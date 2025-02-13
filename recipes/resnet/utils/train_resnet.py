#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, SpecAugment, Reverb
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNetV2, IDRDScheduler, JeffreysLoss

import soundfile as sf

from torch.utils.data import Dataset, DataLoader, Sampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler

import argparse

#import idr_torch
import hostlist
import logging
import os


logger = logging.getLogger(__name__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor = None, feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

        audio, sample_rate = segment.load_audio()
        if self.audio_transforms != None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        if self.feature_extractor != None:
            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

        if self.feature_transforms != None:
            feature = self.feature_transforms(feature)

        return feature, self.labels[ segment.spkid ]



if __name__ == '__main__':

    #os.environ["MASTER_PORT"] = "29500"
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world = int(os.environ["SLURM_JOB_NUM_NODES"])
    world_size = int(os.environ["SLURM_NTASKS"])

    hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

    os.environ["MASTER_ADDR"] = hostnames[0]
    os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids)))


    master_addr = hostnames[0]
    port = int(os.environ["MASTER_PORT"])
    checkpoint = None

    print(f"SLURM_JOB_NODELIST: {os.environ.get('SLURM_JOB_NODELIST', 'non défini')}")
    print(f"Expanded hostnames: {hostnames}")
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
    print(f"RANK: {rank}")
    print(f"LOCAL_RANK: {local_rank}")
    print(f"GPU_IDS: {gpu_ids}")
    print(f"WORLD_SIZE: {world}")


    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--musan", type=str, default="data/musan/")
    parser.add_argument("--rirs_noises", type=str, default = "data/rirs_noises/")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("training_corpus", type=str, metavar="training_corpus")
    parser.add_argument("exp_dir", type=str, metavar="exp_dir")

    args = parser.parse_args()


    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location={"cuda" : "cpu"})


    epochs_start = 0
    if args.checkpoint:
        epochs_start = checkpoint["epochs"]

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    print(device)
    print(f"Using GPU: {torch.cuda.current_device()}, Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
    print(f"Available memory: {torch.cuda.memory_reserved(0) / 1e9} GB")

    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    print(f"Process {rank}/{world_size} initialized on {os.uname().nodename}.")
    print(f"Process {dist.get_rank()} running on {os.uname().nodename}.")

    musan = SegmentSet()
    musan.from_dict(Path(args.musan))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")


    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    training_data = SpeakerTrainingSegmentSet(
                                    audio_transforms=Sometimes( [
                                        Noise(musan_music, snr_range=[5,15]),
                                        Noise(musan_speech, snr_range=[13,20]),
                                        Noise(musan_noise, snr_range=[0,15]),
                                        #Codec(),
                                        #Filtering(),
                                        Normal(),
                                        Reverb(reverb)
                                    ] ),
                                    feature_extractor=Fbank(),
                                    feature_transforms=Linear( [
                                        CMVN(),
                                        Crop(350),
                                        SpecAugment(),
                                    ] ),
                                )


    training_data.from_dict(Path(args.training_corpus))

    train_sampler = DistributedSampler(training_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)

    train_dataloader = DataLoader(training_data, batch_size=64, drop_last=True, shuffle=False, num_workers=15, sampler=train_sampler, pin_memory=True)
    iterator = iter(train_dataloader)


    resnet_model = ResNetV2()
    print(resnet_model)
    if args.checkpoint:
        resnet_model.load_state_dict(  checkpoint["model"]  )
    resnet_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
    resnet_model.to(device)

    resnet_model = torch.nn.parallel.DistributedDataParallel(resnet_model) #, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = torch.optim.SGD([{'params':resnet_model.module.preresnet.parameters(), 'weight_decay':0.0001, 'lr':0.00001},{'params':resnet_model.module.temporal_pooling.parameters(), 'weight_decay':0.0001, 'lr':0.00001},{'params':resnet_model.module.embedding.parameters(), 'weight_decay':0.0001, 'lr':0.00001},{'params':resnet_model.module.output.parameters(), 'lr':0.00001}], momentum=0.9)
    if args.checkpoint:
        optimizer.load_state_dict( checkpoint["optimizer"] )

    criterion = JeffreysLoss(coeff1=0.1, coeff2=0.025)

    scheduler = IDRDScheduler(optimizer, num_epochs=150, initial_lr=0.2, warm_up_epoch=5, plateau_epoch=15, patience=10, factor = 5, amsmloss = 0.3)
    if args.checkpoint:
        scheduler.set_epoch( checkpoint["epochs"] )


    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print('START TRAINING ...')
    for epochs in range(epochs_start, 150):
        iterations = 0
        train_sampler.set_epoch(epochs)
        resnet_model.module.set_m( scheduler.get_amsmloss()   )
        torch.distributed.barrier()
        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1)

            feats = feats.float().to(device)
            iden = iden.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                preds = resnet_model(feats, iden)
                loss = criterion(preds, iden)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            #loss.backward()
            #optimizer.step()

            if iterations%100 == 0:
                msg = "{}: Epoch: [{}/{}] ({}/{}) \t C-Loss:{:.4f} \t LR : {:.8f} \t Margin : {:.4f}".format(time.ctime(), epochs, 150, iterations, len(train_dataloader), loss.item(), get_lr(optimizer), resnet_model.module.get_m())
                print(msg)

            iterations += 1

        scheduler.step()

        if dist.get_rank() == 0:
            checkpoint = {
                "epochs": epochs+1,
                "optimizer": optimizer.state_dict(),
                "model": resnet_model.module.state_dict(),
            }
            torch.save(checkpoint, args.exp_dir+"/model"+str(epochs)+".ckpt")

