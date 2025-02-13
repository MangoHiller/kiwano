#!/bin/bash
#
# compute_score.sh
#
# Usage : compute_score.sh <exp_dir> <epoch_tag>
#   - <exp_dir> = dossier contenant les xvectors (par ex. exp/efficientnet)
#   - <epoch_tag> = suffixe d’époque (par ex. _10) ou un label libre
#
# Exemple : ./compute_score.sh exp/efficientnet 10

dir=$1   # ex: exp/efficientnet
epoch_tag=$2   # ex: 10

echo "----- VoxCeleb1-O -----"
python utils/compute_cosine.py  \
   data/voxceleb1/voxceleb1-o-cleaned.trials  \
   "pkl:cat $dir/voxceleb1.${epoch_tag}/xvector.*.pkl |"  \
   "pkl:cat $dir/voxceleb1.${epoch_tag}/xvector.*.pkl |" \
    > $dir/voxceleb1.${epoch_tag}/scores.txt

python utils/compute_eer.py data/voxceleb1/voxceleb1-o-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
python utils/compute_dcf.py data/voxceleb1/voxceleb1-o-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
python utils/compute_cllr.py data/voxceleb1/voxceleb1-o-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
echo -e "\n"

echo "----- VoxCeleb1-E -----"
python utils/compute_cosine.py  \
   data/voxceleb1/voxceleb1-e-cleaned.trials  \
   "pkl:cat $dir/voxceleb1.${epoch_tag}/xvector.*.pkl |"  \
   "pkl:cat $dir/voxceleb1.${epoch_tag}/xvector.*.pkl |" \
    > $dir/voxceleb1.${epoch_tag}/scores.txt

python utils/compute_eer.py data/voxceleb1/voxceleb1-e-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
python utils/compute_dcf.py data/voxceleb1/voxceleb1-e-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
python utils/compute_cllr.py data/voxceleb1/voxceleb1-e-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
echo -e "\n"


echo "----- VoxCeleb1-H -----"
python utils/compute_cosine.py  \
   data/voxceleb1/voxceleb1-h-cleaned.trials  \
   "pkl:cat $dir/voxceleb1.${epoch_tag}/xvector.*.pkl |"  \
   "pkl:cat $dir/voxceleb1.${epoch_tag}/xvector.*.pkl |" \
    > $dir/voxceleb1.${epoch_tag}/scores.txt

python utils/compute_eer.py data/voxceleb1/voxceleb1-h-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
python utils/compute_dcf.py data/voxceleb1/voxceleb1-h-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
python utils/compute_cllr.py data/voxceleb1/voxceleb1-h-cleaned.trials \
    $dir/voxceleb1.${epoch_tag}/scores.txt
echo -e "\n"

echo "Done scoring with $dir (tag=$epoch_tag)."
