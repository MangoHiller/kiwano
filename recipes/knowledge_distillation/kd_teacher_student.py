# kd_teacher_student.py

import os
import torch

from kiwano.model.efficientnet import EfficientNetV2  # Vous avez déjà la classe dans kiwano.model

class TeacherStudentWrapper:
    """
    Construit un Teacher (EfficientNet-B2) et un Student (EfficientNet-B0),
    charge des checkpoints si fournis, etc.
    """
    def __init__(
        self,
        num_classes,
        teacher_ckpt=None,
        student_ckpt=None,
        teacher_model_name="efficientnet-b2",
        student_model_name="efficientnet-b0"
    ):
        self.teacher = self._build_model(num_classes, teacher_model_name)
        self.student = self._build_model(num_classes, student_model_name)
        
        # Charger checkpoint teacher
        if teacher_ckpt is not None and os.path.isfile(teacher_ckpt):
            print(f"=> Chargement checkpoint Teacher : {teacher_ckpt}")
            ckpt_t = torch.load(teacher_ckpt, map_location="cpu")
            self.teacher.load_state_dict(ckpt_t["model_state_dict"], strict=False)
        else:
            print("=> Teacher initialisé par défaut (attention, si non pré-entraîné).")

        # Charger checkpoint student
        if student_ckpt is not None and os.path.isfile(student_ckpt):
            print(f"=> Chargement checkpoint Student : {student_ckpt}")
            ckpt_s = torch.load(student_ckpt, map_location="cpu")
            self.student.load_state_dict(ckpt_s["model_state_dict"], strict=False)
        else:
            print("=> Student initialisé par défaut (attention, si non pré-entraîné).")

    def _build_model(self, num_classes, model_name):
        # Suppose que EfficientNetV2 sait déjà charger un backbone pretrained
        model = EfficientNetV2(
            num_classes=num_classes,
            input_features=81,
            embed_features=256,
            model_name=model_name
        )
        return model
