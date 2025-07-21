
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from typing import List
import torch.nn as nn


class HDKDLoss(torch.nn.Module):
    """
    This module wraps the Feature distillation criterion which is added to the Logit distillation loss
    and acts as regularization term and enhances the model generalization
    """
    def __init__(self, teacher_model: torch.nn.Module, student_model: torch.nn.Module, alpha: float,
                 teacher_layers: List[int] = [1, 2, 3], student_layers: List[int] =[1, 2, 3]):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha=alpha
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers

    def forward(self,inputs):
        fkd_loss=0
        for i, layers in enumerate(zip(self.teacher_layers,self.student_layers)):
            teacher_layer,student_layer=layers[0],layers[1]
            teacher_layers=nn.Sequential(*list(self.teacher_model.children())[:teacher_layer])
            student_layers=nn.Sequential(*list(self.student_model.children())[:student_layer])
            with torch.no_grad():
                teacher_feature=teacher_layers(inputs)
            student_feature=student_layers(inputs)
            if teacher_feature.shape != student_feature.shape:
                raise ValueError("""This feature distillation loss requires the selected features
                of the teacher and student models to have the same dimensions while
                the given teacher feature dimension is {} and the student feature dimension is {}
                """.format(teacher_feature.shape,student_feature.shape))
            fkd_loss+= nn.MSELoss()(teacher_feature,student_feature)* pow(i+1,self.alpha)

        return fkd_loss

