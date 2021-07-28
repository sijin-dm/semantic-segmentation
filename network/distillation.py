import torch
import torch.nn as nn

import network
from config import cfg
from loss.utils import get_loss_distillation
from runx.logx import logx

# TODO:
# 1. train teacher also.
# 2. AffinityNet
# 3. Spherical Knowledge Disitllation


class Distillation(nn.Module):
    def __init__(self, teacher_net, student_net, criterion=None):
        super(Distillation, self).__init__()
        self.teacher_net = teacher_net
        # self.teacher_net.requires_grad_(False)
        self.student_net = student_net
        self.criterion = criterion
        self.weight_adapt = 10

    def forward(self, inputs):
        assert "images" in inputs
        student_out = self.student_net(inputs)
        if self.training:
            with torch.no_grad():
                teacher_out = self.teacher_net(inputs)
            # teacher_loss = teacher_out['loss']
            student_pred = student_out['pred']
            student_loss = student_out['loss']
            teacher_pred = teacher_out['pred']

            loss_adapt = self.criterion(teacher_pred, student_pred)
            loss = student_loss + self.weight_adapt * loss_adapt
            return loss
        else:
            # student already a dict.
            return student_out


def OCR_HRNet_Mscale_OCR_DDRNET23_SLIM_Mscale(num_classes, criterion):
    assert cfg.MODEL.DISTILLATION.ON
    teacher_arch = 'ocrnet.HRNet_Mscale'
    student_arch = 'ocrnet.DDRNet23_Slim_Mscale'
    logx.msg("Knowledge Distillation between [{}] and [{}].".format(teacher_arch, student_arch))

    ocr_net = network.get_model(network='network.' + teacher_arch, num_classes=num_classes, criterion=criterion)

    ddr_net = network.get_model(network='network.' + student_arch, num_classes=num_classes, criterion=criterion)
    criterion = get_loss_distillation()

    return Distillation(teacher_net=ocr_net, student_net=ddr_net, criterion=criterion)
