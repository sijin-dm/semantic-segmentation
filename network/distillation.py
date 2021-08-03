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


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for name, m in model.named_modules():
        if 'dropout' in m.__class__.__name__.lower():
            m.train()


class Distillation(nn.Module):
    def __init__(self, teacher_net, student_net, criterion=None):
        super(Distillation, self).__init__()
        self.teacher_net = teacher_net
        self.student_net = student_net
        self.criterion = criterion
        self.weight_adapt = 3
        self.weight_student = 1
        if cfg.MODEL.DISTILLATION.DYNAMIC_WEIGHTING:
            self.weight_adapt = nn.Parameter(torch.tensor(self.weight_adapt, dtype=torch.float32), requires_grad=True)
            self.weight_student = nn.Parameter(torch.tensor(self.weight_student, dtype=torch.float32),
                                               requires_grad=True)

    def forward(self, inputs):
        assert "images" in inputs
        student_out = self.student_net(inputs)
        if self.training:
            self.teacher_net.eval()
            with torch.no_grad():
                if cfg.MODEL.DISTILLATION.MONTE_CARLO_DROPOUT_ITERATION is not None:
                    assert cfg.MODEL.DISTILLATION.MONTE_CARLO_DROPOUT_ITERATION > 0
                    enable_dropout(self.teacher_net)
                    teacher_outs = [
                        self.teacher_net.forward_teacher(inputs)
                        for _ in range(cfg.MODEL.DISTILLATION.MONTE_CARLO_DROPOUT_ITERATION)
                    ]

                    teacher_out = {}
                    for key in teacher_outs[0].keys():
                        output_list = [out[key] for out in teacher_outs]
                        mean_tensor = torch.stack(output_list).mean(dim=0)
                        teacher_out[key] = mean_tensor
                else:
                    teacher_out = self.teacher_net(inputs)

            # teacher_loss = teacher_out['loss']
            student_pred = student_out['pred']
            student_loss = student_out['loss']
            teacher_pred = teacher_out['pred']

            loss_adapt = self.criterion(teacher_pred, student_pred)
            if cfg.MODEL.DISTILLATION.DYNAMIC_WEIGHTING:
                loss = torch.exp(-self.weight_student) * student_loss + torch.exp(-self.weight_adapt) * loss_adapt
            else:
                loss = self.weight_student * student_loss + self.weight_adapt * loss_adapt

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
