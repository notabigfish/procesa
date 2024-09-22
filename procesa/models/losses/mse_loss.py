import torch.nn as nn
import torch.nn.functional as F
from models import LOSSES

@LOSSES.register_module()
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        loss = F.mse_loss(pred, target, reduction='mean') * self.loss_weight
        loss_dict = {'loss_reg': loss}
        return loss_dict
        