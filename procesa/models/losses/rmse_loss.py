import torch
import torch.nn as nn
from ..builder import LOSSES


@LOSSES.register_module()
class RMSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(RMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.loss_weight = loss_weight

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        loss = torch.sqrt(self.mseloss(y_pred, y_true))
        loss_dict = {'loss_reg': self.loss_weight * loss}
        return loss_dict