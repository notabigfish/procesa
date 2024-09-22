import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from ..builder import LOSSES

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@LOSSES.register_module()
class TripletLossV2(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(TripletLossV2, self).__init__()
        self.loss_weight = loss_weight

    def get_unsupervised_losses(self, outp1, anchor_f, labels):
        # outp1: [num_nodes, 1024]
        # anchor_f: [num_nodes, 1024]
        # labels: List[num_nodes], ['F', 'K', 'D', 'I', 'T', 'R',...]
        num_samples = 50
        feat_dim = anchor_f.shape[1]
        index = torch.randint(0, outp1.shape[0], size=[num_samples]).to(device)
        f1 = outp1[index]  # [num_samples, feat_dim]
        anchor_out = anchor_f[index]  # [num_samples, feat_dim]

        labels = np.array(labels)
        label_info = labels[index.cpu()]

        # anchor <-> pos
        I = torch.eye(num_samples).to(device)
        factor = 1.0
        anc_pos = torch.mm(anchor_out, f1.T)  # [num_samples, num_samples]
        anc_pos = anc_pos / factor
        anc_pos = I * anc_pos  # diagonal
        anc_pos = torch.nn.functional.logsigmoid(anc_pos) * I
        loss_pos = -torch.sum(anc_pos) / num_samples  # scalar

        # anchor <-> neg
        pos_weight_mat_an = torch.tensor((label_info[:, None] == label_info[None, :]), dtype=torch.float32).to(device)
        negative_samples_minus = 0
        not_same_activity_an = 1 - pos_weight_mat_an - negative_samples_minus
        countneg_an1 = torch.sum(not_same_activity_an)
        anc_neg1 = torch.mm(anchor_out, anchor_out.T)
        anc_neg1 = anc_neg1 / factor
        # anc_neg1 = not_same_activity_an * anc_neg1
        ang_neg1 = torch.nn.functional.logsigmoid(-anc_neg1) * not_same_activity_an
        loss_neg1 = -torch.sum(anc_neg1) / countneg_an1    

        loss =  loss_pos + loss_neg1
        return loss

    def forward(self, projection, labels, anchor_f):
        loss = self.get_unsupervised_losses(
                                    projection,
                                    anchor_f,
                                    labels)
        loss_dict = {'loss_triplet': self.loss_weight * loss}
        return loss_dict
