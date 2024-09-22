import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CriterionClass(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()

    def get_unsupervised_losses_my_combinebatch_double(self, outp1, anchor_f, labels):
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
        num_neg = 5
        loss_neg1 = 0.0
        feat_neg = torch.zeros((num_neg, num_samples, feat_dim)).to(device)

        for i in range(num_samples):
            labellist = np.where(label_info != label_info[i])[0]
            m = np.random.choice(labellist, num_neg)
            for k, ng in enumerate(m):
                feat_neg[k, i, :] = anchor_out[ng]

        for i in range(num_neg):
            anc_neg1 = torch.mm(anchor_out, feat_neg[i].T)
            anc_neg1 = anc_neg1 / factor
            anc_neg1 = I * anc_neg1
            anc_neg1 = torch.nn.functional.logsigmoid(-anc_neg1) * I
            loss_neg1 = loss_neg1 - torch.sum(anc_neg1) / num_samples

        loss =  loss_pos + loss_neg1
        unsupervised_dict_loss = {'contrastive_loss': loss}
        return unsupervised_dict_loss

    def forward(self, projection, labels, anchor_f):
        unsupervised_loss_dict = self.get_unsupervised_losses_my_combinebatch_double(
                                    projection,
                                    anchor_f,
                                    labels)
        loss = unsupervised_loss_dict['contrastive_loss']
        loss_dict = {'full_loss': loss}
        loss_dict.update(unsupervised_loss_dict)
        return loss_dict
