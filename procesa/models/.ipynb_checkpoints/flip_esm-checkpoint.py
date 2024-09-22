import torch
import torch.nn as nn
from sequence_models.structure import Attention1d
from models.builder import MODELS, build_loss
from models import BaseModel

# ESMAttention1d
@MODELS.register_module()
class FLIPESM(BaseModel):
    def __init__(self,
                 max_seq_len=800,
                 d_embedding=1280,
                 loss_reg=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(FLIPESM, self).__init__()

        self.max_seq_len = 800
        
        self.attention1d = Attention1d(in_dim=d_embedding)
        self.linear = nn.Linear(d_embedding, d_embedding)
        self.relu = nn.ReLU()
        self.final = nn.Linear(d_embedding, 1)
        self.loss_reg = build_loss(loss_reg)

    def split_node_feat(self, graphs):
        node_feats = graphs.ndata['x']
        bs = graphs.batch_size
        chunks = graphs.batch_num_nodes().tolist()
        node_feats = torch.split(node_feats, chunks, dim=0)
        ret = torch.zeros((bs, self.max_seq_len, node_feats[0].shape[-1])).to(node_feats[0].device)
        mask = torch.zeros((bs, self.max_seq_len)).to(node_feats[0].device)
        for i, (chunk, node_feat) in enumerate(zip(chunks, node_feats)):
            ret[i, :chunk] = node_feat
            mask[i, :chunk].fill_(1)
        return ret, mask, bs
    
    def extract_feat(self, graphs):
        x, input_mask, bs = self.split_node_feat(graphs)
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x

    def forward_train(self, graphs, labels, sequences):
        output = self.extract_feat(graphs)
        losses = dict()
        loss_reg = self.loss_reg(output, labels)
        losses.update(loss_reg)
        return losses
 
        