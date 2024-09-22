import torch
import torch.nn as nn
from models.attention import AttentionPooling
from models.builder import MODELS, build_loss
from models import BaseModel

@MODELS.register_module()
class GraphModel11(nn.Module):
    def __init__(self,
            in_features=1280,
            hidden_features=1024,
            output_features=1,
            attention_features=128,
            attention_heads=4,
            loss_reg=dict(type='RMSELoss'),
            loss_triplet=dict(type='TripletLoss'),                 
            max_seq_len=800,
            train_cfg=None,
            test_cfg=None):
        super(GraphModel11, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.max_seq_len = max_seq_len
        
        self.layer1 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LeakyReLU(0.1, inplace=True),
            nn.LayerNorm(hidden_features))
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.LeakyReLU(0.1, inplace=True))
        self.pooling = AttentionPooling(
                            hidden_features,
                            attention_heads,
                            attention_features)
        self.pooling2 = AttentionPooling(
                            hidden_features,
                            attention_heads,
                            attention_features)
        self.fc_final = nn.Linear(hidden_features * 2, output_features)

        self.loss_reg = build_loss(loss_reg)
        self.loss_triplet = build_loss(loss_triplet)
    
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
        # node_feat: [bs, seq_len, in_features]
        # mask: [bs, seq_len]
        node_feat, mask, bs = self.split_node_feat(graphs)
        node_feat = self.fc(node_feat)
        seq_feat = self.pooling2(node_feat, mask)
        node_feat = self.layer1(node_feat)
        node_feat = self.layer2(node_feat)
        node_feat = self.pooling(node_feat, mask)
        node_feat = torch.cat((node_feat, seq_feat), 1)
        out = self.fc_final(node_feat)
        return out, None, None
        
    def forward_train(self, graphs, labels, sequences):
        output, f_rep, h0 = self.extract_feat(graphs)
        losses = dict()
        loss_reg = self.loss_reg(labels, output)  # labels: y_true, outputs: y_pred
        losses.update(loss_reg)
        if self.loss_triplet is not None:
            # h0: f_anchor
            loss_triplet = self.loss_triplet(f_rep, sequences, h0)
            losses.update(loss_triplet)
        return losses

    




