import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from models.builder import MODELS, build_loss
from models import BaseModel

@MODELS.register_module()
class GraphModel16(BaseModel):
    def __init__(self,
            in_features=1280,
            hidden_features=256,
            output_features=1,
            attention_features=128,
            attention_heads=4,
            loss_reg=dict(type='RMSELoss'),
            loss_triplet=dict(type='TripletLoss'),                 
            train_cfg=None,
            test_cfg=None):
        super(GraphModel16, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)

        self.gat1 = GATConv(
            in_feats=hidden_features,
            out_feats=hidden_features,
            num_heads=attention_heads,
            residual=True)
        self.ln1 = nn.LayerNorm(hidden_features * attention_heads)
        
        self.gat2 = GATConv(
            in_feats=hidden_features * attention_heads,
            out_feats=hidden_features,
            num_heads=attention_heads,
            residual=True)
        self.ln2 = nn.LayerNorm(hidden_features)
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.fc_final = nn.Linear(hidden_features*1, output_features)

        self.loss_reg = build_loss(loss_reg)
        self.loss_triplet = build_loss(loss_triplet)

    def extract_feat(self, graphs):
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        
        h0 = graphs.ndata['h'].clone()
        graphs.ndata['h'] =  self.ln1(self.relu(self.gat1(graphs, h0).flatten(1)))

        h1 = graphs.ndata['h'].clone()
        
        graphs.ndata['h'] =  self.ln2(self.relu(self.gat2(graphs, h1).mean(1)))

        f_rep =  graphs.ndata['h'].clone()
        feats = h0 + f_rep
        feats = torch.split(feats, graphs.batch_num_nodes().tolist(), dim=0)
        feats = torch.stack([feat.mean(0) for feat in feats], dim=0)

        output = self.fc_final(feats)
        
        graphs.ndata.pop('h')
        
        return output, f_rep, h0
    
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

