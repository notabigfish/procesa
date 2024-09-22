import torch
import torch.nn as nn
from models.utils import GraphConvolution, AttnPooling, AttnPooling_0
from models.builder import MODELS, build_loss
from models import BaseModel
import math


@MODELS.register_module()
class GraphModel15(BaseModel):
    def __init__(self,
            in_features=1280,
            hidden_features=1024,
            output_features=1,
            attention_features=128,
            attention_heads=4,
            loss_reg=dict(type='RMSELoss'),
            loss_triplet=dict(type='TripletLoss'),                 
            train_cfg=None,
            test_cfg=None):
        super(GraphModel15, self).__init__()
        
        self.embed_dim = 21
        self.fc = nn.Linear(in_features + self.embed_dim, hidden_features)
        
        self.w_s = nn.Embedding(self.embed_dim, self.embed_dim)
        
        self.conv1 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        
        self.conv2 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.pooling = AttnPooling(hidden_features, dense_features=attention_features, n_heads=attention_heads)
        self.pooling_0 = AttnPooling_0(hidden_features, dense_features=attention_features, n_heads=attention_heads)
        self.fc_final = nn.Linear(hidden_features*1, output_features)

        self.loss_reg = build_loss(loss_reg)
        self.loss_triplet = build_loss(loss_triplet)

    def extract_feat(self, graphs, seq_num):
        onehot_embeds = self.w_s(seq_num)  # [num_nodes, 20]
        #graphs.ndata['h'] = torch.cat((graphs.ndata['x'], onehot_embeds), dim=1)
        #h_ = graphs.ndata['h'].clone()
        graphs.ndata['h'] = self.fc(torch.cat((graphs.ndata['x'], onehot_embeds), dim=1))
        
        graphs.ndata['h0'] = graphs.ndata['h']
        h0 = graphs.ndata['h'].clone()

        graphs.ndata['h'] =  self.ln1(self.relu(self.conv1(graphs)))

        h1 = graphs.ndata['h'].clone()
        
        graphs.ndata['h'] =  h1 + self.ln2(self.relu(self.conv2(graphs)))

        f_rep =  graphs.ndata['h'].clone()

        output1 = self.pooling(graphs)  #(bs=1,128)
        output2 = self.pooling_0(graphs)
        output = output1 + output2
        output = self.fc_final(output)
        
        graphs.ndata.pop('h')
        
        return output, f_rep, h0
    
    def forward_train(self, graphs, labels, sequences, seq_num):
        output, f_rep, h0 = self.extract_feat(graphs, seq_num)
        losses = dict()
        loss_reg = self.loss_reg(labels, output)  # labels: y_true, outputs: y_pred
        losses.update(loss_reg)
        if self.loss_triplet is not None:
            # h0: f_anchor
            loss_triplet = self.loss_triplet(f_rep, sequences, h0)
            losses.update(loss_triplet)
        return losses

    def forward_test(self, graphs, seq_num):
        output, _, _ = self.extract_feat(graphs, seq_num)
        return output
        
    def forward(self, batch, return_loss=True):
        _, sequences, graphs, labels, seq_num = batch
        graphs = graphs.to('cuda:0')
        labels = labels.to('cuda:0')
        seq_num = seq_num.to('cuda:0')
        if return_loss:
            return self.forward_train(graphs, labels, sequences, seq_num)
        else:
            return self.forward_test(graphs, seq_num), labels

