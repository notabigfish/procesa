import torch
import torch.nn as nn
from models.utils import GraphConvolution, AttnPooling, AttnPooling_0
from models.builder import MODELS, build_loss

@MODELS.register_module()
class GraphModel999(nn.Module):
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
        super(GraphModel999, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)

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

    def extract_feat(self, graphs):       
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
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
        
        # f_rep [num_nodes, 1024]
        # f_anchor [num_nodes, 1024]
        # sequences List[num_nodes], ['M', 'V', 'L', 'G', 'G',...]
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

    def forward_test(self, graphs):
        output, _, _ = self.extract_feat(graphs)
        return output
        
    def forward(self, graphs, labels=None, sequences=None, return_loss=True):
        if return_loss:
            return self.forward_train(graphs, labels, sequences)
        else:
            return self.forward_test(graphs)



