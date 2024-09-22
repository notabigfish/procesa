import torch
import torch.nn as nn
from models.utils import GraphConvolution
from models.attention import AttentionPooling

from models.builder import MODELS, build_loss
from models import BaseModel

@MODELS.register_module()
class GraphModel19(BaseModel):
    def __init__(self,
            in_features=1280,
            hidden_features=1024,
            output_features=1,
            attention_features=128,
            attention_heads=4,
            loss_reg=dict(type='RMSELoss'),
            loss_triplet=dict(type='TripletLoss'),
            max_seq_len=800,
            dropout_rate=0.1,
            train_cfg=None,
            test_cfg=None):
        super(GraphModel19, self).__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.max_seq_len = max_seq_len
        self.hidden_features = hidden_features
        self.dropout_rate = dropout_rate
        self.in_features = in_features
        
        self.conv1 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.conv2 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.q_layer = nn.Linear(hidden_features, hidden_features)
        self.k_layer = nn.Linear(hidden_features, hidden_features)
        self.v_layer = nn.Linear(hidden_features, hidden_features)

        self.multihead_attn_layer = nn.MultiheadAttention(
                                        hidden_features,
                                        attention_heads,
                                        dropout=dropout_rate)
        self.linear1 = nn.Linear(hidden_features, hidden_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln3 = nn.LayerNorm(hidden_features)

        self.pooling = AttentionPooling(
                            hidden_features,
                            attention_heads,
                            attention_features)
        self.fc_final = nn.Linear(hidden_features*1, output_features)

        self.loss_reg = build_loss(loss_reg)
        self.loss_triplet = build_loss(loss_triplet)

    def split_node_feat(self, node_feats, graphs):
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
        # MLP extract features
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        graphs.ndata['h0'] = graphs.ndata['h']
        mlp_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # GCN extract features
        graphs.ndata['h'] = self.ln1(self.relu(self.conv1(graphs)))
        graphs.ndata['h'] = self.ln2(self.relu(self.conv2(graphs)))
        gcn_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # attention
        queries = self.q_layer(mlp_feat)
        keys = self.k_layer(gcn_feat)
        values = self.v_layer(gcn_feat)
        feat = self.multihead_attn_layer(
            queries.transpose(0, 1),
            keys.transpose(0, 1),
            values.transpose(0, 1))[0].transpose(0, 1)  # q, k, v

        # last block
        feat = self.linear1(feat)
        feat = self.dropout(feat)
        feat = self.ln3(feat + values)
        feat = self.pooling(feat, mask)  # 8,1024
        output = self.fc_final(feat)
        
        graphs.ndata.pop('h')
        
        return output, None, None
    
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

