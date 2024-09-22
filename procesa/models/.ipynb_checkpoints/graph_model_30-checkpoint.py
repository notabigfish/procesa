import torch.nn as nn
from models import GraphModel19
from models.builder import MODELS

@MODELS.register_module()
class GraphModel30(GraphModel19):
    def __init__(self, **kwargs):
        super(GraphModel30, self).__init__(**kwargs)
        del self.ln1
        del self.ln2
        del self.ln3
        self.bn1 = nn.BatchNorm1d(self.hidden_features)
        self.bn2 = nn.BatchNorm1d(self.hidden_features)
        self.bn3 = nn.BatchNorm1d(self.hidden_features)

    def extract_feat(self, graphs):
        # MLP extract features
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        graphs.ndata['h0'] = graphs.ndata['h']
        mlp_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # GCN extract features
        graphs.ndata['h'] = self.bn1(self.relu(self.conv1(graphs)))
        graphs.ndata['h'] = self.bn2(self.relu(self.conv2(graphs)))
        gcn_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # attention
        queries = self.q_layer(gcn_feat)
        keys = self.k_layer(mlp_feat)
        values = self.v_layer(gcn_feat)        
        feat = self.multihead_attn_layer(
            queries.transpose(0, 1),
            keys.transpose(0, 1),
            values.transpose(0, 1))[0].transpose(0, 1)  # q, k, v

        # last block
        feat = self.linear1(feat)
        feat = self.dropout(feat)
        feat = self.bn3((feat + values).swapaxes(1, 2)).swapaxes(1, 2)
        feat = self.pooling(feat, mask)  # 8,1024
        output = self.fc_final(feat)
        
        graphs.ndata.pop('h')
        
        return output, None, None
