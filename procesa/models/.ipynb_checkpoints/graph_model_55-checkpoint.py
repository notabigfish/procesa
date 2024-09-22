from models import GraphModel21
from models.builder import MODELS
# from models.utils import GCNDistRaw
from models.utils import GraphConvolution
import torch.nn as nn


@MODELS.register_module()
class GraphModel55(GraphModel21):
    def __init__(self, **kwargs):
        super(GraphModel55, self).__init__(**kwargs)
        self.conv3 = GraphConvolution(in_features=self.hidden_features, out_features=self.hidden_features)
        self.ln3 = nn.LayerNorm(self.hidden_features)
        self.conv4 = GraphConvolution(in_features=self.hidden_features, out_features=self.hidden_features)
        self.ln4 = nn.LayerNorm(self.hidden_features)
        
    def extract_feat(self, graphs):
        # MLP extract features
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        mlp_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # GCN extract features
        # raw dist values as edge feats
        graphs.ndata['hr'] = self.ln1(self.relu(self.conv1(graphs, node_in='h', edge_in='x', feat_out='hr')))
        graphs.ndata['hr'] = self.ln2(self.relu(self.conv2(graphs, node_in='hr', edge_in='x', feat_out='hr')))
        # contact map as edge feats
        graphs.ndata['hc'] = self.ln3(self.relu(self.conv3(graphs, node_in='h', edge_in='c', feat_out='hc')))
        graphs.ndata['hc'] = self.ln4(self.relu(self.conv4(graphs, node_in='hc', edge_in='c', feat_out='hc')))

        graphs.ndata['h'] = graphs.ndata['hr'] + graphs.ndata['hc']
        
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
        feat = self.ln3(feat + values)
        feat = self.pooling(feat, mask)  # 8,1024
        output = self.fc_final(feat)
        
        # graphs.ndata.pop('h')
        
        return output, None, None