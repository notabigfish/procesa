from models import GraphModel21
from models.builder import MODELS
# from models.utils import GCNDistContact

@MODELS.register_module()
class GraphModel51(GraphModel21):
    def __init__(self, **kwargs):
        super(GraphModel51, self).__init__(**kwargs)
        """
        del self.conv1
        del self.conv2
        self.conv1 = GCNDistContact(in_features=self.hidden_features, out_features=self.hidden_features)
        self.conv2 = GCNDistContact(in_features=self.hidden_features, out_features=self.hidden_features)
        """
        
    def extract_feat(self, graphs):
        # MLP extract features
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        graphs.ndata['h0'] = graphs.ndata['h']
        mlp_feat, mask, bs = self.split_node_feat(graphs.ndata['h'], graphs)

        # GCN extract features
        graphs.ndata['h'] = self.ln1(self.relu(self.conv1(graphs, node_in='h', edge_in='c', feat_out='h')))
        graphs.ndata['h'] = self.ln2(self.relu(self.conv2(graphs, node_in='h', edge_in='c', feat_out='h')))
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
        
        graphs.ndata.pop('h')
        
        return output, None, None


