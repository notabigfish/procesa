from models import GraphModel19
from models.builder import MODELS

@MODELS.register_module()
class GraphModel20(GraphModel19):
    def __init__(self, **kwargs):
        super(GraphModel20, self).__init__(**kwargs)

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
        values = self.v_layer(mlp_feat)
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
