from models.builder import MODELS
from models import GraphModel3

@MODELS.register_module()
class GraphModel8(GraphModel3):
    def __init__(self, **kwargs):
        super(GraphModel8, self).__init__(**kwargs)
            
    def extract_feat(self, graphs):
        # node_feat: [bs, seq_len, in_features]
        # mask: [bs, seq_len]
        node_feat, mask, bs = self.split_node_feat(graphs)
        node_feat = self.fc(node_feat)
        # skip connection        
        node_feat = node_feat + self.layer1(node_feat)
        # skip connection        
        node_feat = node_feat + self.layer2(node_feat)      
        node_feat = self.pooling(node_feat, mask)
        out = self.fc_final(node_feat)
        return out, None, None
        

    




