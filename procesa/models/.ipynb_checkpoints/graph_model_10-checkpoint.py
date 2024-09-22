from models import GraphModel4
from models.builder import MODELS

@MODELS.register_module()
class GraphModel10(GraphModel4):
    def __init__(self, **kwargs):
        super(GraphModel10, self).__init__(**kwargs)

    def extract_feat(self, graphs):
        input = graphs.ndata['x']

        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        # graphs.ndata['h0'] = graphs.ndata['h']

        # skip connection        
        h0 = graphs.ndata['h'].clone()
        graphs.ndata['h'] = h0 + self.ln1(self.relu(self.conv1(graphs)))
        
        # skip connection
        h1 = graphs.ndata['h'].clone()
        graphs.ndata['h'] =  h1 + self.ln2(self.relu(self.conv2(graphs)))

        f_rep =  graphs.ndata['h'].clone()

        output = self.pooling(graphs)  #(bs=1,128)
        output = self.fc_final(output)
        
        graphs.ndata.pop('h')
        
        return output, None, None
    




