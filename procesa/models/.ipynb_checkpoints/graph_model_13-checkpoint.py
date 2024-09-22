from models.builder import MODELS
from models import GraphModel0

@MODELS.register_module()
class GraphModel13(GraphModel0):
    def __init__(self, **kwargs):
        super(GraphModel13, self).__init__(**kwargs)
        
    def extract_feat(self, graphs):       
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        graphs.ndata['h0'] = graphs.ndata['h']
        h0 = graphs.ndata['h'].clone()

        graphs.ndata['h'] =  self.ln1(self.relu(self.conv1(graphs)))
        graphs.ndata['h'] =  self.ln2(self.relu(self.conv2(graphs)))

        f_rep =  graphs.ndata['h'].clone()

        output1 = self.pooling(graphs)
        output2 = self.pooling_0(graphs)
        output = self.fc_final(output1)
        graphs.ndata.pop('h')
        
        return output, f_rep, h0