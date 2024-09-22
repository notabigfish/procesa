from models import GraphModel23
from models.builder import MODELS
from models.utils import GCNDist

@MODELS.register_module()
class GraphModel37(GraphModel23):
    def __init__(self, **kwargs):
        super(GraphModel37, self).__init__(**kwargs)
        del self.conv1
        del self.conv2
        self.conv1 = GCNDist(in_features=self.hidden_features, out_features=self.hidden_features)
        self.conv2 = GCNDist(in_features=self.hidden_features, out_features=self.hidden_features)

