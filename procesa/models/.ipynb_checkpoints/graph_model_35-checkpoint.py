from models import GraphModel19
from models.builder import MODELS
import torch.nn as nn

@MODELS.register_module()
class GraphModel35(GraphModel19):
    def __init__(self, **kwargs):
        super(GraphModel35, self).__init__(**kwargs)
        del self.relu
        self.relu = nn.ReLU()

