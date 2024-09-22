import torch.nn as nn
from models.builder import MODELS
from abc import ABCMeta, abstractmethod

@MODELS.register_module()
class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def extract_feat(self, graphs):
        pass

    @abstractmethod
    def forward_train(self, graphs, labels, sequences):
        pass

    def forward_test(self, graphs):
        output, _, _ = self.extract_feat(graphs)
        return output
    
    def forward(self, batch, return_loss=True):
        _, sequences, graphs, labels, _ = batch
        graphs = graphs.to('cuda:0')
        labels = labels.to('cuda:0')
        if return_loss:
            return self.forward_train(graphs, labels, sequences)
        else:
            return self.forward_test(graphs), labels
