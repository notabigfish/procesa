import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn.parameter import Parameter

def make_mlplayers(in_channel, cfg, batch_norm=False, out_layer =None): #1433  cfg=[512,128]
    layers = []
    in_channels = in_channel
    layer_num  = len(cfg)
    for i, v in enumerate(cfg):
        out_channels =  v
        mlp = nn.Linear(in_channels, out_channels)
        if batch_norm:
            layers += [mlp, nn.BatchNorm1d(out_channels, affine=False), nn.ReLU()]
        elif i != (layer_num-1):
            layers += [mlp, nn.ReLU()]
        else:
            layers += [mlp]
        in_channels = out_channels
    if out_layer != None:
        mlp = nn.Linear(in_channels, out_layer)
        layers += [mlp]
    return nn.Sequential(*layers)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.edge = Parameter(torch.FloatTensor(in_features, ))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, graphs, node_in='h', edge_in='x', feat_out='h'):
        # GCNDist: node_in='h', edge_in='d', feat_out='h'
        # GCNDistRaw: node_in='h', edge_in='dr', feat_out='h'
        # GCNDistContact: node_in='h', edge_in='c', feat_out='h'
        
        # u_mul_e: mul node feat h and edge feat x, save to m. then sum up m, save to h
        graphs.update_all(fn.u_mul_e(node_in, edge_in, 'm'), fn.sum('m', feat_out))
        graphs.ndata[feat_out] = graphs.ndata[feat_out] @ self.weight  # equals to torch.matmul
        if self.bias is not None:
            return graphs.ndata[feat_out] + self.bias
        else:
            return graphs.ndata[feat_out]


class GCNDist(GraphConvolution):
    def __init__(self, **kwargs):
        super(GCNDist, self).__init__(**kwargs)

    def forward(self, graphs, node_in='h', edge_in='d', feat_out='h'):
        # u_mul_e: mul node feat h and edge feat x, save to m. then sum up m, save to h
        graphs.update_all(fn.u_mul_e(node_in, edge_in, 'm'), fn.sum('m', feat_out))
        graphs.ndata[feat_out] = graphs.ndata[feat_out] @ self.weight  # equals to torch.matmul
        if self.bias is not None:
            return graphs.ndata[feat_out] + self.bias
        else:
            return graphs.ndata[feat_out]


class GCNDistRaw(GraphConvolution):
    # raw distmap as edge feats
    def __init__(self, **kwargs):
        super(GCNDistRaw, self).__init__(**kwargs)

    def forward(self, graphs):
        # u_mul_e: mul node feat h and edge feat x, save to m. then sum up m, save to h
        graphs.update_all(fn.u_mul_e('h', 'dr', 'm'), fn.sum('m', 'h'))
        graphs.ndata['h'] = graphs.ndata['h'] @ self.weight  # equals to torch.matmul
        if self.bias is not None:
            return graphs.ndata['h'] + self.bias
        else:
            return graphs.ndata['h']


class GCNDistContact(GraphConvolution):
    # contact map as edge feats
    def __init__(self, **kwargs):
        super(GCNDistContact, self).__init__(**kwargs)

    def forward(self, graphs):
        # u_mul_e: mul node feat h and edge feat x, save to m. then sum up m, save to h
        graphs.update_all(fn.u_mul_e('h', 'c', 'm'), fn.sum('m', 'h'))
        graphs.ndata['h'] = graphs.ndata['h'] @ self.weight  # equals to torch.matmul
        if self.bias is not None:
            return graphs.ndata['h'] + self.bias
        else:
            return graphs.ndata['h']



class AttnPooling(nn.Module):
    def __init__(self, in_features, dense_features, n_heads):
        super(AttnPooling, self).__init__()
        self.in_features = in_features
        self.dense_features = dense_features
        self.n_heads = n_heads
        self.fc1 = nn.Linear(in_features, dense_features)
        self.fc2 = nn.Linear(dense_features, n_heads)
    
    def forward(self, graphs):
        with graphs.local_scope():  # out-place ops only works in this func; in-place ops works outside this func
            graphs.ndata['heads'] = torch.tanh(self.fc1(graphs.ndata['h']))
            # (num_nodes, n_heads)
            graphs.ndata['heads'] = self.fc2(graphs.ndata['heads']) #(num_nodes, n_heads)
            attns = dgl.softmax_nodes(graphs, 'heads') #(num_nodes, n_heads) softmax on 'heads' feat
            for i in range(self.n_heads):
                graphs.ndata[f'head_{i}'] = attns[:,i].reshape(-1, 1)
            result = []
            for i in range(self.n_heads):
                result.append(dgl.sum_nodes(graphs, 'h', f'head_{i}').unsqueeze(0))  # sum weighted node_feat h
            output = torch.mean(torch.cat(result, dim=0), dim=0)  # avg all heads
            return output


class AttnPooling_0(nn.Module):
    def __init__(self, in_features, dense_features, n_heads):
        super(AttnPooling_0, self).__init__()
        self.in_features = in_features
        self.dense_features = dense_features
        self.n_heads = n_heads
        self.fc1 = nn.Linear(in_features, dense_features)
        self.fc2 = nn.Linear(dense_features, n_heads)

    def forward(self, graphs):
        with graphs.local_scope():
            graphs.ndata['heads'] = torch.tanh(self.fc1(graphs.ndata['h0']))
            # (num_nodes, n_heads)
            graphs.ndata['heads'] = self.fc2(graphs.ndata['heads'])  # (num_nodes, n_heads)
            attns = dgl.softmax_nodes(graphs, 'heads')  # (num_nodes, n_heads)
            for i in range(self.n_heads):
                graphs.ndata[f'head_{i}'] = attns[:, i].reshape(-1, 1)
            result = []
            for i in range(self.n_heads):
                result.append(dgl.sum_nodes(graphs, 'h0', f'head_{i}').unsqueeze(0))
            output = torch.mean(torch.cat(result, dim=0), dim=0)
            return output

