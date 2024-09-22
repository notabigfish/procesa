#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2022/01/17 16:00:19
@Author  :   Jianwen Chen
@Version :   1.0
@Contact :   chenjw48@mail2.sysu.edu.cn
@License :   (C)Copyright 2021-2022, SAIL-Lab
'''
######################################## import area ########################################

# common library
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from torch.nn.parameter import Parameter
from dgl.nn import GraphConv
from dgl.nn import GATConv
from dgl.nn import AvgPooling
from dgl.nn import SetTransformerEncoder
from dgl.nn import SetTransformerDecoder
######################################## function area ########################################

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
    print("layers----", layers)
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

    def forward(self, graphs):
        graphs.update_all(fn.u_mul_e('h', 'x', 'm'), fn.sum('m', 'h'))
        graphs.ndata['h'] = graphs.ndata['h'] @ self.weight
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
        with graphs.local_scope():
            graphs.ndata['heads'] = torch.tanh(self.fc1(graphs.ndata['h']))
            # (num_nodes, n_heads)
            graphs.ndata['heads'] = self.fc2(graphs.ndata['heads']) #(num_nodes, n_heads)
            attns = dgl.softmax_nodes(graphs, 'heads') #(num_nodes, n_heads)
            for i in range(self.n_heads):
                graphs.ndata[f'head_{i}'] = attns[:,i].reshape(-1, 1)
            result = []
            for i in range(self.n_heads):
                result.append(dgl.sum_nodes(graphs, 'h', f'head_{i}').unsqueeze(0))
            output = torch.mean(torch.cat(result, dim=0), dim=0)
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


class GraphModel(nn.Module):
    
    def __init__(self, in_features, hidden_features, output_features, attention_features, attention_heads):
        super(GraphModel, self).__init__()
        # self.MLP = make_mlplayers(in_features, cfg=[256,128])   #[256,1536]
        # self.MLP = make_mlplayers(in_features, cfg=[1024,512,1024])   #[256,1536]

        self.fc = nn.Linear(in_features, hidden_features)

        self.conv1 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.conv2 = GraphConvolution(in_features=hidden_features, out_features=hidden_features)

        self.ln2 = nn.LayerNorm(hidden_features)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        # self.gatconv1 = GATConv(hidden_features, hidden_features, num_heads=1)

        self.pooling = AttnPooling(hidden_features, dense_features=attention_features, n_heads=attention_heads)
        self.pooling_0 = AttnPooling(hidden_features, dense_features=attention_features, n_heads=attention_heads)

        self.fc_final = nn.Linear(hidden_features*1, output_features)

        self.avgpool = AvgPooling()
        # self.set_trans_dec = SetTransformerDecoder(hidden_features, 1, 256, 256, 1, 3)

    def forward(self, graphs):

        input = graphs.ndata['x']
        # f_anchor = self.MLP(graphs.ndata['x'])
        # graphs.ndata['h2'] = f_anchor
        # print("f_anchor----", f_anchor.shape)

        # graphs.ndata['x'] = F.dropout(graphs.ndata['x'], 0.5, training=self.training)
        graphs.ndata['h'] = self.fc(graphs.ndata['x'])
        graphs.ndata['h0'] = graphs.ndata['h']
        # graphs.ndata['h'] = graphs.ndata['x']
        h0 = graphs.ndata['h'].clone()

        # graphs.ndata['h'] = F.dropout(graphs.ndata['h'], 0.6)  #for huma humn_cell
        graphs.ndata['h'] =  self.ln1(self.relu(self.conv1(graphs)))
        # graphs.ndata['h'] =  self.ln1(self.relu(self.conv1(graphs)))

        # graphs.ndata['h'] = h0 + self.ln1(self.relu(self.conv1(graphs, h0, edge_weight=graphs.edata['x'])))
        # graphs.ndata['h'] = h0 + self.ln1(self.relu(self.conv1(graphs, h0, edge_weight=graphs.edata['x']) + torch.squeeze(self.gatconv1(graphs,h0) )))

        h1 = graphs.ndata['h'].clone()
        
        graphs.ndata['h'] =  h1 + self.ln2(self.relu(self.conv2(graphs)))
        # graphs.ndata['h'] = h0 + self.ln2(self.relu(self.conv2(graphs)))
        # graphs.ndata['h'] = h0 + h1 + self.ln2(self.relu(self.conv2(graphs, h1, edge_weight=graphs.edata['x'])))

        f_rep =  graphs.ndata['h'].clone()
        # f_rep =  h1

        output1 = self.pooling(graphs)  #(bs=1,128)
        output2 = self.pooling_0(graphs)
        # output2 = self.avgpool(graphs, h0)
        # output = self.set_trans_dec(graphs, graphs.ndata['h'])
        # output = output + output2
        # output = torch.cat([output1,output2],1)

        # print("output----",output.shape)
        output = self.fc_final(output1)
        graphs.ndata.pop('h')
        
        return output, input, f_rep, h0
    




