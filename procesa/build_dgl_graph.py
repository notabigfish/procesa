import dgl
import torch
import pickle
import numpy as np
import pandas as pd
import joblib
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='xxxx/dataset/procesa_data/meltome')
parser.add_argument('--dataset', type=str, default='mixed_split')
parser.add_argument('--task', type=str, default='reg')
parser.add_argument('--model', type=str, default='esm1b')
parser.add_argument('--trunc_len', type=int, default=800)

def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result

def get_feats(ptdir, name):
    data = torch.load(os.path.join(ptdir, f'{name}.pt'))
    nodefeat = data['representations'][33].detach().cpu().numpy()
    edgefeat = data['contacts'].detach().cpu().numpy()
    return nodefeat, edgefeat

def build_dgl_graph(
        names,
        sequences,
        labels_reg,
        labels_cls=None,
        dataroot='/datasets/procesa_data/meltome',
        dataset='mixed_split',
        split='train',
        model='esm1b',
        trunc_len=800):

    if labels_cls == None:
        labels_cls = [None] * len(labels_reg)

    start_time = time.time()
    for index, (name, sequence, label_reg, label_cls) in enumerate(zip(names, sequences, labels_reg, labels_cls)):
        names_list, sequences_dict, graphs_dict, labels_reg_dict, labels_cls_dict = list(), dict(), dict(), dict(), dict()
        if index % 1000 == 0:
            print("{} / {}".format(index, len(names)))
        node_features, edge_features = get_feats(os.path.join(dataroot, dataset, model, split), name)

        if len(sequence) > trunc_len:
            sequence = sequence[:trunc_len]
        src, dst = np.nonzero(edge_features)

        # [L, L] -> [num_edges]
        edge_features = normalize_adj(edge_features) #[L,L]
        edge_features = edge_features[np.nonzero(edge_features)]

        graph = dgl.graph((src, dst), num_nodes=len(sequence))
        if not len(sequence) == node_features.shape[0]:
            print(f"{dataset} {model} {split} {name} node features error!")
            assert False
        if not len(edge_features) == len(src) == len(dst):
            print(f"{dataset} {model} {split} {name} edge features error!")
            assert False
        
        # add features
        graph.ndata['x'] = torch.from_numpy(node_features).float()
        graph.edata['x'] = torch.from_numpy(edge_features).float()
        
        # add all
        names_list.append(name)
        sequences_dict[name] = sequence
        graphs_dict[name] = graph
        labels_reg_dict[name] = label_reg if isinstance(label_reg, list) else [label_reg]
        labels_cls_dict[name] = label_cls if isinstance(label_cls, list) else [label_cls]
        
        saveroot = os.path.join(dataroot, dataset, model, 'pkls', split)
        os.makedirs(saveroot, exist_ok=True)
        with open(os.path.join(saveroot, f"{name}.pickle"), 'wb') as fw:
            joblib.dump(
                [names_list,
                 sequences_dict,
                 graphs_dict,
                 labels_reg_dict,
                 labels_cls_dict], fw)
    print(time.time() - start_time)

if __name__ == '__main__':
    args = parser.parse_args()

    # build the dgl graph cache
    for split in ['train', 'test']:
        data = pd.read_csv(os.path.join(args.dataroot, 'splits', f"{args.dataset}_{split}.csv"), sep=',')

        names = data['gene'].values.tolist()
        sequences = data['sequence'].values.tolist()
        labels_reg = data['tgt_reg'].values.tolist()
        labels_cls = None
        if args.task == 'reg+cls':
            labels_cls = data['tgt_cls'].values.tolist()

        build_dgl_graph(
                names,
                sequences,
                labels_reg,
                labels_cls,
                dataroot=args.dataroot,
                dataset=args.dataset,
                split=split,
                model=args.model,
                trunc_len=args.trunc_len)

