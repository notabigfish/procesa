import dgl
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import joblib
import os
import itertools

def get_loader(
        dataroot,
        dataname='human_cell',
        split='train',
        task='reg',
        model='esm1b',
        pklname='pkl',
        batch_size=1,
        shuffle=True,
        num_workers=4,
        return_onehot_num=False):
    
    dataset = ProteinDataset(
            dataroot=dataroot,
            dataname=dataname,
            split=split,
            task=task,
            model=model,
            pklname=pklname,
            return_num=return_onehot_num)
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True)

def collate_fn(samples):
    names, sequences, graphs, labels, seq_num = map(list, zip(*samples))
    str = ''
    sequences = list(str.join(sequences))
    # sequences = np.array(sequences)
    if seq_num[0] is not None:
        return names, sequences, dgl.batch(graphs), torch.from_numpy(np.array(labels)), torch.as_tensor(list(itertools.chain.from_iterable(seq_num)), dtype=torch.long)
    else:
        return names, sequences, dgl.batch(graphs), torch.from_numpy(np.array(labels)), seq_num
        
def normalize_dis(mx): # from SPROF
    return 2 / (1 + np.maximum(mx/4, 1))

class ProteinDataset(Dataset):
    def __init__(self,
            dataroot,
            dataname,
            split,
            task='reg',
            model='esm1b',
            pklname='pkls',
            return_num=False):
        super(ProteinDataset, self).__init__()
        self.data = pd.read_csv(os.path.join(dataroot, 'splits', f"{dataname}_{split}.csv"), sep=',')
        self.split = split
        self.dataroot = dataroot
        self.dataname = dataname
        self.task = task
        self.model = model
        self.pklname = pklname
        self.return_num = return_num
        self.letter_to_num = {
                'G': 0,  'Q': 1,  'D': 2,  'I': 3,  'V': 4,
                'R': 5,  'T': 6,  'K': 7,  'M': 8,  'S': 9,
                'F': 10, 'Y': 11, 'W': 12, 'A': 13, 'H': 14,
                'L': 15, 'P': 16, 'C': 17, 'E': 18, 'N': 19,
                'U': 20}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        names = self.data['gene'].values.tolist()[idx]
        sequences = self.data['sequence'].values.tolist()[idx]
        if self.task == 'reg':
            labels = self.data['tgt_reg'].values.tolist()[idx]
        elif self.task == 'cls':
            labels = self.data['tgt_cls'].values.tolist()[idx]
        
        pklroot = os.path.join(self.dataroot, self.dataname, self.model, self.pklname, self.split)
        with open(os.path.join(pklroot, f"{names}.pickle"), 'rb') as f:
            valid_names, valid_sequences, valid_graphs, _, _ = joblib.load(f)
        labels = labels if isinstance(labels, list) else [labels]
        if self.return_num:
            try:
                seq_num = [self.letter_to_num[a] for a in valid_sequences[names]]
            except:
                print(valid_sequences[names])
        else:
            seq_num = None
            
        return names, valid_sequences[names], valid_graphs[names], labels, seq_num
    
