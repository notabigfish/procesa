import numpy as np
import time
import random

import os
import sys
import torch

import fastaparser
import esm_local as esm
from tqdm import tqdm
import re

import pandas as pd

import argparse 

sys.path.append('../../FLIP/baselines/')
from utils import load_dataset, load_dataset_v2
from train_all import split_dict

esm_dict = {
    'esm1v': 'esm1v_t33_650M_UR90S_1', # use first of 5 models 
    'esm1b': 'esm1b_t33_650M_UR50S',
    'esm_rand': '/../../random_init_esm.pt' # need to randomly initailize an ESM model and save it 
}

def create_parser():
    parser = argparse.ArgumentParser(
        description="create ESM embeddings"
    )

    parser.add_argument(
        "split",
        type=str
    )

    parser.add_argument(
        "esm_version",
        type = str 
    )

    parser.add_argument("--gpu", default='0', type = str)
    
    # zs
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--nogpu", type=int, default=0)
    # zs

    parser.add_argument(
        "--make_fasta",
        action="store_true"
    )
    
    parser.add_argument(
        "--bulk_compute",
        action="store_true"
    )

    parser.add_argument(
        "--concat_tensors",
        action="store_true"
    )

    # zs
    parser.add_argument("--truncate", type=int, default=0)
    parser.add_argument("--trunc_len", type=int, default=1022)
    parser.add_argument("--include", type=str, default='')
    parser.add_argument("--toks_per_batch", type=int, default=4096)
    parser.add_argument("--val_split", action='store_true')
    # zs

    parser.add_argument("--local_esm_model", type=str)

    parser.add_argument(
        "--means_only",
        action="store_true"
    )
    return parser

def make_fasta_v1(outdir, train, test):
    for this_name, this_set in zip(['train', 'test'], [train, test]):
        with open(os.path.join(args.outdir, f'{this_name}.fasta'), 'w') as this_file:
            writer = fastaparser.Writer(this_file)
            for i, seq in enumerate(this_set.sequence):
                writer.writefasta((str(i), seq))
                
def make_fasta_v2(output, train, test):
    for this_name, this_set in zip(['train', 'test'], [train, test]):
        with open(os.path.join(args.outdir, f'{this_name}.fasta'), 'w') as this_file:
            writer = fastaparser.Writer(this_file)
            for (name, seq) in zip(this_set.gene, this_set.sequence):
                writer.writefasta((str(name), seq))    

def main(args):
    split = split_dict[args.split]  # human, s2c2_0, esol
    if 'meltome' in args.split or 'hotprotein' in args.split or 'gb' in args.split:
        dataset = re.findall(r'(\w*)\_', args.split)[0]  # hotprotein, meltome
    elif 'esol' in args.split:
        dataset = args.split
    else:
        raise NotImplementedError

    if dataset == 'meltome':
        train, test, max_length = load_dataset(args.datadir, dataset, split+'.csv', val_split=args.val_split)
    elif 'hotprotein' in dataset or 'esol' in dataset or 'gb1' in dataset:
        train, test, max_length = load_dataset_v2(args.datadir, dataset, split)
    else:
        raise NotImplementedError
    test_len = len(test)
    train_len = len(train)
    max_length += 2
    os.makedirs(args.outdir, exist_ok=True)
    
    if args.make_fasta:
        print('making fasta files')
        if dataset == 'meltome':
            make_fasta_v1(args.outdir, train, test)
        elif 'hotprotein' or 'esol' or 'gb1' in dataset:
            make_fasta_v2(args.outdir, train, test)
    
    if args.bulk_compute:
        print('sending command line arguments')
        if args.local_esm_model:
            esm_dict[args.esm_version] = args.local_esm_model
        for ss in ['train', 'test']:
            os.system(
                    f"python esm_local/extract.py \
                            {esm_dict[args.esm_version]} \
                            {os.path.join(args.outdir, f'{ss}.fasta')} \
                            {os.path.join(args.outdir, f'{ss}')} \
                            --repr_layers 33 \
                            --include {args.include} \
                            --truncate {args.truncate} \
                            --trunc_len {args.trunc_len} \
                            --nogpu {args.nogpu} \
                            --toks_per_batch {args.toks_per_batch}")
           
    if args.concat_tensors:
        for this_len, this_name, this_set in \
                zip([train_len, test_len], 
                    ['train', 'test'], 
                    [train, test]):
            print(f'making empty tensors for {this_name} set')
            if not args.means_only:
                embs_aa = torch.zeros([this_len, args.trunc_len, 1280]) if args.truncate else torch.zeros([this_len, max_length, 1280])
            embs_mean = torch.empty([this_len, 1280])
            labels = torch.empty([this_len])

            print('starting tensor concatenation')
            i = 0
            for l in tqdm(this_set.target):
                e = torch.load(os.path.join(args.outdir, f'{this_name}', (str(i)+'.pt')))
                aa = e['representations'][33]
                if not args.means_only:
                    embs_aa[i, :aa.shape[0], :] = aa
                embs_mean[i] = e['mean_representations'][33]
                labels[i] = l
                i += 1
            if not args.means_only:
                torch.save(embs_aa, os.path.join(args.outdir, f'{this_name}_aa.pt'))
            torch.save(embs_mean, os.path.join(args.outdir, f'{this_name}_mean.pt'))
            torch.save(labels, os.path.join(args.outdir, f'{this_name}_labels.pt'))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
