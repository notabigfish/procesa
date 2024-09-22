import torch
import os
import argparse
import numpy as np
import pickle
import warnings

from dataset_flip_revisedataloader import get_loader
from utils_flip_self import *
from utils import Config, get_root_logger
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('config', help='train config file path')
parser.add_argument('--ckpt_path', type=str, default='./result')
parser.add_argument('--dataroot', type=str, default='./data/preprocess')
parser.add_argument('--dataname', type=str, default='mixed_split')
parser.add_argument('--modelname', type=str, default='esm1b')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()
for arg in vars(args):
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cfg = Config.fromfile(args.config)
    logger = get_root_logger(log_file=
            os.path.join(os.path.dirname(args.ckpt_path), 'test_res.txt'))
    if args.seed:
        seed_everything(args.seed)
        logger.info("SEED FROM ARGS.")
    else:
        seed_everything(cfg.seed)

    valid_loader = get_loader(
            args.dataroot,
            args.dataname,
            split='test',
            model=args.modelname,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers)

    model = build_model(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()

    model, test_epoch = load_ckpt(model, args.ckpt_path)

    metrics = loop_val(
            valid_loader,
            model,
            test_epoch,
            logger)
