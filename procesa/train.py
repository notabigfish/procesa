import torch
import os
import argparse
import numpy as np 
import pickle
import warnings

from dataset_flip_revisedataloader import get_loader
from utils_flip_self import *
from utils import Config, build_scheduler, get_root_logger
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('config', help='train config file path')
parser.add_argument('--result_path', type=str, default='./result')
parser.add_argument('--dataroot', type=str, default='./data/preprocess')
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--dataname', type=str, default='mixed_split')
parser.add_argument('--modelname', type=str, default='esm1b')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--pklname', type=str, default='pkls')

args = parser.parse_args()
for arg in vars(args):
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    cfg = Config.fromfile(args.config)
    save_dir = args.result_path
    os.makedirs(save_dir, exist_ok=True)

    logger = get_root_logger(log_file=os.path.join(save_dir, 'results.txt'))
    
    if args.seed:
        seed_everything(args.seed)
        logger.info("SEED FROM ARGS.")
    else:
        seed_everything(cfg.seed)
    
    train_loader = get_loader(
            args.dataroot,
            args.dataname,
            split='train',
            model=args.modelname,
            pklname=args.pklname,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            return_onehot_num=cfg.return_onehot_num if 'return_onehot_num' in cfg else False)
    valid_loader = get_loader(
            args.dataroot,
            args.dataname,
            split='test',
            model=args.modelname,
            pklname=args.pklname,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            return_onehot_num=cfg.return_onehot_num if 'return_onehot_num' in cfg else False)
    
    model = build_model(cfg.model)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.wd)
    scheduler, warmup_scheduler = build_scheduler(
        cfg.lr_config,
        optimizer,
        len(train_loader),
        cfg.max_epochs)
          
    best_epoch, best_metric, best_results = -1, float('-inf'), None
    
    start_epoch = 1
    if args.resume:
        start_epoch, model = resume_checkpoint(
            model,
            save_dir,
            optimizer)

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        loop_train(
            train_loader,
            model,
            epoch,
            logger,
            optimizer,
            scheduler=scheduler,
            warmup_scheduler=warmup_scheduler,
            sep_train_strategy=cfg.sep_train_strategy)
        
        metrics = loop_val(
            valid_loader,
            model,
            epoch,
            logger)
        
        if sum([metrics[k] for k in cfg.best_metrics]) > best_metric:
            best_epoch, best_metric, best_results = epoch, sum([metrics[k] for k in cfg.best_metrics]), metrics
            save_epochcheckpont(epoch, model, optimizer, os.path.join(save_dir, 'epoch_best.pth'))

        if cfg.earlystop and epoch - best_epoch > 7:
            logger.info("Early stopping at epoch {}.".format(epoch))
            break
            
    # result summary
    logger.info("Best Epoch: {} | {}".format(best_epoch, {k:"{:.3f}".format(v) for k, v in best_results.items()}))

    
