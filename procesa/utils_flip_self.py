import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from scipy.stats import pearsonr, spearmanr
import scipy
import time

def seed_everything(seed=2021):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def initialize_weights(model):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Module.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

def sep_train_hook(epoch, losses, strategy=None):
    if strategy is None:
        return sum(losses.values())
    # strategy = dict(term_epoch=[5], loss_weight=[dict(loss_reg=0, loss_triplet=1)])
    for term_epoch, loss_weight in zip(strategy['term_epoch'], strategy['loss_weight']):
        if epoch < term_epoch:
            for k, v in loss_weight.items():
                losses[k] *= v
            break
    return sum(losses.values())

def loop_train(data_loader, model, epoch, logger, optimizer, scheduler, warmup_scheduler=None, sep_train_strategy=None):
    batch_size = data_loader.batch_size
    model.train()
    duration = 0.0
    for i, batch in enumerate(data_loader):
        start_time = time.time()
        # _, sequences, graphs, labels, seq_num = batch
        # graphs = graphs.to('cuda:0')
        # labels = labels.to('cuda:0')
        losses = model(batch)
        #losses = model(graphs, labels, sequences)
        loss = sep_train_hook(epoch, losses, sep_train_strategy)
        losses.update({'loss': loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            if warmup_scheduler is None:
                scheduler.step()
            else:
                with warmup_scheduler.dampening():
                    if warmup_scheduler.last_step + 1 >= warmup_scheduler.warmup_steps:
                        scheduler.step()
        elif warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_scheduler.warmup_steps:
                    pass
        torch.cuda.empty_cache()
        duration += time.time() - start_time
        
        if i % 10 == 0:
            logger.info("Train | Epoch: {} | Time: {} | Step: {} / {} | {}".format(epoch, duration, i, len(data_loader), {k:"{:.4f}".format(v / batch_size) for k, v in losses.items()}))
            duration = 0.0

def loop_val(data_loader, model, epoch, logger):
    y_true, y_pred = [], []
    model.eval()
    for i, batch in enumerate(data_loader):
        preds, labels = model(batch, return_loss=False)
        preds = preds.detach().cpu().numpy().tolist()
        labels = labels.detach().cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels)

        torch.cuda.empty_cache()

    # metric with threshold 0.5
    results = cal_metric(y_true, y_pred, threshold=0.5)

    logger.info("Val | Epoch: {} | {}".format(epoch, {k:"{:.3f}".format(v) for k, v in results.items()}))

    return results

def cal_metric(y_true, y_pred, threshold=None):
    threshold = 0.5 if threshold is None else threshold
    concatenate_true, concatenate_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    
    binary_true = [1 if true >= threshold else 0 for true in concatenate_true]
    binary_pred = [1 if pred >= threshold else 0 for pred in concatenate_pred]
    
    rmse = np.sqrt(metrics.mean_squared_error(concatenate_true, concatenate_pred))
    pearson = pearsonr(concatenate_true, concatenate_pred)[0]
    spearmanr = scipy.stats.spearmanr(concatenate_true, concatenate_pred)[0]
    r2 = metrics.r2_score(concatenate_true, concatenate_pred)
    return {'rmse': rmse, 'pearson': pearson, 'r2': r2, "spearmanr": spearmanr}

def resume_checkpoint(model, model_dir, optimizer):
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(model_dir + "/epoch_best" + ".ckpt")

    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    print("start epoch:", start_epoch)
    return start_epoch, model

def load_ckpt(model, ckpt_path):
    # Load checkpoint.
    print('==> load test checkpoint..')
    checkpoint = torch.load(ckpt_path)

    model.load_state_dict(checkpoint['net'])
    test_epoch = checkpoint['epoch']
    print("test epoch: ", test_epoch)
    return model, test_epoch

def save_epochcheckpont(epoch, model, optimizer, save_dir):
    print('Saving checkpoint...')
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, save_dir)
