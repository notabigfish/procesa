import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from utils import LinearWarmup

def build_scheduler(lr_config, optimizer, len_loader, max_epochs):
    if lr_config is None:
        return None, None
        
    warmup_steps = 0
    if 'warmup' in lr_config:
        warmup_steps = lr_config['warmup_steps']
    policy_type = lr_config.pop('policy')        
    if policy_type == 'MultiStep':
        milestones = [q * len_loader for q in lr_config['steps']]
        scheduler = MultiStepLR(
                        optimizer,
                        milestones=milestones,
                        gamma=lr_config['decay_ratio'])
    elif policy_type == 'CosineAnnealing':
        # T_max = max_epoch * iters, lr changes every batch
        # T_max = max_epoch, lr stays the same inside each epoch
        scheduler = CosineAnnealingLR(
                        optimizer,
                        T_max=max_epochs * len_loader - warmup_steps,
                        eta_min=lr_config['min_lr'])
    elif policy_type == 'NotChange':
        scheduler = None
    else:
        raise NotImplementedError(f"Not recognised type {policy_type}!")
        
    warmup_scheduler = None
    if 'warmup' in lr_config:
        warmup_scheduler = LinearWarmup(
                            optimizer,
                            warmup_period=warmup_steps)
        
    return scheduler, warmup_scheduler
       
