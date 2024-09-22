model = dict(
    type='GraphModel0',
    loss_reg=dict(type='RMSELoss', loss_weight=1.0),
    loss_triplet=dict(type='TripletLoss', loss_weight=0.01))

seed = 42
num_workers = 4
wd = 0.0
lr_config = dict(
    policy='NotChange')
earlystop = True
sep_train_strategy = None
best_metrics = ['spearmanr', 'pearson']
batch_size = 4
learning_rate = 8e-4
max_epochs = 10
