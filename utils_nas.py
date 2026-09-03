"""
Utilities for BrainNAS.

Reuses some utilities from BQN-demo via path manipulation.
Provides its own data loading for multi-dataset support.
"""

import sys
import os

# Add BQN-demo to path so we can reuse its utilities
_BQN_DEMO_PATH = os.path.join(os.path.dirname(__file__), '..', 'BQN-demo')
if _BQN_DEMO_PATH not in sys.path:
    sys.path.insert(0, _BQN_DEMO_PATH)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import logging
from typing import List, Dict, Any
from sklearn.model_selection import StratifiedShuffleSplit


# ---- Re-export utilities from BQN-demo ----
from utils import (
    StandardScaler_crossROI,
    continues_mixup_data,
    fix_seed,
    count_param,
    isfloat,
)


# ================================================================
# Multi-dataset data loading
# ================================================================

# Map dataset names to file names (handles old + new naming)
_DATASET_FILE_MAP = {
    'ABIDE': 'abide.npy',
}


def load_data(args):
    """Load dataset. Supports original ABIDE + new {dataset}_full_{atlas}.npy files."""
    data_dir = args.data_dir

    # Check if this is a known dataset with a mapped filename
    if args.dataset in _DATASET_FILE_MAP:
        data_path = os.path.join(data_dir, _DATASET_FILE_MAP[args.dataset])
    else:
        # New naming: dataset like "ABIDE_AAL116" → "abide_full_AAL116.npy"
        # The atlas part preserves original case; stem is always lowercase
        ds_lower = args.dataset.lower()
        all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        candidates = [
            f for f in all_files
            if f.lower() == f'{ds_lower.split("_")[0]}_full_{ds_lower.split("_")[1]}.npy'
        ]
        if candidates:
            data_path = os.path.join(data_dir, candidates[0])
        else:
            raise FileNotFoundError(
                f"No file matching dataset={args.dataset} in {data_dir}")

    data = np.load(data_path, allow_pickle=True).item()

    # Handle typo in original data key
    if 'timeseires' in data:
        data_timeseries = data['timeseires']
    elif 'timeseries' in data:
        data_timeseries = data['timeseries']
    else:
        raise KeyError(f"No timeseries key found in {data_path}")

    data_label = data['label']
    data_pearson = data['corr']

    # site is optional (only in original ABIDE)
    if 'site' in data:
        site = data['site']
    else:
        site = data_label  # use labels for stratification

    data_timeseries = StandardScaler_crossROI(data_timeseries)

    (data_timeseries, data_label, data_pearson) = \
        [torch.from_numpy(d).float() for d in (data_timeseries, data_label, data_pearson)]

    # Detect number of classes (use numpy before torch conversion)
    num_classes = len(set(data_label.ravel().tolist()))

    return data_timeseries, data_pearson, data_label, site, num_classes


def init_stratified_dataloader(args,
                               final_timeseries: torch.Tensor,
                               final_pearson: torch.Tensor,
                               labels: torch.Tensor,
                               stratified: np.ndarray,
                               num_classes: int = 2) -> Dict[str, Any]:
    """Create stratified train/val/test dataloaders."""
    labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=num_classes)
    length = final_timeseries.shape[0]
    train_length = int(length * args.Train_prop)
    val_length = int(length * args.Val_prop)
    test_length = length - train_length - val_length

    spilt1 = StratifiedShuffleSplit(
        n_splits=1, train_size=train_length,
        test_size=length - train_length, random_state=args.seed,
    )
    for train_index, val_and_test_index in spilt1.split(final_timeseries, stratified):
        final_timeseries_train = final_timeseries[train_index]
        final_pearson_train = final_pearson[train_index]
        labels_train = labels_onehot[train_index]
        final_timeseries_val_and_test = final_timeseries[val_and_test_index]
        final_pearson_val_and_test = final_pearson[val_and_test_index]
        labels_val_and_test = labels_onehot[val_and_test_index]
        stratified = stratified[val_and_test_index]

    spilt2 = StratifiedShuffleSplit(n_splits=1, test_size=test_length)
    for val_index, test_index in spilt2.split(final_timeseries_val_and_test, stratified):
        final_timeseries_val = final_timeseries_val_and_test[val_index]
        final_pearson_val = final_pearson_val_and_test[val_index]
        labels_val = labels_val_and_test[val_index]
        final_timeseries_test = final_timeseries_val_and_test[test_index]
        final_pearson_test = final_pearson_val_and_test[test_index]
        labels_test = labels_val_and_test[test_index]

    train_dataset = torch.utils.data.TensorDataset(
        final_timeseries_train, final_pearson_train, labels_train)
    val_dataset = torch.utils.data.TensorDataset(
        final_timeseries_val, final_pearson_val, labels_val)
    test_dataset = torch.utils.data.TensorDataset(
        final_timeseries_test, final_pearson_test, labels_test)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    return {
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "test_dataloader": test_dataloader,
    }


# ================================================================
# Helpers
# ================================================================

def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    """Computes the precision@k for the specified values of k."""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def hyper_para_load(args, dataset):
    """Load hyper parameters — always extracts from data shape."""
    node_sz = dataset[0].shape[1]       # number of ROIs
    timeseries_sz = dataset[0].shape[-1] # time series length
    node_feature_sz = dataset[1].shape[-1]  # correlation feature dim

    layers = args.layers
    dropout = args.dropout
    pooling = args.pooling
    cluster_num = args.cluster_num

    return (node_sz, timeseries_sz, node_feature_sz, layers, dropout,
            pooling, cluster_num)


def optimizer_update(optimizer: torch.optim.Optimizer, step: int,
                     total_steps: int, args):
    """Cosine learning rate scheduling."""
    base_lr = args.base_lr
    target_lr = args.target_lr

    current_ratio = step / total_steps
    cosine = math.cos(math.pi * current_ratio)
    lr = target_lr + (base_lr - target_lr) * (1 + cosine) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_formatter() -> logging.Formatter:
    return logging.Formatter(
        '[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s'
    )


def initialize_logger() -> logging.Logger:
    """Initialize logger (reuses BQN-demo pattern)."""
    logger = logging.getLogger('BQN-NAS')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    formatter = get_formatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
