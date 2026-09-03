"""
FairDARTS bi-level optimization for architecture search.

Differences from DARTS:
  - Architecture parameters use sigmoid gating (not softmax)
  - Zero-One loss pushes gates toward 0/1 extremes
  - No temperature annealing needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

from utils_nas import accuracy


def train_search_epoch(model, architect, optimizer_w, optimizer_alpha,
                       train_loader, val_loader, args, epoch):
    """
    One epoch of DARTS bi-level training.

    For each batch:
      1. Update α on val data: L_val(w - ξ∇L_train(w), α) → ∇_α
      2. Update w on train data: L_train(w, α)

    Uses first-order approximation (ξ = 0).
    """
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    criterion = nn.CrossEntropyLoss(reduction='sum')

    model = model.to(device)
    model.train()

    train_loss = 0.0
    val_loss_accum = 0.0
    train_acc_list = []

    # Create an iterator for validation data (cycle if needed)
    val_iter = iter(val_loader)

    for batch_idx, (time_series, node_feature, label) in enumerate(train_loader):
        # ---- Get validation batch ----
        try:
            val_time_series, val_node_feature, val_label = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            val_time_series, val_node_feature, val_label = next(val_iter)

        time_series = time_series.to(device)
        node_feature = node_feature.to(device)
        label = label.to(device)
        val_time_series = val_time_series.to(device)
        val_node_feature = val_node_feature.to(device)
        val_label = val_label.to(device)

        label_float = label.float()
        val_label_float = val_label.float()

        # ---- Step 1: Update architecture α on validation data ----
        # FairDARTS: L_arch = L_val + λ * L_zero_one
        optimizer_alpha.zero_grad()

        val_output = model(val_time_series, val_node_feature)
        val_loss = criterion(val_output, val_label_float)
        zero_one = model.zero_one_loss()
        arch_loss = val_loss + args.zero_one_lambda * zero_one

        arch_loss.backward()
        optimizer_alpha.step()

        val_loss_accum += val_loss.item()

        # ---- Step 2: Update model weights w on training data ----
        optimizer_w.zero_grad()

        train_output = model(time_series, node_feature)
        loss = criterion(train_output, label_float)

        loss.backward()
        optimizer_w.step()

        train_loss += loss.item()
        top1 = accuracy(train_output, label_float[:, 1])[0] / 100
        train_acc_list.append(top1)

    # Normalize (matching BQN-demo: reduction='sum', divide by N)
    n_train = train_loader.dataset.tensors[0].shape[0]
    n_val = val_loader.dataset.tensors[0].shape[0]
    train_loss = train_loss / n_train if n_train > 0 else train_loss
    val_loss_accum = val_loss_accum / n_val if n_val > 0 else val_loss_accum
    train_acc = np.mean(train_acc_list) if train_acc_list else 0.0

    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss_accum,
    }


def search_eval_epoch(model, loader, args):
    """Evaluate model during search phase."""
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    criterion = nn.CrossEntropyLoss(reduction='sum')

    model.eval()
    total_loss = 0.0
    acc_list = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for time_series, node_feature, label in loader:
            time_series = time_series.to(device)
            node_feature = node_feature.to(device)
            label = label.to(device)
            label_float = label.float()

            output = model(time_series, node_feature)
            loss = criterion(output, label_float)
            total_loss += loss.item()

            top1 = accuracy(output, label_float[:, 1])[0] / 100
            acc_list.append(top1)

            preds = F.softmax(output, dim=1)[:, 1].cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(label_float[:, 1].cpu().tolist())

    n = loader.dataset.tensors[0].shape[0]
    avg_loss = total_loss / n if n > 0 else total_loss
    avg_acc = np.mean(acc_list) if acc_list else 0.0
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5

    return {
        'loss': avg_loss,
        'acc': avg_acc,
        'auc': auc,
    }


def run_darts_search(model, args, train_loader, val_loader, test_loader, logger):
    """
    Run DARTS architecture search.

    Returns the trained model with learned architecture parameters.
    """
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Separate optimizers for weights and architecture parameters
    optimizer_w = torch.optim.Adam(
        model.model_parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_alpha = torch.optim.Adam(
        model.arch_parameters(),
        lr=args.arch_lr,
        weight_decay=args.arch_weight_decay,
    )

    # Dummy architect (for API compatibility; the logic is in train_search_epoch)
    architect = None

    best_val_auc = 0.0
    best_arch_state = None
    search_history = []
    last_arch_state = None

    for epoch in range(args.search_epochs):
        result_train = train_search_epoch(
            model, architect, optimizer_w, optimizer_alpha,
            train_loader, val_loader, args, epoch,
        )

        # Evaluate on held-out test set
        result_test = search_eval_epoch(model, test_loader, args)

        logger.info(" | ".join([
            f'Search Epoch[{epoch}/{args.search_epochs}]',
            f'Train Loss:{result_train["train_loss"]:.3f}',
            f'Train Acc:{result_train["train_acc"]:.4f}',
            f'Val Loss:{result_train["val_loss"]:.3f}',
            f'Test Loss:{result_test["loss"]:.3f}',
            f'Test Acc:{result_test["acc"]:.4f}',
            f'Test AUC:{result_test["auc"]:.4f}',
        ]))

        search_history.append({
            'epoch': epoch,
            'train_loss': result_train['train_loss'],
            'train_acc': result_train['train_acc'],
            'val_loss': result_train['val_loss'],
            'test_loss': result_test['loss'],
            'test_acc': result_test['acc'],
            'test_auc': result_test['auc'],
        })

        if result_test['auc'] > best_val_auc:
            best_val_auc = result_test['auc']
            best_arch_state = {
                'epoch': epoch,
                'alpha_src_op': [layer.alpha_src_op.detach().cpu().clone()
                                 for layer in model.nas_layers],
                'alpha_all': [layer.alpha_all.detach().cpu().clone()
                               for layer in model.nas_layers],
            }

        # Always snapshot the last-epoch state (gates fully differentiated)
        last_arch_state = {
            'epoch': epoch,
            'alpha_src_op': [layer.alpha_src_op.detach().cpu().clone()
                             for layer in model.nas_layers],
            'alpha_all': [layer.alpha_all.detach().cpu().clone()
                           for layer in model.nas_layers],
        }

    # Derive the final architecture from the LAST epoch — gates are fully
    # binarized by the Zero-One loss, whereas the best-search-AUC epoch is
    # typically too early (epoch 5-14) and still near-uniform.
    final_state = last_arch_state
    for i, layer in enumerate(model.nas_layers):
        layer.alpha_src_op.data = final_state['alpha_src_op'][i].to(device)
        layer.alpha_all.data = final_state['alpha_all'][i].to(device)
    logger.info(f"Using FINAL-epoch architecture (epoch {final_state['epoch']}) "
                 f"for derivation; best search AUC was {best_val_auc:.4f} "
                 f"at epoch {best_arch_state['epoch'] if best_arch_state else 'N/A'}")

    return model, search_history, final_state
