"""
Standard training and evaluation for the derived (discretized) architecture.

After NAS search completes and we derive a fixed architecture, we retrain
it from scratch using standard training (same pattern as BQN-demo).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

from utils_nas import continues_mixup_data, accuracy, isfloat, optimizer_update


def train(model, optimizer, args, train_loader, epoch):
    """Train the derived model (same pattern as BQN-demo)."""
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    criterion = nn.CrossEntropyLoss(reduction='sum')

    model = model.to(device)
    model.train()
    train_loss = 0.0
    train_acc_list = []
    step_base = epoch * len(train_loader)
    total_steps = len(train_loader) * args.epochs

    for i, (time_series, node_feature, label) in enumerate(train_loader):
        step = step_base + i + 1
        time_series = time_series.to(device)
        node_feature = node_feature.to(device)
        label = label.to(device)

        time_series, node_feature, label = continues_mixup_data(
            time_series, node_feature, y=label, device=device,
        )
        output = model(time_series, node_feature)
        label = label.float()

        optimizer_update(optimizer=optimizer, step=step, total_steps=total_steps, args=args)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        top1 = accuracy(output, label[:, 1])[0] / 100
        train_acc_list.append(top1)

    train_loss = train_loss / (train_loader.dataset.tensors[0].shape[0] // args.batch_size)
    train_acc = np.mean(train_acc_list)
    return {"train_loss": train_loss, "train_acc": train_acc}


def val_test(model, args, val_loader, test_loader):
    """Evaluate on validation and test sets (same pattern as BQN-demo)."""
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    criterion = nn.CrossEntropyLoss(reduction='sum')

    model.eval()

    # ---- Validation ----
    val_loss = 0.0
    val_acc_list = []
    val_result = []
    val_labels = []
    for time_series, node_feature, label in val_loader:
        time_series = time_series.to(device)
        node_feature = node_feature.to(device)
        label = label.to(device).float()

        output = model(time_series, node_feature)
        loss = criterion(output, label)
        val_loss += loss.item()
        top1 = accuracy(output, label[:, 1])[0] / 100
        val_acc_list.append(top1)
        val_result += F.softmax(output, dim=1)[:, 1].tolist()
        val_labels += label[:, 1].tolist()

    val_loss = val_loss / ((val_loader.dataset.tensors[0].shape[0] // args.batch_size) + 1)
    val_acc = np.mean(val_acc_list)
    val_roc = roc_auc_score(val_labels, val_result) if len(set(val_labels)) > 1 else 0.5

    # ---- Test ----
    test_loss = 0.0
    test_acc_list = []
    test_result = []
    test_labels = []
    for time_series, node_feature, label in test_loader:
        time_series = time_series.to(device)
        node_feature = node_feature.to(device)
        label = label.to(device).float()

        output = model(time_series, node_feature)
        loss = criterion(output, label)
        test_loss += loss.item()
        top1 = accuracy(output, label[:, 1])[0] / 100
        test_acc_list.append(top1)
        test_result += F.softmax(output, dim=1)[:, 1].tolist()
        test_labels += label[:, 1].tolist()

    test_loss = test_loss / ((test_loader.dataset.tensors[0].shape[0] // args.batch_size) + 1)
    test_acc = np.mean(test_acc_list)
    test_roc = roc_auc_score(test_labels, test_result)

    test_result_arr = np.array(test_result)
    test_result_arr[test_result_arr > 0.5] = 1
    test_result_arr[test_result_arr <= 0.5] = 0
    labels_arr = np.array(test_labels)

    report = classification_report(labels_arr, test_result_arr, output_dict=True, zero_division=0)
    recall = [0, 0]
    for k in report:
        if isfloat(k):
            recall[int(float(k))] = report[k]['recall']

    return {
        "val_loss": val_loss, "val_acc": val_acc, "val_roc": val_roc,
        "test_loss": test_loss, "test_acc": test_acc, "test_roc": test_roc,
        "test_sensitivity": recall[-1], "test_specificity": recall[-2],
    }


def run_derived_training(model, args, train_loader, val_loader, test_loader, logger):
    """
    Full training loop for the derived architecture.

    Returns best test metrics based on validation loss.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
    )

    epoch_val_roc_list, epoch_val_loss_list = [], []
    epoch_test_roc_list, epoch_test_acc_list = [], []
    epoch_test_sen_list, epoch_test_spec_list = [], []

    for epoch in range(args.epochs):
        result_train = train(model=model, optimizer=optimizer, args=args,
                             train_loader=train_loader, epoch=epoch)
        result_val_test = val_test(model=model, args=args,
                                   val_loader=val_loader, test_loader=test_loader)

        logger.info(" | ".join([
            f'Epoch[{epoch}/{args.epochs}]',
            f'Train Loss:{result_train["train_loss"]:.3f}',
            f'Train Acc:{result_train["train_acc"]:.4f}',
            f'Val Loss:{result_val_test["val_loss"]:.3f}',
            f'Val Acc:{result_val_test["val_acc"]:.4f}',
            f'Val AUC:{result_val_test["val_roc"]:.4f}',
            f'Test Acc:{result_val_test["test_acc"]:.4f}',
            f'Test AUC:{result_val_test["test_roc"]:.4f}',
            f'Test Sen:{result_val_test["test_sensitivity"]:.4f}',
            f'Test Spec:{result_val_test["test_specificity"]:.4f}',
        ]))

        epoch_val_loss_list.append(result_val_test['val_loss'])
        epoch_val_roc_list.append(result_val_test['val_roc'])
        epoch_test_roc_list.append(result_val_test['test_roc'])
        epoch_test_acc_list.append(result_val_test['test_acc'])
        epoch_test_sen_list.append(result_val_test['test_sensitivity'])
        epoch_test_spec_list.append(result_val_test['test_specificity'])

    index_max = epoch_val_loss_list.index(min(epoch_val_loss_list))
    return (
        epoch_test_acc_list[index_max],
        epoch_test_roc_list[index_max],
        epoch_test_sen_list[index_max],
        epoch_test_spec_list[index_max],
    )
