"""

Pipeline:
  1. DARTS Architecture Search — learn α (continuous relaxation)
  2. Architecture Derivation — discretize α → fixed operations
  3. Retrain from Scratch — train derived architecture
  4. Multiple-run Evaluation — report mean±std metrics

Results are saved as CSV files in the result/ directory.
"""

import os
import sys
import time
import torch
import numpy as np
import csv
from datetime import datetime

from parse_nas import get_args
from utils_nas import (
    load_data, init_stratified_dataloader, hyper_para_load,
    count_param, fix_seed, initialize_logger,
)

from model.nas_network import NASNetwork
from model.nas_derived import (
    DerivedNetwork, derive_architecture, describe_architecture,
)
from train_search import run_darts_search
from train_eval import run_derived_training


# Global experiment directory for the current run (set once in run_nas_pipeline)
_EXP_DIR = None


def _get_exp_dir(args) -> str:
    """Get or create the experiment directory for this run.

    If --exp_name is provided (from shell script), use it to ensure all
    configs in a sweep share one folder. Otherwise, auto-generate.
    """
    global _EXP_DIR
    if _EXP_DIR is None:
        exp_name = args.exp_name or datetime.now().strftime('exp_%Y%m%d_%H%M%S')
        _EXP_DIR = os.path.join(args.root_path, 'result', exp_name)
    os.makedirs(_EXP_DIR, exist_ok=True)
    return _EXP_DIR


def _config_id(args) -> str:
    """Generate a unique configuration identifier from hyperparameters."""
    lam = getattr(args, 'zero_one_lambda', 0)
    parts = [
        f"L{args.layers}",
        f"K{args.top_k}",
        f"ALR{args.arch_lr:.0e}".replace('e-0', 'e-'),
        f"LAM{lam:.0e}".replace('e-0', 'e-') if lam else '',
        f"WD{args.weight_decay:.0e}".replace('e-0', 'e-'),
        f"DR{args.dropout}",
        f"C{args.cluster_num}",
    ]
    return '_'.join(p for p in parts if p)


def create_search_dataloaders(args, dataset):
    """
    Create train/val/test splits for architecture search.
    The search uses a portion of training data as validation for DARTS.
    """
    # For DARTS search: split the training data into search_train and search_val
    # We reuse init_stratified_dataloader but with different proportions
    # to get a held-out validation set for architecture optimization

    # Save original proportions
    orig_train_prop = args.Train_prop
    orig_val_prop = args.Val_prop

    # For search: use 60% train, 20% val (for arch), 20% test
    args.Train_prop = 0.6
    args.Val_prop = 0.2

    dataloaders = init_stratified_dataloader(args, *dataset)

    # Restore original proportions
    args.Train_prop = orig_train_prop
    args.Val_prop = orig_val_prop

    return (dataloaders["train_dataloader"],
            dataloaders["val_dataloader"],
            dataloaders["test_dataloader"])


def run_nas_pipeline(args):
    """
    Run the full NAS pipeline:
      Search → Derive → Train → Evaluate

    Returns metrics dict.
    """
    logger = initialize_logger()
    fix_seed(args.seed)

    # ---- Load data ----
    dataset = load_data(args)
    num_classes = dataset[4]  # 5th element
    (node_sz, timeseries_sz, node_feature_sz, layers, dropout,
     pooling, cluster_num) = hyper_para_load(args=args, dataset=dataset[:4])

    logger.info(f"Dataset: {args.dataset}, Nodes: {node_sz}, Features: {node_feature_sz}, Classes: {num_classes}")

    # ================================================================
    # Phase 1: DARTS Architecture Search
    # ================================================================
    logger.info("=" * 60)
    logger.info("Phase 1: DARTS Architecture Search")
    logger.info("=" * 60)

    search_train_loader, search_val_loader, search_test_loader = \
        create_search_dataloaders(args, dataset)

    # Build NAS network with architecture parameters
    nas_model = NASNetwork(
        args=args,
        node_sz=node_sz,
        time_series_sz=timeseries_sz,
        corr_pearson_sz=node_feature_sz,
        layers=layers,
        dropout=dropout,
        cluster_num=cluster_num,
        pooling=pooling,
        num_classes=num_classes,
    )

    arch_params = count_param(nas_model)
    logger.info(f"NAS model parameters (incl. arch): {arch_params}")

    # Run DARTS search
    nas_model, search_history, best_arch = run_darts_search(
        model=nas_model,
        args=args,
        train_loader=search_train_loader,
        val_loader=search_val_loader,
        test_loader=search_test_loader,
        logger=logger,
    )

    # Generate unique config identifier for file naming
    cid = _config_id(args)
    result_dir = _get_exp_dir(args)

    # Save search history (per-config file)
    search_csv_path = os.path.join(result_dir, f'search_history_{cid}.csv')
    with open(search_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=search_history[0].keys())
        writer.writeheader()
        writer.writerows(search_history)
    logger.info(f"Search history saved to {search_csv_path}")

    # ================================================================
    # Phase 2: Architecture Derivation
    # ================================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Architecture Derivation")
    logger.info("=" * 60)

    layer_configs = derive_architecture(nas_model, top_k=args.top_k)
    arch_desc = describe_architecture(layer_configs)
    logger.info(f"\nDiscovered Architecture:\n{arch_desc}")

    # Save architecture description (per-config file)
    arch_txt_path = os.path.join(result_dir, f'architecture_{cid}.txt')
    with open(arch_txt_path, 'w') as f:
        f.write(f"Discovered Architecture (top_k={args.top_k})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Config: arch_lr={args.arch_lr}, layers={args.layers}, "
                f"top_k={args.top_k}, dropout={args.dropout}, "
                f"cluster_num={args.cluster_num}\n")
        f.write("=" * 60 + "\n")
        f.write(arch_desc)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write(f"Best architecture epoch: ")
        f.write(str(best_arch['epoch'] if best_arch else 'N/A'))
        f.write(" (final-epoch derivation)")
    logger.info(f"Architecture description saved to {arch_txt_path}")

    # ================================================================
    # Phase 3 & 4: Retrain Derived Architecture (Multiple Runs)
    # ================================================================
    logger.info("=" * 60)
    logger.info("Phase 3 & 4: Retrain Derived Architecture")
    logger.info("=" * 60)

    # Re-create dataloaders with original splits for training
    dataloaders = init_stratified_dataloader(args, *dataset)
    train_loader = dataloaders["train_dataloader"]
    val_loader = dataloaders["val_dataloader"]
    test_loader = dataloaders["test_dataloader"]

    run_acc_list, run_roc_list = [], []
    run_sen_list, run_spec_list = [], []

    for run_idx in range(args.runs):
        logger.info(f"\n--- Retrain Run {run_idx + 1}/{args.runs} ---")
        fix_seed(args.seed + run_idx)

        # Build derived model with discovered architecture
        derived_model = DerivedNetwork(
            args=args,
            node_sz=node_sz,
            time_series_sz=timeseries_sz,
            corr_pearson_sz=node_feature_sz,
            layers=layers,
            dropout=dropout,
            cluster_num=cluster_num,
            pooling=pooling,
            layer_configs=layer_configs,
            num_classes=num_classes,
        )

        model_params = count_param(derived_model)
        logger.info(f"Derived model parameters: {model_params}")

        acc, roc, sen, spec = run_derived_training(
            model=derived_model,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            logger=logger,
        )

        run_acc_list.append(acc)
        run_roc_list.append(roc)
        run_sen_list.append(sen)
        run_spec_list.append(spec)

        logger.info(f"Run {run_idx + 1} results: "
                     f"Acc={acc:.4f}, AUC={roc:.4f}, "
                     f"Sen={sen:.4f}, Spec={spec:.4f}")

    # ================================================================
    # Compute and save final metrics
    # ================================================================
    acc_mean, acc_std = np.mean(run_acc_list), np.std(run_acc_list)
    roc_mean, roc_std = np.mean(run_roc_list), np.std(run_roc_list)
    sen_mean, sen_std = np.mean(run_sen_list), np.std(run_sen_list)
    spec_mean, spec_std = np.mean(run_spec_list), np.std(run_spec_list)

    logger.info("\n" + "=" * 60)
    logger.info(f"Final Results after {args.runs} runs on {args.dataset}:")
    logger.info(f"  ROC-AUC:  {roc_mean * 100:.2f}% ± {roc_std * 100:.2f}")
    logger.info(f"  Accuracy: {acc_mean * 100:.2f}% ± {acc_std * 100:.2f}")
    logger.info(f"  Sensitivity: {sen_mean * 100:.2f}% ± {sen_std * 100:.2f}")
    logger.info(f"  Specificity: {spec_mean * 100:.2f}% ± {spec_std * 100:.2f}")

    # ---- Save per-run details (per-config file) ----
    run_detail_path = os.path.join(result_dir, f'run_details_{cid}.csv')
    with open(run_detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'acc', 'roc_auc', 'sensitivity', 'specificity'])
        for i in range(args.runs):
            writer.writerow([
                i + 1,
                f"{run_acc_list[i]:.4f}",
                f"{run_roc_list[i]:.4f}",
                f"{run_sen_list[i]:.4f}",
                f"{run_spec_list[i]:.4f}",
            ])
        writer.writerow([])
        writer.writerow(['mean', f"{acc_mean:.4f}", f"{roc_mean:.4f}",
                         f"{sen_mean:.4f}", f"{spec_mean:.4f}"])
        writer.writerow(['std', f"{acc_std:.4f}", f"{roc_std:.4f}",
                         f"{sen_std:.4f}", f"{spec_std:.4f}"])
    logger.info(f"Run details saved to {run_detail_path}")

    # ---- Save to master results CSV (append mode) ----
    result_csv_path = os.path.join(result_dir, f'{args.dataset}_NAS.csv')
    file_exists = os.path.exists(result_csv_path)

    with open(result_csv_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(result_csv_path) == 0:
            writer.writerow([
                'config_id',
                'roc_mean', 'roc_std', 'acc_mean', 'acc_std',
                'sen_mean', 'sen_std', 'spec_mean', 'spec_std',
                'seed', 'runs', 'search_epochs', 'epochs',
                'batch_size', 'base_lr', 'target_lr', 'wd',
                'arch_lr', 'arch_wd', 'zero_one_lambda',
                'layers', 'top_k', 'activation', 'dropout', 'cluster_num',
                'arch_description',
            ])
        writer.writerow([
            cid,
            f"{roc_mean * 100:.2f}", f"{roc_std * 100:.2f}",
            f"{acc_mean * 100:.2f}", f"{acc_std * 100:.2f}",
            f"{sen_mean * 100:.2f}", f"{sen_std * 100:.2f}",
            f"{spec_mean * 100:.2f}", f"{spec_std * 100:.2f}",
            args.seed, args.runs, args.search_epochs, args.epochs,
            args.batch_size, args.base_lr, args.target_lr, args.weight_decay,
            args.arch_lr, args.arch_weight_decay,
            getattr(args, 'zero_one_lambda', ''),
            args.layers, args.top_k,
            args.activation, args.dropout, args.cluster_num,
            arch_desc.replace('\n', ' | ').replace(',', ';'),
        ])
    logger.info(f"Results appended to {result_csv_path}")

    return {
        'roc_mean': roc_mean, 'roc_std': roc_std,
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'sen_mean': sen_mean, 'sen_std': sen_std,
        'spec_mean': spec_mean, 'spec_std': spec_std,
    }


def main():
    args = get_args()
    print("=" * 60)
    print("Neural Architecture Search")
    print("=" * 60)
    print(f"Arguments: {args}")
    run_nas_pipeline(args)


if __name__ == '__main__':
    main()
