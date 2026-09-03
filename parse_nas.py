"""Argument parsing for BrainNAS."""

import argparse


def get_args():
    """Create argument parser for BrainNAS."""
    parser = argparse.ArgumentParser(description='BrainNAS')

    # ---- Paths ----
    parser.add_argument('--device', type=int, default=0, help="CUDA device id")
    parser.add_argument('--root_path', type=str,
                        default="./code/BrainNAS")
    parser.add_argument('--data_dir', type=str,
                        default="./data")
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment folder name (auto-generated if not set)')

    # ---- Dataset ----
    parser.add_argument('--dataset', default='ABIDE',
                        help='Dataset name (e.g., ABIDE, ABIDE_AAL116, PPMI_schaefer100)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', default=5, type=int, help='Number of repeat runs for evaluation')
    parser.add_argument('--Train_prop', default=0.7, type=float)
    parser.add_argument('--Val_prop', default=0.1, type=float)
    parser.add_argument('--batch_size', type=int, default=16)

    # ---- Model architecture ----
    parser.add_argument('--layers', type=int, default=3, help='Number of NAS layers')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='leaky_relu',
                        choices=['gelu', 'leaky_relu', 'elu'])
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--cluster_num', type=int, default=4)

    # ---- NAS Search ----
    parser.add_argument('--search_epochs', type=int, default=50,
                        help='Epochs for DARTS architecture search')
    parser.add_argument('--arch_lr', type=float, default=3e-4,
                        help='Learning rate for architecture parameters α')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        help='Weight decay for architecture parameters')

    # ---- Evaluation (derived model) ----
    parser.add_argument('--epochs', type=int, default=200,
                        help='Epochs for retraining derived architecture')
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--target_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # ---- FairDARTS Zero-One Loss ----
    parser.add_argument('--zero_one_lambda', type=float, default=1.0,
                        help='Weight for Zero-One loss (pushes sigmoid gates to 0/1)')

    # ---- Derivation ----
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top operations to keep per layer after search')

    return parser.parse_args()
