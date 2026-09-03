# BrainNAS: Neural Architecture Search for Brain Network Analysis

Brain‑network analysis plays a key role in studying neurological disorders and computer‑aided diagnosis. However, mainstream message‑passing paradigms (GNNs, Transformers) fit poorly with brain functional connectivity data: the Pearson correlation matrix acts as both graph topology and node features, limiting neighborhood aggregation and attention. Though the Brain Quadratic Network (BQN) solves this via Hadamard‑product quadratic operations, its hand‑crafted structure fails to adapt to diverse brain datasets, parcellation atlases and clinical tasks. We present BrainNAS, the first neural architecture search framework for brain‑network analysis to auto‑identify optimal architectures. It provides a layer‑wise search space with candidate operations. Searched architectures are discretized and retrained for evaluation. Tests over four datasets and two atlases covering autism, Parkinson’s disease and other neurological illnesses show BrainNAS surpasses hand‑designed BQN variants and message‑passing baselines to yield state‑of‑the‑art classification results. Ablation, visualization, interpretability and hyperparameter analyses confirm its efficacy and generalizability.

![](fig/brainnas.png)

## Overview

BQN replaces message passing in brain network modeling with Hadamard-product-based
quadratic operations:

```
H^l = H^{l-1} ⊙ (A · W_A) + (H^{l-1} ⊙ H^{l-1}) · W_H
```

Instead of hand-designing this update rule, BrainNAS searches over a
strictly larger space of 42 candidate computations per layer:

- **Tensor sources** (4): `H^{l-1}`, `A` (adjacency / Pearson correlation), `H^0`
  (initial features / skip connection), `H^{l-1} ⊙ H^{l-1}` (quadratic term).
- **Element-wise ops** (3): Identity, Linear, MLP.
- **Combine ops** (3): Hadamard product, Addition, Concatenation+Projection.
- **Candidates per layer**: 12 unary (4 sources × 3 elem ops) + 30 binary
  (10 unordered source pairs × 3 combine ops) = **42**.

## Pipeline

1. **Search** — bi-level optimization with FairDARTS (200 epochs): architecture
   parameters α are updated on the validation set with
   `L_arch = L_CE + λ · L_01`; model weights are updated on the training set.
2. **Derivation** — final-epoch sigmoid gates are thresholded into a discrete
   architecture (top-K operations per layer).
3. **Retraining** — the derived network is trained from scratch (200 epochs ×
   5 seeds) with mixup augmentation and cosine learning-rate scheduling.
4. **Evaluation** — ACC / AUC / SEN / SPEC reported as mean ± std over 5 runs.

## Installation

```bash
conda create -n dgl python=3.10 -y
conda activate dgl
pip install torch scikit-learn nilearn matplotlib seaborn
```

Data (`.npy` files containing `timeseries`, `corr`, `label`; `site` optional)
should be placed in the `--data_dir` directory. Supported naming conventions:

- Original BQN format: `abide.npy`
- Per-atlas format: `{dataset}_full_{atlas}.npy`, e.g. `taowu_full_AAL116.npy`

## Quick Start

```bash
# Single architecture search + retrain configuration
python main_nas.py \
    --dataset ABIDE \
    --batch_size 16 \
    --base_lr 1e-4 --weight_decay 1e-5 \
    --arch_lr 3e-3 \
    --layers 4 --top_k 4 --cluster_num 8 \
    --search_epochs 200 --epochs 200 \
    --zero_one_lambda 5.0 \
    --exp_name exp_demo
```

All results (search history, architecture files, per-run details, summary CSV)
are written to `result/{exp_name}/`.

## Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--zero_one_lambda` | 1.0 | Weight λ of the Zero-One regularization (paper: 5.0 / 10.0) |
| `--arch_lr` | 3e-4 | Architecture parameter learning rate η_α (paper: 1e-3..5e-3) |
| `--search_epochs` | 50 | Search epochs (paper: 200) |
| `--layers` | 3 | Number of NAS layers L |
| `--top_k` | 3 | Operations retained per layer after search |
| `--cluster_num` | 4 | DEC cluster count C |
| `--base_lr` / `--weight_decay` | 1e-4 / 1e-4 | Model optimizer settings for retraining |
| `--epochs` | 200 | Retraining epochs |
| `--runs` | 5 | Independent evaluation runs |

## Code Structure

```
BQN-NAS-FairDARTS/
├── main_nas.py          # Pipeline: search → derive → retrain → evaluate
├── parse_nas.py         # CLI arguments
├── train_search.py      # FairDARTS bi-level search loop (final-epoch derivation)
├── train_eval.py        # Derived-architecture retraining with mixup + cosine LR
├── utils_nas.py         # Multi-dataset data loading & utilities
├── model/
│   ├── nas_ops.py       # Search space: ops, combine ops, source-pair indices
│   ├── nas_layer.py     # FairDARTS layer: sigmoid gates + Zero-One loss
│   ├── nas_network.py   # Stacked searchable layers + DEC readout + classifier
│   └── nas_derived.py   # Architecture discretization & fixed network
└── result/              # Experiment outputs (one folder per exp_name)
```

## Results Format

Each experiment folder contains:

- `{dataset}_NAS.csv` — one row per hyperparameter configuration with ACC / AUC /
  SEN / SPEC (mean ± std) and the discovered architecture description.
- `architecture_{config_id}.txt` — human-readable per-layer architecture with
  sigmoid gate values.
- `search_history_{config_id}.csv` — per-epoch search metrics.
- `run_details_{config_id}.csv` — per-run metrics for the 5 evaluation seeds.
