"""
Derived (discretized) architecture after DARTS search.

After training with DARTS, we discretize the continuous α into a fixed
architecture by selecting top-k operations per layer. This module creates
a standard (non-searchable) network using only the selected operations.

This derived network is then retrained from scratch for final evaluation.
"""

import sys
import os
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_cluster_pooling():
    """Load cluster_pooling module from BQN-demo using importlib."""
    _bqn_demo = os.path.join(os.path.dirname(__file__), '..', '..', 'BQN-demo')
    _cp_path = os.path.join(_bqn_demo, 'model', 'cluster_pooling.py')
    spec = importlib.util.spec_from_file_location('cluster_pooling', _cp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from .nas_ops import (
    get_elem_ops,
    get_combine_ops,
    build_source_tensors,
    SOURCE_PAIRS,
    NUM_COMBINE_OPS,
)


class DerivedLayer(nn.Module):
    """
    A fixed layer using only the top-k selected operations from NAS search.

    H^l = activation( Σ_{k} candidate_k )  where candidates are top-k by α weight.
    """

    def __init__(self, dim: int, selected_ops: list, src_op_map: dict,
                 activation: str = 'leaky_relu', dropout: float = 0.0):
        """
        Args:
            dim: feature dimension
            selected_ops: list of dicts with keys [pair_idx, combine_idx, weight]
            src_op_map: dict mapping src_idx -> (best_op_idx, weight)
            activation: activation function name
            dropout: dropout rate
        """
        super().__init__()

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.activation = {
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }[activation]

        # Create element-wise ops (we need the specific ops for each source)
        self.elem_ops = get_elem_ops(dim, dim)
        self.combine_ops = get_combine_ops(dim, dim)

        # Store selected operations
        self.selected_ops = selected_ops  # list of dicts
        self.src_op_map = src_op_map      # dict: src_idx -> best_op_idx

        # Total weight for normalization (soft-selection still useful)
        total_w = sum(op['weight'] for op in selected_ops)
        self.op_weights = [op['weight'] / total_w for op in selected_ops]

    def forward(self, h_prev: torch.Tensor, A: torch.Tensor,
                h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using only selected operations.

        Args:
            h_prev: H^{l-1} [B, N, D]
            A: adjacency [B, N, N]
            h0: initial features [B, N, D]

        Returns:
            H^l: [B, N, D]
        """
        sources = build_source_tensors(h_prev, A, h0)

        # Apply the best element-wise op to each source
        transformed = []
        for s_idx, src in enumerate(sources):
            best_op_idx = self.src_op_map[s_idx]
            transformed.append(self.elem_ops[best_op_idx](src))

        # Compute selected candidates (unary + binary)
        outputs = []
        for i, sel in enumerate(self.selected_ops):
            if sel['type'] == 'unary':
                s_idx = sel['src_idx']
                o_idx = sel['op_idx']
                candidate = self.elem_ops[o_idx](sources[s_idx])
            else:
                p_idx = sel['pair_idx']
                c_idx = sel['combine_idx']
                s1, s2 = SOURCE_PAIRS[p_idx]
                t1 = transformed[s1]
                t2 = transformed[s2]
                candidate = self.combine_ops[c_idx](t1, t2)
            outputs.append(self.op_weights[i] * candidate)

        h_new = sum(outputs)
        h_new = self.activation(h_new)
        h_new = self.dropout(h_new)

        return h_new


class DerivedNetwork(nn.Module):
    """
    Fixed network built from the discovered NAS architecture.

    Uses top-k operations per layer (derived from α) with standard training.
    """

    def __init__(self, args, node_sz: int, time_series_sz: int,
                 corr_pearson_sz: int, layers: int, dropout: float = 0.0,
                 cluster_num: int = 4, pooling: bool = True,
                 layer_configs: list = None, num_classes: int = 2):
        """
        Args:
            layer_configs: list of (selected_ops, src_op_map) per layer from NAS search
        """
        super().__init__()

        forward_dim = corr_pearson_sz

        # Build derived layers from discovered architecture
        self.derived_layers = nn.ModuleList()
        if layer_configs is None:
            layer_configs = [([], {}) for _ in range(layers)]

        for i in range(layers):
            selected_ops, src_op_map = layer_configs[i]
            if not selected_ops:
                # Fallback: use BQN-like default (H ⊙ Linear(A) + Linear(H⊙H))
                selected_ops = [
                    {'type': 'binary', 'pair_idx': 1, 'combine_idx': 0, 'weight': 0.5},  # H ⊙ A
                    {'type': 'unary', 'src_idx': 3, 'op_idx': 1, 'weight': 0.5},          # Linear(H⊙H)
                ]
                src_op_map = {0: 0, 1: 1, 2: 0, 3: 1}  # H:Id, A:Linear, H0:Id, H2:Linear

            derived_layer = DerivedLayer(
                dim=forward_dim,
                selected_ops=selected_ops,
                src_op_map=src_op_map,
                activation=args.activation,
                dropout=dropout,
            )
            self.derived_layers.append(derived_layer)

        self.pooling = pooling

        if pooling:
            encoder_hidden_size = 32
            _cp = _load_cluster_pooling()
            DEC = _cp.DEC

            self.encoder = nn.Sequential(
                nn.Linear(forward_dim * node_sz, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, forward_dim * node_sz),
            )
            self.dec = DEC(
                cluster_number=cluster_num,
                hidden_dimension=forward_dim,
                encoder=self.encoder,
                orthogonal=True,
                freeze_center=True,
                project_assignment=True,
            )

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU(),
        )

        if pooling:
            self.fc = nn.Sequential(
                nn.Linear(8 * cluster_num, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
                nn.Linear(32, num_classes),
            )

    def forward(self, timeseries: torch.Tensor,
                corr: torch.Tensor) -> torch.Tensor:
        """Forward pass through derived layers + DEC + classifier."""
        bz, node_sz, corr_sz = corr.shape

        h = corr.clone()
        h0 = corr.clone()

        for layer in self.derived_layers:
            h = layer(h, corr, h0)

        topo = h
        graph_level_topo, assignment = self.dec(topo)
        graph_level_topo = self.dim_reduction(graph_level_topo)
        graph_level_topo = graph_level_topo.reshape(bz, -1)
        result = self.fc(graph_level_topo)

        return result


def derive_architecture(nas_network, top_k: int = 3) -> list:
    """
    Extract the discovered architecture from a trained NAS network.

    Args:
        nas_network: trained NASNetwork with learned α
        top_k: number of top operations to keep per layer

    Returns:
        list of (selected_ops, src_op_map) for each layer
    """
    layer_configs = []
    for layer in nas_network.nas_layers:
        selected_ops = layer.get_derived_ops(top_k=top_k)
        src_op_map = {s: op_idx for s, (op_idx, _) in layer.get_derived_src_ops().items()}
        layer_configs.append((selected_ops, src_op_map))
    return layer_configs


def describe_architecture(layer_configs: list) -> str:
    """Generate a human-readable description of the discovered architecture.

    Source names:
      0: H^{l-1} (previous hidden state)
      1: A (adjacency/correlation matrix)
      2: H^0 (initial features / skip connection)
      3: H^{l-1} ⊙ H^{l-1} (quadratic term)

    Combine op names:
      0: ⊙ (Hadamard product)
      1: + (addition)
      2: Concat (concatenation + projection)

    Elem op names:
      0: Identity
      1: Linear(·W)
      2: MLP(·)
    """
    src_names = ["H", "A", "H^0", "H⊙H"]
    combine_names = ["⊙", "+", "Concat"]
    op_names = ["Id", "Linear", "MLP"]

    lines = []
    for l_idx, (selected_ops, src_op_map) in enumerate(layer_configs):
        lines.append(f"--- Layer {l_idx} ---")
        lines.append(f"  Source ops: " + ", ".join(
            f"{src_names[s]}→{op_names[o]}" for s, o in src_op_map.items()
        ))
        lines.append(f"  Top terms:")
        for i, op in enumerate(selected_ops):
            if op.get('type') == 'unary':
                s_idx = op['src_idx']
                o_idx = op['op_idx']
                lines.append(
                    f"    {i+1}. {op_names[o_idx]}({src_names[s_idx]}) "
                    f"(α={op['weight']:.3f})"
                )
            else:
                s1, s2 = SOURCE_PAIRS[op['pair_idx']]
                c_name = combine_names[op['combine_idx']]
                lines.append(
                    f"    {i+1}. {src_names[s1]} {c_name} {src_names[s2]} "
                    f"(α={op['weight']:.3f})"
                )
    return "\n".join(lines)
