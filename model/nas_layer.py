"""
FairDARTS-based searchable NAS layer for brain network computation.

Key difference from DARTS:
  - sigmoid(alpha) instead of softmax(alpha): each operator independently gated,
    no zero-sum competition between candidates.
  - Zero-One loss pushes sigmoid outputs toward 0/1 extremes.
  - No temperature annealing needed.

Architecture parameters alpha:
  - alpha_src_op:  [4 x 3] — elem op gate per source (sigmoid-gated)
  - alpha_all:     [42]    — candidate gate (sigmoid-gated, 0=off, 1=on)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nas_ops import (
    get_elem_ops,
    get_combine_ops,
    build_source_tensors,
    SOURCE_PAIRS,
    NUM_SOURCES,
    NUM_ELEM_OPS,
    NUM_PAIRS,
    NUM_COMBINE_OPS,
    NUM_UNARY,
    NUM_BINARY,
    NUM_ALL_CANDIDATES,
)

# Pre-compute pair indices for vectorized gathering
_S1_IDX = torch.tensor([s1 for s1, s2 in SOURCE_PAIRS], dtype=torch.long)
_S2_IDX = torch.tensor([s2 for s1, s2 in SOURCE_PAIRS], dtype=torch.long)


class NASLayer(nn.Module):
    """FairDARTS layer: sigmoid gates + Zero-One loss."""

    def __init__(self, dim: int, activation: str = 'leaky_relu', dropout: float = 0.0):
        super().__init__()

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.activation = {
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
        }[activation]

        self.elem_ops = get_elem_ops(dim, dim)
        self.combine_ops = get_combine_ops(dim, dim)

        # Architecture parameters — learned via FairDARTS
        self.alpha_src_op = nn.Parameter(
            1e-3 * torch.randn(NUM_SOURCES, NUM_ELEM_OPS))
        self.alpha_all = nn.Parameter(
            1e-3 * torch.randn(NUM_ALL_CANDIDATES))

        self.register_buffer('s1_idx', _S1_IDX, persistent=False)
        self.register_buffer('s2_idx', _S2_IDX, persistent=False)

        self._arch_parameters = [self.alpha_src_op, self.alpha_all]

    def arch_parameters(self) -> list:
        return self._arch_parameters

    def zero_one_loss(self) -> torch.Tensor:
        """FairDARTS Zero-One loss: pushes sigmoid toward 0 or 1."""
        src_gates = torch.sigmoid(self.alpha_src_op)
        all_gates = torch.sigmoid(self.alpha_all)
        return (src_gates * (1 - src_gates)).mean() + \
               (all_gates * (1 - all_gates)).mean()

    def forward(self, h_prev, A, h0):
        B, N, D = h_prev.shape

        # --- Step 1: Build sources [4, B, N, D] ---
        sources = build_source_tensors(h_prev, A, h0)
        src_stack = torch.stack(sources, dim=0)

        # --- Step 2: Apply element-wise ops [3, 4, B, N, D] ---
        op_outputs = []
        for op in self.elem_ops:
            op_outputs.append(_apply_op_batched(op, src_stack))
        op_stack = torch.stack(op_outputs, dim=0)

        # --- Step 3: Unary candidates [12, B, N, D] ---
        unary = op_stack.reshape(NUM_UNARY, B, N, D)

        # --- Step 4: Source mixing (sigmoid-gated) [4, B, N, D] ---
        src_gates = torch.sigmoid(self.alpha_src_op)
        src_gates = src_gates / (src_gates.sum(dim=-1, keepdim=True) + 1e-8)
        w = src_gates.permute(1, 0).reshape(3, 4, 1, 1, 1)
        transformed = (op_stack * w).sum(dim=0)

        # --- Step 5: Binary candidates [30, B, N, D] ---
        t1 = transformed[self.s1_idx]
        t2 = transformed[self.s2_idx]

        hadamard = t1 * t2
        add = t1 + t2
        cat_input = torch.cat([t1, t2], dim=-1)
        concat = self.combine_ops[2].proj(cat_input)
        binary = torch.cat([hadamard, add, concat], dim=0)

        # --- Step 6: Weighted sum with sigmoid gates, normalized [42, B, N, D] ---
        all_candidates = torch.cat([unary, binary], dim=0)
        cand_gates = torch.sigmoid(self.alpha_all)
        # Normalize to sum=1 for stable output scale (gradients still independent per gate)
        cand_gates = cand_gates / (cand_gates.sum() + 1e-8)
        h_new = (all_candidates * cand_gates.view(NUM_ALL_CANDIDATES, 1, 1, 1)).sum(dim=0)

        h_new = self.activation(h_new)
        h_new = self.dropout(h_new)
        return h_new

    # ---- Architecture derivation ----

    def get_derived_ops(self, top_k: int = 3) -> list:
        """Derive discrete architecture: keep ops with sigmoid(alpha) > 0.5,
        fall back to top-k by gate value."""
        with torch.no_grad():
            all_gates = torch.sigmoid(self.alpha_all).cpu().numpy()

        scored = []
        # Unary (0..11)
        for flat_idx in range(NUM_UNARY):
            s_idx = flat_idx // NUM_ELEM_OPS
            o_idx = flat_idx % NUM_ELEM_OPS
            scored.append({
                'type': 'unary', 'src_idx': s_idx, 'op_idx': o_idx,
                'weight': float(all_gates[flat_idx]), 'flat_idx': flat_idx,
            })
        # Binary (12..41)
        for flat_idx in range(NUM_UNARY, NUM_ALL_CANDIDATES):
            rel_idx = flat_idx - NUM_UNARY
            p_idx = rel_idx // NUM_COMBINE_OPS
            c_idx = rel_idx % NUM_COMBINE_OPS
            s1, s2 = SOURCE_PAIRS[p_idx]
            scored.append({
                'type': 'binary', 'pair_idx': p_idx, 'src1': s1, 'src2': s2,
                'combine_idx': c_idx,
                'weight': float(all_gates[flat_idx]), 'flat_idx': flat_idx,
            })
        scored.sort(key=lambda x: x['weight'], reverse=True)

        # Prefer ops with gate > 0.5, then fall back to top-k
        active = [s for s in scored if s['weight'] > 0.5]
        if len(active) >= top_k:
            return active[:top_k]
        return scored[:top_k]

    def get_derived_src_ops(self) -> dict:
        """Best elem op per source by sigmoid gate."""
        with torch.no_grad():
            src_gates = torch.sigmoid(self.alpha_src_op).cpu().numpy()
        result = {}
        for s in range(NUM_SOURCES):
            best_o = int(src_gates[s].argmax())
            result[s] = (best_o, float(src_gates[s][best_o]))
        return result


def _apply_op_batched(op, x):
    """Apply element-wise op to stacked sources [S, B, N, D]."""
    S, B, N, D = x.shape
    if isinstance(op, nn.Module) and not hasattr(op, 'linear') and not hasattr(op, 'mlp'):
        return x
    x_flat = x.reshape(S * B, N, D)
    out_flat = op(x_flat)
    return out_flat.reshape(S, B, N, D)
