"""
Element-wise operations and Combine operations for the NAS search space.

Element-wise Ops (O):
  - Identity: pass-through, no transformation
  - Linear: x @ W (learnable linear projection)
  - MLP: x -> Linear -> ReLU -> Linear (small multi-layer perceptron)

Combine Ops (C):
  - Hadamard: x1 ⊙ x2 (element-wise product)
  - Add: x1 + x2
  - Concat: [x1 || x2] @ W_proj (concatenation + linear projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Element-wise Operations
# ============================================================

class IdentityOp(nn.Module):
    """Identity operation: returns input unchanged."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearOp(nn.Module):
    """Linear projection: x @ W."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPOp(nn.Module):
    """Small MLP: Linear -> ReLU -> Linear."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================
# Combine Operations
# ============================================================

class HadamardCombine(nn.Module):
    """Hadamard product (element-wise multiplication): x1 ⊙ x2."""
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * x2


class AddCombine(nn.Module):
    """Element-wise addition: x1 + x2."""
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 + x2


class ConcatCombine(nn.Module):
    """Concatenation + linear projection: [x1 || x2] @ W_proj."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim, bias=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([x1, x2], dim=-1)
        return self.proj(cat)


# ============================================================
# Operation Factories
# ============================================================

def get_elem_ops(in_dim: int, out_dim: int) -> nn.ModuleList:
    """Create the 3 element-wise operations."""
    return nn.ModuleList([
        IdentityOp(),
        LinearOp(in_dim, out_dim),
        MLPOp(in_dim, out_dim),
    ])


def get_combine_ops(in_dim: int, out_dim: int) -> nn.ModuleList:
    """Create the 3 combine operations."""
    return nn.ModuleList([
        HadamardCombine(),
        AddCombine(),
        ConcatCombine(in_dim, out_dim),
    ])


# ============================================================
# Source Tensor Builder
# ============================================================

def build_source_tensors(h_prev: torch.Tensor, A: torch.Tensor,
                         h0: torch.Tensor) -> list:
    """
    Build the 4 source tensors from available inputs.

    S = {H^{l-1}, A, H^0, H^{l-1} ⊙ H^{l-1}}

    Args:
        h_prev: H^{l-1}, previous layer output [B, N, D]
        A: adjacency/correlation matrix [B, N, N]
        h0: initial input features (same as corr) [B, N, D]

    Returns:
        list of 4 tensors, each [B, N, D]
    """
    return [
        h_prev,           # S[0]: H^{l-1}
        A,                # S[1]: A (adjacency / correlation)
        h0,               # S[2]: H^0 (initial features, skip connection)
        h_prev * h_prev,  # S[3]: H^{l-1} ⊙ H^{l-1} (quadratic)
    ]


# Unordered source pairs (s1, s2) with s1 <= s2 (0-indexed into sources list)
# 10 pairs: (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3)
SOURCE_PAIRS = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3),
    (3, 3),
]

NUM_SOURCES = 4
NUM_ELEM_OPS = 3
NUM_COMBINE_OPS = 3
NUM_PAIRS = len(SOURCE_PAIRS)  # 10
NUM_UNARY = NUM_SOURCES * NUM_ELEM_OPS   # 12 unary candidates
NUM_BINARY = NUM_PAIRS * NUM_COMBINE_OPS  # 30 binary candidates
NUM_ALL_CANDIDATES = NUM_UNARY + NUM_BINARY  # 42 total
