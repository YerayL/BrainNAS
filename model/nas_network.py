"""
Full NAS Network for brain network analysis.

Stacks NASLayers (searchable) + DEC clustering readout + classifier.
During search, architecture parameters from all layers are jointly optimized.
"""

import sys
import os
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nas_layer import NASLayer


def _load_cluster_pooling():
    """Load cluster_pooling module from BQN-demo using importlib."""
    _bqn_demo = os.path.join(os.path.dirname(__file__), '..', '..', 'BQN-demo')
    _cp_path = os.path.join(_bqn_demo, 'model', 'cluster_pooling.py')
    spec = importlib.util.spec_from_file_location('cluster_pooling', _cp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NASNetwork(nn.Module):
    """
    NAS Network with searchable layers.

    Architecture:
      - Stack of NASLayers (each with its own α for DARTS)
      - DEC clustering for graph-level readout
      - MLP classifier
    """

    def __init__(self, args, node_sz: int, time_series_sz: int,
                 corr_pearson_sz: int, layers: int, dropout: float = 0.0,
                 cluster_num: int = 4, pooling: bool = True,
                 num_classes: int = 2):
        super().__init__()

        forward_dim = corr_pearson_sz

        # Stack of searchable NAS layers
        self.nas_layers = nn.ModuleList()
        for i in range(layers):
            nas_layer = NASLayer(
                dim=forward_dim,
                activation=args.activation,
                dropout=dropout,
            )
            self.nas_layers.append(nas_layer)

        self.pooling = pooling

        # DEC clustering readout (reuses BQN-demo's DEC implementation)
        if pooling:
            _cp = _load_cluster_pooling()
            DEC = _cp.DEC

            encoder_hidden_size = 32
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

    def arch_parameters(self) -> list:
        """Collect architecture parameters from all NAS layers."""
        params = []
        for layer in self.nas_layers:
            params.extend(layer.arch_parameters())
        return params

    def zero_one_loss(self) -> torch.Tensor:
        """Sum of Zero-One loss across all NAS layers."""
        loss = 0.0
        for layer in self.nas_layers:
            loss = loss + layer.zero_one_loss()
        return loss

    def model_parameters(self):
        """Return model weights (excluding architecture parameters)."""
        arch_param_ids = set()
        for p in self.arch_parameters():
            arch_param_ids.add(id(p))

        for p in self.parameters():
            if id(p) not in arch_param_ids:
                yield p

    def forward(self, timeseries: torch.Tensor,
                corr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NAS layers + DEC + classifier.

        Args:
            timeseries: [B, N, T] time series data (not used by NAS layers directly)
            corr:       [B, N, N] Pearson correlation / adjacency

        Returns:
            logits: [B, 2] classification logits
        """
        bz, node_sz, corr_sz = corr.shape

        # Initial hidden state H^0 = corr (pearson correlation as node features)
        h = corr.clone()
        h0 = corr.clone()  # Skip connection source

        # Pass through searchable NAS layers
        for nas_layer in self.nas_layers:
            h = nas_layer(h, corr, h0)

        # DEC clustering readout
        topo = h
        graph_level_topo, assignment = self.dec(topo)
        graph_level_topo = self.dim_reduction(graph_level_topo)
        graph_level_topo = graph_level_topo.reshape(bz, -1)
        result = self.fc(graph_level_topo)

        return result
