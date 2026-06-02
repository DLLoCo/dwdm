"""
Graph Convolution utilities for CCRNN.
=====================================
Provides:
  - GraphConv: single-layer diffusion graph convolution (used inside GRU gates)
  - AdjacencyLearner: learns initial A^(0) from node embeddings + coupled mapping
"""
import torch
import torch.nn as nn


class GraphConv(nn.Module):
    """
    Diffusion graph convolution: h = sum_{k=0}^{K} A^k @ X @ theta_k

    Used inside each GRU gate to replace linear transforms.
    """

    def __init__(self, in_dim, out_dim, cheb_k=3, bias=True):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_dim, out_dim))
            for _ in range(cheb_k)
        ])
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        for w in self.weights:
            nn.init.xavier_uniform_(w)

    def forward(self, x, adj):
        """
        x:   (B, N, D_in)
        adj: (N, N) or (B, N, N)
        out: (B, N, D_out)
        """
        out = x @ self.weights[0]           # k=0: identity diffusion
        power = x
        for k in range(1, self.cheb_k):
            if adj.dim() == 2:
                power = torch.matmul(adj, power)
            else:
                power = torch.bmm(adj, power)
            out = out + power @ self.weights[k]
        if self.bias is not None:
            out = out + self.bias
        return out


class AdjacencyLearner(nn.Module):
    """
    Learns the initial adjacency A^(0) and coupled mappings A^(m+1) = psi(A^(m)).

    Args:
        num_nodes:  N
        embed_dim:  dimension of node embeddings (L in paper)
        num_layers: number of GRU layers (M), need M-1 coupled mappings
    """

    def __init__(self, num_nodes, embed_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Source and target node embeddings for asymmetric learned graph
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

        # Coupled mappings: psi^(m) transforms A^(m) -> A^(m+1)
        if num_layers > 1:
            self.coupled_maps = nn.ModuleList([
                nn.Linear(num_nodes, num_nodes, bias=False)
                for _ in range(num_layers - 1)
            ])
            for cm in self.coupled_maps:
                nn.init.eye_(cm.weight)

        # Multi-level aggregation weights
        self.agg_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def get_all_adj(self, base_adj=None):
        """
        Compute adjacency matrices for all layers.

        Args:
            base_adj: (N, N) optional fixed adjacency to fuse with

        Returns:
            list of M adjacency matrices, each (N, N)
        """
        # A^(0): from learned embeddings
        learned = torch.softmax(torch.relu(self.E1 @ self.E2.t()), dim=-1)

        if base_adj is not None:
            adj = 0.5 * base_adj + 0.5 * learned
        else:
            adj = learned

        adjs = [adj]

        # Coupled mappings for upper layers
        for m in range(self.num_layers - 1):
            adj = torch.relu(self.coupled_maps[m](adj))
            adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            adjs.append(adj)

        return adjs

    def aggregate(self, layer_outputs):
        """
        Multi-level aggregation: weighted sum of all layers' outputs.

        Args:
            layer_outputs: list of M tensors, each (B, N, D)
        Returns:
            (B, N, D) aggregated output
        """
        alpha = torch.softmax(self.agg_weights, dim=0)
        return sum(alpha[m] * layer_outputs[m] for m in range(self.num_layers))
