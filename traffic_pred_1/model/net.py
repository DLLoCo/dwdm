"""
TrafficPredNet — Main Model
============================
Architecture:
  1. M layers of CCGRU (graph-conv GRU), each with its own adjacency A^(m)
  2. Multi-level aggregation of all layers' final hidden states
  3. Output projection to predict next time step

Input:  (B, T, N, C)  — T time steps, N nodes, C features
Output: (B, 1, N, C)  — next 1 time step prediction
"""
import torch
import torch.nn as nn
from model.cgc import AdjacencyLearner
from model.ccgru_cell import CCGRUCell


class TrafficPredNet(nn.Module):
    """
    Args:
        num_nodes:  number of spatial nodes (N)
        input_dim:  input feature dimension (C, e.g. 2 for pickup+dropoff)
        output_dim: output feature dimension (same as input_dim typically)
        hidden_dim: GRU hidden state dimension
        num_layers: number of stacked CCGRU layers (M)
        cheb_k:     diffusion steps in graph convolution
        embed_dim:  node embedding dimension for adjacency learning
    """

    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=3, cheb_k=3, embed_dim=50):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # --- Adjacency learner: produces per-layer adj + aggregation ---
        self.adj_learner = AdjacencyLearner(num_nodes, embed_dim, num_layers)

        # --- Stacked CCGRU layers ---
        self.gru_cells = nn.ModuleList()
        for m in range(num_layers):
            cell_in_dim = input_dim if m == 0 else hidden_dim
            self.gru_cells.append(
                CCGRUCell(cell_in_dim, hidden_dim, num_nodes, cheb_k)
            )

        # --- Output projection: hidden_dim -> output_dim ---
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, base_adj=None):
        """
        Args:
            x:        (B, T, N, C)  input sequence
            base_adj: (N, N) optional fixed adjacency matrix (e.g., grid)

        Returns:
            pred:     (B, 1, N, output_dim)  prediction for next time step
        """
        B, T, N, C = x.shape
        device = x.device

        # Step 1: Get per-layer adjacency matrices
        adjs = self.adj_learner.get_all_adj(base_adj)

        # Step 2: Initialize hidden states for all layers
        h = [cell.init_hidden(B, N, device) for cell in self.gru_cells]

        # Step 3: Process each time step through all layers
        for t in range(T):
            inp = x[:, t, :, :]  # (B, N, C)

            for m in range(self.num_layers):
                h[m] = self.gru_cells[m](inp, h[m], adjs[m])
                inp = h[m]  # output of layer m is input to layer m+1

        # Step 4: Multi-level aggregation of final hidden states
        agg = self.adj_learner.aggregate(h)  # (B, N, hidden_dim)

        # Step 5: Project to output
        pred = self.output_proj(agg)  # (B, N, output_dim)
        pred = pred.unsqueeze(1)      # (B, 1, N, output_dim)

        return pred

    @classmethod
    def from_config(cls, config: dict):
        """Build model from config dict."""
        dcfg = config['data']
        mcfg = config['model']
        return cls(
            num_nodes=dcfg['num_nodes'],
            input_dim=dcfg['input_dim'],
            output_dim=dcfg['output_dim'],
            hidden_dim=mcfg['hidden_dim'],
            num_layers=mcfg['num_layers'],
            cheb_k=mcfg['cheb_k'],
            embed_dim=mcfg['embed_dim'],
        )
