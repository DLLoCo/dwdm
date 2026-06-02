"""
TrafficPredNet v4 — TC + CGC + Flow Gate + Time Embedding
==========================================================
All three challenges addressed:
  Challenge 1 (ST coupling):     TC + CGC in each ST block
  Challenge 2 (hotspot migration): CGC coupled adjacency + multi-level agg
  Challenge 3 (external events):  Time Embedding + Flow Gate (OD attention)

Key insight: Time Embedding tells Flow Gate "what time it is",
so attention can correctly select morning/evening/night OD patterns.
"""
import torch
import torch.nn as nn
from model.cgc import AdjacencyLearner
from model.st_block import STBlock


class TimeEmbedding(nn.Module):
    """Encode hour-of-day (0-47) and day-of-week (0-6) as embeddings."""

    def __init__(self, embed_dim):
        super().__init__()
        self.hour_embed = nn.Embedding(48, embed_dim)
        self.dow_embed = nn.Embedding(7, embed_dim)

    def forward(self, hour_idx, dow_idx):
        """hour_idx: (B,), dow_idx: (B,) → (B, D)"""
        return self.hour_embed(hour_idx) + self.dow_embed(dow_idx)


class TrafficPredNet(nn.Module):

    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=3, cheb_k=3, embed_dim=50, kernel_size=3,
                 dropout=0.1, use_flow_gate=False, use_time_embed=False,
                 num_patterns=48):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_flow_gate = use_flow_gate
        self.use_time_embed = use_time_embed

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_time_embed:
            self.time_embed = TimeEmbedding(hidden_dim)

        self.adj_learner = AdjacencyLearner(num_nodes, embed_dim, num_layers)

        self.st_blocks = nn.ModuleList([
            STBlock(hidden_dim, num_nodes, kernel_size, cheb_k, dropout,
                    use_flow_gate=use_flow_gate, num_patterns=num_patterns)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, base_adj=None, od_patterns=None,
                hour_idx=None, dow_idx=None):
        """
        x:           (B, T, N, C)
        base_adj:    (N, N)
        od_patterns: (P, N, N) for Flow Gate
        hour_idx:    (B,) 0-47
        dow_idx:     (B,) 0-6
        """
        h = self.input_proj(x)  # (B, T, N, D)

        # Add time embedding: broadcast to all T steps and N nodes
        if self.use_time_embed and hour_idx is not None and dow_idx is not None:
            t_emb = self.time_embed(hour_idx, dow_idx)  # (B, D)
            h = h + t_emb[:, None, None, :]  # (B, T, N, D)

        adjs = self.adj_learner.get_all_adj(base_adj)

        for i, block in enumerate(self.st_blocks):
            h = block(h, adjs[i], od_patterns=od_patterns)

        h = h.mean(dim=1)
        pred = self.output_proj(h)
        return pred.unsqueeze(1)

    @classmethod
    def from_config(cls, config):
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
            kernel_size=mcfg.get('kernel_size', 3),
            dropout=mcfg.get('dropout', 0.1),
            use_flow_gate=mcfg.get('use_flow_gate', False),
            use_time_embed=mcfg.get('use_time_embed', False),
        )
