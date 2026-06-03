"""
TrafficPredNet v5 — TC + CGC + Flow Gate + Sin/Cos Time Embedding
=================================================================
v4 → v5 change: TimeEmbedding from learned nn.Embedding to sin/cos
  - Captures cyclical nature: slot 47 (23:30) ≈ slot 0 (00:00)
  - Multi-frequency: low freq = morning/evening, high freq = 8:00/8:30
  - Zero parameters in encoding itself (only a linear projection)
"""
import math
import torch
import torch.nn as nn
from model.cgc import AdjacencyLearner
from model.st_block import STBlock


class TimeEmbedding(nn.Module):
    """Sin/cos multi-frequency encoding for hour-of-day and day-of-week."""

    def __init__(self, embed_dim, num_freqs=4):
        super().__init__()
        self.num_freqs = num_freqs
        # hour: num_freqs×2 dims, dow: num_freqs×2 dims → project to embed_dim
        raw_dim = num_freqs * 2 * 2
        self.proj = nn.Linear(raw_dim, embed_dim)

    def forward(self, hour_idx, dow_idx):
        """hour_idx: (B,) 0-47, dow_idx: (B,) 0-6 → (B, embed_dim)"""
        feats = []
        for f in range(1, self.num_freqs + 1):
            feats.append(torch.sin(2 * math.pi * hour_idx / 48.0 * f))
            feats.append(torch.cos(2 * math.pi * hour_idx / 48.0 * f))
            feats.append(torch.sin(2 * math.pi * dow_idx / 7.0 * f))
            feats.append(torch.cos(2 * math.pi * dow_idx / 7.0 * f))
        raw = torch.stack(feats, dim=-1).float()  # (B, num_freqs*4)
        return self.proj(raw)                      # (B, embed_dim)


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
        h = self.input_proj(x)

        if self.use_time_embed and hour_idx is not None and dow_idx is not None:
            t_emb = self.time_embed(hour_idx, dow_idx)
            h = h + t_emb[:, None, None, :]

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
