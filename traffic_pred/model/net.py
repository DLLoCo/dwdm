"""
TrafficPredNet v6 — TC + CGC + Dual Cluster + Residual Prediction
==================================================================
Ablation flags:
  use_spatial_cluster:  Spatial Cluster Embedding (Challenge 2)
  use_temporal_cluster: Temporal Cluster Embedding (Challenge 3)
  use_residual:         Residual prediction (pred = last_obs + correction)
"""
import torch
import torch.nn as nn
from model.cgc import AdjacencyLearner
from model.st_block import STBlock


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.hour_embed = nn.Embedding(48, embed_dim)
        self.dow_embed = nn.Embedding(7, embed_dim)

    def forward(self, hour_idx, dow_idx):
        return self.hour_embed(hour_idx) + self.dow_embed(dow_idx)


class ClusterEmbedding(nn.Module):
    """Dual cluster conditioning — supports either or both clusters."""

    def __init__(self, n_spatial, n_temporal, embed_dim,
                 use_spatial=True, use_temporal=True):
        super().__init__()
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal

        if use_spatial:
            self.spatial_embed = nn.Embedding(n_spatial, embed_dim)
        if use_temporal:
            self.temporal_embed = nn.Embedding(n_temporal, embed_dim)

        # Gated fusion only when both active
        if use_spatial and use_temporal:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )

    def forward(self, spatial_ids, temporal_ids):
        """Returns (B, 1, N, D) additive embedding."""
        if self.use_spatial and self.use_temporal:
            s = self.spatial_embed(spatial_ids)        # (N, D)
            t = self.temporal_embed(temporal_ids)      # (B, D)
            B, N = t.shape[0], s.shape[0]
            s_exp = s.unsqueeze(0).expand(B, N, -1)
            t_exp = t.unsqueeze(1).expand(B, N, -1)
            gate = self.gate(torch.cat([s_exp, t_exp], dim=-1))
            out = gate * s_exp + (1 - gate) * t_exp
        elif self.use_spatial:
            s = self.spatial_embed(spatial_ids)        # (N, D)
            out = s.unsqueeze(0)                       # (1, N, D) → broadcasts
        elif self.use_temporal:
            t = self.temporal_embed(temporal_ids)      # (B, D)
            out = t.unsqueeze(1)                       # (B, 1, D) → broadcasts
        else:
            return None

        return out.unsqueeze(1)  # (B, 1, N, D) or broadcastable


class TrafficPredNet(nn.Module):

    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=3, cheb_k=3, embed_dim=50, kernel_size=3,
                 dropout=0.1, use_flow_gate=False, use_time_embed=False,
                 num_patterns=48,
                 use_spatial_cluster=False, use_temporal_cluster=False,
                 n_spatial_clusters=5, n_temporal_clusters=5,
                 use_residual=True,
                 # backward compat: old configs use 'use_cluster'
                 use_cluster=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_flow_gate = use_flow_gate
        self.use_time_embed = use_time_embed
        self.use_residual = use_residual

        # Handle old 'use_cluster' flag → both spatial+temporal
        if use_cluster and not use_spatial_cluster and not use_temporal_cluster:
            use_spatial_cluster = True
            use_temporal_cluster = True

        self.use_spatial_cluster = use_spatial_cluster
        self.use_temporal_cluster = use_temporal_cluster
        self.use_any_cluster = use_spatial_cluster or use_temporal_cluster

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        if use_time_embed:
            self.time_embed = TimeEmbedding(hidden_dim)

        if self.use_any_cluster:
            self.cluster_embed = ClusterEmbedding(
                n_spatial_clusters, n_temporal_clusters, hidden_dim,
                use_spatial=use_spatial_cluster,
                use_temporal=use_temporal_cluster)

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
                hour_idx=None, dow_idx=None,
                spatial_cluster=None, temporal_cluster=None):
        h = self.input_proj(x)  # (B, T, N, D)

        if self.use_time_embed and hour_idx is not None and dow_idx is not None:
            t_emb = self.time_embed(hour_idx, dow_idx)
            h = h + t_emb[:, None, None, :]

        if self.use_any_cluster:
            c_emb = self.cluster_embed(
                spatial_cluster if self.use_spatial_cluster else None,
                temporal_cluster if self.use_temporal_cluster else None)
            if c_emb is not None:
                h = h + c_emb

        adjs = self.adj_learner.get_all_adj(base_adj)

        for i, block in enumerate(self.st_blocks):
            h = block(h, adjs[i], od_patterns=od_patterns)

        h = h[:, -1, :, :]                     # (B, N, D) last step
        correction = self.output_proj(h)        # (B, N, C)

        if self.use_residual:
            last_obs = x[:, -1, :, :]           # (B, N, C)
            pred = last_obs + correction
        else:
            pred = correction

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
            use_cluster=mcfg.get('use_cluster', False),
            use_spatial_cluster=mcfg.get('use_spatial_cluster', False),
            use_temporal_cluster=mcfg.get('use_temporal_cluster', False),
            n_spatial_clusters=mcfg.get('n_spatial_clusters', 5),
            n_temporal_clusters=mcfg.get('n_temporal_clusters', 5),
            use_residual=mcfg.get('use_residual', True),
        )
