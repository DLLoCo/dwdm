"""
TrafficPredNet v6 — TC + CGC + Dual Cluster Embedding
======================================================
All three challenges addressed:
  Challenge 1 (ST coupling):       TC + CGC in each ST block
  Challenge 2 (hotspot migration): CGC coupled adjacency + Spatial Cluster Embedding
  Challenge 3 (external events):   Temporal Cluster Embedding (data-driven time awareness)

v4/v5 → v6 change:
  Replace unreliable TimeEmbedding (offset guessing) with data-driven clustering.
  Spatial cluster tells model "what type of region this is" (hot/cold/transit).
  Temporal cluster tells model "what demand regime we're in" (night/peak/plateau).
  Both are learned from data via K-Means, no absolute time needed.

Backward compatible: use_cluster=False → identical to v2 (TC+CGC only).
"""
import math
import torch
import torch.nn as nn
from model.cgc import AdjacencyLearner
from model.st_block import STBlock


# ============================================================
# Time Embedding (v4 — kept for backward compat, use_time_embed)
# ============================================================

class TimeEmbedding(nn.Module):
    """Encode hour-of-day (0-47) and day-of-week (0-6) as embeddings."""

    def __init__(self, embed_dim):
        super().__init__()
        self.hour_embed = nn.Embedding(48, embed_dim)
        self.dow_embed = nn.Embedding(7, embed_dim)

    def forward(self, hour_idx, dow_idx):
        return self.hour_embed(hour_idx) + self.dow_embed(dow_idx)


# ============================================================
# Cluster Embedding (v6 — new)
# ============================================================

class ClusterEmbedding(nn.Module):
    """
    Dual cluster conditioning for spatio-temporal heterogeneity.

    Spatial cluster:  per-node, static → (N, D), broadcast to all B and T
    Temporal cluster: per-sample → (B, D), broadcast to all T and N

    Combined effect: model knows "this is a hot-zone node during late-night"
    vs "this is a cold-zone node during morning peak" — different treatment.
    """

    def __init__(self, n_spatial: int, n_temporal: int, embed_dim: int):
        super().__init__()
        self.spatial_embed = nn.Embedding(n_spatial, embed_dim)
        self.temporal_embed = nn.Embedding(n_temporal, embed_dim)
        # Learnable gate to control how much cluster info to inject
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, spatial_ids, temporal_ids):
        """
        spatial_ids:  (N,) long tensor — node cluster assignment
        temporal_ids: (B,) long tensor — sample temporal cluster
        Returns: (B, 1, N, D) — additive embedding for input tensor
        """
        s_emb = self.spatial_embed(spatial_ids)       # (N, D)
        t_emb = self.temporal_embed(temporal_ids)     # (B, D)

        # Combine: broadcast s_emb to (B, N, D) and t_emb to (B, N, D)
        B = t_emb.shape[0]
        N = s_emb.shape[0]
        s_expanded = s_emb.unsqueeze(0).expand(B, N, -1)   # (B, N, D)
        t_expanded = t_emb.unsqueeze(1).expand(B, N, -1)   # (B, N, D)

        # Gated fusion
        combined = torch.cat([s_expanded, t_expanded], dim=-1)  # (B, N, 2D)
        gate = self.gate(combined)                               # (B, N, D)
        out = gate * s_expanded + (1 - gate) * t_expanded       # (B, N, D)

        return out.unsqueeze(1)  # (B, 1, N, D) for broadcasting over T


# ============================================================
# Main Model
# ============================================================

class TrafficPredNet(nn.Module):

    def __init__(self, num_nodes, input_dim, output_dim, hidden_dim=64,
                 num_layers=3, cheb_k=3, embed_dim=50, kernel_size=3,
                 dropout=0.1, use_flow_gate=False, use_time_embed=False,
                 num_patterns=48,
                 use_cluster=False, n_spatial_clusters=5,
                 n_temporal_clusters=5):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_flow_gate = use_flow_gate
        self.use_time_embed = use_time_embed
        self.use_cluster = use_cluster

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # v4: Time Embedding (optional, kept for backward compat)
        if use_time_embed:
            self.time_embed = TimeEmbedding(hidden_dim)

        # v6: Cluster Embedding (optional)
        if use_cluster:
            self.cluster_embed = ClusterEmbedding(
                n_spatial_clusters, n_temporal_clusters, hidden_dim)

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
        """
        x:                (B, T, N, C)
        base_adj:         (N, N)
        od_patterns:      (P, N, N) for Flow Gate
        hour_idx:         (B,) 0-47  (for TimeEmbed v4)
        dow_idx:          (B,) 0-6   (for TimeEmbed v4)
        spatial_cluster:  (N,) long  (for ClusterEmbed v6)
        temporal_cluster: (B,) long  (for ClusterEmbed v6)
        """
        h = self.input_proj(x)  # (B, T, N, D)

        # v4: Time Embedding
        if self.use_time_embed and hour_idx is not None and dow_idx is not None:
            t_emb = self.time_embed(hour_idx, dow_idx)  # (B, D)
            h = h + t_emb[:, None, None, :]

        # v6: Cluster Embedding
        if (self.use_cluster and spatial_cluster is not None
                and temporal_cluster is not None):
            c_emb = self.cluster_embed(spatial_cluster, temporal_cluster)
            h = h + c_emb  # (B, 1, N, D) broadcasts to (B, T, N, D)

        adjs = self.adj_learner.get_all_adj(base_adj)

        for i, block in enumerate(self.st_blocks):
            h = block(h, adjs[i], od_patterns=od_patterns)

        h = h[:, -1, :, :]      # (B, N, D)
        pred = self.output_proj(h)  # (B, N, output_dim)
        return pred.unsqueeze(1)    # (B, 1, N, output_dim)

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
            n_spatial_clusters=mcfg.get('n_spatial_clusters', 5),
            n_temporal_clusters=mcfg.get('n_temporal_clusters', 5),
        )
