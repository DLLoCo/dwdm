"""
Spatio-Temporal Block (v2 with Flow Gate)
==========================================
One ST block = Gated Temporal Conv -> CGC Graph Conv -> Flow Gate -> Residual

The Flow Gate (from STDN) is applied AFTER graph conv, matching Eq.4:
    Y^(k) = ReLU(Conv(Y^(k-1))) * sigmoid(FlowGate)

When flow_gate is disabled, this is identical to the v1 ST block.
"""
import torch
import torch.nn as nn
from model.temporal_conv import GatedTemporalConv
from model.cgc import GraphConv


class STBlock(nn.Module):
    """
    Single Spatio-Temporal block with optional Flow Gate.

    Input:  (B, T, N, D)
    Output: (B, T, N, D)
    """

    def __init__(self, dim, num_nodes, kernel_size=3, cheb_k=3, dropout=0.1,
                 use_flow_gate=False, num_patterns=48):
        super().__init__()
        self.use_flow_gate = use_flow_gate

        # Temporal: gated 1D conv
        self.temporal = GatedTemporalConv(dim, dim, kernel_size, padding=True)

        # Spatial: graph convolution
        self.spatial = GraphConv(dim, dim, cheb_k, bias=True)

        # Flow Gate (optional): attention-based OD pattern gating
        if use_flow_gate:
            from model.flow_gate import FlowGate
            self.flow_gate = FlowGate(num_nodes, dim, num_patterns)

        # Layer norm + dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj, od_patterns=None):
        """
        x:           (B, T, N, D)
        adj:         (N, N) adjacency matrix
        od_patterns: (P, N, N) OD pattern bank (optional)
        returns:     (B, T, N, D)
        """
        residual = x

        # Step 1: Temporal convolution
        h = self.temporal(x)  # (B, T, N, D)

        # Step 2: Graph convolution at each time step
        B, T, N, D = h.shape
        h = h.reshape(B * T, N, D)
        h = self.spatial(h, adj)  # (B*T, N, D)

        # Step 3: Flow Gate — element-wise gating (STDN Eq.4)
        if self.use_flow_gate and od_patterns is not None:
            gate = self.flow_gate(h, od_patterns)  # (B*T, N, D)
            h = h * gate

        h = h.reshape(B, T, N, D)

        # Step 4: Residual + norm + dropout
        h = self.dropout(h)
        h = self.norm(h + residual)

        return h
