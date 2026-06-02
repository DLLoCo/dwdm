"""
Flow Gate v2 — Attention-based OD pattern gating
=================================================
Adapted from STDN (Yao et al., AAAI 2019) Eq.4 to graph structure.

Design:
  1. Load 48 half-hour OD patterns as a "pattern bank" (48, N, N)
  2. At each ST block, use current hidden state as query to
     attention-select/mix the most relevant OD pattern
  3. Generate gate signal via sigmoid(MLP(selected_pattern))
  4. Element-wise multiply on graph conv output (matches STDN Eq.4)

Correspondence to STDN:
  STDN: Y^(k) = ReLU(Conv(Y^(k-1))) * sigmoid(Conv(F^(k-1)))
  Ours: h = GraphConv(h, adj) * sigmoid(MLP(attn_select(h, OD_bank)))
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FlowGate(nn.Module):
    """
    Attention-based flow gating module.

    Uses 48 OD patterns as a learnable pattern bank.
    Current hidden state queries the bank to select relevant flow pattern,
    then generates a gate signal to modulate graph conv output.

    Args:
        num_nodes:    N
        hidden_dim:   D (hidden feature dim of ST blocks)
        num_patterns: number of OD patterns (48 for half-hour slots)
    """

    def __init__(self, num_nodes, hidden_dim, num_patterns=48):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_patterns = num_patterns

        # Project OD pattern (N,) per node into hidden_dim for attention
        self.pattern_proj = nn.Linear(num_nodes, hidden_dim)

        # Query projection: hidden state -> attention query
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gate generator: selected pattern -> gate signal
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h, od_patterns):
        """
        h:           (BT, N, D) hidden states from graph conv
        od_patterns: (P, N, N) OD pattern bank, P=48
        Returns:     (BT, N, D) gate signal
        """
        P, N, _ = od_patterns.shape
        BT, _, D = h.shape

        # Step 1: Compact query from hidden state (BT, D)
        query = self.query_proj(h.mean(dim=1))

        # Step 2: Compact pattern keys (P, D)
        patterns_emb = self.pattern_proj(od_patterns.mean(dim=1))

        # Step 3: Attention (BT, P) — lightweight, no big expansion
        attn = torch.matmul(query, patterns_emb.t()) / (D ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # Step 4: Select OD pattern (BT, N, N)
        selected_od = torch.einsum('bp,pnm->bnm', attn, od_patterns)

        # Step 5: Project per-node flow features (BT, N, D)
        node_flow = self.pattern_proj(selected_od)

        # Step 6: Gate
        gate = self.gate_net(node_flow)
        return gate


class FlowGateManager:
    """Loads OD pattern data and provides it to the model."""

    def __init__(self, od_path=None, device='cpu'):
        self.device = device
        self.od_patterns = None

        if od_path and os.path.exists(od_path):
            data = np.load(od_path)
            od = data['od'].astype(np.float32)

            if od.ndim == 3:
                # Hourly OD: (48, N, N) — normalize per pattern
                self.od_patterns = torch.FloatTensor(od).to(device)
                row_max = self.od_patterns.amax(dim=-1, keepdim=True).clamp(min=1e-8)
                self.od_patterns = self.od_patterns / row_max
                print(f"[FlowGate] Loaded {self.od_patterns.shape[0]} OD patterns: "
                      f"{self.od_patterns.shape}")
            elif od.ndim == 2:
                # Single OD: (N, N) -> expand to (1, N, N)
                od_t = torch.FloatTensor(od).unsqueeze(0).to(device)
                od_t = od_t / od_t.amax(dim=-1, keepdim=True).clamp(min=1e-8)
                self.od_patterns = od_t
                print(f"[FlowGate] Loaded single OD pattern: {self.od_patterns.shape}")
        else:
            print("[FlowGate] No OD data found, flow gating disabled.")

    @property
    def available(self):
        return self.od_patterns is not None

    def get_patterns(self):
        """Returns (P, N, N) OD pattern tensor."""
        return self.od_patterns
