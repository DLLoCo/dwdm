"""
Gated Temporal Convolution
==========================
From STGCN (Yu et al., IJCAI 2018).

1D causal convolution along time dimension with Gated Linear Unit (GLU):
    output = Conv_a(X) * sigmoid(Conv_b(X))

Processes all time steps in parallel — no sequential assumption.
"""
import torch
import torch.nn as nn


class GatedTemporalConv(nn.Module):
    """
    Gated 1D convolution along the time dimension.

    Input:  (B, T, N, D_in)
    Output: (B, T', N, D_out)  where T' = T - kernel_size + 1 (no padding)
            or T' = T (with padding)

    Args:
        in_dim:      input feature dimension
        out_dim:     output feature dimension
        kernel_size: temporal convolution kernel size
        padding:     whether to pad to keep T unchanged
    """

    def __init__(self, in_dim, out_dim, kernel_size=3, padding=True):
        super().__init__()
        self.padding = padding
        pad = (kernel_size - 1) // 2 if padding else 0

        # Two parallel convolutions for GLU gating
        # Conv1d expects (B, C, L) — we'll reshape to treat N*B as batch
        self.conv_gate = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim * 2,  # split into value + gate
            kernel_size=kernel_size,
            padding=pad,
        )

    def forward(self, x):
        """
        x: (B, T, N, D)
        returns: (B, T, N, D_out)  if padding=True
        """
        B, T, N, D = x.shape

        # Reshape: merge B and N, treat T as sequence length
        # (B, T, N, D) -> (B*N, D, T)  for Conv1d
        x = x.permute(0, 2, 3, 1).reshape(B * N, D, T)

        # Apply gated convolution
        h = self.conv_gate(x)  # (B*N, 2*D_out, T')

        # Split into value and gate
        h_val, h_gate = h.chunk(2, dim=1)  # each (B*N, D_out, T')
        h = h_val * torch.sigmoid(h_gate)  # GLU

        # Reshape back: (B*N, D_out, T') -> (B, T', N, D_out)
        T_out = h.size(-1)
        D_out = h.size(1)
        h = h.reshape(B, N, D_out, T_out).permute(0, 3, 1, 2)

        return h
