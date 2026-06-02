"""
CCGRU Cell — GRU with Graph Convolution gates
==============================================
Standard GRU:
    z = sigmoid(W_z @ [x, h])
    r = sigmoid(W_r @ [x, h])
    h~ = tanh(W @ [x, r*h])
    h_new = z * h + (1-z) * h~

CCGRU replaces W @ [...] with GraphConv([...], A):
    z = sigmoid(GConv_z([x, h], A))
    r = sigmoid(GConv_r([x, h], A))
    h~ = tanh(GConv([x, r*h], A))
    h_new = z * h + (1-z) * h~

This way, each GRU update considers neighbor information via graph convolution.
"""
import torch
import torch.nn as nn
from model.cgc import GraphConv


class CCGRUCell(nn.Module):
    """
    Single time-step CCGRU computation.

    Args:
        in_dim:    input feature dimension (C for first layer, hidden_dim for upper)
        hidden_dim: hidden state dimension
        num_nodes:  N (not used in computation, kept for clarity)
        cheb_k:     diffusion steps in graph convolution
    """

    def __init__(self, in_dim, hidden_dim, num_nodes, cheb_k=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Gate convolutions: input is [x, h] concatenated → dim = in_dim + hidden_dim
        gate_in = in_dim + hidden_dim
        self.conv_z = GraphConv(gate_in, hidden_dim, cheb_k)  # update gate
        self.conv_r = GraphConv(gate_in, hidden_dim, cheb_k)  # reset gate
        self.conv_h = GraphConv(gate_in, hidden_dim, cheb_k)  # candidate

    def forward(self, x_t, h_prev, adj):
        """
        Args:
            x_t:    (B, N, in_dim)   current input
            h_prev: (B, N, hidden_dim) previous hidden state
            adj:    (N, N) adjacency matrix for this layer

        Returns:
            h_new:  (B, N, hidden_dim) updated hidden state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x_t, h_prev], dim=-1)  # (B, N, in_dim + hidden_dim)

        z = torch.sigmoid(self.conv_z(combined, adj))  # update gate
        r = torch.sigmoid(self.conv_r(combined, adj))  # reset gate

        # Candidate with reset gate applied to previous hidden state
        combined_r = torch.cat([x_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.conv_h(combined_r, adj))

        h_new = z * h_prev + (1 - z) * h_tilde
        return h_new

    def init_hidden(self, batch_size, num_nodes, device):
        """Initialize hidden state with zeros."""
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
