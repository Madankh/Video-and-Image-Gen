import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    r"""
    Attention module for DiT    \
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']

        self.att_dim = self.n_heads * self.head_dim
        # QKV projection for the input
        self.qkv_proj = nn.Linear(self.hidden_size, 3* self.hidden_size, bias=True)
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.hidden_size)
        )

        ############################
        # DiT Layer Initialization
        ############################
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.constant_(self.output_proj[0].bias, 0)

    def forward(self, x):
        # Converting to Attention Dimension
        B, N = x.shape[:2]
        q,k,v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h, n h_dim', n_h = self.n_heads, h_dim = self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h, n h_dim', n_h = self.n_heads, h_dim = self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h, n h_dim', n_h = self.n_heads, h_dim = self.head_dim)

        # Compute Attention Weights
        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim **(-0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        out = torch.matmul(att , v)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h n_dim)')
        out = self.output_proj(out)

        return out
