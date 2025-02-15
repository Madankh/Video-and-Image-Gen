import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from einops import rearrange


def get_patch_position_embedding(pos_emb_dim, grid_size, device):
    assert pos_emb_dim % 4 == 0, 'Positional embedding dimension must be divisible by 4'
    grid_size_h , grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w , dtype=torch.float32, device=device)

    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_position -> Number of patch  tokens
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factir = 10000^(2i/d_moodel)
    factor = 10000 ** ((torch.arange(
        start=0, end=pos_emb_dim//4,
        dtype=torch.float32,
        device=device
    ) / (pos_emb_dim // 4))
    )
    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb),torch.cos(grid_h_emb)], dim=-1)

    grid_w_emb = grid_w_positions[:,None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_h_emb)], dim=-1)

    pos_emb = torch.cat([grid_h_emb , grid_w_emb], dim=-1)

    return pos_emb

class PatchEmbed(nn.Module):
    r"""
    Layer to take in the input image and do the following:
        1.  Transform grid of image patches into a sequence of patches.
            Number of patches are decided based on image height,width and
            patch height, width.
        2. Add positional embedding to the above sequence
    """

    def __init__(self,
                image_height, 
                image_width, 
                im_channels,
                patch_height, 
                patch_width, 
                hidden_size):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels

        self.hidden_size = hidden_size

        self.patch_height = patch_height
        self.patch_width = patch_width

        # input dimension for patch Embedding FC layer
        patch_dim = self.im_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_size)
        )

        # DiT layer Initialization
        nn.init.xavier_uniform_(self.patch_embed[0].weight)
        nn.init.constant_(self.patch_embed[0].bias, 0)
    def forward(self, x):
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width
        out = rearrange(x, 'b c (nh, ph) (nw, pw) -> b (nh nw) (ph pw c)',
                        ph = self.patch_height, pw = self.patch_width)
        
        # x = x.reshape(B, C, grid_size_h, self.patch_height , grid_size_w, self.patch_width)          # Shape: (B, C, nh, ph, nw, pw)
        # x = x.permute(0, 2, 4, 3, 5, 1)              # Reorder axes: (B, nh, nw, ph, pw, C)
        # x = x.reshape(B, nh * nw, ph * pw * C)       # Final shape: (B, 196, 768)

    #   out = (32, 14*14=196, 16*16*3=768)
        # BxNumber of tokens x Patch Dimension -> B x Number of tokens x Transformer Dimension
        out = self.patch_embed(out)

        # add 2d sinnusoidal positional encoding
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.hidden_size,
                                                 grid_size=(grid_size_h, grid_size_w),
                                                            device=x.device)
        out += pos_embed
        return out