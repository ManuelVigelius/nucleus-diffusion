"""Small ViT that maps a (noisy) 4-bit image to per-subpixel logits over the clean image."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, channels, image_size, levels=16, patch_size=2, d_in=32,
                 dim=256, depth=6, heads=4, mlp_ratio=4.0):
        super().__init__()
        assert image_size % patch_size == 0
        self.C, self.S, self.L, self.p = channels, image_size, levels, patch_size
        self.grid = image_size // patch_size
        self.value_embed = nn.Parameter(torch.randn(channels, levels, d_in) * 0.02)  # one table per channel
        self.patch_proj = nn.Linear(patch_size * patch_size * channels * d_in, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.grid * self.grid, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * channels * levels)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, C, H, W) long in [0, levels) -> logits (B, C, H, W, levels)."""
        B, C, H, W = x.shape
        p, g, L = self.p, self.grid, self.L
        ch = torch.arange(C, device=x.device).view(1, C, 1, 1)
        e = self.value_embed[ch, x]                                     # (B, C, H, W, d_in)
        e = e.permute(0, 2, 3, 1, 4)                                    # (B, H, W, C, d_in)
        e = e.reshape(B, g, p, g, p, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, g * g, -1)
        h = self.patch_proj(e) + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        out = self.head(self.norm(h))                                   # (B, T, p*p*C*L)
        out = out.view(B, g, g, p, p, C, L).permute(0, 5, 1, 3, 2, 4, 6)  # (B, C, g, p, g, p, L)
        return out.reshape(B, C, H, W, L)


class Marginal(nn.Module):
    """Stage-1 model. Its input is pure uniform noise, independent of x0, so the CE-optimal
    prediction is the per-subpixel data marginal p(x0[c, h, w]). Estimated by counting."""

    def __init__(self, channels, image_size, levels=16):
        super().__init__()
        self.register_buffer("logits", torch.zeros(channels, image_size, image_size, levels))

    @torch.no_grad()
    def fit(self, data, flip=False, pseudocount=0.5):
        """data: (N, C, H, W) ints. flip=True symmetrises over horizontal flips (the train augmentation)."""
        C, H, W, L = self.logits.shape
        data = data.cpu()
        cells = torch.arange(C * H * W) * L
        counts = torch.zeros(C * H * W * L, dtype=torch.float64)
        for chunk in data.split(10000):
            idx = (cells + chunk.reshape(len(chunk), -1).long()).flatten()
            counts += torch.bincount(idx, minlength=counts.numel()).double()
        counts = counts.view(C, H, W, L)
        if flip:
            counts = 0.5 * (counts + counts.flip(2))
        p = (counts + pseudocount) / (counts + pseudocount).sum(-1, keepdim=True)
        self.logits.copy_(p.log().float().to(self.logits.device))
        return self

    def forward(self, x):
        return self.logits.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
