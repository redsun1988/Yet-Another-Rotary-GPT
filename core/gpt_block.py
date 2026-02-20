from core.rotary_linear_attention import RotaryLinearAttention
from gpt_config import GPTConfig


import torch
import torch.nn as nn


class GPTBlock(nn.Module):
    """Residual block: Pre‑LN + Rotary Attention + MLP."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_ln = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attn = RotaryLinearAttention(
            config.embed_dim, config.num_heads, config.dropout, config.block_size
        )
        self.attn_drop = nn.Dropout(config.dropout)

        self.mlp_ln = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, int(config.embed_dim * config.mlp_ratio)),
            nn.GELU(approximate='tanh'),  # GPT‑2 style GELU
            nn.Dropout(config.dropout),
            nn.Linear(int(config.embed_dim * config.mlp_ratio), config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre‑LN residual attention
        attn_out = self.attn(self.attn_ln(x))
        x = x + attn_out

        # Pre‑LN MLP
        mlp_out = self.mlp(self.mlp_ln(x))
        x = x + mlp_out
        return x