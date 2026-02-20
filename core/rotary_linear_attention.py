from core.rotary_embedding import RotaryEmbedding
import torch
import torch.nn as nn

class RotaryLinearAttention(nn.Module):
    """Self‑attention с RoPE: causal, multihead, с ротацией Q."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, block_size: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        
        self.rope = RotaryEmbedding(dim=self.head_dim, max_position_embeddings=2048)

        # Causal mask buffer
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE только для Q и K
        # Применяем к q и k
        q_rot, k_rot = self.rope(q, k)

        # Attention scores
        attn = (q_rot @ k_rot.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.masked_fill(self.causal_mask[:T, :T][None, None, :, :] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, nh, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        return out