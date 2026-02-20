from core.gpt_block import GPTBlock
from core.rotary_embedding import RotaryEmbedding
from gpt_config import GPTConfig


import torch
import torch.nn as nn
from typing import Tuple, Optional


class GPTModel(nn.Module):
    """GPT‑2 like decoder‑only с RoPE и GPT‑2 BPE vocab."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_rope = RotaryEmbedding(config.embed_dim // config.num_heads)
        self.drop_emb = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])

        self.ln_f = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape

        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        x = self.drop_emb(tok_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate_greedy(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Стандартный greedy (argmax) generation с optional top‑k."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        return idx