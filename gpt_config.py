from dataclasses import dataclass

@dataclass
class GPTConfig:
    """Конфигурация модели, гиперпараметры обучения."""
    vocab_size: int = 50257  # GPT‑2 BPE
    block_size: int = 1024   # контекст
    embed_dim: int = 768     # d_model как в GPT‑2 small
    num_layers: int = 12
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    weight_decay: float = 0.1
    lr: float = 6e-4
    warmup_steps: int = 4000
    # max_steps: int = 50_000
    max_steps: int = 2000
    grad_clip_norm: float = 1.0
    batch_size: int = 8
    eval_interval: int = 500
    log_interval: int = 100