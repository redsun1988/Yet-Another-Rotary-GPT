import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        """
        Инициализация роторных позиционных эмбеддингов (RoPE).

        Args:
            dim: размерность головы внимания (должна быть четной)
            max_position_embeddings: максимальная длина последовательности для кэша
            base: базовая частота для геометрического уменьшения (обычно 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # ШАГ 1: Предвычисляем inv_freq = 1 / base^(2i/dim)
        # Для i ∈ {0, 1, ..., dim/2-1}: θ_i = 10000^(-2i/dim)
        # inv_freq имеет размер [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # ШАГ 2: Кэшируем cos/sin значения
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        """
        Предвычисляет и кэширует sin/cos значения для позиций 0..seq_len-1.
        """
        self.max_seq_len_cached = seq_len

        # Создаем вектор позиций t = [0, 1, 2, ..., seq_len-1]
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # Вычисляем freqs = t * inv_freq^T = [seq_len, dim/2]
        # Каждая позиция t умножается на все частоты θ_i
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]

        # ИСПРАВЛЕНИЕ: Дублируем freqs для четных и нечетных индексов
        # Вместо cat((freqs, freqs), dim=-1) используем repeat_interleave
        # Это создает [θ_0, θ_0, θ_1, θ_1, θ_2, θ_2, ...]
        # Размер: [seq_len, dim]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    @staticmethod
    def apply_rotary_pos_emb(q_or_k, cos, sin):
        """
        Применяет роторный поворот к тензору query или key.

        АЛГОРИТМ RoPE:
        Для каждой пары (x_2i, x_2i+1) применяем 2D поворот:
        x'_2i   = x_2i   * cos(m*θ_i) - x_2i+1 * sin(m*θ_i)
        x'_2i+1 = x_2i   * sin(m*θ_i) + x_2i+1 * cos(m*θ_i)

        Args:
            q_or_k: тензор [bs, num_heads, seq_len, head_dim]
            cos/sin: кэшированные [seq_len, dim] с повторяющимися значениями
        """
        seq_len = q_or_k.shape[-2]

        # Обрезаем cos/sin до нужной длины последовательности
        # Добавляем batch и head размерности: [1, 1, seq_len, dim]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # РАЗДЕЛЕНИЕ НА ПАРЫ:
        # x1 = четные индексы [0, 2, 4, 6, ...]  -> размер [..., dim/2]
        # x2 = нечетные индексы [1, 3, 5, 7, ...] -> размер [..., dim/2]
        x1 = q_or_k[..., 0::2]  # четные индексы
        x2 = q_or_k[..., 1::2]  # нечетные индексы

        # ВАЖНО: cos и sin нужно также разделить соответствующим образом
        # cos[0::2] содержит cos(θ_0), cos(θ_1), cos(θ_2), ...
        # sin[0::2] содержит sin(θ_0), sin(θ_1), sin(θ_2), ...
        cos_even = cos[..., 0::2]  # для четных позиций
        sin_even = sin[..., 0::2]  # для четных позиций

        # ПРИМЕНЕНИЕ ПОВОРОТА для каждой пары независимо
        rotated_even = x1 * cos_even - x2 * sin_even  # новая четная часть
        rotated_odd  = x1 * sin_even + x2 * cos_even  # новая нечетная часть

        # Объединяем обратно: [x'_0, x'_1, x'_2, x'_3, ...]
        # Используем stack + flatten для перемежения
        rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

        return rotated

    def forward(self, q, k):
        """
        Применяет RoPE к query и key тензорам.
        Автоматически расширяет кэш если последовательность длиннее.
        """
        # Проверяем, хватает ли кэша для текущей длины
        seq_len = max(q.shape[-2], k.shape[-2])
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        # Применяем поворот к q и k независимо
        q_rotated = self.apply_rotary_pos_emb(q, self.cos_cached, self.sin_cached)
        k_rotated = self.apply_rotary_pos_emb(k, self.cos_cached, self.sin_cached)

        return q_rotated, k_rotated