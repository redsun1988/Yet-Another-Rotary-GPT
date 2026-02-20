import torch
from typing import Tuple
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    """Tiny Shakespeare для демонстрации (скачайте tinyshakespeare/input.txt локально)."""

    def __init__(self, path: str, block_size: int, tokenizer):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.tokenizer = tokenizer
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y