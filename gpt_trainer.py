from core.gpt_model import GPTModel
from gpt_config import GPTConfig


import torch
from typing import Dict
import torch.optim as optim
from torch.utils.data import DataLoader

class GPTTrainer:
    """Trainer с warmup, clipping, logging, eval, checkpoints."""

    def __init__(self, model: GPTModel, config: GPTConfig, tokenizer, device: str = None):
        self.model = model.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=1,
            eta_min=1e-5,
        )

        self.step = 0

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits, loss = self.model(batch_x, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        self.step += 1
        return {'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']}

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                _, loss = self.model(batch_x, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                n += batch_x.size(0)

        return total_loss / n

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None) -> None:
        print(f"Training on {self.device} with batch_size={self.config.batch_size}")
        
        for step in range(self.config.max_steps):
            batch_x, batch_y = next(iter(train_loader))
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            metrics = self.train_step(batch_x, batch_y)

            if step % self.config.log_interval == 0:
                print(f"Step {step:06d}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}")

            if step > 0 and step % self.config.eval_interval == 0 and val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Step {step:06d}: val_loss={val_loss:.4f}")

            if step % 5000 == 0:
                torch.save(self.model.state_dict(), f"checkpoints//gpt_checkpoint_step_{step}.pth")