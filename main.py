import math
import time
import torch
from torch.utils.data import DataLoader
from gpt_trainer import GPTTrainer
from shakespeare_dataset import ShakespeareDataset
from core.gpt_model import GPTModel
from gpt_config import GPTConfig
import tiktoken

if torch.cuda.is_available():
    torch.cuda.empty_cache()

def main():
    
    # Define the desired device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Инициализация
    config = GPTConfig()
    enc = tiktoken.get_encoding("gpt2")
    
    # Dataset (замените на свой tinyshakespeare/input.txt)
    train_dataset = ShakespeareDataset("input.txt", config.block_size, enc)  # Скачайте файл!
    generator=torch.Generator(device=device)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    # Model
    model = GPTModel(config)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = GPTTrainer(model, config, enc, device)
    
    # Train
    trainer.fit(train_loader)
    
    # Demo generation
    context = "ROMEO:\n"
    tokens = torch.tensor([enc.encode(context)], dtype=torch.long)
    generated = model.generate_greedy(tokens, max_new_tokens=100, temperature=0.8)
    print(enc.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
