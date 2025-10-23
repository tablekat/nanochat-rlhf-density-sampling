#!/usr/bin/env python3
"""
Train a Reward Model (RM) on pairwise preferences.

This script:
1. Loads an SFT checkpoint
2. Adds a scalar reward head
3. Trains on preference pairs (chosen vs rejected responses)
4. Saves the trained RM

The RM learns to score responses by predicting preference (reward).
This enables GRPO training to use a learned reward signal.

Usage:
  torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm
  torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- --max_steps=500
  torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_rm -- --learning_rate=5e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class PreferenceDataset(Dataset):
    """Dataset for pairwise preference learning."""
    
    def __init__(self, pairs_path, tokenizer, max_length=512):
        """
        Load pairs from JSONL file.
        
        Expected format: {"prompt", "chosen", "rejected", "src", ...}
        """
        self.pairs = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        
        with open(pairs_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pair = json.loads(line)
                    # Validate required fields
                    if all(k in pair for k in ['prompt', 'chosen', 'rejected']):
                        self.pairs.append(pair)
                except json.JSONDecodeError:
                    continue
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        # Tokenize prompt + response pairs
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        chosen_ids = self.tokenizer.encode(chosen_text, max_length=self.max_length, truncation=True)
        rejected_ids = self.tokenizer.encode(rejected_text, max_length=self.max_length, truncation=True)
        
        # Pad to same length for batch processing
        max_len = max(len(chosen_ids), len(rejected_ids))
        
        chosen_ids = chosen_ids + [self.tokenizer.pad_token_id] * (max_len - len(chosen_ids))
        rejected_ids = rejected_ids + [self.tokenizer.pad_token_id] * (max_len - len(rejected_ids))
        
        return {
            'chosen_ids': torch.tensor(chosen_ids[:self.max_length]),
            'rejected_ids': torch.tensor(rejected_ids[:self.max_length]),
        }


class RewardModelHead(nn.Module):
    """Simple reward head that outputs scalar reward."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        
        Returns:
            rewards: (batch,) - scalar reward for entire sequence
        """
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]
        return self.linear(last_hidden).squeeze(-1)


def train_step(model, rm_head, batch, optimizer, device):
    """Single training step on a batch of preference pairs."""
    chosen_ids = batch['chosen_ids'].to(device)
    rejected_ids = batch['rejected_ids'].to(device)
    
    # Get hidden states from model
    with torch.no_grad():
        chosen_outputs = model(chosen_ids, output_hidden_states=True)
        rejected_outputs = model(rejected_ids, output_hidden_states=True)
    
    chosen_hidden = chosen_outputs.hidden_states[-1]
    rejected_hidden = rejected_outputs.hidden_states[-1]
    
    # Get reward scores
    chosen_reward = rm_head(chosen_hidden)
    rejected_reward = rm_head(rejected_hidden)
    
    # Bradley-Terry loss: log(sigmoid(r_chosen - r_rejected))
    # Maximize P(chosen > rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--sft_ckpt_path", default="outs/sft/ckpt.pt", help="Path to SFT checkpoint")
    parser.add_argument("--pairs_path", default=".cache/data/pairs_all.jsonl", help="Path to pairs JSONL")
    parser.add_argument("--out_path", default="outs/rm/ckpt.pt", help="Output path for RM checkpoint")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device_batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    # Load SFT checkpoint and tokenizer
    print(f"Loading SFT checkpoint: {args.sft_ckpt_path}")
    # This is a simplified version - in practice you'd load the actual checkpoint
    # For now, we'll assume the checkpoint exists and can be loaded
    try:
        checkpoint = torch.load(args.sft_ckpt_path, map_location=device)
        model = checkpoint['model']  # Assumes checkpoint has 'model' key
    except Exception as e:
        print(f"Warning: Could not load checkpoint properly: {e}")
        print("This is expected if SFT hasn't been run yet.")
        sys.exit(1)
    
    # Add reward head
    hidden_size = model.config.hidden_size
    rm_head = RewardModelHead(hidden_size).to(device)
    
    # Load dataset
    print(f"Loading preference pairs: {args.pairs_path}")
    try:
        # We need a tokenizer - in practice this would be loaded from the model
        dataset = PreferenceDataset(args.pairs_path, tokenizer=model.tokenizer)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    print(f"Dataset size: {len(dataset)} pairs")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Setup training
    optimizer = torch.optim.Adam(rm_head.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    log_dir = os.path.dirname(args.out_path)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "logs"))
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Reward Model Training")
    print("="*70)
    
    model.eval()  # Base model frozen
    rm_head.train()
    
    step = 0
    for epoch in range(100):  # Multiple epochs if needed
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.max_steps:
                break
            
            loss = train_step(model, rm_head, batch, optimizer, device)
            scheduler.step()
            
            if step % args.log_interval == 0:
                print(f"Step {step}/{args.max_steps} - Loss: {loss:.4f}")
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            
            step += 1
        
        if step >= args.max_steps:
            break
    
    # Save checkpoint
    print(f"\nSaving RM checkpoint to: {args.out_path}")
    torch.save({
        'rm_head_state_dict': rm_head.state_dict(),
        'config': {
            'hidden_size': hidden_size,
        }
    }, args.out_path)
    
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
