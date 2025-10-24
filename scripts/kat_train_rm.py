#!/usr/bin/env python3
"""
Train a Reward Model (RM) on pairwise preferences.

This script:
1. Loads an SFT checkpoint via checkpoint_manager
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

from nanochat.checkpoint_manager import load_model, save_checkpoint, get_base_dir
from nanochat.tokenizer import get_tokenizer


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
            for line_num, line in enumerate(f, 1):
                try:
                    pair = json.loads(line)
                    # Validate required fields
                    if all(k in pair for k in ['prompt', 'chosen', 'rejected']):
                        self.pairs.append(pair)
                except json.JSONDecodeError:
                    if line_num <= 5:  # Only warn about first few errors
                        print(f"⚠️  Warning: Could not parse line {line_num}: {line[:50]}...")
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
        
        # Encode without max_length/truncation parameters
        chosen_ids = self.tokenizer.encode(chosen_text)
        rejected_ids = self.tokenizer.encode(rejected_text)
        
        # Apply truncation manually
        if len(chosen_ids) > self.max_length:
            chosen_ids = chosen_ids[:self.max_length]
        if len(rejected_ids) > self.max_length:
            rejected_ids = rejected_ids[:self.max_length]
        
        # Pad to same length for batch processing
        max_len = max(len(chosen_ids), len(rejected_ids))
        
        pad_token_id = self.tokenizer.encode_special("<|assistant_end|>")
        
        chosen_ids = chosen_ids + [pad_token_id] * (max_len - len(chosen_ids))
        rejected_ids = rejected_ids + [pad_token_id] * (max_len - len(rejected_ids))
        
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
    
    # Get hidden states from model (model frozen, just getting representations)
    with torch.no_grad():
        chosen_outputs = model(chosen_ids)
        rejected_outputs = model(rejected_ids)
        # For now, use the last layer outputs as hidden states
        # In a more sophisticated version, would hook intermediate layers
        chosen_hidden = chosen_outputs[:, :, -1] if isinstance(chosen_outputs, torch.Tensor) else chosen_outputs
        rejected_hidden = rejected_outputs[:, :, -1] if isinstance(rejected_outputs, torch.Tensor) else rejected_outputs
    
    # Get reward scores
    chosen_reward = rm_head(chosen_hidden.unsqueeze(0) if chosen_hidden.dim() == 2 else chosen_hidden)
    rejected_reward = rm_head(rejected_hidden.unsqueeze(0) if rejected_hidden.dim() == 2 else rejected_hidden)
    
    # Bradley-Terry loss: log(sigmoid(r_chosen - r_rejected))
    # Maximize P(chosen > rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    
    # Paths
    default_pairs = os.path.join(get_base_dir(), "data", "pairs_all.jsonl")
    parser.add_argument("--pairs_path", default=default_pairs, help="Path to pairs JSONL")
    parser.add_argument("--out_dir", default=os.path.join(get_base_dir(), "rm_checkpoints", "d20"), 
                        help="Output directory for RM checkpoint")
    
    # Model source
    parser.add_argument("--sft_source", default="sft", help="Source to load SFT model from (sft|mid|base)")
    
    # Training params
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device_batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load SFT checkpoint properly using checkpoint_manager
    print(f"Loading {args.sft_source} model...")
    try:
        model, tokenizer, meta_data = load_model(
            source=args.sft_source,
            device=device,
            phase="eval"
        )
        model.eval()  # Freeze for RM training
        for param in model.parameters():
            param.requires_grad = False
        print(f"✓ Model loaded from {args.sft_source}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Get hidden size from model config
    try:
        hidden_size = model.config.n_embd if hasattr(model, 'config') else 768
    except Exception as e:
        print(f"Warning: Could not get hidden size from config: {e}, using default 768")
        hidden_size = 768
    
    # Load dataset
    print(f"Loading preference pairs: {args.pairs_path}")
    try:
        if not os.path.exists(args.pairs_path):
            raise FileNotFoundError(f"Pairs file not found: {args.pairs_path}\nRun kat_download_pairs first")
        
        dataset = PreferenceDataset(args.pairs_path, tokenizer=tokenizer)
        
        if len(dataset) == 0:
            raise RuntimeError(f"No valid pairs loaded from {args.pairs_path}")
        
        print(f"✓ Loaded {len(dataset)} preference pairs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    
    # Setup training
    rm_head = RewardModelHead(hidden_size).to(device)
    optimizer = torch.optim.Adam(rm_head.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    log_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Reward Model Training")
    print("="*70)
    
    rm_head.train()
    
    step = 0
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.max_steps:
                break
            
            try:
                loss = train_step(model, rm_head, batch, optimizer, device)
                scheduler.step()
                
                if step % args.log_interval == 0:
                    print(f"Step {step}/{args.max_steps} - Loss: {loss:.4f}")
                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                
                step += 1
            except Exception as e:
                print(f"Error in training step: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if step >= args.max_steps:
            break
    
    # Save checkpoint
    print(f"\nSaving RM checkpoint to: {args.out_dir}")
    
    # Save reward head state
    model_state = rm_head.state_dict()
    meta_info = {
        "hidden_size": hidden_size,
        "model_config": meta_data.get("model_config", {}),
        "training_steps": step,
    }
    
    # Save using checkpoint_manager pattern
    checkpoint_data = {
        'rm_head_state_dict': model_state,
        'config': {
            'hidden_size': hidden_size,
        }
    }
    
    checkpoint_path = os.path.join(args.out_dir, "model_000000.pt")
    torch.save(checkpoint_data, checkpoint_path)
    print(f"✓ Saved checkpoint to {checkpoint_path}")
    
    writer.close()
    print("✓ Training complete!")


if __name__ == "__main__":
    main()
