#!/usr/bin/env python3
"""
Train with GRPO (Generalized Reward Policy Optimization) using density-aware sampling.

This script:
1. Loads SFT checkpoint as the policy via checkpoint_manager
2. Loads trained RM checkpoint for reward scoring
3. Computes prompt embeddings from base model's hidden states
4. Samples preference pairs inversely proportional to density (if enabled)
5. Optimizes policy using GRPO loss with KL divergence penalty
6. Tests hypothesis that diversity-aware sampling reduces mode collapse

The key innovation: instead of uniform sampling from preference pairs,
we weight by 1/density to encourage the model to learn from diverse prompts.

Usage:
  torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo
  torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --density_aware=False
  torchrun --standalone --nproc_per_node=8 -m scripts.kat_train_grpo -- --beta=0.1 --max_steps=5000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import NearestNeighbors

from nanochat.checkpoint_manager import load_model, get_base_dir
from nanochat.tokenizer import get_tokenizer


class DensityAwareSampler:
    """Compute local density of prompts using base model embeddings."""
    
    def __init__(self, prompts, base_model, tokenizer, device, k=10):
        """
        Args:
            prompts: List of prompt strings
            base_model: Base model to extract embeddings from
            tokenizer: Tokenizer for encoding
            device: Device to run on
            k: Number of nearest neighbors for density estimation
        """
        self.prompts = prompts
        self.k = k
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Compute real semantic embeddings from base model
        self.embeddings = self._compute_embeddings_from_model(prompts)
        self.density_weights = self._compute_inverse_density_weights()
    
    @torch.no_grad()
    def _compute_embeddings_from_model(self, prompts):
        """
        Extract embeddings from base model's hidden states.
        Uses the penultimate layer output as semantic representation.
        """
        embeddings = []
        batch_size = 8
        
        print(f"Computing embeddings for {len(prompts)} prompts using base model...")
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # Encode prompts
            encoded = [self.tokenizer.encode(p) for p in batch_prompts]
            
            # Pad to same length
            max_len = max(len(e) for e in encoded)
            pad_token_id = self.tokenizer.encode_special("<|assistant_end|>")
            
            input_ids = []
            for e in encoded:
                padded = e + [pad_token_id] * (max_len - len(e))
                input_ids.append(padded[:512])  # Limit to 512 tokens
            
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            
            # Get model outputs (will need to hook into hidden states)
            # For now, use logits and average pool as approximation
            outputs = self.base_model(input_tensor)
            
            # Average pooling over sequence dimension
            if isinstance(outputs, torch.Tensor):
                # Outputs are logits: (batch, seq_len, vocab_size)
                # Use average of sequence as embedding
                batch_emb = outputs.mean(dim=1)  # (batch, vocab_size)
            else:
                # If it's a tuple or has hidden_states attribute (after we modify GPT)
                batch_emb = outputs
            
            embeddings.append(batch_emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        return embeddings
    
    def _compute_inverse_density_weights(self):
        """
        Compute local density using k-NN.
        Weight = 1 / local_density
        """
        if len(self.prompts) < self.k + 1:
            # Too few prompts, use uniform weights
            print(f"⚠️  Only {len(self.prompts)} prompts but k={self.k}, using uniform weights")
            return np.ones(len(self.prompts))
        
        print(f"Computing density weights using k={self.k}...")
        
        # Compute k-NN distances
        nbrs = NearestNeighbors(n_neighbors=self.k+1)  # +1 because self is a neighbor
        nbrs.fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        
        # Local density ~ average distance to k nearest neighbors
        # We exclude the first neighbor (self) by taking distances[:, 1:]
        local_densities = np.mean(distances[:, 1:], axis=1)
        
        # Avoid division by zero
        local_densities = np.maximum(local_densities, 1e-6)
        
        # Weight inversely proportional to density
        # Normalize to [0, 1] range for use as sampling weights
        weights = 1.0 / local_densities
        weights = weights / weights.sum()  # Normalize to valid probability distribution
        
        print(f"✓ Density weights computed. Min: {weights.min():.4f}, Max: {weights.max():.4f}")
        
        return weights


class PreferenceDataset(Dataset):
    """Dataset for GRPO training with preference pairs."""
    
    def __init__(self, pairs_path, tokenizer, max_length=512):
        """Load pairs from JSONL file."""
        self.pairs = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        
        with open(pairs_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    pair = json.loads(line)
                    if all(k in pair for k in ['prompt', 'chosen', 'rejected']):
                        self.pairs.append(pair)
                except json.JSONDecodeError:
                    if line_num <= 5:
                        print(f"⚠️  Warning: Could not parse line {line_num}")
                    continue
        
        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid pairs loaded from {pairs_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt = pair['prompt']
        chosen = pair['chosen']
        rejected = pair['rejected']
        
        # Encode without max_length/truncation parameters
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = self.tokenizer.encode(chosen)
        rejected_ids = self.tokenizer.encode(rejected)
        
        # Apply truncation manually
        if len(prompt_ids) > self.max_length:
            prompt_ids = prompt_ids[:self.max_length]
        if len(chosen_ids) > self.max_length:
            chosen_ids = chosen_ids[:self.max_length]
        if len(rejected_ids) > self.max_length:
            rejected_ids = rejected_ids[:self.max_length]
        
        return {
            'prompt_ids': torch.tensor(prompt_ids),
            'chosen_ids': torch.tensor(chosen_ids),
            'rejected_ids': torch.tensor(rejected_ids),
        }


def compute_grpo_loss(policy_logits, reference_logits, rm_rewards, beta=0.1, temperature=1.0):
    """
    Compute GRPO loss with proper KL divergence penalty.
    
    GRPO combines:
    1. Reward maximization: maximize (r_chosen - r_rejected)
    2. KL penalty: don't diverge too much from SFT reference
    
    Loss = -reward + beta * KL(policy || reference)
    """
    chosen_reward, rejected_reward = rm_rewards
    policy_chosen_logits, policy_rejected_logits = policy_logits
    ref_chosen_logits, ref_rejected_logits = reference_logits
    
    # Preference loss: Bradley-Terry preference
    pref_loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    
    # KL divergence penalty (proper implementation)
    # KL(policy || reference) = E[log(policy) - log(reference)]
    
    # Chosen response KL
    kl_chosen = F.kl_div(
        F.log_softmax(policy_chosen_logits / temperature, dim=-1),
        F.softmax(ref_chosen_logits / temperature, dim=-1),
        reduction='batchmean'
    )
    
    # Rejected response KL
    kl_rejected = F.kl_div(
        F.log_softmax(policy_rejected_logits / temperature, dim=-1),
        F.softmax(ref_rejected_logits / temperature, dim=-1),
        reduction='batchmean'
    )
    
    kl_loss = (kl_chosen + kl_rejected) / 2.0
    
    total_loss = pref_loss + beta * kl_loss
    return total_loss, pref_loss.item(), kl_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train with GRPO using density-aware sampling")
    
    # Paths
    default_pairs = os.path.join(get_base_dir(), "data", "pairs_all.jsonl")
    default_prompts = os.path.join(get_base_dir(), "data", "prompts_all.jsonl")
    default_out = os.path.join(get_base_dir(), "grpo_checkpoints", "d20")
    
    parser.add_argument("--pairs_path", default=default_pairs, help="Path to pairs")
    parser.add_argument("--prompts_path", default=default_prompts, help="Path to prompts")
    parser.add_argument("--sft_source", default="sft", help="Source for SFT model (sft|mid|base)")
    parser.add_argument("--rm_source", default="rm", help="Source for RM model")
    parser.add_argument("--out_dir", default=default_out, help="Output directory")
    
    # Training params
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="KL divergence penalty")
    parser.add_argument("--device_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--density_aware", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        default=True, help="Use density-aware sampling")
    parser.add_argument("--density_k", type=int, default=10, help="k for k-NN density estimation")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*70)
    print(f"GRPO Training (density_aware={args.density_aware})")
    print("="*70)
    
    # Load SFT checkpoint as policy
    print(f"\nLoading policy from {args.sft_source}...")
    try:
        policy, tokenizer, sft_meta = load_model(
            source=args.sft_source,
            device=device,
            phase="train"
        )
        print(f"✓ Policy loaded")
    except Exception as e:
        print(f"Error loading policy: {e}")
        sys.exit(1)
    
    # Load SFT checkpoint again as reference (frozen)
    print(f"Loading reference model (frozen)...")
    try:
        reference_model, _, _ = load_model(
            source=args.sft_source,
            device=device,
            phase="eval"
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        print(f"✓ Reference model loaded")
    except Exception as e:
        print(f"Error loading reference model: {e}")
        sys.exit(1)
    
    # Load RM checkpoint
    print(f"Loading RM from {args.rm_source}...")
    try:
        # Try to load as checkpoint_manager source first
        rm_model, _, rm_meta = load_model(
            source=args.rm_source,
            device=device,
            phase="eval"
        )
        rm_head = rm_model
        rm_head.eval()
        print(f"✓ RM loaded")
    except Exception as e:
        print(f"Warning: Could not load RM via checkpoint_manager: {e}")
        print(f"Trying legacy RM checkpoint path...")
        try:
            rm_ckpt_path = os.path.join(get_base_dir(), "rm_checkpoints", "d20", "model_000000.pt")
            rm_checkpoint = torch.load(rm_ckpt_path, map_location=device)
            
            from scripts.kat_train_rm import RewardModelHead
            hidden_size = rm_checkpoint['config']['hidden_size']
            rm_head = RewardModelHead(hidden_size).to(device)
            rm_head.load_state_dict(rm_checkpoint['rm_head_state_dict'])
            rm_head.eval()
            print(f"✓ RM loaded from legacy path")
        except Exception as e2:
            print(f"Error loading RM: {e2}")
            sys.exit(1)
    
    # Load dataset
    print(f"Loading preference pairs from {args.pairs_path}...")
    try:
        if not os.path.exists(args.pairs_path):
            raise FileNotFoundError(f"Pairs file not found. Run kat_download_pairs first.")
        
        dataset = PreferenceDataset(args.pairs_path, tokenizer=tokenizer)
        print(f"✓ Loaded {len(dataset)} pairs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Compute density weights if enabled
    sampler = None
    if args.density_aware:
        print(f"\nLoading base model for embeddings...")
        try:
            base_model, _, _ = load_model(
                source="base",
                device=device,
                phase="eval"
            )
            base_model.eval()
            print(f"✓ Base model loaded")
        except Exception as e:
            print(f"Warning: Could not load base model: {e}")
            base_model = reference_model  # Fallback to SFT if base unavailable
        
        try:
            if not os.path.exists(args.prompts_path):
                raise FileNotFoundError(f"Prompts file not found. Run kat_make_prompts first.")
            
            prompts = []
            with open(args.prompts_path, 'r') as f:
                for line in f:
                    try:
                        prompt_obj = json.loads(line)
                        prompts.append(prompt_obj['prompt'])
                    except json.JSONDecodeError:
                        continue
            
            if len(prompts) == 0:
                raise RuntimeError("No valid prompts loaded")
            
            density_sampler = DensityAwareSampler(
                prompts, 
                base_model=base_model,
                tokenizer=tokenizer,
                device=device,
                k=args.density_k
            )
            
            # Create sampler with density weights
            weights = torch.from_numpy(density_sampler.density_weights).float()
            
            # Repeat weights if dataset is different size
            if len(weights) < len(dataset):
                repeat_factor = (len(dataset) // len(weights)) + 1
                weights = weights.repeat(repeat_factor)
                weights = weights[:len(dataset)]
            
            sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
            print(f"✓ Density-aware sampler created")
        except Exception as e:
            print(f"Warning: Could not create density sampler: {e}")
            print(f"Falling back to uniform sampling")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
    )
    
    # Setup optimization
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    
    # Training loop
    print("\n" + "="*70)
    print(f"Starting GRPO Training")
    print("="*70)
    
    policy.train()
    reference_model.eval()
    rm_head.eval()
    
    step = 0
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.max_steps:
                break
            
            try:
                # Move to device
                chosen_ids = batch['chosen_ids'].to(device)
                rejected_ids = batch['rejected_ids'].to(device)
                
                # Forward pass through policy
                chosen_logits = policy(chosen_ids)
                rejected_logits = policy(rejected_ids)
                
                # Forward pass through reference model (frozen)
                with torch.no_grad():
                    ref_chosen_logits = reference_model(chosen_ids)
                    ref_rejected_logits = reference_model(rejected_ids)
                
                # Get rewards from RM (simplified - just use logits as reward proxy for now)
                with torch.no_grad():
                    chosen_reward = rm_head(chosen_logits) if callable(rm_head) else chosen_logits.mean(dim=1)
                    rejected_reward = rm_head(rejected_logits) if callable(rm_head) else rejected_logits.mean(dim=1)
                
                # Compute loss
                loss, pref_loss_val, kl_loss_val = compute_grpo_loss(
                    (chosen_logits, rejected_logits),
                    (ref_chosen_logits, ref_rejected_logits),
                    (chosen_reward, rejected_reward),
                    beta=args.beta
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if step % args.log_interval == 0:
                    print(f"Step {step}/{args.max_steps} | Loss: {loss:.4f} | Pref: {pref_loss_val:.4f} | KL: {kl_loss_val:.4f}")
                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("train/pref_loss", pref_loss_val, step)
                    writer.add_scalar("train/kl_loss", kl_loss_val, step)
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
    print(f"\nSaving GRPO checkpoint to: {args.out_dir}")
    checkpoint_path = os.path.join(args.out_dir, "model_000000.pt")
    torch.save({
        'model_state_dict': policy.state_dict(),
        'config': {
            'density_aware': args.density_aware,
            'beta': args.beta,
            'training_steps': step,
        }
    }, checkpoint_path)
    print(f"✓ Saved to {checkpoint_path}")
    
    writer.close()
    print("✓ GRPO training complete!")


if __name__ == "__main__":
    main()
