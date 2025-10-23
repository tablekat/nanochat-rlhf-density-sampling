#!/usr/bin/env python3
"""
Train with GRPO (Generalized Reward Policy Optimization) using density-aware sampling.

This script:
1. Loads SFT checkpoint as the policy
2. Loads trained RM checkpoint for reward scoring
3. Computes prompt embeddings and local density
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


class DensityAwareSampler:
    """Compute local density of prompts and create inverse-density weights."""
    
    def __init__(self, prompts, k=10, embedding_dim=384):
        """
        Args:
            prompts: List of prompt strings
            k: Number of nearest neighbors for density estimation
            embedding_dim: Dimension of embeddings (for simple hashing initially)
        """
        self.prompts = prompts
        self.k = k
        
        # Simple embedding: use sentence-transformers in practice
        # For now, use a basic hash-based embedding
        self.embeddings = self._compute_simple_embeddings(prompts, embedding_dim)
        self.density_weights = self._compute_inverse_density_weights()
    
    def _compute_simple_embeddings(self, prompts, dim):
        """
        Create simple embeddings from text hashes.
        In practice, use sentence-transformers.sentence_transformers import SentenceTransformer
        """
        embeddings = []
        np.random.seed(42)
        for prompt in prompts:
            # Simple approach: use hash to generate pseudo-random embedding
            # This is just a placeholder for testing
            hash_val = hash(prompt)
            rng = np.random.RandomState(hash_val % (2**31))
            emb = rng.normal(0, 1, dim)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def _compute_inverse_density_weights(self):
        """
        Compute local density using k-NN.
        Weight = 1 / local_density
        """
        if len(self.prompts) < self.k + 1:
            # Too few prompts, use uniform weights
            return np.ones(len(self.prompts))
        
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
            for line in f:
                try:
                    pair = json.loads(line)
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
        
        prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_length, truncation=True)
        chosen_ids = self.tokenizer.encode(chosen, max_length=self.max_length, truncation=True)
        rejected_ids = self.tokenizer.encode(rejected, max_length=self.max_length, truncation=True)
        
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
    # Using F.kl_div which expects log-probabilities as input
    
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
    parser.add_argument("--sft_ckpt_path", default="outs/sft/ckpt.pt", help="Path to SFT checkpoint")
    parser.add_argument("--rm_ckpt_path", default="outs/rm/ckpt.pt", help="Path to RM checkpoint")
    parser.add_argument("--pairs_path", default=".cache/data/pairs_all.jsonl", help="Path to pairs")
    parser.add_argument("--prompts_path", default=".cache/data/prompts_all.jsonl", help="Path to prompts")
    parser.add_argument("--out_dir", default="outs/grpo", help="Output directory")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="KL divergence penalty")
    parser.add_argument("--device_batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--density_aware", type=bool, default=True, help="Use density-aware sampling")
    parser.add_argument("--density_k", type=int, default=10, help="k for k-NN density estimation")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load SFT checkpoint as policy
    print(f"Loading SFT checkpoint: {args.sft_ckpt_path}")
    try:
        checkpoint = torch.load(args.sft_ckpt_path, map_location=device)
        policy = checkpoint['model']
    except Exception as e:
        print(f"Error loading SFT checkpoint: {e}")
        sys.exit(1)
    
    # Load SFT checkpoint again as reference (frozen)
    print(f"Loading SFT checkpoint as reference model")
    try:
        checkpoint = torch.load(args.sft_ckpt_path, map_location=device)
        reference_model = checkpoint['model']
        reference_model.eval()  # Keep frozen
        for param in reference_model.parameters():
            param.requires_grad = False
    except Exception as e:
        print(f"Error loading reference model: {e}")
        sys.exit(1)
    
    # Load RM checkpoint
    print(f"Loading RM checkpoint: {args.rm_ckpt_path}")
    try:
        rm_checkpoint = torch.load(args.rm_ckpt_path, map_location=device)
        # Reconstruct RM head
        from scripts.kat_train_rm import RewardModelHead
        hidden_size = rm_checkpoint['config']['hidden_size']
        rm_head = RewardModelHead(hidden_size).to(device)
        rm_head.load_state_dict(rm_checkpoint['rm_head_state_dict'])
        rm_head.eval()
    except Exception as e:
        print(f"Error loading RM checkpoint: {e}")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading preference pairs: {args.pairs_path}")
    dataset = PreferenceDataset(args.pairs_path, tokenizer=policy.tokenizer)
    print(f"Dataset size: {len(dataset)} pairs")
    
    # Compute density weights if enabled
    sampler = None
    if args.density_aware:
        print("Computing local density of prompts for inverse-density weighting...")
        prompts = []
        with open(args.prompts_path, 'r') as f:
            for line in f:
                prompt_obj = json.loads(line)
                prompts.append(prompt_obj['prompt'])
        
        density_sampler = DensityAwareSampler(prompts, k=args.density_k)
        
        # Create sampler with density weights
        # Map dataset indices to prompt indices
        # For simplicity, use uniform weights for now
        weights = torch.from_numpy(density_sampler.density_weights).float()
        # Repeat weights if dataset is different size
        if len(weights) < len(dataset):
            weights = weights.repeat((len(dataset) // len(weights)) + 1)
            weights = weights[:len(dataset)]
        
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True)
        print(f"Weights min/max: {weights.min():.4f} / {weights.max():.4f}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.device_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if not using weighted sampler
    )
    
    # Setup optimization
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "logs"))
    
    # Training loop
    print("\n" + "="*70)
    print(f"Starting GRPO Training (density_aware={args.density_aware})")
    print("="*70)
    
    policy.train()
    rm_head.eval()
    
    step = 0
    for epoch in range(100):
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.max_steps:
                break
            
            # Simple training step
            # In practice, this would be more sophisticated
            try:
                chosen_ids = batch['chosen_ids'].to(device)
                rejected_ids = batch['rejected_ids'].to(device)
                
                # Forward pass through policy
                chosen_out = policy(chosen_ids, output_hidden_states=True)
                rejected_out = policy(rejected_ids, output_hidden_states=True)
                
                # Forward pass through reference model (frozen)
                with torch.no_grad():
                    ref_chosen_out = reference_model(chosen_ids, output_hidden_states=True)
                    ref_rejected_out = reference_model(rejected_ids, output_hidden_states=True)
                
                # Get rewards from RM
                with torch.no_grad():
                    chosen_reward = rm_head(chosen_out.hidden_states[-1])
                    rejected_reward = rm_head(rejected_out.hidden_states[-1])
                
                # Compute loss
                loss, pref_loss_val, kl_loss_val = compute_grpo_loss(
                    (chosen_out.logits, rejected_out.logits),
                    (ref_chosen_out.logits, ref_rejected_out.logits),
                    (chosen_reward, rejected_reward),
                    beta=args.beta
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if step % args.log_interval == 0:
                    print(f"Step {step}/{args.max_steps} - Loss: {loss:.4f}")
                    print(f"  Pref Loss: {pref_loss_val:.4f}, KL Loss: {kl_loss_val:.4f}, Total: {loss:.4f}")
                    writer.add_scalar("train/loss", loss, step)
                    writer.add_scalar("train/pref_loss", pref_loss_val, step)
                    writer.add_scalar("train/kl_loss", kl_loss_val, step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                
                step += 1
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        if step >= args.max_steps:
            break
    
    # Save checkpoint
    print(f"\nSaving GRPO checkpoint to: {args.out_dir}")
    torch.save({
        'model_state_dict': policy.state_dict(),
        'config': {
            'density_aware': args.density_aware,
            'beta': args.beta,
        }
    }, os.path.join(args.out_dir, "ckpt.pt"))
    
    writer.close()
    print("GRPO training complete!")


if __name__ == "__main__":
    main()
