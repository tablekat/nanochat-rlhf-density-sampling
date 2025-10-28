#!/usr/bin/env python3
"""
Compute embeddings for conversations/prompts and project to 3D.

This script:
1. Loads preference pairs or prompts
2. Computes embeddings using sentence-transformers
3. Projects to 3D using UMAP (cluster-preserving)
4. Saves as JSON for 3D visualization in web UI

The 3D visualization shows:
- Each point = one prompt/conversation
- Point color = source dataset (hh-rlhf, ultrafeedback, stack-exchange)
- Point size = inverse of local density (rare = bigger, common = smaller)
- Interactive: hover for preview, click to select in chat UI

Usage:
  python -m scripts.kat_viz_embeddings \
    --pairs_path .cache/data/pairs_all.jsonl \
    --output_path .cache/embeddings_3d.json \
    --model_name sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from nanochat.common import get_base_dir


def load_pairs(path):
    """Load preference pairs from JSONL."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pairs file not found: {path}")
    
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                pair = json.loads(line)
                pairs.append(pair)
            except json.JSONDecodeError:
                continue
    return pairs


def extract_text_from_pair(pair):
    """Extract text for embedding from either prefix object or legacy prompt field."""
    # New format: prefix is a full conversation object
    if 'prefix' in pair and isinstance(pair['prefix'], dict):
        prefix = pair['prefix']
        messages = prefix.get('messages', [])
        text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if content:
                text_parts.append(f"{role}: {content}")
        if text_parts:
            return "\n".join(text_parts)
    
    # Legacy format: direct prompt field
    if 'prompt' in pair:
        return pair['prompt']
    
    return ""


def compute_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Compute embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    model.eval()
    
    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    return embeddings


def project_to_3d(embeddings, n_neighbors=15, min_dist=0.1):
    """Project embeddings to 3D using UMAP (cluster-preserving)."""
    try:
        import umap
    except ImportError:
        print("Error: umap not installed")
        print("Install with: pip install umap-learn")
        sys.exit(1)
    
    print(f"Projecting {len(embeddings)} embeddings to 3D with UMAP...")
    print(f"  (n_neighbors={n_neighbors}, min_dist={min_dist})")
    
    # UMAP for cluster-preserving projection
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    
    coordinates_3d = reducer.fit_transform(embeddings)
    
    # Normalize to [-1, 1] range for visualization
    coords_min = coordinates_3d.min(axis=0)
    coords_max = coordinates_3d.max(axis=0)
    coords_normalized = 2 * (coordinates_3d - coords_min) / (coords_max - coords_min + 1e-8) - 1
    
    return coords_normalized


def compute_local_density(embeddings, k=10):
    """Compute local density using k-NN."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        print("Error: scikit-learn not installed")
        print("Install with: pip install scikit-learn")
        sys.exit(1)
    
    print(f"Computing local density with k={k}...")
    
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Local density ~ 1 / average distance to k-nearest neighbors
    local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
    
    # Normalize to [0, 1]
    local_density = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-8)
    
    return local_density


def get_color_for_source(src):
    """Map source dataset to RGB color."""
    colors = {
        'hh-rlhf': [0.2, 0.8, 0.2],  # Green
        'ultrafeedback-binarized': [0.2, 0.2, 0.8],  # Blue
        'stack-exchange-preferences': [0.8, 0.2, 0.2],  # Red
    }
    return colors.get(src, [0.5, 0.5, 0.5])  # Gray default


def create_visualization_data(pairs, embeddings_3d, local_density):
    """Create JSON data for 3D visualization."""
    assert len(pairs) == len(embeddings_3d) == len(local_density)
    
    points = []
    for i, (pair, coords, density) in enumerate(zip(pairs, embeddings_3d, local_density)):
        src = pair.get('src', 'unknown')
        color = get_color_for_source(src)
        
        # Size inversely proportional to density (rare items bigger)
        size = 0.5 + (1.0 - density) * 2.0  # Range [0.5, 2.5]
        
        point = {
            'id': i,
            'x': float(coords[0]),
            'y': float(coords[1]),
            'z': float(coords[2]),
            'size': float(size),
            'color': color,
            'density': float(density),
            'source': src,
            'prompt': extract_text_from_pair(pair)[:200],  # First 200 chars
            'chosen': pair.get('chosen', '')[:100],
            'rejected': pair.get('rejected', '')[:100],
        }
        points.append(point)
    
    return {
        'points': points,
        'metadata': {
            'total_points': len(points),
            'sources': list(set(p.get('src', 'unknown') for p in pairs)),
            'colors': {
                'hh-rlhf': [0.2, 0.8, 0.2],
                'ultrafeedback-binarized': [0.2, 0.2, 0.8],
                'stack-exchange-preferences': [0.8, 0.2, 0.2],
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings and project to 3D")
    
    # Use get_base_dir() for default paths
    base_dir = get_base_dir()
    default_pairs = os.path.join(base_dir, "data", "pairs_all.jsonl")
    default_output = os.path.join(base_dir, "embeddings_3d.json")
    
    parser.add_argument("--pairs_path", default=default_pairs,
                        help="Path to pairs JSONL")
    parser.add_argument("--output_path", default=default_output,
                        help="Output JSON path")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model name")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="UMAP n_neighbors for cluster preservation")
    parser.add_argument("--min_dist", type=float, default=0.1,
                        help="UMAP min_dist")
    parser.add_argument("--density_k", type=int, default=10,
                        help="k for local density computation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("KAT Embedding Visualization: 3D Projection")
    print("=" * 70)
    print("")
    
    # Load pairs
    print(f"Loading pairs from: {args.pairs_path}")
    pairs = load_pairs(args.pairs_path)
    print(f"Loaded {len(pairs)} pairs")
    
    if args.max_samples:
        pairs = pairs[:args.max_samples]
        print(f"Limited to {len(pairs)} samples")
    print("")
    
    # Extract prompts
    prompts = [extract_text_from_pair(p) for p in pairs]
    
    # Compute embeddings
    embeddings = compute_embeddings(prompts, args.model_name)
    print(f"Embeddings shape: {embeddings.shape}")
    print("")
    
    # Project to 3D
    coords_3d = project_to_3d(embeddings, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    print(f"3D coordinates shape: {coords_3d.shape}")
    print("")
    
    # Compute local density
    density = compute_local_density(embeddings, k=args.density_k)
    print(f"Density range: [{density.min():.4f}, {density.max():.4f}]")
    print("")
    
    # Create visualization data
    viz_data = create_visualization_data(pairs, coords_3d, density)
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"✓ Saved visualization data to: {args.output_path}")
    print(f"  Total points: {len(viz_data['points'])}")
    print(f"  File size: {os.path.getsize(args.output_path) / 1e6:.1f} MB")
    print("")
    print("Next: Launch web UI and navigate to /viz to see 3D visualization")


if __name__ == "__main__":
    main()
