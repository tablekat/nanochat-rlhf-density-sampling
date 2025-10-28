#!/usr/bin/env python3
"""
Visualize precomputed embeddings in 3D.

This script:
1. Loads precomputed embeddings from kat_compute_embeddings_offline
2. Loads prefixes from prefixes_all.jsonl (the exact data that was embedded)
3. Projects to 3D using UMAP (cluster-preserving)
4. Saves as JSON for 3D visualization in web UI
5. Uses the SAME embeddings as RM/GRPO training

The 3D visualization shows:
- Each point = one unique prefix (conversation start)
- Point color = source dataset (hh-rlhf, ultrafeedback, stack-exchange)
- Point size = inverse of local density (rare = bigger, common = smaller)
- Interactive: hover for preview, click to select in chat UI

Usage:
  python -m scripts.kat_viz_embeddings \\
    --embeddings_dir data/embeddings_offline \\
    --prefixes_path data/prefixes_all.jsonl \\
    --output_path embeddings_3d.json

Note: Requires kat_compute_embeddings_offline to have been run first!
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from nanochat.common import get_base_dir


def load_precomputed_embeddings(embeddings_dir):
    """Load embeddings and density weights from precomputed files."""
    embeddings_path = os.path.join(embeddings_dir, "embeddings.npy")
    weights_path = os.path.join(embeddings_dir, "density_weights.npy")
    ids_path = os.path.join(embeddings_dir, "ids.json")
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}\n"
                              "Run kat_compute_embeddings_offline first!")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Density weights not found: {weights_path}")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"IDs not found: {ids_path}")
    
    embeddings = np.load(embeddings_path)  # [N, D]
    weights = np.load(weights_path)  # [N,]
    ids = json.load(open(ids_path))  # List of IDs
    
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded weights: {weights.shape}")
    print(f"Loaded IDs: {len(ids)}")
    
    assert len(embeddings) == len(weights) == len(ids), \
        "Embeddings, weights, and IDs length mismatch"
    
    return embeddings, weights, ids


def load_prefixes(path):
    """Load prefixes from prefixes_all.jsonl."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prefixes file not found: {path}")
    
    prefixes = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                prefix_obj = json.loads(line)
                prefixes.append(prefix_obj)
            except json.JSONDecodeError:
                continue
    return prefixes


def extract_text_from_prefix(prefix_obj):
    """Extract text for display from prefix object."""
    if 'prefix' in prefix_obj and isinstance(prefix_obj['prefix'], dict):
        prefix = prefix_obj['prefix']
        messages = prefix.get('messages', [])
        text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if content:
                text_parts.append(f"{role}: {content}")
        if text_parts:
            return "\n".join(text_parts)
    
    return ""


def get_color_for_source(src):
    """Map source dataset to RGB color."""
    colors = {
        'hh-rlhf': [0.2, 0.8, 0.2],  # Green
        'ultrafeedback-binarized': [0.2, 0.2, 0.8],  # Blue
        'stack-exchange-preferences': [0.8, 0.2, 0.2],  # Red
    }
    return colors.get(src, [0.5, 0.5, 0.5])  # Gray default


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
        n_neighbors=min(n_neighbors, len(embeddings) - 1),  # Handle small datasets
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
    
    k = min(k, len(embeddings) - 1)  # Handle small datasets
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Local density ~ 1 / average distance to k-nearest neighbors
    local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-6)
    
    # Normalize to [0, 1]
    local_density = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-8)
    
    return local_density


def create_visualization_data(prefixes, ids, embeddings_3d, local_density, precomputed_weights):
    """Create JSON data for 3D visualization."""
    # Direct 1:1 mapping since prefixes are already in same order as embeddings
    assert len(prefixes) == len(ids) == len(embeddings_3d) == len(local_density) == len(precomputed_weights), \
        f"Length mismatch: {len(prefixes)} prefixes vs {len(ids)} ids vs {len(embeddings_3d)} embeddings"
    
    points = []
    for prefix_obj, prefix_id, coords, density, weight in zip(prefixes, ids, embeddings_3d, local_density, precomputed_weights):
        # Try to infer source from the prefix (may not be stored, so default to unknown)
        src = prefix_obj.get('src', 'unknown')
        color = get_color_for_source(src)
        
        # Size inversely proportional to density (rare items bigger)
        size = 0.5 + (1.0 - density) * 2.0  # Range [0.5, 2.5]
        
        point = {
            'id': prefix_id,
            'x': float(coords[0]),
            'y': float(coords[1]),
            'z': float(coords[2]),
            'size': float(size),
            'color': color,
            'density': float(density),
            'weight': float(weight),
            'source': src,
            'prefix': extract_text_from_prefix(prefix_obj)[:500],  # First 500 chars
        }
        points.append(point)
    
    return {
        'points': points,
        'metadata': {
            'embedding_source': 'kat_compute_embeddings_offline',
            'embedding_dim': int(embeddings_3d.shape[1]),
            'projection': '3D UMAP',
            'total_points': len(points),
            'colors': {
                'hh-rlhf': [0.2, 0.8, 0.2],
                'ultrafeedback-binarized': [0.2, 0.2, 0.8],
                'stack-exchange-preferences': [0.8, 0.2, 0.2],
            },
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize precomputed embeddings in 3D"
    )
    
    base_dir = get_base_dir()
    default_embeddings_dir = os.path.join(base_dir, "data", "embeddings_offline")
    default_prefixes = os.path.join(base_dir, "data", "prefixes_all.jsonl")
    default_output = os.path.join(base_dir, "embeddings_3d.json")
    
    parser.add_argument("--embeddings_dir", default=default_embeddings_dir,
                        help="Path to embeddings_offline directory (from kat_compute_embeddings_offline)")
    parser.add_argument("--prefixes_path", default=default_prefixes,
                        help="Path to prefixes_all.jsonl (exact data that was embedded)")
    parser.add_argument("--output_path", default=default_output,
                        help="Output JSON path for 3D visualization")
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
    print("KAT Embedding Visualization: 3D Projection from Precomputed Embeddings")
    print("=" * 70)
    print("")
    
    # Load precomputed embeddings
    print(f"Loading precomputed embeddings from: {args.embeddings_dir}")
    try:
        embeddings, weights, ids = load_precomputed_embeddings(args.embeddings_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    print("")
    
    # Load prefixes
    print(f"Loading prefixes from: {args.prefixes_path}")
    try:
        prefixes = load_prefixes(args.prefixes_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    print(f"Loaded {len(prefixes)} prefixes")
    
    # Validate consistency
    if len(prefixes) != len(embeddings):
        print(f"⚠️  Warning: {len(prefixes)} prefixes but {len(embeddings)} embeddings")
        print(f"   Taking minimum ({min(len(prefixes), len(embeddings))})")
        min_len = min(len(prefixes), len(embeddings))
        prefixes = prefixes[:min_len]
        embeddings = embeddings[:min_len]
        weights = weights[:min_len]
        ids = ids[:min_len]
    
    if args.max_samples:
        prefixes = prefixes[:args.max_samples]
        embeddings = embeddings[:args.max_samples]
        weights = weights[:args.max_samples]
        ids = ids[:args.max_samples]
        print(f"Limited to {len(prefixes)} samples")
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
    viz_data = create_visualization_data(prefixes, ids, coords_3d, density, weights)
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"✓ Saved visualization data to: {args.output_path}")
    print(f"  Total points: {len(viz_data['points'])}")
    print(f"  Embedding source: {viz_data['metadata']['embedding_source']}")
    print(f"  File size: {os.path.getsize(args.output_path) / 1e6:.1f} MB")
    print("")
    print("Next: Launch web UI and navigate to /viz to see 3D visualization")


if __name__ == "__main__":
    main()
