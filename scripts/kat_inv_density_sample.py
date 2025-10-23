"""
python scripts/kat_inv_density_sample.py \
  --emb data/my_conversations_emb.pt \
  --k 20 \
  --budget 50000 \
  --out_indices data/ids.invden.sampled.txt
"""
import argparse, torch, random, math
import numpy as np
import faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help=".pt from embed.py")
    ap.add_argument("--k", type=int, default=20, help="neighbors for density")
    ap.add_argument("--out_indices", required=True, help="text file: one id per line (sampled)")
    ap.add_argument("--budget", type=int, required=True, help="how many to sample")
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()

    blob = torch.load(args.emb, map_location="cpu")
    ids = blob["ids"]
    X = blob["emb"].numpy().astype("float32")  # [N,d], already L2-normalized

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    # self-included knn
    D, I = index.search(X, args.k + 1)  # first neighbor is self
    # density = mean cosine sim to k nearest neighbors (excluding self)
    sim = D[:,1:].mean(axis=1)
    # map similarity to density proxy in [0, +]:
    # higher sim -> higher density; keep non-negative
    density = np.maximum(0.0, sim)
    weights = 1.0 / (args.eps + density)
    # normalize
    p = weights / weights.sum()

    # sample without replacement using probabilities
    # numpy choice with p and replace=False
    chosen = np.random.choice(len(ids), size=min(args.budget, len(ids)), replace=False, p=p)
    with open(args.out_indices, "w", encoding="utf-8") as f:
        for i in chosen:
            f.write(str(ids[i]) + "\n")

    print(f"wrote {args.out_indices} with {len(chosen)} ids")

if __name__ == "__main__":
    main()
