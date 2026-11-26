"""Fixed train/val episode lists for the 500-episode π0.5 study.

Train: 500 episodes drawn (seed 0) from the repo's train split (episodes with index % 10 != 0).
Val:    50 episodes drawn (seed 0) from the repo's held-out split (index % 10 == 0).
Same split rule as chronos_policy simulation/export.py::split (the zarr stores no split).
"""
import argparse, json
from pathlib import Path
import numpy as np
import zarr

p = argparse.ArgumentParser()
p.add_argument("--zarr", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--train", type=int, default=500)
p.add_argument("--val", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
n = len(zarr.open(str(a.zarr), "r").keys())
idx = np.arange(n)
rng = np.random.default_rng(a.seed)
train = sorted(rng.permutation(idx[idx % 10 != 0])[: a.train].tolist())
val = sorted(rng.permutation(idx[idx % 10 == 0])[: a.val].tolist())
assert not set(train) & set(val)
a.output.write_text(json.dumps(dict(seed=a.seed, n_episodes=n, train=train, val=val), indent=1) + "\n")
print(f"train {len(train)} val {len(val)} -> {a.output}")
