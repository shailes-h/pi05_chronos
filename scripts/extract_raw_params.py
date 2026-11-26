"""Make a servable checkpoint from the RAW (non-EMA) params stored in an openpi train_state.

openpi saves EMA params as `<step>/params` and the raw params inside `<step>/train_state`. With only ~400 steps an
EMA decay of 0.999 leaves the EMA weights mostly at pi05_base, so we serve the raw weights instead.

    python extract_raw_params.py <step_dir> <out_dir>
Writes <out_dir>/params (same format as openpi's) and copies <step_dir>/assets.
"""
import pathlib
import shutil
import sys

import jax
import numpy as np
import orbax.checkpoint as ocp

src, dst = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
with ocp.PyTreeCheckpointer() as ckptr:
    meta = ckptr.metadata(src / "train_state")
    print("train_state keys:", list(meta.keys()))
    item = {"params": meta["params"]}
    args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), item)
    try:
        raw = ckptr.restore(src / "train_state", ocp.args.PyTreeRestore(item=item, restore_args=args, partial_restore=True))
    except TypeError:  # older orbax without partial_restore
        full = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), dict(meta))
        raw = ckptr.restore(src / "train_state", ocp.args.PyTreeRestore(item=dict(meta), restore_args=full))
    params = raw["params"]
    n = sum(x.size for x in jax.tree.leaves(params))
    print(f"raw params: {n / 1e9:.3f} B values, dtype {jax.tree.leaves(params)[0].dtype}")
    dst.mkdir(parents=True, exist_ok=True)
    ckptr.save(dst / "params", {"params": params}, force=True)
shutil.copytree(src / "assets", dst / "assets", dirs_exist_ok=True)
print("wrote", dst)
