"""Build the shared BiCAT training data (one GPU job, chronos env).

    python -m pi05_chronos.bicat_build --raw data/bicat/raw/catchball --out data/bicat/catchball

Writes to --out:
  episodes.npz            per-frame state [N, 20], actions [N, 20], timestamp, episode_index; per-episode offsets
  frames224_<cam>.npy     uint8 [N, 224, 224, 3], openpi resize_with_pad (π0.5 inputs)
  siglip_<cam>.npy        float16 [N, 256, 768] frozen SigLIP2 patch features (first, frozen-encoder recipe; superseded)
  frames240_<cam>.npy     uint8 [N, 240, 240, 3] antialiased resize (--frames240): input of the paper's real-world
                          image pipeline (212 px crop -> 256 px, fine-tuned SigLIP2) in pi05_chronos.bicat_train_paper
  windows_{train,val}.npz window table (bicat.build_windows): train randomized (seed 0), val nominal
  chronos/{train,val}/episode_<id>.pt + chronos/config.json   Chronos trainer format; camera observations are
                          global feature rows (resolved lazily by pi05_chronos.bicat_train_chronos)
Video frame for a data row = round((episode from_timestamp + timestamp) * fps), LeRobot's convention.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from pi05_chronos import bicat

OPENPI_IMAGE_TOOLS = Path(__file__).resolve().parents[2] / "openpi/packages/openpi-client/src/openpi_client/image_tools.py"


def image_tools():
    spec = importlib.util.spec_from_file_location("image_tools", OPENPI_IMAGE_TOOLS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_table(raw):
    d = pq.read_table(raw / "data/chunk-000/file-000.parquet").to_pydict()
    order = np.argsort(d["index"])
    assert (np.asarray(d["index"])[order] == np.arange(len(order))).all()
    col = lambda k: np.asarray(d[k])[order]  # noqa: E731
    eps = pq.read_table(raw / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    return {
        "state": bicat.pose20(np.stack(col("observation.state_eef_absolute"))),
        "actions": bicat.pose20(np.stack(col("action_eef_absolute"))),
        "timestamp": col("timestamp").astype(np.float64),
        "episode_index": col("episode_index").astype(np.int64),
        "frame_index": col("frame_index").astype(np.int64),
    }, eps


def decode_camera(raw, cam, table, eps, fps, out, encoder, device):
    from simulation.policy import encode_images

    tools = image_tools()
    n = len(table["timestamp"])
    starts = {e["episode_index"]: e[f"videos/observation.images.{cam}/from_timestamp"] for e in eps}
    video_frame = np.round((np.array([starts[e] for e in table["episode_index"]]) + table["timestamp"]) * fps)
    video_frame = video_frame.astype(np.int64)
    rows_by_frame = {}
    for row, f in enumerate(video_frame):
        rows_by_frame.setdefault(int(f), []).append(row)
    frames224 = np.lib.format.open_memmap(out / f"frames224_{cam}.npy", "w+", np.uint8, (n, 224, 224, 3))
    feats = None
    filled = np.zeros(n, bool)
    pending_rows, pending_imgs = [], []

    def flush():
        nonlocal feats
        if not pending_rows:
            return
        f = encode_images(encoder, np.stack(pending_imgs), device, batch_size=256).half().cpu().numpy()
        if feats is None:
            feats = np.lib.format.open_memmap(out / f"siglip_{cam}.npy", "w+", np.float16, (n, *f.shape[1:]))
        for rows, x in zip(pending_rows, f):
            feats[rows] = x
        pending_rows.clear()
        pending_imgs.clear()

    container = av.open(str(raw / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    t0, decoded = time.time(), 0
    for frame in container.decode(stream):
        decoded += 1
        f = int(round(float(frame.pts * stream.time_base) * fps))
        rows = rows_by_frame.get(f)
        if rows is None:
            continue
        img = frame.to_ndarray(format="rgb24")
        frames224[rows] = tools.resize_with_pad(img, 224, 224, method=Image.BILINEAR)
        filled[rows] = True
        pending_rows.append(rows)
        pending_imgs.append(img)
        if len(pending_imgs) == 512:
            flush()
    flush()
    container.close()
    frames224.flush()
    feats.flush()
    missing = np.flatnonzero(~filled)
    print(f"{cam}: decoded {decoded} video frames in {time.time() - t0:.0f} s, {n - len(missing)}/{n} rows filled",
          flush=True)
    if len(missing):
        raise RuntimeError(f"{cam}: {len(missing)} rows without a video frame, e.g. {missing[:5]}")


def frames240(raw, cam, table, eps, fps, out):
    """Squash-resize (as Chronos's preprocessing does) with antialiasing to 240 x 240 uint8."""
    n = len(table["timestamp"])
    starts = {e["episode_index"]: e[f"videos/observation.images.{cam}/from_timestamp"] for e in eps}
    video_frame = np.round((np.array([starts[e] for e in table["episode_index"]]) + table["timestamp"]) * fps)
    rows_by_frame = {}
    for row, f in enumerate(video_frame.astype(np.int64)):
        rows_by_frame.setdefault(int(f), []).append(row)
    store = np.lib.format.open_memmap(out / f"frames240_{cam}.npy", "w+", np.uint8, (n, 240, 240, 3))
    filled = np.zeros(n, bool)
    container = av.open(str(raw / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    t0 = time.time()
    for frame in container.decode(stream):
        rows = rows_by_frame.get(int(round(float(frame.pts * stream.time_base) * fps)))
        if rows is None:
            continue
        store[rows] = np.asarray(frame.to_image().resize((240, 240), Image.BILINEAR, reducing_gap=None))
        filled[rows] = True
    container.close()
    store.flush()
    print(f"{cam}: frames240 in {time.time() - t0:.0f} s, {filled.sum()}/{n} rows", flush=True)
    if not filled.all():
        raise RuntimeError(f"{cam}: rows without a video frame")


def chronos_episodes(table, windows, directory):
    directory.mkdir(parents=True, exist_ok=True)
    per_arm = {"robot0_10d": slice(0, 10), "robot1_10d": slice(10, 20)}
    for e in np.unique(windows["episode"]):
        m = windows["episode"] == e
        w = {k: v[m] for k, v in windows.items()}
        ep = {"observations": {}, "capture_times": {}, "valid": {}, "feature_rows": {}}
        for name in bicat.STREAMS:
            ep["capture_times"][name] = torch.from_numpy(w[f"{name}/times"])
            ep["valid"][name] = torch.from_numpy(w[f"{name}/valid"])
            if name in bicat.CAMERAS:
                ep["feature_rows"][name] = torch.from_numpy(w[f"{name}/index"])
            else:
                ep["observations"][name] = torch.from_numpy(table["state"][w[f"{name}/index"]][..., per_arm[name]])
        ep["actions"] = torch.from_numpy(table["actions"][w["targets"]])
        ep["action_times"] = torch.from_numpy(w["action_times"])
        torch.save(ep, directory / f"episode_{int(e)}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--frames240", action="store_true", help="Only (re)build windows and the 240 px frame store.")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.raw / "meta/info.json").read_text())
    fps = info["fps"]
    assert fps == bicat.FPS
    table, eps = read_table(args.raw)
    ids = np.unique(table["episode_index"])
    offsets = np.searchsorted(table["episode_index"], ids)
    lengths = np.diff(np.append(offsets, len(table["episode_index"])))
    assert (table["frame_index"] == np.arange(len(table["frame_index"])) - np.repeat(offsets, lengths)).all()
    np.savez(args.out / "episodes.npz", **table, ids=ids, offsets=offsets, lengths=lengths)
    train_ids, val_ids = bicat.split(ids.tolist())
    split_info = {"train": train_ids, "val": val_ids}
    (args.out / "split.json").write_text(json.dumps(split_info) + "\n")
    for name, chosen in split_info.items():
        episodes = [{"id": e, "offset": int(offsets[i]), "timestamp": table["timestamp"][offsets[i]:offsets[i] + lengths[i]]}
                    for i, e in enumerate(ids) if e in set(chosen)]
        w = bicat.build_windows(episodes, randomized=name == "train", seed=args.seed)
        np.savez(args.out / f"windows_{name}.npz", **w)
        chronos_episodes(table, w, args.out / "chronos" / name)
        print(f"{name}: {len(chosen)} episodes, {len(w['episode'])} windows", flush=True)
    config = dict(
        action_dim=20,
        proprio_dims={"robot0_10d": 10, "robot1_10d": 10},
        cameras=list(bicat.CAMERAS),
        observation_horizons={name: 2 for name in bicat.STREAMS},
        feature_dim=768,
        patch_grid=16,
        horizon=bicat.HORIZON,
        normalizer_ignore_indices={"robot0_10d": bicat.POSE_ROTATION, "robot1_10d": bicat.POSE_ROTATION,
                                   "actions": bicat.POSE_ROTATION + [i + 10 for i in bicat.POSE_ROTATION]},
    )
    (args.out / "chronos" / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.frames240:
        for cam in bicat.CAMERAS:
            frames240(args.raw, cam, table, eps, fps, args.out)
        return
    if args.skip_video:
        return
    from simulation.policy import load_encoder

    encoder = load_encoder(args.device)
    for cam in bicat.CAMERAS:
        decode_camera(args.raw, cam, table, eps, fps, args.out, encoder, args.device)


if __name__ == "__main__":
    main()
