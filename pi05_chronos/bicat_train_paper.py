"""Train Chronos / DT on BiCAT with the paper's REAL-WORLD recipe (Chronos Policy, App. A.1/A.3).

    python -m pi05_chronos.bicat_train_paper --method chronos --output runs/bicat_paper/chronos

The released `scripts/train.py` only trains on frozen, precomputed SigLIP features. The paper's real-world policies
instead fine-tune the image encoder, so this trainer keeps chronos_policy's model code unmodified
(`chronos.policy.ChronosPolicy` with `vision=SiglipVisionModel`, its RGB path) and adds the paper's training loop:

  * SigLIP2 base (google/siglip2-base-patch16-256), fine-tuned at 0.1x the learning rate;
  * images: 212 px random crop during training, center crop at evaluation, antialiased resize to 256 x 256; color
    jitter with probability 0.3 (magnitudes not given in the paper: brightness/contrast/saturation x U(0.7, 1.3));
    the input is the 240 x 240 squash-resized frame (the paper does not state the pre-crop size);
  * 16-layer, width-512, 8-head diffusion transformer (chronos_policy defaults); l1 loss on clean actions;
    cosine noise schedule; DDIM 48 train / 16 inference steps (paper: 50/16, but chronos_policy requires
    train steps divisible by inference steps);
  * AdamW lr 3e-4, betas (0.95, 0.999), weight decay 1e-6, cosine decay with 2,000 warmup updates, BF16 autocast,
    batch 128, 50 epochs; grad clip 1.0 (as in scripts/train.py);
  * range normalization with 6D rotation coordinates left unchanged (scripts/train.py's fit_normalizers);
  * Chronos = time positional encoding with capture/action timestamps; DT (the paper's "DP") = index positions.

Checkpoints use scripts/train.py's format (best.pt by validation loss on nominal-timing windows, last.pt) plus
`vision` = encoder name, so they load with ChronosPolicy(**config, vision=SiglipVisionModel.from_pretrained(...)).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pi05_chronos import bicat

CHRONOS = Path("/scratch/shailesh.xml/chronos_policy")
sys.path.insert(0, str(CHRONOS))
sys.path.insert(0, str(CHRONOS / "scripts"))
import train as chronos_train  # noqa: E402  (scripts/train.py: fit_normalizers, Episodes)
from chronos.policy import ChronosPolicy  # noqa: E402

ENCODER = "google/siglip2-base-patch16-256"
CROP, SIZE = 212, 256


class Windows(Dataset):
    def __init__(self, root, split):
        self.root, self.split = Path(root), split
        self._load()

    def _load(self):
        with np.load(self.root / f"windows_{self.split}.npz") as z:
            self.w = {k: z[k] for k in z.files}
        with np.load(self.root / "episodes.npz") as z:
            self.state, self.actions = z["state"], z["actions"]
        self.frames = None

    def __getstate__(self):
        return {"root": self.root, "split": self.split}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load()

    def __len__(self):
        return len(self.w["episode"])

    def __getitem__(self, i):
        w = self.w
        if self.frames is None:
            self.frames = {c: np.load(self.root / f"frames240_{c}.npy", mmap_mode="r") for c in bicat.CAMERAS}
        obs, times, valid = {}, {}, {}
        for name in bicat.STREAMS:
            index = w[f"{name}/index"][i]
            if name in bicat.CAMERAS:
                obs[name] = torch.from_numpy(np.ascontiguousarray(self.frames[name][index]))  # [2, 240, 240, 3]
            else:
                arm = bicat.PROPRIO.index(name)
                obs[name] = torch.from_numpy(self.state[index][:, 10 * arm : 10 * arm + 10])
            times[name] = torch.from_numpy(w[f"{name}/times"][i])
            valid[name] = torch.from_numpy(w[f"{name}/valid"][i])
        return {"observations": obs, "capture_times": times, "valid": valid,
                "actions": torch.from_numpy(self.actions[w["targets"][i]]),
                "action_times": torch.from_numpy(w["action_times"][i])}


def images(rgb_uint8, train, generator):
    """[B, T, H, W, 3] uint8 -> [B, T, 3, 256, 256] float in [0, 1]; one crop / jitter per sample and camera."""
    b, t, h, w, _ = rgb_uint8.shape
    x = rgb_uint8.permute(0, 1, 4, 2, 3).float().div_(255)
    if train:
        top = torch.randint(0, h - CROP + 1, (b,), generator=generator, device="cpu").tolist()
        left = torch.randint(0, w - CROP + 1, (b,), generator=generator, device="cpu").tolist()
    else:
        top, left = [(h - CROP) // 2] * b, [(w - CROP) // 2] * b
    x = torch.stack([x[k, :, :, top[k] : top[k] + CROP, left[k] : left[k] + CROP] for k in range(b)])
    x = F.interpolate(x.flatten(0, 1), size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    x = x.view(b, t, 3, SIZE, SIZE)
    if train:
        apply = (torch.rand(b, generator=generator) < 0.3).to(x.device)
        f = (0.7 + 0.6 * torch.rand(b, 3, generator=generator)).to(x.device)
        f = torch.where(apply[:, None], f, torch.ones_like(f))
        x = x * f[:, 0, None, None, None, None]  # brightness
        mean = x.mean((2, 3, 4), keepdim=True)
        x = (x - mean) * f[:, 1, None, None, None, None] + mean  # contrast
        gray = (0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2])[:, :, None]
        x = (x - gray) * f[:, 2, None, None, None, None] + gray  # saturation
        x = x.clamp_(0, 1)
    return x


def prepare(batch, normalizers, device, train, generator):
    batch = chronos_train.tree_map(lambda v: v.to(device, non_blocking=True), batch)
    obs = batch["observations"]
    for name in bicat.CAMERAS:
        obs[name] = images(obs[name], train, generator)
    for name in bicat.PROPRIO:
        s = normalizers[name]
        obs[name] = (obs[name].float() - s["offset"]) / s["scale"]
    s = normalizers["actions"]
    actions = (batch["actions"].float() - s["offset"]) / s["scale"]
    return obs, batch, actions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("chronos", "dt"), required=True)
    p.add_argument("--root", type=Path, default=Path("data/bicat/catchball"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder-lr-mult", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    # Paper: 50 train / 16 inference steps; chronos_policy requires divisibility -> 48 (closest), 16 kept.
    p.add_argument("--train-steps", type=int, default=48)
    p.add_argument("--inference-steps", type=int, default=16)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0, help="Stop after N updates (smoke test).")
    args = p.parse_args()
    from transformers import SiglipVisionModel

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = json.loads((args.root / "chronos" / "config.json").read_text())
    ignored = config.pop("normalizer_ignore_indices")
    config.update(positional_encoding="time" if args.method == "chronos" else "index", seed=args.seed,
                  train_steps=args.train_steps, inference_steps=args.inference_steps)
    normalizers = chronos_train.fit_normalizers(chronos_train.Episodes(args.root / "chronos" / "train"), config,
                                                ignored)
    gpu_norm = chronos_train.tree_map(lambda v: v.to(device), normalizers)
    vision = SiglipVisionModel.from_pretrained(ENCODER)
    policy = ChronosPolicy(**config, vision=vision).to(device)
    enc = [p_ for n, p_ in policy.named_parameters() if n.startswith("vision.")]
    rest = [p_ for n, p_ in policy.named_parameters() if not n.startswith("vision.")]
    assert enc and rest
    optimizer = torch.optim.AdamW(
        [{"params": rest, "lr": args.lr}, {"params": enc, "lr": args.lr * args.encoder_lr_mult}],
        betas=(0.95, 0.999), weight_decay=args.weight_decay)
    train_set, val_set = Windows(args.root, "train"), Windows(args.root, "val")
    loader_gen = torch.Generator().manual_seed(args.seed)
    aug_gen = torch.Generator().manual_seed(args.seed + 1)
    opts = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_set, shuffle=True, generator=loader_gen, drop_last=False, **opts)
    val_loader = DataLoader(val_set, shuffle=False, **opts)
    total = args.epochs * len(train_loader)
    if args.max_steps:
        total = min(total, args.max_steps)

    def lr_factor(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - args.warmup) / max(1, total - args.warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    settings.update(vision=ENCODER, crop=CROP, size=SIZE, frames="frames240", total_updates=total)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{args.method}: {len(train_set)} train / {len(val_set)} val windows, {len(train_loader)} updates/epoch, "
          f"{total} total; params {sum(p_.numel() for p_ in policy.parameters()) / 1e6:.1f} M "
          f"(encoder {sum(p_.numel() for p_ in enc) / 1e6:.1f} M)", flush=True)
    step, best, history = 0, float("inf"), []
    for epoch in range(args.epochs):
        policy.train()
        train_sum, n_train, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            obs, b, actions = prepare(batch, gpu_norm, device, True, aug_gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = policy.loss(obs, b["capture_times"], b["action_times"], actions, valid=b["valid"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            train_sum += loss.item() * len(actions)
            n_train += len(actions)
            if step % 50 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / (n_train / args.batch_size):.2f} s/step", flush=True)
            if step >= total:
                break
        policy.eval()
        val_sum = 0.0
        with torch.random.fork_rng(devices=[device.index or 0]), torch.no_grad():
            torch.manual_seed(args.seed)
            for batch in val_loader:
                obs, b, actions = prepare(batch, gpu_norm, device, False, None)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = policy.loss(obs, b["capture_times"], b["action_times"], actions, valid=b["valid"])
                val_sum += loss.item() * len(actions)
        val_loss = val_sum / len(val_set)
        improved = val_loss < best
        best = min(best, val_loss)
        record = {"epoch": epoch + 1, "step": step, "train_loss": train_sum / max(n_train, 1), "val_loss": val_loss,
                  "epoch_s": round(time.time() - t0)}
        history.append(record)
        ckpt = dict(config=config, settings=settings, normalizers=normalizers, normalizer_ignore_indices=ignored,
                    vision=ENCODER, state_dict=policy.state_dict(), epoch=epoch + 1, best_val_loss=best,
                    history=history)
        chronos_train.save_checkpoint(args.output / "last.pt", ckpt)
        if improved:
            chronos_train.save_checkpoint(args.output / "best.pt", ckpt)
        (args.output / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        if step >= total:
            break


if __name__ == "__main__":
    main()
