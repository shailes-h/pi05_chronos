"""Results table: π0.5-orig / π0.5-chronos (500 eps) vs the published Chronos / DT checkpoints (M0 reproduction).

Success % per condition; π0.5 rows = mean ± std over the 3 paper scene batches (one training seed);
Chronos/DT rows = mean ± std over 3 training seeds (each averaging the 3 batches), as in the paper table.
"""
import glob, json, sys
import numpy as np

W = "/scratch/Projects/CFP-05/CFP05-CF-002/chronos_pi05"
CONDS = ["nominal", "sensor_id", "sensor_ood", "inference_0_100ms", "inference_0_200ms",
         "combined_sensor_ood_inference_0_200ms"]
NAMES = ["Nominal", "Sensor ID", "Sensor OOD", "Inf 0–100 ms", "Inf 0–200 ms", "Combined"]


def rate(f):
    return json.load(open(f))["success_rate"] * 100


rows = []
for root, label, name in ((f"{W}/results/pi05_d500", "pi05_orig", "π0.5-orig (500 eps)"),
                          (f"{W}/results/pi05_d500", "pi05_chronos", "π0.5-chronos (500 eps)"),
                          (f"{W}/results/pi05_full", "pi05_orig_full", "π0.5-orig (full, 1,567 eps)"),
                          (f"{W}/results/pi05_full", "pi05_chronos_full", "π0.5-chronos (full, 1,567 eps)"),
                         (f"{W}/results/pi05_full_e10", "pi05_chronos_full_e10", "π0.5-chronos (full, 10 epochs)"),
                         (f"{W}/results/pi05_full_e20", "pi05_chronos_full_e20", "π0.5-chronos (full, 20 epochs)"),
                         (f"{W}/results/pi05_full_1frame_e20", "pi05_chronos_full_1frame_e20", "π0.5-chronos (full, 20 epochs, 1 frame)")):
    cells = []
    for c in CONDS:
        v = [rate(f) for f in sorted(glob.glob(f"{root}/{label}/scenes*/{c}.json"))]
        cells.append(f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f} (n={len(v)})" if len(v) > 1 else (f"{v[0]:.1f}" if v else "—"))
    rows.append((name, cells))
for m, name in (("dt", "DT (published ckpt, full data)"), ("chronos", "Chronos (published ckpt, full data)")):
    cells = []
    for c in CONDS:
        seeds = [np.mean([rate(f) for f in glob.glob(f"{W}/results/m0_paper/conveyor/{m}/seed{s}/*/{c}.json")]) for s in range(3)]
        cells.append(f"{np.mean(seeds):.1f} ± {np.std(seeds, ddof=1):.1f}")
    rows.append((name, cells))
out = ["| Model | " + " | ".join(NAMES) + " |", "| --- |" + " ---: |" * len(NAMES)]
out += [f"| {n} | " + " | ".join(c) + " |" for n, c in rows]
print("\n".join(out))
