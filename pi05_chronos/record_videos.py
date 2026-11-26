"""Record MP4s of π0.5 Conveyor rollouts (the harness is unchanged; env.step is wrapped only to keep the frames).

    python -m pi05_chronos.record_videos --port 8300 --label pi05_chronos --condition nominal --episodes 4 --out DIR

Episodes are the first N of the paper's scene batch (default 600851, sensor seed 1801, inference seed 1811),
run in sequence in one simulator exactly as in evaluation.
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from simulation.conveyor.env import ConveyorEnv
from simulation.conveyor.evaluate import CONDITIONS, run_episode

from .eval_conveyor import Pi05ServerPolicy


def annotate(frame, lines, color):
    img = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_NEAREST)
    bar = np.full((70, 448, 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color if i == 0 else (0, 0, 0), 1,
                    cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--condition", choices=CONDITIONS, default="nominal")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--first-episode", type=int, default=600851)
    p.add_argument("--sensor-seed", type=int, default=1801)
    p.add_argument("--inference-seed", type=int, default=1811)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    policy = Pi05ServerPolicy("127.0.0.1", args.port)
    env = ConveyorEnv()
    frames = []
    step = env.step

    def recording_step(tcp, gripper):
        out = step(tcp, gripper)
        frames.extend(f["image"] for f in out)
        return out

    env.step = recording_step
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for index in range(args.episodes):
            frames.clear()
            seed = args.first_episode + index
            r = run_episode(env, policy, args.condition, seed, index, args)
            verdict = "SUCCESS" if r["success"] else ("MISSED" if r["missed"] else ("FELL" if r["fell"] else "FAIL"))
            color = (0, 140, 0) if r["success"] else (200, 0, 0)
            name = f"{args.label}_{args.condition}_scene{seed}_{verdict.lower()}.mp4"
            head = f"{args.label} | {args.condition} | {verdict}"
            sub = f"scene {seed} | belt {r['conveyor_speed']:.2f} m/s | {r['control_steps']} steps"
            # 100 Hz frames, written at 50 fps (every other frame) -> real time.
            with imageio.get_writer(args.out / name, fps=50, codec="libx264", quality=7, macro_block_size=1) as w:
                for k, f in enumerate(frames[::2]):
                    w.append_data(annotate(f, [head, sub, f"t = {2 * k / 100:.2f} s"], color))
            print(name, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
