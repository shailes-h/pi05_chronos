"""Evaluate a π0.5 policy server on Chronos Conveyor with the repo's unchanged harness.

    python -m pi05_chronos.eval_conveyor --port 8000 --condition nominal --first-episode 600851 \
        --sensor-seed 1801 --inference-seed 1811 --output results/x.json

`run_episode`, the timing conditions, the env and the success check all come from
`simulation.conveyor.evaluate` (not modified). This file only replaces the policy object: observations, their
capture times (relative to the request) and the requested action times go to an openpi websocket server, and the
16 returned 10-D poses come back. π0.5 latency does not matter: the harness clock is simulated.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from simulation.conveyor.evaluate import CAMERA, CONDITIONS, PROPRIO, run_episode
from simulation.conveyor.env import ConveyorEnv
from simulation.conveyor.timing import ACTION_PERIOD_S

PROMPT = "pick up the cube from the moving conveyor belt and place it in the bin"


def to_request(observations, action_times):
    """Harness observations -> the flat dict produced by openpi.training.chronos_conveyor for training."""
    images, cam_times, cam_valid = observations[CAMERA]
    proprio, proprio_times, proprio_valid = observations[PROPRIO]
    return {
        "image": np.ascontiguousarray(images[-1], dtype=np.uint8),
        "image_prev": np.ascontiguousarray(images[0], dtype=np.uint8),
        "image_valid": np.asarray(cam_valid, dtype=bool),
        "state": np.concatenate([proprio[-1], proprio[0]]).astype(np.float32),
        "cam_times": np.asarray(cam_times, dtype=np.float32),
        "proprio_times": np.asarray(proprio_times, dtype=np.float32),
        "proprio_valid": np.asarray(proprio_valid, dtype=bool),
        "request_action_times": np.asarray(action_times, dtype=np.float32),
        "prompt": PROMPT,
    }


class Pi05ServerPolicy:
    """Duck-types simulation.policy.Policy for run_episode."""

    horizon = 16
    horizons = {CAMERA: 2, PROPRIO: 2}

    def __init__(self, host, port):
        import functools

        import websockets.sync.client as ws_client
        from openpi_client import websocket_client_policy

        # The server blocks its event loop while it JIT-compiles on the first request, so keepalive pings can time
        # out. The harness clock is simulated, so waiting is harmless: disable client pings.
        if not getattr(ws_client.connect, "_no_ping", False):
            connect = functools.partial(ws_client.connect, ping_interval=None, ping_timeout=None)
            connect._no_ping = True
            ws_client.connect = connect

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.calls, self.server_ms = 0, 0.0

    def reset(self):
        pass

    def __call__(self, observations, episode, action_times=None):
        out = self.client.infer(to_request(observations, action_times))
        self.calls += 1
        self.server_ms += float(out.get("policy_timing", {}).get("infer_ms", 0.0))
        actions = np.asarray(out["actions"], dtype=np.float64)
        assert actions.shape == (self.horizon, 10), actions.shape
        return actions


def warmup(port, host="127.0.0.1"):
    """One dummy request so the server compiles before the evaluation clients start."""
    policy = Pi05ServerPolicy(host, port)
    obs = {CAMERA: (np.zeros((2, 224, 224, 3), np.uint8), np.array([-0.05, 0.0]), np.array([True, True])),
           PROPRIO: (np.zeros((2, 10), np.float32), np.array([-0.05, 0.0]), np.array([True, True]))}
    t = time.time()
    policy(obs, 0, np.arange(16) / 20.0)
    print(f"port {port} warm in {time.time() - t:.0f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--condition", choices=CONDITIONS, default="nominal")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--first-episode", type=int, default=600851)
    p.add_argument("--sensor-seed", type=int, default=1801)
    p.add_argument("--inference-seed", type=int, default=1811)
    p.add_argument("--execute-steps", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--label", default="")
    p.add_argument("--warmup", action="store_true", help="Only send one request to compile the server.")
    args = p.parse_args()
    if args.warmup:
        return warmup(args.port, args.host)
    if round(CONDITIONS[args.condition][1] / ACTION_PERIOD_S) > 16 - args.execute_steps:
        p.error("The inference delay must leave enough fresh actions for the executed chunk")
    policy = Pi05ServerPolicy(args.host, args.port)
    env = ConveyorEnv()
    episodes, start = [], time.time()
    try:
        for index in range(args.episodes):
            episodes.append(run_episode(env, policy, args.condition, args.first_episode + index, index, args))
            print(f"episode {args.first_episode + index}: success={episodes[-1]['success']}", flush=True)
    finally:
        env.close()
    successes = sum(e["success"] for e in episodes)
    result = dict(
        task="conveyor",
        condition=args.condition,
        checkpoint=args.label,
        success_rate=successes / len(episodes),
        successes=successes,
        settings={k: v for k, v in vars(args).items() if k not in ("output",)},
        wall_s=time.time() - start,
        policy_calls=policy.calls,
        mean_server_infer_ms=policy.server_ms / max(policy.calls, 1),
        episodes=episodes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.condition}: {successes}/{len(episodes)} successes")


if __name__ == "__main__":
    main()
