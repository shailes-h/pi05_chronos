"""BiCAT YAM (LeRobot v3.0, 30 Hz, bimanual) -> Chronos-style training windows shared by all four policies.

Follows Chronos's export (`simulation/export.py::windows`, same RNG draws in the same order) with the paper's
REAL-WORLD timing ranges (Table 6), on a 15 Hz control grid because BiCAT is recorded at 30 Hz (the paper's 20 Hz
would fall between frames):

  * one window every control step (query frames 0, 2, 4, ...), 16 action targets at 15 Hz (1.07 s);
  * every stream (3 cameras + 2 arm poses) keeps 2 samples, nominally one control step apart;
  * randomized timing (train split only), drawn once per window with rng([seed, episode position]):
      inference delay D in {0..3} steps (0-200 ms, the paper's inference condition);
      per stream an older-sample interval drawn from the paper's rate range (RGB 12.5-33.33 Hz, pose/gripper
      8.33-33.33 Hz) and an independent delivery delay U{0..3} frames (0-100 ms);
  * validation windows use the nominal grid;
  * action targets: frames q + 2 (D + j), action_times = (D + j) * 2 / 30 s (action_offset = 0).

Poses: per arm pos(3) + rot6d(6) + gripper(1) = 10-D (rot6d = first two rows of the rotation matrix);
state = 2 x 10-D, actions = 20-D (left arm first).
"""

import numpy as np
from scipy.spatial.transform import Rotation

FPS = 30
PERIOD = 2  # frames per 15 Hz control step
HORIZON = 16
QUERY_PHASE = 0
ACTION_OFFSET = 0
INTERVAL_SCALE = (0.6, 1.65)
SENSOR_DELAY_FRAMES = 3  # up to 100 ms (paper Table 6)
INFERENCE_DELAYS = (0, 1, 2, 3)  # 0-200 ms in 66.7 ms steps
ACTION_PERIOD_S = PERIOD / FPS

CAMERAS = ("top", "left", "right")  # observation.images.<name>; left/right are the wrist cameras
PROPRIO = ("robot0_10d", "robot1_10d")  # left arm, right arm
STREAMS = (*CAMERAS, *PROPRIO)  # draw order = Chronos's horizons dict order (cameras, then proprio)
POSE_ROTATION = [3, 4, 5, 6, 7, 8]
PROMPT = "catch the ball"
# Older-sample interval in FRAMES per stream, from the paper's real-world rate ranges (Table 6).
RGB_INTERVAL = (FPS / 33.33, FPS / 12.5)  # 0.9-2.4 frames
POSE_INTERVAL = (FPS / 33.33, FPS / 8.33)  # 0.9-3.6 frames
INTERVALS = {**{c: RGB_INTERVAL for c in CAMERAS}, **{p: POSE_INTERVAL for p in PROPRIO}}


def rot6d(quat_wxyz):
    rotation = Rotation.from_quat(np.asarray(quat_wxyz)[:, [1, 2, 3, 0]])
    return rotation.as_matrix()[:, :2].reshape(-1, 6).astype(np.float32)


def pose20(eef16):
    """[T, 16] (per arm x y z qw qx qy qz gripper) -> [T, 20] (per arm xyz, rot6d, gripper)."""
    eef16 = np.asarray(eef16, dtype=np.float64)
    arms = []
    for arm in range(2):
        a = eef16[:, 8 * arm : 8 * arm + 8]
        arms.append(np.concatenate([a[:, :3], rot6d(a[:, 3:7]), a[:, 7:8]], 1))
    return np.concatenate(arms, 1).astype(np.float32)


def window_timing(frames, rng, streams=STREAMS, period=PERIOD, horizon=HORIZON, query_phase=QUERY_PHASE,
                  action_offset=ACTION_OFFSET, interval_scale=INTERVAL_SCALE, sensor_delay=SENSOR_DELAY_FRAMES,
                  inference_delays=INFERENCE_DELAYS, history=2, intervals=None):
    """Index arithmetic of chronos_policy `simulation/export.py::windows` (same RNG draws, same order), with the
    constants as arguments. `intervals` (stream -> (lo, hi) frames) replaces interval_scale * period.
    Returns per-window frame indices, validity, delays and action targets."""
    reserve = max(inference_delays) if rng is not None else 0
    last = frames - 1 - action_offset - period * (horizon - 1 + reserve)
    queries = np.arange(query_phase, last + 1, period)
    count = len(queries)
    if count == 0:
        return None
    delays = rng.choice(inference_delays, count) if rng is not None else np.zeros(count, int)
    out = {"queries": queries, "delays": delays}
    for name in streams:
        if rng is None:
            ages = period * np.arange(history - 1, -1, -1)[None].repeat(count, 0)
        else:
            if intervals is None:
                steps = rng.uniform(*interval_scale, (count, history - 1)) * period
            else:
                steps = rng.uniform(*intervals[name], (count, history - 1))
            gaps = np.round(np.cumsum(steps[:, ::-1], 1)[:, ::-1]).astype(int)
            delay = rng.integers(0, sensor_delay + 1, (count, 1))
            ages = np.concatenate([gaps, np.zeros((count, 1), int)], 1) + delay
        index = queries[:, None] - ages
        out[f"{name}/valid"] = index >= 0
        out[f"{name}/index"] = index.clip(0)
    out["targets"] = queries[:, None] + action_offset + period * (delays[:, None] + np.arange(horizon))
    out["action_steps"] = delays[:, None] + np.arange(horizon)
    return out


def split(episode_ids):
    """Chronos's rule when a dataset has no split: hold out every tenth episode."""
    ids = sorted(episode_ids)
    return [e for i, e in enumerate(ids) if i % 10], [e for i, e in enumerate(ids) if not i % 10]


def build_windows(episodes, randomized, seed=0):
    """episodes: list of dicts with `id`, `offset` (global row of frame 0), `timestamp` [T].

    Returns flat arrays over all windows; frame indices are GLOBAL rows (episode offset + frame)."""
    cols = {}
    for position, ep in enumerate(episodes):
        rng = np.random.default_rng([seed, position]) if randomized else None
        t = window_timing(len(ep["timestamp"]), rng, intervals=INTERVALS)
        if t is None:
            continue
        q, stamps, off = t["queries"], ep["timestamp"], ep["offset"]
        row = {"episode": np.full(len(q), ep["id"]), "query": off + q, "delays": t["delays"],
               "targets": off + t["targets"], "action_times": t["action_steps"] * ACTION_PERIOD_S}
        for name in STREAMS:
            index = t[f"{name}/index"]
            row[f"{name}/index"] = off + index
            row[f"{name}/valid"] = t[f"{name}/valid"]
            row[f"{name}/times"] = stamps[index] - stamps[q][:, None]
        for k, v in row.items():
            cols.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    for k in out:
        if k.endswith("/times") or k == "action_times":
            out[k] = out[k].astype(np.float32)
    return out
