"""bicat.window_timing with Chronos's constants must reproduce chronos_policy simulation/export.py::windows exactly
(same RNG draws in the same order), so the 30 Hz BiCAT windows follow the paper's recipe.

    envs/chronos/bin/python -m pytest pi05_chronos/tests/test_bicat_windows.py -q
"""
import numpy as np
import pytest

from simulation import export

from pi05_chronos import bicat


def fake_episode(frames, rng):
    ep = {"timestamp": (np.arange(frames) + 1) * 0.01}
    for name in ("s0", "s1", "s2"):
        ep[name] = rng.normal(size=(frames, 4)).astype(np.float32)
    ep["action0"] = rng.normal(size=(frames, 3)).astype(np.float32)
    return ep


@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("frames", [200, 431])
def test_matches_chronos_export(randomized, frames):
    ep = fake_episode(frames, np.random.default_rng(1))
    spec = dict(cameras=(), proprio={"s0": 4, "s1": 4, "s2": 4}, actions=("action0",),
                horizons={"s0": 2, "s1": 2, "s2": 2}, periods={"s0": 5, "s1": 5, "s2": 5},
                query_phase=4, action_offset=5)
    seed = [0, 7]
    ref = export.windows(ep, spec, None, "cpu", np.random.default_rng(seed) if randomized else None)
    mine = bicat.window_timing(frames, np.random.default_rng(seed) if randomized else None,
                               streams=("s0", "s1", "s2"), period=5, query_phase=4, action_offset=5,
                               sensor_delay=5, inference_delays=(0, 1, 2, 3, 4))
    q = mine["queries"]
    for name in ("s0", "s1", "s2"):
        np.testing.assert_array_equal(ref["observations"][name].numpy(), ep[name][mine[f"{name}/index"]])
        np.testing.assert_array_equal(ref["valid"][name].numpy(), mine[f"{name}/valid"])
        np.testing.assert_allclose(ref["capture_times"][name].numpy(),
                                   ep["timestamp"][mine[f"{name}/index"]] - ep["timestamp"][q][:, None])
    np.testing.assert_array_equal(ref["actions"].numpy(), ep["action0"][mine["targets"]])
    np.testing.assert_allclose(ref["action_times"].numpy(), 0.05 * mine["action_steps"])


def test_bicat_constants():
    t = bicat.window_timing(430, None)
    assert t["queries"][:3].tolist() == [0, 2, 4]
    assert t["top/index"][5].tolist() == [8, 10]  # nominal: previous sample one control step (2 frames) back
    assert t["targets"][0, :3].tolist() == [0, 2, 4]
    r = bicat.window_timing(430, np.random.default_rng([0, 0]), intervals=bicat.INTERVALS)
    assert set(np.unique(r["delays"])) <= {0, 1, 2, 3}
    for name, hi in (("top", 2), ("robot0_10d", 4)):  # RGB 12.5-33.33 Hz, pose 8.33-33.33 Hz at 30 fps
        ok = r[f"{name}/valid"].all(1)  # at the episode start the older sample is clamped to frame 0 (invalid)
        gaps = (r[f"{name}/index"][:, 1] - r[f"{name}/index"][:, 0])[ok]
        assert gaps.min() == 1 and gaps.max() == hi
    ages = r["query"] if "query" in r else r["queries"]
    delay = ages - r["top/index"][:, 1]
    assert delay.min() == 0 and delay.max() == 3  # 0-100 ms delivery delay


def test_pose20_identity():
    eef = np.array([[0.1, 0.2, 0.3, 1, 0, 0, 0, 0.5, -0.1, 0, 0.2, 1, 0, 0, 0, 0.9]], np.float32)
    p = bicat.pose20(eef)
    assert p.shape == (1, 20)
    np.testing.assert_allclose(p[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(p[0, [0, 1, 2, 9, 10, 19]], [0.1, 0.2, 0.3, 0.5, -0.1, 0.9], atol=1e-6)
