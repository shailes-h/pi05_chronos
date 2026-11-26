"""Our Conveyor window sampler == chronos_policy simulation/export.py::windows (randomized and nominal).

Run in the chronos env:  python -I tests/test_export_parity.py <zarr> <subset json>
"""
import importlib.util, json, sys
import numpy as np, torch, zarr

W = "/scratch/Projects/CFP-05/CFP05-CF-002/chronos_pi05"
spec = importlib.util.spec_from_file_location("cc", f"{W}/openpi/src/openpi/training/chronos_conveyor.py")
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
import simulation.export as ex

# Stub SigLIP: "features" = the first 12 raw pixel values of each frame.
ex.encode_images = lambda encoder, frames, device: torch.from_numpy(
    np.asarray(frames).reshape(len(frames), -1)[:, :12].astype(np.float32))[:, None]

zarr_path, subset = sys.argv[1], sys.argv[2]
store = zarr.open(zarr_path, "r")
for split, timing in (("train", "randomized"), ("val", "nominal")):
    n = 6
    ep = json.load(open(subset))[split][:n]
    tmp = f"{W}/.cache/tmp/subset_{split}.json"
    json.dump({split: ep}, open(tmp, "w"))
    ours = cc.ConveyorWindows(zarr_path, tmp, split, timing, seed=0)
    r = 0
    for pos, e in enumerate(ep):
        rng = np.random.default_rng([0, pos]) if timing == "randomized" else None
        ref = ex.windows(store[f"episode_{e}"], ex.TASKS["conveyor"], None, "cpu", rng)
        for w in range(len(ref["actions"])):
            item = ours[r]; r += 1
            np.testing.assert_allclose(item["actions"], ref["actions"][w].numpy())
            np.testing.assert_allclose(item["request_action_times"], ref["action_times"][w].numpy(), atol=1e-6)
            np.testing.assert_allclose(item["cam_times"], ref["capture_times"][cc.CAMERA][w].numpy(), atol=1e-6)
            np.testing.assert_allclose(item["proprio_times"], ref["capture_times"][cc.PROPRIO][w].numpy(), atol=1e-6)
            assert (item["image_valid"] == ref["valid"][cc.CAMERA][w].numpy()).all()
            proprio = ref["observations"][cc.PROPRIO][w].numpy()
            np.testing.assert_allclose(item["state"], np.concatenate([proprio[-1], proprio[0]]))
            feats = ref["observations"][cc.CAMERA][w].float().numpy()[:, 0]
            np.testing.assert_allclose(item["image"].reshape(-1)[:12], feats[-1])
            np.testing.assert_allclose(item["image_prev"].reshape(-1)[:12], feats[0])
    assert r == len(ours), (r, len(ours))
    at = np.stack([ours[i]["request_action_times"][0] for i in range(len(ours))])
    ct = np.stack([ours[i]["cam_times"] for i in range(len(ours))])
    print(f"{split}/{timing}: {r} windows match export.windows; D values {sorted(set(np.round(at/0.05).astype(int)))}, "
          f"newest-cam age ms [{-ct[:,1].max()*1e3:.0f},{-ct[:,1].min()*1e3:.0f}], frame gap ms [{(ct[:,1]-ct[:,0]).min()*1e3:.0f},{(ct[:,1]-ct[:,0]).max()*1e3:.0f}]")
print("PARITY OK")
