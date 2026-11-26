"""π0.5-chronos time-RoPE (timing tokens off) == stock π0.5 at nominal timing; differs under delay.

Loads pi05_base weights into both configs, feeds one identical batch with identical rng/noise.
Run on a GPU node in the openpi venv:  python tests/test_equivalence.py
"""
import dataclasses
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config

PARAMS = pathlib.Path(os.environ["OPENPI_DATA_HOME"]) / "openpi-assets/checkpoints/pi05_base/params"


def make_batch(model_cfg, b=4, seed=0):
    rng = np.random.default_rng(seed)
    obs_spec, act_spec = model_cfg.inputs_spec(batch_size=b)
    images = {k: jnp.asarray(rng.uniform(-1, 1, v.shape), jnp.float32) for k, v in obs_spec.images.items()}
    masks = {"base_0_rgb": jnp.ones(b, bool), "left_wrist_0_rgb": jnp.ones(b, bool), "right_wrist_0_rgb": jnp.zeros(b, bool)}
    L = model_cfg.max_token_len
    lengths = rng.integers(40, 90, b)  # variable prompt lengths -> per-sample prefix lengths differ
    tokens = jnp.asarray(rng.integers(1, 1000, (b, L)), jnp.int32)
    tmask = jnp.asarray(np.arange(L)[None] < lengths[:, None])
    state = jnp.asarray(rng.uniform(-1, 1, (b, model_cfg.action_dim)), jnp.float32)
    actions = jnp.asarray(rng.uniform(-1, 1, act_spec.shape), jnp.float32)
    return images, masks, tokens, tmask, state, actions


def obs(images, masks, tokens, tmask, state, action_times=None):
    return _model.Observation(images=images, image_masks=masks, state=state, tokenized_prompt=tokens,
                              tokenized_prompt_mask=tmask, action_times=action_times)


def main():
    base = _config.get_config("pi05_conveyor_orig").model
    chronos = dataclasses.replace(base, time_rope_actions=True, timing_tokens=False)
    params = _model.restore_params(PARAMS, dtype=jnp.bfloat16)
    m_orig, m_chr = base.load(params), chronos.load(params)
    images, masks, tokens, tmask, state, actions = make_batch(base)
    b, H = actions.shape[0], base.action_horizon
    nominal = jnp.broadcast_to(0.05 * jnp.arange(H, dtype=jnp.float32), (b, H))
    delayed = nominal + jnp.asarray([0.0, 0.05, 0.1, 0.2])[:, None]  # sample 0 still nominal
    rng = jax.random.key(0)

    loss_o = m_orig.compute_loss(rng, obs(images, masks, tokens, tmask, state), actions)
    loss_n = m_chr.compute_loss(rng, obs(images, masks, tokens, tmask, state, nominal), actions)
    loss_d = m_chr.compute_loss(rng, obs(images, masks, tokens, tmask, state, delayed), actions)
    noise = jax.random.normal(jax.random.key(1), actions.shape)
    act_o = m_orig.sample_actions(rng, obs(images, masks, tokens, tmask, state), noise=noise)
    act_n = m_chr.sample_actions(rng, obs(images, masks, tokens, tmask, state, nominal), noise=noise)
    act_d = m_chr.sample_actions(rng, obs(images, masks, tokens, tmask, state, delayed), noise=noise)

    d_loss_nom = float(jnp.max(jnp.abs(loss_o - loss_n)))
    d_act_nom = float(jnp.max(jnp.abs(act_o - act_n)))
    d_act_del = np.asarray(jnp.max(jnp.abs(act_o - act_d), axis=(1, 2)))
    print(f"nominal: max|Δloss| = {d_loss_nom:.3e}, max|Δactions| = {d_act_nom:.3e}")
    print(f"delayed (0, 50, 100, 200 ms): per-sample max|Δactions| = {np.array2string(d_act_del, precision=4)}")
    print(f"loss orig {np.asarray(loss_o.mean()):.5f}  chronos-nominal {np.asarray(loss_n.mean()):.5f}  chronos-delayed {np.asarray(loss_d.mean()):.5f}")
    assert d_loss_nom < 1e-3 and d_act_nom < 1e-2, "nominal timing must reproduce stock π0.5"
    assert d_act_del[0] < 1e-2, "an undelayed sample must stay identical"
    assert (d_act_del[1:] > 1e-2).all(), "delayed samples must change the output"
    print("EQUIVALENCE OK")


if __name__ == "__main__":
    main()
