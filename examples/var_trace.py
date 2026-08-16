"""Trace posterior parameter variance per layer during FineWeb training.

Answers: does var_w contract as observations accumulate (TAGI's native
annealing), or is it frozen at the prior? Uses fineweb_gpt2 defaults.
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
from tqdm import tqdm

import pytagi
from examples.fineweb_gpt2 import (
    VOCAB_SIZE,
    build_mingpt,
    get_batch,
    sampled_metrics,
)
from pytagi import Utils
from pytagi.nn import OutputUpdater


def layer_labels(num_layers: int):
    labels = ["emb"]
    for i in range(num_layers):
        labels += [f"L{i}.attn.rms", f"L{i}.attn.qkv", f"L{i}.attn.proj"]
        labels += [
            f"L{i}.ffn.rms",
            f"L{i}.ffn.fc1",
            f"L{i}.ffn.fc2",
            f"L{i}.ffn.fc3",
        ]
    labels += ["final.rms", "head"]
    return labels


def snapshot(net):
    net.params_to_host()
    out = []
    for mu_w, var_w, _, _ in net.parameters():
        if len(mu_w) == 0:
            continue
        out.append(
            (
                np.asarray(mu_w, dtype=np.float64),
                np.asarray(var_w, dtype=np.float64),
            )
        )
    return out


def main(
    num_layers: int = 4,
    max_iters: int = 2000,
    trace_interval: int = 200,
    seq_len: int = 64,
    embed_dim: int = 768,
    num_heads: int = 8,
    ffn_hidden: int = 1024,
    batch_size: int = 16,
    sigma_v: float = 10.0,
    sigma_v_min: float = 4.0,
    decay_factor: float = 0.95,
    decay_interval: int = 200,
    obs_scale: float = 3.0,
    qkv_gain: float = 1.0,
    gain_w_rms: float = 1.0,
    embed_scale: float = 0.15,
    metric_samples: int = 64,
    var_decay_tau: float = 0.0,
):
    utils = Utils()
    hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)

    net = build_mingpt(
        vocab_size=VOCAB_SIZE,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_hidden=ffn_hidden,
        output_size=hrc.len,
        qkv_gain=qkv_gain,
        gain_w_rms=gain_w_rms,
        embed_scale=embed_scale,
        debug=False,
    )
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    if var_decay_tau > 0.0:
        net.set_var_decay(var_decay_tau)

    out_updater = OutputUpdater(net.device)
    current_sigma_v = sigma_v
    var_y = np.full(
        (batch_size * seq_len * hrc.num_obs,), sigma_v**2, dtype=np.float32
    )

    init = snapshot(net)
    labels = layer_labels(num_layers)
    if len(labels) != len(init):
        labels = [f"p{i}" for i in range(len(init))]
    print(f"[var-trace] {len(init)} parameter tensors")
    for name, (mu_w, var_w) in zip(labels, init):
        print(
            f"[var-trace]   {name:16s} n={mu_w.size:9d} "
            f"var_w0={var_w.mean():.6e} |mu_w0|={np.abs(mu_w).mean():.6e}"
        )

    prev = init
    ce_hist = []
    pbar = tqdm(range(max_iters + 1), desc="var-trace")
    for iter_num in pbar:
        x, labels_batch = get_batch("train", batch_size, seq_len)
        m_pred, v_pred = net(x)
        y_obs, y_idx, _ = utils.label_to_obs(
            labels=labels_batch, num_classes=VOCAB_SIZE
        )
        y_obs = np.asarray(y_obs, dtype=np.float32) * obs_scale
        out_updater.update_using_indices(
            output_states=net.output_z_buffer,
            mu_obs=y_obs,
            var_obs=var_y,
            selected_idx=y_idx,
            delta_states=net.input_delta_z_buffer,
        )
        net.backward()
        net.step()

        if iter_num > 0 and iter_num % decay_interval == 0:
            current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
            var_y[:] = current_sigma_v**2

        if iter_num % 50 == 0:
            ce, _, _ = sampled_metrics(
                utils,
                hrc,
                m_pred,
                v_pred,
                labels_batch,
                obs_scale,
                metric_samples,
            )
            ce_hist.append(ce)
            pbar.set_description(
                f"var-trace {iter_num} | ce {np.mean(ce_hist[-20:]):.4f}"
            )

        if iter_num % trace_interval == 0:
            cur = snapshot(net)
            print(
                f"\n[var-trace] step={iter_num} "
                f"ce={np.mean(ce_hist[-20:]):.4f} sigma_v={current_sigma_v:.3f}"
            )
            print(
                "[var-trace] "
                f"{'layer':16s} {'var/var0':>10s} {'varmin/var0':>12s} "
                f"{'|mu|/|mu0|':>11s} {'drift_sig':>10s} {'drift_p999':>11s}"
            )
            for name, (mu0, var0), (mu_p, _), (mu_c, var_c) in zip(
                labels, init, prev, cur
            ):
                sigma = np.sqrt(var_c)
                step_move = np.abs(mu_c - mu_p) / (
                    sigma * max(trace_interval, 1)
                )
                print(
                    f"[var-trace] {name:16s} "
                    f"{var_c.mean() / var0.mean():10.6f} "
                    f"{var_c.min() / var0.mean():12.6f} "
                    f"{np.abs(mu_c).mean() / np.abs(mu0).mean():11.6f} "
                    f"{step_move.mean():10.3e} "
                    f"{np.percentile(step_move, 99.9):11.3e}"
                )
            prev = cur


if __name__ == "__main__":
    fire.Fire(main)
