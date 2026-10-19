"""Time one examples/fineweb_gpt2.py train step per phase and its metric paths.
Run from the repository root: python -m benchmarks.bench_fineweb_step
"""

import ctypes
import inspect
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(REPO_DIR, "build"))

import fire
import numpy as np

import pytagi
from examples.fineweb_gpt2 import (
    VOCAB_SIZE,
    build_mingpt,
    fit_level_calibration,
    get_batch,
    main,
    sampled_metrics,
)
from pytagi import Utils
from pytagi.nn import OutputUpdater

cudart = ctypes.CDLL("libcudart.so")
DEFAULTS = {k: p.default for k, p in inspect.signature(main).parameters.items()}


class PhaseTimer:
    """Wall-clock ms between consecutive marks, each after a device sync."""

    def __init__(self):
        self.samples = {}

    def start(self):
        cudart.cudaDeviceSynchronize()
        self._begin = self._last = time.perf_counter()

    def mark(self, phase):
        cudart.cudaDeviceSynchronize()
        now = time.perf_counter()
        self.samples.setdefault(phase, []).append((now - self._last) * 1e3)
        self._last = now

    def stop(self, name):
        self.samples.setdefault(name, []).append(
            (self._last - self._begin) * 1e3
        )

    def medians(self):
        return {k: float(np.median(v)) for k, v in self.samples.items()}


def time_train_step(
    net, utils, hrc, batch_size, seq_len, dense, warmup, steps, seed
):
    out_updater = OutputUpdater(net.device)
    var_y = np.full(
        (batch_size * seq_len * hrc.num_obs,),
        DEFAULTS["sigma_v"] ** 2,
        dtype=np.float32,
    )
    timer = PhaseTimer()
    for it in range(warmup + steps):
        if it == warmup:
            timer = PhaseTimer()
        np.random.seed(seed)  # same train batch every iteration
        timer.start()
        x, labels = get_batch("train", batch_size, seq_len)
        timer.mark("get_batch")
        y_obs, y_idx, _ = utils.label_to_obs(
            labels=labels, num_classes=VOCAB_SIZE
        )
        y_obs = np.asarray(y_obs, dtype=np.float32) * DEFAULTS["obs_scale"]
        timer.mark("label_to_obs")
        if not dense:
            net.set_output_indices(y_idx, hrc.num_obs)
        net._cpp_backend.forward(x, None)
        timer.mark("forward")
        net.get_outputs()
        timer.mark("get_outputs")
        if dense:
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y_obs,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
        else:
            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y_obs,
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )
        timer.mark("update")
        net.backward()
        timer.mark("backward")
        net.step()
        timer.mark("step")
        timer.stop("total")
    return timer.medians()


def time_metric_paths(net, utils, hrc, batch_size, seq_len, reps):
    obs_scale = DEFAULTS["obs_scale"]
    calib_args = (net, utils, hrc, batch_size, seq_len, obs_scale)
    calib = fit_level_calibration(*calib_args, DEFAULTS["calib_batches"])
    timer = PhaseTimer()
    net.eval()
    for _ in range(reps):
        x, labels = get_batch("val", batch_size, seq_len)
        timer.start()
        net._cpp_backend.forward(x, None)
        timer.mark("eval_forward")
        m_pred, v_pred = net.get_outputs()
        timer.mark("eval_get_outputs")
        sampled_metrics(
            utils,
            hrc,
            m_pred,
            v_pred,
            labels,
            obs_scale,
            DEFAULTS["metric_samples"],
            calib,
            device=net.device,
        )
        timer.mark("eval_sampled_metrics")
        timer.stop("eval_batch")
    net.train()
    for _ in range(reps):
        timer.start()
        fit_level_calibration(*calib_args, DEFAULTS["calib_batches"])
        timer.mark("calibration")
    medians = timer.medians()
    medians["amortized_per_iter"] = (
        medians["eval_sampled_metrics"] / DEFAULTS["log_interval"]
        + medians["calibration"] / DEFAULTS["calib_interval"]
        + medians["eval_batch"]
        * DEFAULTS["eval_iters"]
        / DEFAULTS["eval_interval"]
    )
    return medians


def mean_touched_rows(utils, batch_size, seq_len, batches):
    counts = []
    for _ in range(batches):
        _, labels = get_batch("train", batch_size, seq_len)
        _, y_idx, _ = utils.label_to_obs(labels=labels, num_classes=VOCAB_SIZE)
        counts.append(np.unique(y_idx).size)
    return float(np.mean(counts))


def bench(
    num_layers: int = DEFAULTS["num_layers"],
    dense: bool = False,
    warmup: int = 20,
    steps: int = 100,
    metric_reps: int = 5,
    batch_size: int = DEFAULTS["batch_size"],
    seq_len: int = DEFAULTS["seq_len"],
    seed: int = DEFAULTS["seed"],
):
    np.random.seed(seed)
    pytagi.manual_seed(seed)
    utils = Utils()
    hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)
    net = build_mingpt(
        vocab_size=VOCAB_SIZE,
        seq_len=seq_len,
        embed_dim=DEFAULTS["embed_dim"],
        num_heads=DEFAULTS["num_heads"],
        num_layers=num_layers,
        ffn_hidden=DEFAULTS["ffn_hidden"],
        output_size=hrc.len,
        qkv_gain=DEFAULTS["qkv_gain"],
        gain_w_rms=DEFAULTS["gain_w_rms"],
        embed_scale=DEFAULTS["embed_scale"],
        center_score_delta=DEFAULTS["center_score_delta"],
    )
    net.to_device("cuda")
    net.set_cap_factor(DEFAULTS["cap_factor"])
    net.set_var_decay(
        DEFAULTS["var_decay_tau"], min_gain=DEFAULTS["var_decay_min_gain"]
    )

    step_ms = time_train_step(
        net, utils, hrc, batch_size, seq_len, dense, warmup, steps, seed
    )
    metrics_ms = time_metric_paths(
        net, utils, hrc, batch_size, seq_len, metric_reps
    )
    touched = mean_touched_rows(utils, batch_size, seq_len, batches=100)

    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR, text=True
    ).strip()
    mode = "dense" if dense else "sparse"
    result = {
        "sha": sha,
        "L": num_layers,
        "B": batch_size,
        "T": seq_len,
        "mode": mode,
        "step_ms": {k: v for k, v in step_ms.items() if k != "total"},
        "total_ms": step_ms["total"],
        "metrics_ms": metrics_ms,
        "touched_rows_mean": touched,
    }
    results_dir = os.path.join(REPO_DIR, "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{sha}_L{num_layers}_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{sha} L={num_layers} B={batch_size} T={seq_len} {mode}")
    for name, ms in step_ms.items():
        print(f"  {name:22s} {ms:8.2f} ms")
    print("metric paths")
    for name, ms in metrics_ms.items():
        print(f"  {name:22s} {ms:8.2f} ms")
    print(f"touched readout rows   {touched:8.1f}")
    print(out_path)


if __name__ == "__main__":
    fire.Fire(bench)
