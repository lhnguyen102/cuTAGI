"""Diagnose CPU vs CUDA training divergence by comparing per-layer weights
after each step on the SAME batches.

Usage:
    python -m examples.cpu_vs_cuda_diag --steps 20 --batch_size 16 --seq_len 128

The script builds two identical attention models (CPU and CUDA), copies the
CPU's weights into the CUDA model so they start from the same point, then for
each step:
  1. Pulls the same batch from a deterministic dataset.
  2. Runs forward + update on both.
  3. Pulls the CUDA params back to host and prints, per layer:
       max|cpu.mu_w - cuda.mu_w|,  max|cpu.var_w - cuda.var_w|

The first layer whose diff jumps by ~2 orders of magnitude is where the bug
(or the dominant fp drift) originates.
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import (
    Embedding,
    Linear,
    MultiheadAttention,
    OutputUpdater,
    PositionalEncoding,
    RMSNorm,
    Sequential,
)


def build_model(
    vocab_size,
    embed_dim,
    num_heads,
    seq_len,
    hrc_class_len,
    use_causal_mask,
    num_layers,
):
    layers = [
        Embedding(vocab_size, embed_dim, input_size=seq_len, scale=0.15),
        PositionalEncoding(embed_dim),
    ]
    for _ in range(num_layers):
        layers.extend(
            [
                MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    seq_len=seq_len,
                    bias=False,
                    gain_weight=0.25,
                    gain_bias=0.5,
                    init_method="He",
                    pos_emb="",
                    use_causal_mask=use_causal_mask,
                ),
                RMSNorm([embed_dim]),
                Linear(embed_dim, embed_dim),
            ]
        )
    layers.append(Linear(embed_dim, hrc_class_len))
    return Sequential(*layers)


def deterministic_batches(steps, vocab_size, batch_size, seq_len, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(steps):
        x = rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(
            np.float32
        )
        x = x.reshape(batch_size, seq_len, 1)
        labels = rng.integers(
            0, vocab_size, size=(batch_size * seq_len,)
        ).astype(np.int32)
        batches.append((x, labels))
    return batches


def diff_layer(cpu_layer, cuda_layer):
    if not cpu_layer.mu_w:
        return None
    cpu_mu = np.asarray(cpu_layer.mu_w, dtype=np.float64)
    cpu_var = np.asarray(cpu_layer.var_w, dtype=np.float64)
    cuda_mu = np.asarray(cuda_layer.mu_w, dtype=np.float64)
    cuda_var = np.asarray(cuda_layer.var_w, dtype=np.float64)
    return {
        "n": cpu_mu.size,
        "max_mu": float(np.max(np.abs(cpu_mu - cuda_mu))),
        "max_var": float(np.max(np.abs(cpu_var - cuda_var))),
    }


def diff_layer_deltas(cpu_layer, cuda_layer):
    if not cpu_layer.delta_mu_w:
        return None
    cpu_dmu = np.asarray(cpu_layer.delta_mu_w, dtype=np.float64)
    cpu_dvar = np.asarray(cpu_layer.delta_var_w, dtype=np.float64)
    cuda_dmu = np.asarray(cuda_layer.delta_mu_w, dtype=np.float64)
    cuda_dvar = np.asarray(cuda_layer.delta_var_w, dtype=np.float64)
    return {
        "n": cpu_dmu.size,
        "max_dmu": float(np.max(np.abs(cpu_dmu - cuda_dmu))),
        "max_dvar": float(np.max(np.abs(cpu_dvar - cuda_dvar))),
    }


def cuda_delta_params_to_host(cuda_net):
    """Sync per-layer device delta_w/delta_b to host for inspection."""
    for layer in cuda_net.layers:
        # Each pytagi CUDA layer wraps a C++ BaseLayerCuda; the binding
        # exposes delta_params_to_host where it makes sense.
        if hasattr(layer, "delta_params_to_host"):
            layer.delta_params_to_host()


def sync_weights(src_cpu_net, dst_cpu_net):
    """Copy weights between two CPU-side Sequentials. Layer names match because
    both are CPU; this avoids the Embedding vs EmbeddingCuda key mismatch."""
    dst_cpu_net.load_state_dict(src_cpu_net.state_dict())


def main(
    steps: int = 4,
    batch_size: int = 16,
    seq_len: int = 28,
    vocab_size: int = 65,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 1,
    use_causal_mask: bool = True,
    sigma_v: float = 4.5,
    seed: int = 42,
):
    pytagi.manual_seed(seed)
    np.random.seed(seed)

    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=vocab_size)
    hrc = utils.get_hierarchical_softmax(vocab_size)

    cpu_net = build_model(
        vocab_size,
        embed_dim,
        num_heads,
        seq_len,
        hrc.len,
        use_causal_mask,
        num_layers,
    )
    cuda_net = build_model(
        vocab_size,
        embed_dim,
        num_heads,
        seq_len,
        hrc.len,
        use_causal_mask,
        num_layers,
    )

    # Both still on CPU here so layer names match -> state_dict copy works.
    sync_weights(cpu_net, cuda_net)
    cuda_net.to_device("cuda")

    cpu_updater = OutputUpdater(cpu_net.device)
    cuda_updater = OutputUpdater(cuda_net.device)

    var_y = np.full(
        (batch_size * seq_len * hrc.num_obs,),
        sigma_v**2,
        dtype=np.float32,
    )

    batches = deterministic_batches(
        steps, vocab_size, batch_size, seq_len, seed=seed
    )

    print(
        f"\nRunning {steps} steps with use_causal_mask={use_causal_mask} "
        f"(B={batch_size}, T={seq_len}, D={embed_dim}, H={num_heads}, "
        f"layers={num_layers})\n"
    )

    for step, (x, labels) in enumerate(batches):
        for net, updater in ((cpu_net, cpu_updater), (cuda_net, cuda_updater)):
            net.train()
            m_pred, v_pred = net(x)
            print(f"m_pred: {m_pred[:10]}")
            y_obs, y_idx, _ = utils.label_to_obs(
                labels=labels, num_classes=vocab_size
            )
            updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y_obs,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()


if __name__ == "__main__":
    fire.Fire(main)
