import csv
import os
import sys
from datetime import datetime

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import (
    Embedding,
    LayerBlock,
    Linear,
    MultiheadAttention,
    MultiheadAttentionV2,
    OutputUpdater,
    ReLU,
    ResNetBlock,
    RMSNorm,
    Sequential,
)

try:
    from scipy.special import ndtr as _scipy_ndtr

    def norm_cdf(x):
        return _scipy_ndtr(x)

except ImportError:

    def norm_cdf(x):
        # Abramowitz & Stegun 7.1.26, fully vectorized via numpy.
        x = np.asarray(x, dtype=np.float64)
        sign = np.sign(x)
        a1, a2, a3, a4, a5, p = (
            0.254829592,
            -0.284496736,
            1.421413741,
            -1.453152027,
            1.061405429,
            0.3275911,
        )
        abs_x = np.abs(x) / np.sqrt(2.0)
        t = 1.0 / (1.0 + p * abs_x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(
            -abs_x * abs_x
        )
        return 0.5 * (1.0 + sign * y)


np.random.seed(1384)
pytagi.manual_seed(1384)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "shakespeare", "input.txt"
)


def cross_entropy(
    prob: np.ndarray, labels: np.ndarray, num_classes: int
) -> float:
    prob = np.asarray(prob).reshape(len(labels), num_classes)
    prob = prob / prob.sum(axis=1, keepdims=True)
    p_true = prob[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, 1e-9, 1.0)).mean())


def arch_rows_from_state_dict(net, prev_mu_w: dict) -> tuple:
    """Snapshot per-layer weight stats. Requires params_to_host() upstream
    so the host-side state_dict reflects current GPU values.

    Returns (rows, new_prev_mu_w) where rows is a list of per-layer dicts and
    new_prev_mu_w maps layer-name -> mu_w array (for next-epoch delta).
    """
    sd = net.state_dict()
    rows = []
    new_prev = {}
    for name, params in sd.items():
        mu_w, var_w, _mu_b, _var_b = params
        mu_w = np.asarray(mu_w, dtype=np.float32)
        var_w = np.asarray(var_w, dtype=np.float32)
        if mu_w.size == 0:
            continue
        row = {
            "layer": name,
            "n_w": int(mu_w.size),
            "mu_w_abs": float(np.abs(mu_w).mean()),
            "mu_w_max": float(np.abs(mu_w).max()),
            "var_w_mean": float(var_w.mean()),
            "var_w_med": float(np.median(var_w)),
            "var_w_min": float(var_w.min()),
        }
        if name in prev_mu_w:
            d = mu_w - prev_mu_w[name]
            row["dmu_w_abs"] = float(np.abs(d).mean())
            row["dmu_w_max"] = float(np.abs(d).max())
        else:
            row["dmu_w_abs"] = 0.0
            row["dmu_w_max"] = 0.0
        rows.append(row)
        new_prev[name] = mu_w
    return rows, new_prev


def attention_diag(net) -> dict:
    """Aggregate attention scores: entropy and top-1 per (layer, head)."""
    try:
        scores = net.get_attention_scores()
    except Exception:
        return {}
    out = {}
    for li, (mu, _) in enumerate(scores.values()):
        mu = np.asarray(mu)  # (batch, heads, seq, seq)
        p = np.clip(mu, 1e-12, 1.0)
        entropy = -(p * np.log(p)).sum(axis=-1)  # (batch, heads, seq)
        top1 = mu.max(axis=-1)  # (batch, heads, seq)
        out[f"attn_L{li}_entropy_mean"] = float(entropy.mean())
        out[f"attn_L{li}_entropy_med"] = float(np.median(entropy))
        out[f"attn_L{li}_top1_mean"] = float(top1.mean())
        out[f"attn_L{li}_top1_max"] = float(top1.max())
    return out


def build_dense_hrc_obs(
    labels: np.ndarray,
    hrc,
    obs_scale: float,
):
    """Dense HRC observation means: every bit in the binary tree gets a target.

    Path bits (the log2(V) bits encoding the true class) get `+/-obs_scale`;
    off-path bits get 0. Variance is uniform (sigma_v**2) for every bit and
    is built once per epoch by the caller.

    Returns flat `mu_obs` of length `len(labels) * hrc.len` in the row-major
    layout expected by `OutputUpdater.update`.
    """
    n = len(labels)
    L = hrc.len
    n_obs = hrc.num_obs

    hrc_obs = np.asarray(hrc.obs, dtype=np.float32).reshape(-1, n_obs)
    hrc_idx = np.asarray(hrc.idx, dtype=np.int64).reshape(-1, n_obs)

    mu_obs = np.zeros((n, L), dtype=np.float32)
    rows = np.arange(n)[:, None]
    mu_obs[rows, hrc_idx[labels] - 1] = hrc_obs[labels] * obs_scale
    return mu_obs.ravel()


def hrc_diag_stats(
    m_pred_m: np.ndarray,
    v_pred_m: np.ndarray,
    prob: np.ndarray,
    num_samples: int,
    vocab_size: int,
    alpha: float,
) -> dict:
    """Per-step diagnostics on the obs_scale-divided model output.

    `m_pred_m`, `v_pred_m`: what is actually passed to HRC decode in C++.
    `prob`: HRC-decoded P[r] per (sample, class), un-renormalized.
    `alpha`: the same hardcoded value used inside cost.cpp::obs_to_class.
    """
    mz = np.asarray(m_pred_m, dtype=np.float32).ravel()
    Sz = np.asarray(v_pred_m, dtype=np.float32).ravel()
    q = norm_cdf(np.abs(mz) / np.sqrt(1.0 / alpha**2 + Sz))
    prob_2d = np.asarray(prob, dtype=np.float64).reshape(
        num_samples, vocab_size
    )
    p_sum = prob_2d.sum(axis=1)
    return {
        "mz_abs": float(np.abs(mz).mean()),
        "mz_std": float(mz.std()),
        "Sz_mean": float(Sz.mean()),
        "Sz_med": float(np.median(Sz)),
        "q_mean": float(q.mean()),
        "q_p10": float(np.percentile(q, 10)),
        "p_sum_mean": float(p_sum.mean()),
        "p_sum_std": float(p_sum.std()),
    }


def visualize_attention(net, dataset, prompt, seq_len, last_k=32, top_k=10):
    """Show which prior characters the model attends to when guessing the
    next character after `prompt`: (1) bar chart per layer/head of the
    last-query row, (2) top-K characters heatmap across layers/heads."""
    prompt_ids = [dataset.stoi[c] for c in prompt][-seq_len:]
    pad_len = seq_len - len(prompt_ids)
    x = np.array([0] * pad_len + prompt_ids, dtype=np.float32).reshape(
        1, seq_len, 1
    )
    net(x)
    scores = net.get_attention_scores()

    def readable(c):
        return "↵" if c == "\n" else c

    full_labels = ["·"] * pad_len + [
        readable(dataset.itos[i]) for i in prompt_ids
    ]
    start = max(0, seq_len - last_k)
    labels = full_labels[start:]
    positions = np.arange(len(labels))

    num_layers = len(scores)
    num_heads = next(iter(scores.values()))[0].shape[1]

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(num_heads * 5, num_layers * 2.5),
        squeeze=False,
    )
    for li, (mu, _) in enumerate(scores.values()):
        for hi in range(num_heads):
            last_row = mu[0, hi, -1, start:]
            ax = axes[li][hi]
            ax.bar(positions, last_row)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"Layer {li + 1}, Head {hi + 1}")
            ax.set_ylabel("attention")
    fig.suptitle(
        f"Which characters does the model look at to guess after: "
        f"{prompt!r}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    rows, row_labels, cell_chars = [], [], []
    for li, (mu, _) in enumerate(scores.values()):
        for hi in range(num_heads):
            last_row = mu[0, hi, -1, :]
            top_idx = np.argsort(-last_row)[:top_k]
            rows.append(last_row[top_idx])
            cell_chars.append([full_labels[i] for i in top_idx])
            row_labels.append(f"L{li + 1}H{hi + 1}")
    mat = np.array(rows)
    fig, ax = plt.subplots(figsize=(top_k * 0.9, len(rows) * 0.5 + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0)
    vmax = mat.max() if mat.size else 1.0
    for i in range(len(rows)):
        for j in range(top_k):
            color = "white" if mat[i, j] < 0.5 * vmax else "black"
            ax.text(
                j,
                i,
                cell_chars[i][j],
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"#{i + 1}" for i in range(top_k)])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("rank (most → least attended)")
    fig.colorbar(im, ax=ax, label="attention")
    ax.set_title(
        f"Top-{top_k} characters attended to when predicting next token"
    )
    plt.tight_layout()
    plt.show()


class CharDataset:
    def __init__(self, text, seq_len):
        # text = text.replace("$", "")
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        self.seq_len = seq_len

    def next_batch(self, batch_size):
        max_start = len(self.data) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([self.data[s : s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts])
        x = x.reshape(batch_size, self.seq_len, 1).astype(np.float32)
        y = y.reshape(-1).astype(np.int32)
        return x, y


def build_mingpt(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    ffn_hidden: int,
    output_size: int,
    qkv_gain: float = 0.25,
    gain_w_rms: float = 0.1,
    embed_scale: float = 0.15,
    post_embed_gain_w: float = 0.0,
    debug: bool = False,
    debug_interval: int = 200,
) -> Sequential:
    """TAGI pre-norm transformer for next-token prediction.

    Architecture differs from standard pre-norm in two coupled ways that
    together substitute for the implicit gradient-driven regularization a
    backprop transformer would have on the embedding:

      1. A RMSNorm is placed AFTER the embedding (outside any ResNet block)
         to bound what enters the residual stream. `post_embed_gain_w=0`
         freezes its gain at 1.0 so it acts as pure normalization without
         a learnable scale that could drift over training.

      2. The FIRST attention block has NO pre-attn RMSNorm inside its
         ResNet (its input is already RMS-1 from the post-embed norm).
         Every subsequent sub-layer (FFN1, L2 attn, FFN2, ...) keeps the
         standard pre-block RMSNorm because its input is the un-normalized
         residual sum.

    Flow for num_layers=2:
        x_norm = RMSNorm_postembed(x_emb)
        y1 = x_norm + MHA(x_norm)                # L1: no pre-attn RMSN
        y2 = y1     + FFN(RMSNorm(y1))
        y3 = y2     + MHA(RMSNorm(y2))           # L2: with pre-attn RMSN
        y4 = y3     + FFN(RMSNorm(y3))
        out = Linear(RMSNorm_final(y4))
    """
    layers = [
        Embedding(vocab_size, embed_dim, input_size=seq_len, scale=embed_scale),
        RMSNorm(
            [embed_dim],
            gain_w=post_embed_gain_w,
            debug=debug,
            debug_interval=debug_interval,
        ),
    ]
    for i in range(num_layers):
        # First attention block omits the pre-attn RMSNorm because its
        # input (post-embed normalized stream) is already RMS-1.
        attn_block_layers = []
        if i != 0:
            attn_block_layers.append(
                RMSNorm(
                    [embed_dim],
                    gain_w=gain_w_rms,
                    debug=debug,
                    debug_interval=debug_interval,
                )
            )
        attn_block_layers.append(
            MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                seq_len=seq_len,
                bias=False,
                gain_weight=qkv_gain,
                gain_bias=1.0,
                init_method="He",
                pos_emb="rope",
                use_causal_mask=True,
                debug=debug,
                debug_interval=debug_interval,
            )
        )
        layers.append(ResNetBlock(LayerBlock(*attn_block_layers)))
        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm(
                        [embed_dim],
                        gain_w=gain_w_rms,
                        debug=debug,
                        debug_interval=debug_interval,
                    ),
                    Linear(
                        embed_dim,
                        ffn_hidden,
                        bias=False,
                        debug=debug,
                        debug_interval=debug_interval,
                    ),
                    ReLU(),
                    Linear(
                        ffn_hidden,
                        embed_dim,
                        bias=False,
                        debug=debug,
                        debug_interval=debug_interval,
                    ),
                )
            )
        )
    layers.append(
        RMSNorm(
            [embed_dim],
            gain_w=gain_w_rms,
            debug=debug,
            debug_interval=debug_interval,
        )
    )
    layers.append(
        Linear(
            embed_dim,
            output_size,
            bias=False,
            debug=debug,
            debug_interval=debug_interval,
        )
    )

    return Sequential(*layers)


def main(
    num_epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 64,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 3,
    ffn_hidden: int = 512,
    steps_per_epoch: int = 200,
    sigma_v: float = 8.0,
    sigma_v_min: float = 2.0,
    decay_factor: float = 0.97,
    max_new_tokens: int = 200,
    gen_sigma_v: float = 0.3,
    network: str = "mingpt",
    qkv_gain: float = 0.25,
    gain_w_rms: float = 1.0,
    embed_scale: float = 0.15,
    obs_scale: float = 1.0,
    post_embed_gain_w: float = 0.0,
    debug: bool = False,
    debug_interval: int = 200,
    alpha: float = 3.0,
    log_dir: str = "./diag_logs",
    dense_update: bool = True,
):
    """Train a character-level next-token predictor (TAGI) on Shakespeare text."""
    text = open(DATA_PATH, "r").read()
    dataset = CharDataset(text, seq_len)
    vocab_size = dataset.vocab_size
    print(f"Data: {len(text)} chars, {vocab_size} unique")

    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=vocab_size)
    hrc = utils.get_hierarchical_softmax(vocab_size)

    if network == "mingpt":
        net = build_mingpt(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_hidden=ffn_hidden,
            output_size=hrc.len,
            qkv_gain=qkv_gain,
            gain_w_rms=gain_w_rms,
            embed_scale=embed_scale,
            post_embed_gain_w=post_embed_gain_w,
            debug=debug,
            debug_interval=debug_interval,
        )
    else:
        layers = [
            Embedding(vocab_size, embed_dim, input_size=seq_len, scale=0.15),
        ]
        for _ in range(num_layers):
            layers.extend(
                [
                    RMSNorm([embed_dim]),
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        seq_len=seq_len,
                        bias=False,
                        gain_weight=0.25,
                        gain_bias=1.0,
                        init_method="He",
                        pos_emb="rope",
                        use_causal_mask=True,
                    ),
                    RMSNorm([embed_dim]),
                    Linear(embed_dim, ffn_hidden),
                    ReLU(),
                    Linear(ffn_hidden, embed_dim),
                    ReLU(),
                    RMSNorm([embed_dim]),
                ]
            )
        layers.append(Linear(embed_dim, hrc.len))
        net = Sequential(*layers)
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")

    out_updater = OutputUpdater(net.device)
    out_updater._cpp_backend.debug = debug
    out_updater._cpp_backend.debug_interval = debug_interval
    current_sigma_v = sigma_v

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"diag_{ts}.csv")
    arch_log_path = os.path.join(log_dir, f"arch_{ts}.csv")
    attn_log_path = os.path.join(log_dir, f"attn_{ts}.csv")

    diag_fields = [
        "epoch",
        "sigma_v",
        "train_ce",
        "train_err",
        "mz_abs",
        "mz_std",
        "Sz_mean",
        "Sz_med",
        "q_mean",
        "q_p10",
        "p_sum_mean",
        "p_sum_std",
    ]
    log_file = open(log_path, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=diag_fields)
    log_writer.writeheader()

    arch_fields = [
        "epoch",
        "layer",
        "n_w",
        "mu_w_abs",
        "mu_w_max",
        "var_w_mean",
        "var_w_med",
        "var_w_min",
        "dmu_w_abs",
        "dmu_w_max",
    ]
    arch_file = open(arch_log_path, "w", newline="")
    arch_writer = csv.DictWriter(arch_file, fieldnames=arch_fields)
    arch_writer.writeheader()

    attn_file = open(attn_log_path, "w", newline="")
    attn_writer = None  # header written on first row

    print(f"Diagnostics CSV: {log_path}")
    print(f"Arch CSV:        {arch_log_path}")
    print(f"Attn CSV:        {attn_log_path}")
    print(
        f"HRC update mode: {'DENSE (all bits)' if dense_update else 'SPARSE (path only)'}"
    )

    prev_mu_w = {}  # epoch-over-epoch delta state for arch_rows_from_state_dict

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        n_per_token = hrc.len if dense_update else hrc.num_obs
        var_y = np.full(
            (batch_size * seq_len * n_per_token,),
            current_sigma_v**2,
            dtype=np.float32,
        )
        net.train()
        error_rates = []
        losses = []
        diag_rows = []  # per-step dicts from hrc_diag_stats
        num_samples = batch_size * seq_len
        for _ in range(steps_per_epoch):
            x, labels = dataset.next_batch(batch_size)
            m_pred, v_pred = net(x)

            if dense_update:
                mu_obs = build_dense_hrc_obs(labels, hrc, obs_scale)
                out_updater.update(
                    output_states=net.output_z_buffer,
                    mu_obs=mu_obs,
                    var_obs=var_y,
                    delta_states=net.input_delta_z_buffer,
                )
            else:
                y_obs, y_idx, _ = utils.label_to_obs(
                    labels=labels, num_classes=vocab_size
                )
                if obs_scale != 1.0:
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

            if obs_scale != 1.0:
                m_pred_m = (np.asarray(m_pred) / obs_scale).tolist()
                v_pred_m = (np.asarray(v_pred) / (obs_scale**2)).tolist()
            else:
                m_pred_m, v_pred_m = m_pred, v_pred
            er, prob = utils.get_errors(
                m_pred_m, v_pred_m, labels, vocab_size, num_samples
            )

            error_rates.append(float(np.mean(er)))
            losses.append(cross_entropy(prob, labels, vocab_size))
            diag_rows.append(
                hrc_diag_stats(
                    m_pred_m, v_pred_m, prob, num_samples, vocab_size, alpha
                )
            )

        avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
        current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)

        last = diag_rows[-100:]
        agg = {k: float(np.mean([r[k] for r in last])) for k in last[0].keys()}
        log_writer.writerow(
            {
                "epoch": epoch + 1,
                "sigma_v": current_sigma_v,
                "train_ce": avg_loss,
                "train_err": avg_error,
                **agg,
            }
        )
        log_file.flush()

        # Per-layer architectural snapshot at end of epoch
        net.params_to_host()
        arch_rows, prev_mu_w = arch_rows_from_state_dict(net, prev_mu_w)
        for row in arch_rows:
            row["epoch"] = epoch + 1
            arch_writer.writerow(row)
        arch_file.flush()

        # Attention diagnostic — last batch's scores are still in memory
        attn_row = attention_diag(net)
        if attn_row:
            attn_row["epoch"] = epoch + 1
            if attn_writer is None:
                attn_fields = ["epoch"] + sorted(
                    k for k in attn_row if k != "epoch"
                )
                attn_writer = csv.DictWriter(attn_file, fieldnames=attn_fields)
                attn_writer.writeheader()
            attn_writer.writerow(attn_row)
            attn_file.flush()

        pbar.set_description(
            f"E{epoch + 1}/{num_epochs} | ce={avg_loss:.3f} err={avg_error * 100:.1f}% "
            f"sv={current_sigma_v:.2f} | mz|={agg['mz_abs']:.3f} Sz={agg['Sz_mean']:.3f} "
            f"q={agg['q_mean']:.3f} p_sum={agg['p_sum_mean']:.3f}"
        )

    log_file.close()
    arch_file.close()
    attn_file.close()

    # Generation
    net.eval()
    prompt = text[:seq_len]
    assert len(prompt) == seq_len
    prompt_ids = [dataset.stoi[c] for c in prompt]
    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = generated[-seq_len:]
        x = np.array(context, dtype=np.float32).reshape(1, seq_len, 1)
        m_pred, v_pred = net(x)
        m_last = np.asarray(m_pred).reshape(seq_len, hrc.len)[-1]
        v_last = np.asarray(v_pred).reshape(seq_len, hrc.len)[-1]
        v_last = v_last
        probs = np.asarray(
            utils.obs_to_label_prob(m_last, v_last, hrc, vocab_size)
        ).reshape(-1)
        probs = probs / probs.sum()
        next_token = int(np.random.choice(vocab_size, p=probs))
        generated.append(next_token)

    new_tokens = generated[len(prompt_ids) :]
    new_text = "".join([dataset.itos[i] for i in new_tokens])
    print(f"\n--- prompt ---\n{prompt}")
    print(f"--- continuation ---\n\033[31m{new_text}\033[0m")
    print(f"--- full ---\n{prompt}\033[31m{new_text}\033[0m")

    # visualize_attention(net, dataset, prompt, seq_len)


if __name__ == "__main__":
    fire.Fire(main)
