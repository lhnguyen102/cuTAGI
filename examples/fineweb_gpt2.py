import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
import tiktoken
from tqdm import tqdm

import pytagi
from pytagi import Utils
from pytagi.nn import (
    Embedding,
    LayerBlock,
    Linear,
    MultiheadAttention,
    OutputUpdater,
    ReLU,
    ResNetBlock,
    RMSNorm,
    Sequential,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fineweb")
VOCAB_SIZE = 50257  # gpt2 BPE
ALPHA = 3.0  # probit scale in obs_to_class (src/cost.cpp)


def get_batch(split: str, batch_size: int, seq_len: int):
    # recreate memmap every batch to avoid leaking page cache (see nanoGPT)
    data = np.memmap(
        os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r"
    )
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i : i + seq_len] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + seq_len] for i in ix])
    x = x.reshape(batch_size, seq_len, 1).astype(np.float32)
    y = y.reshape(-1).astype(np.int32)
    return x, y


def sampled_metrics(
    utils,
    hrc,
    m_pred,
    v_pred,
    labels,
    obs_scale,
    num_samples,
    calib=None,
):
    n = len(labels)
    m = np.asarray(m_pred).reshape(n, hrc.len) / obs_scale
    v = np.asarray(v_pred).reshape(n, hrc.len) / (obs_scale**2)
    idx = np.random.choice(n, size=min(num_samples, n), replace=False)
    ce, ce_raw, correct, prob_sum = 0.0, 0.0, 0, 0.0
    scale, offset = calib if calib is not None else (None, None)
    for i in idx:
        raw = np.asarray(
            utils.obs_to_label_prob(m[i], v[i], hrc, VOCAB_SIZE)
        ).reshape(-1)
        ce_raw -= np.log(max(raw[labels[i]] / raw.sum(), 1e-9))
        probs = raw
        if calib is not None:
            probs = np.asarray(
                utils.obs_to_label_prob(
                    m[i], v[i], hrc, VOCAB_SIZE, scale, offset
                )
            ).reshape(-1)
        prob_sum += float(probs.sum())
        probs = probs / probs.sum()
        ce -= np.log(max(probs[labels[i]], 1e-9))
        correct += int(np.argmax(probs) == labels[i])
    k = len(idx)
    return ce / k, 1.0 - correct / k, prob_sum / k, ce_raw / k


def _phi(x):
    """Standard normal CDF, vectorized (Abramowitz-Stegun 7.1.26)."""
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x) / np.sqrt(2.0))
    poly = t * (
        0.254829592
        + t
        * (
            -0.284496736
            + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))
        )
    )
    erf_abs = 1.0 - poly * np.exp(-(x**2) / 2.0)
    return 0.5 * (1.0 + np.sign(x) * erf_abs)


def fit_level_calibration(
    net, utils, hrc, batch_size, seq_len, obs_scale, n_batches
):
    """Fit a per-level scale and offset of the probit argument on train data.

    The metric reads every tree level through one alpha, but the levels have
    very different marginals, so one link cannot fit all of them. Scaling the
    probit argument per level is equivalent to a per-level alpha; the offset
    covers the shallowest levels, where the marginal is skewed enough that no
    scale suffices. Drawn with the training RNG state saved and restored so
    the run stays step-comparable.
    """
    rng_state = np.random.get_state()
    net.eval()
    zs, obs_all = [], []
    for _ in range(n_batches):
        x, labels = get_batch("train", batch_size, seq_len)
        m_pred, v_pred = net(x)
        n = len(labels)
        m = np.asarray(m_pred).reshape(n, hrc.len) / obs_scale
        v = np.asarray(v_pred).reshape(n, hrc.len) / (obs_scale**2)
        obs, idx, n_obs = utils.label_to_obs(
            labels=labels.astype(np.int32), num_classes=VOCAB_SIZE
        )
        obs = np.asarray(obs, dtype=np.float32).reshape(n, n_obs)
        idx = np.asarray(idx, dtype=np.int64).reshape(n, n_obs) - 1
        rows = np.arange(n)[:, None]
        zs.append(m[rows, idx] / np.sqrt(1.0 / ALPHA**2 + v[rows, idx]))
        obs_all.append(obs)
    net.train()
    np.random.set_state(rng_state)

    z = np.concatenate(zs)
    obs = np.concatenate(obs_all)
    n_obs = z.shape[1]
    scale = np.ones(n_obs, dtype=np.float32)
    offset = np.zeros(n_obs, dtype=np.float32)
    for l in range(n_obs):
        zl, ol = z[:, l], obs[:, l]
        s_lo, s_hi, b_lo, b_hi = 0.2, 2.0, -2.0, 2.0
        for _ in range(2):
            grid_s = np.linspace(s_lo, s_hi, 19)
            grid_b = np.linspace(b_lo, b_hi, 25)
            p = _phi(
                grid_s[:, None, None] * zl[None, None, :]
                + grid_b[None, :, None]
            )
            p = np.where(ol[None, None, :] == 1.0, p, 1.0 - p)
            ce = -np.log(np.clip(p, 1e-12, None)).mean(axis=2)
            i, j = np.unravel_index(int(np.argmin(ce)), ce.shape)
            ds = (s_hi - s_lo) / 18.0
            db = (b_hi - b_lo) / 24.0
            s_lo, s_hi = grid_s[i] - ds, grid_s[i] + ds
            b_lo, b_hi = grid_b[j] - db, grid_b[j] + db
            scale[l], offset[l] = grid_s[i], grid_b[j]
    return scale, offset


def max_causal_var_qk(net) -> float:
    """Largest pre-Remax score variance over the positions the causal mask
    keeps, across all attention layers. A runaway value signals that training
    is about to diverge."""
    qk_max = 0.0
    for scores in net.get_attention_scores().values():
        causal = np.tril(np.ones((scores.timestep, scores.timestep), bool))
        qk_max = max(qk_max, float(scores.var_qk[..., causal].max()))
    return qk_max


def build_mingpt(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    ffn_hidden: int,
    output_size: int,
    qkv_gain: float = 1.0,
    gain_w_rms: float = 1.0,
    embed_scale: float = 0.15,
    center_score_delta: bool = True,
) -> Sequential:

    layers = [
        Embedding(vocab_size, embed_dim, input_size=seq_len, scale=embed_scale),
    ]
    for _ in range(num_layers):
        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm([embed_dim], gain_w=gain_w_rms),
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        seq_len=seq_len,
                        bias=False,
                        gain_weight=qkv_gain,
                        init_method="He",
                        use_rope=True,
                        use_causal_mask=True,
                        center_score_delta=center_score_delta,
                    ),
                    Linear(embed_dim, embed_dim, bias=False),
                )
            )
        )
        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm([embed_dim], gain_w=gain_w_rms),
                    Linear(embed_dim, ffn_hidden, bias=False),
                    ReLU(),
                    Linear(ffn_hidden, ffn_hidden, bias=False),
                    ReLU(),
                    Linear(ffn_hidden, embed_dim, bias=False),
                )
            )
        )
    layers.append(RMSNorm([embed_dim], gain_w=gain_w_rms))
    layers.append(Linear(embed_dim, output_size))

    return Sequential(*layers)


def main(
    seq_len: int = 64,
    embed_dim: int = 768,
    num_heads: int = 8,
    num_layers: int = 4,
    ffn_hidden: int = 1024,
    max_iters: int = 100_000,
    batch_size: int = 16,
    sigma_v: float = 4.0,
    sigma_v_min: float = 2.0,
    decay_factor: float = 0.95,
    decay_interval: int = 200,
    obs_scale: float = 3.0,
    qkv_gain: float = 1.0,
    gain_w_rms: float = 1.0,
    embed_scale: float = 0.15,
    embed_var_scale: float = 1.0,
    cap_factor: float = 50.0,
    var_decay_tau: float = 1500.0,
    var_decay_min_gain: float = 0.2,
    center_score_delta: bool = True,
    calibrate_readout: bool = True,
    calib_batches: int = 8,
    calib_interval: int = 500,
    ckpt_interval: int = 0,
    eval_interval: int = 2000,
    eval_iters: int = 50,
    log_interval: int = 50,
    metric_samples: int = 64,
    out_dir: str = "out/fineweb",
    init_from: str = "scratch",
    eval_only: bool = False,
    max_new_tokens: int = 200,
    seed: int = 44,
):
    """Train a TAGI GPT-2 on FineWeb (BPE token level)."""
    np.random.seed(seed)
    pytagi.manual_seed(seed)
    train_path = os.path.join(DATA_DIR, "train.bin")
    if not os.path.exists(train_path):
        sys.exit(f"{train_path} not found. Run: python -m data.fineweb.prepare")

    enc = tiktoken.get_encoding("gpt2")
    utils = Utils()
    hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)
    tokens_per_iter = batch_size * seq_len
    print(f"tokens per iteration: {tokens_per_iter:,}")

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
        center_score_delta=center_score_delta,
    )
    if embed_var_scale != 1.0:
        sd = net.state_dict()
        mu_w, var_w, mu_b, var_b = sd["Embedding.0"]
        var_w = (np.asarray(var_w) * embed_var_scale).tolist()
        sd["Embedding.0"] = (mu_w, var_w, mu_b, var_b)
        net.load_state_dict(sd)
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    net.set_cap_factor(cap_factor)
    net.set_var_decay(var_decay_tau, min_gain=var_decay_min_gain)

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.bin")
    if init_from == "resume":
        print(f"resuming from {ckpt_path}")
        net.load(ckpt_path)

    calib = None
    out_updater = OutputUpdater(net.device)
    current_sigma_v = sigma_v
    var_y = np.full(
        (batch_size * seq_len * hrc.num_obs,),
        current_sigma_v**2,
        dtype=np.float32,
    )

    def estimate_loss():
        rng_state = np.random.get_state()
        net.eval()
        ces, errs, psums, raws = [], [], [], []
        for _ in range(eval_iters):
            x, labels = get_batch("val", batch_size, seq_len)
            m_pred, v_pred = net(x)
            ce, err, psum, ce_raw = sampled_metrics(
                utils,
                hrc,
                m_pred,
                v_pred,
                labels,
                obs_scale,
                metric_samples,
                calib,
            )
            ces.append(ce)
            errs.append(err)
            psums.append(psum)
            raws.append(ce_raw)
        net.train()
        np.random.set_state(rng_state)
        return (
            float(np.mean(ces)),
            float(np.mean(errs)),
            float(np.mean(psums)),
            float(np.mean(raws)),
        )

    best_val_loss = float("inf")
    losses = []
    error_rates = []
    prob_sums = []
    raw_losses = []
    pbar = tqdm(range(max_iters + 1), desc="Training")
    refit_every = calib_interval if calib_interval > 0 else eval_interval
    for iter_num in pbar:
        if calibrate_readout and iter_num % refit_every == 0:
            calib = fit_level_calibration(
                net, utils, hrc, batch_size, seq_len, obs_scale, calib_batches
            )
            np.savez(
                os.path.join(out_dir, "calibration.npz"),
                scale=calib[0],
                offset=calib[1],
                step=iter_num,
            )
        if iter_num % eval_interval == 0:
            val_ce, val_err, val_psum, val_raw = estimate_loss()
            tqdm.write(
                f"step {iter_num}: val ce {val_ce:.4f} (raw {val_raw:.4f}), "
                f"val error {val_err * 100:.2f}%, val psum {val_psum:.3f}"
            )
            if val_ce < best_val_loss and iter_num > 0:
                best_val_loss = val_ce
                tqdm.write(f"saving checkpoint to {ckpt_path}")
                net.save(ckpt_path)
            if eval_only:
                break

        x, labels = get_batch("train", batch_size, seq_len)
        y_obs, y_idx, _ = utils.label_to_obs(
            labels=labels, num_classes=VOCAB_SIZE
        )
        if obs_scale != 1.0:
            y_obs = np.asarray(y_obs, dtype=np.float32) * obs_scale
        # Metric iterations need the full vocabulary; every other step only
        # computes and updates the path units (bit-equal, see ADR 0001).
        metric_iteration = iter_num % log_interval == 0
        if not metric_iteration:
            net.set_output_indices(y_idx, hrc.num_obs)
        m_pred, v_pred = net(x)
        if metric_iteration:
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
        net.backward()
        net.step()

        if ckpt_interval > 0 and iter_num > 0 and iter_num % ckpt_interval == 0:
            step_ckpt = os.path.join(out_dir, f"ckpt_{iter_num}.bin")
            net.save(step_ckpt)
            tqdm.write(f"saved {step_ckpt}")

        if iter_num > 0 and iter_num % decay_interval == 0:
            current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
            var_y[:] = current_sigma_v**2

        if metric_iteration:
            ce, err, psum, ce_raw = sampled_metrics(
                utils,
                hrc,
                m_pred,
                v_pred,
                labels,
                obs_scale,
                metric_samples,
                calib,
            )
            losses.append(ce)
            error_rates.append(err)
            prob_sums.append(psum)
            raw_losses.append(ce_raw)
            avg_loss = sum(losses[-100:]) / min(len(losses), 100)
            avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
            avg_psum = sum(prob_sums[-100:]) / min(len(prob_sums), 100)
            avg_raw = sum(raw_losses[-100:]) / min(len(raw_losses), 100)
            pbar.set_description(
                f"Iter {iter_num}/{max_iters} | ce_loss: {avg_loss:.4f} | "
                f"ce_raw: {avg_raw:.4f} | error: {avg_error * 100:.2f}% | "
                f"psum: {avg_psum:.3f} | sigma_v: {current_sigma_v:.3f} | "
                f"QKmax: {max_causal_var_qk(net):.3g}"
            )

    # Generation sample
    net.eval()
    val_data = np.memmap(
        os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r"
    )
    prompt_ids = val_data[:seq_len].astype(np.int64).tolist()
    generated = list(prompt_ids)
    for _ in range(max_new_tokens):
        context = generated[-seq_len:]
        x = np.array(context, dtype=np.float32).reshape(1, seq_len, 1)
        m_pred, v_pred = net(x)
        m_last = np.asarray(m_pred).reshape(seq_len, hrc.len)[-1] / obs_scale
        v_last = np.asarray(v_pred).reshape(seq_len, hrc.len)[-1] / (
            obs_scale**2
        )
        probs = np.asarray(
            utils.obs_to_label_prob(
                m_last,
                v_last,
                hrc,
                VOCAB_SIZE,
                None if calib is None else calib[0],
                None if calib is None else calib[1],
            )
        ).reshape(-1)
        probs = probs / probs.sum()
        next_token = int(np.random.choice(VOCAB_SIZE, p=probs))
        generated.append(next_token)

    prompt_text = enc.decode(prompt_ids)
    new_text = enc.decode(generated[len(prompt_ids) :])
    print(f"\n--- prompt ---\n{prompt_text}")
    print(f"--- continuation ---\n\033[31m{new_text}\033[0m")


if __name__ == "__main__":
    fire.Fire(main)
