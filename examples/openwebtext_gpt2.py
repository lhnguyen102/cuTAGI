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

np.random.seed(44)
pytagi.manual_seed(44)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "openwebtext")
VOCAB_SIZE = 50257  # gpt2 BPE


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


def sampled_metrics(utils, hrc, m_pred, v_pred, labels, obs_scale, num_samples):
    """CE loss and error rate on a random subset of positions. Computing the
    full 50k-class probability for every position is too expensive."""
    n = len(labels)
    m = np.asarray(m_pred).reshape(n, hrc.len) / obs_scale
    v = np.asarray(v_pred).reshape(n, hrc.len) / (obs_scale**2)
    idx = np.random.choice(n, size=min(num_samples, n), replace=False)
    ce, correct = 0.0, 0
    for i in idx:
        probs = np.asarray(
            utils.obs_to_label_prob(m[i], v[i], hrc, VOCAB_SIZE)
        ).reshape(-1)
        probs = probs / probs.sum()
        ce -= np.log(max(probs[labels[i]], 1e-9))
        correct += int(np.argmax(probs) == labels[i])
    return ce / len(idx), 1.0 - correct / len(idx)


def build_mingpt(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    ffn_hidden: int,
    output_size: int,
    qkv_gain: float = 0.25,
    gain_w_rms: float = 5.0,
    embed_scale: float = 0.15,
    prior_pull: float = 0.0,
    debug: bool = False,
    debug_interval: int = 200,
) -> Sequential:

    layers = [
        Embedding(
            vocab_size,
            embed_dim,
            input_size=seq_len,
            scale=embed_scale,
            debug=debug,
            debug_interval=debug_interval,
        ),
    ]
    for _ in range(num_layers):
        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm(
                        [embed_dim],
                        gain_w=gain_w_rms,
                        prior_pull=prior_pull,
                        debug=debug,
                        debug_interval=debug_interval,
                    ),
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
                        prior_pull=prior_pull,
                        debug=debug,
                        debug_interval=debug_interval,
                    ),
                    Linear(embed_dim, embed_dim, bias=True),
                )
            )
        )

        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm(
                        [embed_dim],
                        gain_w=gain_w_rms,
                        prior_pull=prior_pull,
                        debug=debug,
                        debug_interval=debug_interval,
                    ),
                    Linear(
                        embed_dim,
                        ffn_hidden,
                        debug=debug,
                        debug_interval=debug_interval,
                        bias=False,
                    ),
                    ReLU(),
                    Linear(
                        ffn_hidden,
                        ffn_hidden,
                        debug=debug,
                        debug_interval=debug_interval,
                        bias=False,
                    ),
                    ReLU(),
                    Linear(
                        ffn_hidden,
                        embed_dim,
                        debug=debug,
                        debug_interval=debug_interval,
                        bias=False,
                    ),
                )
            )
        )
    layers.append(
        RMSNorm(
            [embed_dim],
            gain_w=gain_w_rms,
            prior_pull=prior_pull,
            debug=debug,
            debug_interval=debug_interval,
        )
    )
    head = Linear(
        embed_dim, output_size, debug=debug, debug_interval=debug_interval
    )
    layers.append(head)

    return Sequential(*layers)


def main(
    seq_len: int = 128,
    embed_dim: int = 384,
    num_heads: int = 8,
    num_layers: int = 1,
    ffn_hidden: int = 1024,
    max_iters: int = 100_000,
    batch_size: int = 8,
    sigma_v: float = 10.0,
    sigma_v_min: float = 4.0,
    decay_factor: float = 0.95,
    decay_interval: int = 200,
    obs_scale: float = 2.0,
    qkv_gain: float = 0.25,
    gain_w_rms: float = 5.0,
    embed_scale: float = 0.15,
    prior_pull: float = 0.0005,
    eval_interval: int = 2000,
    eval_iters: int = 50,
    log_interval: int = 50,
    metric_samples: int = 64,
    out_dir: str = "out/openwebtext",
    init_from: str = "scratch",
    eval_only: bool = False,
    max_new_tokens: int = 200,
    debug: bool = False,
    debug_interval: int = 50,
):
    """Train a TAGI GPT-2 on OpenWebText (BPE token level)."""
    train_path = os.path.join(DATA_DIR, "train.bin")
    if not os.path.exists(train_path):
        sys.exit(
            f"{train_path} not found. Run: python -m data.openwebtext.prepare"
        )

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
        prior_pull=prior_pull,
        debug=debug,
        debug_interval=debug_interval,
    )
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.bin")
    if init_from == "resume":
        print(f"resuming from {ckpt_path}")
        net.load(ckpt_path)

    out_updater = OutputUpdater(net.device)
    current_sigma_v = sigma_v
    var_y = np.full(
        (batch_size * seq_len * hrc.num_obs,),
        current_sigma_v**2,
        dtype=np.float32,
    )

    def estimate_loss():
        net.eval()
        ces, errs = [], []
        for _ in range(eval_iters):
            x, labels = get_batch("val", batch_size, seq_len)
            m_pred, v_pred = net(x)
            ce, err = sampled_metrics(
                utils, hrc, m_pred, v_pred, labels, obs_scale, metric_samples
            )
            ces.append(ce)
            errs.append(err)
        net.train()
        return float(np.mean(ces)), float(np.mean(errs))

    best_val_loss = float("inf")
    losses = []
    error_rates = []
    pbar = tqdm(range(max_iters + 1), desc="Training")
    for iter_num in pbar:
        if iter_num % eval_interval == 0:
            val_ce, val_err = estimate_loss()
            tqdm.write(
                f"step {iter_num}: val ce {val_ce:.4f}, "
                f"val error {val_err * 100:.2f}%"
            )
            if val_ce < best_val_loss and iter_num > 0:
                best_val_loss = val_ce
                tqdm.write(f"saving checkpoint to {ckpt_path}")
                net.save(ckpt_path)
            if eval_only:
                break

        x, labels = get_batch("train", batch_size, seq_len)
        m_pred, v_pred = net(x)
        y_obs, y_idx, _ = utils.label_to_obs(
            labels=labels, num_classes=VOCAB_SIZE
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

        if iter_num > 0 and iter_num % decay_interval == 0:
            current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
            var_y[:] = current_sigma_v**2

        if iter_num % log_interval == 0:
            ce, err = sampled_metrics(
                utils, hrc, m_pred, v_pred, labels, obs_scale, metric_samples
            )
            losses.append(ce)
            error_rates.append(err)
            avg_loss = sum(losses[-100:]) / min(len(losses), 100)
            avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
            pbar.set_description(
                f"Iter {iter_num}/{max_iters} | ce_loss: {avg_loss:.4f} | "
                f"error: {avg_error * 100:.2f}% | sigma_v: {current_sigma_v:.3f}"
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
            utils.obs_to_label_prob(m_last, v_last, hrc, VOCAB_SIZE)
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
