import os
import sys

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
    LayerNorm,
    Linear,
    MixtureReLU,
    MultiheadAttention,
    MultiheadAttentionV2,
    OutputUpdater,
    PositionalEncoding,
    ReLU,
    ResNetBlock,
    RMSNorm,
    Sequential,
)

np.random.seed(42)
pytagi.manual_seed(42)

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
    debug: bool = False,
    debug_interval: int = 200,
) -> Sequential:

    layers = [
        Embedding(vocab_size, embed_dim, input_size=seq_len, scale=0.15),
    ]
    gain_w_rms = 5.0
    for li in range(num_layers):
        # Only print diagnostics from the first transformer block to keep
        # the log readable.
        first = li == 0
        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm(
                        [embed_dim],
                        debug=debug and first,
                        debug_interval=debug_interval,
                        gain_w=gain_w_rms,
                    ),
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        seq_len=seq_len,
                        bias=False,
                        gain_weight=0.25,
                        gain_bias=1.0,
                        init_method="He",
                        pos_emb="rope",
                        debug=debug and first,
                        debug_interval=debug_interval,
                        use_causal_mask=True,
                    ),
                )
            )
        )
        layers.append(
            ResNetBlock(
                LayerBlock(
                    RMSNorm(
                        [embed_dim],
                        debug=debug and first,
                        debug_interval=debug_interval,
                        gain_w=gain_w_rms,
                    ),
                    Linear(embed_dim, ffn_hidden, bias=False),
                    ReLU(),
                    Linear(ffn_hidden, embed_dim, bias=False),
                )
            )
        )
    layers.append(
        RMSNorm(
            [embed_dim],
            debug=debug,
            debug_interval=debug_interval,
            gain_w=gain_w_rms,
        )
    )
    layers.append(Linear(embed_dim, output_size))

    return Sequential(*layers)


def main(
    num_epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 64,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 1,
    ffn_hidden: int = 512,
    steps_per_epoch: int = 200,
    sigma_v: float = 10.0,
    sigma_v_min: float = 0.3,
    decay_factor: float = 0.995,
    max_new_tokens: int = 200,
    gen_sigma_v: float = 0.3,
    network: str = "mingpt",
    debug: bool = False,
    debug_interval: int = 200,
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
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        seq_len=seq_len,
                        bias=False,
                        gain_weight=0.25,
                        gain_bias=1.0,
                        init_method="He",
                        pos_emb="rope",
                        debug=False,
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
    current_sigma_v = sigma_v

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        var_y = np.full(
            (batch_size * seq_len * hrc.num_obs,),
            current_sigma_v**2,
            dtype=np.float32,
        )
        net.train()
        error_rates = []
        losses = []
        num_samples = batch_size * seq_len
        for _ in range(steps_per_epoch):
            x, labels = dataset.next_batch(batch_size)
            m_pred, v_pred = net(x)

            y_obs, y_idx, _ = utils.label_to_obs(
                labels=labels, num_classes=vocab_size
            )
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y_obs,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()

            er, prob = utils.get_errors(
                m_pred, v_pred, labels, vocab_size, num_samples
            )
            error_rates.append(float(np.mean(er)))
            losses.append(cross_entropy(prob, labels, vocab_size))

        avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
        current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | ce_loss: {avg_loss:.4f} | error: {avg_error * 100:.2f}% | sigma_v: {current_sigma_v:.3f}"
        )

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
        v_last = v_last + gen_sigma_v**2
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

    visualize_attention(net, dataset, prompt, seq_len)


if __name__ == "__main__":
    fire.Fire(main)
