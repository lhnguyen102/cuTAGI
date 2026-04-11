import os
import sys

from matplotlib.tri import TriContourSet

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric, Utils
from pytagi.nn import (
    Embedding,
    LayerNorm,
    Linear,
    MultiheadAttention,
    MultiheadAttentionV2,
    OutputUpdater,
    PositionalEncoding,
    ReLU,
    RMSNorm,
    Sequential,
)

np.random.seed(42)

torch.manual_seed(42)


class ReverseDataset:
    """Generates random sequences and their reversed versions."""

    def __init__(self, vocab_size: int = 10, seq_len: int = 16):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def next_batch(self, batch_size: int):
        x = np.random.randint(self.vocab_size, size=(batch_size, self.seq_len))
        x = x.reshape(batch_size, self.seq_len, 1).astype(np.float32)
        y = np.flip(x, axis=1)
        return x, y.reshape(-1).astype(np.int32)

    def next_batch_onehot(self, batch_size: int):
        x_idx = np.random.randint(
            self.vocab_size, size=(batch_size, self.seq_len)
        )
        x_onehot = np.zeros(
            (batch_size, self.seq_len, self.vocab_size), dtype=np.float32
        )
        for i in range(batch_size):
            for j in range(self.seq_len):
                x_onehot[i, j, x_idx[i, j]] = 1.0
        y = np.flip(x_idx, axis=1).copy()
        return x_onehot, y.reshape(-1).astype(np.int32)


def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx]
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx] for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(
        num_layers,
        num_heads,
        figsize=(num_heads * fig_size, num_layers * fig_size),
    )
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(
                attn_maps[row][column], origin="lower", vmin=0
            )
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title(f"Layer {row + 1}, Head {column + 1}")
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def main(
    num_epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = 5,
    vocab_size: int = 8,
    embed_dim: int = 64,
    num_heads: int = 1,
    sigma_v: float = 3.5,
    sigma_v_min: float = 0.3,
    decay_factor: float = 1.0,
    steps_per_epoch: int = 100,
    no_attn: bool = False,
):
    """Train a TAGI attention model on the sequence reversal task."""
    task = ReverseDataset(vocab_size=vocab_size, seq_len=seq_len)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=vocab_size)

    hrc = utils.get_hierarchical_softmax(vocab_size)
    hrc_class_len = hrc.len

    if no_attn:
        net = Sequential(
            Embedding(vocab_size, embed_dim, input_size=seq_len, scale=1.0),
            Linear(embed_dim, hrc_class_len),
        )
    else:
        net = Sequential(
            Embedding(vocab_size, embed_dim, input_size=seq_len, scale=0.25),
            PositionalEncoding(embed_dim),
            MultiheadAttentionV2(
                embed_dim=embed_dim,
                num_heads=num_heads,
                seq_len=seq_len,
                bias=False,
                gain_weight=0.25,
                gain_bias=0.5,
                init_method="He",
                pos_emb="",
                use_causal_mask=False,
            ),
            RMSNorm([embed_dim]),
            Linear(embed_dim, hrc_class_len),
        )

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
        for _ in range(steps_per_epoch):

            x, labels = task.next_batch(batch_size)
            m_pred, v_pred = net(x)
            attention_scores = net.get_attention_scores()

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
            error_rate = metric.error_rate(m_pred, v_pred, labels)
            error_rates.append(error_rate)

        avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
        current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | error: {avg_error * 100:.2f}% | sigma_v: {current_sigma_v:.3f}"
        )

    test_batch_size = 256
    x_test, y_test = task.next_batch(test_batch_size)
    net.eval()
    m_pred, v_pred = net(x_test)
    predicted = metric.get_predicted_labels(m_pred, v_pred)

    x_test = x_test.reshape(test_batch_size, seq_len, -1)
    x_display = x_test.squeeze(-1).astype(int)
    y_test = y_test.reshape(test_batch_size, seq_len)
    predicted = predicted.reshape(test_batch_size, seq_len)

    num_show = min(5, test_batch_size)
    print(f"\nTest Results (showing {num_show} of {test_batch_size}):")
    for i in range(num_show):
        print(f"  Input:      {x_display[i].tolist()}")
        print(f"  Target:     {y_test[i].tolist()}")
        print(f"  Prediction: {predicted[i].tolist()}")
        print()

    accuracy = np.mean(predicted == y_test)
    print(f"Test accuracy: {accuracy:.2%}")

    if not no_attn:
        attention_scores = net.get_attention_scores()
        mu_scores = [mu for mu, var in attention_scores.values()]
        var_scores = [var for mu, var in attention_scores.values()]
        print(f"\nAttention scores (sample 0, head 0):")
        print(f"  mu:\n{mu_scores[0][0, 0]}")
        print(f"  var:\n{var_scores[0][0, 0]}")
        plot_attention_maps(x_display, mu_scores)


if __name__ == "__main__":
    fire.Fire(main)
