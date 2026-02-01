# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import string

import fire
import numpy as np
from tqdm import tqdm

import pytagi
from pytagi import HRCSoftmaxMetric, Utils, exponential_scheduler
from pytagi.nn import (
    Embedding,
    Linear,
    MultiheadAttention,
    OutputUpdater,
    RMSNorm,
    Sequential,
)


class CountingTask:
    """Generates random letter sequences and their per-character counts."""

    def __init__(self, max_len: int = 10, vocab_size: int = 3):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.letters = list(string.ascii_uppercase[: self.vocab_size])

    def next_batch(self, batch_size: int):
        x = np.random.randint(
            0, self.vocab_size + 1, size=(batch_size, self.max_len)
        )
        counts = np.zeros((batch_size, self.vocab_size), dtype=np.int32)
        for i in range(self.vocab_size):
            counts[:, i] = np.sum(x == (i + 1), axis=1)
        return x.reshape(-1).astype(np.float32), counts.reshape(-1).astype(
            np.int32
        )

    def prettify(self, x_flat, batch_size):
        x = x_flat.reshape(batch_size, self.max_len).astype(int)
        mapping = [" "] + self.letters
        return [[mapping[idx] for idx in seq] for seq in x]


def main(
    num_epochs: int = 50,
    batch_size: int = 2,
    max_len: int = 4,
    vocab_size: int = 2,
    embed_dim: int = 4,
    num_heads: int = 2,
    sigma_v: float = 1.0,
    steps_per_epoch: int = 100,
):
    """Train a TAGI attention model on the letter counting task."""
    task = CountingTask(max_len=max_len, vocab_size=vocab_size)
    utils = Utils()
    metric = HRCSoftmaxMetric(num_classes=max_len + 1)

    # HRC softmax for count classification (0..max_len)
    num_classes = max_len + 1
    hrc = utils.get_hierarchical_softmax(num_classes)
    total_output = hrc.len

    net = Sequential(
        Embedding(vocab_size + 1, embed_dim, input_size=max_len),
        MultiheadAttention(
            embed_dim,
            num_heads,
            num_heads,
            seq_len=max_len,
            bias=False,
            init_method="Xavier",
        ),
        RMSNorm([embed_dim]),
        Linear(embed_dim, total_output),
    )
    var_y = np.full(
        (batch_size * vocab_size * hrc.num_obs,),
        sigma_v**2,
        dtype=np.float32,
    )

    out_updater = OutputUpdater(net.device)

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        net.train()
        error_rates = []
        for _ in range(steps_per_epoch):
            x, counts = task.next_batch(batch_size)
            m_pred, v_pred = net(x)

            y, y_idx, _ = utils.label_to_obs(
                labels=counts, num_classes=num_classes
            )
            out_updater.update_using_indices(
                output_states=net.output_z_buffer,
                mu_obs=y,
                var_obs=var_y,
                selected_idx=y_idx,
                delta_states=net.input_delta_z_buffer,
            )

            net.backward()
            net.step()

            error_rate = metric.error_rate(m_pred, v_pred, counts)
            error_rates.append(error_rate)

        avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}"
            f" | error: {avg_error * 100:.2f}%"
        )

    # Testing
    x_test, counts_test = task.next_batch(batch_size)
    net.eval()
    m_pred, v_pred = net(x_test)
    predicted = metric.get_predicted_labels(m_pred, v_pred)

    sequences = task.prettify(x_test, batch_size)
    num_show = min(5, batch_size)
    print(f"\nTest Results (showing {num_show} of {batch_size}):")
    for i in range(num_show):
        print(f"  Input: {sequences[i]}")
        print(f"  Target:     {dict(zip(task.letters, counts_test[i]))}")
        print(f"  Prediction: {dict(zip(task.letters, predicted[i]))}")
        print()

    accuracy = np.mean(predicted == counts_test)
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    fire.Fire(main)
