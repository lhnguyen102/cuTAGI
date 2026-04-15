"""Attention + LSTM time series forecasting using TAGI.

Usage:
    python -m examples.attention_lstm
"""

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
import pytagi.metric as metric
from pytagi.nn import (
    LSTM,
    Linear,
    MultiheadAttention,
    MultiheadAttentionV2,
    OutputUpdater,
    PositionalEncoding,
    RMSNorm,
    Sequential,
)

np.random.seed(42)
pytagi.manual_seed(42)


class SineDataset:
    """Sine waves at different speeds used as time series data.

    Args:
        n_features: How many sine waves (each one faster than the last).
        seq_len: How many past time steps the model sees as input.
        n_samples: Total number of time steps to generate.

    The model receives a window of seq_len steps and predicts the next one.
    Data is split 80/20 into train/test by time order.
    """

    def __init__(
        self,
        n_features: int = 6,
        seq_len: int = 20,
        n_samples: int = 4000,
        corrupt_steps: list = None,
    ):
        t = np.linspace(-40, 40, n_samples)
        data = np.sin(
            np.stack(
                [t / (np.pi / k) for k in range(1, n_features + 1)], axis=1
            )
        )
        x = np.stack(
            [data[i : i + seq_len] for i in range(n_samples - seq_len)]
        )
        y = data[seq_len:]

        # Corrupt fixed steps in all samples.
        self.corrupt_steps = corrupt_steps or []
        for s in self.corrupt_steps:
            x[:, s, :] = np.random.randn(len(x), n_features) * 5.0

        split = int(len(x) * 0.8)
        self.x_train = x[:split].astype(np.float32)
        self.y_train = y[:split].astype(np.float32)
        self.x_test = x[split:].astype(np.float32)
        self.y_test = y[split:].astype(np.float32)

    def next_batch(self, batch_size: int):
        idx = np.random.choice(len(self.x_train), batch_size, replace=False)
        return self.x_train[idx], self.y_train[idx]


def plot_attention_maps(attn_maps, seq_len, corrupt_steps=None):
    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
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
    labels = list(range(seq_len))
    for row in range(num_layers):
        for column in range(num_heads):
            im = ax[row][column].imshow(
                attn_maps[row][column], origin="lower", vmin=0
            )
            ax[row][column].set_xticks(labels)
            ax[row][column].set_yticks(labels)
            ax[row][column].set_xlabel("Key (time step)")
            ax[row][column].set_ylabel("Query (time step)")
            ax[row][column].set_title(f"Layer {row + 1}, Head {column + 1}")
            fig.colorbar(im, ax=ax[row][column])
            # Mark corrupted columns with red lines
            if corrupt_steps:
                for s in corrupt_steps:
                    ax[row][column].axvline(
                        x=s, color="red", linestyle="--", lw=1.5, alpha=0.7
                    )
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def main(
    num_epochs: int = 50,
    batch_size: int = 16,
    seq_len: int = 20,
    n_features: int = 6,
    num_heads: int = 1,
    hidden_size: int = 16,
    sigma_v: float = 4.0,
    sigma_v_min: float = 0.3,
    decay_factor: float = 0.95,
    steps_per_epoch: int = 50,
    corrupt_steps: list = [5, 15],
):
    dataset = SineDataset(
        n_features=n_features, seq_len=seq_len, corrupt_steps=corrupt_steps
    )
    if dataset.corrupt_steps:
        print(f"Corrupted steps: {dataset.corrupt_steps} (in all samples)")
        print("Expect LOW attention weights at these columns.")

    net = Sequential(
        PositionalEncoding(n_features),
        MultiheadAttention(
            embed_dim=n_features,
            num_heads=num_heads,
            seq_len=seq_len,
            bias=False,
            gain_weight=0.25,
            gain_bias=0.5,
            pos_emb="",
            use_causal_mask=False,
            init_method="He",
        ),
        RMSNorm([n_features]),
        LSTM(n_features, hidden_size, False, seq_len),
        LSTM(hidden_size, hidden_size, True, seq_len),
        Linear(hidden_size, n_features),
    )

    out_updater = OutputUpdater(net.device)
    current_sigma_v = sigma_v

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        var_y = np.full(
            (batch_size * n_features,), current_sigma_v**2, dtype=np.float32
        )
        net.train()
        mses = []
        for _ in range(steps_per_epoch):
            x, y = dataset.next_batch(batch_size)
            m_pred, v_pred = net(x)

            out_updater.update(
                output_states=net.output_z_buffer,
                mu_obs=y.flatten(),
                var_obs=var_y,
                delta_states=net.input_delta_z_buffer,
            )
            net.backward()
            net.step()
            mses.append(metric.mse(m_pred, y.flatten()))

        net.reset_lstm_states()
        current_sigma_v = max(sigma_v_min, current_sigma_v * decay_factor)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"| MSE: {np.mean(mses[-50:]):.4f} | sigma_v: {current_sigma_v:.3f}"
        )

    # Testing: sequential prediction over the full test horizon
    net.eval()
    mu_preds = []
    var_preds = []
    y_test = []

    for i in range(0, len(dataset.x_test) - batch_size + 1, batch_size):
        xb = dataset.x_test[i : i + batch_size]
        yb = dataset.y_test[i : i + batch_size]
        m_pred, v_pred = net(xb)

        mu_preds.extend(m_pred)
        var_preds.extend(v_pred + current_sigma_v**2)
        y_test.extend(yb.flatten())

    mu_preds = np.array(mu_preds)
    std_preds = np.array(var_preds) ** 0.5
    y_test = np.array(y_test)

    mse = metric.mse(mu_preds, y_test)
    log_lik = metric.log_likelihood(
        prediction=mu_preds, observation=y_test, std=std_preds
    )

    print(f"\nTest MSE           : {mse:.4f}")
    print(f"Test Log-likelihood: {log_lik:.4f}")

    # Plot all features: prediction vs actual with uncertainty band
    n_test = len(mu_preds) // n_features
    mu_2d = mu_preds.reshape(n_test, n_features)
    std_2d = std_preds.reshape(n_test, n_features)
    y_2d = y_test.reshape(n_test, n_features)
    t = np.arange(n_test)

    ncols = 2
    nrows = (n_features + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.ravel()
    for f in range(n_features):
        ax = axes[f]
        ax.plot(t, y_2d[:, f], "k", lw=1, label="Actual")
        ax.plot(t, mu_2d[:, f], "r", lw=1, label="Prediction")
        ax.fill_between(
            t,
            mu_2d[:, f] - std_2d[:, f],
            mu_2d[:, f] + std_2d[:, f],
            color="red",
            alpha=0.3,
        )
        ax.set_title(f"Feature {f}")
        ax.set_xlabel("Time step")
    axes[0].legend()
    fig.suptitle("TAGI Attention+LSTM Forecast", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Attention scores from a test sample
    x_probe = dataset.x_test[:1]
    net(x_probe)
    attention_scores = net.get_attention_scores()
    mu_scores = [mu for mu, var in attention_scores.values()]
    plot_attention_maps(
        [m[0] for m in mu_scores], seq_len, dataset.corrupt_steps
    )


if __name__ == "__main__":
    fire.Fire(main)
