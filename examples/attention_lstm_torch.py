"""Attention + LSTM time series forecasting using PyTorch.

Usage:
    python -m examples.attention_lstm_torch
    python -m examples.attention_lstm_torch --corrupt_steps '[5,15]'
"""

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)


class SineDataset:
    """Sine waves at different speeds used as time series data.

    Args:
        n_features: How many sine waves (each one faster than the last).
        seq_len: How many past time steps the model sees as input.
        n_samples: Total number of time steps to generate.
        corrupt_steps: List of step indices to replace with noise in all samples.

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

        self.corrupt_steps = corrupt_steps or []
        for s in self.corrupt_steps:
            x[:, s, :] = np.random.randn(len(x), n_features) * 5.0

        split = int(len(x) * 0.8)
        self.x_train = torch.tensor(x[:split], dtype=torch.float32)
        self.y_train = torch.tensor(y[:split], dtype=torch.float32)
        self.x_test = torch.tensor(x[split:], dtype=torch.float32)
        self.y_test = torch.tensor(y[split:], dtype=torch.float32)

    def next_batch(self, batch_size: int):
        idx = torch.randint(len(self.x_train), (batch_size,))
        return self.x_train[idx], self.y_train[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.shape[1]]


class AttentionLSTM(nn.Module):
    def __init__(self, embed_dim, out_size, num_heads=3, hidden_size=17):
        super().__init__()
        self.pos_enc = PositionalEncoding(embed_dim)
        self.att = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.rms_norm = nn.RMSNorm(embed_dim)
        self.lstm1 = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.pos_enc(x)
        out, attn_weights = self.att(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        out = self.rms_norm(out)
        out, _ = self.lstm1(out)
        out, _ = self.lstm2(out)
        return self.lin(out[:, -1, :]), attn_weights


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
            if corrupt_steps:
                for s in corrupt_steps:
                    ax[row][column].axvline(
                        x=s, color="red", linestyle="--", lw=1.5, alpha=0.7
                    )
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def main(
    num_epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = 20,
    n_features: int = 6,
    num_heads: int = 1,
    hidden_size: int = 16,
    lr: float = 0.03,
    steps_per_epoch: int = 50,
    corrupt_steps: list = [5, 15],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SineDataset(
        n_features=n_features, seq_len=seq_len, corrupt_steps=corrupt_steps
    )
    if dataset.corrupt_steps:
        print(f"Corrupted steps: {dataset.corrupt_steps} (in all samples)")
        print("Expect LOW attention weights at these columns.")

    model = AttentionLSTM(n_features, n_features, num_heads, hidden_size).to(
        device
    )
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            x, y = dataset.next_batch(batch_size)
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | Loss: {np.mean(losses[-50:]):.4f}"
        )

    # Testing: sequential prediction over the full test horizon
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for i in range(0, len(dataset.x_test) - batch_size + 1, batch_size):
            xb = dataset.x_test[i : i + batch_size].to(device)
            yb = dataset.y_test[i : i + batch_size]
            pred, _ = model(xb)
            preds.append(pred.cpu().numpy())
            targets.append(yb.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    test_mse = np.mean((preds - targets) ** 2)

    print(f"\nTest MSE: {test_mse:.4f}")

    # Plot all features: prediction vs actual
    n_test = len(preds)
    t = np.arange(n_test)

    ncols = 2
    nrows = (n_features + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.ravel()
    for f in range(n_features):
        ax = axes[f]
        ax.plot(t, targets[:, f], "k", lw=1, label="Actual")
        ax.plot(t, preds[:, f], "r", lw=1, label="Prediction")
        ax.set_title(f"Feature {f}")
        ax.set_xlabel("Time step")
    axes[0].legend()
    fig.suptitle("PyTorch Attention+LSTM Forecast", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Attention scores from a test sample
    x_probe = dataset.x_test[:1].to(device)
    with torch.no_grad():
        _, attn_weights = model(x_probe)
    attn_np = attn_weights[0].cpu().numpy()  # (num_heads, seq_len, seq_len)
    print(f"\nAttention scores (head 0):")
    print(f"  mu:\n{attn_np[0, 0]}")
    plot_attention_maps([attn_np], seq_len, dataset.corrupt_steps)


if __name__ == "__main__":
    fire.Fire(main)
