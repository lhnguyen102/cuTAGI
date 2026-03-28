import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class ReverseDataset:
    def __init__(self, vocab_size: int = 10, seq_len: int = 16):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def next_batch(self, batch_size: int):
        x = np.random.randint(self.vocab_size, size=(batch_size, self.seq_len))
        y = np.flip(x, axis=1).copy()
        return torch.tensor(x, dtype=torch.long), torch.tensor(
            y, dtype=torch.long
        )


class RotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0
    ):
        super().__init__()
        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        positions = torch.arange(max_seq_len).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, x):
        S = x.shape[2]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = self.cos_cache[:S].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:S].unsqueeze(0).unsqueeze(0)
        out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return out.flatten(-2)


class RoPEEncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.use_causal_mask = use_causal_mask
        self.W_q = nn.Linear(model_dim, model_dim, bias=False)
        self.W_k = nn.Linear(model_dim, model_dim, bias=False)
        self.W_v = nn.Linear(model_dim, model_dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.ff = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        B, S, _ = x.shape
        q = (
            self.W_q(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.W_k(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.W_v(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = self.rope(q)
        k = self.rope(k)
        v = self.rope(v)

        scale = self.head_dim**-0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        if self.use_causal_mask:
            mask = torch.tril(torch.ones(S, S, device=x.device))
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))
        attn_weights = attn_weights.softmax(dim=-1)

        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, S, -1)
        x = self.ff(attn_out)
        return x, attn_weights


class ReversePredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                RoPEEncoderBlock(
                    embed_dim,
                    num_heads,
                    embed_dim,
                    dropout,
                    use_causal_mask,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_net = nn.Sequential(
            nn.Linear(embed_dim, vocab_size),
        )

    def forward(self, x):
        h = self.embedding(x)
        attn_maps = []
        for block in self.blocks:
            h, attn_w = block(h)
            attn_maps.append(attn_w)
        return self.output_net(h), attn_maps


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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.use_causal_mask = use_causal_mask
        self.W_qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.ff = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**-0.5
        attn_weights = (q @ k.transpose(-2, -1)) * scale

        if self.use_causal_mask:
            mask = torch.triu(
                torch.full((S, S), float("-inf"), device=x.device), diagonal=1
            )
            attn_weights = attn_weights + mask
        attn_weights = attn_weights.softmax(dim=-1)

        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, S, -1)
        x = self.ff(attn_out)
        return x, attn_weights


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        seq_len: int,
        dropout: float = 0.0,
        use_causal_mask: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_enc = PositionalEncoding(model_dim)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    model_dim,
                    num_heads,
                    model_dim,
                    dropout,
                    use_causal_mask,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_net = nn.Sequential(
            nn.Linear(model_dim, vocab_size),
        )

    def forward(self, x):
        h = self.pos_enc(self.embedding(x))
        attn_maps = []
        for block in self.blocks:
            h, attn_w = block(h)
            attn_maps.append(attn_w)
        return self.output_net(h), attn_maps


def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

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
    embed_dim: int = 32,
    num_heads: int = 1,
    num_layers: int = 1,
    lr: float = 1e-2,
    dropout: float = 0.0,
    steps_per_epoch: int = 100,
    arch: str = "positional",
    use_causal_mask: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = ReverseDataset(vocab_size=vocab_size, seq_len=seq_len)

    if arch == "positional":
        model = TransformerPredictor(
            vocab_size=vocab_size,
            model_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_len=seq_len,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
        ).to(device)
    else:
        model = ReversePredictor(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            seq_len=seq_len,
            num_layers=num_layers,
            dropout=dropout,
            use_causal_mask=use_causal_mask,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        error_rates = []
        for _ in range(steps_per_epoch):
            x, y = task.next_batch(batch_size)
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=-1)
            error_rate = (preds != y).float().mean().item()
            error_rates.append(error_rate)

        avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | error: {avg_error * 100:.2f}%"
        )

    model.eval()
    test_batch_size = 100
    x_test, y_test = task.next_batch(test_batch_size)
    x_test, y_test = x_test.to(device), y_test.to(device)
    with torch.no_grad():
        logits, attn_weights = model(x_test)
    predicted = logits.argmax(dim=-1)

    x_np = x_test.cpu().numpy()
    y_np = y_test.cpu().numpy()
    pred_np = predicted.cpu().numpy()

    num_show = min(5, test_batch_size)
    print(f"\nTest Results (showing {num_show} of {test_batch_size}):")
    for i in range(num_show):
        print(f"  Input:      {x_np[i].tolist()}")
        print(f"  Target:     {y_np[i].tolist()}")
        print(f"  Prediction: {pred_np[i].tolist()}")
        print()

    accuracy = (predicted == y_test).float().mean().item()
    print(f"Test accuracy: {accuracy:.2%}")

    print(f"\nAttention scores (sample 0, head 0):")
    print(attn_weights[0][0, 0].detach().cpu().numpy())

    block = model.blocks[0]
    print(
        f"\nEmbedding weight stats: mean={model.embedding.weight.data.mean():.6f}, "
        f"std={model.embedding.weight.data.std():.6f}"
    )

    plot_attention_maps(x_test, attn_weights)


if __name__ == "__main__":
    fire.Fire(main)
