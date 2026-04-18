import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "shakespeare", "input.txt"
)


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
        return torch.tensor(x, dtype=torch.long), torch.tensor(
            y, dtype=torch.long
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(
                1, 1, seq_len, seq_len
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        self.last_attn = att.detach()
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden, seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ffn_hidden,
        seq_len,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, ffn_hidden, seq_len)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(
            0
        )
        x = self.embedding(idx) + self.pos_enc(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            context = idx[:, -self.seq_len :]
            logits = self(context)
            logits = logits[:, -1] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def visualize_attention(model, dataset, prompt, device, last_k=32, top_k=10):
    """Show which prior characters the model attends to when guessing the
    next character after `prompt`: (1) bar chart per layer/head of the
    last-query row, (2) top-K characters heatmap across layers/heads."""
    prompt_ids = [dataset.stoi[c] for c in prompt][-model.seq_len :]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        model(x)

    def readable(c):
        return "↵" if c == "\n" else c

    full_labels = [readable(dataset.itos[i]) for i in prompt_ids]
    start = max(0, len(full_labels) - last_k)
    labels = full_labels[start:]
    positions = np.arange(len(labels))

    num_layers = len(model.blocks)
    num_heads = model.blocks[0].attn.num_heads

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(num_heads * 5, num_layers * 2.5),
        squeeze=False,
    )
    attn_mats = [
        block.attn.last_attn[0].cpu().numpy() for block in model.blocks
    ]
    for li, attn in enumerate(attn_mats):
        for hi in range(num_heads):
            last_row = attn[hi, -1, start:]
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
    for li, attn in enumerate(attn_mats):
        for hi in range(num_heads):
            last_row = attn[hi, -1, :]
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


def main(
    num_epochs: int = 20,
    batch_size: int = 64,
    seq_len: int = 128,
    embed_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    ffn_hidden: int = 512,
    steps_per_epoch: int = 200,
    lr: float = 3e-4,
    max_new_tokens: int = 200,
):
    """Train a character-level GPT on Shakespeare text."""
    text = open(DATA_PATH).read()
    dataset = CharDataset(text, seq_len)
    print(f"Data: {len(text)} chars, {dataset.vocab_size} unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(
        dataset.vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ffn_hidden,
        seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            x, y = dataset.next_batch(batch_size)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(
                logits.reshape(-1, dataset.vocab_size), y.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses[-100:]) / min(len(losses), 100)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | ce_loss: {avg_loss:.4f}"
        )

    model.eval()
    prompt = (
        "First Citizen:\nBefore we proceed any further, hear me speak.\n"
        "\nAll:\nSpeak, speak.\n\nFirst Citi"
    )
    x = torch.tensor(
        [[dataset.stoi[c] for c in prompt]], dtype=torch.long, device=device
    )
    generated = model.generate(x, max_new_tokens)
    output = "".join([dataset.itos[i] for i in generated[0].tolist()])
    print(f"\n{output}")

    visualize_attention(model, dataset, prompt, device)


if __name__ == "__main__":
    fire.Fire(main)
