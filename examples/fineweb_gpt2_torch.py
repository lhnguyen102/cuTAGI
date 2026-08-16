import os
import sys

import fire
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from tqdm import tqdm

np.random.seed(44)
torch.manual_seed(44)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fineweb")
VOCAB_SIZE = 50257  # gpt2 BPE


def get_batch(split: str, batch_size: int, seq_len: int, device):
    # recreate memmap every batch to avoid leaking page cache (see nanoGPT)
    data = np.memmap(
        os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r"
    )
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i : i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + seq_len] for i in ix]).astype(np.int64)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def build_rope_cache(seq_len, head_dim, theta=10000.0):
    """Mirror generate_rope_cache in src/attention.cpp."""
    half_dim = head_dim // 2
    log_theta = -float(np.log(theta)) / head_dim
    i = torch.arange(half_dim, dtype=torch.float32)
    freqs = torch.exp(2.0 * i * log_theta)
    pos = torch.arange(seq_len, dtype=torch.float32)
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos_cache, sin_cache):
    """Paired-adjacent RoPE matching apply_rope in src/attention.cpp:
    out[..., 2d]   = x1*cos - x2*sin
    out[..., 2d+1] = x1*sin + x2*cos
    x has shape (B, H, T, D)."""
    T = x.size(-2)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos = cos_cache[:T].unsqueeze(0).unsqueeze(0)
    sin = sin_cache[:T].unsqueeze(0).unsqueeze(0)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class RopeCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len, rope_theta=10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(
                1, 1, seq_len, seq_len
            ),
        )
        cos, sin = build_rope_cache(seq_len, self.head_dim, rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    """Mirror the TAGI ResNetBlock pair: RMSNorm -> MHA -> proj residual,
    then RMSNorm -> Linear x3 with ReLU residual (all bias-free)."""

    def __init__(self, embed_dim, num_heads, ffn_hidden, seq_len, rope_theta):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = RopeCausalSelfAttention(
            embed_dim, num_heads, seq_len, rope_theta
        )
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(ffn_hidden, ffn_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(ffn_hidden, embed_dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
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
        rope_theta=10000.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, ffn_hidden, seq_len, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.norm_f = RMSNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, idx):
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
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


def main(
    seq_len: int = 64,
    embed_dim: int = 768,
    num_heads: int = 8,
    num_layers: int = 4,
    ffn_hidden: int = 1024,
    max_iters: int = 5_000,
    batch_size: int = 16,
    lr: float = 3e-4,
    rope_theta: float = 10000.0,
    eval_interval: int = 2000,
    eval_iters: int = 50,
    log_interval: int = 50,
    out_dir: str = "out/fineweb_torch",
    init_from: str = "scratch",
    eval_only: bool = False,
    max_new_tokens: int = 200,
):
    """Train a PyTorch GPT-2 on FineWeb (BPE token level), mirroring the
    TAGI architecture in examples/fineweb_gpt2.py."""
    train_path = os.path.join(DATA_DIR, "train.bin")
    if not os.path.exists(train_path):
        sys.exit(f"{train_path} not found. Run: python -m data.fineweb.prepare")

    enc = tiktoken.get_encoding("gpt2")
    tokens_per_iter = batch_size * seq_len
    print(f"tokens per iteration: {tokens_per_iter:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(
        VOCAB_SIZE,
        embed_dim,
        num_heads,
        num_layers,
        ffn_hidden,
        seq_len,
        rope_theta,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.2f}M")

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if init_from == "resume":
        print(f"resuming from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        ces, errs = [], []
        for _ in range(eval_iters):
            x, y = get_batch("val", batch_size, seq_len, device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            ces.append(loss.item())
            errs.append((logits.argmax(dim=-1) != y).float().mean().item())
        model.train()
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
                torch.save(model.state_dict(), ckpt_path)
            if eval_only:
                break

        x, y = get_batch("train", batch_size, seq_len, device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % log_interval == 0:
            err = (logits.argmax(dim=-1) != y).float().mean().item()
            losses.append(loss.item())
            error_rates.append(err)
            avg_loss = sum(losses[-100:]) / min(len(losses), 100)
            avg_error = sum(error_rates[-100:]) / min(len(error_rates), 100)
            pbar.set_description(
                f"Iter {iter_num}/{max_iters} | ce_loss: {avg_loss:.4f} | "
                f"error: {avg_error * 100:.2f}%"
            )

    # Generation sample
    model.eval()
    val_data = np.memmap(
        os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r"
    )
    prompt_ids = val_data[:seq_len].astype(np.int64).tolist()
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(x, max_new_tokens)[0].tolist()

    prompt_text = enc.decode(prompt_ids)
    new_text = enc.decode(generated[len(prompt_ids) :])
    print(f"\n--- prompt ---\n{prompt_text}")
    print(f"--- continuation ---\n\033[31m{new_text}\033[0m")


if __name__ == "__main__":
    fire.Fire(main)
