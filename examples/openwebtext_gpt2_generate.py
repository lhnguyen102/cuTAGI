import os
import sys
import time

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
import tiktoken

import pytagi
from examples.openwebtext_gpt2 import (
    DATA_DIR,
    VOCAB_SIZE,
    build_mingpt,
)
from pytagi import Utils

PAD_TOKEN = 198  # gpt2 BPE id for "\n", used to left-pad short prompts


def _fit_context(ids, seq_len):
    """Left-pad or crop a token list to exactly seq_len."""
    ids = list(ids)[-seq_len:]
    if len(ids) < seq_len:
        ids = [PAD_TOKEN] * (seq_len - len(ids)) + ids
    return ids


def _next_token_probs(net, utils, hrc, context, seq_len, obs_scale):
    x = np.array(context, dtype=np.float32).reshape(1, seq_len, 1)
    m_pred, v_pred = net(x)
    m_last = np.asarray(m_pred).reshape(seq_len, hrc.len)[-1] / obs_scale
    v_last = np.asarray(v_pred).reshape(seq_len, hrc.len)[-1] / (obs_scale**2)
    probs = np.asarray(
        utils.obs_to_label_prob(m_last, v_last, hrc, VOCAB_SIZE)
    ).reshape(-1)
    return probs / probs.sum()


def _sample(probs, temperature, top_k, greedy, rng):
    if greedy:
        return int(np.argmax(probs))
    if temperature != 1.0:
        logits = np.log(np.maximum(probs, 1e-12)) / max(temperature, 1e-6)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
    if top_k and top_k > 0:
        keep = np.argpartition(probs, -top_k)[-top_k:]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()
    return int(rng.choice(VOCAB_SIZE, p=probs))


def generate(net, utils, hrc, prompt_ids, cfg, rng, on_token=None):
    generated = list(prompt_ids)
    new_ids = []
    for _ in range(cfg["tokens"]):
        context = _fit_context(generated, cfg["seq_len"])
        probs = _next_token_probs(
            net, utils, hrc, context, cfg["seq_len"], cfg["obs_scale"]
        )
        tok = _sample(probs, cfg["temp"], cfg["top_k"], cfg["greedy"], rng)
        generated.append(tok)
        new_ids.append(tok)
        if on_token is not None:
            on_token(new_ids)
    return new_ids


HELP = """
Commands:
  <text>            generate continuation(s) from the typed prompt
  :val [offset]     use a prompt of seq_len tokens from val.bin (default 0)
  :samples N        number of samples per prompt (default 3)
  :tokens N         new tokens to generate (default 200)
  :temp F           sampling temperature (default 1.0)
  :topk N           top-k filtering, 0 to disable (default 0)
  :greedy           toggle greedy (argmax) decoding
  :show             show current settings
  :help             show this help
  :quit / :q        exit
"""


def main(
    seq_len: int = 128,
    embed_dim: int = 384,
    num_heads: int = 8,
    num_layers: int = 1,
    ffn_hidden: int = 1024,
    obs_scale: float = 2.0,
    qkv_gain: float = 0.25,
    gain_w_rms: float = 5.0,
    embed_scale: float = 0.15,
    prior_pull: float = 0.0005,
    ckpt_path: str = "ckpt/ckpt.bin",
    seed: int = 44,
):
    """Interactive generation from a trained TAGI GPT-2 checkpoint."""
    if not os.path.exists(ckpt_path):
        sys.exit(f"{ckpt_path} not found.")

    np.random.seed(seed)
    pytagi.manual_seed(seed)
    rng = np.random.default_rng(seed)

    enc = tiktoken.get_encoding("gpt2")
    utils = Utils()
    hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)

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
    )
    net.to_device("cuda" if pytagi.cuda.is_available() else "cpu")
    net.load(ckpt_path)
    net.eval()
    print(f"loaded checkpoint from {ckpt_path} (device: {net.device})")

    val_path = os.path.join(DATA_DIR, "val.bin")
    val_data = (
        np.memmap(val_path, dtype=np.uint16, mode="r")
        if os.path.exists(val_path)
        else None
    )

    cfg = {
        "seq_len": seq_len,
        "obs_scale": obs_scale,
        "samples": 3,
        "tokens": 200,
        "temp": 1.0,
        "top_k": 0,
        "greedy": False,
    }
    print(HELP)

    def run(prompt_ids, prompt_text):
        n = 1 if cfg["greedy"] else cfg["samples"]
        mode = (
            "greedy"
            if cfg["greedy"]
            else f"temp={cfg['temp']}, top_k={cfg['top_k']}"
        )
        print(f"=== {n} sample(s) | {mode} | {cfg['tokens']} tokens ===")
        for s in range(n):
            print(f"\n--- sample {s + 1}/{n} ---\n{prompt_text}", end="")
            sys.stdout.write("\033[31m")

            shown = {"n": 0}

            def on_token(new_ids):
                text = enc.decode(new_ids)
                sys.stdout.write(text[shown["n"] :])
                sys.stdout.flush()
                shown["n"] = len(text)

            t0 = time.perf_counter()
            new_ids = generate(net, utils, hrc, prompt_ids, cfg, rng, on_token)
            dt = time.perf_counter() - t0
            sys.stdout.write("\033[0m\n")
            tps = len(new_ids) / dt if dt > 0 else float("inf")
            print(f"[{len(new_ids)} tokens in {dt:.2f}s | {tps:.1f} tok/s]")
            sys.stdout.flush()

    while True:
        try:
            line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith(":"):
            parts = line.split()
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else None)
            if cmd in (":quit", ":q"):
                break
            elif cmd == ":help":
                print(HELP)
            elif cmd == ":show":
                print(cfg)
            elif cmd == ":greedy":
                cfg["greedy"] = not cfg["greedy"]
                print(f"greedy = {cfg['greedy']}")
            elif cmd == ":samples" and arg:
                cfg["samples"] = int(arg)
            elif cmd == ":tokens" and arg:
                cfg["tokens"] = int(arg)
            elif cmd == ":temp" and arg:
                cfg["temp"] = float(arg)
            elif cmd == ":topk" and arg:
                cfg["top_k"] = int(arg)
            elif cmd == ":val":
                if val_data is None:
                    print(f"{val_path} not found.")
                    continue
                off = int(arg) if arg else 0
                prompt_ids = (
                    val_data[off : off + seq_len].astype(np.int64).tolist()
                )
                run(prompt_ids, enc.decode(prompt_ids))
            else:
                print(f"unknown command: {line}")
            continue

        prompt_ids = _fit_context(enc.encode(line), seq_len)
        run(prompt_ids, line)


if __name__ == "__main__":
    fire.Fire(main)
