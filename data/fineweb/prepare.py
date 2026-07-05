"""Prepare a subset of FineWeb for TAGI GPT-2 training.

Streams documents from HuggingFace (no full download), tokenizes with
GPT-2 BPE (tiktoken), and writes train.bin/val.bin as uint16 token ids --
the same raw format used by data/openwebtext/prepare.py.

FineWeb has two flavours (see coderef/llm.c-master/dev/data/fineweb.py):
    --dataset_type=edu      HuggingFaceFW/fineweb-edu   (default, higher quality)
    --dataset_type=classic  HuggingFaceFW/fineweb

Dependencies (see data/fineweb/requirements.txt):
    pip install tiktoken datasets

python -m data.fineweb.prepare --train_tokens=200000000 --val_tokens=2000000
"""

import os

import fire
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = os.path.dirname(__file__)

DATASETS = {
    "edu": "HuggingFaceFW/fineweb-edu",
    "classic": "HuggingFaceFW/fineweb",
}


def write_split(doc_iter, enc, path, target_tokens, batch_docs=512):
    written = 0
    pbar = tqdm(
        total=target_tokens,
        desc=f"writing {os.path.basename(path)}",
        unit="tok",
        unit_scale=True,
    )
    with open(path, "wb") as f:
        while written < target_tokens:
            texts = []
            for _ in range(batch_docs):
                try:
                    texts.append(next(doc_iter)["text"])
                except StopIteration:
                    break
            if not texts:
                break
            for ids in enc.encode_ordinary_batch(texts):
                ids.append(enc.eot_token)
                arr = np.array(ids, dtype=np.uint16)
                f.write(arr.tobytes())
                written += len(arr)
                pbar.update(len(arr))
                if written >= target_tokens:
                    break
    pbar.close()
    return written


def main(
    train_tokens: int = 200_000_000,
    val_tokens: int = 2_000_000,
    dataset_type: str = "edu",
    sample: str = "sample-10BT",
    shuffle_buffer: int = 10_000,
    seed: int = 2357,
):
    if dataset_type not in DATASETS:
        raise ValueError(
            f"dataset_type must be one of {list(DATASETS)}, got {dataset_type}"
        )
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(
        DATASETS[dataset_type], name=sample, split="train", streaming=True
    )
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    doc_iter = iter(ds)

    n_val = write_split(
        doc_iter, enc, os.path.join(OUT_DIR, "val.bin"), val_tokens
    )
    n_train = write_split(
        doc_iter, enc, os.path.join(OUT_DIR, "train.bin"), train_tokens
    )
    print(f"val.bin: {n_val:,} tokens | train.bin: {n_train:,} tokens")
    print(f"vocab: gpt2 BPE, {enc.n_vocab} tokens (eot={enc.eot_token})")


if __name__ == "__main__":
    fire.Fire(main)
