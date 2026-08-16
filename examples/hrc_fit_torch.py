"""Can the hierarchical-softmax head express the torch model's distribution?

examples/hrc_ceiling.py showed the head is not a sharpness ceiling: a correct
mean decodes to CE 0.50 at the predictive variance TAGI actually reaches. That
test used one-hot targets, so it says nothing about the diffuse distributions
real next-token prediction requires, which is where the 6.456 vs 6.070 gap
lives.

This test closes that. It takes the torch reference's predicted distribution
p(.|context), converts it to the ideal HRC node statistics - for each internal
node on some token's path, the branch probability implied by summing leaf
probabilities on each side - and decodes those through obs_to_label_prob. The
resulting CE is what TAGI's head would score if it fit the torch model's
distribution exactly.

  CE ~ 6.07  -> head is adequate, the whole gap is optimization
  CE ~ 6.4   -> head is the binding constraint, and every optimizer-side lever
                bottoming out at the same place is explained

Node statistics use the Bernoulli moments of the +/-1 encoding, which is what
the observation model targets: m = 2p - 1, v = 1 - m^2.

Usage:
    python -m examples.hrc_fit_torch --n_contexts=200
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np
import torch

import pytagi
from examples.fineweb_gpt2_torch import GPT, get_batch
from pytagi import Utils

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fineweb")
VOCAB_SIZE = 50257


def build_path_table(utils):
    """(idx, sign) for every vocab entry, as int32 arrays [VOCAB, num_obs]."""
    labels = np.arange(VOCAB_SIZE, dtype=np.int32)
    obs, obs_idx, _ = utils.label_to_obs(labels, VOCAB_SIZE)
    n = obs_idx.size // VOCAB_SIZE
    # obs_idx is 1-based with respect to the array obs_to_label_prob consumes
    return (
        obs_idx.reshape(VOCAB_SIZE, n).astype(np.int64) - 1,
        obs.reshape(VOCAB_SIZE, n).astype(np.float32),
    )


def hrc_stats_from_dist(probs, idx_tab, sign_tab, hrc_len):
    """Ideal node mean/variance for a distribution over the vocabulary."""
    w = np.repeat(probs[:, None], idx_tab.shape[1], axis=1).reshape(-1)
    flat_idx = idx_tab.reshape(-1)
    flat_sgn = sign_tab.reshape(-1)

    mass = np.bincount(flat_idx, weights=w, minlength=hrc_len)
    signed = np.bincount(flat_idx, weights=w * flat_sgn, minlength=hrc_len)

    m = np.zeros(hrc_len, dtype=np.float32)
    seen = mass > 0
    m[seen] = (signed[seen] / mass[seen]).astype(np.float32)
    v = (1.0 - m**2).astype(np.float32)
    # unvisited nodes carry no information: coin flip
    v[~seen] = 1.0
    return m, v


def main(
    n_contexts: int = 200,
    seq_len: int = 64,
    batch_size: int = 16,
    ckpt: str = "out/fineweb_torch/ckpt.pt",
    var_floor: float = 1e-4,
):
    np.random.seed(44)
    pytagi.manual_seed(44)
    torch.manual_seed(44)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(VOCAB_SIZE, 768, 8, 4, 1024, seq_len, 10000.0).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    utils = Utils()
    hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)
    idx_tab, sign_tab = build_path_table(utils)
    print(f"hrc.len={hrc.len} path_len={idx_tab.shape[1]}")

    ce_torch, ce_hrc, psum_acc, n = 0.0, 0.0, 0.0, 0
    while n < n_contexts:
        x, y = get_batch("val", batch_size, seq_len, device)
        with torch.no_grad():
            logits = model(x)
        logp = torch.log_softmax(logits.reshape(-1, VOCAB_SIZE), dim=-1)
        probs = logp.exp().cpu().numpy()
        labels = y.reshape(-1).cpu().numpy()

        pick = np.random.choice(
            len(labels), size=min(batch_size, n_contexts - n), replace=False
        )
        for i in pick:
            p = probs[i].astype(np.float64)
            p = p / p.sum()
            ce_torch -= float(np.log(max(p[labels[i]], 1e-12)))

            m, v = hrc_stats_from_dist(p, idx_tab, sign_tab, hrc.len)
            v = np.maximum(v, var_floor)
            q = np.asarray(
                utils.obs_to_label_prob(m, v, hrc, VOCAB_SIZE)
            ).reshape(-1)
            psum_acc += float(q.sum())
            q = q / q.sum()
            ce_hrc -= float(np.log(max(q[labels[i]], 1e-12)))
            n += 1

    print(f"\ncontexts            {n}")
    print(f"torch CE            {ce_torch / n:.4f}")
    print(f"HRC-refit CE        {ce_hrc / n:.4f}")
    print(f"head cost           {(ce_hrc - ce_torch) / n:+.4f}")
    print(f"mean psum           {psum_acc / n:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
