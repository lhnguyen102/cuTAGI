"""CE floor imposed by the hierarchical-softmax observation model.

The TAGI runs decode CE from predicted means AND predictive variances through
obs_to_label_prob, so CE improves both as the mean gets right and as the
predictive variance shrinks. The torch reference uses a full 50257-way softmax
with no observation noise. This asks what CE the TAGI output model can reach at
all, independent of the optimizer: it builds a prediction that is perfect up to
a mean-scale factor k and a uniform predictive variance v, decodes it exactly
as sampled_metrics() does, and reports CE(k, v).

k = 1 is a perfectly fit mean (|m|/obs_scale = 1, the encoding's own target).
The tau=1500 sigma_v 4->2 run sits near k ~ 0.47, v ~ 0.18 (from the
out-diag mu_a[idx] line: |mu| mean 1.40 against obs_scale 3, var mean 1.59).

Usage:
    python -m examples.hrc_ceiling
    python -m examples.hrc_ceiling --n_labels=200
"""

import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import numpy as np

import pytagi
from pytagi import Utils

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fineweb")
VOCAB_SIZE = 50257


def decode_ce(utils, hrc, m, v, label):
    probs = np.asarray(utils.obs_to_label_prob(m, v, hrc, VOCAB_SIZE)).reshape(
        -1
    )
    psum = float(probs.sum())
    probs = probs / psum
    return -np.log(max(probs[label], 1e-9)), psum


def main(n_labels: int = 100, seed: int = 44):
    np.random.seed(seed)
    pytagi.manual_seed(seed)

    utils = Utils()
    hrc = utils.get_hierarchical_softmax(VOCAB_SIZE)

    data = np.memmap(
        os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r"
    )
    labels = np.asarray(
        data[np.random.randint(0, len(data), size=n_labels)], dtype=np.int32
    )

    ks = [0.47, 0.75, 1.0, 1.5, 2.0, 4.0]
    vs = [0.5, 0.18, 0.05, 0.01, 1e-3, 1e-4]

    print(f"hrc.len={hrc.len} num_obs={hrc.num_obs} n_labels={n_labels}")
    print("CE decoded from a perfect-up-to-scale mean (rows k, cols v)\n")
    header = "  k \\ v " + "".join(f"{v:>10.4g}" for v in vs)
    print(header)

    for k in ks:
        row = f"{k:6.2f} "
        for v in vs:
            ce_sum, psum_sum = 0.0, 0.0
            for label in labels:
                obs, obs_idx, _ = utils.label_to_obs(
                    np.array([label], dtype=np.int32), VOCAB_SIZE
                )
                m = np.zeros(hrc.len, dtype=np.float32)
                m[obs_idx - 1] = k * obs
                sa = np.full(hrc.len, v, dtype=np.float32)
                ce, psum = decode_ce(utils, hrc, m, sa, int(label))
                ce_sum += ce
                psum_sum += psum
            row += f"{ce_sum / len(labels):>10.4f}"
        print(row)

    print("\npsum at k=1 (sanity, should be ~1):")
    for v in vs:
        ps = 0.0
        for label in labels:
            obs, obs_idx, _ = utils.label_to_obs(
                np.array([label], dtype=np.int32), VOCAB_SIZE
            )
            m = np.zeros(hrc.len, dtype=np.float32)
            m[obs_idx - 1] = obs
            sa = np.full(hrc.len, v, dtype=np.float32)
            ps += decode_ce(utils, hrc, m, sa, int(label))[1]
        print(f"  v={v:<8.4g} psum={ps / len(labels):.4f}")


if __name__ == "__main__":
    fire.Fire(main)
