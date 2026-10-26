# cuTAGI

Tractable Approximate Gaussian Inference (TAGI) library: Bayesian neural
networks trained by closed-form Gaussian updates instead of gradients. This
glossary covers the language used in the transformer training work.

## Language

**Readout**:
The final Linear layer that maps the hidden width to the vocabulary width.
_Avoid_: output layer, head, classifier

**Path units**:
The output units a token's label selects in the hierarchical softmax, one per
tree level. Only these units are observed for that token.
_Avoid_: target nodes, selected indices, observed units

**Touched rows**:
The weight rows that receive a non-zero delta in one training step. For the
readout, the union of all tokens' path units in the batch; for the embedding,
the distinct non-padding tokens in the batch. The step updates only those rows;
the embedding backward writes only those delta rows and leaves the rest stale.

**Dense path**:
Training in which the readout computes, ships and updates every vocabulary
unit, whether observed or not.

**Sparse readout**:
Training in which the readout computes, ships and updates only the path units,
yielding the same posterior as the dense path.

**Kernel parity**:
Evaluation level one: a new kernel reproduces the old kernel's outputs on the
same inputs, to a stated tolerance.
_Avoid_: unit check, check 1

**Cross-entropy check**:
Evaluation level two: a training run with the new code reaches the same
validation cross-entropy as the previous runs at the same depth.
_Avoid_: training check, check 3

**Metric iteration**:
A training step on which sampled metrics are computed. It runs the dense path
so the full vocabulary distribution is available.
