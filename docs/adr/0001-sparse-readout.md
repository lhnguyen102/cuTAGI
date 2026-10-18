---
status: accepted
date: 2026-10-18
---

# Sparse readout is an exact, deterministic mode of Linear

The readout `Linear(768 -> 50267)` costs 324 of 369 ms per training step in
`examples/fineweb_gpt2.py`, although the hierarchical softmax observes only 16
path units per token and every other unit's delta is exactly zero. We add a
mode to the existing CUDA Linear layer, reached through
`Sequential::set_output_indices`, that computes, ships and updates only the
path units of the next forward and its backward and step. The result is
bit-equal to the dense path, not merely close, so a training run on the new
code is comparable to the old one without a tolerance argument.

## Decisions

- **Mode, not a new layer.** The model definition, checkpoints and eval path
  stay unchanged. A new layer type would have changed all three.
- **Indices are consumed.** They apply to the next forward only. A forward
  without fresh indices is dense, so an eval loop can never silently score 16
  units. There is no clear call.
- **Deterministic duplicate handling.** Every token observes the root unit, so
  one readout row collects deltas from all 1024 tokens of a batch. We sort
  (token, unit) pairs by unit and reduce each group sequentially in token
  order. Atomics were rejected: they would make the readout nondeterministic
  and break bit-equality.
- **Bit-equal is the parity criterion.** The dense v3 kernels keep one
  accumulator per output element and chain `__fmaf_rn` over the reduction
  index in increasing order. The sparse kernels use the same fma form and the
  same order; the skipped terms are exact zeros, which an fma leaves unchanged.
  A 1e-6 tolerance would hide a changed sum order, and the trust-region clip
  turns any changed sum into a diverging run.
- **1-based unit ids**, as everywhere else in the library (`class_to_obs`
  numbers the root 1; the output updaters, `obs_to_class`, calibration and the
  examples all subtract 1). The layer subtracts 1 once.
- **CUDA only.** The CPU Linear throws. A CPU implementation would have no
  caller; the parity reference is the dense CUDA kernel gathered at the same
  indices.

## Consequences

- Only touched rows move in the step. This is exact because var decay is never
  applied to the last layer. Any future change that decays the readout, or
  adds a per-weight term that is non-zero for a zero delta, breaks the
  equivalence and must be applied densely.
- The output state's `actual_size` becomes `B*T*16` in sparse mode, so the
  jcb fill, the host copy and the delta memset shrink with it. Code that
  reshapes the training output to `(n, vocab)` must only run on iterations
  that used the dense path.
- Kernel changes elsewhere (RMSNorm reductions, tiled attention, cuBLAS) do
  change the sum order and are validated against a cross-entropy check at
  L=8, not against bit-equality.
