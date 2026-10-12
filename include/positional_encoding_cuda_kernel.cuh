#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// mu_z = mu_a + pe[t, d], var_z = var_a, jcb = 1.
// One thread per element of [batch_size, seq_len, embed_dim].
__global__ void positional_encoding_fwd(const float *mu_a, const float *var_a,
                                        const float *pe_cache, int batch_size,
                                        int seq_len, int embed_dim, float *mu_z,
                                        float *var_z, float *jcb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;
    if (idx >= total) return;

    int pe_idx = idx % (seq_len * embed_dim);

    mu_z[idx] = mu_a[idx] + pe_cache[pe_idx];
    var_z[idx] = var_a[idx];
    jcb[idx] = 1.0f;
}
