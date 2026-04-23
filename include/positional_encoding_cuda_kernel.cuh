#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// Forward kernel: y = x + pe[t, d], var_y = var_x, jcb = 1.
// Total threads = batch_size * seq_len * embed_dim, mapped 1-D.
__global__ void positional_encoding_fwd(const float *mu_a, const float *var_a,
                                        const float *pe_cache, int batch_size,
                                        int seq_len, int embed_dim, float *mu_z,
                                        float *var_z, float *jcb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;
    if (idx >= total) return;

    int d = idx % embed_dim;
    int t = (idx / embed_dim) % seq_len;
    int pe_idx = t * embed_dim + d;

    mu_z[idx] = mu_a[idx] + pe_cache[pe_idx];
    var_z[idx] = var_a[idx];
    jcb[idx] = 1.0f;
}
