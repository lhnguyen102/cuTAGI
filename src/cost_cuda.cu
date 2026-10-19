#include <cuda_runtime.h>

#include "../include/cost_cuda.cuh"
#include "../include/cuda_error_checking.cuh"

__global__ void node_prob_cuda(const float *mz, const float *Sz,
                               const int *node_level, const float *level_scale,
                               const float *level_offset, int len, int size,
                               float *P_z)
/* P(node) = Phi(s * z + b) with z = mz / sqrt(1/alpha^2 + Sz), alpha = 3.
 * Same float/double op sequence as obs_to_class in cost.cpp, so z is
 * bit-equal; erfc runs in double to stay within one float ulp of the CPU.
 */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const double inv_alpha_sq = (1.0f / 3.0f) * (double)(1.0f / 3.0f);
    float z = (float)(mz[i] / sqrt(inv_alpha_sq + (double)Sz[i]));
    if (level_scale != nullptr) {
        z *= level_scale[node_level[i % len]];
    }
    if (level_offset != nullptr) {
        z += level_offset[node_level[i % len]];
    }
    float arg = -z * 0.7071067811865475f;
    P_z[i] = 0.5f * (float)erfc((double)arg);
}

__global__ void class_prob_cuda(const float *P_z, const float *obs,
                                const int *idx, int n_classes, int n_obs,
                                int len, int size, float *P)
/* Product of the node probabilities along each class's path, in level order
 * like obs_to_class in cost.cpp.
 */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    int r = i / n_classes;
    int c = i % n_classes;
    const float *row = P_z + (size_t)r * len;
    float tmp = 1.0f;
    for (int k = 0; k < n_obs; k++) {
        float p = row[idx[c * n_obs + k] - 1];
        if (obs[c * n_obs + k] == -1.0f) {
            tmp *= fabsf(p - 1.0f);
        } else {
            tmp *= p;
        }
    }
    P[i] = tmp;
}

namespace {
template <typename T>
T *to_device(const T *host, size_t count) {
    T *d = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d, count * sizeof(T)));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
}  // namespace

void obs_to_class_cuda(const float *mz, const float *Sz, int rows,
                       const HRCSoftmax &hs, int n_classes,
                       const std::vector<float> &level_scale,
                       const std::vector<float> &level_offset, float *P) {
    // Column c of hs.idx holds the level-c node on each class's path.
    std::vector<int> node_level(hs.len, 0);
    for (int r = 0; r < n_classes; r++) {
        for (int c = 0; c < hs.n_obs; c++) {
            node_level[hs.idx[r * hs.n_obs + c] - 1] = c;
        }
    }

    size_t num_nodes = static_cast<size_t>(rows) * hs.len;
    size_t num_probs = static_cast<size_t>(rows) * n_classes;
    float *d_mz = to_device(mz, num_nodes);
    float *d_Sz = to_device(Sz, num_nodes);
    int *d_node_level = to_device(node_level.data(), node_level.size());
    float *d_scale = level_scale.empty()
                         ? nullptr
                         : to_device(level_scale.data(), level_scale.size());
    float *d_offset = level_offset.empty()
                          ? nullptr
                          : to_device(level_offset.data(), level_offset.size());
    float *d_obs = to_device(hs.obs.data(), hs.obs.size());
    int *d_idx = to_device(hs.idx.data(), hs.idx.size());
    float *d_P_z = nullptr, *d_P = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_P_z, num_nodes * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, num_probs * sizeof(float)));

    const int threads = 256;
    node_prob_cuda<<<(num_nodes + threads - 1) / threads, threads>>>(
        d_mz, d_Sz, d_node_level, d_scale, d_offset, hs.len, num_nodes, d_P_z);
    class_prob_cuda<<<(num_probs + threads - 1) / threads, threads>>>(
        d_P_z, d_obs, d_idx, n_classes, hs.n_obs, hs.len, num_probs, d_P);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(
        cudaMemcpy(P, d_P, num_probs * sizeof(float), cudaMemcpyDeviceToHost));

    for (void *d : {(void *)d_mz, (void *)d_Sz, (void *)d_node_level,
                    (void *)d_scale, (void *)d_offset, (void *)d_obs,
                    (void *)d_idx, (void *)d_P_z, (void *)d_P}) {
        cudaFree(d);
    }
}
