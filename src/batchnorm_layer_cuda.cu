#include "../include/batchnorm_layer_cuda.cuh"
#include "../include/param_init.h"

// Sum reduction kernels
__device__ void warp_smem_reduction(volatile float *smem_mu, int tx, int ty,
                                    int BLOCK_DIM)
/*
 */
{
    float mu_x = smem_mu[ty * BLOCK_DIM + tx];

    if (blockDim.x >= WARP_SIZE * 2) {
        mu_x += smem_mu[ty * BLOCK_DIM + tx + 32];
        __syncwarp();
        smem_mu[ty * BLOCK_DIM + tx] = mu_x;
        __syncwarp();
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        mu_x += smem_mu[ty * BLOCK_DIM + tx + offset];
        __syncwarp();
        smem_mu[ty * BLOCK_DIM + tx] = mu_x;
        __syncwarp();
    }
}

__device__ void dual_warp_smem_reduction(volatile float *smem_mu,
                                         volatile float *smem_var, int tx,
                                         int ty, int BLOCK_DIM)
/*
 */
{
    float mu_x = smem_mu[ty * BLOCK_DIM + tx];
    float var_x = smem_var[ty * BLOCK_DIM + tx];

    if (blockDim.x >= WARP_SIZE * 2) {
        mu_x += smem_mu[ty * BLOCK_DIM + tx + 32];
        var_x += smem_var[ty * BLOCK_DIM + tx + 32];
        __syncwarp();
        smem_mu[ty * BLOCK_DIM + tx] = mu_x;
        smem_var[ty * BLOCK_DIM + tx] = var_x;
        __syncwarp();
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        mu_x += smem_mu[ty * BLOCK_DIM + tx + offset];
        var_x += smem_var[ty * BLOCK_DIM + tx + offset];
        __syncwarp();
        smem_mu[ty * BLOCK_DIM + tx] = mu_x;
        smem_var[ty * BLOCK_DIM + tx] = var_x;
        __syncwarp();
    }
}

template <int BLOCK_TILE_X, int BLOCK_TILE_Y>
__global__ void dual_sum_reduction(float const *delta_mu_in,
                                   float const *delta_var_in, size_t len_x,
                                   size_t len_y, float *delta_mu_out,
                                   float *delta_var_out)
/*
 */
{
    __shared__ float smem_mu[BLOCK_TILE_Y * BLOCK_TILE_X];
    __shared__ float smem_var[BLOCK_TILE_Y * BLOCK_TILE_X];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t col = blockIdx.x * BLOCK_TILE_X + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < len_x && row < len_y) {
        smem_mu[ty * BLOCK_TILE_X + tx] = delta_mu_in[row * len_x + col];
        smem_var[ty * BLOCK_TILE_X + tx] = delta_var_in[row * len_x + col];
    } else {
        smem_mu[ty * BLOCK_TILE_X + tx] = 0.0f;
        smem_var[ty * BLOCK_TILE_X + tx] = 0.0f;
    }

    __syncthreads();

    for (size_t i = BLOCK_TILE_X / 2; i > WARP_SIZE; i >>= 1) {
        if (tx < i) {
            smem_mu[ty * BLOCK_TILE_X + tx] +=
                smem_mu[ty * BLOCK_TILE_X + tx + i];
            smem_var[ty * BLOCK_TILE_X + tx] +=
                smem_var[ty * BLOCK_TILE_X + tx + i];
        }
        __syncthreads();
    }

    if (tx < WARP_SIZE) {
        dual_warp_smem_reduction(smem_mu, smem_var, tx, ty, BLOCK_TILE_X);
    }

    if (tx == 0 && row < len_y) {
        delta_mu_out[row * gridDim.x + blockIdx.x] =
            smem_mu[ty * BLOCK_TILE_X + tx];
        delta_var_out[row * gridDim.x + blockIdx.x] =
            smem_var[ty * BLOCK_TILE_X + tx];
    }
}

template <int BLOCK_TILE_X, int BLOCK_TILE_Y>
__global__ void sum_reduction(float const *mu_in, size_t len_x, size_t len_y,
                              float *mu_out)
/*
 */
{
    __shared__ float smem_mu[BLOCK_TILE_Y * BLOCK_TILE_X];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t col = blockIdx.x * BLOCK_TILE_X + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < len_x && row < len_y) {
        smem_mu[ty * BLOCK_TILE_X + tx] = mu_in[row * len_x + col];
    } else {
        smem_mu[ty * BLOCK_TILE_X + tx] = 0.0f;
    }

    __syncthreads();

    for (size_t i = BLOCK_TILE_X / 2; i > WARP_SIZE; i >>= 1) {
        if (tx < i) {
            smem_mu[ty * BLOCK_TILE_X + tx] +=
                smem_mu[ty * BLOCK_TILE_X + tx + i];
        }
        __syncthreads();
    }

    if (tx < WARP_SIZE) {
        warp_smem_reduction(smem_mu, tx, ty, BLOCK_TILE_X);
    }

    if (tx == 0 && row < len_y) {
        mu_out[row * gridDim.x + blockIdx.x] = smem_mu[ty * BLOCK_TILE_X + tx];
    }
}

__global__ void running_mean_var_cuda(float const *mu_s, float const *var_s,
                                      float momentum, int num_states,
                                      float *mu_ra, float *var_ra)
/*Copute the running average for the normalization layers.
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_states) {
        mu_ra[col] = mu_ra[col] * momentum + mu_s[col] * (1.0f - momentum);
        var_ra[col] = var_ra[col] * momentum + var_s[col] * (1.0f - momentum);
    }
}

__global__ void batchnorm_stat_mean_var_cuda(float const *mu_a,
                                             float const *var_a, int ni,
                                             int batch_size, float *mu_s,
                                             float *var_s)
/*Compute sample mean and variance of activation units of full-connected layer
for each batch.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0;
    float sum_var = 0;
    if (col < ni) {
        for (int i = 0; i < batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col + i * ni];
            sum_var += var_a[col + i * ni];
        }
        mu_s[col] = sum_mu / batch_size;
        var_s[col] = sum_var;
    }
}

__global__ void batchnorm_sample_var_cuda(float const *mu_a, float const *mu_s,
                                          float const *var_s, int ni,
                                          int batch_size, float *var)
/*Compute statistical mean and variance of activation units for full-connected
layer for each batch.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < ni) {
        for (int i = 0; i < batch_size; i++) {
            sum += (mu_a[col + i * ni] - mu_s[col]) *
                   (mu_a[col + i * ni] - mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (batch_size - 1);
    }
}

__global__ void batchnorm_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, bool bias, float epsilon, int ni,
    int batch_size, float *mu_z, float *var_z)
/*Compute pmean of product WA of batch-normalization layer.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[col] + epsilon);
        int idx = col + row * ni;
        float mu_a_tilde = mu_a[idx] - mu_ra[col];

        float tmp_mu_z = inv_sqrt_var_ra * mu_a_tilde * mu_w[col];
        float tmp_var_z = inv_sqrt_var_ra * inv_sqrt_var_ra *
                          (var_a[idx] * (mu_w[col] * mu_w[col] + var_w[col]) +
                           var_w[col] * mu_a_tilde * mu_a_tilde);

        mu_z[idx] = bias ? tmp_mu_z + mu_b[col] : tmp_mu_z;
        var_z[idx] = bias ? tmp_var_z + var_b[col] : tmp_var_z;
    }
}

template <int BLOCK_TILE_X, int BLOCK_TILE_Y>
__global__ void batchnorm2d_dual_sum_reduction(float const *delta_mu_e,
                                               float const *delta_var_e,
                                               int wihi, int fi, int batch_size,
                                               float *delta_mu,
                                               float *delta_var) {
    __shared__ float smem_mu[BLOCK_TILE_Y * BLOCK_TILE_X];
    __shared__ float smem_var[BLOCK_TILE_Y * BLOCK_TILE_X];

    const size_t col = blockIdx.x * BLOCK_TILE_X + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;

    const size_t idx = row * wihi + (col / wihi) * wihi * fi + col % wihi;

    if (col < wihi * batch_size && row < fi) {
        smem_mu[ty * BLOCK_TILE_X + tx] = delta_mu_e[idx];
        smem_var[ty * BLOCK_TILE_X + tx] = delta_var_e[idx];
    } else {
        smem_mu[ty * BLOCK_TILE_X + tx] = 0.0f;
        smem_var[ty * BLOCK_TILE_X + tx] = 0.0f;
    }

    __syncthreads();
    for (size_t i = BLOCK_TILE_X / 2; i > WARP_SIZE; i >>= 1) {
        if (tx < i) {
            smem_mu[ty * BLOCK_TILE_X + tx] +=
                smem_mu[ty * BLOCK_TILE_X + tx + i];
            smem_var[ty * BLOCK_TILE_X + tx] +=
                smem_var[ty * BLOCK_TILE_X + tx + i];
        }
        __syncthreads();
    }

    if (tx < WARP_SIZE) {
        dual_warp_smem_reduction(smem_mu, smem_var, tx, ty, BLOCK_TILE_X);
    }
    if (tx == 0 && row < fi) {
        delta_mu[row * gridDim.x + blockIdx.x] =
            smem_mu[ty * BLOCK_TILE_X + tx];
        delta_var[row * gridDim.x + blockIdx.x] =
            smem_var[ty * BLOCK_TILE_X + tx];
    }
}

template <int BLOCK_TILE_X, int BLOCK_TILE_Y>
__global__ void batchnorm2d_sample_sum_reduction(float const *sample,
                                                 float const *mu_s, int wihi,
                                                 int fi, int batch_size,
                                                 float *var) {
    __shared__ float smem_mu[BLOCK_TILE_Y * BLOCK_TILE_X];

    const size_t col = blockIdx.x * BLOCK_TILE_X + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;

    const size_t idx = row * wihi + (col / wihi) * wihi * fi + col % wihi;

    if (col < wihi * batch_size && row < fi) {
        float diff = sample[idx] - mu_s[row];
        smem_mu[ty * BLOCK_TILE_X + tx] = __fmul_rn(diff, diff);

    } else {
        smem_mu[ty * BLOCK_TILE_X + tx] = 0.0f;
    }

    __syncthreads();

    for (size_t i = BLOCK_TILE_X / 2; i > WARP_SIZE; i >>= 1) {
        if (tx < i) {
            smem_mu[ty * BLOCK_TILE_X + tx] +=
                smem_mu[ty * BLOCK_TILE_X + tx + i];
        }
        __syncthreads();
    }

    if (tx < WARP_SIZE) {
        warp_smem_reduction(smem_mu, tx, ty, BLOCK_TILE_X);
    }
    if (tx == 0 && row < fi) {
        var[row * gridDim.x + blockIdx.x] = smem_mu[ty * BLOCK_TILE_X + tx];
    }
}

void batchnorm2d_fwd_sum_reduction(float *&sample, float *&mu_s, int batch_size,
                                   int wihi, int fi, float *&buf_mu_in,
                                   float *&buf_mu_out, float *&mu_out)
/*
 */
{
    // TODO: remove this hard code
    constexpr unsigned int BLOCK_SIZE_X = 64;
    constexpr unsigned int BLOCK_SIZE_Y = 16;
    const dim3 block_dim_rd(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1U);
    unsigned int grid_size_y = (fi + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    unsigned int grid_size_x =
        (batch_size * wihi + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid_dim_rd(grid_size_x, grid_size_y, 1U);
    size_t reduced_size = grid_size_x;

    // Stage 1: Perform custom sum reduction
    batchnorm2d_sample_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_rd, block_dim_rd>>>(sample, mu_s, wihi, fi, batch_size,
                                        buf_mu_out);

    // Stage 2: Perform recursive reduction sum
    while (grid_size_x > BLOCK_SIZE_X) {
        grid_size_x = (grid_size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
        grid_dim_rd.x = grid_size_x;
        sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
            <<<grid_dim_rd, block_dim_rd>>>(buf_mu_out, reduced_size, fi,
                                            buf_mu_in);

        // Swap the buffers
        std::swap(buf_mu_out, buf_mu_in);

        reduced_size = grid_size_x;
    }

    // Stage 3: Perform the final reduction
    dim3 grid_dim_1b(1, grid_size_y);
    sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_1b, block_dim_rd>>>(buf_mu_out, reduced_size, fi, mu_out);
}

void batchnorm2d_fwd_dual_sum_reduction(float *&mu_in, float *&var_in,
                                        int batch_size, int wihi, int fi,
                                        float *&buf_mu_in, float *&buf_var_in,
                                        float *&buf_mu_out, float *&buf_var_out,
                                        float *&mu_out, float *&var_out)
/*
 */
{
    // TODO: remove this hard code
    constexpr unsigned int BLOCK_SIZE_X = 64U;
    constexpr unsigned int BLOCK_SIZE_Y = 16U;
    const dim3 block_dim_rd(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1U);
    unsigned int grid_size_y = (fi + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    unsigned int grid_size_x =
        (batch_size * wihi + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid_dim_rd(grid_size_x, grid_size_y, 1U);
    size_t reduced_size = grid_size_x;

    // Stage 1: Perform custom sum reduction
    batchnorm2d_dual_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_rd, block_dim_rd>>>(mu_in, var_in, wihi, fi, batch_size,
                                        buf_mu_out, buf_var_out);

    // Stage 2: Perform recursive reduction sum
    while (grid_size_x > BLOCK_SIZE_X) {
        grid_size_x = (grid_size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
        grid_dim_rd.x = grid_size_x;
        dual_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
            <<<grid_dim_rd, block_dim_rd>>>(buf_mu_out, buf_var_out,
                                            reduced_size, fi, buf_mu_in,
                                            buf_var_in);

        // Swap the buffers
        std::swap(buf_mu_out, buf_mu_in);
        std::swap(buf_var_out, buf_var_in);

        reduced_size = grid_size_x;
    }

    // Stage 3: Perform the final reduction
    dim3 grid_dim_1b(1, grid_size_y);
    dual_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_1b, block_dim_rd>>>(buf_mu_out, buf_var_out, reduced_size,
                                        fi, mu_out, var_out);
}

void batchnorm2d_bwd_dual_sum_reduction(int batch_size, int wihi, int fi,
                                        float *&buf_mu_in, float *&buf_var_in,
                                        float *&buf_mu_out, float *&buf_var_out,
                                        float *&delta_mu, float *&delta_var)
/*
 */
{
    // TODO: remove this hard code
    constexpr unsigned BLOCK_SIZE_X = 64;
    constexpr unsigned BLOCK_SIZE_Y = 16;
    const dim3 block_dim_rd(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1U);
    unsigned int grid_size_y =
        (static_cast<unsigned int>(fi) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    unsigned int grid_size_x =
        (static_cast<unsigned int>(batch_size * wihi) + BLOCK_SIZE_X - 1) /
        BLOCK_SIZE_X;
    dim3 grid_dim_rd(grid_size_x, grid_size_y, 1U);
    size_t reduced_size = grid_size_x;

    // Stage 1: Perform custom sum reduction
    batchnorm2d_dual_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_rd, block_dim_rd>>>(buf_mu_in, buf_var_in, wihi, fi,
                                        batch_size, buf_mu_out, buf_var_out);

    // Stage 2: Perform recursive reduction sum
    while (grid_size_x > BLOCK_SIZE_X) {
        grid_size_x = (grid_size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
        grid_dim_rd.x = grid_size_x;
        dual_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
            <<<grid_dim_rd, block_dim_rd>>>(buf_mu_out, buf_var_out,
                                            reduced_size, fi, buf_mu_in,
                                            buf_var_in);

        // Swap the buffers
        std::swap(buf_mu_out, buf_mu_in);
        std::swap(buf_var_out, buf_var_in);

        reduced_size = grid_size_x;
    }

    // Stage 3: Perform the final reduction
    dim3 grid_dim_1b(1, grid_size_y);
    dual_sum_reduction<BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_1b, block_dim_rd>>>(buf_mu_out, buf_var_out, reduced_size,
                                        fi, delta_mu, delta_var);
}

__global__ void batchnorm2d_stat_mean_var_cuda(float const *mu_a,
                                               float const *var_a, int wihi,
                                               int fi, int batch_size,
                                               float *mu_s, float *var_s)
/*Compute sample mean and variance of activation units for batch-normalization
layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0;
    float sum_var = 0;
    if (col < fi) {
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sum_var += var_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        mu_s[col] = sum_mu / (wihi * batch_size);
        var_s[col] = sum_var;
    }
}

__global__ void batchnorm2d_sample_var_cuda(float const *mu_a,
                                            float const *mu_s,
                                            float const *var_s, int wihi,
                                            int fi, int batch_size, float *var)
/*Compute statistical mean and variance of activation units for
batch-normalization layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < fi) {
        for (int i = 0; i < wihi * batch_size; i++) {
            sum += (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]) *
                   (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (wihi * batch_size - 1);
    }
}

__global__ void batchnorm2d_sample_mu_post_processing(float const *data_in,
                                                      int fi, float scale,
                                                      float *data_out)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < fi) {
        data_out[col] = data_in[col] / scale;
    }
}

__global__ void batchnorm2d_sample_var_post_processing(float const *data_in,
                                                       float const *bias,
                                                       int fi, float scale,
                                                       float *data_out)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < fi) {
        data_out[col] = (data_in[col] + bias[col]) / scale;
    }
}

__global__ void batchnorm2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, bool bias, float epsilon, int wihi,
    int fi, int m, float *mu_z, float *var_z)
/*Compute mean of product WA of batch-normalization. Note that the previous
layer is a convolutional layer.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        int div_idx = row % fi;
        float inv_var_ra = 1.0f / (var_ra[div_idx] + epsilon);
        float inv_var_ra_sqrt = sqrtf(inv_var_ra);

        int idx = col + row * k;

        float tmp_mu_a = mu_a[idx];
        float tmp_var_a = var_a[idx];
        float tmp_mu_w = mu_w[div_idx];
        float tmp_mu_w_2 = tmp_mu_w * tmp_mu_w;
        float tmp_mu_ra = mu_ra[div_idx];
        float tmp_mu_a_tilde = tmp_mu_a - tmp_mu_ra;

        float tmp_mu_z = inv_var_ra_sqrt * tmp_mu_a_tilde * tmp_mu_w;

        float tmp_var_z =
            inv_var_ra * (tmp_var_a * (tmp_mu_w_2 + var_w[div_idx]) +
                          var_w[div_idx] * tmp_mu_a_tilde * tmp_mu_a_tilde);

        // OLD FORMULATION
        // float tmp_var_z =
        //     inv_var_ra * (tmp_var_a * tmp_mu_w_2 +
        //                   var_w[div_idx] * (tmp_mu_a * tmp_mu_a -
        //                                     tmp_mu_ra * tmp_mu_ra +
        //                                     tmp_var_a));

        mu_z[idx] = bias ? tmp_mu_z + mu_b[div_idx] : tmp_mu_z;
        var_z[idx] = bias ? tmp_var_z + var_b[div_idx] : tmp_var_z;
    }
}

__global__ void batchnorm_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is full-connected layer.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float inv_var_hat = 1.0f / (var_hat[col] + epsilon);
        float inv_var_hat_sqrt = sqrtf(inv_var_hat);
        float tmp = mu_w[col] * jcb[col + row * ni];

        delta_mu[col + row * ni] =
            tmp * delta_mu_out[col + row * ni] * inv_var_hat_sqrt;
        delta_var[col + row * ni] =
            tmp * delta_var_out[col + row * ni] * tmp * inv_var_hat;
    }
}

__global__ void batchnorm2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int wihi, int fi, int m, float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is convolutional layer.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi * fi, m = B
    {
        float inv_var_hat = 1.0f / (var_hat[row % fi] + epsilon);
        float inv_var_hat_sqrt = sqrtf(inv_var_hat);
        float tmp = mu_w[row % fi] * jcb[col + row * wihi];

        delta_mu[col + row * wihi] =
            tmp * delta_mu_out[col + row * wihi] * inv_var_hat_sqrt;

        delta_var[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp * inv_var_hat;
    }
}

__global__ void batchnorm_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_ra,
    float const *var_ra, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int ni, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to full-connected layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < batch_size; i++) {
            float inv_var_ra = 1.0f / (var_ra[col] + epsilon);
            float inv_var_ra_sqrt = sqrtf(inv_var_ra);

            float tmp = (mu_a[col + i * ni] - mu_ra[col]) * var_w[col];
            sum_mu += tmp * delta_mu_out[col + i * ni] * inv_var_ra_sqrt;
            sum_var += tmp * delta_var_out[col + i * ni] * tmp * inv_var_ra;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

__global__ void batchnorm_bwd_delta_b_cuda(float const *var_b,
                                           float const *delta_mu_out,
                                           float const *delta_var_out,
                                           float epsilon, int ni,
                                           int batch_size, float *delta_mu_b,
                                           float *delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to full-connected layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float tmp = var_b[col];
            sum_mu += tmp * delta_mu_out[col + i * ni];
            sum_var += tmp * delta_var_out[col + i * ni] * tmp;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

__global__ void batchnorm2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_ra,
    float const *var_ra, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int wihi, int fi, int m, float *delta_mu_w,
    float *delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to convolutional layer.
*/
// TODO: remove the duplicates
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi, m = fi*B
    {
        float inv_var_ra = 1.0f / (var_ra[row % fi] + epsilon);
        float inv_var_ra_sqrt = sqrtf(inv_var_ra);
        float tmp =
            (mu_a[col + row * wihi] - mu_ra[row % fi]) * var_w[row % fi];

        delta_mu_w[col + row * wihi] =
            tmp * delta_mu_out[col + row * wihi] * inv_var_ra_sqrt;
        delta_var_w[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp * inv_var_ra;
    }
}

__global__ void batchnorm2d_bwd_delta_b_cuda(float const *var_b,
                                             float const *delta_mu_out,
                                             float const *delta_var_out,
                                             float epsilon, int wihi, int fi,
                                             int m, float *delta_mu_b,
                                             float *delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to convolutional layer.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi, m = fi*B
    {
        float tmp = var_b[row % fi];

        delta_mu_b[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
        delta_var_b[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp;
    }
}

////////////////////////////////////////////////////////////////////////////////
//// Batch Norm
////////////////////////////////////////////////////////////////////////////////
BatchNorm2dCuda::BatchNorm2dCuda(int num_features, float eps, float momentum,
                                 bool bias, float gain_weight, float gain_bias)
    : num_features(num_features),
      epsilon(eps),
      momentum(momentum),
      gain_w(gain_weight),
      gain_b(gain_bias)
/*
 */
{
    this->bias = bias;
    this->init_weight_bias();
    this->allocate_running_mean_var();
    if (this->training) {
        this->allocate_param_delta();
    }
}

BatchNorm2dCuda::~BatchNorm2dCuda()
/*
 */
{
    cudaFree(d_mu_ra);
    cudaFree(d_var_ra);
}

std::string BatchNorm2dCuda::get_layer_info() const
/*
 */
{
    return "BatchNorm2d()";
}

std::string BatchNorm2dCuda::get_layer_name() const
/*
 */
{
    return "BatchNorm2dCuda";
}

LayerType BatchNorm2dCuda::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void BatchNorm2dCuda::init_weight_bias()
/*
 */
{
    this->num_weights = this->num_features;
    this->num_biases = this->bias ? this->num_features : 0;
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_norm("", this->gain_w, this->gain_b,
                              this->num_features, this->num_features,
                              this->num_weights, this->num_biases);

    this->allocate_param_memory();
    this->params_to_device();
}

void BatchNorm2dCuda::deallocate_running_mean_var() {
    cudaFree(this->d_mu_ra);
    cudaFree(this->d_var_ra);
    cudaFree(this->d_mu_norm_batch);
    cudaFree(this->d_var_norm_batch);
    this->d_mu_ra = nullptr;
    this->d_var_ra = nullptr;
    this->d_mu_norm_batch = nullptr;
    this->d_var_norm_batch = nullptr;
}

void BatchNorm2dCuda::allocate_running_mean_var()
/*
 */
{
    this->mu_ra.resize(this->num_features, 0.0f);
    this->var_ra.resize(this->num_features, 1.0f);
    this->mu_norm_batch.resize(this->num_features, 0.0f);
    this->var_norm_batch.resize(this->num_features, 0.0f);

    // Memory aligment
    unsigned int size_num_features =
        ((this->num_features + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    cudaMalloc(&this->d_mu_ra, size_num_features * sizeof(float));
    cudaMalloc(&this->d_var_ra, size_num_features * sizeof(float));
    cudaMalloc(&this->d_mu_norm_batch, size_num_features * sizeof(float));
    cudaMalloc(&this->d_var_norm_batch, size_num_features * sizeof(float));

    CHECK_LAST_CUDA_ERROR();
    this->running_mean_var_to_device();
}

void BatchNorm2dCuda::running_mean_var_to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_ra, this->mu_ra.data(),
               this->mu_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_ra, this->var_ra.data(),
               this->var_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_norm_batch, this->mu_norm_batch.data(),
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_norm_batch, this->var_norm_batch.data(),
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();
}

void BatchNorm2dCuda::running_mean_var_to_host()
/*
 */
{
    cudaMemcpy(this->mu_ra.data(), this->d_mu_ra,
               this->mu_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ra.data(), this->d_var_ra,
               this->var_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(this->mu_norm_batch.data(), this->d_mu_norm_batch,
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_norm_batch.data(), this->d_var_norm_batch,
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}

void BatchNorm2dCuda::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    if (this->input_size == 0 || this->output_size == 0) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }
    float _momentum = this->momentum;

    // if (this->first_batch) {
    //     if (this->training) {
    //         _momentum = 0.0f;
    //     }
    //     this->first_batch = false;
    // }

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    auto d_mu_target = this->training ? this->d_mu_norm_batch : this->d_mu_ra;
    auto d_var_target =
        this->training ? this->d_var_norm_batch : this->d_var_ra;

    if (this->num_features != this->in_channels) {
        unsigned int grid_size_ra =
            (this->input_size + num_threads - 1) / num_threads;

        if (this->training) {
            batchnorm_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                cu_input_states->d_mu_a, cu_input_states->d_var_a,
                this->input_size, batch_size, this->d_mu_norm_batch,
                cu_temp_states->d_tmp_2);

            batchnorm_sample_var_cuda<<<grid_size_ra, num_threads>>>(
                cu_input_states->d_mu_a, this->d_mu_norm_batch,
                cu_temp_states->d_tmp_2, this->input_size, batch_size,
                this->d_var_norm_batch);

            if (this->training) {
                running_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                    this->d_mu_norm_batch, this->d_var_norm_batch, _momentum,
                    this->input_size, this->d_mu_ra, this->d_var_ra);
            }
        }
        unsigned int grid_col =
            (this->input_size + num_threads - 1) / num_threads;
        unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, d_mu_target,
            d_var_target, this->bias, this->epsilon, this->input_size,
            batch_size, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        int wihi = this->in_height * this->in_width;
        unsigned int grid_size_ra =
            (this->in_channels + num_threads - 1) / num_threads;

        // Local pointer for swapping. Leverage the existing and
        // not-yet-used memory block defined in GPU device to reduce the
        // memory allocation
        float *buf_mu_out = cu_output_states->d_mu_a;
        float *buf_var_out = cu_output_states->d_var_a;
        float *buf_mu_in = cu_temp_states->d_tmp_1;
        float *buf_var_in = cu_temp_states->d_tmp_2;
        if (this->training) {
            // OLD KERNELS
            // batchnorm2d_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
            //     cu_input_states->d_mu_a, cu_input_states->d_var_a, wihi,
            //     this->in_channels, batch_size, this->d_mu_norm_batch,
            //     cu_temp_states->d_tmp_2);

            // batchnorm2d_sample_var_cuda<<<grid_size_ra, num_threads>>>(
            //     cu_input_states->d_mu_a, this->d_mu_norm_batch,
            //     cu_temp_states->d_tmp_2, wihi, this->in_channels, batch_size,
            //     this->d_var_norm_batch);

            // Calculate  sum_val = \sum (samples)
            batchnorm2d_fwd_dual_sum_reduction(
                cu_input_states->d_mu_a, cu_input_states->d_var_a, batch_size,
                wihi, this->in_channels, buf_mu_in, buf_var_in, buf_mu_out,
                buf_var_out, this->d_mu_norm_batch, this->d_var_norm_batch);

            // Calculate mean, mu_val = sum_val / (wihi * batch_size)
            float scale = wihi * batch_size;
            batchnorm2d_sample_mu_post_processing<<<grid_size_ra,
                                                    num_threads>>>(
                this->d_mu_norm_batch, this->in_channels, scale,
                this->d_mu_norm_batch);

            // Calculate variance, stat_var = (sample - mu)^2
            batchnorm2d_fwd_sum_reduction(cu_input_states->d_mu_a,
                                          this->d_mu_norm_batch, batch_size,
                                          wihi, this->in_channels, buf_mu_in,
                                          buf_mu_out, cu_temp_states->d_tmp_2);

            // Statistical sample variance, var = (sum_val + stat var) / (wihi *
            // batch_size)
            // scale = scale - 1.0f;
            batchnorm2d_sample_var_post_processing<<<grid_size_ra,
                                                     num_threads>>>(
                this->d_var_norm_batch, cu_temp_states->d_tmp_2,
                this->in_channels, scale, this->d_var_norm_batch);

            running_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                this->d_mu_norm_batch, this->d_var_norm_batch, _momentum,
                this->in_channels, this->d_mu_ra, this->d_var_ra);
        }

        int fi_batch = this->in_channels * batch_size;
        unsigned int grid_row = (fi_batch + num_threads - 1) / num_threads;
        unsigned int grid_col = (wihi + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm2d_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, d_mu_target,
            d_var_target, this->bias, this->epsilon, wihi, this->in_channels,
            fi_batch, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void BatchNorm2dCuda::backward(BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_delta_states,
                               BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int batch_size = cu_input_delta_states->block_size;
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    if (param_update) {
        TempStateCuda *cu_temp_states =
            dynamic_cast<TempStateCuda *>(&temp_states);

        if (this->in_channels == 0) {
            unsigned int grid_size_p =
                (this->input_size + num_threads - 1) / num_threads;

            batchnorm_bwd_delta_w_cuda<<<grid_size_p, num_threads>>>(
                this->d_var_w, cu_next_bwd_states->d_mu_a,
                this->d_mu_norm_batch, this->d_var_norm_batch,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon,
                this->input_size, batch_size, this->d_delta_mu_w,
                this->d_delta_var_w);

            if (this->bias) {
                batchnorm_bwd_delta_b_cuda<<<grid_size_p, num_threads>>>(
                    this->d_var_b, cu_input_delta_states->d_delta_mu,
                    cu_input_delta_states->d_delta_var, this->epsilon,
                    this->input_size, batch_size, this->d_delta_mu_b,
                    this->d_delta_var_b);
            }

        } else {
            int wihi = this->in_width * this->in_height;
            int fi_batch = this->in_channels * batch_size;

            unsigned int grid_row_p =
                (fi_batch + num_threads - 1) / num_threads;
            unsigned int grid_col_p = (wihi + num_threads - 1) / num_threads;
            dim3 dim_grid_p(grid_col_p, grid_row_p);

            batchnorm2d_bwd_delta_w_cuda<<<dim_grid_p, block_dim>>>(
                this->d_var_w, cu_next_bwd_states->d_mu_a,
                this->d_mu_norm_batch, this->d_var_norm_batch,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon, wihi,
                this->in_channels, fi_batch, cu_temp_states->d_tmp_1,
                cu_temp_states->d_tmp_2);

            // Local pointer for swapping. Leverage the existing and
            // not-yet-used memory blocks defined in GPU device to reduce the
            // memory allocation
            float *buf_mu_out = cu_output_delta_states->d_delta_mu;
            float *buf_var_out = cu_output_delta_states->d_delta_var;
            float *buf_mu_in = cu_temp_states->d_tmp_1;
            float *buf_var_in = cu_temp_states->d_tmp_2;

            batchnorm2d_bwd_dual_sum_reduction(
                batch_size, wihi, this->in_channels, buf_mu_in, buf_var_in,
                buf_mu_out, buf_var_out, this->d_delta_mu_w,
                this->d_delta_var_w);

            if (this->bias) {
                batchnorm2d_bwd_delta_b_cuda<<<dim_grid_p, block_dim>>>(
                    this->d_var_b, cu_input_delta_states->d_delta_mu,
                    cu_input_delta_states->d_delta_var, this->epsilon, wihi,
                    this->in_channels, fi_batch, buf_mu_in, buf_var_in);

                batchnorm2d_bwd_dual_sum_reduction(
                    batch_size, wihi, this->in_channels, buf_mu_in, buf_var_in,
                    buf_mu_out, buf_var_out, this->d_delta_mu_b,
                    this->d_delta_var_b);
            }
        }
    }
    if (state_udapte) {
        if (this->in_channels == 0) {
            unsigned int grid_row =
                (batch_size + num_threads - 1) / num_threads;
            unsigned int grid_col =
                (this->input_size + num_threads - 1) / num_threads;
            dim3 grid_size(grid_col, grid_row);

            batchnorm_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
                this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_norm_batch,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon,
                this->input_size, batch_size,
                cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);

        } else {
            int fi_batch = this->in_channels * batch_size;
            int wihi = this->in_width * this->in_height;

            unsigned int grid_row = (fi_batch + num_threads - 1) / num_threads;
            unsigned int grid_col = (wihi + num_threads - 1) / num_threads;
            dim3 grid_size(grid_col, grid_row);

            batchnorm2d_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
                this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_norm_batch,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon, wihi,
                this->in_channels, fi_batch, cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
        }
    }
}

std::unique_ptr<BaseLayer> BatchNorm2dCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<BatchNorm2d>(
        this->num_features, this->epsilon, this->momentum, this->bias,
        this->gain_w, this->gain_b);

    host_layer->mu_w = this->mu_w;
    host_layer->var_w = this->var_w;
    host_layer->mu_b = this->mu_b;
    host_layer->var_b = this->var_b;

    return host_layer;
}

void BatchNorm2dCuda::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }
    // Transfer data to host
    this->params_to_host();
    this->running_mean_var_to_host();

    // Save the name length and name
    auto layer_name = this->get_layer_info();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (const auto &m_ra : this->mu_ra) {
        file.write(reinterpret_cast<const char *>(&m_ra), sizeof(m_ra));
    }
    for (const auto &v_ra : this->var_ra) {
        file.write(reinterpret_cast<const char *>(&v_ra), sizeof(v_ra));
    }
}

void BatchNorm2dCuda::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_info();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (auto &m_ra : this->mu_ra) {
        file.read(reinterpret_cast<char *>(&m_ra), sizeof(m_ra));
    }
    for (auto &v_ra : this->var_ra) {
        file.read(reinterpret_cast<char *>(&v_ra), sizeof(v_ra));
    }

    this->num_weights = this->mu_w.size();
    this->num_biases = this->mu_b.size();
    if (this->training) {
        this->allocate_param_delta();
    }

    // It wont set momentum to zero for running average of norm's mean & var
    this->first_batch = false;

    // Transfer data to device
    this->params_to_device();
    this->running_mean_var_to_device();
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
BatchNorm2dCuda::get_norm_mean_var() {
    this->running_mean_var_to_host();
    std::vector<std::vector<float>> mu_ras = {this->mu_ra};
    std::vector<std::vector<float>> var_ras = {this->var_ra};
    std::vector<std::vector<float>> mu_norms = {this->mu_norm_batch};
    std::vector<std::vector<float>> var_norms = {this->var_norm_batch};

    return std::make_tuple(mu_ras, var_ras, mu_norms, var_norms);
}
