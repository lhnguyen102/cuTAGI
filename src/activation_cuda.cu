#include "../include/activation.h"
#include "../include/activation_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        float tmp = fmaxf(mu_z[col], 0.0f);
        mu_a[col] = tmp;

        bool is_zero = (tmp == 0.0f);
        jcb[col] = is_zero ? 0.0f : 1.0f;
        var_a[col] = is_zero ? 0.0f : var_z[col];
    }
}

__global__ void relu_mean_var_cuda_vectorized(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx < num_states) {
        float4 mu_z_vec, var_z_vec, mu_a_vec, jcb_vec, var_a_vec;

        // Load 4 float values into float4 vectors
        mu_z_vec.x = mu_z[vec_idx];
        mu_z_vec.y = vec_idx + 1 < num_states ? mu_z[vec_idx + 1] : 0.0f;
        mu_z_vec.z = vec_idx + 2 < num_states ? mu_z[vec_idx + 2] : 0.0f;
        mu_z_vec.w = vec_idx + 3 < num_states ? mu_z[vec_idx + 3] : 0.0f;

        var_z_vec.x = var_z[vec_idx];
        var_z_vec.y = vec_idx + 1 < num_states ? var_z[vec_idx + 1] : 0.0f;
        var_z_vec.z = vec_idx + 2 < num_states ? var_z[vec_idx + 2] : 0.0f;
        var_z_vec.w = vec_idx + 3 < num_states ? var_z[vec_idx + 3] : 0.0f;

        // Process the data
        mu_a_vec.x = fmaxf(mu_z_vec.x, 0.0f);
        mu_a_vec.y = fmaxf(mu_z_vec.y, 0.0f);
        mu_a_vec.z = fmaxf(mu_z_vec.z, 0.0f);
        mu_a_vec.w = fmaxf(mu_z_vec.w, 0.0f);

        jcb_vec.x = (mu_a_vec.x == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.y = (mu_a_vec.y == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.z = (mu_a_vec.z == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.w = (mu_a_vec.w == 0.0f) ? 0.0f : 1.0f;

        var_a_vec.x = (mu_a_vec.x == 0.0f) ? 0.0f : var_z_vec.x;
        var_a_vec.y = (mu_a_vec.y == 0.0f) ? 0.0f : var_z_vec.y;
        var_a_vec.z = (mu_a_vec.z == 0.0f) ? 0.0f : var_z_vec.z;
        var_a_vec.w = (mu_a_vec.w == 0.0f) ? 0.0f : var_z_vec.w;

        // Store the results back as individual floats
        mu_a[vec_idx] = mu_a_vec.x;
        jcb[vec_idx] = jcb_vec.x;
        var_a[vec_idx] = var_a_vec.x;

        if (vec_idx + 1 < num_states) {
            mu_a[vec_idx + 1] = mu_a_vec.y;
            jcb[vec_idx + 1] = jcb_vec.y;
            var_a[vec_idx + 1] = var_a_vec.y;
        }

        if (vec_idx + 2 < num_states) {
            mu_a[vec_idx + 2] = mu_a_vec.z;
            jcb[vec_idx + 2] = jcb_vec.z;
            var_a[vec_idx + 2] = var_a_vec.z;
        }

        if (vec_idx + 3 < num_states) {
            mu_a[vec_idx + 3] = mu_a_vec.w;
            jcb[vec_idx + 3] = jcb_vec.w;
            var_a[vec_idx + 3] = var_a_vec.w;
        }
    }
}

__global__ void sigmoid_mean_var_cuda(float const *mu_z, float const *var_z,
                                      int num_states, float *mu_a, float *jcb,
                                      float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;

    if (col < num_states) {
        tmp = 1.0f / (1.0f + expf(-mu_z[col]));
        mu_a[col] = tmp;
        jcb[col] = tmp * (1.0f - tmp);
        var_a[col] = tmp * (1.0f - tmp) * var_z[col] * tmp * (1.0f - tmp);
    }
}

__global__ void tanh_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = tanhf(mu_z[col]);
        float tmp_2 = tmp * tmp;
        mu_a[col] = tmp;
        jcb[col] = (1.0f - tmp_2);
        var_a[col] = (1.0f - tmp_2) * var_z[col] * (1.0f - tmp_2);
    }

    // printf("mean_var_cuda: mu_z[%d]=%f, var_z[%d]=%f\n", col, mu_z[col],
    //            col, var_z[col]);

    //     float delta = 20.0f;
    //     float sign = (mu_z[col] >= 0.0f) ? 1.0f : -1.0f;

    //     if (std::abs(mu_z[col]) > delta) {
    //         mu_a[col] = sign * delta;
    //         jcb[col] = 0.0f;
    //         var_a[col] = 0.0f;
    //     } else {
    //         mu_a[col] = mu_z[col];
    //         var_a[col] = var_z[col];
    //         jcb[col] = 1.0f;
    //     }
}

__device__ float normcdf_cuda(float x)
/*
Normal cumulative distribution function
 */
{
    return 0.5f * erfcf(-x * 0.7071067811865475f);
}

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float SQRT_2PI = 2.5066282746310002f;
    if (col < num_states) {
        // Reused components for moments calculations
        float tmp_mu_z = mu_z[col];
        float std_z = powf(var_z[col], 0.5);
        float alpha = tmp_mu_z / std_z;
        float pdf_alpha = (1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha);
        float cdf_alpha = normcdf_cuda(alpha);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = mu_z[col] * cdf_alpha + std_z * pdf_alpha;
        mu_a[col] = fmaxf(0.000001f, tmp_mu_a);
        float tmp_var_a = fmaxf(
            0.000001f, -tmp_mu_a * tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z -
                           tmp_mu_z * std_z * pdf_alpha +
                           (var_z[col] - tmp_mu_z * tmp_mu_z) * cdf_alpha);
        var_a[col] = tmp_var_a;
        jcb[col] = cdf_alpha;
    }
}

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    constexpr float SQRT_2PI = 2.5066282746310002f;

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[col], 0.5);
        alpha_l = (1.0f + mu_z[col]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[col]) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_z = mu_z[col];
        float tmp_mu_z_2 = tmp_mu_z * tmp_mu_z;
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = fmaxf(0.0000001f, tmp_mu_a);
        var_a[col] =
            max(0.0000001f,
                (cdf_l * (var_z[col] - tmp_mu_z_2 - 2 * tmp_mu_z - 1) +
                 cdf_u * (var_z[col] - tmp_mu_z_2 + 2 * tmp_mu_z - 1) +
                 std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                 tmp_mu_a * tmp_mu_a + 2 * mu_a[col] * tmp_mu_z +
                 tmp_mu_z * tmp_mu_z - var_z[col] + 2) /
                    4.0f);
        mu_a[col] = tmp_mu_a / 2.0f + 0.5f;
        jcb[col] = (cdf_u + cdf_l - 1) / 2.0f;
    }
}

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    constexpr float SQRT_2PI = 2.5066282746310002f;

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        float tmp_mu_z = mu_z[col];
        std_z = powf(var_z[col], 0.5);
        alpha_l = (1.0f + tmp_mu_z) / std_z;  // Lower truncation
        alpha_u = (1.0f - tmp_mu_z) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = tmp_mu_a;
        var_a[col] = max(
            0.000001f,
            cdf_l * (var_z[col] - tmp_mu_z * tmp_mu_z - 2 * tmp_mu_z - 1) +
                cdf_u * (var_z[col] - tmp_mu_z * tmp_mu_z + 2 * tmp_mu_z - 1) +
                std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z + tmp_mu_z - var_z[col] + 2);

        jcb[col] = cdf_u + cdf_l - 1;
    }
}

__global__ void celu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float SQRT_2PI = 2.5066282746310002;
    constexpr float INV_SQRT2 = 0.7071067811865475;
    constexpr float ALPHA = 0.2;  // slope of negative part

    if (col < num_states) {
        // inside your kernel, per-element:
        float mz = mu_z[col];
        float varz = var_z[col];

        // 1) std-dev and standardize
        float sz = sqrt(varz);
        float z = mz / sz;

        // 2) shift amount
        float a = sz / ALPHA;
        float z_a = z + a;
        float z_2a = z + 2.0 * a;

        // 3) φ(z) and tail probs P[Z < −x] = 0.5*erfc(x/√2), clamped
        float phi_z = exp(-0.5 * z * z) / SQRT_2PI;

        float tail_z = 0.5 * erfc(z * INV_SQRT2);
        float tail_za = 0.5 * erfc(z_a * INV_SQRT2);
        float tail_z2a = 0.5 * erfc(z_2a * INV_SQRT2);

        // 4) analytic ratios instead of φ(z)/φ(z+k·a)
        float exp_a = exp(a * z + 0.5 * (a * a));
        float exp_2a = exp(2 * a * z + 0.5 * (2 * a) * (2 * a));

        // 5) Mean E[CELU(z)]
        float mean_d =
            mz + sz * phi_z - (ALPHA + mz) * tail_z + ALPHA * tail_za * exp_a;

        // 6) Second moment E[CELU(z)²]
        float E2 = mz * mz + mz * sz * phi_z + varz -
                   2.0 * ALPHA * ALPHA * tail_za * exp_a +
                   ALPHA * ALPHA * tail_z2a * exp_2a +
                   (ALPHA * ALPHA - mz * mz - varz) * tail_z;

        // 7) Variance = E2 – mean², with floor
        float var_d = E2 - mean_d * mean_d;

        // 8) Covariance Cov[z, CELU(z)] = varz * (P[Z> -z] + tail_za·exp_a)
        float cov_d = varz * ((1.0 - tail_z) + tail_za * exp_a);

        // 9) Jacobian in [0,1]
        float jcb_d = cov_d / varz;
        // jcb_d = fmin(fmax(jcb_d, 0.0), 1.0);

        // 10) Store back (cast to float, clamp mean+α > 0)
        mu_a[col] = fmax(mean_d + ALPHA, 0.000001f);
        var_a[col] = fmaxf(var_d, 0.000001f);
        jcb[col] = jcb_d;
    }
}

__global__ void softplus_mean_var_cuda(float const *mu_z, float const *var_z,
                                       int num_states, float *mu_a, float *jcb,
                                       float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < num_states) {
        mu_a[col] = logf(1 + expf(mu_z[col]));
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = tmp * var_z[col] * tmp;
    }
}

__global__ void leakyrelu_mean_var_cuda(float const *mu_z, float const *var_z,
                                        float alpha, int num_states,
                                        float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0.0f;
    float one_pad = 1.0f;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = max(mu_z[col], zero_pad);
        if (tmp == 0) {
            mu_a[col] = alpha * mu_z[col];
            jcb[col] = alpha;
            var_a[col] = alpha * var_z[col] * alpha;

        } else {
            mu_a[col] = tmp;
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

__global__ void softmax_mean_var_cuda(float const *mu_z, float *var_z,
                                      size_t output_size, int batch_size,
                                      float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float max_mu = mu_z[0];
    float max_var = var_z[0];

    for (int j = 1; j < output_size; j++) {
        if (mu_z[j + i * output_size] > max_mu) {
            max_mu = mu_z[j + i * output_size];
            max_var = var_z[j + i * output_size];
        }
    }

    float sum_mu = 0.0f;
    for (int j = 0; j < output_size; j++) {
        sum_mu += expf(mu_z[j + i * output_size] - max_mu);
    }

    float tmp_mu;
    for (int j = 0; j < output_size; j++) {
        tmp_mu = expf(mu_z[j + output_size * i] - max_mu) / sum_mu;

        mu_a[j + i * output_size] = tmp_mu;

        jcb[j + output_size * i] = tmp_mu * (1 - tmp_mu);

        var_a[j + output_size * i] = jcb[j + output_size * i] *
                                     (var_z[j + output_size * i] + max_var) *
                                     jcb[j + output_size * i];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Remax kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void to_log_cuda(float const *mu_m, float const *var_m,
                            int hidden_size, int batch_size, float *mu_log,
                            float *var_log)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < hidden_size && row < batch_size) {
        float tmp_var = logf(1.0f + (var_m[row * hidden_size + col] /
                                     powf(mu_m[row * hidden_size + col], 2)));
        float tmp_mu = logf(mu_m[row * hidden_size + col]) - 0.5 * tmp_var;

        mu_log[row * hidden_size + col] = tmp_mu;
        var_log[row * hidden_size + col] = tmp_var;
    }
}

__global__ void compute_mean_var_sum_cuda(float const *mu_m, float const *var_m,
                                          int hidden_size, int batch_size,
                                          float *mu_sum, float *var_sum)
/*
 */
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    if (row < batch_size) {
        for (int i = 0; i < hidden_size; i++) {
            sum_mu += mu_m[row * hidden_size + i];
            sum_var += var_m[row * hidden_size + i];
        }
    }
    mu_sum[row] = sum_mu;
    var_sum[row] = sum_var;
}

__global__ void compute_cov_log_m_mt_cuda(float const *mu_m, float const *var_m,
                                          float const *mu_mt, int hidden_size,
                                          int batch_size, float *cov_log_m_mt)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < hidden_size && row < batch_size) {
        cov_log_m_mt[row * hidden_size + col] =
            logf(1.0f + var_m[row * hidden_size + col] * (1.0f / mu_mt[row]) *
                            (1.0f / mu_m[row * hidden_size + col]));
    }
}

__global__ void compute_remax_mean_var_cuda(
    float const *mu_log_m, float const *var_log_m, float const *mu_log_mt,
    float const *var_log_mt, float const *cov_log_m_mt, int hidden_size,
    int batch_size, float *mu_a, float *var_a)
/*
 */
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size) {
        float sum_mu = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            float tmp_mu = mu_log_m[row * hidden_size + j] - mu_log_mt[row];
            float tmp_var = var_log_m[row * hidden_size + j] + var_log_mt[row] -
                            2 * cov_log_m_mt[row * hidden_size + j];

            mu_a[row * hidden_size + j] =
                fmaxf(0.000001f, expf(tmp_mu + 0.5f * tmp_var));
            sum_mu += mu_a[row * hidden_size + j];
            var_a[row * hidden_size + j] = expf(tmp_var) - 1.0f;
        }
        for (int j = 0; j < hidden_size; j++) {
            float tmp_mu_norm = mu_a[row * hidden_size + j] / sum_mu;
            mu_a[row * hidden_size + j] = tmp_mu_norm;
            var_a[row * hidden_size + j] *= tmp_mu_norm * tmp_mu_norm;
        }
    }
}

__global__ void compute_cov_a_z_cuda(float const *mu_a, float const *var_a,
                                     float const *var_z, float const *mu_m,
                                     float const *var_m, float const *var_log_m,
                                     float const *cov_log_m_mt,
                                     float const *cdfn, int hidden_size,
                                     int batch_size, float *cov_a_z)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < hidden_size && row < batch_size) {
        float cov_log_a_log_m = var_log_m[row * hidden_size + col] -
                                cov_log_m_mt[row * hidden_size + col];
        float cov_a_m = (expf(cov_log_a_log_m) - 1.0f) *
                        mu_a[row * hidden_size + col] *
                        mu_m[row * hidden_size + col];

        cov_a_z[row * hidden_size + col] =
            fminf(powf(var_a[row * hidden_size + col], 0.5f) *
                      powf(var_z[row * hidden_size + col], 0.5f),
                  cov_a_m / cdfn[row * hidden_size + col]);

        cov_a_z[row * hidden_size + col] /= var_z[row * hidden_size + col];
    }
}

__global__ void compute_cov_a_z_cuda_v2(float const *mu_a, float const *var_a,
                                        float const *var_z, float const *mu_m,
                                        float const *var_m,
                                        float const *var_log_m,
                                        float const *cov_log_m_mt,
                                        float const *cdfn, int hidden_size,
                                        int batch_size, float *cov_a_z)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < hidden_size && row < batch_size) {
        float cov_log_a_log_m = var_log_m[row * hidden_size + col] -
                                cov_log_m_mt[row * hidden_size + col];
        float cov_a_m = (expf(cov_log_a_log_m) - 1.0f) *
                        mu_a[row * hidden_size + col] *
                        mu_m[row * hidden_size + col];

        cov_a_z[row * hidden_size + col] = min(
            powf(var_a[row * hidden_size + col], 0.5f) *
                powf(var_z[row * hidden_size + col], 0.5f),
            cov_a_m * var_z[row * hidden_size + col] *
                cdfn[row * hidden_size + col] / var_m[row * hidden_size + col]);

        cov_a_z[row * hidden_size + col] /= var_z[row * hidden_size + col];
    }
}

__global__ void split_stream_kernel(const float *d_in_mu, const float *d_in_var,
                                    const float *d_in_jcb, int half_size,
                                    float *d_even_mu, float *d_even_var,
                                    float *d_even_jcb, float *d_odd_mu,
                                    float *d_odd_var, float *d_odd_jcb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half_size) {
        // Handle even index
        int even_idx = 2 * i;
        d_even_mu[i] = d_in_mu[even_idx];
        d_even_var[i] = d_in_var[even_idx];
        d_even_jcb[i] = d_in_jcb[even_idx];

        // Handle odd index
        int odd_idx = 2 * i + 1;
        d_odd_mu[i] = d_in_mu[odd_idx];
        d_odd_var[i] = d_in_var[odd_idx];
        d_odd_jcb[i] = d_in_jcb[odd_idx];
    }
}

__global__ void merge_stream_kernel(
    const float *d_even_mu, const float *d_even_var, const float *d_even_jcb,
    const float *d_odd_mu, const float *d_odd_var, const float *d_odd_jcb,
    int half_size, float *d_out_mu, float *d_out_var, float *d_out_jcb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half_size) {
        // Handle even index
        int even_idx = 2 * i;
        d_out_mu[even_idx] = d_even_mu[i];
        d_out_var[even_idx] = d_even_var[i];
        d_out_jcb[even_idx] = d_even_jcb[i];

        // Handle odd index
        int odd_idx = 2 * i + 1;
        d_out_mu[odd_idx] = d_odd_mu[i];
        d_out_var[odd_idx] = d_odd_var[i];
        d_out_jcb[odd_idx] = d_odd_jcb[i];
    }
}

__global__ void exp_mean_var_cuda(float const *mu_z, float const *var_z,
                                  float const *jcb_z, int num_states,
                                  float *mu_a, float *var_a, float *jcb_a,
                                  float scale, float shift)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        float new_var_z = var_z[col] * (scale * scale);
        // Calculate the new mean of the input z, and constrain it
        float new_mu_z = mu_z[col] * scale + shift;

        // Calculate the output mean using the constrained new_mu_z
        mu_a[col] = fmaxf(expf(new_mu_z + 0.5f * new_var_z), 0.0000001f);

        // Calculate the output variance using the constrained new_mu_z
        var_a[col] =
            fmaxf(expf(2.0f * new_mu_z + new_var_z) * (expf(new_var_z) - 1.0f),
                  0.0000001f);

        // Calculate the Jacobian based on the constrained new_mu_z
        jcb_a[col] = mu_a[col] * scale;
    }
}

__global__ void agvi_extract_odd_stream_kernel(
    const float *d_input_mu, const float *d_input_var, const float *d_input_jcb,
    int half_size, float *d_odd_mu, float *d_odd_var, float *d_odd_jcb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half_size) {
        int input_idx = 2 * i + 1;
        d_odd_mu[i] = d_input_mu[input_idx];
        d_odd_var[i] = d_input_var[input_idx];
        d_odd_jcb[i] = d_input_jcb[input_idx];
    }
}

__global__ void agvi_forward_combine_kernel(
    const float *d_input_mu, const float *d_input_var, const float *d_input_jcb,
    const float *d_inner_output_mu, int half_size, float *d_output_mu,
    float *d_output_var, float *d_output_jcb, bool agvi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half_size) {
        int even_input_idx = 2 * i;
        d_output_mu[i] = d_input_mu[even_input_idx];
        d_output_jcb[i] = d_input_jcb[even_input_idx];
        if (agvi)
            d_output_var[i] =
                d_input_var[even_input_idx] + d_inner_output_mu[i];
        else
            d_output_var[i] = d_input_var[even_input_idx];
    }
}

__global__ void agvi_backward_kernel(
    const float *d_incoming_delta_mu, const float *d_incoming_delta_var,
    const float *d_stored_output_mu_a, const float *d_stored_output_var_a,
    const float *d_stored_inner_mu_a, const float *d_stored_inner_var_a,
    const float *d_stored_inner_jcb, const float *d_stored_input_var_a,
    int half_size, float *d_output_delta_mu, float *d_output_delta_var,
    bool overfit_mu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const float epsilon = 1e-12f;

    if (i < half_size) {
        float mu_a = d_stored_output_mu_a[i];
        float var_a = fmaxf(d_stored_output_var_a[i], epsilon);

        float mu_v2_bar_tilde = d_stored_inner_mu_a[i];
        float var_v2_bar_tilde = d_stored_inner_var_a[i];
        float jcb_v2_bar_tilde = d_stored_inner_jcb[i];

        float incoming_delta_mu = d_incoming_delta_mu[i];
        float incoming_delta_var = d_incoming_delta_var[i];

        float var_z = d_stored_input_var_a[i * 2];

        float delta_a_mu = incoming_delta_mu * var_a;
        float delta_a_var = incoming_delta_var * var_a * var_a;

        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 =
            3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde;

        float Jv = mu_v2_bar_tilde / fmaxf(var_a, epsilon);

        float mu_v_pos = Jv * delta_a_mu;
        float var_v_pos = mu_v2_bar_tilde + Jv * Jv * delta_a_var;

        float mu_v2_pos = mu_v_pos * mu_v_pos + var_v_pos;
        float var_v2_pos = 2.0f * var_v_pos * var_v_pos +
                           4.0f * var_v_pos * mu_v_pos * mu_v_pos;

        float Jv2_bar_tilde = var_v2_bar_tilde / fmaxf(var_v2, epsilon);
        float mu_v2_bar_tilde_pos =
            mu_v2_bar_tilde + Jv2_bar_tilde * (mu_v2_pos - mu_v2);
        float var_v2_bar_tilde_pos =
            var_v2_bar_tilde +
            Jv2_bar_tilde * Jv2_bar_tilde * (var_v2_pos - var_v2);

        int even_idx = 2 * i;
        int odd_idx = 2 * i + 1;

        float Jv2_bar = jcb_v2_bar_tilde / fmaxf(var_v2_bar_tilde, epsilon);

        float odd_delta_mu = Jv2_bar * (mu_v2_bar_tilde_pos - mu_v2_bar_tilde);
        float odd_delta_var =
            Jv2_bar * Jv2_bar * (var_v2_bar_tilde_pos - var_v2_bar_tilde);

        if (isnan(odd_delta_mu) || isinf(odd_delta_mu) ||
            isnan(odd_delta_var) || isinf(odd_delta_var)) {
            odd_delta_mu = 0.0f;
            odd_delta_var = 0.0f;
        }

        d_output_delta_mu[odd_idx] = odd_delta_mu;
        d_output_delta_var[odd_idx] = odd_delta_var;

        float Jz = 1.0f / (var_a);
        float Jz_mu;
        if (overfit_mu) {
            Jz_mu = 1.0f / (var_z);
        } else {
            Jz_mu = Jz;
        }

        float even_delta_mu = Jz_mu * delta_a_mu;
        float even_delta_var = Jz * Jz * delta_a_var;

        // Check and handle potential NaN or Inf values for even stream deltas.
        if (isnan(even_delta_mu) || isinf(even_delta_mu) ||
            isnan(even_delta_var) || isinf(even_delta_var)) {
            even_delta_mu = 0.0f;
            even_delta_var = 0.0f;
        }

        d_output_delta_mu[even_idx] = even_delta_mu;
        d_output_delta_var[even_idx] = even_delta_var;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////
ReLUCuda::ReLUCuda() {}
ReLUCuda::~ReLUCuda() {}

std::string ReLUCuda::get_layer_info() const
/*
 */
{
    return "Relu()";
}

std::string ReLUCuda::get_layer_name() const
/*
 */
{
    return "ReLUCuda";
}

LayerType ReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ReLUCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    constexpr unsigned int THREADS = 256;
    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks = (num_states + THREADS - 1) / THREADS;

    relu_mean_var_cuda<<<blocks, THREADS>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }
}

std::unique_ptr<BaseLayer> ReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<ReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
SigmoidCuda::SigmoidCuda() {}
SigmoidCuda::~SigmoidCuda() {}

std::string SigmoidCuda::get_layer_info() const
/*
 */
{
    return "Sigmoid()";
}

std::string SigmoidCuda::get_layer_name() const
/*
 */
{
    return "SigmoidCuda";
}

LayerType SigmoidCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SigmoidCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    sigmoid_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SigmoidCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Sigmoid>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
TanhCuda::TanhCuda() {}
TanhCuda::~TanhCuda() {}

std::string TanhCuda::get_layer_info() const
/*
 */
{
    return "Tanh()";
}

std::string TanhCuda::get_layer_name() const
/*
 */
{
    return "TanhCuda";
}

LayerType TanhCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void TanhCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    tanh_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> TanhCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Tanh>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
MixtureReLUCuda::MixtureReLUCuda() {}
MixtureReLUCuda ::~MixtureReLUCuda() {}

std::string MixtureReLUCuda::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureReLUCuda::get_layer_name() const
/*
 */
{
    return "MixtureReLUCuda";
}

LayerType MixtureReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureReLUCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_relu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    // cu_output_states->to_device();

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoidCuda::MixtureSigmoidCuda() {}
MixtureSigmoidCuda ::~MixtureSigmoidCuda() {}

std::string MixtureSigmoidCuda::get_layer_info() const
/*
 */
{
    return "MixtureSigmoid()";
}

std::string MixtureSigmoidCuda::get_layer_name() const
/*
 */
{
    return "MixtureSigmoidCuda";
}

LayerType MixtureSigmoidCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureSigmoidCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_sigmoid_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureSigmoidCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureSigmoid>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanhCuda::MixtureTanhCuda() {}
MixtureTanhCuda ::~MixtureTanhCuda() {}

std::string MixtureTanhCuda::get_layer_info() const
/*
 */
{
    return "MixtureTanh()";
}

std::string MixtureTanhCuda::get_layer_name() const
/*
 */
{
    return "MixtureTanhCuda";
}

LayerType MixtureTanhCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureTanhCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_tanh_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureTanhCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureTanh>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
// CELU + alpha
////////////////////////////////////////////////////////////////////////////////

CELUCuda::CELUCuda() {}
CELUCuda ::~CELUCuda() {}

std::string CELUCuda::get_layer_info() const
/*
 */
{
    return "CELU()";
}

std::string CELUCuda::get_layer_name() const
/*
 */
{
    return "CELUCuda";
}

LayerType CELUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void CELUCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    celu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> CELUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<CELU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
SoftplusCuda::SoftplusCuda() {}
SoftplusCuda::~SoftplusCuda() {}

std::string SoftplusCuda::get_layer_info() const
/*
 */
{
    return "Softplus()";
}

std::string SoftplusCuda::get_layer_name() const
/*
 */
{
    return "SoftplusCuda";
}

LayerType SoftplusCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SoftplusCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    softplus_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SoftplusCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Softplus>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// LeakyRelu
////////////////////////////////////////////////////////////////////////////////
LeakyReLUCuda::LeakyReLUCuda() {}
LeakyReLUCuda::~LeakyReLUCuda() {}

std::string LeakyReLUCuda::get_layer_info() const
/*
 */
{
    return "leakyRelu()";
}

std::string LeakyReLUCuda::get_layer_name() const
/*
 */
{
    return "leakyReluCuda";
}

LayerType LeakyReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyReLUCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    leakyrelu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->alpha,
        num_states, cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> LeakyReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<LeakyReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Softmax
////////////////////////////////////////////////////////////////////////////////
SoftmaxCuda::SoftmaxCuda() {}
SoftmaxCuda::~SoftmaxCuda() {}

std::string SoftmaxCuda::get_layer_info() const
/*
 */
{
    return "Softmax()";
}

std::string SoftmaxCuda::get_layer_name() const
/*
 */
{
    return "SoftmaxCuda";
}

LayerType SoftmaxCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SoftmaxCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    unsigned int blocks =
        (input_states.block_size + this->num_cuda_threads - 1) /
        this->num_cuda_threads;

    softmax_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->actual_size, cu_input_states->block_size,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SoftmaxCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Softmax>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
RemaxCuda::RemaxCuda() {}
RemaxCuda::~RemaxCuda() { this->deallocate_memory(); }

std::string RemaxCuda::get_layer_info() const
/*
 */
{
    return "Remax()";
}

std::string RemaxCuda::get_layer_name() const
/*
 */
{
    return "RemaxCuda";
}

LayerType RemaxCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void RemaxCuda::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/*
 */
{
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = input_states.block_size;
    int hidden_size = input_states.actual_size;
    if (this->batch_size_ != batch_size) {
        this->batch_size_ = batch_size;
        this->deallocate_memory();
        this->allocate_memory(hidden_size, batch_size);
    }
    int num_states = batch_size * hidden_size;
    constexpr int THREADS = 256;
    constexpr int THREADS_BATCH = 16;
    constexpr int THREADS_HIDDEN = 16;
    unsigned int blocks = (num_states + THREADS - 1) / THREADS;

    mixture_relu_mean_var_cuda<<<blocks, THREADS>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        this->d_mu_m, this->d_jcb_m, this->d_var_m);

    // Compute mean and variance of Mt
    unsigned int blocks_sum = (batch_size + THREADS - 1) / THREADS;
    compute_mean_var_sum_cuda<<<blocks_sum, THREADS>>>(
        this->d_mu_m, this->d_var_m, hidden_size, batch_size, this->d_mu_mt,
        this->d_var_mt);

    // Compute mean and variance of log(M)
    unsigned int hidden_blocks =
        (hidden_size + THREADS_HIDDEN - 1) / THREADS_HIDDEN;
    unsigned int batch_blocks =
        (batch_size + THREADS_BATCH - 1) / THREADS_BATCH;
    dim3 dim_grid_log(hidden_blocks, batch_blocks);
    dim3 dim_block_log(THREADS_HIDDEN, THREADS_BATCH);

    to_log_cuda<<<dim_grid_log, dim_block_log>>>(
        this->d_mu_m, this->d_var_m, hidden_size, batch_size, this->d_mu_log_m,
        this->d_var_log_m);

    // Compute mean and variance of log(Mt)
    unsigned int blocks_log_mt = (batch_size + THREADS - 1) / THREADS;
    dim3 dim_grid_log_mt(1, blocks_log_mt);
    dim3 dim_block_log_mt(1, THREADS);
    to_log_cuda<<<dim_grid_log_mt, dim_block_log_mt>>>(
        this->d_mu_mt, this->d_var_mt, 1, batch_size, this->d_mu_log_mt,
        this->d_var_log_mt);

    // Compute covariance of log(M) and log(Mt)
    compute_cov_log_m_mt_cuda<<<dim_grid_log, dim_block_log>>>(
        this->d_mu_m, this->d_var_m, this->d_mu_mt, hidden_size, batch_size,
        this->d_cov_log_m_mt);

    // Compute mean and variance of A
    compute_remax_mean_var_cuda<<<blocks_sum, THREADS>>>(
        this->d_mu_log_m, this->d_var_log_m, this->d_mu_log_mt,
        this->d_var_log_mt, this->d_cov_log_m_mt, hidden_size, batch_size,
        cu_output_states->d_mu_a, cu_output_states->d_var_a);

    // Compute covariance of A and Z i.e., Jacobian.
    compute_cov_a_z_cuda<<<dim_grid_log, dim_block_log>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_input_states->d_var_a, this->d_mu_m, this->d_var_m,
        this->d_var_log_m, this->d_cov_log_m_mt, this->d_jcb_m, hidden_size,
        batch_size, cu_output_states->d_jcb);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> RemaxCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Remax>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

void RemaxCuda::allocate_memory(int hidden_size, int batch_size)
/*
 */
{
    int size = hidden_size * batch_size;
    this->mu_m.resize(size, 0.0f);
    this->var_m.resize(size, 0.0f);
    this->jcb_m.resize(size, 0.0f);
    this->mu_log_m.resize(size, 0.0f);
    this->var_log_m.resize(size, 0.0f);
    this->mu_mt.resize(batch_size, 0.0f);
    this->var_mt.resize(batch_size, 0.0f);
    this->mu_log_mt.resize(batch_size, 0.0f);
    this->var_log_mt.resize(batch_size, 0.0f);
    this->cov_log_m_mt.resize(size, 0.0f);

    cudaMalloc(&this->d_mu_m, size * sizeof(float));
    cudaMalloc(&this->d_var_m, size * sizeof(float));
    cudaMalloc(&this->d_jcb_m, size * sizeof(float));
    cudaMalloc(&this->d_mu_log_m, size * sizeof(float));
    cudaMalloc(&this->d_var_log_m, size * sizeof(float));
    cudaMalloc(&this->d_mu_mt, batch_size * sizeof(float));
    cudaMalloc(&this->d_var_mt, batch_size * sizeof(float));
    cudaMalloc(&this->d_mu_log_mt, batch_size * sizeof(float));
    cudaMalloc(&this->d_var_log_mt, batch_size * sizeof(float));
    cudaMalloc(&this->d_cov_log_m_mt, size * sizeof(float));
}

void RemaxCuda::deallocate_memory()
/*
 */
{
    cudaFree(this->d_mu_m);
    this->d_mu_m = nullptr;
    cudaFree(this->d_var_m);
    this->d_var_m = nullptr;
    cudaFree(this->d_jcb_m);
    this->d_jcb_m = nullptr;
    cudaFree(this->d_mu_log_m);
    this->d_mu_log_m = nullptr;
    cudaFree(this->d_var_log_m);
    this->d_var_log_m = nullptr;
    cudaFree(this->d_mu_mt);
    this->d_mu_mt = nullptr;
    cudaFree(this->d_var_mt);
    this->d_var_mt = nullptr;
    cudaFree(this->d_mu_log_mt);
    this->d_mu_log_mt = nullptr;
    cudaFree(this->d_var_log_mt);
    this->d_var_log_mt = nullptr;
    cudaFree(this->d_cov_log_m_mt);
    this->d_cov_log_m_mt = nullptr;
    this->mu_m.clear();
    this->var_m.clear();
    this->jcb_m.clear();
    this->mu_log_m.clear();
    this->var_log_m.clear();
    this->mu_mt.clear();
    this->var_mt.clear();
    this->mu_log_mt.clear();
    this->var_log_mt.clear();
    this->cov_log_m_mt.clear();
}

void RemaxCuda::data_to_host()
/*
 */
{
    cudaMemcpy(this->mu_m.data(), this->d_mu_m,
               this->mu_m.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_m.data(), this->d_var_m,
               this->var_m.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_m.data(), this->d_jcb_m,
               this->jcb_m.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_log_m.data(), this->d_mu_log_m,
               this->mu_log_m.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_log_m.data(), this->d_var_log_m,
               this->var_log_m.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_mt.data(), this->d_mu_mt,
               this->mu_mt.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_mt.data(), this->d_var_mt,
               this->var_mt.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_log_mt.data(), this->d_mu_log_mt,
               this->mu_log_mt.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_log_mt.data(), this->d_var_log_mt,
               this->var_log_mt.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cov_log_m_mt.data(), this->d_cov_log_m_mt,
               this->cov_log_m_mt.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void RemaxCuda::data_to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_m, this->mu_m.data(),
               this->mu_m.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_m, this->var_m.data(),
               this->var_m.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb_m, this->jcb_m.data(),
               this->jcb_m.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_log_m, this->mu_log_m.data(),
               this->mu_log_m.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_log_m, this->var_log_m.data(),
               this->var_log_m.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_mt, this->mu_mt.data(),
               this->mu_mt.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_mt, this->var_mt.data(),
               this->var_mt.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_log_mt, this->mu_log_mt.data(),
               this->mu_log_mt.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_log_mt, this->var_log_mt.data(),
               this->var_log_mt.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_cov_log_m_mt, this->cov_log_m_mt.data(),
               this->cov_log_m_mt.size() * sizeof(float),
               cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
/// ClosedFormSoftmax
////////////////////////////////////////////////////////////////////////////////
__global__ void compute_mean_var_exp_sum_cuda(const float *mu_z,
                                              const float *var_z,
                                              int hidden_size, int batch_size,
                                              float *mu_e_sum,
                                              float *var_e_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum_mu += expf(mu_z[idx * hidden_size + j] +
                           0.5f * var_z[idx * hidden_size + j]);
            sum_var += expf(2.0f * mu_z[idx * hidden_size + j] +
                            var_z[idx * hidden_size + j]) *
                       (expf(var_z[idx * hidden_size + j]) - 1.0f);
        }
        mu_e_sum[idx] = sum_mu;
        var_e_sum[idx] = sum_var;
    }
}

__global__ void compute_mean_var_log_a_cuda(
    const float *mu_z, const float *var_z, const float *mu_log_e_sum,
    const float *var_log_e_sum, const float *mu_e_sum, const float *var_e_sum,
    int hidden_size, int batch_size, float *mu_log_a, float *var_log_a,
    float *cov_log_a_z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < hidden_size) {
        float cov_e_e_sum = expf(2.0f * mu_z[row * hidden_size + col] +
                                 var_z[row * hidden_size + col]) *
                            (expf(var_z[row * hidden_size + col]) - 1.0f);
        float mu_e = expf(mu_z[row * hidden_size + col] +
                          0.5f * var_z[row * hidden_size + col]);
        float tmp_inverse_mu = 1.0f / (mu_e_sum[row] * mu_e);
        float cov_z_log_e_sum = logf(1.0f + cov_e_e_sum * tmp_inverse_mu);
        mu_log_a[row * hidden_size + col] =
            mu_z[row * hidden_size + col] - mu_log_e_sum[row];
        var_log_a[row * hidden_size + col] = var_z[row * hidden_size + col] +
                                             var_log_e_sum[row] -
                                             2.0f * cov_z_log_e_sum;
        cov_log_a_z[row * hidden_size + col] =
            var_z[row * hidden_size + col] - cov_z_log_e_sum;
    }
}

__global__ void compute_cfsoftmax_mean_var_cuda(
    const float *mu_log_a, const float *var_log_a, const float *cov_log_a_z,
    const float *var_z, int hidden_size, int batch_size, float *mu_a,
    float *var_a, float *jcb_a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < hidden_size) {
        float tmp_mu = expf(mu_log_a[row * hidden_size + col] +
                            0.5f * var_log_a[row * hidden_size + col]);
        if (isnan(tmp_mu)) {
            tmp_mu = 0.00001f;
        } else {
            tmp_mu = min(1.0f, max(0.00001f, tmp_mu));
        }
        mu_a[row * hidden_size + col] = tmp_mu;
        var_a[row * hidden_size + col] =
            max(0.00001f, (expf(var_log_a[row * hidden_size + col]) - 1.0f) *
                              tmp_mu * tmp_mu);
        if (isnan(var_a[row * hidden_size + col])) {
            var_a[row * hidden_size + col] = 0.00001f;
        }
        jcb_a[row * hidden_size + col] =
            max(0.00001f, min(powf(var_a[row * hidden_size + col], 0.5f) *
                                  powf(var_z[row * hidden_size + col], 0.5f),
                              tmp_mu * cov_log_a_z[row * hidden_size + col])) /
            var_z[row * hidden_size + col];
    }
}

ClosedFormSoftmaxCuda::ClosedFormSoftmaxCuda() {}
ClosedFormSoftmaxCuda::~ClosedFormSoftmaxCuda() {}

std::string ClosedFormSoftmaxCuda::get_layer_info() const
/*
 */
{
    return "ClosedFormSoftmax()";
}

std::string ClosedFormSoftmaxCuda::get_layer_name() const
/*
 */
{
    return "ClosedFormSoftmaxCuda";
}

LayerType ClosedFormSoftmaxCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ClosedFormSoftmaxCuda::forward(BaseHiddenStates &input_states,
                                    BaseHiddenStates &output_states,
                                    BaseTempStates &temp_states)
/*
 */
{
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = input_states.block_size;
    int hidden_size = input_states.actual_size;
    if (this->batch_size_ != batch_size) {
        this->batch_size_ = batch_size;
        this->deallocate_memory();
        this->allocate_memory(hidden_size, batch_size);
    }
    constexpr int THREADS = 256;
    unsigned int blocks = (batch_size + THREADS - 1) / THREADS;

    // Compute mean and variance of softmax's denominator sum[exp(z)]
    compute_mean_var_exp_sum_cuda<<<blocks, THREADS>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, hidden_size,
        batch_size, this->d_mu_e_sum, this->d_var_e_sum);

    // Transform to log space
    dim3 dim_grid_log(1, blocks);
    dim3 dim_block_log(1, THREADS);
    to_log_cuda<<<dim_grid_log, dim_block_log>>>(
        this->d_mu_e_sum, this->d_var_e_sum, 1, batch_size,
        this->d_mu_log_e_sum, this->d_var_log_e_sum);

    // Compute mean and variance of log[softmax(z)]
    constexpr int THREADS_BATCH = 16;
    constexpr int THREADS_HIDDEN = 16;
    const int batch_blocks = (batch_size + THREADS_BATCH - 1) / THREADS_BATCH;
    const int hidden_blocks =
        (hidden_size + THREADS_HIDDEN - 1) / THREADS_HIDDEN;
    dim3 dim_grid_a(hidden_blocks, batch_blocks);
    dim3 dim_block_a(THREADS_HIDDEN, THREADS_BATCH);
    compute_mean_var_log_a_cuda<<<dim_grid_a, dim_block_a>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_mu_log_e_sum,
        this->d_var_log_e_sum, this->d_mu_e_sum, this->d_var_e_sum, hidden_size,
        batch_size, this->d_mu_log_a, this->d_var_log_a, this->d_cov_log_a_z);

    // Compute mean and variance of softmax(z)
    compute_cfsoftmax_mean_var_cuda<<<dim_grid_a, dim_block_a>>>(
        this->d_mu_log_a, this->d_var_log_a, this->d_cov_log_a_z,
        cu_input_states->d_var_a, hidden_size, batch_size,
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> ClosedFormSoftmaxCuda::to_host()
/*
 */
{
    std::unique_ptr<BaseLayer> host_layer =
        std::make_unique<ClosedFormSoftmax>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

void ClosedFormSoftmaxCuda::allocate_memory(int hidden_size, int batch_size)
/*
 */
{
    int size = hidden_size * batch_size;
    this->mu_e_sum.resize(batch_size, 0.0f);
    this->var_e_sum.resize(batch_size, 0.0f);
    this->cov_z_log_e_sum.resize(size, 0.0f);
    this->mu_log_e_sum.resize(batch_size, 0.0f);
    this->var_log_e_sum.resize(batch_size, 0.0f);
    this->cov_log_a_z.resize(size, 0.0f);
    this->mu_log_a.resize(size, 0.0f);
    this->var_log_a.resize(size, 0.0f);

    cudaMalloc(&this->d_mu_e_sum, batch_size * sizeof(float));
    cudaMalloc(&this->d_var_e_sum, batch_size * sizeof(float));
    cudaMalloc(&this->d_cov_z_log_e_sum,
               batch_size * hidden_size * sizeof(float));
    cudaMalloc(&this->d_mu_log_e_sum, batch_size * sizeof(float));
    cudaMalloc(&this->d_var_log_e_sum, batch_size * sizeof(float));
    cudaMalloc(&this->d_cov_log_a_z, size * sizeof(float));
    cudaMalloc(&this->d_mu_log_a, size * sizeof(float));
    cudaMalloc(&this->d_var_log_a, size * sizeof(float));
}

void ClosedFormSoftmaxCuda::deallocate_memory() {
    cudaFree(this->d_mu_e_sum);
    this->d_mu_e_sum = nullptr;
    cudaFree(this->d_var_e_sum);
    this->d_var_e_sum = nullptr;
    cudaFree(this->d_cov_z_log_e_sum);
    this->d_cov_z_log_e_sum = nullptr;
    cudaFree(this->d_mu_log_e_sum);
    this->d_mu_log_e_sum = nullptr;
    cudaFree(this->d_var_log_e_sum);
    this->d_var_log_e_sum = nullptr;
    cudaFree(this->d_cov_log_a_z);
    this->d_cov_log_a_z = nullptr;
    cudaFree(this->d_mu_log_a);
    this->d_mu_log_a = nullptr;
    cudaFree(this->d_var_log_a);
    this->d_var_log_a = nullptr;
    this->mu_e_sum.clear();
    this->var_e_sum.clear();
    this->cov_z_log_e_sum.clear();
    this->mu_log_e_sum.clear();
    this->var_log_e_sum.clear();
    this->cov_log_a_z.clear();
    this->mu_log_a.clear();
    this->var_log_a.clear();
}

void ClosedFormSoftmaxCuda::data_to_host() {
    cudaMemcpy(this->mu_e_sum.data(), this->d_mu_e_sum,
               this->mu_e_sum.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_e_sum.data(), this->d_var_e_sum,
               this->var_e_sum.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cov_z_log_e_sum.data(), this->d_cov_z_log_e_sum,
               this->cov_z_log_e_sum.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_log_e_sum.data(), this->d_mu_log_e_sum,
               this->mu_log_e_sum.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_log_e_sum.data(), this->d_var_log_e_sum,
               this->var_log_e_sum.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cov_log_a_z.data(), this->d_cov_log_a_z,
               this->cov_log_a_z.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_log_a.data(), this->d_mu_log_a,
               this->mu_log_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_log_a.data(), this->d_var_log_a,
               this->var_log_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
}

void ClosedFormSoftmaxCuda::data_to_device() {
    cudaMemcpy(this->d_mu_e_sum, this->mu_e_sum.data(),
               this->mu_e_sum.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_e_sum, this->var_e_sum.data(),
               this->var_e_sum.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_cov_z_log_e_sum, this->cov_z_log_e_sum.data(),
               this->cov_z_log_e_sum.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_log_e_sum, this->mu_log_e_sum.data(),
               this->mu_log_e_sum.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_log_e_sum, this->var_log_e_sum.data(),
               this->var_log_e_sum.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_cov_log_a_z, this->cov_log_a_z.data(),
               this->cov_log_a_z.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_log_a, this->mu_log_a.data(),
               this->mu_log_a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_log_a, this->var_log_a.data(),
               this->var_log_a.size() * sizeof(float), cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
/// SplitActivationCuda
////////////////////////////////////////////////////////////////////////////////

SplitActivationCuda::SplitActivationCuda(std::unique_ptr<BaseLayer> odd_layer,
                                         std::unique_ptr<BaseLayer> even_layer)
    : odd_layer(std::move(odd_layer)), even_layer(std::move(even_layer)) {}

SplitActivationCuda::~SplitActivationCuda() {}

std::string SplitActivationCuda::get_layer_info() const {
    std::string even_layer_name = "Identity";
    if (even_layer) {
        even_layer_name = even_layer->get_layer_name();
    }
    return "SplitActivationCuda(odd=" + odd_layer->get_layer_name() +
           ", even=" + even_layer_name + ")";
}

std::string SplitActivationCuda::get_layer_name() const {
    return "SplitActivationCuda";
}

LayerType SplitActivationCuda::get_layer_type() const {
    return LayerType::Activation;
}

std::unique_ptr<BaseLayer> SplitActivationCuda::to_host() {
    auto *odd_cuda_layer = dynamic_cast<BaseLayerCuda *>(odd_layer.get());
    auto host_odd_layer = odd_cuda_layer->to_host();

    std::shared_ptr<BaseLayer> host_even_layer = nullptr;
    if (even_layer) {
        auto *even_cuda_layer = dynamic_cast<BaseLayerCuda *>(even_layer.get());
        host_even_layer = even_cuda_layer->to_host();
    }

    return std::make_unique<SplitActivation>(std::move(host_odd_layer),
                                             std::move(host_even_layer));
}

void SplitActivationCuda::forward(BaseHiddenStates &input_states,
                                  BaseHiddenStates &output_states,
                                  BaseTempStates &temp_states) {
    // 1. Cast states to CUDA-specific types
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    // 2. Calculate sizes
    int total_elements =
        cu_input_states->actual_size * cu_input_states->block_size;
    int half_size = total_elements / 2;

    // 3. Prepare temporary states for split streams
    HiddenStateCuda odd_input_states, even_input_states;
    cudaMalloc(&odd_input_states.d_mu_a, half_size * sizeof(float));
    cudaMalloc(&odd_input_states.d_var_a, half_size * sizeof(float));
    cudaMalloc(&odd_input_states.d_jcb, half_size * sizeof(float));
    cudaMalloc(&even_input_states.d_mu_a, half_size * sizeof(float));
    cudaMalloc(&even_input_states.d_var_a, half_size * sizeof(float));
    cudaMalloc(&even_input_states.d_jcb, half_size * sizeof(float));

    odd_input_states.block_size = cu_input_states->block_size;
    odd_input_states.actual_size = half_size / cu_input_states->block_size;
    even_input_states.block_size = cu_input_states->block_size;
    even_input_states.actual_size = half_size / cu_input_states->block_size;

    // 4. Launch kernel to split the input stream
    unsigned int threads = 256;
    unsigned int blocks = (half_size + threads - 1) / threads;
    split_stream_kernel<<<blocks, threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, half_size, even_input_states.d_mu_a,
        even_input_states.d_var_a, even_input_states.d_jcb,
        odd_input_states.d_mu_a, odd_input_states.d_var_a,
        odd_input_states.d_jcb);

    // 5. Process the streams
    HiddenStateCuda odd_output_states, even_output_states;
    // odd_output_states.allocate(half_size);
    // even_output_states.allocate(half_size);
    cudaMalloc(&odd_output_states.d_mu_a, half_size * sizeof(float));
    cudaMalloc(&odd_output_states.d_var_a, half_size * sizeof(float));
    cudaMalloc(&odd_output_states.d_jcb, half_size * sizeof(float));
    cudaMalloc(&even_output_states.d_mu_a, half_size * sizeof(float));
    cudaMalloc(&even_output_states.d_var_a, half_size * sizeof(float));
    cudaMalloc(&even_output_states.d_jcb, half_size * sizeof(float));

    // Process odd stream (mandatory)
    odd_layer->forward(odd_input_states, odd_output_states, temp_states);

    // Process even stream (optional)
    if (even_layer) {
        even_layer->forward(even_input_states, even_output_states, temp_states);
    } else {
        // Identity: Copy even input directly to even output
        cudaMemcpy(even_output_states.d_mu_a, even_input_states.d_mu_a,
                   half_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(even_output_states.d_var_a, even_input_states.d_var_a,
                   half_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(even_output_states.d_jcb, even_input_states.d_jcb,
                   half_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // 6. Launch kernel to merge the processed streams
    merge_stream_kernel<<<blocks, threads>>>(
        even_output_states.d_mu_a, even_output_states.d_var_a,
        even_output_states.d_jcb, odd_output_states.d_mu_a,
        odd_output_states.d_var_a, odd_output_states.d_jcb, half_size,
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb);

    // 7. Update output state metadata
    cu_output_states->actual_size = cu_input_states->actual_size;
    cu_output_states->block_size = cu_input_states->block_size;
    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // 8. Free temporary device memory
    cudaFree(odd_input_states.d_mu_a);
    cudaFree(odd_input_states.d_var_a);
    cudaFree(odd_input_states.d_jcb);
    cudaFree(even_input_states.d_mu_a);
    cudaFree(even_input_states.d_var_a);
    cudaFree(even_input_states.d_jcb);
    cudaFree(odd_output_states.d_mu_a);
    cudaFree(odd_output_states.d_var_a);
    cudaFree(odd_output_states.d_jcb);
    cudaFree(even_output_states.d_mu_a);
    cudaFree(even_output_states.d_var_a);
    cudaFree(even_output_states.d_jcb);
    odd_input_states.d_mu_a = nullptr;
    odd_input_states.d_var_a = nullptr;
    odd_input_states.d_jcb = nullptr;
    even_input_states.d_mu_a = nullptr;
    even_input_states.d_var_a = nullptr;
    even_input_states.d_jcb = nullptr;
    odd_output_states.d_mu_a = nullptr;
    odd_output_states.d_var_a = nullptr;
    odd_output_states.d_jcb = nullptr;
    even_output_states.d_mu_a = nullptr;
    even_output_states.d_var_a = nullptr;
    even_output_states.d_jcb = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Exp
////////////////////////////////////////////////////////////////////////////////
ExpCuda::ExpCuda(float scale, float shift) : scale(scale), shift(shift) {}
ExpCuda::~ExpCuda() {}

std::string ExpCuda::get_layer_info() const
/*
 */
{
    return "Exp()";
}

std::string ExpCuda::get_layer_name() const
/*
 */
{
    return "ExpCuda";
}

LayerType ExpCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ExpCuda::forward(BaseHiddenStates &input_states,
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

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    exp_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, num_states, cu_output_states->d_mu_a,
        cu_output_states->d_var_a, cu_output_states->d_jcb, scale, shift);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> ExpCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Exp>(scale, shift);
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// AGVICuda
////////////////////////////////////////////////////////////////////////////////

AGVICuda::AGVICuda(std::unique_ptr<BaseLayer> activation_layer, bool overfit_mu,
                   bool agvi)
    : m_activation_layer(std::move(activation_layer)),
      m_overfit_mu(overfit_mu),
      m_agvi(agvi) {}

AGVICuda::~AGVICuda() {}

std::string AGVICuda::get_layer_info() const {
    return "AGVICuda(" + m_activation_layer->get_layer_name() + ")";
}

std::string AGVICuda::get_layer_name() const { return "AGVICuda"; }

LayerType AGVICuda::get_layer_type() const { return LayerType::AGVI; }

std::unique_ptr<BaseLayer> AGVICuda::to_host() {
    auto *cuda_layer = dynamic_cast<BaseLayerCuda *>(m_activation_layer.get());
    if (!cuda_layer) {
        throw std::runtime_error(
            "AGVICuda::to_host(): Failed to cast inner layer to "
            "BaseLayerCuda.");
    }
    auto host_inner_layer = cuda_layer->to_host();
    return std::make_unique<AGVI>(std::move(host_inner_layer), m_overfit_mu,
                                  m_agvi);
}

void AGVICuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states) {
    // Cast to CUDA-specific types
    auto *cu_input_states = dynamic_cast<HiddenStateCuda *>(&input_states);
    auto *cu_output_states = dynamic_cast<HiddenStateCuda *>(&output_states);

    // Calculate sizes
    int total_elements =
        cu_input_states->actual_size * cu_input_states->block_size;
    int half_size = total_elements / 2;

    // Set output dimensions
    cu_output_states->actual_size = cu_input_states->actual_size / 2;
    cu_output_states->block_size = cu_input_states->block_size;
    this->input_size = cu_input_states->actual_size;
    this->output_size = cu_output_states->actual_size;

    // 1. Prepare states for the inner activation layer (odd stream)
    HiddenStateCuda odd_input_states;
    odd_input_states.block_size = cu_input_states->block_size;
    odd_input_states.actual_size = cu_input_states->actual_size / 2;

    // Allocate memory for odd input states
    {
        size_t size = odd_input_states.block_size *
                      odd_input_states.actual_size * sizeof(float);
        cudaMalloc(&odd_input_states.d_mu_a, size);
        cudaMalloc(&odd_input_states.d_var_a, size);
        cudaMalloc(&odd_input_states.d_jcb, size);
    }

    // 2. Launch kernel to extract the odd-indexed elements from the input
    unsigned int threads = 256;
    unsigned int blocks = (half_size + threads - 1) / threads;
    agvi_extract_odd_stream_kernel<<<blocks, threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, half_size, odd_input_states.d_mu_a,
        odd_input_states.d_var_a, odd_input_states.d_jcb);

    // 3. Allocate memory for the inner activation output and call its forward
    // pass
    m_stored_inner_output_states.block_size = odd_input_states.block_size;
    m_stored_inner_output_states.actual_size = odd_input_states.actual_size;

    // Allocate memory for the inner activation output states
    {
        size_t size = m_stored_inner_output_states.block_size *
                      m_stored_inner_output_states.actual_size * sizeof(float);
        cudaMalloc(&m_stored_inner_output_states.d_mu_a, size);
        cudaMalloc(&m_stored_inner_output_states.d_var_a, size);
        cudaMalloc(&m_stored_inner_output_states.d_jcb, size);
    }

    // Forward the inner activation layer
    m_activation_layer->forward(odd_input_states, m_stored_inner_output_states,
                                temp_states);

    // 4. Launch kernel to combine the even stream and inner activation output
    agvi_forward_combine_kernel<<<blocks, threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, m_stored_inner_output_states.d_mu_a, half_size,
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, m_agvi);

    // 5. Store pointers for the backward pass
    d_stored_input_var_a = cu_input_states->d_var_a;
    d_stored_output_mu_a = cu_output_states->d_mu_a;
    d_stored_output_var_a = cu_output_states->d_var_a;

    // Deallocate the odd input states memory
    cudaFree(odd_input_states.d_mu_a);
    cudaFree(odd_input_states.d_var_a);
    cudaFree(odd_input_states.d_jcb);
    odd_input_states.d_mu_a = nullptr;
    odd_input_states.d_var_a = nullptr;
    odd_input_states.d_jcb = nullptr;
}

void AGVICuda::backward(BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_delta_states,
                        BaseTempStates &temp_states, bool state_update) {
    auto *cu_input_delta = dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    auto *cu_output_delta =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int half_size = this->output_size * cu_input_delta->block_size;

    unsigned int threads = 256;
    unsigned int blocks = (half_size + threads - 1) / threads;
    agvi_backward_kernel<<<blocks, threads>>>(
        cu_input_delta->d_delta_mu, cu_input_delta->d_delta_var,
        d_stored_output_mu_a, d_stored_output_var_a,
        m_stored_inner_output_states.d_mu_a,
        m_stored_inner_output_states.d_var_a,
        m_stored_inner_output_states.d_jcb, d_stored_input_var_a, half_size,
        cu_output_delta->d_delta_mu, cu_output_delta->d_delta_var,
        m_overfit_mu);  // Pass the flag here

    cu_output_delta->actual_size = this->input_size;
    cu_output_delta->block_size = cu_input_delta->block_size;

    // Free memory used for storing inner states.
    // The other d_stored_* pointers are just views and don't own memory.
    cudaFree(m_stored_inner_output_states.d_mu_a);
    cudaFree(m_stored_inner_output_states.d_var_a);
    cudaFree(m_stored_inner_output_states.d_jcb);
    m_stored_inner_output_states.d_mu_a = nullptr;
    m_stored_inner_output_states.d_var_a = nullptr;
    m_stored_inner_output_states.d_jcb = nullptr;
    d_stored_input_var_a = nullptr;
    d_stored_output_mu_a = nullptr;
    d_stored_output_var_a = nullptr;
}
