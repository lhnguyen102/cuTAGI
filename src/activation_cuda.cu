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
        alpha_l =
            (1.0f + (mu_z[col] * 2.0f - 1.0f)) / std_z;  // Lower truncation
        alpha_u =
            (1.0f - (mu_z[col] * 2.0f - 1.0f)) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_z = mu_z[col] * 2.0f - 1.0f;
        float tmp_mu_z_2 = tmp_mu_z * tmp_mu_z;
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = tmp_mu_a;
        var_a[col] =
            fmaxf(0.0000001f,
                  (cdf_l * (var_z[col] - tmp_mu_z_2 - 2 * tmp_mu_z - 1) +
                   cdf_u * (var_z[col] - tmp_mu_z_2 + 2 * tmp_mu_z - 1) +
                   std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                   tmp_mu_a * tmp_mu_a + 2 * mu_a[col] * tmp_mu_z +
                   tmp_mu_z * tmp_mu_z - var_z[col] + 2) /
                      4.0f);
        mu_a[col] = fmaxf(tmp_mu_a / 2.0f + 0.5f, 0.0000001f);
        jcb[col] = (cdf_u + cdf_l - 1) / 2.0f;
    }

    // // Double sided mReLU
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // float lower_bound = 0.0f;
    // float upper_bound = 1.0f;
    // constexpr float INV_SQRT_2 = 0.7071067811865475f;
    // constexpr float TINY_FLOAT = 0.0000001f;

    // if (idx < num_states) {
    //     // Load input mean and variance for the current element
    //     float mu_y_i = mu_z[idx];
    //     float var_y_i = var_z[idx];
    //     float sigma_y_i = sqrtf(fmaxf(var_y_i, TINY_FLOAT));

    //     // Standardize the bounds
    //     float alpha = (lower_bound - mu_y_i) / sigma_y_i; // Corresponds to
    //     z_lower float beta = (upper_bound - mu_y_i) / sigma_y_i;   //
    //     Corresponds to z_upper

    //     // Compute PDF and CDF values
    //     float cdf_alpha = normcdf_cuda(alpha);
    //     float cdf_beta = normcdf_cuda(beta);
    //     float pdf_alpha = normpdf_cuda(alpha);
    //     float pdf_beta = normpdf_cuda(beta);

    //     float cdf_diff = cdf_beta - cdf_alpha;

    //     // ---- 1. Calculate Mean E[z] ----
    //     float tmp_mu_z = lower_bound * cdf_alpha +
    //                      mu_y_i * cdf_diff +
    //                      sigma_y_i * (pdf_alpha - pdf_beta) +
    //                      upper_bound * (1.0f - cdf_beta);

    //     // ---- 2. Calculate Second Moment E[z^2] ----
    //     float term1_Ez2 = lower_bound * lower_bound * cdf_alpha;
    //     float term2_Ez2 = (mu_y_i * mu_y_i + var_y_i) * cdf_diff;
    //     float term3_Ez2 = -sigma_y_i * ((mu_y_i + upper_bound) * pdf_beta -
    //     (mu_y_i + lower_bound) * pdf_alpha); float term4_Ez2 = upper_bound *
    //     upper_bound * (1.0f - cdf_beta); float Ez2 = term1_Ez2 + term2_Ez2 +
    //     term3_Ez2 + term4_Ez2;

    //     // ---- 3. Calculate Variance Var[z] ----
    //     float tmp_var_z = Ez2 - tmp_mu_z * tmp_mu_z;

    //     // ---- 4. Calculate Cross Moment E[y*z] to find Cov(y, z) ----
    //     float term1_Eyz = lower_bound * (mu_y_i * cdf_alpha - sigma_y_i *
    //     pdf_alpha);
    //     // term2_Eyz is identical to term2_Ez2
    //     // term3_Eyz is identical to term3_Ez2
    //     float term4_Eyz = upper_bound * (mu_y_i * (1.0f - cdf_beta) +
    //     sigma_y_i * pdf_beta); float Eyz = term1_Eyz + term2_Ez2 + term3_Ez2
    //     + term4_Eyz;

    //     float cov_yz = Eyz - tmp_mu_z * mu_y_i;

    //     // Store the final results
    //     mu_a[idx] = fmaxf(tmp_mu_z, TINY_FLOAT);
    //     var_a[idx] = fmaxf(tmp_var_z, TINY_FLOAT);
    //     jcb[idx] = cov_yz / fmaxf(var_y_i, TINY_FLOAT);
    // }
}

__global__ void normalize_means(float *mu_a, float *var_a, float *jcb,
                                int num_states, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < num_states; j++) {
            sum += mu_a[j + idx * num_states];
        }
        for (int j = 0; j < num_states; j++) {
            mu_a[j + idx * num_states] /= sum;
            var_a[j + idx * num_states] /= (sum * sum);
            jcb[j + idx * num_states] /= sum;
        }
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
                                   float *var_a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= num_states) {
        return;
    }

    // --- Constants ---
    constexpr float SQRT_2PI = 2.5066282746310002f;
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    constexpr float ALPHA = 0.1f;
    constexpr float VAR_MIN = 1e-8f;  // Threshold for small variance
    constexpr float OUT_MIN = 1e-7f;  // Floor for output mean and variance

    // --- Inputs ---
    float scale = 1.0f;
    float shift = 0.0f;
    float mz = mu_z[col] * scale + shift;
    float varz = var_z[col] * (scale * scale);

    // --- Path 1: Deterministic case for small variance ---
    // Handles near-zero variance to avoid division by zero and catastrophic
    // cancellation.
    if (varz < VAR_MIN) {
        float out_mean, out_var, out_jcb;
        if (mz > 0.0f) {
            out_mean = mz;
            out_jcb = 1.0f;  // Derivative of CELU is 1 for x > 0
            out_var = varz;  // Propagate variance with (derivative)^2
        } else {
            // Derivative of CELU is exp(x/ALPHA) for x <= 0
            float exp_mz_div_alpha = expf(mz / ALPHA);
            out_mean = ALPHA * (exp_mz_div_alpha - 1.0f);
            out_jcb = exp_mz_div_alpha;
            out_var = varz * out_jcb *
                      out_jcb;  // Propagate variance via Taylor expansion
        }

        // Store back, applying final shift and floor
        mu_a[col] = fmaxf(out_mean + ALPHA, OUT_MIN);
        var_a[col] = fmaxf(out_var, OUT_MIN);
        jcb[col] = out_jcb * scale;
        return;
    }

    // --- Path 2: Full stochastic calculation for non-trivial variance ---

    // 1) Standardize input
    float sz = sqrtf(varz);
    float z = mz / sz;  // Standardized mean

    // 2) Pre-compute common Gaussian terms
    float exp_m_half_z2 = expf(-0.5f * z * z);
    float phi_z = exp_m_half_z2 / SQRT_2PI;  // Standard normal PDF: φ(z)

    // 3) Stably compute CDF and tail probability
    // tail_z = P[Z > z]
    float tail_z = 0.5f * erfcf(z * INV_SQRT2);
    // cdf_z = P[Z <= z] = 1 - tail_z. The below is more stable than 1.0 -
    // tail_z
    float cdf_z = 0.5f * erfcf(-z * INV_SQRT2);

    // 4) Compute key exponential terms stably using erfcx
    // The scaled complementary error function, erfcx(x) = exp(x^2)erfc(x),
    // avoids overflow that occurs when multiplying a large exp() by a small
    // erfc().
    float a = sz / ALPHA;
    float T1 = 0.5f * erfcxf((z + a) * INV_SQRT2) * exp_m_half_z2;
    float T2 = 0.5f * erfcxf((z + 2.0f * a) * INV_SQRT2) * exp_m_half_z2;

    // 5) Mean E[CELU(z)]
    // Rearranged formula to be more explicit about positive and negative parts
    float mean_d = mz * cdf_z + sz * phi_z + ALPHA * (T1 - tail_z);

    // 6) Second moment E[CELU(z)²]
    // Separated into E[max(0,Z)²] and terms from the negative part
    float E2_pos = (mz * mz + varz) * cdf_z + mz * sz * phi_z;
    float E2_neg = ALPHA * ALPHA * (T2 - 2.0f * T1 + tail_z);
    float E2 = E2_pos + E2_neg;

    // 7) Variance = E[X²] – E[X]²
    float var_d = E2 - mean_d * mean_d;

    // 8) Jacobian term (proportional to Cov[z, CELU(z)])
    float jcb_d = cdf_z + T1;

    // 9) Store back, applying final shift and floor
    mu_a[col] = fmaxf(mean_d + ALPHA, OUT_MIN);
    var_a[col] = fmaxf(var_d, OUT_MIN);
    jcb[col] = jcb_d * scale;
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
        float lambda = 0.1f;
        if (mu_z[col] > lambda) {
            mu_a[col] = lambda;
            jcb[col] = -0.001f;
            var_a[col] = 0.001f;
        } else {
            mu_a[col] = fmaxf(logf(1 + expf(mu_z[col])), 0.000001f);
            tmp = 1 / (1 + expf(-mu_z[col]));
            jcb[col] = tmp;
            var_a[col] = fmaxf(tmp * var_z[col] * tmp, 0.000001f);
        }
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

/**
 * @brief Extracts the odd-indexed elements from a full input stream.
 * * Each thread 'i' computes the source index as 2*i + 1 and copies the data.
 * This prepares the input for the aleatoric uncertainty branch (odd stream).
 */
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

__global__ void agvi_forward_combine_kernel(const float *d_even_stream_mu,
                                            const float *d_even_stream_var,
                                            const float *d_inner_output_mu,
                                            int half_size, float *d_output_mu,
                                            float *d_output_var,
                                            float *d_output_jcb, bool agvi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < half_size) {
        // Output mean comes from the even stream (epistemic part)
        d_output_mu[i] = d_even_stream_mu[i];

        // The Jacobian of the output of this layer with respect to the even
        // stream is 1.
        d_output_jcb[i] = 1.0f;

        // If AGVI is enabled, add the learned aleatoric variance
        if (agvi) {
            d_output_var[i] = d_even_stream_var[i] + d_inner_output_mu[i];
        } else {
            d_output_var[i] = d_even_stream_var[i];
        }
    }
}

__global__ void agvi_backward_kernel(
    const float *d_incoming_delta_mu, const float *d_incoming_delta_var,
    const float *d_stored_output_mu_a, const float *d_stored_output_var_a,
    const float *d_stored_inner_mu_a, const float *d_stored_inner_var_a,
    const float *d_stored_inner_jcb, const float *d_stored_even_var_a,
    const float *d_stored_even_jcb, int half_size, float *d_output_delta_mu,
    float *d_output_delta_var, bool overfit_mu, bool has_even_layer) {
    const float epsilon = 1.0e-6f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < half_size) {
        // --- 1. Load stored values from forward pass ---
        float var_zv =
            fmaxf(d_stored_output_var_a[i], epsilon);  // Total variance of Z+V
        float var_z = fmaxf(d_stored_even_var_a[i],
                            epsilon);  // Epistemic variance from even stream

        // Outputs of the inner (odd stream) activation
        float mu_v2_bar_tilde = d_stored_inner_mu_a[i];
        float var_v2_bar_tilde = fmaxf(d_stored_inner_var_a[i], epsilon);
        float jcb_v2_bar_tilde = d_stored_inner_jcb[i];

        float incoming_delta_mu = d_incoming_delta_mu[i];
        float incoming_delta_var = d_incoming_delta_var[i];

        // --- 2. Backward pass for the ODD stream (aleatoric part) ---
        // Prior moments for v^2 (aleatoric variance)
        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 = fmaxf(
            3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde,
            epsilon);

        // Moments for the noise term V ~ N(0, mu_v2)
        float var_v = mu_v2;

        // Posterior moments for V after observing the output
        float mu_v_pos = var_v * incoming_delta_mu;
        float var_v_pos =
            fmaxf(var_v + var_v * incoming_delta_var * var_v, epsilon);

        // Posterior moments for V^2
        float mu_v2_pos = mu_v_pos * mu_v_pos + var_v_pos;
        float var_v2_pos = 2.0f * var_v_pos * var_v_pos +
                           4.0f * var_v_pos * mu_v_pos * mu_v_pos;

        // Propagate updates back to V2_bar_tilde (pre-activation aleatoric
        // variance)
        float Jv2_bar_tilde = var_v2_bar_tilde / var_v2;
        float mu_v2_bar_tilde_pos =
            mu_v2_bar_tilde + Jv2_bar_tilde * (mu_v2_pos - mu_v2);
        float var_v2_bar_tilde_pos =
            var_v2_bar_tilde +
            Jv2_bar_tilde * Jv2_bar_tilde * (var_v2_pos - var_v2);

        // Final delta for the odd stream input
        float Jv2_bar = jcb_v2_bar_tilde / var_v2_bar_tilde;
        float odd_delta_mu = Jv2_bar * (mu_v2_bar_tilde_pos - mu_v2_bar_tilde);
        float odd_delta_var =
            Jv2_bar * Jv2_bar * (var_v2_bar_tilde_pos - var_v2_bar_tilde);

        int odd_idx = 2 * i + 1;
        d_output_delta_mu[odd_idx] =
            (isnan(odd_delta_mu) || isinf(odd_delta_mu)) ? 0.0f : odd_delta_mu;
        d_output_delta_var[odd_idx] =
            (isnan(odd_delta_var) || isinf(odd_delta_var)) ? 0.0f
                                                           : odd_delta_var;

        float mu_zv_pos = var_zv * incoming_delta_mu;
        float var_zv_pos = var_zv * incoming_delta_var * var_zv;

        // --- 3. Backward pass for the EVEN stream (epistemic part) ---
        float Jz = has_even_layer ? d_stored_even_jcb[i] : 1.0f;

        float even_delta_mu = Jz / var_zv * incoming_delta_mu;
        // Kalman gain for the variance delta
        float even_delta_var = Jz / var_zv * (var_zv_pos)*Jz / var_zv;

        // Kalman gain for the mean delta
        if (overfit_mu) {
            // Use epistemic variance only, for a more direct update to the mean
            even_delta_mu = Jz / var_z * (mu_zv_pos);
        }

        int even_idx = 2 * i;
        d_output_delta_mu[even_idx] =
            (isnan(even_delta_mu) || isinf(even_delta_mu)) ? 0.0f
                                                           : even_delta_mu;
        d_output_delta_var[even_idx] =
            (isnan(even_delta_var) || isinf(even_delta_var)) ? 0.0f
                                                             : even_delta_var;
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

    // We calculate stats for only the first item in the batch for debugging.
    int num_states_to_stat = hidden_size;

    // --- 1. Get statistics using loops (debugging only) ---
    if (num_states_to_stat > 0) {
        // Step A: Create host vectors to hold the data from the GPU.
        std::vector<float> host_mu_a(num_states_to_stat);
        std::vector<float> host_var_a(num_states_to_stat);

        // Step B: Copy the data from the GPU device to the CPU host.
        cudaMemcpy(host_mu_a.data(), cu_input_states->d_mu_a,
                   num_states_to_stat * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_var_a.data(), cu_input_states->d_var_a,
                   num_states_to_stat * sizeof(float), cudaMemcpyDeviceToHost);

        // Step C: Now that the data is on the CPU, use standard loops.
        // This code is copied from your Remax::forward CPU version.
        float sum_mu_z = 0.0f;
        for (float n : host_mu_a) {
            sum_mu_z += n;
        }
        float mean_mu_z = sum_mu_z / num_states_to_stat;

        float sum_var_z = 0.0f;
        for (float n : host_var_a) {
            sum_var_z += n;
        }
        float mean_var_z = sum_var_z / num_states_to_stat;

        float std_mu_z = 0.0f;
        float std_var_z = 0.0f;
        for (int i = 0; i < num_states_to_stat; i++) {
            std_mu_z += (host_mu_a[i] - mean_mu_z) * (host_mu_a[i] - mean_mu_z);
            std_var_z +=
                (host_var_a[i] - mean_var_z) * (host_var_a[i] - mean_var_z);
        }
        std_mu_z = sqrtf(std_mu_z / num_states_to_stat);
        std_var_z = sqrtf(std_var_z / num_states_to_stat);

        float min_mu_z = *std::min_element(host_mu_a.begin(), host_mu_a.end());
        float max_mu_z = *std::max_element(host_mu_a.begin(), host_mu_a.end());

        // Step D: Print the statistics.
        std::cout << "RemaxCuda layer (Debug Stats): mean_mu_z = " << mean_mu_z
                  << ", std_mu_z = " << std_mu_z
                  << ", mean_var_z = " << mean_var_z
                  << ", std_var_z = " << std_var_z << std::endl;

        std::cout << "RemaxCuda layer (Debug Stats): min_mu_z = " << min_mu_z
                  << ", max_mu_z = " << max_mu_z << std::endl;
    }
    // --- End of statistics calculation ---

    int total_num_states = batch_size * hidden_size;
    if (this->batch_size_ != batch_size) {
        this->batch_size_ = batch_size;
        this->deallocate_memory();
        this->allocate_memory(hidden_size, batch_size);
    }

    constexpr int THREADS = 256;
    constexpr int THREADS_BATCH = 16;
    constexpr int THREADS_HIDDEN = 16;
    unsigned int blocks = (total_num_states + THREADS - 1) / THREADS;

    mixture_relu_mean_var_cuda<<<blocks, THREADS>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, total_num_states,
        this->d_mu_m, this->d_jcb_m, this->d_var_m);

    // ... (rest of the function remains the same)
    unsigned int blocks_sum = (batch_size + THREADS - 1) / THREADS;
    compute_mean_var_sum_cuda<<<blocks_sum, THREADS>>>(
        this->d_mu_m, this->d_var_m, hidden_size, batch_size, this->d_mu_mt,
        this->d_var_mt);

    unsigned int hidden_blocks =
        (hidden_size + THREADS_HIDDEN - 1) / THREADS_HIDDEN;
    unsigned int batch_blocks =
        (batch_size + THREADS_BATCH - 1) / THREADS_BATCH;
    dim3 dim_grid_log(hidden_blocks, batch_blocks);
    dim3 dim_block_log(THREADS_HIDDEN, THREADS_BATCH);

    to_log_cuda<<<dim_grid_log, dim_block_log>>>(
        this->d_mu_m, this->d_var_m, hidden_size, batch_size, this->d_mu_log_m,
        this->d_var_log_m);

    unsigned int blocks_log_mt = (batch_size + THREADS - 1) / THREADS;
    dim3 dim_grid_log_mt(1, blocks_log_mt);
    dim3 dim_block_log_mt(1, THREADS);
    to_log_cuda<<<dim_grid_log_mt, dim_block_log_mt>>>(
        this->d_mu_mt, this->d_var_mt, 1, batch_size, this->d_mu_log_mt,
        this->d_var_log_mt);

    compute_cov_log_m_mt_cuda<<<dim_grid_log, dim_block_log>>>(
        this->d_mu_m, this->d_var_m, this->d_mu_mt, hidden_size, batch_size,
        this->d_cov_log_m_mt);

    compute_remax_mean_var_cuda<<<blocks_sum, THREADS>>>(
        this->d_mu_log_m, this->d_var_log_m, this->d_mu_log_mt,
        this->d_var_log_mt, this->d_cov_log_m_mt, hidden_size, batch_size,
        cu_output_states->d_mu_a, cu_output_states->d_var_a);

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

AGVICuda::AGVICuda(std::unique_ptr<BaseLayer> odd_layer,
                   std::unique_ptr<BaseLayer> even_layer, bool overfit_mu,
                   bool agvi)
    : m_odd_layer(std::move(odd_layer)),
      m_even_layer(std::move(even_layer)),
      m_overfit_mu(overfit_mu),
      m_agvi(agvi) {}

AGVICuda::~AGVICuda() {
    // The unique_ptrs for layers and HiddenStateCuda members will automatically
    // handle the deallocation of their respective resources.
}

std::string AGVICuda::get_layer_info() const {
    std::string even_name =
        m_even_layer ? m_even_layer->get_layer_name() : "Identity";
    return "AGVICuda(odd=" + m_odd_layer->get_layer_name() +
           ", even=" + even_name + ")";
}

std::string AGVICuda::get_layer_name() const { return "AGVICuda"; }

LayerType AGVICuda::get_layer_type() const { return LayerType::AGVI; }

std::unique_ptr<BaseLayer> AGVICuda::to_host() {
    auto *odd_cuda_layer = dynamic_cast<BaseLayerCuda *>(m_odd_layer.get());
    if (!odd_cuda_layer) {
        throw std::runtime_error(
            "AGVICuda::to_host(): Failed to cast odd_layer to BaseLayerCuda.");
    }
    auto host_odd_layer = odd_cuda_layer->to_host();

    std::shared_ptr<BaseLayer> host_even_layer = nullptr;
    if (m_even_layer) {
        auto *even_cuda_layer =
            dynamic_cast<BaseLayerCuda *>(m_even_layer.get());
        if (!even_cuda_layer) {
            throw std::runtime_error(
                "AGVICuda::to_host(): Failed to cast even_layer to "
                "BaseLayerCuda.");
        }
        host_even_layer = even_cuda_layer->to_host();
    }

    auto host_agvi = std::make_unique<AGVI>(std::move(host_odd_layer),
                                            std::move(host_even_layer),
                                            m_overfit_mu, m_agvi);
    return host_agvi;
}

void AGVICuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states) {
    // 1. Cast states to CUDA-specific types
    auto *cu_input_states = dynamic_cast<HiddenStateCuda *>(&input_states);
    auto *cu_output_states = dynamic_cast<HiddenStateCuda *>(&output_states);

    // 2. Calculate sizes
    int total_elements =
        cu_input_states->actual_size * cu_input_states->block_size;
    int half_size = total_elements / 2;
    size_t half_bytes = half_size * sizeof(float);

    // 3. Set output dimensions (output is half the size of the input)
    cu_output_states->actual_size = cu_input_states->actual_size / 2;
    cu_output_states->block_size = cu_input_states->block_size;
    this->input_size = cu_input_states->actual_size;
    this->output_size = cu_output_states->actual_size;

    // 4. Prepare temporary states for the odd stream processing
    HiddenStateCuda odd_input_states;
    odd_input_states.block_size = cu_input_states->block_size;
    odd_input_states.actual_size = cu_input_states->actual_size / 2;
    cudaMalloc(&odd_input_states.d_mu_a, half_bytes);
    cudaMalloc(&odd_input_states.d_var_a, half_bytes);
    cudaMalloc(&odd_input_states.d_jcb, half_bytes);

    // 5. Setup CUDA launch configuration
    unsigned int threads = 256;
    unsigned int blocks = (half_size + threads - 1) / threads;

    // 6. Extract odd stream from input
    agvi_extract_odd_stream_kernel<<<blocks, threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, half_size, odd_input_states.d_mu_a,
        odd_input_states.d_var_a, odd_input_states.d_jcb);

    // 7. Process the odd stream through its activation layer
    m_stored_inner_output_states.actual_size =
        half_size / cu_input_states->block_size;
    m_stored_inner_output_states.block_size = cu_input_states->block_size;
    cudaMalloc(&m_stored_inner_output_states.d_mu_a, half_bytes);
    cudaMalloc(&m_stored_inner_output_states.d_var_a, half_bytes);
    cudaMalloc(&m_stored_inner_output_states.d_jcb, half_bytes);
    m_odd_layer->forward(odd_input_states, m_stored_inner_output_states,
                         temp_states);

    // 8. Process the even stream
    m_stored_even_output_states.actual_size =
        half_size / cu_input_states->block_size;
    m_stored_even_output_states.block_size = cu_input_states->block_size;
    cudaMalloc(&m_stored_even_output_states.d_mu_a, half_bytes);
    cudaMalloc(&m_stored_even_output_states.d_var_a, half_bytes);
    cudaMalloc(&m_stored_even_output_states.d_jcb, half_bytes);

    if (m_even_layer) {
        // If an even layer is provided, extract even inputs and process them
        HiddenStateCuda even_input_states;
        even_input_states.actual_size = half_size / cu_input_states->block_size;
        even_input_states.block_size = cu_input_states->block_size;
        cudaMalloc(&even_input_states.d_mu_a, half_bytes);
        cudaMalloc(&even_input_states.d_var_a, half_bytes);
        cudaMalloc(&even_input_states.d_jcb, half_bytes);

        // Use the general-purpose split kernel to get even inputs
        split_stream_kernel<<<blocks, threads>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a,
            cu_input_states->d_jcb, half_size, even_input_states.d_mu_a,
            even_input_states.d_var_a, even_input_states.d_jcb,
            odd_input_states.d_mu_a, odd_input_states.d_var_a,
            odd_input_states.d_jcb);

        m_even_layer->forward(even_input_states, m_stored_even_output_states,
                              temp_states);

        cudaFree(even_input_states.d_mu_a);
        cudaFree(even_input_states.d_var_a);
        cudaFree(even_input_states.d_jcb);
        even_input_states.d_mu_a = nullptr;
        even_input_states.d_var_a = nullptr;
        even_input_states.d_jcb = nullptr;
    } else {
        // Identity: just extract the even stream directly into the stored state
        split_stream_kernel<<<blocks, threads>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a,
            cu_input_states->d_jcb, half_size,
            m_stored_even_output_states.d_mu_a,
            m_stored_even_output_states.d_var_a,
            m_stored_even_output_states.d_jcb, odd_input_states.d_mu_a,
            odd_input_states.d_var_a, odd_input_states.d_jcb);
    }

    // 9. Combine results into the final output using the CORRECTED kernel
    agvi_forward_combine_kernel<<<blocks, threads>>>(
        m_stored_even_output_states.d_mu_a, m_stored_even_output_states.d_var_a,
        m_stored_inner_output_states.d_mu_a, half_size,
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, m_agvi);

    // 10. Store pointers to final output states for the backward pass
    d_stored_output_mu_a = cu_output_states->d_mu_a;
    d_stored_output_var_a = cu_output_states->d_var_a;

    // 11. Clean up temporary odd input memory
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
    // 1. Cast states to CUDA-specific types
    auto *cu_input_delta = dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    auto *cu_output_delta =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    // 2. Calculate size for kernel launch
    int half_size = this->output_size * cu_input_delta->block_size;

    // 3. Set up CUDA launch configuration
    unsigned int threads = 256;
    unsigned int blocks = (half_size + threads - 1) / threads;

    // 4. Launch the backward kernel
    agvi_backward_kernel<<<blocks, threads>>>(
        cu_input_delta->d_delta_mu, cu_input_delta->d_delta_var,
        d_stored_output_mu_a, d_stored_output_var_a,
        m_stored_inner_output_states.d_mu_a,
        m_stored_inner_output_states.d_var_a,
        m_stored_inner_output_states.d_jcb, m_stored_even_output_states.d_var_a,
        m_stored_even_output_states.d_jcb, half_size,
        cu_output_delta->d_delta_mu, cu_output_delta->d_delta_var, m_overfit_mu,
        m_even_layer != nullptr);

    // 5. Update output delta metadata
    cu_output_delta->actual_size = this->input_size;
    cu_output_delta->block_size = cu_input_delta->block_size;

    // 6. Free memory that was stored for the backward pass
    cudaFree(m_stored_inner_output_states.d_mu_a);
    cudaFree(m_stored_inner_output_states.d_var_a);
    cudaFree(m_stored_inner_output_states.d_jcb);
    m_stored_inner_output_states.d_mu_a = nullptr;
    m_stored_inner_output_states.d_var_a = nullptr;
    m_stored_inner_output_states.d_jcb = nullptr;

    cudaFree(m_stored_even_output_states.d_mu_a);
    cudaFree(m_stored_even_output_states.d_var_a);
    cudaFree(m_stored_even_output_states.d_jcb);
    m_stored_even_output_states.d_mu_a = nullptr;
    m_stored_even_output_states.d_var_a = nullptr;
    m_stored_even_output_states.d_jcb = nullptr;

    // Reset non-owning pointers to prevent dangling references
    d_stored_output_mu_a = nullptr;
    d_stored_output_var_a = nullptr;
}
