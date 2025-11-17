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

__global__ void even_exp_mean_var_cuda(float const *mu_z, float const *var_z,
                                       float const *jcb_z, int num_states,
                                       float *mu_a, float *var_a, float *jcb_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        if (col % 2 == 0) {
            mu_a[col] = mu_z[col];
            var_a[col] = var_z[col];
            jcb_a[col] = jcb_z[col];
        } else {
            mu_a[col] = expf(mu_z[col] + 0.5f * var_z[col]);
            var_a[col] =
                expf(2.0f * mu_z[col] + var_z[col]) * (expf(var_z[col]) - 1.0f);
            jcb_a[col] = mu_a[col];
        }
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
/// EvenExp
////////////////////////////////////////////////////////////////////////////////
EvenExpCuda::EvenExpCuda() {}
EvenExpCuda::~EvenExpCuda() {}

std::string EvenExpCuda::get_layer_info() const
/*
 */
{
    return "EvenExp()";
}

std::string EvenExpCuda::get_layer_name() const
/*
 */
{
    return "EvenExpCuda";
}

LayerType EvenExpCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void EvenExpCuda::forward(BaseHiddenStates &input_states,
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

    even_exp_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, num_states, cu_output_states->d_mu_a,
        cu_output_states->d_var_a, cu_output_states->d_jcb);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> EvenExpCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<EvenExp>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}
