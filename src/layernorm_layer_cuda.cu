#include "../include/layernorm_layer_cuda.cuh"
#include "../include/param_init.h"

__global__ void layernorm_stat_mean_var_cuda(float const *mu_a,
                                             float const *var_a, int ni,
                                             int batch_size, float *mu_s,
                                             float *var_s)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < batch_size) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < ni; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col * ni + i];
            sum_var += var_a[col * ni + i];
        }
        mu_s[col] = sum_mu / ni;
        var_s[col] = sum_var;
    }
}

__global__ void layernorm_sample_var_cuda(float const *mu_a, float const *mu_s,
                                          float const *var_s, int ni,
                                          int batch_size, float *var_sample)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < ni; i++) {
            sum += (mu_a[col * ni + i] - mu_s[col]) *
                   (mu_a[col * ni + i] - mu_s[col]);
        }
        var_sample[col] = (sum + var_s[col]) / (ni - 1);
    }
}

__global__ void layernorm_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, bool bias, float epsilon, int ni,
    int B, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < ni && row < B) {
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[row] + epsilon);
        int idx = col + row * ni;
        float mu_w_term = mu_w[col];
        float mu_a_term = mu_a[idx];
        float mu_ra_term = mu_ra[row];
        float mu_a_tilde = mu_a_term - mu_ra_term;

        float tmp_mu = inv_sqrt_var_ra * mu_a_tilde * mu_w_term;
        float tmp_var = inv_sqrt_var_ra * inv_sqrt_var_ra *
                        (var_a[idx] * (mu_w_term * mu_w_term + var_w[col]) +
                         var_w[col] * mu_a_tilde * mu_a_tilde);

        mu_z[idx] = bias ? tmp_mu + mu_b[col] : tmp_mu;
        var_z[idx] = bias ? tmp_var + var_b[col] : tmp_var;
    }
}

__global__ void layernorm2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, bool bias, float epsilon, int wihi,
    int m, int k, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi * fi, m = B
    {
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[row] + epsilon);
        float mu_ra_term = mu_ra[row];
        int idx = col + row * k;
        int div_idx = col / wihi;
        float mu_w_term = mu_w[div_idx];
        float mu_a_term = mu_a[idx];
        float mu_a_tilde = mu_a_term - mu_ra_term;

        float tmp_mu_z = inv_sqrt_var_ra * mu_a_tilde * mu_w_term;
        float tmp_var_z =
            inv_sqrt_var_ra * inv_sqrt_var_ra *
            (var_a[idx] * (mu_w_term * mu_w_term + var_w[div_idx]) +
             var_w[div_idx] * mu_a_tilde * mu_a_tilde);

        mu_z[idx] = bias ? tmp_mu_z + mu_b[div_idx] : tmp_mu_z;
        var_z[idx] = bias ? tmp_var_z + var_b[div_idx] : tmp_var_z;
    }
}

////
// Layer Norm's backward
////
__global__ void layernorm_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float tmp = (1.0f / sqrtf(var_hat[row] + epsilon)) * mu_w[col] *
                    jcb[col + row * ni];

        delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];
        delta_var[col + row * ni] = tmp * delta_var_out[col + row * ni] * tmp;
    }
}

__global__ void layernorm_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_hat,
    float const *var_hat, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int ni, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = (1.0f / sqrtf(var_hat[i] + epsilon)) *
                      (mu_a[col + i * ni] - mu_hat[i]) * var_w[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

__global__ void layernorm_bwd_delta_b_cuda(float const *var_b,
                                           float const *delta_mu_out,
                                           float const *delta_var_out,
                                           float epsilon, int ni,
                                           int batch_size, float *delta_mu_b,
                                           float *delta_var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = var_b[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

__global__ void layernorm2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int wihi, int fi, int batch_size, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi * fi;
    if (col < k && row < batch_size)  // k = wihi * fi, m = B
    {
        float tmp = (1 / sqrtf(var_hat[row] + epsilon)) * mu_w[col / wihi] *
                    jcb[col + row * k];

        delta_mu[col + row * k] = tmp * delta_mu_out[col + row * k];
        delta_var[col + row * k] = tmp * delta_var_out[col + row * k] * tmp;
    }
}

__global__ void layernorm2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_ra,
    float const *var_ra, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int wihi, int fi, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi * fi;
    if (col < k && row < batch_size)  // k = wihi*fi, m = B
    {
        float A = (1.0f / sqrtf(var_ra[row] + epsilon)) *
                  (mu_a[col + row * k] - mu_ra[row]) * var_w[col / wihi];
        delta_mu_w[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_w[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

__global__ void layernorm2d_bwd_delta_b_cuda(float const *var_b,
                                             float const *delta_mu_out,
                                             float const *delta_var_out,
                                             float epsilon, int wihi, int fi,
                                             int m, float *delta_mu_b,
                                             float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi * fi;
    if (col < k && row < m)  // k = wihi*f, m = B
    {
        float A = var_b[col / wihi];
        delta_mu_b[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_b[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

__global__ void delta_param_sum_cuda(float const *delta_mu_e,
                                     float const *delta_var_e, int wihi, int fi,
                                     int batch_size, float *delta_mu,
                                     float *delta_var) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < fi) {
        float sum_delta_mu = 0.0f;
        float sum_delta_var = 0.0f;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi * B
        {
            sum_delta_mu +=
                delta_mu_e[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sum_delta_var +=
                delta_var_e[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        delta_mu[col] = sum_delta_mu;
        delta_var[col] = sum_delta_var;
    }
}

////////////////////////////////////////////////////////////////////////////////
//// Layer Norm
////////////////////////////////////////////////////////////////////////////////
LayerNormCuda::LayerNormCuda(const std::vector<int> &normalized_shape,
                             float eps, bool bias)
/*
 */
{
    this->normalized_shape = normalized_shape;
    this->epsilon = eps;
    this->bias = bias;
    if (this->normalized_shape.size() == 1) {
        this->input_size = this->normalized_shape[0];
        this->output_size = normalized_shape[0];
    } else if (this->normalized_shape.size() == 3) {
        this->in_channels = this->normalized_shape[0];
        this->in_width = this->normalized_shape[1];
        this->in_height = this->normalized_shape[2];
        this->out_channels = this->normalized_shape[0];
        this->out_width = this->normalized_shape[1];
        this->out_height = this->normalized_shape[2];
        this->input_size = this->in_channels * this->in_width * this->in_height;
        this->output_size =
            this->out_channels * this->out_width * this->out_height;
    } else {
        throw std::runtime_error(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Normalized shape provided are not supported.");
    }
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
}

LayerNormCuda::~LayerNormCuda()
/*
 */
{
    this->deallocate_running_mean_var();
}

void LayerNormCuda::deallocate_running_mean_var() {
    if (d_mu_ra != nullptr) {
        cudaFree(d_mu_ra);
    }
    if (d_var_ra != nullptr) {
        cudaFree(d_var_ra);
    }
}

std::string LayerNormCuda::get_layer_info() const
/*
 */
{
    return "LayerNorm()";
}

std::string LayerNormCuda::get_layer_name() const
/*
 */
{
    return "LayerNormCuda";
}

LayerType LayerNormCuda::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void LayerNormCuda::init_weight_bias()
/*
 */
{
    int num_features = this->normalized_shape[0];
    this->num_weights = this->normalized_shape[0];
    this->num_biases = this->bias ? this->normalized_shape[0] : 0;
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_norm("", 1.0f, 1.0f, num_features, num_features,
                              this->num_weights, this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void LayerNormCuda::allocate_running_mean_var()
/*
 */
{
    this->mu_ra.resize(this->_batch_size, 0.0f);
    this->var_ra.resize(this->_batch_size, 1.0f);

    // Memory aligment
    unsigned int size_batch_size =
        ((this->_batch_size + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    cudaMalloc(&this->d_mu_ra, size_batch_size * sizeof(float));
    cudaMalloc(&this->d_var_ra, size_batch_size * sizeof(float));
    this->running_mean_var_to_device();
    CHECK_LAST_CUDA_ERROR();
}

void LayerNormCuda::running_mean_var_to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_ra, this->mu_ra.data(),
               this->mu_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_ra, this->var_ra.data(),
               this->var_ra.size() * sizeof(float), cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();
}

void LayerNormCuda::running_mean_var_to_host()
/*
 */
{
    cudaMemcpy(this->mu_ra.data(), this->d_mu_ra,
               this->mu_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ra.data(), this->d_var_ra,
               this->var_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}

void LayerNormCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states)
/*
 */
{
    // Checkout input size
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    int batch_size = input_states.block_size;

    if (this->_batch_size < batch_size) {
        this->_batch_size = batch_size;
        this->set_cap_factor_udapte(batch_size);
        this->deallocate_running_mean_var();
        this->allocate_running_mean_var();
    }

    int num_threads = this->num_cuda_threads;
    unsigned int grid_size_ra = (batch_size + num_threads - 1) / num_threads;
    dim3 block_dim(num_threads, num_threads);

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
    unsigned int grid_col = (this->input_size + num_threads - 1) / num_threads;
    dim3 grid_size(grid_col, grid_row);

    layernorm_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->input_size,
        batch_size, this->d_mu_ra, cu_temp_states->d_tmp_2);

    layernorm_sample_var_cuda<<<grid_size_ra, num_threads>>>(
        cu_input_states->d_mu_a, this->d_mu_ra, cu_temp_states->d_tmp_2,
        this->input_size, batch_size, this->d_var_ra);

    if (this->normalized_shape.size() == 1) {
        layernorm_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_mu_ra,
            this->d_var_ra, this->bias, this->epsilon, this->input_size,
            batch_size, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        int wihi = this->in_height * this->in_width;
        layernorm2d_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_mu_ra,
            this->d_var_ra, this->bias, this->epsilon, wihi, batch_size,
            this->input_size, cu_output_states->d_mu_a,
            cu_output_states->d_var_a);
    }

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void LayerNormCuda::backward(BaseDeltaStates &input_delta_states,
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

    // Initialization
    int batch_size = input_delta_states.block_size;
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
    unsigned int grid_col = (this->input_size + num_threads - 1) / num_threads;
    dim3 grid_size(grid_col, grid_row);

    if (state_udapte) {
        if (this->normalized_shape.size() == 1) {
            layernorm_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
                this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_ra,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon,
                this->input_size, batch_size,
                cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
        } else {
            int wihi = this->in_height * this->in_width;

            layernorm2d_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
                this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_ra,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon, wihi,
                this->in_channels, batch_size,
                cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
        }
    }

    if (param_update) {
        TempStateCuda *cu_temp_states =
            dynamic_cast<TempStateCuda *>(&temp_states);

        unsigned int grid_col_p =
            (this->input_size + num_threads - 1) / num_threads;

        if (this->normalized_shape.size() == 1) {
            layernorm_bwd_delta_w_cuda<<<grid_col_p, num_threads>>>(
                this->d_var_w, cu_next_bwd_states->d_mu_a, this->d_mu_ra,
                this->d_var_ra, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon,
                this->input_size, batch_size, this->d_delta_mu_w,
                this->d_delta_var_w);

            if (this->bias) {
                layernorm_bwd_delta_b_cuda<<<grid_col_p, num_threads>>>(
                    this->d_var_b, cu_input_delta_states->d_delta_mu,
                    cu_input_delta_states->d_delta_var, this->epsilon,
                    this->input_size, batch_size, this->d_delta_mu_b,
                    this->d_delta_var_b);
            }

        } else {
            int wihi = this->in_height * this->in_width;
            unsigned int grid_row_p =
                (batch_size + num_threads - 1) / num_threads;
            dim3 dim_grid_p(grid_col_p, grid_row_p);
            unsigned int sum_grid_size =
                (this->in_channels + num_threads - 1) / num_threads;

            // Weights
            // TODO: Not sure if it should be batch_size or batch_size * fi
            layernorm2d_bwd_delta_w_cuda<<<dim_grid_p, block_dim>>>(
                this->d_var_w, cu_next_bwd_states->d_mu_a, this->d_mu_ra,
                this->d_var_ra, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->epsilon, wihi,
                this->in_channels, batch_size, cu_temp_states->d_tmp_1,
                cu_temp_states->d_tmp_2);

            delta_param_sum_cuda<<<sum_grid_size, num_threads>>>(
                cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2, wihi,
                this->in_channels, batch_size, this->d_delta_mu_w,
                this->d_delta_var_w);

            // Biases
            if (this->bias) {
                layernorm2d_bwd_delta_b_cuda<<<dim_grid_p, block_dim>>>(
                    this->d_var_b, cu_input_delta_states->d_delta_mu,
                    cu_input_delta_states->d_delta_var, this->epsilon, wihi,
                    this->in_channels, batch_size, cu_temp_states->d_tmp_1,
                    cu_temp_states->d_tmp_2);

                delta_param_sum_cuda<<<sum_grid_size, num_threads>>>(
                    cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2, wihi,
                    this->in_channels, batch_size, this->d_delta_mu_b,
                    this->d_delta_var_b);
            }
        }
    }
}

std::unique_ptr<BaseLayer> LayerNormCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<LayerNorm>(
        this->normalized_shape, this->epsilon, this->bias);

    host_layer->mu_w = this->mu_w;
    host_layer->var_w = this->var_w;
    host_layer->mu_b = this->mu_b;
    host_layer->var_b = this->var_b;

    return host_layer;
}

std::tuple<std::vector<float>, std::vector<float>>
LayerNormCuda::get_running_mean_var()
/*
 */
{
    this->running_mean_var_to_host();
    return {this->mu_ra, this->var_ra};
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
LayerNormCuda::get_norm_mean_var() {
    this->running_mean_var_to_host();
    std::vector<std::vector<float>> mu_ras = {this->mu_ra};
    std::vector<std::vector<float>> var_ras = {this->var_ra};
    std::vector<std::vector<float>> mu_norms;
    std::vector<std::vector<float>> var_norms;
    return {mu_ras, var_ras, mu_norms, var_norms};
}

void LayerNormCuda::save(std::ofstream &file)
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
}

void LayerNormCuda::load(std::ifstream &file)
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

    this->num_weights = this->mu_w.size();
    this->num_biases = this->mu_b.size();
    if (this->training) {
        this->allocate_param_delta();
    }

    // Transfer data to device
    this->params_to_device();
}