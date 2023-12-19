///////////////////////////////////////////////////////////////////////////////
// File:         fc_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 03, 2023
// Updated:      December 13, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/fc_cuda.cuh"

__global__ void fwd_mean_var(float const *mu_w, float const *var_w,
                             float const *mu_b, float const *var_b,
                             const float *mu_a, const float *var_a,
                             size_t input_size, size_t output_size,
                             int batch_size, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    float mu_a_tmp = 0.0f;
    float var_a_tmp = 0.0f;
    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            mu_a_tmp = mu_a[input_size * col + i];
            var_a_tmp = var_a[input_size * col + i];

            if (mu_a_tmp != 0) {
                sum_mu += mu_w[row * input_size + i] * mu_a_tmp;
                sum_var +=
                    (mu_w[row * input_size + i] * mu_w[row * input_size + i] +
                     var_w[row * input_size + i]) *
                        var_a_tmp +
                    var_w[row * input_size + i] * mu_a_tmp * mu_a_tmp;
            }
        }
        mu_z[col * output_size + row] = sum_mu + mu_b[row];
        var_z[col * output_size + row] = sum_var + var_b[row];
    }
}

__global__ void fwd_full_cov(float const *mu_w, float const *var_a_f,
                             size_t input_size, size_t output_size,
                             int batch_size, float *var_z_fp)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tu = 0, k = 0;
    float sum = 0.0f;
    float var_a_in = 0.0f;

    if (col <= (row % output_size) && row < output_size * batch_size) {
        for (int i = 0; i < input_size * input_size; i++) {
            int row_in = i / input_size;
            int col_in = i % input_size;
            if (row_in > col_in)  // lower triangle
            {
                tu = (input_size * col_in - ((col_in * (col_in + 1)) / 2) +
                      row_in);
            } else {
                tu = (input_size * row_in - ((row_in * (row_in + 1)) / 2) +
                      col_in);
            }
            var_a_in = var_a_f[tu + (row / output_size) *
                                        (input_size * (input_size + 1)) / 2];

            sum += mu_w[i % input_size + (row % output_size) * input_size] *
                   var_a_in *
                   mu_w[i / input_size + (col % output_size) * input_size];
        }
        k = output_size * col - ((col * (col + 1)) / 2) + row % output_size +
            (row / output_size) * (((output_size + 1) * output_size) / 2);
        var_z_fp[k] = sum;
    }
}

__global__ void fwd_full_var(float const *mu_w, float const *var_w,
                             float const *var_b, float const *mu_a,
                             float const *var_a, float const *var_z_fp,
                             size_t input_size, size_t output_size,
                             int batch_size, float *var_z, float *var_z_f)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    float final_sum = 0;
    int k;

    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            sum += var_w[row * input_size + i] * var_a[input_size * col + i] +
                   var_w[row * input_size + i] * mu_a[input_size * col + i] *
                       mu_a[input_size * col + i];
        }
        k = output_size * row - (row * (row - 1)) / 2 +
            col * (output_size * (output_size + 1)) / 2;

        final_sum = sum + var_b[row] + var_z_fp[k];

        var_z[col * output_size + row] = final_sum;
    }
}

__global__ void bwd_delta_z(float const *mu_w, float const *jcb,
                            float const *delta_mu_out,
                            float const *delta_var_out, size_t input_size,
                            size_t output_size, int batch_size,
                            float *delta_mu_in, float *delta_var_in)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    if (col < batch_size && row < input_size) {
        for (int i = 0; i < output_size; i++) {
            sum_mu += mu_w[input_size * i + row] *
                      delta_mu_out[col * output_size + i];

            sum_var += mu_w[input_size * i + row] *
                       delta_var_out[col * output_size + i] *
                       mu_w[input_size * i + row];
        }
        delta_mu_in[col * input_size + row] =
            sum_mu * jcb[col * input_size + row];

        delta_var_in[col * input_size + row] =
            sum_var * jcb[col * input_size + row] * jcb[col * input_size + row];
    }
}

__global__ void bwd_delta_w(float const *var_w, float const *mu_a,
                            float const *delta_mu_out,
                            float const *delta_var_out, size_t input_size,
                            size_t output_size, int batch_size,
                            float *delta_mu_w, float *delta_var_w)
/**/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < output_size && row < input_size) {
        for (int i = 0; i < batch_size; i++) {
            sum_mu += mu_a[input_size * i + row] *
                      delta_mu_out[output_size * i + col];

            sum_var += mu_a[input_size * i + row] * mu_a[input_size * i + row] *
                       delta_var_out[output_size * i + col];
        }

        delta_mu_w[col * input_size + row] =
            sum_mu * var_w[col * input_size + row];

        delta_var_w[col * input_size + row] = sum_var *
                                              var_w[col * input_size + row] *
                                              var_w[col * input_size + row];
    }
}

__global__ void bwd_delta_b(float const *var_b, float const *delta_mu_out,
                            float const *delta_var_out, size_t input_size,
                            size_t output_size, int batch_size,
                            float *delta_mu_b, float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < output_size && row < input_size) {
        for (int i = 0; i < batch_size; i++) {
            sum_mu += delta_mu_out[input_size * i + row];
            sum_var += delta_var_out[input_size * i + row];
        }

        delta_mu_b[col * input_size + row] =
            sum_mu * var_b[col * input_size + row];

        delta_var_b[col * input_size + row] = sum_var *
                                              var_b[col * input_size + row] *
                                              var_b[col * input_size + row];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Fully Connected Layer
////////////////////////////////////////////////////////////////////////////////

LinearCuda::LinearCuda(size_t ip_size, size_t op_size, float gain_weight,
                       float gain_bias, std::string method)
    : gain_w(gain_weight),
      gain_b(gain_bias),
      init_method(method)
/*
 */
{
    this->input_size = ip_size;
    this->output_size = op_size;

    // Initalize weights and bias
    this->init_weight_bias();
}

LinearCuda::~LinearCuda() {}

std::string LinearCuda::get_layer_info() const
/*
 */
{
    return "Linear(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LinearCuda::get_layer_name() const
/*
 */
{
    return "LinearCuda";
}

LayerType LinearCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LinearCuda::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->input_size, this->output_size);
}

void LinearCuda::allocate_param_delta()
/*
 */
{
    cudaMalloc(&this->d_delta_mu_w,
               this->input_size * this->output_size * sizeof(float));
    cudaMalloc(&this->d_delta_var_w,
               this->input_size * this->output_size * sizeof(float));
    cudaMalloc(&this->d_delta_mu_b, this->output_size * sizeof(float));
    cudaMalloc(&this->d_delta_var_b, this->output_size * sizeof(float));
}

void LinearCuda::forward(HiddenStateCuda &input_states,
                         HiddenStateCuda &output_states,
                         TempStateCuda &temp_state)
/*
 */
{
    // Gert batch size
    int batch_size = input_states.block_size;

    // Forward pass
    unsigned int grid_rows = (this->output_size + this->num_cuda_threads - 1) /
                             this->num_cuda_threads;
    unsigned int grid_cols =
        (batch_size + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 grid_dim(grid_cols, grid_rows);
    dim3 block_dim(this->num_cuda_threads, this->num_cuda_threads);

    fwd_mean_var<<<grid_dim, block_dim>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        input_states.d_mu_a, input_states.d_var_a, this->input_size,
        this->output_size, input_states.block_size, output_states.d_mu_z,
        output_states.d_var_z);

    // Lazy initialization
    if (this->bwd_states.size == 0 && this->training) {
        this->bwd_states.size = input_states.actual_size * batch_size;
        this->bwd_states.allocate_memory();
    }

    // Update backward state for inferring parameres
    if (this->training) {
        int act_size = input_states.actual_size * batch_size;
        unsigned int blocks =
            (act_size + this->num_cuda_threads - 1) / this->num_cuda_threads;

        fill_bwd_states_on_device<<<blocks, this->num_cuda_threads>>>(
            input_states.d_mu_a, input_states.d_jcb, act_size,
            this->bwd_states.d_mu_a, this->bwd_states.d_jcb);

        int out_size = this->output_size * batch_size;
        unsigned int out_blocks =
            (out_size + this->num_cuda_threads - 1) / this->num_cuda_threads;

        fill_output_states_on_device<<<out_blocks, this->num_cuda_threads>>>(
            output_states.d_mu_z, output_states.d_var_z, out_size,
            output_states.d_mu_a, output_states.d_jcb, output_states.d_var_a);
    }

    // Update number of actual states.
    output_states.size = this->output_size * batch_size;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;
}

void LinearCuda::state_backward(BackwardStateCuda &next_bwd_states,
                                DeltaStateCuda &input_delta_states,
                                DeltaStateCuda &output_delta_states,
                                TempStateCuda &temp_states)
/*
 */
{
    // Initialization
    int batch_size = input_delta_states.block_size;

    // Compute inovation vector
    unsigned int grid_row = (this->input_size + this->num_cuda_threads - 1) /
                            this->num_cuda_threads;
    unsigned int grid_col =
        (batch_size + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 grid_dim(grid_col, grid_row);
    dim3 block_dim(this->num_cuda_threads, this->num_cuda_threads);

    bwd_delta_z<<<grid_dim, block_dim>>>(
        this->d_mu_w, this->bwd_states.d_jcb, input_delta_states.d_delta_mu,
        input_delta_states.d_delta_var, this->input_size, this->output_size,
        batch_size, output_delta_states.d_delta_mu,
        output_delta_states.d_delta_var);
}

void LinearCuda::param_backward(BackwardStateCuda &bwd_states,
                                DeltaStateCuda &delta_states,
                                TempStateCuda &temp_states)
/*
 */
{
    // Initalization
    int batch_size = delta_states.block_size;
    dim3 block_dim(this->num_cuda_threads, this->num_cuda_threads);

    // Updated values for weights
    unsigned int grid_row_w = (this->input_size + this->num_cuda_threads - 1) /
                              this->num_cuda_threads;
    unsigned int grid_col_w = (this->output_size + this->num_cuda_threads - 1) /
                              this->num_cuda_threads;
    dim3 grid_dim_w(grid_col_w, grid_row_w);

    bwd_delta_w<<<grid_dim_w, block_dim>>>(
        this->d_var_w, bwd_states.d_mu_a, delta_states.d_delta_mu,
        delta_states.d_delta_var, this->input_size, this->output_size,
        batch_size, this->d_delta_mu_w, this->d_delta_var_w);

    // Updated values for biases
    unsigned int grid_row_b = (this->output_size + this->num_cuda_threads - 1) /
                              this->num_cuda_threads;
    dim3 grid_dim_b(1, grid_row_b);

    bwd_delta_b<<<grid_dim_b, block_dim>>>(
        this->d_var_b, delta_states.d_delta_mu, delta_states.d_delta_var,
        this->input_size, this->output_size, batch_size, this->d_delta_mu_b,
        this->d_delta_var_b);
}
