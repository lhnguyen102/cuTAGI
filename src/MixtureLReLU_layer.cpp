#include "../include/MixtureLReLU_layer.h"

#include "../include/common.h"
#include "../include/custom_logger.h"

// #ifdef USE_CUDA
// #include "../include/linear_layer_cuda.cuh"
// #endif

void MixtureLReLU_fwd_mean_var(float slope,
                               std::vector<float> &mu_z, std::vector<float> &var_z,
                               int start_chunk, int end_chunk, size_t input_size,
                               int batch_size, std::vector<float> &mu_a, std::vector<float> &var_a)
/*Compute output moment for the leaky ReLU activation function

Args:
  mu_z: Mean of hidden units
    this->mu_z.resize(this->size, 0.0f);
    this->var_z.resize(this->size, 0.0f);

    this->deallocate_memory();
    this->allocate_memory();
  slope: Slope of the negative segment for z < 0
  mu_a: Mean of activation units
  start_chunk: Start index of the chunk
  end_chunk: End index of the chunk
  n: Input/output node
  k: Number of batches
*/
{
    float std_z, alpha, pdf_alpha, cdf_alpha;
    for (int i = start_chunk; i < end_chunk; i++)
    {
        // Reused components for moments calculations
        std_z = powf(var_z[i], 0.5);
        alpha = mu_z[i] / std_z;
        pdf_alpha = normpdf_cpu(alpha, 0.0f, 1.0f);
        cdf_alpha = normcdf_cpu(alpha);

        // Moments calculations (L. Alric, 2025)
        mu_a[i] = slope * mu_z[i] + (1 - slope) *
                                        (std_z * pdf_alpha + mu_z[i] * cdf_alpha);
        var_a[i] = powf(slope, 2) * (powf(mu_z[i], 2) + var_z[i]) + (1 - powf(slope, 2)) * ((mu_z[i] * std_z * pdf_alpha + (powf(mu_z[i], 2) + var_z[i]) * cdf_alpha)) - powf(mu_a[i], 2);
    }
}

void MixtureLReLU_fwd_mean_var_mp(float slope,
                                  std::vector<float> &mu_z, std::vector<float> &var_z,
                                  size_t input_size,
                                  int batch_size, unsigned int num_threads,
                                  std::vector<float> &mu_a, std::vector<float> &var_a)
/*Multi-processing verion of forward pass for fc layer
 */
{
    const int tot_ops = input_size * batch_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++)
    {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &slope, &mu_z, &var_z,
                              &input_size, &batch_size,
                              &mu_a, &var_a]
                             { MixtureLReLU_fwd_mean_var(slope, mu_z, var_z,
                                                         start_chunk, end_chunk, input_size,
                                                         batch_size, mu_a, var_a); });
    }

    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

void MixtureLReLU_bwd_delta_z(float slope,
                              std::vector<float> &mu_a, std::vector<float> &var_a,
                              std::vector<float> &mu_z_input, std::vector<float> &var_z_input,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t input_size,
                              int B, int start_chunk,
                              int end_chunk, std::vector<float> &delta_mu_z,
                              std::vector<float> &delta_var_z)
/*Compute input moments for the inverse of the leaky ReLU activation function
 */
{
    int ni = input_size;
    for (int j = start_chunk; j < end_chunk; j++)
    {
        int row = j / B;
        int col = j % B;
        int i = col * ni; // Not sure of this one...
        float mu_z, var_z, std_a, alpha, pdf_alpha, cdf_alpha;

        mu_a[i] = mu_a[i] + delta_mu[i];
        var_a[i] = var_a[i] + delta_var[i];

        // Reused components for moments calculations
        std_a = powf(var_a[i], 0.5);
        alpha = mu_a[i] / std_a;
        pdf_alpha = normpdf_cpu(alpha, 0.0f, 1.0f);
        cdf_alpha = normcdf_cpu(alpha);

        // Moments calculations (L. Alric, 2025)
        mu_z = slope * mu_a[i] + (1 - 1 / slope) *
                                     (std_a * pdf_alpha + mu_a[i] * cdf_alpha);
        var_z = powf(1 / slope, 2) * (powf(mu_a[i], 2) + var_a[i]) + (1 - powf(slope, 2)) * ((mu_a[i] * std_a * pdf_alpha + (powf(mu_a[i], 2) + var_a[i]) * cdf_alpha)) - powf(mu_z, 2);

        delta_mu_z[col * ni + row] = mu_z - mu_z_input[i];
        delta_var_z[col * ni + row] = var_z - var_z_input[i];
    }
}
// {
//     int ni = input_size;
//     for (int j = start_chunk; j < end_chunk; j++)
//     {
//         int row = j / B;
//         int col = j % B;
//         float sum_mu_z = 0.0f;
//         float sum_var_z = 0.0f;
//         for (int i = 0; i < no; i++)
//         {
//             sum_mu_z += mu_w[ni * i + row] * delta_mu[col * no + i];

//             sum_var_z += mu_w[ni * i + row] * delta_var[col * no + i] *
//                          mu_w[ni * i + row];
//         }

//         // NOTE: Compute directly inovation vector
//         delta_mu_z[col * ni + row] = sum_mu_z * jcb[col * ni + row];
//         delta_var_z[col * ni + row] =
//             sum_var_z * jcb[col * ni + row] * jcb[col * ni + row];
//     }
// }

void MixtureLReLU_bwd_delta_z_mp(float slope,
                                 std::vector<float> &mu_a, std::vector<float> &var_a,
                                 std::vector<float> &mu_z_input, std::vector<float> &var_z_input,
                                 std::vector<float> &delta_mu,
                                 std::vector<float> &delta_var, size_t input_size,
                                 int batch_size,
                                 unsigned int num_threads,
                                 std::vector<float> &delta_mu_z,
                                 std::vector<float> &delta_var_z)
/*
 */
{
    const int tot_ops = input_size * batch_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++)
    {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &slope, &mu_a, &var_a, &mu_z_input, &var_z_input, &delta_mu, &delta_var,
                              &input_size, &batch_size,
                              &delta_mu_z, &delta_var_z]
                             { MixtureLReLU_bwd_delta_z(slope, mu_a, var_a, mu_z_input, var_z_input, delta_mu, delta_var, input_size,
                                                        batch_size, start_chunk,
                                                        end_chunk, delta_mu_z, delta_var_z); });
    }

    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
}

MixtureLReLU::MixtureLReLU(size_t ip_size, float slope_segment)
    : slope(slope_segment)
/*
 */
{
}

MixtureLReLU::~MixtureLReLU() {}

std::string MixtureLReLU::get_layer_info() const
/*
 */
{
    return "MixtureLReLU(" + std::to_string(this->input_size) + ")";
}

std::string MixtureLReLU::get_layer_name() const
/*
 */
{
    return "MixtureLReLU";
}

LayerType MixtureLReLU::get_layer_type() const
/*
 */
{
    return LayerType::MixtureLReLU;
}

void MixtureLReLU::forward(BaseHiddenStates &input_states,
                           BaseHiddenStates &output_states,
                           BaseTempStates &temp_states)
/*
 */
{
    // Initialization
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    // Checkout input size
    if (this->input_size != input_states.actual_size)
    {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    // Forward pass
    if (this->num_threads > 1)
    {
        MixtureLReLU_fwd_mean_var_mp(this->slope,
                                     input_states.mu_a, input_states.var_a,
                                     this->input_size, batch_size,
                                     this->num_threads,
                                     output_states.mu_a, output_states.var_a);
    }
    else
    {
        int start_chunk = 0;
        int end_chunk = this->output_size * batch_size;
        MixtureLReLU_fwd_mean_var(this->slope,
                                  input_states.mu_a, input_states.var_a, start_chunk,
                                  end_chunk, this->input_size,
                                  batch_size, output_states.mu_a,
                                  output_states.var_a);
    }
    // Update number of actual states.
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;
    temp_states.tmp_1 = input_states.mu_a;
    temp_states.tmp_2 = input_states.var_a;
    temp_states.tmp_3 = output_states.mu_a;
    temp_states.tmp_4 = output_states.var_a;

    if (this->training)
    {
        this->storing_states_for_training(input_states, output_states);
    }
}

void MixtureLReLU::backward(BaseDeltaStates &input_delta_states,
                            BaseDeltaStates &output_delta_states,
                            BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    // Initialization
    int batch_size = input_delta_states.block_size;

    // Compute inovation vector
    if (this->num_threads > 1)
    {
        MixtureLReLU_bwd_delta_z_mp(this->slope,
            temp_states.tmp_3, temp_states.tmp_4,//this->bwd_states->mu_a, this->bwd_states->var_a,
                                    temp_states.tmp_1, temp_states.tmp_2,
                                    input_delta_states.delta_mu, input_delta_states.delta_var,
                                    this->input_size, batch_size, this->num_threads,
                                    output_delta_states.delta_mu, output_delta_states.delta_var);
    }
    else
    {
        int start_chunk = 0;
        int end_chunk = batch_size * this->input_size;
        MixtureLReLU_bwd_delta_z(this->slope,
            temp_states.tmp_3, temp_states.tmp_4,//this->bwd_states->mu_a, this->bwd_states->var_a,
                                 temp_states.tmp_1, temp_states.tmp_2,
                                 input_delta_states.delta_mu, input_delta_states.delta_var,
                                 this->input_size, batch_size, start_chunk, end_chunk,
                                 output_delta_states.delta_mu, output_delta_states.delta_var);
    }
}

// #ifdef USE_CUDA
// std::unique_ptr<BaseLayer> Linear::to_cuda()
// {
//     this->device = "cuda";
//     auto cuda_layer = std::make_unique<LinearCuda>(
//         this->input_size, this->output_size, this->bias, this->gain_w,
//         this->gain_b, this->init_method);

//     // Move params from this->layer to cuda_layer
//     auto base_cuda = dynamic_cast<BaseLayerCuda *>(cuda_layer.get());
//     base_cuda->copy_params_from(*this);

//     return cuda_layer;
// }
// #endif
// #ifdef USE_CUDA
// std::unique_ptr<BaseLayer> MixtureLReLU::to_cuda() {
//     this->device = "cuda";
//     return std::make_unique<MixtureLReLUCuda>();
// }
// #endif
