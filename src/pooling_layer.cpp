///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 08, 2024
// Updated:      January 23, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/pooling_layer.h"

#include "../include/conv2d_layer.h"
#include "../include/indices.h"
#ifdef USE_CUDA
#include "../include/pooling_layer_cuda.cuh"
#endif
#include <thread>

////////////////////////////////////////////////////////////////////////////////
/// AvgPool2d
////////////////////////////////////////////////////////////////////////////////
AvgPool2d::AvgPool2d(size_t kernel_size, int stride, int padding,
                     int padding_type)
    : kernel_size(kernel_size),
      stride(stride),
      padding_type(padding_type),
      padding(padding)
/**/
{
    if (this->training) {
        this->bwd_states = std::make_unique<BaseBackwardStates>();
    }
}

AvgPool2d::~AvgPool2d() {}

std::string AvgPool2d::get_layer_info() const {
    return "AvgPool2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

std::string AvgPool2d::get_layer_name() const { return "AvgPool2d"; }

LayerType AvgPool2d::get_layer_type() const { return LayerType::Pool2d; }

void AvgPool2d::compute_input_output_size(const InitArgs &args)
/*
 */
{
    this->in_width = args.width;
    this->in_height = args.height;
    this->in_channels = args.depth;
    this->out_channels = args.depth;

    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride,
                                       this->in_width, this->in_height,
                                       this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void AvgPool2d::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_states.block_size;

    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
    }

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int num_states = woho * this->out_channels * batch_size;
    int pad_idx_in = wihi * this->in_channels * batch_size + 1;

    if (this->num_threads > 1) {
        if (this->overlap) {
            avgpool2d_fwd_overlapped_mean_var_mp(
                input_states.mu_a, input_states.var_a, this->pool_idx, woho,
                wihi, this->kernel_size, num_states, pad_idx_in,
                this->num_threads, output_states.mu_a, output_states.var_a);
        } else {
            avgpool2d_fwd_mean_var_mp(
                input_states.mu_a, input_states.var_a, this->pool_idx, woho,
                wihi, this->kernel_size, num_states, this->num_threads,
                output_states.mu_a, output_states.var_a);
        }
    } else {
        if (this->overlap) {
            avgpool2d_fwd_overlapped_mean_var(
                input_states.mu_a, input_states.var_a, this->pool_idx, woho,
                wihi, this->kernel_size, num_states, pad_idx_in, 0, num_states,
                output_states.mu_a, output_states.var_a);
        } else {
            avgpool2d_fwd_mean_var(input_states.mu_a, input_states.var_a,
                                   this->pool_idx, woho, wihi,
                                   this->kernel_size, num_states, 0, num_states,
                                   output_states.mu_a, output_states.var_a);
        }
    }

    if (training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void AvgPool2d::backward(BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         BaseTempStates &temp_states, bool state_udapte)
/**/
{
    // Initialization
    int batch_size = input_delta_states.block_size;

    // Launch kernel
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int pad_out_idx = woho * this->out_channels * batch_size + 1;

    if (state_udapte) {
        if (this->num_threads > 1) {
            if (this->overlap) {
                int num_in_states = this->in_width * this->in_height *
                                    this->in_channels * batch_size;

                avgpool2d_bwd_overlapped_delta_z_mp(
                    this->bwd_states->jcb, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->z_ud_idx, woho, wihi,
                    this->kernel_size, this->col_z_ud, num_in_states,
                    pad_out_idx, this->num_threads,
                    output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            } else {
                int kiwo = this->kernel_size * this->out_width;
                int nums = wihi * this->in_channels * batch_size / kiwo;

                avgpool2d_bwd_delta_z_mp(
                    this->bwd_states->jcb, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->out_width,
                    this->kernel_size, nums, this->num_threads,
                    output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            }
        } else {
            if (this->overlap) {
                int num_in_states = this->in_width * this->in_height *
                                    this->in_channels * batch_size;

                avgpool2d_bwd_overlapped_delta_z(
                    this->bwd_states->jcb, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->z_ud_idx, woho, wihi,
                    this->kernel_size, this->col_z_ud, num_in_states,
                    pad_out_idx, 0, num_in_states, output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            } else {
                int kiwo = this->kernel_size * this->out_width;
                int nums = wihi * this->in_channels * batch_size / kiwo;
                int end_chunk = this->kernel_size * this->out_width * nums;

                avgpool2d_bwd_delta_z(
                    this->bwd_states->jcb, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->out_width,
                    this->kernel_size, nums, 0, end_chunk,
                    output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            }
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> AvgPool2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<AvgPool2dCuda>(this->kernel_size, this->stride,
                                           this->padding, this->padding_type);
}
#endif

void AvgPool2d::lazy_index_init()
/*
 */
{
    if (this->kernel_size == this->stride ||
        this->kernel_size == this->in_width) {
        this->overlap = false;
    }

    int pad_idx_in = -1;
    int pad_idx_out = -1;

    auto idx = get_pool_index(this->kernel_size, this->stride, this->in_width,
                              this->in_height, this->out_width,
                              this->out_height, this->padding,
                              this->padding_type, pad_idx_in, pad_idx_out);

    this->pool_idx = idx.pool_idx;
    this->z_ud_idx = idx.z_ud_idx;
    this->row_zw = idx.w;
    this->col_z_ud = idx.h;
}

void AvgPool2d::preinit_layer() {
    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Pool2d Backward and Forward
////////////////////////////////////////////////////////////////////////////////
void avgpool2d_fwd_overlapped_mean_var(const std::vector<float> &mu_a,
                                       const std::vector<float> &var_a,
                                       const std::vector<int> &a_idx, int woho,
                                       int wihi, int ki, int k, int pad_idx,
                                       int start_chunk, int end_chunk,
                                       std::vector<float> &mu_z,
                                       std::vector<float> &var_z)
/**/
{
    int ki2 = ki * ki;
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu_z = 0;
        float sum_var_z = 0;
        for (int i = 0; i < ki2; i++) {
            int a_idx_tmp = a_idx[col % woho + woho * i];
            if (a_idx_tmp > -1) {
                a_idx_tmp += (col / woho) * wihi;
                // index in a_idx starts at 1
                sum_mu_z += mu_a[a_idx_tmp - 1];
                sum_var_z += var_a[a_idx_tmp - 1];
            }
        }
        mu_z[col] = sum_mu_z / ki2;
        var_z[col] = sum_var_z / (ki2 * ki2);
    }
}

void avgpool2d_fwd_overlapped_mean_var_mp(const std::vector<float> &mu_a,
                                          const std::vector<float> &var_a,
                                          const std::vector<int> &a_idx,
                                          int woho, int wihi, int ki, int k,
                                          int pad_idx, unsigned int num_threads,
                                          std::vector<float> &mu_z,
                                          std::vector<float> &var_z)
/**/
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = k / num_threads;
    int extra = k % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &var_a, &a_idx, &mu_z, &var_z] {
            avgpool2d_fwd_overlapped_mean_var(mu_a, var_a, a_idx, woho, wihi,
                                              ki, k, pad_idx, start_chunk,
                                              end_chunk, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void avgpool2d_fwd_mean_var(const std::vector<float> &mu_a,
                            const std::vector<float> &var_a,
                            const std::vector<int> a_idx, int woho, int wihi,
                            int ki, int k, int start_chunk, int end_chunk,
                            std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    int ki2 = ki * ki;
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu_z = 0;
        float sum_var_z = 0;
        for (int i = 0; i < ki2; i++) {
            // index in a_idx starts at 1
            int a_idx_tmp =
                a_idx[col % woho + woho * i] + (col / woho) * wihi - 1;
            sum_mu_z += mu_a[a_idx_tmp];
            sum_var_z += var_a[a_idx_tmp];
        }
        mu_z[col] = sum_mu_z / ki2;
        var_z[col] = sum_var_z / (ki2 * ki2);
    }
}

void avgpool2d_fwd_mean_var_mp(const std::vector<float> &mu_a,
                               const std::vector<float> &var_a,
                               const std::vector<int> a_idx, int woho, int wihi,
                               int ki, int k, unsigned int num_threads,
                               std::vector<float> &mu_z,
                               std::vector<float> &var_z)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = k / num_threads;
    int extra = k % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &var_a, &a_idx, &mu_z, &var_z] {
            avgpool2d_fwd_mean_var(mu_a, var_a, a_idx, woho, wihi, ki, k,
                                   start_chunk, end_chunk, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void avgpool2d_bwd_overlapped_delta_z(
    const std::vector<float> &jcb, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &z_ud_idx,
    int woho, int wihi, int ki, int n, int k, int pad_idx, int start_chunk,
    int end_chunk, std::vector<float> &delta_mu, std::vector<float> &delta_var)
/**/
{
    int ki2 = ki * ki;

    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_delta_mu = 0;
        float sum_delta_var = 0;
        for (int i = 0; i < n; i++) {
            int z_idx_tmp = z_ud_idx[col % wihi + wihi * i];
            if (z_idx_tmp > -1) {
                z_idx_tmp += (col / wihi) * woho;
                sum_delta_mu += delta_mu_out[z_idx_tmp - 1];
                sum_delta_var += delta_var_out[z_idx_tmp - 1];
            }
        }
        delta_mu[col] = sum_delta_mu * jcb[col] / ki2;
        delta_var[col] = sum_delta_var * jcb[col] * jcb[col] / (ki2 * ki2);
    }
}

void avgpool2d_bwd_overlapped_delta_z_mp(
    const std::vector<float> &jcb, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &z_ud_idx,
    int woho, int wihi, int ki, int n, int k, int pad_idx,
    unsigned int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/**/
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = k / num_threads;
    int extra = k % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &jcb, &delta_mu_out, &delta_var_out, &z_ud_idx,
                              &delta_mu, &delta_var] {
            avgpool2d_bwd_overlapped_delta_z(
                jcb, delta_mu_out, delta_var_out, z_ud_idx, woho, wihi, ki, n,
                k, pad_idx, start_chunk, end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void avgpool2d_bwd_delta_z(const std::vector<float> &jcb,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out, int wo,
                           int ki, int k, int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var)
/*
 */
{
    int ki2 = ki * ki;
    int m = ki * wo;
    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / k;
        int col = j % k;

        delta_mu[row + col * m] =
            delta_mu_out[row / ki + (col / ki) * wo] * jcb[row + col * m] / ki2;

        delta_var[row + col * m] = delta_var_out[row / ki + (col / ki) * wo] *
                                   jcb[row + col * m] * jcb[row + col * m] /
                                   (ki2 * ki2);
    }
}

void avgpool2d_bwd_delta_z_mp(const std::vector<float> &jcb,
                              const std::vector<float> &delta_mu_out,
                              const std::vector<float> &delta_var_out, int wo,
                              int ki, int k, unsigned int num_threads,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var)
/*
 */
{
    int m = ki * wo;
    const int tot_ops = k * m;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &jcb, &delta_mu_out, &delta_var_out, &delta_mu,
                              &delta_var] {
            avgpool2d_bwd_delta_z(jcb, delta_mu_out, delta_var_out, wo, ki, k,
                                  start_chunk, end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

Pool2dIndex get_pool_index(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out) {
    // Initialize pointers
    std::vector<int> raw_img, img, padded_img, Fmwa_2_idx, tmp;
    std::vector<int> Szz_ud_idx;
    RefIndexOut Fmwa_2;
    int w_p, h_p;

    // Generate image indices
    std::tie(raw_img, img, padded_img, w_p, h_p) =
        image_construction(wi, hi, pad, pad_idx_in, pad_type);

    // Get indices for receptive field
    tmp =
        get_receptive_field(img, padded_img, kernel, stride, wo, ho, w_p, h_p);
    if (!(kernel == stride || (kernel == wi && stride == 1))) {
        // Get unique indices and its frequency of the receptive field
        Fmwa_2 = get_ref_idx(tmp, pad, pad_idx_in);

        // Get indices for Szz ud
        Szz_ud_idx =
            get_Szz_ud_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                           Fmwa_2.base_idx, pad_idx_out, Fmwa_2.w, Fmwa_2.h);
    }

    // NOTE THAT DOUBLE CHECK WHY WE NEED THE TRANSPOSE HEAR AND SIZE OF MATRIX
    Fmwa_2_idx = transpose_matrix(tmp, kernel * kernel, wo * ho);

    return {Fmwa_2_idx, Szz_ud_idx, Fmwa_2.w, Fmwa_2.h};
}