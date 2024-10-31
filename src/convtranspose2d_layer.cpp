///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      August 06, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/convtranspose2d_layer.h"
#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "../include/base_layer.h"
#include "../include/common.h"
#include "../include/indices.h"
#include "../include/param_init.h"

#ifdef USE_CUDA
#include "../include/convtranspose2d_layer_cuda.cuh"
#endif
#include <thread>

void convtranspose2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &widx, const std::vector<int> &aidx, int woho,
    int fo, int wihi, int fi, int ki, int rf, int start_chunk, int end_chunk,
    bool bias, std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    int ki2 = ki * ki;
    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < woho * fo; col++) {
            int div_idx = col / woho;
            int mod_idx = col % woho;
            float sum_mu = 0;
            float sum_var = 0;
            int aidx_tmp = 0;
            int widx_tmp = 0;

            for (int i = 0; i < rf * fi; i++) {
                int i_div_rf = i / rf;

                // minus 1 due to the index starting at 1
                widx_tmp = widx[mod_idx * rf + i % rf];
                aidx_tmp = aidx[mod_idx * rf + i % rf];

                if (aidx_tmp > -1 && widx_tmp > -1) {
                    widx_tmp += div_idx * ki2 + i_div_rf * ki2 * fo - 1;
                    aidx_tmp += row * wihi * fi + i_div_rf * wihi - 1;

                    sum_mu += mu_w[widx_tmp] * mu_a[aidx_tmp];

                    sum_var +=
                        (mu_w[widx_tmp] * mu_w[widx_tmp] + var_w[widx_tmp]) *
                            var_a[aidx_tmp] +
                        var_w[widx_tmp] * mu_a[aidx_tmp] * mu_a[aidx_tmp];
                }
            }

            mu_z[col + row * woho * fo] = sum_mu;
            var_z[col + row * woho * fo] = sum_var;
            if (bias) {
                mu_z[col + row * woho * fo] += mu_b[div_idx];
                var_z[col + row * woho * fo] += var_b[div_idx];
            }
        }
    }
}

void convtranspose2d_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &widx, const std::vector<int> &aidx, int woho,
    int fo, int wihi, int fi, int ki, int rf, int batch_size, bool bias,
    unsigned int num_threads, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int batch_per_thread = batch_size / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk = i * batch_per_thread;
        int end_chunk = (i + 1) * batch_per_thread;

        if (i == num_threads - 1) {
            end_chunk = batch_size;
        }

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &widx, &aidx, &mu_z, &var_z]() {
            convtranspose2d_fwd_mean_var(
                mu_w, var_w, mu_b, var_b, mu_a, var_a, widx, aidx, woho, fo,
                wihi, fi, ki, rf, start_chunk, end_chunk, bias, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        };
    }
}

void convtranspose2d_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &widx,
    const std::vector<int> &zidx, int woho, int fo, int wihi, int fi, int ki,
    int rf, int start_chunk, int end_chunk, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/*
 */
{
    int input_size = wihi * fi;

    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < input_size; col++) {
            float sum_mu = 0.0f;
            float sum_var = 0.0f;
            int widx_tmp;
            int zidx_tmp;  // updated index (idxSzzUd)
            for (int i = 0; i < rf * fo; i++) {
                // minus 1 due to the index starting at 1
                zidx_tmp = zidx[(col % wihi) * ki * ki + i % rf];
                widx_tmp = widx[(col % wihi) * ki * ki + i % rf];

                if (zidx_tmp > -1 && widx_tmp > -1) {
                    widx_tmp +=
                        (i / rf) * ki * ki + (col / wihi) * ki * ki * fo - 1;
                    zidx_tmp += (i / rf) * woho + row * woho * fo - 1;

                    sum_mu += delta_mu_out[zidx_tmp] * mu_w[widx_tmp];
                    sum_var += mu_w[widx_tmp] * delta_var_out[zidx_tmp] *
                               mu_w[widx_tmp];
                }
            }

            delta_mu[col + row * input_size] =
                sum_mu * jcb[col + row * input_size];
            delta_var[col + row * input_size] = sum_var *
                                                jcb[col + row * input_size] *
                                                jcb[col + row * input_size];
        }
    }
}

void convtranspose2d_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &widx,
    const std::vector<int> &zidx, int woho, int fo, int wihi, int fi, int ki,
    int rf, int batch_size, unsigned int num_threads,
    std::vector<float> &delta_mu, std::vector<float> &delta_var)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int batch_per_thread = batch_size / num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk = i * batch_per_thread;
        int end_chunk = (i + 1) * batch_per_thread;

        // Ensure the last thread covers any remainder
        if (i == num_threads - 1) {
            end_chunk = batch_size;
        }

        threads.emplace_back([=, &mu_w, &jcb, &delta_mu_out, &delta_var_out,
                              &widx, &zidx, &delta_mu, &delta_var]() {
            convtranspose2d_bwd_delta_z(
                mu_w, jcb, delta_mu_out, delta_var_out, widx, zidx, woho, fo,
                wihi, fi, ki, rf, start_chunk, end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void convtranspose2d_bwd_delta_w(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &aidx,
    const std::vector<int> &zidx, int woho, int fo, int wihi, int fi, int ki,
    int batch_size, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/*
 */
{
    int num_params = ki * ki * fo;
    int ki2 = ki * ki;
    for (int col = start_chunk; col < end_chunk; col++) {
        for (int row = 0; row < fi; row++) {
            float sum_mu = 0.0f;
            float sum_var = 0.0f;
            int zidx_tmp;  // updated index (idxSzzUd)
            int aidx_tmp;
            int col_div_ki2 = col / ki2;
            int col_mod_ki2 = col % ki2;
            for (int i = 0; i < wihi * batch_size; i++)  // n = wihi * B
            {
                int i_div_wihi = i / wihi;
                int i_mod_wihi = i % wihi;

                // minus 1 due to the index starting at 1
                aidx_tmp = aidx[col_mod_ki2 * wihi + i_mod_wihi];

                if (aidx_tmp > -1) {
                    // minus 1 due to the index starting at 1
                    zidx_tmp = zidx[col_mod_ki2 * wihi + i_mod_wihi] +
                               col_div_ki2 * woho + i_div_wihi * woho * fo - 1;
                    aidx_tmp += +row * wihi + i_div_wihi * wihi * fi - 1;

                    sum_mu += mu_a[aidx_tmp] * delta_mu_out[zidx_tmp];
                    sum_var += mu_a[aidx_tmp] * mu_a[aidx_tmp] *
                               delta_var_out[zidx_tmp];
                }
            }

            delta_mu_w[col + row * num_params] =
                sum_mu * var_w[col + row * num_params];
            delta_var_w[col + row * num_params] =
                sum_var * var_w[col + row * num_params] *
                var_w[col + row * num_params];
        }
    }
}

void convtranspose2d_bwd_delta_w_mp(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &aidx,
    const std::vector<int> &zidx, int woho, int fo, int wihi, int fi, int ki,
    int batch_size, unsigned int num_threads, std::vector<float> &delta_mu_w,
    std::vector<float> &delta_var_w)
/*
 */
{
    int num_params = ki * ki * fo;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int params_per_thread = num_params / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_chunk = t * params_per_thread;
        int end_chunk = (t + 1) * params_per_thread;

        // Adjust for the last thread to cover any remainder
        if (t == num_threads - 1) {
            end_chunk = num_params;
        }

        threads.emplace_back([=, &var_w, &mu_a, &delta_mu_out, &delta_var_out,
                              &aidx, &zidx, &delta_mu_w, &delta_var_w]() {
            convtranspose2d_bwd_delta_w(var_w, mu_a, delta_mu_out,
                                        delta_var_out, aidx, zidx, woho, fo,
                                        wihi, fi, ki, batch_size, start_chunk,
                                        end_chunk, delta_mu_w, delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void convtranspose2d_bwd_delta_b(const std::vector<float> &var_b,
                                 const std::vector<float> &delta_mu_out,
                                 const std::vector<float> &delta_var_out,
                                 int woho, int fo, int batch_size,
                                 int start_chunk, int end_chunk,
                                 std::vector<float> &delta_mu_b,
                                 std::vector<float> &delta_var_b)
/*
 */
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < woho * batch_size; i++)  // n = woho * B
        {
            int idx = col * woho + (i % woho) + (i / woho) * woho * fo;

            sum_mu += delta_mu_out[idx];
            sum_var += delta_var_out[idx];
        }

        delta_mu_b[col] = sum_mu * var_b[col];
        delta_var_b[col] = var_b[col] * sum_var * var_b[col];
    }
}

void convtranspose2d_bwd_delta_b_mp(const std::vector<float> &var_b,
                                    const std::vector<float> &delta_mu_out,
                                    const std::vector<float> &delta_var_out,
                                    int woho, int fo, int batch_size,
                                    unsigned int num_threads,
                                    std::vector<float> &delta_mu_b,
                                    std::vector<float> &delta_var_b)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int fo_per_thread = fo / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        int start_chunk = t * fo_per_thread;
        int end_chunk = (t + 1) * fo_per_thread;

        // Adjust for the last thread to cover any remainder
        if (t == num_threads - 1) {
            end_chunk = fo;
        }

        threads.emplace_back([=, &var_b, &delta_mu_out, &delta_var_out,
                              &delta_mu_b, &delta_var_b]() {
            convtranspose2d_bwd_delta_b(var_b, delta_mu_out, delta_var_out,
                                        woho, fo, batch_size, start_chunk,
                                        end_chunk, delta_mu_b, delta_var_b);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// CONV. TRANSPOSE 2D
////////////////////////////////////////////////////////////////////////////////

ConvTranspose2dIndex get_tconv_idx(int pad_idx_in, int pad_idx_out,
                                   int param_pad_idx, Conv2dIndex &convIndex)
/**/
{
    // Initialize pointers
    std::vector<int> FCwz_2_idx, Swz_ud_idx;
    std::vector<int> FCzwa_1_idx, Szz_ud_idx;
    std::vector<int> FCzwa_1_idx_t, FCzwa_2_idx_t, Szz_ud_idx_t, tmp_1, tmp_2,
        tmp_3, tmp_4;
    int pad = 1;

    // Transpose convolutional index matrix
    FCzwa_1_idx_t =
        transpose_matrix(convIndex.FCzwa_1_idx, convIndex.w, convIndex.h);
    FCzwa_2_idx_t =
        transpose_matrix(convIndex.FCzwa_2_idx, convIndex.w, convIndex.h);
    Szz_ud_idx_t =
        transpose_matrix(convIndex.Szz_ud_idx, convIndex.w, convIndex.h);

    ///////////////////////////////////////////
    /* Indices for FCwz 2 and Swz ud */
    // Get unique indices and its frequency of the receptive field
    auto FCwz_1 = get_ref_idx(convIndex.FCzwa_1_idx, pad, param_pad_idx);

    // Get indices for FCwz_2
    tmp_1 = reorganize_idx_from_ref(Szz_ud_idx_t, pad, FCwz_1.pad_pos,
                                    pad_idx_out, FCwz_1.ref, FCwz_1.base_idx,
                                    FCwz_1.w, FCwz_1.h);
    FCwz_2_idx = transpose_matrix(tmp_1, FCwz_1.w, FCwz_1.h);

    // Get indices for Swz ud
    tmp_2 = reorganize_idx_from_ref(FCzwa_2_idx_t, pad, FCwz_1.pad_pos,
                                    pad_idx_in, FCwz_1.ref, FCwz_1.base_idx,
                                    FCwz_1.w, FCwz_1.h);

    Swz_ud_idx = transpose_matrix(tmp_2, FCwz_1.w, FCwz_1.h);

    //////////////////////////////////////////
    /* Indices for FCzz 2 and Szz ud */
    // Get unique indices and its frequency of the receptive field
    auto Szz_ud = get_ref_idx(Szz_ud_idx_t, pad, pad_idx_out);

    // Get indices for FCwz_2
    tmp_3 = reorganize_idx_from_ref(convIndex.FCzwa_1_idx, pad, Szz_ud.pad_pos,
                                    param_pad_idx, Szz_ud.ref, Szz_ud.base_idx,
                                    Szz_ud.w, Szz_ud.h);

    FCzwa_1_idx = transpose_matrix(tmp_3, Szz_ud.w, Szz_ud.h);

    // Get indices for Szz ud
    tmp_4 = reorganize_idx_from_ref(FCzwa_2_idx_t, pad, Szz_ud.pad_pos,
                                    pad_idx_in, Szz_ud.ref, Szz_ud.base_idx,
                                    Szz_ud.w, Szz_ud.h);

    Szz_ud_idx = transpose_matrix(tmp_4, Szz_ud.w, Szz_ud.h);

    return {FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx,
            FCwz_1.w,   FCwz_1.h,   Szz_ud.w,    Szz_ud.h};
}

std::tuple<int, int> compute_upsample_img_size_v2(int kernel, int stride,
                                                  int wi, int hi, int pad,
                                                  int pad_type)
/* Compute the size of upsampling images i.e. increase of image size.
 *
 * Args:
 *    Kernel: size of the receptive field
 *    stride: Stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    pad: Number of paddings
 *    pad_type: Padding type
 *
 * Returns:
 *    wo: Width of the output image
 *    ho: Height of the output image
 *    */
{
    int wo, ho, nom_w, nom_h;
    // Compute nominator of tconv. formulation given a padding type
    if (pad_type == 1) {
        wo = stride * (wi - 1) + kernel - 2 * pad;
        ho = stride * (hi - 1) + kernel - 2 * pad;
        nom_w = wo - kernel + 2 * pad;
        nom_h = ho - kernel + 2 * pad;
    }

    // Check validity of the conv. hyper-parameters such as wi, hi, kernel,
    // stride
    else if (pad_type == 2) {
        wo = stride * (wi - 1) + kernel - pad;
        ho = stride * (hi - 1) + kernel - pad;
        nom_w = wo - kernel + pad;
        nom_h = ho - kernel + pad;
    }

    if (nom_w % stride != 0 || nom_h % stride != 0) {
        throw std::invalid_argument(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Invalid hyperparameters for ConvTranspose2d layer");
    }

    return {wo, ho};
}

ConvTranspose2d::ConvTranspose2d(size_t in_channels, size_t out_channels,
                                 size_t kernel_size, bool bias, int stride,
                                 int padding, int padding_type, size_t in_width,
                                 size_t in_height, float gain_w, float gain_b,
                                 std::string init_method)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/**/
{
    this->in_width = in_width;
    this->in_height = in_height;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->bias = bias;

    if (this->training) {
        this->bwd_states = std::make_unique<BaseBackwardStates>();
    }
}

ConvTranspose2d::~ConvTranspose2d() {}

std::string ConvTranspose2d::get_layer_name() const {
    return "ConvTranspose2d";
}

std::string ConvTranspose2d::get_layer_info() const {
    return "ConvTranspose2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

LayerType ConvTranspose2d::get_layer_type() const {
    return LayerType::ConvTranspose2d;
};

void ConvTranspose2d::compute_input_output_size(const InitArgs &args)
/*
 */
{
    if (this->in_height == 0 || this->in_height == 0) {
        this->in_width = args.width;
        this->in_height = args.height;
    }
    std::tie(this->out_width, this->out_height) = compute_upsample_img_size_v2(
        this->kernel_size, this->stride, this->in_width, this->in_height,
        this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void ConvTranspose2d::get_number_param()
/*
 */
{
    this->num_weights = this->kernel_size * this->kernel_size *
                        this->in_channels * this->out_channels;
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = this->out_channels;
    }
}

void ConvTranspose2d::init_weight_bias()
/**/
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
}

void ConvTranspose2d::lazy_index_init()
/*
 */
{
    // int ki2 = this->kernel_size * this->kernel_size;
    // int param_pad_idx = ki2 * this->in_channels * this->out_channels + 1;

    // TODO: remove all -1 when unittest is done
    auto conv_idx =
        get_conv2d_idx(this->kernel_size, this->stride, this->out_width,
                       this->out_height, this->in_width, this->in_height,
                       this->padding, this->padding_type, -1, -1, -1);

    auto conv_transpose_idx = get_tconv_idx(-1, -1, -1, conv_idx);

    this->idx_mwa_1 = conv_idx.FCzwa_1_idx;
    this->idx_mwa_2 =
        transpose_matrix(conv_idx.Szz_ud_idx, conv_idx.w, conv_idx.h);
    this->idx_cov_wz_2 = conv_transpose_idx.FCwz_2_idx;
    this->idx_var_wz_ud = conv_transpose_idx.Swz_ud_idx;
    this->idx_cov_z_wa_1 = conv_transpose_idx.FCzwa_1_idx;
    this->idx_var_z_ud = conv_transpose_idx.Szz_ud_idx;

    // Dimension
    this->row_zw = conv_transpose_idx.w_wz;
    this->col_z_ud = conv_transpose_idx.w_zz;
    this->col_cov_mwa_1 = conv_idx.h;
}

void ConvTranspose2d::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_1.size() == 0) {
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

    if (this->num_threads > 1) {
        convtranspose2d_fwd_mean_var_mp(
            this->mu_w, this->var_w, this->mu_b, this->var_b, input_states.mu_a,
            input_states.var_a, this->idx_mwa_1, this->idx_mwa_2, woho,
            this->out_channels, wihi, this->in_channels, this->kernel_size,
            this->col_cov_mwa_1, batch_size, this->bias, this->num_threads,
            output_states.mu_a, output_states.var_a);
    } else {
        convtranspose2d_fwd_mean_var(
            this->mu_w, this->var_w, this->mu_b, this->var_b, input_states.mu_a,
            input_states.var_a, this->idx_mwa_1, this->idx_mwa_2, woho,
            this->out_channels, wihi, this->in_channels, this->kernel_size,
            this->col_cov_mwa_1, 0, batch_size, this->bias, output_states.mu_a,
            output_states.var_a);
    }
    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void ConvTranspose2d::backward(BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_delta_states,
                               BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    // Initialization
    int batch_size = input_delta_states.block_size;

    int wihi = this->in_height * this->in_width;
    int woho = this->out_width * this->out_height;

    if (this->num_threads > 1) {
        convtranspose2d_bwd_delta_z_mp(
            this->mu_w, this->bwd_states->jcb, input_delta_states.delta_mu,
            input_delta_states.delta_var, this->idx_cov_z_wa_1,
            this->idx_var_z_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, this->row_zw, batch_size,
            this->num_threads, output_delta_states.delta_mu,
            output_delta_states.delta_var);
    } else {
        convtranspose2d_bwd_delta_z(
            this->mu_w, this->bwd_states->jcb, input_delta_states.delta_mu,
            input_delta_states.delta_var, this->idx_cov_z_wa_1,
            this->idx_var_z_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, this->row_zw, 0, batch_size,
            output_delta_states.delta_mu, output_delta_states.delta_var);
    }

    // Parameters

    int ki2 = this->kernel_size * this->kernel_size;

    if (this->num_threads > 1) {
        convtranspose2d_bwd_delta_w_mp(
            this->var_w, this->bwd_states->mu_a, input_delta_states.delta_mu,
            input_delta_states.delta_var, this->idx_cov_wz_2,
            this->idx_var_wz_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, batch_size, this->num_threads,
            this->delta_mu_w, this->delta_var_w);
        if (this->bias) {
            convtranspose2d_bwd_delta_b_mp(
                this->var_b, input_delta_states.delta_mu,
                input_delta_states.delta_var, woho, this->out_channels,
                batch_size, this->num_threads, this->delta_mu_b,
                this->delta_var_b);
        }
    } else {
        convtranspose2d_bwd_delta_w(
            this->var_w, this->bwd_states->mu_a, input_delta_states.delta_mu,
            input_delta_states.delta_var, this->idx_cov_wz_2,
            this->idx_var_wz_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, batch_size, 0,
            ki2 * this->out_channels, this->delta_mu_w, this->delta_var_w);
        if (this->bias) {
            convtranspose2d_bwd_delta_b(
                this->var_b, input_delta_states.delta_mu,
                input_delta_states.delta_var, woho, this->out_channels,
                batch_size, 0, this->out_channels, this->delta_mu_b,
                this->delta_var_b);
        }
    }
}

// void ConvTranspose2d::state_backward(BaseBackwardStates &next_bwd_states,
//                                      BaseDeltaStates &input_delta_states,
//                                      BaseDeltaStates &output_delta_states,
//                                      BaseTempStates &temp_states)
// /*
//  */
// {
//     // Initialization
//     int batch_size = input_delta_states.block_size;

//     int wihi = this->in_height * this->in_width;
//     int woho = this->out_width * this->out_height;

//     if (this->num_threads > 1) {
//         convtranspose2d_bwd_delta_z_mp(
//             this->mu_w, next_bwd_states.jcb, input_delta_states.delta_mu,
//             input_delta_states.delta_var, this->idx_cov_z_wa_1,
//             this->idx_var_z_ud, woho, this->out_channels, wihi,
//             this->in_channels, this->kernel_size, this->row_zw, batch_size,
//             this->num_threads, output_delta_states.delta_mu,
//             output_delta_states.delta_var);
//     } else {
//         convtranspose2d_bwd_delta_z(
//             this->mu_w, next_bwd_states.jcb, input_delta_states.delta_mu,
//             input_delta_states.delta_var, this->idx_cov_z_wa_1,
//             this->idx_var_z_ud, woho, this->out_channels, wihi,
//             this->in_channels, this->kernel_size, this->row_zw, 0,
//             batch_size, output_delta_states.delta_mu,
//             output_delta_states.delta_var);
//     }
// }

// void ConvTranspose2d::param_backward(BaseBackwardStates &next_bwd_states,
//                                      BaseDeltaStates &delta_states,
//                                      BaseTempStates &temp_states)
// /*
//  */
// {
//     int batch_size = delta_states.block_size;

//     int ki2 = this->kernel_size * this->kernel_size;
//     int wihi = this->in_height * this->in_width;
//     int woho = this->out_width * this->out_height;

//     if (this->num_threads > 1) {
//         convtranspose2d_bwd_delta_w_mp(
//             this->var_w, next_bwd_states.mu_a, delta_states.delta_mu,
//             delta_states.delta_var, this->idx_cov_wz_2, this->idx_var_wz_ud,
//             woho, this->out_channels, wihi, this->in_channels,
//             this->kernel_size, batch_size, this->num_threads,
//             this->delta_mu_w, this->delta_var_w);
//         if (this->bias) {
//             convtranspose2d_bwd_delta_b_mp(
//                 this->var_b, delta_states.delta_mu, delta_states.delta_var,
//                 woho, this->out_channels, batch_size, this->num_threads,
//                 this->delta_mu_b, this->delta_var_b);
//         }
//     } else {
//         convtranspose2d_bwd_delta_w(
//             this->var_w, next_bwd_states.mu_a, delta_states.delta_mu,
//             delta_states.delta_var, this->idx_cov_wz_2, this->idx_var_wz_ud,
//             woho, this->out_channels, wihi, this->in_channels,
//             this->kernel_size, batch_size, 0, ki2 * this->out_channels,
//             this->delta_mu_w, this->delta_var_w);
//         if (this->bias) {
//             convtranspose2d_bwd_delta_b(
//                 this->var_b, delta_states.delta_mu, delta_states.delta_var,
//                 woho, this->out_channels, batch_size, 0, this->out_channels,
//                 this->delta_mu_b, this->delta_var_b);
//         }
//     }
// }

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> ConvTranspose2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<ConvTranspose2dCuda>(
        this->in_channels, this->out_channels, this->kernel_size, this->bias,
        this->stride, this->padding, this->padding_type, this->in_width,
        this->in_height, this->gain_w, this->gain_b, this->init_method);
}
#endif

void ConvTranspose2d::preinit_layer() {
    this->get_number_param();
    this->init_weight_bias();
    this->lazy_index_init();
    if (this->training) {
        this->allocate_param_delta();
    }
}
