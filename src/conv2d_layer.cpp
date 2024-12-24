#include "../include/conv2d_layer.h"

#include "../include/custom_logger.h"
#include "../include/indices.h"
#include "../include/param_init.h"
#ifdef USE_CUDA
#include "../include/conv2d_layer_cuda.cuh"
#endif
#include <thread>

////////////////////////////////////////////////////////////////////////////////
/// Conv2d
////////////////////////////////////////////////////////////////////////////////
Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
               bool bias, int stride, int padding, int padding_type,
               size_t in_width, size_t in_height, float gain_w, float gain_b,
               std::string init_method)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/*
 */
{
    this->in_width = in_width;
    this->in_height = in_height;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->bias = bias;
}

Conv2d::~Conv2d() {}

std::string Conv2d::get_layer_name() const { return "Conv2d"; }

std::string Conv2d::get_layer_info() const {
    return "Conv2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

LayerType Conv2d::get_layer_type() const { return LayerType::Conv2d; };

void Conv2d::compute_input_output_size(const InitArgs &args)
/*
 */
{
    if (this->in_height == 0 || this->in_height == 0) {
        this->in_width = args.width;
        this->in_height = args.height;
    }
    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride,
                                       this->in_width, this->in_height,
                                       this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void Conv2d::get_number_param()

/* Get the number of parameters for conv. and tconv. layer.

 *    */
{
    int n_w, n_b;
    n_w = this->kernel_size * this->kernel_size * this->in_channels *
          this->out_channels;
    if (this->bias) {
        n_b = this->out_channels;
    } else {
        n_b = 0;
    }
    this->num_weights = n_w;
    this->num_biases = n_b;
}

void Conv2d::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
}

void Conv2d::lazy_index_init()
/*
 */
{
    // Get precomputed conv indices
    int param_pad_idx =
        pow(this->kernel_size, 2) * this->in_channels * this->out_channels + 1;

    auto conv_idx = get_conv2d_idx(
        this->kernel_size, this->stride, this->in_width, this->in_height,
        this->out_width, this->out_height, this->padding, this->padding_type,
        -1, -1, param_pad_idx);

    this->idx_mwa_2 = conv_idx.Fmwa_2_idx;
    this->idx_cov_zwa_1 = conv_idx.FCzwa_1_idx;
    this->idx_var_z_ud = conv_idx.Szz_ud_idx;

    this->row_zw = conv_idx.h;
    this->col_z_ud = conv_idx.h;
}

void Conv2d::forward(BaseHiddenStates &input_states,
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

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    // Only need to initalize at the first iteration
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_2.size() == 0) {
        this->lazy_index_init();
    }

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    // Launch kernel
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int pad_idx = wihi * this->in_channels * batch_size + 1;

    if (this->num_threads > 1) {
        conv2d_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               input_states.mu_a, input_states.var_a,
                               this->idx_mwa_2, woho, this->out_channels, wihi,
                               this->in_channels, this->kernel_size, batch_size,
                               pad_idx, this->bias, this->num_threads,
                               output_states.mu_a, output_states.var_a);
    } else {
        conv2d_fwd_mean_var(
            this->mu_w, this->var_w, this->mu_b, this->var_b, input_states.mu_a,
            input_states.var_a, this->idx_mwa_2, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, batch_size, pad_idx,
            this->bias, 0, woho * batch_size * this->out_channels,
            output_states.mu_a, output_states.var_a);
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void Conv2d::backward(BaseDeltaStates &input_delta_states,
                      BaseDeltaStates &output_delta_states,
                      BaseTempStates &temp_states, bool state_udapte)
/**/
{
    // Initialization
    int batch_size = input_delta_states.block_size;

    int wihi = this->in_width * this->in_height;
    int woho = this->out_width * this->out_height;
    int row_zw_fo = this->row_zw * this->out_channels;
    int pad_idx = woho * this->out_channels * batch_size + 1;

    if (state_udapte) {
        permute_jacobian(this->bwd_states->jcb, wihi, this->in_channels,
                         batch_size, temp_states.tmp_1);

        if (this->num_threads > 1) {
            conv2d_bwd_delta_z_mp(
                this->mu_w, temp_states.tmp_1, input_delta_states.delta_mu,
                input_delta_states.delta_var, this->idx_cov_zwa_1,
                this->idx_var_z_ud, woho, this->out_channels, wihi,
                this->in_channels, this->kernel_size, this->row_zw, row_zw_fo,
                batch_size, pad_idx, this->num_threads,
                output_delta_states.delta_mu, output_delta_states.delta_var);
        } else {
            conv2d_bwd_delta_z(
                this->mu_w, temp_states.tmp_1, input_delta_states.delta_mu,
                input_delta_states.delta_var, this->idx_cov_zwa_1,
                this->idx_var_z_ud, woho, this->out_channels, wihi,
                this->in_channels, this->kernel_size, this->row_zw, row_zw_fo,
                batch_size, pad_idx, 0, wihi * batch_size * this->in_channels,
                output_delta_states.delta_mu, output_delta_states.delta_var);
        }
    }

    if (this->param_update) {
        int woho_batch = woho * batch_size;
        int wohofo = woho * this->out_channels;
        int param_pad_idx = wihi * this->in_channels * batch_size + 1;

        permute_delta(input_delta_states.delta_mu, input_delta_states.delta_var,
                      woho, wohofo, batch_size, temp_states.tmp_1,
                      temp_states.tmp_2);

        if (this->num_threads > 1) {
            conv2d_bwd_delta_w_mp(
                this->var_w, this->bwd_states->mu_a, temp_states.tmp_1,
                temp_states.tmp_2, this->idx_mwa_2, batch_size,
                this->out_channels, woho, wihi, this->in_channels,
                this->kernel_size, param_pad_idx, this->num_threads,
                this->delta_mu_w, this->delta_var_w);
        } else {
            int end_chunk = this->kernel_size * this->kernel_size *
                            this->in_channels * this->out_channels;
            conv2d_bwd_delta_w(this->var_w, this->bwd_states->mu_a,
                               temp_states.tmp_1, temp_states.tmp_2,
                               this->idx_mwa_2, batch_size, this->out_channels,
                               woho, wihi, this->in_channels, this->kernel_size,
                               param_pad_idx, 0, end_chunk, this->delta_mu_w,
                               this->delta_var_w);
        }
        if (this->bias) {
            conv2d_bwd_delta_b(
                this->var_b, temp_states.tmp_1, temp_states.tmp_2, woho_batch,
                this->out_channels, this->delta_mu_b, this->delta_var_b);
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Conv2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<Conv2dCuda>(
        this->in_channels, this->out_channels, this->kernel_size, this->bias,
        this->stride, this->padding, this->padding_type, this->in_width,
        this->in_height, this->gain_w, this->gain_b, this->init_method);
}
#endif

void Conv2d::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
    }

    if (this->idx_mwa_2.size() == 0) {
        this->lazy_index_init();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Conv2d Backward and Forward
////////////////////////////////////////////////////////////////////////////////
void conv2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &aidx, int woho, int fo, int wihi, int fi, int ki,
    int batch_size, int pad_idx, bool bias, int start_chunk, int end_chunk,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    int ki2 = ki * ki;
    int n = ki2 * fi;

    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / (woho * batch_size);
        int col = j % (woho * batch_size);
        float sum_mu = 0;
        float sum_var = 0;
        float mu_a_tmp = 0;
        float var_a_tmp = 0;
        float mu_w_tmp = 0;
        float var_w_tmp = 0;
        for (int i = 0; i < n; i++) {
            int aidx_tmp = aidx[(col % woho) * ki2 + i % ki2];

            if (aidx_tmp > -1) {
                // aidx's lowest value starts at 1
                aidx_tmp += (i / ki2) * wihi + (col / woho) * wihi * fi;
                mu_a_tmp = mu_a[aidx_tmp - 1];
                var_a_tmp = var_a[aidx_tmp - 1];

                mu_w_tmp = mu_w[row * n + i];
                var_w_tmp = var_w[row * n + i];

                sum_mu += mu_w_tmp * mu_a_tmp;
                sum_var += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                           var_w_tmp * mu_a_tmp * mu_a_tmp;
            }
        }

        int out_idx = woho * (col / woho) * fo + col % woho + row * woho;
        if (bias) {
            mu_z[out_idx] = sum_mu + mu_b[row];
            var_z[out_idx] = sum_var + var_b[row];
        } else {
            mu_z[out_idx] = sum_mu;
            var_z[out_idx] = sum_var;
        }
    }
}

void conv2d_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &aidx, int woho, int fo, int wihi, int fi, int ki,
    int batch_size, int pad_idx, bool bias, unsigned int num_threads,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    const int tot_ops = woho * batch_size * fo;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &aidx, &mu_z, &var_z] {
            conv2d_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a, aidx,
                                woho, fo, wihi, fi, ki, batch_size, pad_idx,
                                bias, start_chunk, end_chunk, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void conv2d_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &zw_idx,
    const std::vector<int> &zud_idx, int woho, int fo, int wihi, int fi, int ki,
    int nr, int n, int batch_size, int pad_idx, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu, std::vector<float> &delta_var)
/**/
{
    float mu_w_tmp;
    int k = wihi * batch_size;
    int ki2 = ki * ki;

    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < n; i++) {
            // indices for mw. Note that nr = n / fo. Indices's lowest value
            // starts at 1
            int widx_tmp = zw_idx[(col % wihi) * nr + i % nr] +
                           (i / nr) * ki2 * fi + row * ki2 - 1;

            // indices for deltaM
            int aidx_tmp = zud_idx[col % wihi + wihi * (i % nr)];

            if (aidx_tmp > -1) {
                aidx_tmp += (i / nr) * woho + (col / wihi) * woho * fo;
                mu_w_tmp = mu_w[widx_tmp];

                sum_mu += delta_mu_out[aidx_tmp - 1] * mu_w_tmp;
                sum_var += mu_w_tmp * delta_var_out[aidx_tmp - 1] * mu_w_tmp;
            }
        }

        int out_idx = wihi * (col / wihi) * fi + col % wihi + row * wihi;

        delta_mu[out_idx] = sum_mu * jcb[row * k + col];
        delta_var[out_idx] = sum_var * jcb[row * k + col] * jcb[row * k + col];
    }
}

void conv2d_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &zw_idx,
    const std::vector<int> &zud_idx, int woho, int fo, int wihi, int fi, int ki,
    int nr, int n, int batch_size, int pad_idx, unsigned int num_threads,
    std::vector<float> &delta_mu, std::vector<float> &delta_var)
/**/
{
    const int tot_ops = wihi * batch_size * fi;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &jcb, &delta_mu_out, &delta_var_out,
                              &zw_idx, &zud_idx, &delta_mu, &delta_var] {
            conv2d_bwd_delta_z(mu_w, jcb, delta_mu_out, delta_var_out, zw_idx,
                               zud_idx, woho, fo, wihi, fi, ki, nr, n,
                               batch_size, pad_idx, start_chunk, end_chunk,
                               delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void permute_jacobian(std::vector<float> &jcb_0, int wihi, int fi,
                      int batch_size, std::vector<float> &jcb)
/*
 */
{
    for (int col = 0; col < wihi * fi; col++) {
        for (int row = 0; row < batch_size; row++) {
            // Note that (col/(w * h)) equivalent to floorf((col/(w * h)))
            // because of interger division
            jcb[wihi * (col / wihi) * batch_size + col % wihi + row * wihi] =
                jcb_0[row * wihi * fi + col];
        }
    }
}

void conv2d_bwd_delta_w(const std::vector<float> &var_w,
                        const std::vector<float> &mu_a,
                        const std::vector<float> &delta_mu_out,
                        const std::vector<float> &delta_var_out,
                        const std::vector<int> &aidx, int batch_size, int k,
                        int woho, int wihi, int fi, int ki, int pad_idx,
                        int start_chunk, int end_chunk,
                        std::vector<float> &delta_mu_w,
                        std::vector<float> &delta_var_w)
/*
 */
{
    int ki2 = ki * ki;
    int m = ki2 * fi;
    int n = woho * batch_size;

    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < n; i++) {
            int aidx_tmp = aidx[ki2 * (i % woho) + row % ki2];

            if (aidx_tmp > -1) {
                aidx_tmp += (row / ki2) * wihi + (i / woho) * wihi * fi;
                // Indices's lowest value starts at 1
                float mu_a_tmp = mu_a[aidx_tmp - 1];
                sum_mu += mu_a_tmp * delta_mu_out[col * n + i];
                sum_var += mu_a_tmp * delta_var_out[col * n + i] * mu_a_tmp;
            }
        }
        float var_w_tmp = var_w[col * m + row];
        delta_mu_w[col * m + row] = sum_mu * var_w_tmp;
        delta_var_w[col * m + row] = sum_var * var_w_tmp * var_w_tmp;
    }
}

void conv2d_bwd_delta_w_mp(const std::vector<float> &var_w,
                           const std::vector<float> &mu_a,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           const std::vector<int> &aidx, int batch_size, int k,
                           int woho, int wihi, int fi, int ki, int pad_idx,
                           unsigned int num_threads,
                           std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w)
/*
 */
{
    int ki2 = ki * ki;
    const int tot_ops = ki2 * fi * k;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &mu_a, &delta_mu_out, &delta_var_out,
                              &aidx, &delta_mu_w, &delta_var_w] {
            conv2d_bwd_delta_w(var_w, mu_a, delta_mu_out, delta_var_out, aidx,
                               batch_size, k, woho, wihi, fi, ki, pad_idx,
                               start_chunk, end_chunk, delta_mu_w, delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void conv2d_bwd_delta_b(const std::vector<float> &var_b,
                        const std::vector<float> &delta_mu_out,
                        const std::vector<float> &delta_var_out, int n, int k,
                        std::vector<float> &delta_mu_b,
                        std::vector<float> &delta_var_b)
/*
 */
{
    for (int col = 0; col < k; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_mu += delta_mu_out[col * n + i];
            sum_var += delta_var_out[col * n + i];
        }
        delta_mu_b[col] = sum_mu * var_b[col];
        delta_var_b[col] = sum_var * var_b[col] * var_b[col];
    }
}

void permute_delta(const std::vector<float> &delta_mu_0,
                   const std::vector<float> &delta_var_0, int woho, int kp,
                   int batch_size, std::vector<float> &delta_mu,
                   std::vector<float> &delta_var)
/*
 */
{
    for (int col = 0; col < kp; col++) {
        for (int row = 0; row < batch_size; row++) {
            // Note that (col/(w * h)) equvalent to floorf((col/(w * h)))
            // because of interger division
            delta_mu[woho * (col / woho) * batch_size + col % woho +
                     row * woho] = delta_mu_0[row * kp + col];
            delta_var[woho * (col / woho) * batch_size + col % woho +
                      row * woho] = delta_var_0[row * kp + col];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

std::tuple<int, int> compute_downsample_img_size_v2(int kernel, int stride,
                                                    int wi, int hi, int pad,
                                                    int pad_type)
/* compute the size of downsampling images i.e. reduction of image size
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: width of the input image
 *    hi: height of the input image
 *    pad: number of paddings
 *    pad_type: padding type
 *
 * returns:
 *    wo: width of the output image
 *    ho: height of the output image
 *    */
{
    int wo, ho, nom_w, nom_h;

    // Compute nominator of conv. formulation given a padding type
    if (pad_type == 1) {
        nom_w = wi - kernel + 2 * pad;
        nom_h = hi - kernel + 2 * pad;
    } else if (pad_type == 2) {
        nom_w = wi - kernel + pad;
        nom_h = hi - kernel + pad;
    } else {
        nom_w = wi - kernel;
        nom_h = hi - kernel;
    }

    // Check validity of the conv. hyper-parameters such as wi, hi, kernel,
    // stride
    if (nom_w % stride == 0 && nom_h % stride == 0) {
        wo = nom_w / stride + 1;
        ho = nom_h / stride + 1;
    } else {
        LOG(LogLevel::ERROR,
            "Invalid hyperparameters for conv2d layer: "
            "wi=" +
                std::to_string(wi) + ", hi=" + std::to_string(hi) +
                ", kernel=" + std::to_string(kernel) +
                ", stride=" + std::to_string(stride));
    }

    return {wo, ho};
}

std::tuple<int, int> get_number_param_conv_v2(int kernel, int fi, int fo,
                                              bool use_bias)
/* Get the number of parameters for conv. and tconv. layer.
 *
 * Args:
 *    kernel: Size of the receptive field
 *    fi: Number of filters for input image
 *    fo: Number of filters for output image
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = kernel * kernel * fi * fo;
    if (use_bias) {
        n_b = fo;
    } else {
        n_b = 0;
    }

    return {n_w, n_b};
}

Conv2dIndex get_conv2d_idx(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out, int param_pad_idx)
/*
 * Get index matrices for convolutional layer.
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    wo: width of the output image
 *    ho: height of the output image
 *    pad: Number of padding
 *    pad_type: Type of paddings
 *    pad_idx_in: Padding index for the input image
 *    pad_idx_out: Index for the padding of the output image
 *    param_pad_idx: Index for the padding of the parameters
 *
 * Returns:
 *    FCzwa_1_idx: Index for the parameters sorted as the input hidden state
 *      ordering
 *    FCzwa_2_idx: Index for the receptive field indices sorted as the input
 *      hidden state ordering
 *    Szz_ud_idx: Index for the output hidden states sorted as the input
 *      hidden state ordering
 *    w: Width of three above-mentioned index matrix
 *    h: Height of three above_mentioned idex matrix
 * */
{
    // Initialize pointers
    std::vector<int> raw_img, img, padded_img, Fmwa_2_idx;
    std::vector<int> FCzwa_1_idx, FCzwa_2_idx, Szz_ud_idx, tmp;
    int w_p, h_p, num_elements;

    // Generate image indices
    std::tie(raw_img, img, padded_img, w_p, h_p) =
        image_construction(wi, hi, pad, pad_idx_in, pad_type);

    // Get indices for receptive field
    Fmwa_2_idx =
        get_receptive_field(img, padded_img, kernel, stride, wo, ho, w_p, h_p);

    // Get unique indices and its frequency of the receptive field
    auto Fmwa_2 = get_ref_idx(Fmwa_2_idx, pad, pad_idx_in);

    // Get indices for FCzwa 1
    tmp = get_FCzwa_1_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                          Fmwa_2.base_idx, param_pad_idx, Fmwa_2.w, Fmwa_2.h);

    FCzwa_1_idx = transpose_matrix(tmp, Fmwa_2.w, Fmwa_2.h);

    // Get indices for FCzwa 2
    FCzwa_2_idx = get_FCzwa_2_idx(Fmwa_2_idx, pad, pad_idx_in, Fmwa_2.ref,
                                  Fmwa_2.base_idx, Fmwa_2.w, Fmwa_2.h);

    // Get indices for Szz ud
    Szz_ud_idx =
        get_Szz_ud_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                       Fmwa_2.base_idx, pad_idx_out, Fmwa_2.w, Fmwa_2.h);

    return {Fmwa_2_idx, FCzwa_1_idx, FCzwa_2_idx,
            Szz_ud_idx, Fmwa_2.w,    Fmwa_2.h};
}
