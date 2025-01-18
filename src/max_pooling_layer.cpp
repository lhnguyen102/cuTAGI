#include "../include/max_pooling_layer.h"

#include "../include/conv2d_layer.h"

MaxPool2d::MaxPool2d(size_t kernel_size, int stride, int padding,
                     int padding_type)
    : kernel_size(kernel_size),
      stride(stride),
      padding_type(padding_type),
      padding(padding) {}

MaxPool2d::~MaxPool2d() {}

std::string MaxPool2d::get_layer_info() const {
    return "MaxPool2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

std::string MaxPool2d::get_layer_name() const { return "MaxPool2d"; }

LayerType MaxPool2d::get_layer_type() const { return LayerType::Pool2d; }

void MaxPool2d::compute_input_output_size(const InitArgs &args) {
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

////////////////////////////////////////////////////////////////////////////////
// CPU Kernels for MaxPool2d
////////////////////////////////////////////////////////////////////////////////
void max2dpool_overlapped_mean_var(
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &pool_idx, const std::vector<int> a_idx, int woho,
    int wihi, int ki, int k, int start_chunk, int end_chunk,
    std::vector<int> &max_pool_idx, std::vector<float> &mu_z,
    std::vector<float> &var_z) {
    int ki2 = ki * ki;
    for (int col = start_chunk; col < end_chunk; col++) {
        float max_mu_z = 0;
        float max_var_z = 0;
        int max_pool_idx_tmp = -1;
        for (int i = 0; i < ki2; i++) {
            int a_idx_tmp = a_idx[col % woho + woho * i];

            if (a_idx_tmp > -1) {
                a_idx_tmp += (col / woho) * wihi;
                // index in a_idx starts at 1
                float tmp_mu = mu_a[a_idx_tmp];
                if (tmp_mu > max_mu_z) {
                    max_mu_z = tmp_mu;
                    max_var_z = var_a[a_idx_tmp];
                    max_pool_idx_tmp = a_idx_tmp;
                }
            }
        }
        mu_z[col] = max_mu_z;
        var_z[col] = max_var_z;
        max_pool_idx[col] = max_pool_idx_tmp;
    }
}

void max2dpool_mean_var(const std::vector<float> &mu_a,
                        const std::vector<float> &var_a,
                        const std::vector<int> &pool_idx,
                        const std::vector<int> a_idx, int woho, int wihi,
                        int ki, int k, int start_chunk, int end_chunk,
                        std::vector<int> &max_pool_idx,
                        std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    int ki2 = ki * ki;
    for (int col = start_chunk; col < end_chunk; col++) {
        float max_mu_z = 0;
        float max_var_z = 0;
        int max_pool_idx_tmp = -1;
        for (int i = 0; i < ki2; i++) {
            int a_idx_tmp =
                a_idx[col % woho + woho * i] + (col / woho) * wihi - 1;
            // index in a_idx starts at 1
            float tmp_mu = mu_a[a_idx_tmp];
            if (tmp_mu > max_mu_z) {
                max_mu_z = tmp_mu;
                max_var_z = var_a[a_idx_tmp];
                max_pool_idx_tmp = a_idx_tmp;
            }
        }
        mu_z[col] = max_mu_z;
        var_z[col] = max_var_z;
        max_pool_idx[col] = max_pool_idx_tmp;
    }
}

void max2dpool_bwd_overlapped_delta_z(
    const std::vector<float> &jcb, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &z_ud_idx,
    int woho, int wihi, int ki, int n, int k, int pad_idx, int start_chunk,
    int end_chunk, std::vector<float> &delta_mu,
    std::vector<float> &delta_var) {
    int ki2 = ki * ki;
}

void max2dpool_bwd_delta_z(const std::vector<int> &max_pool_idx,
                           const std::vector<float> &jcb,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out, int wo,
                           int ki, int k, int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var) {
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
