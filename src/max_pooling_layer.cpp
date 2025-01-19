#include "../include/max_pooling_layer.h"

#include "../include/conv2d_layer.h"
#include "../include/max_pooling_layer_cuda.cuh"
#include "../include/pooling_layer.h"

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

void MaxPool2d::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;

    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
        // TODO: trigger if changed batch size
        this->max_pool_idx.resize(this->output_size * batch_size);
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

    if (this->overlap) {
        max2dpool_overlapped_mean_var(
            input_states.mu_a, input_states.var_a, this->pool_idx, woho, wihi,
            this->kernel_size, num_states, 0, num_states, this->max_pool_idx,
            output_states.mu_a, output_states.var_a);
    } else {
        max2dpool_mean_var(input_states.mu_a, input_states.var_a,
                           this->pool_idx, woho, wihi, this->kernel_size,
                           num_states, 0, num_states, this->max_pool_idx,
                           output_states.mu_a, output_states.var_a);
    }

    if (training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void MaxPool2d::backward(BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         BaseTempStates &temp_states, bool state_update) {
    // Initialization
    int batch_size = input_delta_states.block_size;

    // Launch kernel
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int num_states = woho * this->out_channels * batch_size;

    if (state_update) {
        if (this->overlap) {
            max2dpool_bwd_overlapped_delta_z(
                this->max_pool_idx, this->bwd_states->jcb,
                input_delta_states.delta_mu, input_delta_states.delta_var, 0,
                num_states, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        } else {
            max2dpool_bwd_delta_z(this->max_pool_idx, this->bwd_states->jcb,
                                  input_delta_states.delta_mu,
                                  input_delta_states.delta_var, 0, num_states,
                                  output_delta_states.delta_mu,
                                  output_delta_states.delta_var);
        }
    }
}
#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MaxPool2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<MaxPool2dCuda>(this->kernel_size, this->stride,
                                           this->padding, this->padding_type);
}
#endif

void MaxPool2d::lazy_index_init()
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
}

void MaxPool2d::preinit_layer() {
    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
    }
}

////////////////////////////////////////////////////////////////////////////////
// CPU Kernels for MaxPool2d
////////////////////////////////////////////////////////////////////////////////
void max2dpool_overlapped_mean_var(
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> a_idx, int woho, int wihi, int ki, int k,
    int start_chunk, int end_chunk, std::vector<int> &max_pool_idx,
    std::vector<float> &mu_z, std::vector<float> &var_z) {
    int ki2 = ki * ki;
    for (int col = start_chunk; col < end_chunk; col++) {
        float max_mu_z = 0;
        float max_var_z = 0;
        int max_pool_idx_tmp = -1;
        for (int i = 0; i < ki2; i++) {
            int a_idx_tmp = a_idx[col % woho + woho * i];

            if (a_idx_tmp > -1) {
                a_idx_tmp += (col / woho) * wihi - 1;
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
            // index in a_idx starts at 1
            int a_idx_tmp =
                a_idx[col % woho + woho * i] + (col / woho) * wihi - 1;
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

void max2dpool_bwd_overlapped_delta_z(const std::vector<int> &max_pool_idx,
                                      const std::vector<float> &jcb,
                                      const std::vector<float> &delta_mu_out,
                                      const std::vector<float> &delta_var_out,
                                      int start_chunk, int end_chunk,
                                      std::vector<float> &delta_mu,
                                      std::vector<float> &delta_var)
/*
 */
{
    // delta_mu and delta_var reset to zero before this step
    for (int col = start_chunk; col < end_chunk; col++) {
        int idx = max_pool_idx[col];
        delta_mu[idx] += delta_mu_out[col] * jcb[idx];
        delta_var[idx] += delta_var_out[col] * jcb[idx] * jcb[idx];
    }
}

void max2dpool_bwd_delta_z(const std::vector<int> &max_pool_idx,
                           const std::vector<float> &jcb,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var) {
    for (int col = start_chunk; col < end_chunk; col++) {
        int idx = max_pool_idx[col];
        delta_mu[idx] = delta_mu_out[col] * jcb[idx];
        delta_var[idx] = delta_var_out[col] * jcb[idx] * jcb[idx];
    }
}
