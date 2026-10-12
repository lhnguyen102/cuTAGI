#include "../include/positional_encoding.h"

#include <cmath>
#include <string>

#include "../include/custom_logger.h"

#ifdef USE_CUDA
#include "../include/positional_encoding_cuda.cuh"
#endif

std::vector<float> build_sinusoidal_pe(int embed_dim, size_t max_seq_len) {
    std::vector<float> pe(max_seq_len * embed_dim);
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (int d = 0; d < embed_dim; d++) {
            float exponent = static_cast<float>(2 * (d / 2)) / embed_dim;
            float angle = pos / powf(10000.0f, exponent);
            pe[pos * embed_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
    return pe;
}

void check_positional_encoding_input(int actual_size, int seq_len,
                                     int embed_dim, size_t max_seq_len) {
    if (actual_size != embed_dim) {
        LOG(LogLevel::ERROR, "PositionalEncoding: input size " +
                                 std::to_string(actual_size) +
                                 " != embed_dim " + std::to_string(embed_dim));
    }
    if (static_cast<size_t>(seq_len) > max_seq_len) {
        LOG(LogLevel::ERROR, "PositionalEncoding: seq_len " +
                                 std::to_string(seq_len) + " > max_seq_len " +
                                 std::to_string(max_seq_len));
    }
}

PositionalEncoding::PositionalEncoding(int embed_dim, size_t max_seq_len,
                                       int device_idx)
    : embed_dim(embed_dim),
      max_seq_len(max_seq_len),
      pe_cache(build_sinusoidal_pe(embed_dim, max_seq_len)) {
    this->device_idx = device_idx;
    this->input_size = embed_dim;
    this->output_size = embed_dim;
}

PositionalEncoding::~PositionalEncoding() {}

std::string PositionalEncoding::get_layer_info() const {
    return "PositionalEncoding(dim=" + std::to_string(this->embed_dim) + ")";
}

std::string PositionalEncoding::get_layer_name() const {
    return "PositionalEncoding";
}

// Activation type: Sequential passes delta states through unchanged, which is
// the exact backward of a constant addition.
LayerType PositionalEncoding::get_layer_type() const {
    return LayerType::Activation;
}

void PositionalEncoding::forward(BaseHiddenStates &input_states,
                                 BaseHiddenStates &output_states,
                                 BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;
    int seq_len = input_states.seq_len;
    int actual_size = input_states.actual_size;
    check_positional_encoding_input(actual_size, seq_len, this->embed_dim,
                                    this->max_seq_len);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < actual_size; d++) {
                int idx = (b * seq_len + t) * actual_size + d;
                output_states.mu_a[idx] =
                    input_states.mu_a[idx] + pe_cache[t * actual_size + d];
                output_states.var_a[idx] = input_states.var_a[idx];
                output_states.jcb[idx] = 1.0f;
            }
        }
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = actual_size;
    output_states.seq_len = seq_len;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> PositionalEncoding::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    return std::make_unique<PositionalEncodingCuda>(
        this->embed_dim, this->max_seq_len, device_idx);
}
#endif
