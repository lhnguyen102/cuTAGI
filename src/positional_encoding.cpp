#include "../include/positional_encoding.h"

#include <cmath>
#include <string>

PositionalEncoding::PositionalEncoding(int embed_dim, size_t max_seq_len)
    : embed_dim(embed_dim), max_seq_len(max_seq_len) {
    this->num_weights = 0;
    this->num_biases = 0;

    pe_cache.resize(max_seq_len * embed_dim);
    for (int pos = 0; pos < (int)max_seq_len; pos++) {
        for (int d = 0; d < embed_dim; d++) {
            float freq =
                1.0f / powf(10000.0f, (2.0f * (static_cast<float>(d) / 2)) /
                                          static_cast<float>(embed_dim));
            float angle = pos * freq;
            int idx = pos * embed_dim + d;
            pe_cache[idx] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }
}

PositionalEncoding::~PositionalEncoding() {}

std::string PositionalEncoding::get_layer_info() const {
    return "PositionalEncoding(dim=" + std::to_string(this->embed_dim) + ")";
}

std::string PositionalEncoding::get_layer_name() const {
    return "PositionalEncoding";
}

LayerType PositionalEncoding::get_layer_type() const {
    // TODO: replace by embedding layer
    return LayerType::Activation;
}

void PositionalEncoding::forward(BaseHiddenStates &input_states,
                                 BaseHiddenStates &output_states,
                                 BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;
    int seq_len = input_states.seq_len;
    int actual_size = input_states.actual_size;

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < actual_size; d++) {
                int idx = (b * seq_len + t) * actual_size + d;
                int pe_idx = t * this->embed_dim + d;
                output_states.mu_a[idx] =
                    input_states.mu_a[idx] + pe_cache[pe_idx];
                output_states.var_a[idx] = input_states.var_a[idx];
                output_states.jcb[idx] = 1.0f;
            }
        }
    }

    this->input_size = actual_size;
    this->output_size = actual_size;

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = actual_size;
    output_states.seq_len = seq_len;
}
