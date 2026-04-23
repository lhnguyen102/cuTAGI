#include <cmath>

#include "../include/cuda_error_checking.cuh"
#include "../include/data_struct_cuda.cuh"
#include "../include/positional_encoding_cuda.cuh"
#include "../include/positional_encoding_cuda_kernel.cuh"

PositionalEncodingCuda::PositionalEncodingCuda(int embed_dim,
                                               size_t max_seq_len,
                                               int device_idx)
    : embed_dim(embed_dim), max_seq_len(max_seq_len) {
    this->num_weights = 0;
    this->num_biases = 0;
    this->device_idx = device_idx;

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

    cudaSetDevice(this->device_idx);
    CHECK_CUDA_ERROR(
        cudaMalloc((void **)&d_pe_cache, pe_cache.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pe_cache, pe_cache.data(),
                                pe_cache.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
}

PositionalEncodingCuda::~PositionalEncodingCuda() {
    if (d_pe_cache) {
        cudaFree(d_pe_cache);
        d_pe_cache = nullptr;
    }
}

std::string PositionalEncodingCuda::get_layer_info() const {
    return "PositionalEncoding(dim=" + std::to_string(this->embed_dim) + ")";
}

std::string PositionalEncodingCuda::get_layer_name() const {
    return "PositionalEncodingCuda";
}

LayerType PositionalEncodingCuda::get_layer_type() const {
    return LayerType::Activation;
}

void PositionalEncodingCuda::forward(BaseHiddenStates &input_states,
                                     BaseHiddenStates &output_states,
                                     BaseTempStates &temp_states) {
    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = cu_in->block_size;
    int seq_len = cu_in->seq_len;
    int actual_size = cu_in->actual_size;
    size_t total = (size_t)batch_size * seq_len * actual_size;

    if (total > 0) {
        constexpr int THREADS = 256;
        int blocks = (int)((total + THREADS - 1) / THREADS);
        positional_encoding_fwd<<<blocks, THREADS>>>(
            cu_in->d_mu_a, cu_in->d_var_a, d_pe_cache, batch_size, seq_len,
            actual_size, cu_out->d_mu_a, cu_out->d_var_a, cu_out->d_jcb);
        CHECK_LAST_CUDA_ERROR();
    }

    this->input_size = actual_size;
    this->output_size = actual_size;

    cu_out->width = this->out_width;
    cu_out->height = this->out_height;
    cu_out->depth = this->out_channels;
    cu_out->block_size = batch_size;
    cu_out->actual_size = actual_size;
    cu_out->seq_len = seq_len;
}

std::unique_ptr<BaseLayer> PositionalEncodingCuda::to_host() {
    return std::make_unique<PositionalEncoding>(this->embed_dim,
                                                this->max_seq_len);
}
