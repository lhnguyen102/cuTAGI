#include "../include/cuda_error_checking.cuh"
#include "../include/data_struct_cuda.cuh"
#include "../include/positional_encoding_cuda.cuh"
#include "../include/positional_encoding_cuda_kernel.cuh"

PositionalEncodingCuda::PositionalEncodingCuda(int embed_dim,
                                               size_t max_seq_len,
                                               int device_idx)
    : embed_dim(embed_dim), max_seq_len(max_seq_len) {
    this->device_idx = device_idx;
    this->input_size = embed_dim;
    this->output_size = embed_dim;

    std::vector<float> pe_cache = build_sinusoidal_pe(embed_dim, max_seq_len);
    cudaSetDevice(this->device_idx);
    CHECK_CUDA_ERROR(
        cudaMalloc((void **)&d_pe_cache, pe_cache.size() * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pe_cache, pe_cache.data(),
                                pe_cache.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
}

PositionalEncodingCuda::~PositionalEncodingCuda() { cudaFree(d_pe_cache); }

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
    check_positional_encoding_input(actual_size, seq_len, this->embed_dim,
                                    this->max_seq_len);

    int total = batch_size * seq_len * actual_size;
    constexpr int THREADS = 256;
    int blocks = (total + THREADS - 1) / THREADS;
    positional_encoding_fwd<<<blocks, THREADS>>>(
        cu_in->d_mu_a, cu_in->d_var_a, d_pe_cache, batch_size, seq_len,
        actual_size, cu_out->d_mu_a, cu_out->d_var_a, cu_out->d_jcb);
    CHECK_LAST_CUDA_ERROR();

    cu_out->width = this->out_width;
    cu_out->height = this->out_height;
    cu_out->depth = this->out_channels;
    cu_out->block_size = batch_size;
    cu_out->actual_size = actual_size;
    cu_out->seq_len = seq_len;
}

std::unique_ptr<BaseLayer> PositionalEncodingCuda::to_host() {
    return std::make_unique<PositionalEncoding>(
        this->embed_dim, this->max_seq_len, this->device_idx);
}
