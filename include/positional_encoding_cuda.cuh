#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct.h"
#include "positional_encoding.h"

class PositionalEncodingCuda : public BaseLayerCuda {
   public:
    int embed_dim;
    size_t max_seq_len;
    std::vector<float> pe_cache;
    float *d_pe_cache = nullptr;

    PositionalEncodingCuda(int embed_dim, size_t max_seq_len = 2048,
                           int device_idx = 0);
    ~PositionalEncodingCuda();

    PositionalEncodingCuda(const PositionalEncodingCuda &) = delete;
    PositionalEncodingCuda &operator=(const PositionalEncodingCuda &) = delete;
    PositionalEncodingCuda(PositionalEncodingCuda &&) = default;
    PositionalEncodingCuda &operator=(PositionalEncodingCuda &&) = default;

    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};
    void update_weights() override {};
    void update_biases() override {};
    void save(std::ofstream &file) override {};
    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};
