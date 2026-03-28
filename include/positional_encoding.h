#pragma once
#include <cmath>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

class PositionalEncoding : public BaseLayer {
   public:
    int embed_dim;
    size_t max_seq_len;
    std::vector<float> pe_cache;

    PositionalEncoding(int embed_dim, size_t max_seq_len = 2048);
    ~PositionalEncoding();

    PositionalEncoding(const PositionalEncoding &) = delete;
    PositionalEncoding &operator=(const PositionalEncoding &) = delete;

    PositionalEncoding(PositionalEncoding &&) = default;
    PositionalEncoding &operator=(PositionalEncoding &&) = default;

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
};
