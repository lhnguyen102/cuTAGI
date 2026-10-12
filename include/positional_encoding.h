#pragma once
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

// Sinusoidal table [max_seq_len, embed_dim]:
// PE[t, d] = sin(t / 10000^(2*floor(d/2)/embed_dim)) for even d, cos for odd d.
std::vector<float> build_sinusoidal_pe(int embed_dim, size_t max_seq_len);

class PositionalEncoding : public BaseLayer {
   public:
    int embed_dim;
    size_t max_seq_len;
    std::vector<float> pe_cache;

    PositionalEncoding(int embed_dim, size_t max_seq_len = 2048,
                       int device_idx = 0);
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

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

// Throws when the states do not fit the layer's table.
void check_positional_encoding_input(int actual_size, int seq_len,
                                     int embed_dim, size_t max_seq_len);
