#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

struct EmbeddingProp {
    std::vector<size_t> cat_sizes, emb_sizes;
};

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    int num_embeddings, int embedding_dim, float scale,
    unsigned int *seed = nullptr);

void fwd_emb(std::vector<float> &ma, std::vector<float> &mu_w,
             std::vector<float> &var_w, int embedding_dim, int num_inputs,
             int batch_size, int padding_idx, std::vector<float> &mu_z,
             std::vector<float> &var_z);

void bwd_emb(std::vector<float> &ma, std::vector<float> &var_w,
             std::vector<float> &delta_mu, std::vector<float> &delta_var,
             int embedding_dim, int num_inputs, int batch_size, int padding_idx,
             std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w);

int calculate_embedding_size(int num_categories);

class Embedding : public BaseLayer {
   public:
    int embedding_dim;
    int num_embeddings;
    float scale;
    int padding_idx;

    Embedding(int num_embeddings, int embedding_dim, int input_size,
              float scale = 1.0f, int padding_idx = -1, int device_idx = 0);

    ~Embedding();

    // Delete copy constructor and copy assignment
    Embedding(const Embedding &) = delete;
    Embedding &operator=(const Embedding &) = delete;

    // Optionally implement move constructor and move assignment
    Embedding(Embedding &&) = default;
    Embedding &operator=(Embedding &&) = default;

    virtual std::string get_layer_info() const override;

    virtual std::string get_layer_name() const override;

    virtual LayerType get_layer_type() const override;

    void init_weight_bias() override;

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states) override;

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};
