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

std::tuple<std::vector<float>, std::vector<float>> get_embedding_values(
    size_t num_classes, size_t emb_size, float scale,
    unsigned int *seed = nullptr);

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
    int num_cat_var, float scale, unsigned int *seed = nullptr);

void fwd_emb(std::vector<float> &ma, std::vector<float> &mu_w,
             std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
             std::vector<size_t> &emb_sizes, int num_cat, int batch_size,
             std::vector<float> &mu_z, std::vector<float> &var_z);

void bwd_emb(std::vector<float> &ma, std::vector<float> &var_w,
             std::vector<float> &delta_mu, std::vector<float> &delta_var,
             std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
             int num_cat, int batch_size, std::vector<float> &delta_mu_w,
             std::vector<float> &delta_var_w);

int calculate_embedding_size(int num_categories);

class Embedding : public BaseLayer {
   public:
    std::vector<size_t> cat_sizes;
    std::vector<size_t> emb_sizes;
    int num_cat;
    float scale;

    Embedding(const std::vector<size_t> &cat_sizes,
              const std::vector<size_t> &emb_sizes, float scale = 1.0f,
              int device_idx = 0);

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

    // #ifdef USE_CUDA
    //     std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
    // #endif

   private:
    void calculate_sizes();
};
