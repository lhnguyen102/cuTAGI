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

struct EmbeddingBagProp {
    std::vector<size_t> cat_sizes, emb_sizes, num_bags, bag_sizes;
};

struct Offsets {
    std::vector<int> batch_in_offsets;
    std::vector<int> batch_out_offsets;
    std::vector<int> cat_in_offsets;
    std::vector<int> cat_out_offsets;
};

Offsets precompute_offsets(const std::vector<size_t> &num_bags,
                           const std::vector<size_t> &bag_sizes, int num_cat,
                           int batch_size);

void fwd_bag_emb(std::vector<float> &mu_a, std::vector<float> &mu_w,
                 std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
                 std::vector<size_t> &emb_sizes, std::vector<size_t> &num_bags,
                 std::vector<size_t> &bag_sizes, int num_cat, int batch_size,
                 std::vector<float> &mu_z, std::vector<float> &var_z);

void bwd_bag_emb(std::vector<float> &mu_a, std::vector<float> &var_w,
                 std::vector<float> &delta_mu, std::vector<float> &delta_var,
                 std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
                 std::vector<size_t> &num_bags, std::vector<size_t> &bag_sizes,
                 int num_cat, int batch_size, std::vector<float> &delta_mu_w,
                 std::vector<float> &delta_var_w);

class EmbeddingBag : public BaseLayer {
   public:
    std::vector<size_t> cat_sizes;
    std::vector<size_t> emb_sizes;
    std::vector<size_t> num_bags;
    std::vector<size_t> bag_sizes;
    int num_cat;
    float scale;

    EmbeddingBag(const std::vector<size_t> &cat_sizes,
                 const std::vector<size_t> &emb_sizes,
                 const std::vector<size_t> &num_bags,
                 const std::vector<size_t> &bag_sizes, float scale = 1.0f,
                 int device_idx = 0);

    ~EmbeddingBag();

    // Delete copy constructor and copy assignment
    EmbeddingBag(const EmbeddingBag &) = delete;
    EmbeddingBag &operator=(const EmbeddingBag &) = delete;

    // Optionally implement move constructor and move assignment
    EmbeddingBag(EmbeddingBag &&) = default;
    EmbeddingBag &operator=(EmbeddingBag &&) = default;

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

   private:
    void calculate_sizes();
};
