#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct.h"
#include "row_segments.cuh"

__global__ void embedding_fwd_mean_var(const float *mu_a, const float *mu_w,
                                       const float *var_w, int embedding_dim,
                                       int num_inputs, int batch_size,
                                       int padding_idx, float *mu_z,
                                       float *var_z);

__global__ void embedding_token_rows(const float *mu_a, int num_tokens,
                                     int padding_idx, int padding_row,
                                     int *rows);

__global__ void embedding_bwd_delta_w(
    const float *var_w, const float *delta_mu, const float *delta_var,
    const int *touched_row, const int *segment_offset, const int *segment_count,
    const int *sorted_token, int embedding_dim, float *delta_mu_w,
    float *delta_var_w);

class EmbeddingCuda : public BaseLayerCuda {
   public:
    int embedding_dim;
    int num_embeddings;
    float scale;
    int padding_idx;

    EmbeddingCuda(int num_embeddings, int embedding_dim, int input_size,
                  float scale = 1.0f, int padding_idx = -1, int device_idx = 0);

    ~EmbeddingCuda();

    EmbeddingCuda(const EmbeddingCuda &) = delete;
    EmbeddingCuda &operator=(const EmbeddingCuda &) = delete;

    EmbeddingCuda(EmbeddingCuda &&) = default;
    EmbeddingCuda &operator=(EmbeddingCuda &&) = default;

    int get_max_num_states() override;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states, bool state_udapte) override;

    // Touched rows only; delta_mu_w / delta_var_w hold stale values elsewhere.
    void update_weights() override;

    std::unique_ptr<BaseLayer> to_host() override;

   protected:
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;

   private:
    // Tokens of the last backward grouped by embedding row; padding tokens
    // get row num_embeddings, sort last and are dropped from num_touched_rows.
    RowSegments segments;
    int num_touched_rows = 0;
};
