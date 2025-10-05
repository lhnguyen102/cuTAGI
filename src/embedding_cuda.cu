#include "../include/custom_logger.h"
#include "../include/embedding_cpu.h"
#include "../include/embedding_cuda.cuh"

__global__ void embedding_fwd_mean_var(const float *mu_a, const float *mu_w,
                                       const float *var_w, int embedding_dim,
                                       int num_inputs, int batch_size,
                                       int padding_idx, float *mu_z,
                                       float *var_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_inputs;

    if (idx < total_size) {
        int batch_idx = idx / num_inputs;
        int input_idx = idx % num_inputs;

        int cat_idx = static_cast<int>(mu_a[idx]);
        int out_idx =
            batch_idx * num_inputs * embedding_dim + input_idx * embedding_dim;

        if (cat_idx == padding_idx) {
            for (int k = 0; k < embedding_dim; k++) {
                mu_z[out_idx + k] = 0.0f;
                var_z[out_idx + k] = 0.0f;
            }
        } else {
            for (int k = 0; k < embedding_dim; k++) {
                mu_z[out_idx + k] = mu_w[cat_idx * embedding_dim + k];
                var_z[out_idx + k] = var_w[cat_idx * embedding_dim + k];
            }
        }
    }
}

__global__ void embedding_bwd_delta_w(const float *mu_a, const float *var_w,
                                      const float *delta_mu,
                                      const float *delta_var, int embedding_dim,
                                      int num_inputs, int batch_size,
                                      int padding_idx, float *delta_mu_w,
                                      float *delta_var_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * num_inputs;

    if (idx < total_size) {
        int batch_idx = idx / num_inputs;
        int input_idx = idx % num_inputs;

        int cat_idx = static_cast<int>(mu_a[idx]);

        if (cat_idx != padding_idx) {
            int out_idx = batch_idx * num_inputs * embedding_dim +
                          input_idx * embedding_dim;

            for (int k = 0; k < embedding_dim; k++) {
                int emb_idx = cat_idx * embedding_dim + k;
                float grad_mu = delta_mu[out_idx + k] * var_w[emb_idx];
                float grad_var = delta_var[out_idx + k] * var_w[emb_idx];

                atomicAdd(&delta_mu_w[emb_idx], grad_mu);
                atomicAdd(&delta_var_w[emb_idx], grad_var);
            }
        }
    }
}

EmbeddingCuda::EmbeddingCuda(int num_embeddings, int embedding_dim, float scale,
                             int padding_idx, int device_idx)
    : embedding_dim(embedding_dim),
      num_embeddings(num_embeddings),
      scale(scale),
      padding_idx(padding_idx) {
    this->device_idx = device_idx;
    this->num_weights = num_embeddings * embedding_dim;
    this->num_biases = 0;

    if (this->training) {
        this->allocate_param_delta();
    }
}

EmbeddingCuda::~EmbeddingCuda() {}

std::string EmbeddingCuda::get_layer_info() const {
    return "Embedding(" + std::to_string(this->num_embeddings) + "->" +
           std::to_string(this->embedding_dim) + ")";
}

std::string EmbeddingCuda::get_layer_name() const { return "EmbeddingCuda"; }

LayerType EmbeddingCuda::get_layer_type() const { return LayerType::Embedding; }

void EmbeddingCuda::init_weight_bias() {
    auto weights = initialize_embedding_values(
        this->num_embeddings, this->embedding_dim, this->scale);
    this->mu_w = std::get<0>(weights);
    this->var_w = std::get<1>(weights);

    this->allocate_param_memory();
    this->params_to_device();
    if (this->training) {
        this->allocate_param_delta();
    }
}

void EmbeddingCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states) {
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = this->input_size * this->embedding_dim;
    }

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    int total_threads = batch_size * this->input_size;
    constexpr int NUM_THREADS = 256;
    int num_blocks = (total_threads + NUM_THREADS - 1) / NUM_THREADS;

    embedding_fwd_mean_var<<<num_blocks, NUM_THREADS>>>(
        cu_input_states->d_mu_a, this->d_mu_w, this->d_var_w,
        this->embedding_dim, this->input_size, batch_size, this->padding_idx,
        cu_output_states->d_mu_a, cu_output_states->d_var_a);

    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void EmbeddingCuda::backward(BaseDeltaStates &input_delta_states,
                             BaseDeltaStates &output_delta_states,
                             BaseTempStates &temp_states, bool state_udapte) {
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);

    int batch_size = input_delta_states.block_size;

    if (this->param_update) {
        cudaMemset(this->d_delta_mu_w, 0, this->num_weights * sizeof(float));
        cudaMemset(this->d_delta_var_w, 0, this->num_weights * sizeof(float));

        int total_threads = batch_size * this->input_size;
        constexpr int NUM_THREADS = 256;
        int num_blocks = (total_threads + NUM_THREADS - 1) / NUM_THREADS;

        embedding_bwd_delta_w<<<num_blocks, NUM_THREADS>>>(
            cu_next_bwd_states->d_mu_a, this->d_var_w,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->embedding_dim,
            this->input_size, batch_size, this->padding_idx, this->d_delta_mu_w,
            this->d_delta_var_w);
    }
}

std::unique_ptr<BaseLayer> EmbeddingCuda::to_host() {
    std::unique_ptr<BaseLayer> host_emb =
        std::make_unique<Embedding>(this->num_embeddings, this->embedding_dim,
                                    this->scale, this->padding_idx);
    host_emb->mu_w = this->mu_w;
    host_emb->var_w = this->var_w;

    return host_emb;
}
