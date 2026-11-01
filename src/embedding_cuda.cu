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

__global__ void embedding_token_rows(const float *mu_a, int num_tokens,
                                     int padding_idx, int padding_row,
                                     int *rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens) {
        int cat_idx = static_cast<int>(mu_a[idx]);
        rows[idx] = cat_idx == padding_idx ? padding_row : cat_idx;
    }
}

// One block per touched row; tokens of a segment are summed in token order
// with the unfused multiply-add of the CPU bwd_emb (the host build has no FMA
// target), so the sums are bit-equal.
__global__ void embedding_bwd_delta_w(
    const float *var_w, const float *delta_mu, const float *delta_var,
    const int *touched_row, const int *segment_offset, const int *segment_count,
    const int *sorted_token, int embedding_dim, float *delta_mu_w,
    float *delta_var_w) {
    int row = touched_row[blockIdx.x];
    int begin = segment_offset[blockIdx.x];
    int end = begin + segment_count[blockIdx.x];
    for (int k = threadIdx.x; k < embedding_dim; k += blockDim.x) {
        size_t w = static_cast<size_t>(row) * embedding_dim + k;
        float var_w_tmp = var_w[w];
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int s = begin; s < end; s++) {
            size_t z = static_cast<size_t>(sorted_token[s]) * embedding_dim + k;
            sum_mu = __fadd_rn(sum_mu, __fmul_rn(delta_mu[z], var_w_tmp));
            sum_var = __fadd_rn(
                sum_var,
                __fmul_rn(__fmul_rn(delta_var[z], var_w_tmp), var_w_tmp));
        }
        delta_mu_w[w] = sum_mu;
        delta_var_w[w] = sum_var;
    }
}

EmbeddingCuda::EmbeddingCuda(int num_embeddings, int embedding_dim,
                             int input_size, float scale, int padding_idx,
                             int device_idx)
    : embedding_dim(embedding_dim),
      num_embeddings(num_embeddings),
      scale(scale),
      padding_idx(padding_idx) {
    this->device_idx = device_idx;
    this->num_weights = num_embeddings * embedding_dim;
    this->num_biases = 0;
    this->bias = false;

    if (input_size > 0) {
        this->input_size = input_size;
        this->output_size = embedding_dim;
    }

    if (this->training) {
        this->allocate_param_delta();
    }
}

EmbeddingCuda::~EmbeddingCuda() {}

int EmbeddingCuda::get_max_num_states() {
    return this->input_size * this->embedding_dim;
}

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

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    int total_threads = batch_size * this->input_size;
    constexpr int NUM_THREADS = 256;
    int num_blocks = (total_threads + NUM_THREADS - 1) / NUM_THREADS;

    embedding_fwd_mean_var<<<num_blocks, NUM_THREADS>>>(
        cu_input_states->d_mu_a, this->d_mu_w, this->d_var_w,
        this->embedding_dim, this->input_size, batch_size, this->padding_idx,
        cu_output_states->d_mu_a, cu_output_states->d_var_a);

    cu_output_states->width = this->out_width;
    cu_output_states->height = this->out_height;
    cu_output_states->depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.seq_len = this->input_size;
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
        int num_tokens = batch_size * this->input_size;
        constexpr int NUM_THREADS = 256;
        int num_blocks = (num_tokens + NUM_THREADS - 1) / NUM_THREADS;

        this->segments.reserve(num_tokens);
        embedding_token_rows<<<num_blocks, NUM_THREADS>>>(
            cu_next_bwd_states->d_mu_a, num_tokens, this->padding_idx,
            this->num_embeddings, this->segments.d_row);
        this->segments.group(num_tokens);
        this->num_touched_rows = this->segments.num_touched;
        if (this->num_touched_rows > 0) {
            int last_row;
            cudaMemcpy(
                &last_row,
                this->segments.d_touched_row + this->num_touched_rows - 1,
                sizeof(int), cudaMemcpyDeviceToHost);
            if (last_row == this->num_embeddings) this->num_touched_rows--;
        }
        if (this->num_touched_rows > 0) {
            embedding_bwd_delta_w<<<this->num_touched_rows, NUM_THREADS>>>(
                this->d_var_w, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var,
                this->segments.d_touched_row, this->segments.d_segment_offset,
                this->segments.d_segment_count, this->segments.d_sorted_item,
                this->embedding_dim, this->d_delta_mu_w, this->d_delta_var_w);
        }
    }
}

void EmbeddingCuda::update_weights() {
    this->update_weight_rows(this->segments.d_touched_row,
                             this->num_touched_rows, this->embedding_dim);
}

std::unique_ptr<BaseLayer> EmbeddingCuda::to_host() {
    std::unique_ptr<BaseLayer> host_emb = std::make_unique<Embedding>(
        this->num_embeddings, this->embedding_dim, this->input_size,
        this->scale, this->padding_idx);
    host_emb->mu_w = this->mu_w;
    host_emb->var_w = this->var_w;

    return host_emb;
}
