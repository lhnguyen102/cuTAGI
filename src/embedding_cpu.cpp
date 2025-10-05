#include "../include/embedding_cpu.h"

#include "../include/common.h"
#include "../include/custom_logger.h"

#ifdef USE_CUDA
#include "../include/embedding_cuda.cuh"
#endif

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    int num_embeddings, int embedding_dim, float scale, unsigned int *seed)
/* Initialize embedding values for each class separately
 */
{
    std::vector<float> mu_emb;
    std::vector<float> var_emb;

    std::mt19937 gen(seed ? *seed : std::random_device{}());

    std::normal_distribution<float> norm_dist(0.0f, scale);

    for (int i = 0; i < num_embeddings; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            mu_emb.push_back(norm_dist(gen));
            var_emb.push_back(pow(scale, 2));
        }
    }

    return {mu_emb, var_emb};
}

///////////////////////////////////////////////////////////////////////////////
// Embedding Layer
///////////////////////////////////////////////////////////////////////////////
void fwd_emb(std::vector<float> &mu_a, std::vector<float> &mu_w,
             std::vector<float> &var_w, int embedding_dim, int num_inputs,
             int batch_size, int padding_idx, std::vector<float> &mu_z,
             std::vector<float> &var_z)
/**/
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_inputs; j++) {
            int cat_idx = mu_a[j + i * num_inputs];
            int out_idx = i * num_inputs * embedding_dim + j * embedding_dim;
            if (cat_idx == padding_idx) {
                for (int k = 0; k < embedding_dim; k++) {
                    mu_z[out_idx + k] = 0;
                    var_z[out_idx + k] = 0;
                }
                continue;
            }
            for (int k = 0; k < embedding_dim; k++) {
                mu_z[out_idx + k] = mu_w[cat_idx * embedding_dim + k];
                var_z[out_idx + k] = var_w[cat_idx * embedding_dim + k];
            }
        }
    }
}

void bwd_emb(std::vector<float> &ma, std::vector<float> &var_w,
             std::vector<float> &delta_mu, std::vector<float> &delta_var,
             int embedding_dim, int num_inputs, int batch_size, int padding_idx,
             std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_inputs; j++) {
            int cat_idx = ma[j + i * num_inputs];
            if (cat_idx == padding_idx) {
                continue;
            }
            int out_idx = i * num_inputs * embedding_dim + j * embedding_dim;

            // Note: delta_mu_w and delta_var_w are not zeroed out before the
            // backward pass as we can have more than one input for the same
            // embedding.
            for (int k = 0; k < embedding_dim; k++) {
                delta_mu_w[cat_idx * embedding_dim + k] +=
                    delta_mu[out_idx + k] * var_w[cat_idx * embedding_dim + k];
                delta_var_w[cat_idx * embedding_dim + k] +=
                    delta_var[out_idx + k] * var_w[cat_idx * embedding_dim + k];
            }
        }
    }
}

int calculate_embedding_size(int num_classes)
/*
Calculate the embedding size based on the number of categories using fast.ai
heuristic.

Args:
    num_classes: Number of classes

Return:
    int: Embedding size

 */
{
    int emb_size = 1.6 * powf(num_classes, 0.56);

    return std::max(600, emb_size);
}

///////////////////////////////////////////////////////////////////////////////
// Embedding Layer Class
///////////////////////////////////////////////////////////////////////////////

Embedding::Embedding(int num_embeddings, int embedding_dim, int input_size,
                     float scale, int padding_idx, int device_idx)
    : embedding_dim(embedding_dim),
      num_embeddings(num_embeddings),
      scale(scale),
      padding_idx(padding_idx) {
    this->device_idx = device_idx;
    this->num_weights = num_embeddings * embedding_dim;
    this->num_biases = 0;

    if (input_size > 0) {
        this->input_size = input_size;
        this->output_size = input_size * embedding_dim;
    }

    if (this->device.compare("cpu") == 0) {
        this->init_weight_bias();
    }

    if (this->training && this->device.compare("cpu") == 0) {
        this->allocate_param_delta();
    }
}

Embedding::~Embedding() {}

std::string Embedding::get_layer_info() const {
    std::string info = "Embedding(";
    info += std::to_string(this->num_embeddings) + "->" +
            std::to_string(this->embedding_dim);
    info += ")";
    return info;
}

std::string Embedding::get_layer_name() const { return "Embedding"; }

LayerType Embedding::get_layer_type() const { return LayerType::Embedding; }

void Embedding::init_weight_bias() {
    auto weights = initialize_embedding_values(
        this->num_embeddings, this->embedding_dim, this->scale);
    this->mu_w = std::get<0>(weights);
    this->var_w = std::get<1>(weights);
}

void Embedding::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = this->input_size * this->embedding_dim;
    }

    fwd_emb(input_states.mu_a, this->mu_w, this->var_w, this->embedding_dim,
            this->input_size, batch_size, this->padding_idx, output_states.mu_a,
            output_states.var_a);

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void Embedding::backward(BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         BaseTempStates &temp_states, bool state_udapte) {
    int batch_size = input_delta_states.block_size;

    if (this->param_update) {
        bwd_emb(this->bwd_states->mu_a, this->var_w,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->embedding_dim, this->input_size, batch_size,
                this->padding_idx, this->delta_mu_w, this->delta_var_w);
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Embedding::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    auto cuda_layer = std::make_unique<EmbeddingCuda>(
        this->num_embeddings, this->embedding_dim, this->input_size,
        this->scale, this->padding_idx, this->device_idx);

    auto base_cuda = dynamic_cast<BaseLayerCuda *>(cuda_layer.get());
    base_cuda->copy_params_from(*this);

    return cuda_layer;
}
#endif
