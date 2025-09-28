#include "../include/embedding_cpu.h"

#include "../include/common.h"
#include "../include/custom_logger.h"

// #ifdef USE_CUDA
// #include "../include/embedding_cuda.cuh"
// #endif

std::tuple<std::vector<float>, std::vector<float>> get_embedding_values(
    size_t num_classes, size_t emb_size, float scale, unsigned int *seed)
/*
 */
{
    // Initialize pointer
    std::vector<float> mu_w(num_classes * emb_size, 0);
    std::vector<float> var_w(num_classes * emb_size, pow(scale, 2));

    // Mersenne twister PRGN - seed
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Create normal distribution
    std::normal_distribution<float> norm_dist(0.0f, scale);

    // Get sample for weight
    for (size_t i = 0; i < num_classes * emb_size; i++) {
        mu_w[i] = norm_dist(gen);
    }

    return {mu_w, var_w};
}

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
    int num_cat_var, float scale, unsigned int *seed)
/*
 */
{
    // Check dim
    if (cat_sizes.size() != emb_sizes.size() ||
        cat_sizes.size() != num_cat_var) {
        LOG(LogLevel::ERROR, "Mismatch in vector sizes or num_cat_var.");
    }
    // Initialize the embedding vectors
    std::vector<float> mu_emb;
    std::vector<float> var_emb;

    for (int i = 0; i < num_cat_var; i++) {
        auto weight_dist =
            get_embedding_values(cat_sizes[i], emb_sizes[i], scale, seed);

        // Insert the values to the embedding vectors directly using std::get
        mu_emb.insert(mu_emb.end(), std::get<0>(weight_dist).begin(),
                      std::get<0>(weight_dist).end());
        var_emb.insert(var_emb.end(), std::get<1>(weight_dist).begin(),
                       std::get<1>(weight_dist).end());
    }

    return {mu_emb, var_emb};
}

///////////////////////////////////////////////////////////////////////////////
// Embedding Layer
///////////////////////////////////////////////////////////////////////////////

void fwd_emb(std::vector<float> &ma, std::vector<float> &mu_w,
             std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
             std::vector<size_t> &emb_sizes, int num_cat, int batch_size,
             std::vector<float> &mu_z, std::vector<float> &var_z)
/**/
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_cat; j++) {
            int cat_idx = ma[j + i * batch_size];
            int emb_size = emb_sizes[j];
            for (int k = 0; k < emb_size; k++) {
                mu_z[k] = mu_w[cat_idx * emb_size + k];
                var_z[k] = var_w[cat_idx * emb_size + k];
            }
        }
    }
}

void bwd_emb(std::vector<float> &ma, std::vector<float> &var_w,
             std::vector<float> &delta_mu, std::vector<float> &delta_var,
             std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
             int num_cat, int batch_size, std::vector<float> &delta_mu_w,
             std::vector<float> &delta_var_w)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_cat; j++) {
            int cat_idx = ma[j + i * batch_size];
            int emb_size = emb_sizes[j];
            for (int k = 0; k < emb_size; k++) {
                delta_mu_w[cat_idx * emb_size + k] =
                    delta_mu[k] * var_w[cat_idx * emb_size + k];
                delta_var_w[cat_idx * emb_size + k] =
                    delta_var[k] * var_w[cat_idx * emb_size + k];
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

Embedding::Embedding(const std::vector<size_t> &cat_sizes,
                     const std::vector<size_t> &emb_sizes, float scale,
                     int device_idx)
    : cat_sizes(cat_sizes), emb_sizes(emb_sizes), scale(scale) {
    this->device_idx = device_idx;
    this->num_cat = cat_sizes.size();

    this->calculate_sizes();

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
    for (size_t i = 0; i < this->cat_sizes.size(); ++i) {
        if (i > 0) info += ",";
        info += std::to_string(this->cat_sizes[i]) + "->" +
                std::to_string(this->emb_sizes[i]);
    }
    info += ")";
    return info;
}

std::string Embedding::get_layer_name() const { return "Embedding"; }

LayerType Embedding::get_layer_type() const { return LayerType::Embedding; }

void Embedding::calculate_sizes() {
    this->input_size = this->num_cat;
    this->num_weights = 0;
    this->num_biases = 0;

    for (int i = 0; i < this->num_cat; i++) {
        this->num_weights += this->cat_sizes[i] * this->emb_sizes[i];
    }

    this->output_size = 0;
    for (int i = 0; i < this->num_cat; i++) {
        this->output_size += this->emb_sizes[i];
    }
}

void Embedding::init_weight_bias() {
    auto weights = initialize_embedding_values(this->cat_sizes, this->emb_sizes,
                                               this->num_cat, this->scale);
    this->mu_w = std::get<0>(weights);
    this->var_w = std::get<1>(weights);
}

void Embedding::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states) {
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    fwd_emb(input_states.mu_a, this->mu_w, this->var_w, this->cat_sizes,
            this->emb_sizes, this->num_cat, batch_size, output_states.mu_a,
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
                this->cat_sizes, this->emb_sizes, this->num_cat, batch_size,
                this->delta_mu_w, this->delta_var_w);
    }
}

// #ifdef USE_CUDA
// std::unique_ptr<BaseLayer> Embedding::to_cuda(int device_idx) {
//     this->device = "cuda";
//     this->device_idx = device_idx;
//     auto cuda_layer = std::make_unique<EmbeddingCuda>(
//         this->cat_sizes, this->emb_sizes, this->num_bags, this->bag_sizes,
//         this->scale, this->device_idx);

//     auto base_cuda = dynamic_cast<BaseLayerCuda *>(cuda_layer.get());
//     base_cuda->copy_params_from(*this);

//     return cuda_layer;
// }
// #endif
