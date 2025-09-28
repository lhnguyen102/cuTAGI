#include "../include/embedding_bag_cpu.h"

#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/embedding_cpu.h"

Offsets precompute_offsets(const std::vector<size_t> &num_bags,
                           const std::vector<size_t> &bag_sizes, int num_cat,
                           int batch_size) {
    Offsets offsets;
    offsets.batch_in_offsets.resize(batch_size);
    offsets.batch_out_offsets.resize(batch_size);
    offsets.cat_in_offsets.resize(num_cat);
    offsets.cat_out_offsets.resize(num_cat);

    // Precompute batch offsets
    int batch_size_in_offset = 0;
    int batch_size_out_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        offsets.batch_in_offsets[i] = batch_size_in_offset;
        offsets.batch_out_offsets[i] = batch_size_out_offset;

        int cat_in_offset = 0;
        int cat_out_offset = 0;

        for (int j = 0; j < num_cat; ++j) {
            size_t bag = num_bags[j];
            size_t bag_size = bag_sizes[j];
            offsets.cat_in_offsets[j] = cat_in_offset;
            offsets.cat_out_offsets[j] = cat_out_offset;

            cat_in_offset += bag_size * bag;
            cat_out_offset += bag;
        }

        batch_size_in_offset += cat_in_offset;
        batch_size_out_offset += cat_out_offset;
    }

    return offsets;
}

void fwd_bag_emb(std::vector<float> &mu_a, std::vector<float> &mu_w,
                 std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
                 std::vector<size_t> &emb_sizes, std::vector<size_t> &num_bags,
                 std::vector<size_t> &bag_sizes, int num_cat, int batch_size,
                 std::vector<float> &mu_z, std::vector<float> &var_z) {
    // Compute the offsets for each bags due to we store all data in a single
    // vector
    Offsets offsets =
        precompute_offsets(num_bags, bag_sizes, num_cat, batch_size);
    for (int i = 0; i < batch_size; i++) {
        int batch_size_in_offset = offsets.batch_in_offsets[i];
        int batch_size_out_offset = offsets.batch_out_offsets[i];
        for (int j = 0; j < num_cat; j++) {
            size_t bag = num_bags[j];
            size_t emb_size = emb_sizes[j];
            size_t bag_size = bag_sizes[j];
            int cat_in_offset = offsets.cat_in_offsets[j];
            int cat_out_offset = offsets.cat_out_offsets[j];
            for (int m = 0; m < bag; m++) {
                float sum_mu = 0.0f;
                float sum_var = 0.0f;
                for (int n = 0; n < bag_size; n++) {
                    // Convert categorical index in each bag to integer. TODO:
                    // need to avoid this conversion for computing performance
                    int cat_idx = mu_a[n + m * bag_size + cat_in_offset +
                                       batch_size_in_offset];

                    int offset = cat_idx * emb_size;
                    // Sum over all embedding values for each bag
                    for (int k = 0; k < emb_size; k++) {
                        sum_mu += mu_w[offset + k];
                        sum_var += var_w[offset + k];
                    }
                }

                // Average the embedding values. Output size (batch_size,
                // num_cat, bag)
                int z_idx = m + cat_out_offset + batch_size_out_offset;
                mu_z[z_idx] = sum_mu / bag_size;
                var_z[z_idx] = sum_var / bag_size;
            }
        }
    }
}

void bwd_bag_emb(std::vector<float> &mu_a, std::vector<float> &var_w,
                 std::vector<float> &delta_mu, std::vector<float> &delta_var,
                 std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
                 std::vector<size_t> &num_bags, std::vector<size_t> &bag_sizes,
                 int num_cat, int batch_size, std::vector<float> &delta_mu_w,
                 std::vector<float> &delta_var_w) {
    // Compute the offsets for each bags due to we store all data in a single
    // vector
    Offsets offsets =
        precompute_offsets(num_bags, bag_sizes, num_cat, batch_size);
    for (int i = 0; i < batch_size; i++) {
        int batch_size_in_offset = offsets.batch_in_offsets[i];
        int batch_size_out_offset = offsets.batch_out_offsets[i];
        for (int j = 0; j < num_cat; j++) {
            size_t bag = num_bags[j];
            size_t emb_size = emb_sizes[j];
            size_t bag_size = bag_sizes[j];
            int cat_in_offset = offsets.cat_in_offsets[j];
            int cat_out_offset = offsets.cat_out_offsets[j];

            for (int m = 0; m < bag; m++) {
                for (int n = 0; n < bag_size; n++) {
                    // Convert categorical index in each bag to integer. TODO:
                    // need to avoid this conversion for computing performance
                    int cat_idx = mu_a[n + m * bag_size + cat_in_offset +
                                       batch_size_in_offset];

                    // Index for the updating quantities of the output layer
                    int ino_idx = m + cat_out_offset + batch_size_out_offset;

                    // Calculate the updating quanties for embeddings inside
                    // each bag
                    for (int k = 0; k < emb_size; k++) {
                        // Index for embedding
                        int w_idx = cat_idx * emb_size + k;

                        // Updating quantities for embedding. TODO Need to set
                        // all delta values to zero
                        delta_mu_w[w_idx] += delta_mu[ino_idx] * var_w[w_idx];
                        delta_var_w[w_idx] += delta_var[ino_idx] * var_w[w_idx];
                    }
                }
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// EmbeddingBag Layer Class
///////////////////////////////////////////////////////////////////////////////

EmbeddingBag::EmbeddingBag(const std::vector<size_t> &cat_sizes,
                           const std::vector<size_t> &emb_sizes,
                           const std::vector<size_t> &num_bags,
                           const std::vector<size_t> &bag_sizes, float scale,
                           int device_idx)
    : cat_sizes(cat_sizes),
      emb_sizes(emb_sizes),
      num_bags(num_bags),
      bag_sizes(bag_sizes),
      scale(scale) {
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

EmbeddingBag::~EmbeddingBag() {}

std::string EmbeddingBag::get_layer_info() const {
    std::string info = "EmbeddingBag(";
    for (size_t i = 0; i < this->cat_sizes.size(); ++i) {
        if (i > 0) info += ",";
        info += std::to_string(this->cat_sizes[i]) + "->" +
                std::to_string(this->emb_sizes[i]) + "x" +
                std::to_string(this->num_bags[i]);
    }
    info += ")";
    return info;
}

std::string EmbeddingBag::get_layer_name() const { return "EmbeddingBag"; }

LayerType EmbeddingBag::get_layer_type() const {
    return LayerType::EmbeddingBag;
}

void EmbeddingBag::calculate_sizes() {
    this->input_size = this->num_cat;
    this->num_weights = 0;
    this->num_biases = 0;

    for (int i = 0; i < this->num_cat; i++) {
        this->num_weights += this->cat_sizes[i] * this->emb_sizes[i];
    }

    this->output_size = 0;
    for (int i = 0; i < this->num_cat; i++) {
        this->output_size += this->num_bags[i];
    }
}

void EmbeddingBag::init_weight_bias() {
    auto weights = initialize_embedding_values(this->cat_sizes, this->emb_sizes,
                                               this->num_cat, this->scale);
    this->mu_w = std::get<0>(weights);
    this->var_w = std::get<1>(weights);
}

void EmbeddingBag::forward(BaseHiddenStates &input_states,
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

    fwd_bag_emb(input_states.mu_a, this->mu_w, this->var_w, this->cat_sizes,
                this->emb_sizes, this->num_bags, this->bag_sizes, this->num_cat,
                batch_size, output_states.mu_a, output_states.var_a);

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void EmbeddingBag::backward(BaseDeltaStates &input_delta_states,
                            BaseDeltaStates &output_delta_states,
                            BaseTempStates &temp_states, bool state_udapte) {
    int batch_size = input_delta_states.block_size;

    if (this->param_update) {
        bwd_bag_emb(this->bwd_states->mu_a, this->var_w,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->cat_sizes, this->emb_sizes, this->num_bags,
                    this->bag_sizes, this->num_cat, batch_size,
                    this->delta_mu_w, this->delta_var_w);
    }
}
