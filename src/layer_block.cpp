#include "../include/layer_block.h"

#include "../include/attention.h"
#include "../include/custom_logger.h"

#ifdef USE_CUDA
#include "../include/attention_cuda.cuh"
#include "../include/base_layer_cuda.cuh"
#endif

LayerBlock::~LayerBlock() {}

LayerBlock::LayerBlock() {}

void LayerBlock::add_layers()
/*
 */
{
    this->input_size = this->layers.front()->input_size;

    auto layer_type = this->layers.back()->get_layer_type();
    int num_layers = this->layers.size();
    int i = num_layers - 2;
    while (layer_type == LayerType::Activation && i >= 0) {
        this->output_size = this->layers[i]->output_size;
        layer_type = this->layers[i]->get_layer_type();
        i--;
    }
}

void LayerBlock::add_layer(std::shared_ptr<BaseLayer> layer)
/*
NOTE: The output buffer size is determinated based on the output size for each
layer assuming that batch size = 1. If the batch size in the forward pass > 1,
it will be corrected at the first run in the forward pass.
 */
{
    // Stack layer
    if (this->device.compare("cpu") == 0) {
        this->layers.push_back(std::move(layer));
    } else if (this->device.compare("cuda") == 0) {
        this->layers.push_back(std::move(layer->to_cuda(this->device_idx)));
    } else {
        LOG(LogLevel::ERROR, "Invalid device: [" + this->device + "]");
    }
}

void LayerBlock::switch_to_cuda() {
    for (size_t i = 0; i < this->layers.size(); ++i) {
        auto cuda_layer = layers[i]->to_cuda(this->device_idx);
        layers[i] = std::move(cuda_layer);
    }
}

std::string LayerBlock::get_layer_info() const
/*
 */
{
    return "LayerBlock(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LayerBlock::get_layer_name() const
/*
 */
{
    return "LayerBlock";
}

LayerType LayerBlock::get_layer_type() const
/*
 */
{
    return LayerType::LayerBlock;
}

int LayerBlock::get_max_num_states()
/**/
{
    int max_size = 0;
    for (const auto &layer : this->layers) {
        int layer_max_size = layer->get_max_num_states();
        max_size = std::max(layer_max_size, max_size);
    }
    return max_size;
}

std::string LayerBlock::get_device()
/*
 */
{
    std::string block_device =
        this->device + ":" + std::to_string(this->device_idx);
    for (auto &layer : this->layers) {
        auto layer_device = layer->get_device();
        if (layer_device != block_device) {
            LOG(LogLevel::ERROR, "Layer device [" + layer_device +
                                     "] does not match block device [" +
                                     block_device + "]");
        }
    }
    return block_device;
}

void LayerBlock::init_weight_bias()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->init_weight_bias();
    }
}

void LayerBlock::set_threads(int num)
/*
 */
{
    for (auto &layer : this->layers) {
        layer->set_threads(num);
    }
}

void LayerBlock::train()
/*
 */
{
    for (auto &layer : this->layers) {
        layer->train();
    }
}

void LayerBlock::eval()
/*
 */
{
    for (auto &layer : this->layers) {
        layer->eval();
    }
}

#ifdef USE_CUDA
void LayerBlock::set_cuda_threads(int num)
/*
 */
{
    for (auto &layer : this->layers) {
        BaseLayerCuda *cu_layer = dynamic_cast<BaseLayerCuda *>(layer.get());
        cu_layer->set_cuda_threads(num);
    }
}
#endif

void LayerBlock::forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states)
/*
 */
{
    BaseHiddenStates *casted_input_states =
        dynamic_cast<BaseHiddenStates *>(&input_states);
    BaseHiddenStates *casted_output_states =
        dynamic_cast<BaseHiddenStates *>(&output_states);

    int batch_size = input_states.block_size;
    int seq_len = input_states.seq_len;
    int num_layers = this->layers.size();

    for (int i = 0; i < num_layers; ++i) {
        auto *current_layer = this->layers[i].get();
        current_layer->forward(*casted_input_states, *casted_output_states,
                               temp_states);
        std::swap(casted_input_states, casted_output_states);
    }

    // Ensure output states contains the output of the last layer in the block
    if (num_layers % 2 == 0) {
        output_states.swap(input_states);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.seq_len = seq_len;
    output_states.actual_size = this->output_size;
}

void LayerBlock::backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states, bool state_update)
/*
 */
{
    for (auto layer = this->layers.rbegin(); layer != this->layers.rend() - 1;
         ++layer) {
        auto *current_layer = layer->get();
        current_layer->backward(input_delta_states, output_delta_states,
                                temp_states);
        if (current_layer->get_layer_type() != LayerType::Activation) {
            input_delta_states.swap(output_delta_states);
        }
    }

    if (state_update) {
        this->layers[0]->backward(input_delta_states, output_delta_states,
                                  temp_states, state_update);
    }

    if (this->layers[0]->get_layer_type() == LayerType::Activation ||
        !state_update) {
        output_delta_states.swap(input_delta_states);
    }
    output_delta_states.seq_len = input_delta_states.seq_len;
}

void LayerBlock::update_weights()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->update_weights();
    }
}

void LayerBlock::set_var_decay(float tau)
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->set_var_decay(tau);
    }
}

void LayerBlock::apply_var_decay()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->apply_var_decay();
    }
}

void LayerBlock::update_biases()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->update_biases();
    }
}

void LayerBlock::compute_input_output_size(const InitArgs &args)
/*
 */
{
    this->in_channels = args.depth;
    this->in_height = args.height;
    this->in_width = args.width;

    InitArgs tmp = InitArgs(args.width, args.height, args.depth);

    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i]->compute_input_output_size(tmp);

        tmp.width = this->layers[i]->out_width;
        tmp.height = this->layers[i]->out_height;
        tmp.depth = this->layers[i]->out_channels;
    }

    this->out_channels = this->layers.back()->out_channels;
    this->out_height = this->layers.back()->out_height;
    this->out_width = this->layers.back()->out_width;

    int spatial_in = this->in_width * this->in_height * this->in_channels;
    int spatial_out = this->out_width * this->out_height * this->out_channels;
    this->input_size =
        spatial_in > 0 ? spatial_in : this->layers.front()->input_size;
    this->output_size =
        spatial_out > 0 ? spatial_out : this->layers.back()->output_size;
}

void LayerBlock::save(std::ofstream &file)
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->save(file);
    }
}

void LayerBlock::load(std::ifstream &file)
/*
 */
{
    for (auto &layer : this->layers) {
        layer->load(file);
    }
}

ParameterMap LayerBlock::get_parameters_as_map(std::string suffix) {
    ParameterMap params;
    for (size_t i = 0; i < this->layers.size(); i++) {
        if (layers[i]->get_layer_type() == LayerType::Activation ||
            layers[i]->get_layer_type() == LayerType::Pool2d) {
            continue;
        }
        std::string layer_id_suffix = suffix + "." + std::to_string(i);
        auto layer_params =
            this->layers[i]->get_parameters_as_map(layer_id_suffix);
        params.insert(layer_params.begin(), layer_params.end());
    }
    return params;
}

void LayerBlock::load_parameters_from_map(const ParameterMap &param_map,
                                          const std::string &suffix) {
    for (size_t i = 0; i < this->layers.size(); i++) {
        if (layers[i]->get_layer_type() == LayerType::Activation ||
            layers[i]->get_layer_type() == LayerType::Pool2d) {
            continue;
        }
        std::string layer_id_suffix = suffix + "." + std::to_string(i);
        this->layers[i]->load_parameters_from_map(param_map, layer_id_suffix);
    }
}

std::vector<ParameterTuple> LayerBlock::parameters() {
    std::vector<ParameterTuple> params;
    for (const auto &layer : this->layers) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LayerBlock::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    this->switch_to_cuda();
    return std::make_unique<LayerBlock>(std::move(*this));
}
#endif

void LayerBlock::preinit_layer() {
    for (auto &layer : this->layers) {
        layer->preinit_layer();
    }
}

std::vector<AttentionScores> LayerBlock::get_attention_scores() {
    std::vector<AttentionScores> result;
    for (auto &layer : this->layers) {
        BaseLayer *raw = layer.get();
        if (auto *l = dynamic_cast<MultiheadAttention *>(raw)) {
            result.push_back(l->get_attention_scores());
        } else if (auto *l = dynamic_cast<MultiheadAttentionV2 *>(raw)) {
            result.push_back(l->get_attention_scores());
        }
#ifdef USE_CUDA
        else if (auto *l = dynamic_cast<MultiheadAttentionCuda *>(raw)) {
            result.push_back(l->get_attention_scores());
        } else if (auto *l = dynamic_cast<MultiheadAttentionV2Cuda *>(raw)) {
            result.push_back(l->get_attention_scores());
        }
#endif
    }
    return result;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
LayerBlock::get_norm_mean_var() {
    std::vector<std::vector<float>> mu_ras, var_ras, mu_norms, var_norms;
    for (const auto &layer : this->layers) {
        std::vector<std::vector<float>> mu_ra, var_ra, mu_norm, var_norm;
        std::tie(mu_ra, var_ra, mu_norm, var_norm) = layer->get_norm_mean_var();
        for (size_t i = 0; i < mu_ra.size(); i++) {
            mu_ras.push_back(mu_ra[i]);
            var_ras.push_back(var_ra[i]);
            mu_norms.push_back(mu_norm[i]);
            var_norms.push_back(var_norm[i]);
        }
    }
    return std::make_tuple(mu_ras, var_ras, mu_norms, var_norms);
}
