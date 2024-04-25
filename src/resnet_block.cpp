
#include "../include/resnet_block.h"

#ifdef USE_CUDA
#include "../include/resnet_block_cuda.cuh"
#endif

void add_shortcut_mean_var(const std::vector<float> &mu_s,
                           const std::vector<float> &var_s, int num_states,
                           std::vector<float> &mu_a, std::vector<float> &var_a)
/*
 */
{
    for (int i = 0; i < num_states; i++) {
        mu_a[i] += mu_s[i];
        var_a[i] += var_s[i];
    }
}

ResNetBlock::~ResNetBlock() {}

std::string ResNetBlock::get_layer_info() const
/*
 */
{
    return "ResNetBlock(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string ResNetBlock::get_layer_name() const
/*
 */
{
    return "ResNetBlock";
}

LayerType ResNetBlock::get_layer_type() const
/*
 */
{
    return LayerType::ResNetBlock;
}

int ResNetBlock::get_max_num_states()
/**/
{
    auto max_main_block = this->main_block->get_max_num_states();
    int max_shortcut = 0;
    if (this->shortcut != nullptr) {
        max_shortcut = this->shortcut->get_max_num_states();
    }

    return std::max(max_main_block, max_shortcut);
}

void ResNetBlock::compute_input_output_size(const InitArgs &args) {
    this->main_block->compute_input_output_size(args);
    if (this->shortcut != nullptr) {
        this->shortcut->compute_input_output_size(args);
    }
}

void ResNetBlock::init_shortcut_state()
/*
 */
{
    this->shortcut_output_z = std::make_shared<BaseHiddenStates>(
        this->shortcut->get_max_num_states(), this->_batch_size);
}

void ResNetBlock::init_shortcut_delta_state()
/*
 */
{
    this->shortcut_output_delta_z = std::make_shared<BaseDeltaStates>(
        this->shortcut->get_max_num_states(), this->_batch_size);
}

void ResNetBlock::init_weight_bias()
/*
 */
{
    this->main_block->init_weight_bias();
    if (this->shortcut != nullptr) {
        this->shortcut->init_weight_bias();
    }
}

void ResNetBlock::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/**/

{
    int batch_size = input_states.block_size;

    // Main block
    if (batch_size > this->_batch_size && this->shortcut != nullptr) {
        this->_batch_size = batch_size;
        this->init_shortcut_state();
        if (this->training) {
            this->init_shortcut_delta_state();
        }
    }
    this->main_block->forward(input_states, output_states, temp_states);
    int num_states = output_states.block_size * output_states.actual_size;

    // Shortcut
    if (this->shortcut != nullptr) {
        this->shortcut->forward(input_states, *this->shortcut_output_z,
                                temp_states);

        add_shortcut_mean_var(shortcut_output_z->mu_a, shortcut_output_z->var_a,
                              num_states, output_states.mu_a,
                              output_states.var_a);

    } else {
        add_shortcut_mean_var(input_states.mu_a, input_states.var_a, num_states,
                              output_states.mu_a, output_states.var_a);
    }
}

void ResNetBlock::backward(BaseDeltaStates &input_delta_states,
                           BaseDeltaStates &output_delta_states,
                           BaseTempStates &temp_states, bool state_update)
/**/
{
    this->main_block->backward(input_delta_states, output_delta_states,
                               temp_states, state_update);

    int num_states =
        output_delta_states.block_size * output_delta_states.actual_size;

    if (this->shortcut != nullptr) {
        this->shortcut->backward(input_delta_states,
                                 *this->shortcut_output_delta_z, temp_states,
                                 state_update);
        add_shortcut_mean_var(this->shortcut_output_delta_z->delta_mu,
                              this->shortcut_output_delta_z->delta_var,
                              num_states, output_delta_states.delta_mu,
                              output_delta_states.delta_var);

    } else {
        add_shortcut_mean_var(input_delta_states.delta_mu,
                              input_delta_states.delta_var, num_states,
                              output_delta_states.delta_mu,
                              output_delta_states.delta_var);
    }
}

void ResNetBlock::update_weights()
/*
 */
{
    this->main_block->update_weights();
    if (this->shortcut != nullptr) {
        this->shortcut->update_weights();
    }
}

void ResNetBlock::update_biases()
/*
 */
{
    this->main_block->update_biases();
    if (this->shortcut != nullptr) {
        this->shortcut->update_biases();
    }
}

void ResNetBlock::save(std::ofstream &file)
/*
 */
{
    this->main_block->save(file);
    if (this->shortcut != nullptr) {
        this->shortcut->save(file);
    }
}

void ResNetBlock::load(std::ifstream &file)
/*
 */
{
    this->main_block->load(file);
    if (this->shortcut != nullptr) {
        this->shortcut->load(file);
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> ResNetBlock::to_cuda() {
    this->device = "cuda";
    return std::make_unique<ResNetBlockCuda>(std::move(*this->main_block),
                                             std::move(*this->shortcut));
}
#endif
