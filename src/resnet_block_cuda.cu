#include <cuda.h>

#include "../include/resnet_block.h"
#include "../include/resnet_block_cuda.cuh"

__global__ void add_shortcut_mean_var_cuda(float const *mu_s,
                                           float const *var_s, int num_states,
                                           float *mu_a, float *var_a)
/**/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        mu_a[col] += mu_s[col];
        var_a[col] += mu_s[col];
    }
}

ResNetBlockCuda::ResNetBlockCuda(std::shared_ptr<LayerBlock> main_block_layer,
                                 std::shared_ptr<BaseLayer> shortcut_layer)
    : main_block(std::move(main_block_layer)),
      shortcut(std::move(shortcut_layer))
/**/
{}
ResNetBlockCuda::~ResNetBlockCuda() {}

std::string ResNetBlockCuda::get_layer_info() const
/*
 */
{
    return "ResNetBlockCuda(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string ResNetBlockCuda::get_layer_name() const
/*
 */
{
    return "ResNetBlockCuda";
}

LayerType ResNetBlockCuda::get_layer_type() const
/*
 */
{
    return LayerType::ResNetBlock;
}

int ResNetBlockCuda::get_max_num_states()
/**/
{
    auto max_main_block = this->main_block->get_max_num_states();
    int max_shortcut = 0;
    if (this->shortcut != nullptr) {
        max_shortcut = this->shortcut->get_max_num_states();
    }

    return std::max(max_main_block, max_shortcut);
}

void ResNetBlockCuda::compute_input_output_size(const InitArgs &args) {
    this->main_block->compute_input_output_size(args);
    if (this->shortcut != nullptr) {
        this->shortcut->compute_input_output_size(args);
    }
}

void ResNetBlockCuda::init_shortcut_state()
/*
 */
{
    this->shortcut_output_z = std::make_shared<HiddenStateCuda>(
        this->shortcut->get_max_num_states(), this->_batch_size);
}

void ResNetBlockCuda::init_shortcut_delta_state()
/*
 */
{
    this->shortcut_output_delta_z = std::make_shared<BaseDeltaStates>(
        this->shortcut->get_max_num_states(), this->_batch_size);
}

void ResNetBlockCuda::init_weight_bias()
/*
 */
{
    this->main_block->init_weight_bias();
    if (this->shortcut != nullptr) {
        this->shortcut->init_weight_bias();
    }
}

void ResNetBlockCuda::forward(BaseHiddenStates &input_states,
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

        HiddenStateCuda *cu_shortcut_output_z =
            dynamic_cast<HiddenStateCuda *>(this->shortcut_output_z.get());

        HiddenStateCuda *cu_output_states =
            dynamic_cast<HiddenStateCuda *>(&output_states);

        unsigned int grid_size =
            (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_output_z->d_mu_a, cu_shortcut_output_z->d_var_a,
            num_states, cu_output_states->d_mu_a, cu_output_states->d_var_a);

    } else {
        HiddenStateCuda *cu_shortcut_output_z =
            dynamic_cast<HiddenStateCuda *>(&input_states);

        HiddenStateCuda *cu_output_states =
            dynamic_cast<HiddenStateCuda *>(&output_states);

        unsigned int grid_size =
            (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_output_z->d_mu_a, cu_shortcut_output_z->d_var_a,
            num_states, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }
}

void ResNetBlockCuda::backward(BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_delta_states,
                               BaseTempStates &temp_states, bool state_update)
/**/
{
    this->main_block->backward(input_delta_states, output_delta_states,
                               temp_states, state_update);

    int num_states =
        output_delta_states.block_size * output_delta_states.actual_size;
    unsigned int grid_size =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    if (this->shortcut != nullptr) {
        this->shortcut->backward(input_delta_states,
                                 *this->shortcut_output_delta_z, temp_states,
                                 state_update);

        DeltaStateCuda *cu_shortcut_output_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->shortcut_output_delta_z.get());

        DeltaStateCuda *cu_output_delta_states =
            dynamic_cast<DeltaStateCuda *>(&output_delta_states);

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_output_delta_z->d_delta_mu,
            cu_shortcut_output_delta_z->d_delta_var, num_states,
            cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    } else {
        DeltaStateCuda *cu_shortcut_output_delta_z =
            dynamic_cast<DeltaStateCuda *>(&input_delta_states);

        DeltaStateCuda *cu_output_delta_states =
            dynamic_cast<DeltaStateCuda *>(&output_delta_states);

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_output_delta_z->d_delta_mu,
            cu_shortcut_output_delta_z->d_delta_var, num_states,
            cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    }
}

void ResNetBlockCuda::update_weights()
/*
 */
{
    this->main_block->update_weights();
    if (this->shortcut != nullptr) {
        this->shortcut->update_weights();
    }
}

void ResNetBlockCuda::update_biases()
/*
 */
{
    this->main_block->update_biases();
    if (this->shortcut != nullptr) {
        this->shortcut->update_biases();
    }
}

void ResNetBlockCuda::save(std::ofstream &file)
/*
 */
{
    this->main_block->save(file);
    if (this->shortcut != nullptr) {
        this->shortcut->save(file);
    }
}

void ResNetBlockCuda::load(std::ifstream &file)
/*
 */
{
    this->main_block->load(file);
    if (this->shortcut != nullptr) {
        this->shortcut->load(file);
    }
}

std::unique_ptr<BaseLayer> ResNetBlockCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer =
        std::make_unique<ResNetBlock>(this->main_block, this->shortcut);

    return host_layer;
}
