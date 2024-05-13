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
        var_a[col] += var_s[col];
    }
}

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

std::string ResNetBlockCuda::get_device()
/*
 */
{
    auto main_block_device = this->main_block->get_device();
    if (main_block_device != this->device) {
        return main_block_device;
    }
    if (this->shortcut != nullptr) {
        auto shortcut_device = this->shortcut->get_device();
        if (shortcut_device != this->device) {
            return shortcut_device;
        }
    }
    return this->device;
}

void ResNetBlockCuda::compute_input_output_size(const InitArgs &args)

{
    this->in_channels = args.depth;
    this->in_height = args.height;
    this->in_width = args.width;

    this->main_block->compute_input_output_size(args);
    if (this->shortcut != nullptr) {
        this->shortcut->compute_input_output_size(args);
    }

    this->out_channels = this->main_block->out_channels;
    this->out_height = this->main_block->out_height;
    this->out_width = this->main_block->out_width;

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void ResNetBlockCuda::init_shortcut_state()
/*
 */
{
    int max_num_states = this->shortcut->get_max_num_states();
    int size = max_num_states * this->_batch_size;
    this->shortcut_output_z =
        std::make_shared<HiddenStateCuda>(size, this->_batch_size);
}

void ResNetBlockCuda::init_shortcut_delta_state()
/*
 */
{
    int max_num_states = this->shortcut->get_max_num_states();
    int size = max_num_states * this->_batch_size;
    this->shortcut_output_delta_z =
        std::make_shared<DeltaStateCuda>(size, this->_batch_size);
}

void ResNetBlockCuda::init_input_buffer()
/*
 */
{
    int max_num_states = this->input_size;
    if (this->shortcut != nullptr) {
        max_num_states = this->shortcut->get_max_num_states();
    }
    int size = max_num_states * this->_batch_size;
    this->input_z = std::make_shared<HiddenStateCuda>(size, this->_batch_size);
    this->input_delta_z =
        std::make_shared<DeltaStateCuda>(size, this->_batch_size);
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

void ResNetBlockCuda::set_threads(int num)
/*
 */
{
    this->main_block->set_threads(num);
    if (this->shortcut != nullptr) {
        this->shortcut->set_threads(num);
    }
}

void ResNetBlockCuda::set_cuda_threads(int num)
/*
 */
{
    // TODO: Any better way?
    BaseLayerCuda *cu_main_block =
        dynamic_cast<BaseLayerCuda *>(this->main_block.get());
    if (cu_main_block) {
        cu_main_block->set_cuda_threads(num);
    } else {
        LayerBlock *layer_block =
            dynamic_cast<LayerBlock *>(this->main_block.get());
        if (layer_block) {
            layer_block->set_cuda_threads(num);
        } else {
            throw std::invalid_argument(
                "Error in file: " + std::string(__FILE__) + " at line: " +
                std::to_string(__LINE__) + ". Set cuda threads.");
        }
    }

    if (this->shortcut != nullptr) {
        BaseLayerCuda *cu_shortcut =
            dynamic_cast<BaseLayerCuda *>(this->shortcut.get());
        if (cu_shortcut) {
            cu_shortcut->set_cuda_threads(num);
        } else {
            throw std::invalid_argument(
                "Error in file: " + std::string(__FILE__) + " at line: " +
                std::to_string(__LINE__) + ". Set cuda threads.");
        }
    }
}

void ResNetBlockCuda::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/**/

{
    int batch_size = input_states.block_size;

    // Main block
    if (batch_size != this->_batch_size) {
        this->_batch_size = batch_size;
        this->init_input_buffer();
        if (this->shortcut != nullptr) {
            this->init_shortcut_state();
            if (this->training) {
                this->init_shortcut_delta_state();
            }
        }
    }

    // Make a copy of input states for residual connection
    this->input_z->copy_from(input_states, this->input_size * batch_size);

    this->main_block->forward(input_states, output_states, temp_states);

    int num_states = output_states.block_size * this->output_size;
    unsigned int grid_size =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // Shortcut
    if (this->shortcut != nullptr) {
        this->shortcut->forward(*this->input_z, *this->shortcut_output_z,
                                temp_states);

        HiddenStateCuda *cu_shortcut_output_z =
            dynamic_cast<HiddenStateCuda *>(this->shortcut_output_z.get());

        HiddenStateCuda *cu_output_states =
            dynamic_cast<HiddenStateCuda *>(&output_states);

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_output_z->d_mu_a, cu_shortcut_output_z->d_var_a,
            num_states, cu_output_states->d_mu_a, cu_output_states->d_var_a);

    } else {
        HiddenStateCuda *cu_shortcut_output_z =
            dynamic_cast<HiddenStateCuda *>(this->input_z.get());

        HiddenStateCuda *cu_output_states =
            dynamic_cast<HiddenStateCuda *>(&output_states);

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_output_z->d_mu_a, cu_shortcut_output_z->d_var_a,
            num_states, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;
}

void ResNetBlockCuda::backward(BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_delta_states,
                               BaseTempStates &temp_states, bool state_update)
/**/
{
    // Make a copy of delta input used later for residual connection
    this->input_delta_z->copy_from(
        input_delta_states, this->input_size * input_delta_states.block_size);

    this->main_block->backward(input_delta_states, output_delta_states,
                               temp_states, state_update);

    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int num_states = output_delta_states.block_size * this->input_size;
    unsigned int grid_size =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    if (this->shortcut != nullptr) {
        this->shortcut->backward(*this->input_delta_z,
                                 *this->shortcut_output_delta_z, temp_states,
                                 state_update);

        DeltaStateCuda *cu_shortcut_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->shortcut_output_delta_z.get());

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_shortcut_delta_z->d_delta_mu, cu_shortcut_delta_z->d_delta_var,
            num_states, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);

    } else {
        DeltaStateCuda *cu_input_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->input_delta_z.get());

        add_shortcut_mean_var_cuda<<<grid_size, this->num_cuda_threads>>>(
            cu_input_delta_z->d_delta_mu, cu_input_delta_z->d_delta_var,
            num_states, cu_output_delta_states->d_delta_mu,
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
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<ResNetBlock>(
        std::move(*this->main_block), std::move(*this->shortcut));

    return host_layer;
}
void ResNetBlockCuda::preinit_layer() {
    this->main_block->preinit_layer();

    if (this->shortcut != nullptr) {
        this->shortcut->preinit_layer();
    }
}
