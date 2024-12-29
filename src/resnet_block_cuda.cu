#include <cuda.h>

#include "../include/custom_logger.h"
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

__global__ void add_shortcut_delta_cuda(float const *mu_s, float const *var_s,
                                        float const *jcb_s, int num_states,
                                        float *mu_a, float *var_a)
/**/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        mu_a[col] += mu_s[col] * jcb_s[col];
        var_a[col] += var_s[col] * jcb_s[col] * jcb_s[col];
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

void ResNetBlockCuda::train()
/*
 */
{
    this->main_block->train();
    if (this->shortcut != nullptr) {
        this->shortcut->train();
    }
}

void ResNetBlockCuda::eval()
/*
 */
{
    this->main_block->eval();
    if (this->shortcut != nullptr) {
        this->shortcut->eval();
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
            LOG(LogLevel::ERROR, "Set cuda threads.");
        }
    }

    if (this->shortcut != nullptr) {
        BaseLayerCuda *cu_shortcut =
            dynamic_cast<BaseLayerCuda *>(this->shortcut.get());
        if (cu_shortcut) {
            cu_shortcut->set_cuda_threads(num);
        } else {
            LOG(LogLevel::ERROR, "Set cuda threads.");
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
    // Store jacobian matrix for backward pass
    if (this->training) {
        HiddenStateCuda *cu_input_states =
            dynamic_cast<HiddenStateCuda *>(&input_states);
        BackwardStateCuda *cu_bwd_states =
            dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

        int act_size = cu_input_states->actual_size * batch_size;
        if (cu_bwd_states->size != act_size) {
            cu_bwd_states->size = act_size;
            cu_bwd_states->allocate_memory();
        }
        cudaMemcpy(cu_bwd_states->d_mu_a, cu_input_states->d_mu_a,
                   act_size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(cu_bwd_states->d_jcb, cu_input_states->d_jcb,
                   act_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Make a copy of input states for residual connection
    this->input_z->copy_from(input_states, this->input_size * batch_size);

    this->main_block->forward(input_states, output_states, temp_states);

    int num_states = output_states.block_size * this->output_size;
    constexpr unsigned int THREADS = 256;
    unsigned int grid_size = (num_states + THREADS - 1) / THREADS;
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    // Shortcut
    if (this->shortcut != nullptr)  // use a transformation function to match
                                    // the output size of the main block
    {
        this->shortcut->forward(*this->input_z, *this->shortcut_output_z,
                                temp_states);

        HiddenStateCuda *cu_shortcut_output_z =
            dynamic_cast<HiddenStateCuda *>(this->shortcut_output_z.get());

        add_shortcut_mean_var_cuda<<<grid_size, THREADS>>>(
            cu_shortcut_output_z->d_mu_a, cu_shortcut_output_z->d_var_a,
            num_states, cu_output_states->d_mu_a, cu_output_states->d_var_a);

    } else {
        HiddenStateCuda *cu_shortcut_output_z =
            dynamic_cast<HiddenStateCuda *>(this->input_z.get());

        add_shortcut_mean_var_cuda<<<grid_size, THREADS>>>(
            cu_shortcut_output_z->d_mu_a, cu_shortcut_output_z->d_var_a,
            num_states, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    // Fill jacobian matrix for output with ones
    if (this->training) {
        HiddenStateCuda *cu_input_states =
            dynamic_cast<HiddenStateCuda *>(this->input_z.get());

        int out_size = this->output_size * batch_size;
        unsigned int out_blocks = (out_size + THREADS - 1) / THREADS;
        fill_output_states_on_device<<<out_blocks, THREADS>>>(
            out_size, cu_output_states->d_jcb);
    }
}

void ResNetBlockCuda::backward(BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_delta_states,
                               BaseTempStates &temp_states, bool state_update)
/**/
{
    int batch_size = input_delta_states.block_size;
    // Make a copy of delta input used later for residual connection
    this->input_delta_z->copy_from(input_delta_states,
                                   this->output_size * batch_size);

    this->main_block->backward(input_delta_states, output_delta_states,
                               temp_states, state_update);

    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int num_states = batch_size * this->input_size;
    constexpr unsigned int THREADS = 256;
    unsigned int grid_size = (num_states + THREADS - 1) / THREADS;

    if (this->shortcut != nullptr) {
        this->shortcut->backward(*this->input_delta_z,
                                 *this->shortcut_output_delta_z, temp_states,
                                 state_update);

        DeltaStateCuda *cu_shortcut_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->shortcut_output_delta_z.get());

        add_shortcut_mean_var_cuda<<<grid_size, THREADS>>>(
            cu_shortcut_delta_z->d_delta_mu, cu_shortcut_delta_z->d_delta_var,
            num_states, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);

    } else {
        DeltaStateCuda *cu_input_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->input_delta_z.get());

        BackwardStateCuda *cu_bwd_states =
            dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

        add_shortcut_delta_cuda<<<grid_size, THREADS>>>(
            cu_input_delta_z->d_delta_mu, cu_input_delta_z->d_delta_var,
            cu_bwd_states->d_jcb, num_states,
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

ParameterMap ResNetBlockCuda::get_parameters_as_map(std::string suffix) {
    std::string main_suffix = "main." + suffix;
    ParameterMap params = this->main_block->get_parameters_as_map(main_suffix);
    if (this->shortcut != nullptr) {
        std::string shortcut_suffix = "shortcut." + suffix;
        auto shortcut_params =
            this->shortcut->get_parameters_as_map(shortcut_suffix);
        params.insert(shortcut_params.begin(), shortcut_params.end());
    }
    return params;
}

void ResNetBlockCuda::load_parameters_from_map(const ParameterMap &param_map,
                                               const std::string &suffix) {
    std::string main_suffix = "main." + suffix;
    this->main_block->load_parameters_from_map(param_map, main_suffix);
    if (this->shortcut != nullptr) {
        std::string shortcut_suffix = "shortcut." + suffix;
        this->shortcut->load_parameters_from_map(param_map, shortcut_suffix);
    }
}

std::vector<ParameterTuple> ResNetBlockCuda::parameters() {
    std::vector<ParameterTuple> params = this->main_block->parameters();
    if (this->shortcut != nullptr) {
        auto shortcut_params = this->shortcut->parameters();
        params.insert(params.end(), shortcut_params.begin(),
                      shortcut_params.end());
    }
    return params;
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

// DEBUG
std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
ResNetBlockCuda::get_norm_mean_var() {
    std::vector<std::vector<float>> mu_ras, var_ras, mu_norms, var_norms;
    std::tie(mu_ras, var_ras, mu_norms, var_norms) =
        this->main_block->get_norm_mean_var();

    if (this->shortcut != nullptr) {
        std::vector<std::vector<float>> mu_ra, var_ra, mu_norm, var_norm;
        std::tie(mu_ra, var_ra, mu_norm, var_norm) =
            this->shortcut->get_norm_mean_var();
        for (size_t i = 0; i < mu_ra.size(); i++) {
            mu_ras.push_back(mu_ra[i]);
            var_ras.push_back(var_ra[i]);
            mu_norms.push_back(mu_norm[i]);
            var_norms.push_back(var_norm[i]);
        }
    }

    return {mu_ras, var_ras, mu_norms, var_norms};
}
