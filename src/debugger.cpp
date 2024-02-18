///////////////////////////////////////////////////////////////////////////////
// File:         debugger.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 14, 2024
// Updated:      February 15, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/debugger.h"

#include <memory>
#include <vector>

#include "../include/data_struct.h"
#ifdef USE_CUDA
#include "data_struct_cuda.cuh"
#endif

ModelDebugger::ModelDebugger(Sequential &test_model, Sequential &ref_model,
                             OutputUpdater &output_updater) {
    this->test_model = test_model;
    this->ref_model = ref_model;
    this->output_updater = output_updater;
}

ModelDebugger::~ModelDebugger() {}

void ModelDebugger::lazy_init(int batch_size, int z_buffer_size)
/*
 */
{
    if (test_model.device.compare("cpu") == 0) {
        test_output_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        test_input_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        test_temp_states =
            std::make_shared<BaseTempStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (test_model.device.compare("cuda") == 0) {
        test_output_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        test_input_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        test_temp_states =
            std::make_shared<TempStateCuda>(z_buffer_size, batch_size);
    }
#endif

    if (ref_model.device.compare("cpu") == 0) {
        ref_output_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        ref_input_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        ref_temp_states =
            std::make_shared<BaseTempStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        ref_output_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        ref_input_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        ref_temp_states =
            std::make_shared<TempStateCuda>(z_buffer_size, batch_size);
    }
#endif

    if (test_model.device.compare("cpu") == 0) {
        test_output_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
        test_input_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        test_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        test_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
#endif

    if (ref_model.device.compare("cpu") == 0) {
        ref_output_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
        ref_input_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        ref_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        ref_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
#endif
}

void ModelDebugger::debug_forward(const std::vector<float> &mu_x,
                                  const std::vector<float> &var_x)
/*
 */
{
    int batch_size = mu_x.size() / test_model.layers.front()->input_size;
    int z_buffer_size = batch_size * test_model.z_buffer_size;

    this->lazy_init(batch_size, z_buffer_size);

    // Merge input data to the input buffer
    this->test_input_z_buffer->set_input_x(mu_x, var_x, batch_size);
    this->ref_input_z_buffer->set_input_x(mu_x, var_x, batch_size);

    int num_layers = this->test_model.layers.size();

    for (int i = 0; i < num_layers; i++) {
        auto *test_current_layer = this->test_model.layers[i].get();
        auto *ref_current_layer = this->ref_model.layers[i].get();

        test_current_layer->forward(*this->test_input_z_buffer,
                                    *this->test_output_z_buffer,
                                    *this->test_temp_states);

        ref_current_layer->forward(*this->ref_input_z_buffer,
                                   *this->ref_output_z_buffer,
                                   *this->ref_temp_states);

        // Copy to host for gpu model
#ifdef USE_CUDA
        if (this->test_model.device.compare("cuda") == 0) {
            HiddenStateCuda *test_output_z_buffer_cu =
                dynamic_cast<HiddenStateCuda *>(
                    this->test_output_z_buffer.get());
            test_output_z_buffer_cu->to_host();
        }
        if (this->ref_model.device.compare("cuda") == 0) {
            HiddenStateCuda *ref_output_z_buffer_cu =
                dynamic_cast<HiddenStateCuda *>(
                    this->ref_output_z_buffer.get());
            ref_output_z_buffer_cu->to_host();
        }
#endif

        // Test here
        auto layer_name = test_current_layer->get_layer_name();
        for (int j = 0; j < this->test_output_z_buffer->mu_a.size(); i++) {
            if (this->test_output_z_buffer->mu_a[j] !=
                this->ref_output_z_buffer->mu_a[j]) {
                std::cout << "Layer name: " << layer_name << " "
                          << "Layer no" << i << "\n"
                          << std::endl;
                int check = 1;
                break;
            }
        }
        std::swap(this->test_input_z_buffer, this->test_output_z_buffer);
        std::swap(this->ref_input_z_buffer, this->ref_output_z_buffer);
    }
    // Output buffer is considered as the final output of network
    std::swap(this->test_output_z_buffer, this->test_input_z_buffer);
    std::swap(this->ref_output_z_buffer, this->ref_input_z_buffer);
}

void ModelDebugger::debug_backward(std::vector<float> &y_batch,
                                   std::vector<float> &var_obs,
                                   std::vector<int> &idx_ud_batch)
/*
 */
{
    // Output layer
    this->output_updater.update_using_indices(*test_output_z_buffer, y_batch,
                                              var_obs, idx_ud_batch,
                                              *test_input_delta_z_buffer);

    this->output_updater.update_using_indices(*ref_output_z_buffer, y_batch,
                                              var_obs, idx_ud_batch,
                                              *ref_input_delta_z_buffer);

    int num_layers = test_model.layers.size();

    for (int i = num_layers - 1; i < 1; i--) {
        auto *test_current_layer = test_model.layers[i].get();
        auto *ref_current_layer = ref_model.layers[i].get();

        // Backward pass for parameters and hidden states
        if (test_model.param_update) {
            test_current_layer->param_backward(*test_current_layer->bwd_states,
                                               *test_input_delta_z_buffer,
                                               *test_temp_states);

            ref_current_layer->param_backward(*ref_current_layer->bwd_states,
                                              *ref_input_delta_z_buffer,
                                              *ref_temp_states);
        }

        // Backward pass for hidden states
        test_current_layer->state_backward(
            *test_current_layer->bwd_states, *test_input_delta_z_buffer,
            *test_output_delta_z_buffer, *test_temp_states);

        ref_current_layer->state_backward(
            *ref_current_layer->bwd_states, *ref_input_delta_z_buffer,
            *ref_output_delta_z_buffer, *ref_temp_states);

        // Copy to host for gpu model
#ifdef USE_CUDA
        if (this->test_model.device.compare("cuda") == 0) {
            DeltaStateCuda *test_output_delta_z_buffer_cu =
                dynamic_cast<DeltaStateCuda *>(
                    this->test_output_z_buffer.get());
            test_output_delta_z_buffer_cu->to_host();
        }
        if (this->ref_model.device.compare("cuda") == 0) {
            DeltaStateCuda *ref_output_delta_z_buffer_cu =
                dynamic_cast<DeltaStateCuda *>(
                    this->ref_output_delta_z_buffer.get());
            ref_output_delta_z_buffer_cu->to_host();
        }
#endif

        // Test here
        auto layer_name = test_current_layer->get_layer_name();
        for (int j = 0; j < this->test_output_delta_z_buffer->delta_mu.size();
             i++) {
            if (this->test_output_delta_z_buffer->delta_mu[j] !=
                this->ref_output_delta_z_buffer->delta_mu[j]) {
                std::cout << "Layer name: " << layer_name << " "
                          << "Layer no" << i << "\n"
                          << std::endl;
                break;
            }
        }

        // Pass new input data for next iteration
        if (test_current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(test_input_delta_z_buffer, test_output_delta_z_buffer);
        }

        if (ref_current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(ref_input_delta_z_buffer, ref_output_delta_z_buffer);
        }
    }

    // Parameter update for input layer
    if (test_model.param_update) {
        test_model.layers[0]->param_backward(*test_model.layers[0]->bwd_states,
                                             *test_model.input_delta_z_buffer,
                                             *test_model.temp_states);
    }

    // State update for input layer
    if (test_model.input_hidden_state_update) {
        test_model.layers[0]->state_backward(
            *test_model.layers[0]->bwd_states, *test_model.input_delta_z_buffer,
            *test_model.output_delta_z_buffer, *test_model.temp_states);
    }

    if (ref_model.param_update) {
        ref_model.layers[0]->param_backward(*ref_model.layers[0]->bwd_states,
                                            *ref_model.input_delta_z_buffer,
                                            *ref_model.temp_states);
    }

    if (ref_model.input_hidden_state_update) {
        ref_model.layers[0]->state_backward(
            *ref_model.layers[0]->bwd_states, *ref_model.input_delta_z_buffer,
            *ref_model.output_delta_z_buffer, *ref_model.temp_states);
    }
}
