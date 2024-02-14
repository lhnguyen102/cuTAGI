///////////////////////////////////////////////////////////////////////////////
// File:         debugger.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 14, 2024
// Updated:      February 14, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/debugger.h"

#include <memory>
#include <vector>

#include "../include/base_output_updater.h"
#include "../include/data_struct.h"
#ifdef USE_CUDA
#include "data_struct_cuda.cuh"
#endif

void debug_forward(Sequential &test_model, Sequential &ref_model,
                   const std::vector<float> &mu_x,
                   const std::vector<float> &var_x)
/*
 */
{
    // Test model
    int batch_size = mu_x.size() / test_model.layers.front()->input_size;
    int z_buffer_size = batch_size * test_model.z_buffer_size;
    std::shared_ptr<BaseHiddenStates> test_output_z_buffer;
    std::shared_ptr<BaseHiddenStates> test_input_z_buffer;
    std::shared_ptr<BaseTempStates> test_temp_states;

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

    // Reference Model
    std::shared_ptr<BaseHiddenStates> ref_output_z_buffer;
    std::shared_ptr<BaseHiddenStates> ref_input_z_buffer;
    std::shared_ptr<BaseTempStates> ref_temp_states;

    if (test_model->device.compare("cpu") == 0) {
        ref_output_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        ref_input_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        ref_temp_states =
            std::make_shared<BaseTempStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (test_model->device.compare("cuda") == 0) {
        ref_output_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        ref_input_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        ref_temp_states =
            std::make_shared<TempStateCuda>(z_buffer_size, batch_size);
    }
#endif

    // Merge input data to the input buffer
    test_input_z_buffer->set_input_x(mu_x, var_x, batch_size);
    ref_input_z_buffer->set_input_x(mu_x, var_x, batch_size);

    int num_layers = test_model->layers.size();

    for (int i = 0; i < num_layers; i++) {
        auto *test_current_layer = test_model.layers[i].get();
        auto *ref_current_layer = ref_model.layers[i].get();

        test_current_layer->forward(*test_input_z_buffer, *test_output_z_buffer,
                                    *test_temp_states);

        ref_current_layer->forward(*ref_input_z_buffer, *ref_output_z_buffer,
                                   *ref_temp_states);

        // Copy to host for gpu model
#ifdef USE_CUDA
        if (test_model->device.compare("cuda") == 0) {
            test_input_z_buffer.to_host();
            test_output_z_buffer.to_host();
        }
        if (ref_model->device.compare("cuda") == 0) {
            ref_input_z_buffer.to_host();
            ref_output_z_buffer.to_host();
        }
#endif

        std::swap(test_input_z_buffer, test_output_z_buffer);
        std::swap(ref_input_z_buffer, ref_output_z_buffer);
    }
}

void debug_backward(Sequential &test_model, Sequential &ref_model,
                    OutputUpdater &output_updater, std::vector<float> &y_batch,
                    std::vector<float> &var_obs, std::vector<int> &idx_ud_batch)
/*
 */
{
    // Test Model
    int batch_size = y_batch.size() / test_model.layers.back()->output_size;
    int z_buffer_size = batch_size * test_model.z_buffer_size;
    std::shared_ptr<BaseDeltaStates> test_output_delta_z_buffer;
    std::shared_ptr<BaseDeltaStates> test_input_delta_z_buffer;

    if (test_model.device.compare("cpu") == 0) {
        test_output_delta_z_buffer = std::make_shared<BaseDeltaStates>(
            z_buffer_size, batch_size
        test_input_delta_z_buffer = std::make_shared<BaseDeltaStates>(
            z_buffer_size, batch_size
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        test_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        test_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
#endif

    // Ref Model
    std::shared_ptr<BaseDeltaStates> ref_output_delta_z_buffer;
    std::shared_ptr<BaseDeltaStates> ref_input_delta_z_buffer;

    if (test_model.device.compare("cpu") == 0) {
        ref_output_delta_z_buffer = std::make_shared<BaseDeltaStates>(
            z_buffer_size, batch_size
        ref_input_delta_z_buffer = std::make_shared<BaseDeltaStates>(
            z_buffer_size, batch_size
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        ref_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        ref_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
#endif

    // Output layer
    output_updater.update_using_indices(*test_output_z_buffer, y_batch, var_obs,
                                        idx_ud_batch,
                                        *test_input_delta_z_buffer);

    output_updater.update_using_indices(*ref_output_z_buffer, y_batch, var_obs,
                                        idx_ud_batch,
                                        *ref_input_delta_z_buffer);

    int num_layers = test_model->layers.size();

    for (int i = num_layer - 1; i < 1; i--) {
        auto *test_current_layer = test_model.layers[i]->get();
        auto *ref_current_layer = ref_model.layers[i]->get();

        // Backward pass for parameters and hidden states
        if (this->param_update) {
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