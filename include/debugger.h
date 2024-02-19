///////////////////////////////////////////////////////////////////////////////
// File:         debugger.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 14, 2024
// Updated:      February 15, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "base_output_updater.h"
#include "sequential.h"

class ModelDebugger {
   public:
    std::shared_ptr<BaseHiddenStates> test_output_z_buffer;
    std::shared_ptr<BaseHiddenStates> test_input_z_buffer;
    std::shared_ptr<BaseTempStates> test_temp_states;

    std::shared_ptr<BaseDeltaStates> test_output_delta_z_buffer;
    std::shared_ptr<BaseDeltaStates> test_input_delta_z_buffer;

    std::shared_ptr<BaseHiddenStates> ref_output_z_buffer;
    std::shared_ptr<BaseHiddenStates> ref_input_z_buffer;
    std::shared_ptr<BaseTempStates> ref_temp_states;

    std::shared_ptr<BaseDeltaStates> ref_output_delta_z_buffer;
    std::shared_ptr<BaseDeltaStates> ref_input_delta_z_buffer;

    Sequential test_model;
    Sequential ref_model;
    OutputUpdater output_updater;

    ModelDebugger(Sequential &test_model, Sequential &ref_model,
                  OutputUpdater &output_updater);
    ~ModelDebugger();

    void lazy_init(int batch_size, int z_buffer_size);
    void debug_forward(const std::vector<float> &mu_x,
                       const std::vector<float> &var_x = std::vector<float>());
    void debug_backward(std::vector<float> &y_batch,
                        std::vector<float> &var_obs,
                        std::vector<int> &idx_ud_batch);
};
