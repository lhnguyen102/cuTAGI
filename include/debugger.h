///////////////////////////////////////////////////////////////////////////////
// File:         debugger.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 14, 2024
// Updated:      March 31, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "base_output_updater.h"
#include "sequential.h"
#include "tagi_network_cpu.h"
#ifdef USE_CUDA
#include "tagi_network.cuh"
#endif

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
    OutputUpdater cpu_output_updater;
    OutputUpdater cuda_output_updater;

    ModelDebugger(Sequential &test_model, Sequential &ref_model);
    ~ModelDebugger();

    void lazy_init(int batch_size, int z_buffer_size);
    void debug_forward(const std::vector<float> &mu_x,
                       const std::vector<float> &var_x = std::vector<float>());
    void debug_backward(std::vector<float> &y_batch,
                        std::vector<float> &var_obs,
                        std::vector<int> &idx_ud_batch);
};

#ifdef USE_CUDA
class CrossValidator
// Compare new version with the previous one
{
   public:
    std::shared_ptr<BaseHiddenStates> test_output_z_buffer;
    std::shared_ptr<BaseHiddenStates> test_input_z_buffer;
    std::shared_ptr<BaseTempStates> test_temp_states;

    std::shared_ptr<BaseDeltaStates> test_output_delta_z_buffer;
    std::shared_ptr<BaseDeltaStates> test_input_delta_z_buffer;

    Sequential test_model;
    TagiNetwork *ref_model;

    OutputUpdater cpu_output_updater;
    OutputUpdater cuda_output_updater;

    CrossValidator(Sequential &test_model, TagiNetwork *ref_model,
                   std::string &param_prefix);
    ~CrossValidator();

    void lazy_init(int batch_size, int z_buffer_size);

    void validate_forward(
        const std::vector<float> &mu_x,
        const std::vector<float> &var_x = std::vector<float>());

    void validate_backward(std::vector<float> &y_batch,
                           std::vector<float> &var_obs,
                           std::vector<int> &idx_ud_batch);
};

#endif
