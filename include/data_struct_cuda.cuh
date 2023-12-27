///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      December 20, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_struct.h"

class HiddenStateCuda : public BaseHiddenStates {
   public:
    float *d_mu_z = nullptr;
    float *d_var_z = nullptr;
    float *d_mu_a = nullptr;
    float *d_var_a = nullptr;
    float *d_jcb = nullptr;

    HiddenStateCuda(size_t size, size_t block_size);
    HiddenStateCuda();
    ~HiddenStateCuda();

    std::string get_name() const override { return "HiddenStateCuda"; };
    void allocate_memory();
    void to_device();
    void to_host();
};

class DeltaStateCuda : public BaseDeltaStates {
   public:
    float *d_delta_mu = nullptr;
    float *d_delta_var = nullptr;

    DeltaStateCuda(size_t size, size_t block_size);
    DeltaStateCuda();
    ~DeltaStateCuda();

    std::string get_name() const override { return "DeltaStateCuda"; };
    void allocate_memory();
    void to_device();
    void to_host();
    void reset_zeros();
};

class TempStateCuda : public BaseTempStates {
   public:
    float *d_tmp_1 = nullptr;
    float *d_tmp_2 = nullptr;

    TempStateCuda(size_t size, size_t block_size);
    TempStateCuda();
    ~TempStateCuda();

    std::string get_name() const override { return "TempStateCuda"; };

    void allocate_memory();
    void to_device();
    void to_host();
};

class BackwardStateCuda : public BaseBackwardStates {
   public:
    float *d_mu_a = nullptr;
    float *d_jcb = nullptr;

    BackwardStateCuda();
    ~BackwardStateCuda();

    std::string get_name() const override { return "BackwardStateCuda"; };

    void allocate_memory();
    void to_device();
    void to_host();
};

class ObservationCuda : public BaseObservation {
   public:
    float *d_mu_obs = nullptr;
    float *d_var_obs = nullptr;
    int *d_selected_idx = nullptr;

    ObservationCuda();
    ~ObservationCuda();

    std::string get_name() const override { return "ObservationCuda"; };

    void allocate_memory();
    void to_device();
    void to_host();
};