///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      December 15, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_struct.h"

class HiddenStateCuda : public BaseHiddenStates {
   public:
    float *d_mu_z;
    float *d_var_z;
    float *d_mu_a;
    float *d_var_a;
    float *d_jcb;

    HiddenStateCuda(size_t size, size_t block_size);
    HiddenStateCuda();
    ~HiddenStateCuda();

    std::string get_name() const override { return "cudaHidden"; };
    void allocate_memory();
    void to_device();
    void to_host();
};

class DeltaStateCuda : public BaseDeltaStates {
   public:
    float *d_delta_mu;
    float *d_delta_var;

    DeltaStateCuda(size_t size, size_t block_size);
    DeltaStateCuda();
    ~DeltaStateCuda();

    void allocate_memory();
    void to_device();
    void to_host();
};

class TempStateCuda : public BaseTempStates {
   public:
    float *d_tmp_1;
    float *d_tmp_2;

    TempStateCuda(size_t size, size_t block_size);
    TempStateCuda();
    ~TempStateCuda();

    void allocate_memory();
    void to_device();
    void to_host();
};

class BackwardStateCuda : public BaseBackwardStates {
   public:
    float *d_mu_a;
    float *d_jcb;
    int size = 0;

    BackwardStateCuda();
    ~BackwardStateCuda();

    void allocate_memory();
    void to_device();
    void to_host();
};
