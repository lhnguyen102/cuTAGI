///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      April 10, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>

#include "data_struct.h"

class HiddenStateCuda : public BaseHiddenStates {
   public:
    float *d_mu_a = nullptr;
    float *d_var_a = nullptr;
    float *d_jcb = nullptr;

    HiddenStateCuda(size_t size, size_t block_size);
    HiddenStateCuda();
    ~HiddenStateCuda();

    void set_input_x(const std::vector<float> &mu_x,
                     const std::vector<float> &var_x,
                     const size_t block_size) override;

    std::string get_name() const override { return "HiddenStateCuda"; };
    void allocate_memory();
    void deallocate_memory();
    void to_device();
    void chunks_to_device(const size_t chunk_size);
    void to_host();
    void set_size(size_t size, size_t block_size) override;
    void copy_from(const BaseHiddenStates &source, int num_data = -1) override;

    // Move constructor
    HiddenStateCuda(HiddenStateCuda &&other) noexcept
        : BaseHiddenStates(std::move(other)) {
        d_mu_a = other.d_mu_a;
        d_var_a = other.d_var_a;
        d_jcb = other.d_jcb;

        other.d_mu_a = nullptr;
        other.d_var_a = nullptr;
        other.d_jcb = nullptr;
    }

    // Move assignment operator
    HiddenStateCuda &operator=(HiddenStateCuda &&other) noexcept {
        BaseHiddenStates::operator=(std::move(other));
        if (this != &other) {
            deallocate_memory();

            // Transfer ownership
            d_mu_a = other.d_mu_a;
            d_var_a = other.d_var_a;
            d_jcb = other.d_jcb;

            other.d_mu_a = nullptr;
            other.d_var_a = nullptr;
            other.d_jcb = nullptr;
        }
        return *this;
    }

    void swap(BaseHiddenStates &other) override;
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
    void deallocate_memory();
    void to_device();
    void to_host();
    void reset_zeros() override;
    void copy_from(const BaseDeltaStates &source, int num_data = -1) override;
    void set_size(size_t size, size_t block_size) override;
    void swap(BaseDeltaStates &other) override;
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
    void deallocate_memory();
    void to_device();
    void to_host();
    void set_size(size_t size, size_t block_size) override;
};

class BackwardStateCuda : public BaseBackwardStates {
   public:
    float *d_mu_a = nullptr;
    float *d_jcb = nullptr;

    BackwardStateCuda();
    ~BackwardStateCuda();

    std::string get_name() const override { return "BackwardStateCuda"; };

    void allocate_memory();
    void deallocate_memory();
    void to_device();
    void to_host();
    void set_size(size_t size) override;
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
    void deallocate_memory();
    void to_device();
    void to_host();
    void set_size(size_t size, size_t block_size) override;
};

class LSTMStateCuda : public BaseLSTMStates {
   public:
    float *d_mu_ha = nullptr, *d_var_ha = nullptr, *d_mu_f_ga = nullptr,
          *d_var_f_ga = nullptr, *d_jcb_f_ga = nullptr, *d_mu_i_ga = nullptr,
          *d_var_i_ga = nullptr, *d_jcb_i_ga = nullptr, *d_mu_c_ga = nullptr,
          *d_var_c_ga = nullptr, *d_jcb_c_ga = nullptr, *d_mu_o_ga = nullptr,
          *d_var_o_ga = nullptr, *d_jcb_o_ga = nullptr, *d_mu_ca = nullptr,
          *d_var_ca = nullptr, *d_jcb_ca = nullptr, *d_mu_c = nullptr,
          *d_var_c = nullptr, *d_cov_i_c = nullptr, *d_cov_o_tanh_c = nullptr;
    float *d_mu_c_prev = nullptr, *d_var_c_prev = nullptr,
          *d_mu_h_prev = nullptr, *d_var_h_prev = nullptr,
          *d_mu_c_prior = nullptr, *d_var_c_prior = nullptr,
          *d_mu_h_prior = nullptr, *d_var_h_prior = nullptr;

    LSTMStateCuda(size_t num_states, size_t num_inputs);
    LSTMStateCuda();
    ~LSTMStateCuda();
    std::string get_name() const override { return "LSTMStateCuda"; };
    void set_num_states(size_t num_states, size_t num_inputs) override;
    void allocate_memory();
    void deallocate_memory();
    void to_device();
    void to_host();
};