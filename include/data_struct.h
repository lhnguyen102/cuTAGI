///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 01, 2023
// Updated:      March 28, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <string>
#include <vector>

class BaseHiddenStates {
   public:
    std::vector<float> mu_a;
    std::vector<float> var_a;
    std::vector<float> jcb;
    size_t size = 0;         // size of data including buffer
    size_t block_size = 1;   // batch size
    size_t actual_size = 0;  // actual size of data
    size_t width = 0, height = 0, depth = 0;

    BaseHiddenStates(size_t n, size_t m);
    BaseHiddenStates();
    ~BaseHiddenStates() = default;

    virtual void set_input_x(const std::vector<float> &mu_x,
                             const std::vector<float> &var_x,
                             const size_t block_size);

    virtual std::string get_name() const { return "BaseHiddenStates"; };
};

class BaseDeltaStates {
   public:
    std::vector<float> delta_mu;
    std::vector<float> delta_var;
    size_t size = 0, block_size = 1, actual_size = 0;

    BaseDeltaStates(size_t n, size_t m);
    BaseDeltaStates();
    ~BaseDeltaStates() = default;
    virtual std::string get_name() const { return "BaseDeltaStates"; };
    virtual void reset_zeros();
    virtual void copy_from(const BaseDeltaStates &source, int num_data = -1);
};

class BaseTempStates {
   public:
    std::vector<float> tmp_1;
    std::vector<float> tmp_2;
    size_t size = 0, block_size = 1;

    BaseTempStates(size_t n, size_t m);
    BaseTempStates();
    ~BaseTempStates() = default;
    virtual std::string get_name() const { return "BaseTempStates"; };
};

class BaseBackwardStates {
   public:
    std::vector<float> mu_a;
    std::vector<float> jcb;
    size_t size = 0;

    BaseBackwardStates(int num);
    BaseBackwardStates();
    ~BaseBackwardStates() = default;
    virtual std::string get_name() const { return "BaseBackwardStates"; };
};

class BaseObservation {
   public:
    std::vector<float> mu_obs;
    std::vector<float> var_obs;
    std::vector<int> selected_idx;
    size_t size = 0, block_size = 1, actual_size = 0;
    size_t idx_size = 0;

    BaseObservation(size_t n, size_t m, size_t k);
    BaseObservation();
    ~BaseObservation() = default;

    virtual std::string get_name() const { return "BaseObservation"; };

    void set_obs(std::vector<float> &mu_obs, std::vector<float> &var_obs);
    void set_selected_idx(std::vector<int> &selected_idx);
};

class BaseLSTMStates {
   public:
    size_t num_states;
    size_t num_inputs;
    std::vector<float> mu_ha, var_ha, mu_f_ga, var_f_ga, jcb_f_ga, mu_i_ga,
        var_i_ga, jcb_i_ga, mu_c_ga, var_c_ga, jcb_c_ga, mu_o_ga, var_o_ga,
        jcb_o_ga, mu_ca, var_ca, jcb_ca, mu_c, var_c, mu_c_prev, var_c_prev,
        mu_h_prev, var_h_prev, cov_i_c, cov_o_tanh_c;

    BaseLSTMStates(size_t num_states, size_t num_inputs);
    BaseLSTMStates();
    ~BaseLSTMStates() = default;
    virtual void set_num_states(size_t num_states, size_t num_inputs);
    virtual std::string get_name() const { return "BaseLSTMStates"; };
    void reset_zeros();
};
