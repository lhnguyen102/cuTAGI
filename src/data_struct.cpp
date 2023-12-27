///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      December 27, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/data_struct.h"

BaseHiddenStates::BaseHiddenStates(size_t n, size_t m)
    : mu_z(n, 0.0f),
      var_z(n, 0.0f),
      mu_a(n, 0.0f),
      var_a(n, 0.0f),
      jcb(n, 1.0f),
      size(n),
      block_size(m) {}

BaseHiddenStates::BaseHiddenStates() {}

BaseDeltaStates::BaseDeltaStates(size_t n, size_t m)
    : delta_mu(n, 0.0f), delta_var(n, 0.0f), size(n), block_size(m) {}

BaseDeltaStates::BaseDeltaStates() {}

BaseTempStates::BaseTempStates(size_t n, size_t m)
    : tmp_1(n, 0.0f), tmp_2(n, 0.0f), size(n), block_size(m) {}

BaseTempStates::BaseTempStates() {}

BaseBackwardStates::BaseBackwardStates(int num) : size(num) {}
BaseBackwardStates::BaseBackwardStates() {}

BaseObservation::BaseObservation(size_t n, size_t m, size_t k)
    : size(n), block_size(m), idx_size(k) {}

BaseObservation::BaseObservation() {}

void BaseObservation::set_obs(std::vector<float> &mu_obs,
                              std::vector<float> &var_obs) {
    this->mu_obs = mu_obs;
    this->var_obs = var_obs;
}