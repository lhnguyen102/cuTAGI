///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      December 10, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/data_struct.h"

HiddenStateBase::HiddenStateBase(size_t n, size_t m)
    : mu_z(n, 0.0f),
      var_z(n, 0.0f),
      mu_a(n, 0.0f),
      var_a(n, 0.0f),
      jcb(n, 1.0f),
      size(n),
      block_size(m) {}

DeltaStateBase::DeltaStateBase(size_t n, size_t m)
    : delta_mu(n, 0.0f), delta_var(n, 0.0f), size(n), block_size(m) {}

TempStateBase::TempStateBase(size_t n, size_t m)
    : tmp_1(n, 0.0f), tmp_2(n, 0.0f), size(n), block_size(m) {}