///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calculation_cpu.cpp
// Description:  Network properties
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 12, 2022
// Updated:      July 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/derivative_calculation_cpu.h"

void compute_derivative_mean_var(std::vector<float> &mw, std::vector<float> &Sw,
                                 std::vector<float> &mda,
                                 std::vector<float> &Sda, int ni, int no, int B,
                                 std::vector<float> &md, std::vector<float> &Sd)
/* Compute derivatives for each node of fully-connected layer*/
{
    int k;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            k = (i / ni) * j * B;
            md[ni * B * j + i] = mw[k] * mda[i];
            Sd[ni * B * j + i] = Sw[k] * Sda[i] + Sw[k] * mda[i] * mda[i] +
                                 Sda[i] * mw[k] * mw[k];
        }
    }
}

void compute_cov_aw_aa(std::vector<float> &mw, std::float<float> &Sw,
                       std::vector<float> &J_o, std::vector<float> &ma_i,
                       std::vector<float> &Sa_i, int ni, int no, int B,
                       std::vector<float> &Caw, std::vector<float> &Caa)
/*Compute covraince between activation units and weights & between activation
   units */
{}