///////////////////////////////////////////////////////////////////////////////
// File:         data_transfer_cpu.h
// Description:  Header file for data transfer within CPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 20, 2022
// Updated:      September 10, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

//////////////////////////////
// DELTA STATE
//////////////////////////////
class DeltaState {
   public:
    std::vector<float> delta_mz, delta_Sz, delta_mdsc, delta_Sdsc, delta_msc;
    std::vector<float> delta_Ssc, delta_mzsc, delta_Szsc, dummy_m, dummy_S;
    std::vector<float> delta_m, delta_S, delta_mx, delta_Sx;
    MultiHeadAttentionDelta mha;

    DeltaState();
    ~DeltaState();
    void set_values(Network &net_prop);
    void reset_updated_values(int n);
};

//////////////////////////////
// DELTA PARAM
//////////////////////////////
class DeltaParam {
   public:
    std::vector<float> delta_mw, delta_Sw, delta_mb, delta_Sb, delta_mw_sc;
    std::vector<float> delta_Sw_sc, delta_mb_sc, delta_Sb_sc;

    DeltaParam();
    void set_values(int w, int b, int w_sc, int b_sc);
    void reset_zero();

    ~DeltaParam();
};

//////////////////////////////
// INPUT
//////////////////////////////
class Input {
   public:
    std::vector<float> x_batch, Sx_batch, Sx_f_batch;
    Input();
    void set_values(std::vector<float> &x, std::vector<float> &Sx,
                    std::vector<float> &Sx_f);
    ~Input();
};

//////////////////////////////
// OBSERVATION
//////////////////////////////
class Obs {
   public:
    std::vector<float> y_batch, V_batch;
    std::vector<int> idx_ud_batch;
    Obs();
    void set_values(std::vector<float> &y, std::vector<float> &Sy,
                    std::vector<int> &idx_ud);
    ~Obs();
};