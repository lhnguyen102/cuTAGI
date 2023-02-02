///////////////////////////////////////////////////////////////////////////////
// File:         data_transfer_cpu.h
// Description:  Header file for data transfer within CPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 20, 2022
// Updated:      January 05, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

class DeltaStateSoftmax {
   public:
    std::vector<float> delta_mu_y_check, delta_var_y_check, delta_mu_zy_check,
        delta_var_zy_check;
    int n;
    DeltaStateSoftmax();
    ~DeltaStateSoftmax();
    void set_values(int n);
    void reset_delta();
};

//////////////////////////////
// DELTA STATE
//////////////////////////////
class DeltaState {
   public:
    std::vector<float> delta_mz, delta_Sz, delta_mdsc, delta_Sdsc, delta_msc;
    std::vector<float> delta_Ssc, delta_mzsc, delta_Szsc, dummy_m, dummy_S;
    std::vector<float> delta_m, delta_S, delta_mx, delta_Sx;
    DeltaStateSoftmax delta_state_softmax;

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