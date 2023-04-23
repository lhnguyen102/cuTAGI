///////////////////////////////////////////////////////////////////////////////
// File:         data_transfer_cpu.cpp
// Description:  CPU version for data transfer
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 20, 2022
// Updated:      March 05, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "../include/data_transfer_cpu.h"
//////////////////////////////
// DELTA STATE
//////////////////////////////
DeltaState::DeltaState() {}

void DeltaState::set_values(Network &net_prop) {
    int s = net_prop.n_state;
    int sc = net_prop.n_state_sc;
    int dsc = net_prop.n_state_sc;
    int max_n_s = net_prop.n_max_state;

    this->delta_mz.resize(max_n_s, 0);
    this->delta_Sz.resize(max_n_s, 0);
    this->delta_mdsc.resize(dsc, 0);
    this->delta_Sdsc.resize(dsc, 0);
    this->delta_msc.resize(sc, 0);
    this->delta_Ssc.resize(sc, 0);
    this->delta_mzsc.resize(max_n_s, 0);
    this->delta_Szsc.resize(max_n_s, 0);
    this->dummy_m.resize(max_n_s, 0);
    this->dummy_S.resize(max_n_s, 0);
    this->delta_m.resize(s, 0);
    this->delta_S.resize(s, 0);
    this->delta_mx.resize(dsc, 0);
    this->delta_Sx.resize(dsc, 0);
}

void DeltaState::reset_updated_values(int n) {
    for (int i = 0; i < n; i++) {
        this->delta_mz[i] = 0.0f;
        this->delta_Sz[i] = 0.0f;
    }
}

DeltaState::~DeltaState() {}

//////////////////////////////
// DELTA PARAM
//////////////////////////////
DeltaParam::DeltaParam(){};

void DeltaParam::set_values(int w, int b, int w_sc, int b_sc) {
    this->delta_mw.resize(w, 0);
    this->delta_Sw.resize(w, 0);
    this->delta_mb.resize(b, 0);
    this->delta_Sb.resize(b, 0);
    this->delta_mw_sc.resize(w_sc, 0);
    this->delta_Sw_sc.resize(w_sc, 0);
    this->delta_mb_sc.resize(b_sc, 0);
    this->delta_Sb_sc.resize(b_sc, 0);
}

DeltaParam::~DeltaParam() {}

//////////////////////////////
// INPUT
//////////////////////////////
Input::Input() {}
void Input::set_values(std::vector<float> &x, std::vector<float> &Sx,
                       std::vector<float> &Sx_f) {
    this->x_batch = x;
    this->Sx_batch = Sx;
    this->Sx_f_batch = Sx_f;
}
Input::~Input() {}

//////////////////////////////
// OBSERVATION
//////////////////////////////
Obs::Obs() {}
void Obs::set_values(std::vector<float> &y, std::vector<float> &Sy,
                     std::vector<int> &idx_ud) {
    this->y_batch = y;
    this->V_batch = Sy;
    this->idx_ud_batch = idx_ud;
}
Obs::~Obs() {}