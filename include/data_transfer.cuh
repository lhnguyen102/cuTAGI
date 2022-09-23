///////////////////////////////////////////////////////////////////////////////
// File:         data_transfer.cuh
// Description:  Header file for data transfer between CPU and GPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2022
// Updated:      September 23, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

class LSTMStateGPU {
   public:
    size_t n_state_bytes, n_max_state_bytes;
    float *d_mha, *d_Sha, *d_mf_ga, *d_Sf_ga, *d_Jf_ga, *d_mi_ga, *d_Si_ga,
        *d_Ji_ga, *d_mc_ga, *d_Sc_ga, *d_Jc_ga, *d_mo_ga, *d_So_ga, *d_Jo_ga,
        *d_mca, *d_Sca, *d_Jca, *d_mc, *d_Sc, *d_mc_prev, *d_Sc_prev,
        *d_mh_prev, *d_Sh_prev, *d_Ci_c, *d_Co_tanh_c;
    LSTMState *lstm;
    LSTMStateGPU();
    void set_values(LSTMState &_lstm);
    void compute_bytes(int n_state, int n_max_state);
    void allocate_cuda_memory();
    void copy_host_to_device();
    void copy_device_to_host();
    ~LSTMStateGPU();
};

class NoiseStateGPU {
   public:
    size_t n_bytes;
    float *d_ma_mu, *d_Sa_mu, *d_J_mu, *d_Sz_mu, *d_ma_v2b_prior,
        *d_Sa_v2b_prior, *d_Sa_v2_prior;
    float *d_Cza_v2, *d_J_v2, *d_ma_v2_post, *d_Sa_v2_post, *d_J_v, *d_delta_mv;
    float *d_delta_Sv, *d_delta_mz_mu, *d_delta_Sz_mu, *d_delta_mz_v2b;
    float *d_delta_Sz_v2b;

    NoiseStateGPU();
    void compute_bytes(int _n);
    void allocate_cuda_memory();
    void copy_host_to_device(NoiseState &noise_state);
    void copy_device_to_host(NoiseState &noise_state);

    ~NoiseStateGPU();
};

class DerivativeStateGPU
/*Derivative state*/
{
   public:
    size_t n_state_bytes, n_tmp_bytes;
    float *d_mda, *d_Sda, *d_md_node, *d_Sd_node, *d_Cdo_diwi, *d_md_layer,
        *d_Sd_layer, *d_md_layer_m, *d_Sd_layer_m, *d_md_layer_m_o, *d_Cdi_zi,
        *d_Cdo_zi, *d_Cld_zi, *d_Cld_zi_m;
    DerivativeStateGPU();
    void compute_bytes(int n_state, int n_max_nodes, int batch_size);
    void allocate_cuda_memory();
    void copy_host_to_device(DerivativeState &derv_state);
    void copy_device_to_host(DerivativeState &derv_state);
    ~DerivativeStateGPU();
};

class StateGPU {
   public:
    size_t s_bytes, sc_bytes, dsc_bytes, ra_bytes, max_full_cov_bytes;
    std::vector<float> mra_prev, Sra_prev, ms, Ss, SsTmp;
    float *d_mz, *d_Sz, *d_ma, *d_Sa, *d_J, *d_msc, *d_Ssc, *d_mdsc, *d_Sdsc;
    float *d_mra, *d_Sra, *d_mra_prev, *d_Sra_prev, *d_ms, *d_Ss, *d_SsTmp;
    float *d_Sz_f, *d_Sa_f, *d_Sz_fp;
    NetState *state_cpu;
    NoiseStateGPU noise_state;
    DerivativeStateGPU derv_state;
    LSTMStateGPU lstm;

    StateGPU();
    void set_values(NetState &state, Network &net);
    void allocate_cuda_memory();
    void copy_host_to_device();
    void copy_device_to_host();

    ~StateGPU();
};

class ParamGPU {
   public:
    size_t w_bytes, b_bytes, w_sc_bytes, b_sc_bytes;
    float *d_mw, *d_Sw, *d_mb, *d_Sb, *d_mw_sc, *d_Sw_sc, *d_mb_sc, *d_Sb_sc;

    ParamGPU();
    void set_values(int w, int b, int w_sc, int b_sc);
    void allocate_cuda_memory();
    void copy_host_to_device(Param &theta);
    void copy_device_to_host(Param &theta);

    ~ParamGPU();
};

class IndexGPU {
   public:
    size_t Fmwa_1_bytes, Fmwa_2_bytes, FCzwa_1_bytes, FCzwa_2_bytes;
    size_t Szz_ud_bytes, pooling_bytes, FCwz_2_bytes, Swz_ud_bytes;
    size_t Fmwa_2_sc_bytes, FCzwa_1_sc_bytes, FCzwa_2_sc_bytes, Szz_ud_sc_bytes;

    int *d_Fmwa_1, *d_Fmwa_2, *d_FCzwa_1, *d_FCzwa_2, *d_Szz_ud, *d_pooling;
    int *d_FCwz_2, *d_Swz_ud, *d_Fmwa_2_sc, *d_FCzwa_1_sc, *d_FCzwa_2_sc;
    int *d_Szz_ud_sc;

    IndexGPU();
    void set_values(IndexOut &idx);
    void allocate_cuda_memory();
    void copy_host_to_device(IndexOut &idx);

    ~IndexGPU();
};

class DeltaStateGPU {
   public:
    std::vector<float> delta_mz, delta_Sz, delta_mdsc, delta_Sdsc, delta_msc;
    std::vector<float> delta_Ssc, delta_mzsc, delta_Szsc, dummy_m, dummy_S;
    std::vector<float> delta_m, delta_S, delta_mx, delta_Sx;

    size_t s_bytes, sc_bytes, dsc_bytes, max_n_s_bytes;
    float *d_delta_mz, *d_delta_Sz, *d_delta_mdsc, *d_delta_Sdsc, *d_delta_msc;
    float *d_delta_Ssc, *d_delta_mzsc, *d_delta_Szsc, *d_dummy_m, *d_dummy_S;
    float *d_delta_m, *d_delta_S, *d_delta_mx, *d_delta_Sx;

    DeltaStateGPU();
    void set_values(int s, int sc, int dsc, int max_n_s);
    void allocate_cuda_memory();
    void copy_host_to_device();
    void copy_device_to_host();

    ~DeltaStateGPU();
};

class DeltaParamGPU {
   public:
    std::vector<float> delta_mw, delta_Sw, delta_mb, delta_Sb, delta_mw_sc;
    std::vector<float> delta_Sw_sc, delta_mb_sc, delta_Sb_sc;
    size_t w_bytes, b_bytes, w_sc_bytes, b_sc_bytes;
    float *d_delta_mw, *d_delta_Sw, *d_delta_mb, *d_delta_Sb, *d_delta_mw_sc;
    float *d_delta_Sw_sc, *d_delta_mb_sc, *d_delta_Sb_sc;

    DeltaParamGPU();
    void set_values(int w, int b, int w_sc, int b_sc);
    void allocate_cuda_memory();
    void copy_host_to_device();
    void copy_device_to_host();

    ~DeltaParamGPU();
};

class InputGPU {
   public:
    size_t id_bytes, id_f_bytes;
    float *d_x_batch, *d_Sx_batch, *d_Sx_f_batch;

    InputGPU(Network &net);
    void allocate_cuda_memory();
    void copy_host_to_device(std::vector<float> &x_batch,
                             std::vector<float> &Sx_batch,
                             std::vector<float> &Sx_f_batch);

    void copy_device_to_host(std::vector<float> &x_batch,
                             std::vector<float> &Sx_batch,
                             std::vector<float> &Sx_f_batch);

    ~InputGPU();
};

class ObsGPU {
   public:
    size_t od_bytes, ode_bytes;
    float *d_y_batch, *d_V_batch;
    int *d_idx_ud_batch;

    ObsGPU(int ny, int nye, int B);
    void allocate_cuda_memory();

    void copy_host_to_device(std::vector<float> &y_batch,
                             std::vector<int> &idx_ud_batch,
                             std::vector<float> &V_batch);

    void copy_device_to_host(std::vector<float> &y_batch,
                             std::vector<int> &idx_ud_batch,
                             std::vector<float> &V_batch);

    ~ObsGPU();
};
