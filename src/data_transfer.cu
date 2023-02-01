///////////////////////////////////////////////////////////////////////////////
// File:         data_transfer.cu
// Description:  Data transfer between CPU and GPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2022
// Updated:      February, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/data_transfer.cuh"
////////////////////////
// CLOSED-FORM SOFTMAX GPU
///////////////////////
CfSoftmaxGPU::CfSoftmaxGPU() {
    this->n_state_bytes = 0 * sizeof(float);
    this->d_mu_e = nullptr;
    this->d_var_e = nullptr;
    this->d_mu_e_tilde = nullptr;
    this->d_var_e_tilde = nullptr;
    this->d_mu_e_check = nullptr;
    this->d_var_e_check = nullptr;
    this->d_rho_e_e_tilde = nullptr;
    this->d_cov_z_e = nullptr;
    this->d_cov_z_e_check = nullptr;
    this->d_cov_y_y_check = nullptr;
    this->d_cov_z_y_check = nullptr;
    this->d_mu_y_check = nullptr;
    this->d_var_y_check = nullptr;
}

void CfSoftmax::~CfSoftmaxGPU() {
    cudaFree(d_mu_e);
    cudaFree(d_var_e);
    cudaFree(d_mu_e_tilde);
    cudaFree(d_var_e_tilde);
    cudaFree(d_mu_e_check);
    cudaFree(d_var_e_check);
    cudaFree(d_rho_e_e_tilde);
    cudaFree(d_cov_z_e);
    cudaFree(d_cov_z_e_check);
    cudaFree(d_cov_y_y_check);
    cudaFree(d_cov_z_y_check);
    cudaFree(d_mu_y_check);
    cudaFree(d_var_y_check);
}

void CfSoftmax::set_values(CfSoftmax &_cf_softmax) {
    this->cf_softmax_cpu = &_cf_softmax;
    this->n_state_bytes = _cf_softmax.mu_e.size() * sizeof(float);
}

void CfSoftmax::allocate_cuda_memory() {
    cudaMalloc(&this->d_mu_e, this->n_state_bytes);
    cudaMalloc(&this->d_var_e, this->n_state_bytes);
    cudaMalloc(&this->d_mu_e_tilde, this->n_state_bytes);
    cudaMalloc(&this->d_var_e_tilde, this->n_state_bytes);
    cudaMalloc(&this->d_mu_e_check, this->n_state_bytes);
    cudaMalloc(&this->d_var_e_check, this->n_state_bytes);
    cudaMalloc(&this->d_rho_e_e_tilde, this->n_state_bytes);
    cudaMalloc(&this->d_cov_z_e, this->n_state_bytes);
    cudaMalloc(&this->d_cov_z_e_check, this->n_state_bytes);
    cudaMalloc(&this->d_cov_y_y_check, this->n_state_bytes);
    cudaMalloc(&this->d_cov_y_e_check, this->n_state_bytes);
    cudaMalloc(&this->d_mu_y_check, this->n_state_bytes);
    cudaMalloc(&this->d_var_y_check, this->n_state_bytes);
}

void CfSoftmax::copy_host_to_device() {
    cudaMemcpy(&this->d_mu_e, this->cf_softmax_cpu.mu_e.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_var_e, this->cf_softmax_cpu.var_e.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_mu_e_tilde, this->cf_softmax_cpu.mu_e_tilde.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_var_e_tilde, this->cf_softmax_cpu.var_e_tilde.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_mu_e_check, this->cf_softmax_cpu.mu_e_check.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_var_e_check, this->cf_softmax_cpu.var_e_check.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_rho_e_e_tilde,
               this->cf_softmax_cpu.rho_e_e_tilde.data(), this->n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_cov_z_e, this->cf_softmax_cpu.cov_z_e.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_cov_z_e_check,
               this->cf_softmax_cpu.cov_z_e_check.data(), this->n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_cov_y_y_check,
               this->cf_softmax_cpu.cov_y_y_check.data(), this->n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_cov_y_e_check,
               this->cf_softmax_cpu.cov_y_e_check.data(), this->n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_mu_y_check, this->cf_softmax_cpu.mu_y_check.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(&this->d_var_y_check, this->cf_softmax_cpu.var_y_check.data(),
               this->n_state_bytes, cudaMemcpyHostToDevice);
}

void CfSoftmax::copy_device_to_host() {
    cudaMemcpy(this->cf_softmax_cpu.mu_e.data(), this->d_mu_e, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.var_e.data(), this->d_var_e, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.mu_e_tilde.data(), this->d_mu_e_tilde,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.var_e_tilde.data(), this->d_var_e_tilde,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.mu_e_check.data(), this->d_mu_e_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.var_e_check.data(), this->d_var_e_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.rho_e_e_tilde.data(), this->d_rho_e_e_tilde,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.cov_z_e.data(), this->d_cov_z_e,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.cov_z_e_check.data(), this->d_cov_z_e_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.cov_y_y_check.data(), this->d_cov_y_y_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.cov_y_e_check.data(), this->d_cov_y_e_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.mu_y_check.data(), this->d_mu_y_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cf_softmax_cpu.var_y_check.data(), this->d_var_y_check,
               this->n_state_bytes, cudaMemcpyDeviceToHost);
}

////////////////////////
// LSTM STATE GPU
///////////////////////
LSTMStateGPU::LSTMStateGPU() {
    this->n_state_bytes = 0 * sizeof(float);
    this->n_max_state_bytes = 0 * sizeof(float);
    this->d_mha = nullptr;
    this->d_Sha = nullptr;
    this->d_mf_ga = nullptr;
    this->d_Sf_ga = nullptr;
    this->d_Jf_ga = nullptr;
    this->d_mi_ga = nullptr;
    this->d_Si_ga = nullptr;
    this->d_Ji_ga = nullptr;
    this->d_mc_ga = nullptr;
    this->d_Sc_ga = nullptr;
    this->d_Jc_ga = nullptr;
    this->d_mo_ga = nullptr;
    this->d_So_ga = nullptr;
    this->d_Jo_ga = nullptr;
    this->d_mca = nullptr;
    this->d_Sca = nullptr;
    this->d_Jca = nullptr;
    this->d_mc = nullptr;
    this->d_Sc = nullptr;
    this->d_mc_prev = nullptr;
    this->d_Sc_prev = nullptr;
    this->d_mh_prev = nullptr;
    this->d_Sh_prev = nullptr;
    this->d_Ci_c = nullptr;
    this->d_Co_tanh_c = nullptr;
}
void LSTMStateGPU::set_values(LSTMState &_lstm) { this->lstm = &_lstm; }
void LSTMStateGPU::compute_bytes(int n_state, int n_max_state) {
    this->n_state_bytes = n_state * sizeof(float);
    this->n_max_state_bytes = n_max_state * sizeof(float);
}

void LSTMStateGPU::allocate_cuda_memory() {
    cudaMalloc(&d_mha, n_state_bytes);
    cudaMalloc(&d_Sha, n_state_bytes);
    cudaMalloc(&d_mf_ga, n_state_bytes);
    cudaMalloc(&d_Sf_ga, n_state_bytes);
    cudaMalloc(&d_Jf_ga, n_state_bytes);
    cudaMalloc(&d_mi_ga, n_state_bytes);
    cudaMalloc(&d_Si_ga, n_state_bytes);
    cudaMalloc(&d_Ji_ga, n_state_bytes);
    cudaMalloc(&d_mc_ga, n_state_bytes);
    cudaMalloc(&d_Sc_ga, n_state_bytes);
    cudaMalloc(&d_Jc_ga, n_state_bytes);
    cudaMalloc(&d_mo_ga, n_state_bytes);
    cudaMalloc(&d_So_ga, n_state_bytes);
    cudaMalloc(&d_Jo_ga, n_state_bytes);
    cudaMalloc(&d_mca, n_state_bytes);
    cudaMalloc(&d_Sca, n_state_bytes);
    cudaMalloc(&d_Jca, n_state_bytes);
    cudaMalloc(&d_mc, n_state_bytes);
    cudaMalloc(&d_Sc, n_state_bytes);
    cudaMalloc(&d_mc_prev, n_state_bytes);
    cudaMalloc(&d_Sc_prev, n_state_bytes);
    cudaMalloc(&d_mh_prev, n_state_bytes);
    cudaMalloc(&d_Sh_prev, n_state_bytes);
    cudaMalloc(&d_Ci_c, n_max_state_bytes);
    cudaMalloc(&d_Co_tanh_c, n_max_state_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for LSTM state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void LSTMStateGPU::copy_host_to_device() {
    cudaMemcpy(d_mha, this->lstm->mha.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sha, this->lstm->Sha.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mf_ga, this->lstm->mf_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sf_ga, this->lstm->Sf_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jf_ga, this->lstm->Jf_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mi_ga, this->lstm->mi_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Si_ga, this->lstm->Si_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ji_ga, this->lstm->Ji_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mc_ga, this->lstm->mc_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sc_ga, this->lstm->Sc_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jc_ga, this->lstm->Jc_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mo_ga, this->lstm->mo_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_So_ga, this->lstm->So_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jo_ga, this->lstm->Jo_ga.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mca, this->lstm->mca.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sca, this->lstm->Sca.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jca, this->lstm->Jca.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mc, this->lstm->mc.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sc, this->lstm->Sc.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mc_prev, this->lstm->mc_prev.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sc_prev, this->lstm->Sc_prev.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mh_prev, this->lstm->mh_prev.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sh_prev, this->lstm->Sh_prev.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ci_c, this->lstm->Ci_c.data(), n_max_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Co_tanh_c, this->lstm->Co_tanh_c.data(), n_max_state_bytes,
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for LSTM state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void LSTMStateGPU::copy_device_to_host() {
    cudaMemcpy(this->lstm->mha.data(), d_mha, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sha.data(), d_Sha, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mf_ga.data(), d_mf_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sf_ga.data(), d_Sf_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Jf_ga.data(), d_Jf_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mi_ga.data(), d_mi_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Si_ga.data(), d_Si_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Ji_ga.data(), d_Ji_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mc_ga.data(), d_mc_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sc_ga.data(), d_Sc_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Jc_ga.data(), d_Jc_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mo_ga.data(), d_mo_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->So_ga.data(), d_So_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Jo_ga.data(), d_Jo_ga, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mca.data(), d_mca, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sca.data(), d_Sca, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Jca.data(), d_Jca, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mc.data(), d_mc, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sc.data(), d_Sc, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mc_prev.data(), d_mc_prev, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sc_prev.data(), d_Sc_prev, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->mh_prev.data(), d_mh_prev, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Sh_prev.data(), d_Sh_prev, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Ci_c.data(), d_Ci_c, n_max_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->lstm->Co_tanh_c.data(), d_Co_tanh_c, n_max_state_bytes,
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for LSTM state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

LSTMStateGPU::~LSTMStateGPU() {
    cudaFree(d_mha);
    cudaFree(d_Sha);
    cudaFree(d_mf_ga);
    cudaFree(d_Sf_ga);
    cudaFree(d_Jf_ga);
    cudaFree(d_mi_ga);
    cudaFree(d_Si_ga);
    cudaFree(d_Ji_ga);
    cudaFree(d_mc_ga);
    cudaFree(d_Sc_ga);
    cudaFree(d_Jc_ga);
    cudaFree(d_mo_ga);
    cudaFree(d_So_ga);
    cudaFree(d_Jo_ga);
    cudaFree(d_mca);
    cudaFree(d_Sca);
    cudaFree(d_Jca);
    cudaFree(d_mc);
    cudaFree(d_Sc);
    cudaFree(d_mc_prev);
    cudaFree(d_Sc_prev);
    cudaFree(d_mh_prev);
    cudaFree(d_Sh_prev);
    cudaFree(d_Ci_c);
    cudaFree(d_Co_tanh_c);
}

////////////////////////
// NOISE STATE GPU
///////////////////////
NoiseStateGPU::NoiseStateGPU() {
    this->n_bytes = 0 * sizeof(float);
    this->d_ma_mu = nullptr;
    this->d_Sa_mu = nullptr;
    this->d_Sz_mu = nullptr;
    this->d_J_mu = nullptr;
    this->d_ma_v2b_prior = nullptr;
    this->d_Sa_v2b_prior = nullptr;
    this->d_Sa_v2_prior = nullptr;
    this->d_Cza_v2 = nullptr;
    this->d_J_v2 = nullptr;
    this->d_ma_v2_post = nullptr;
    this->d_Sa_v2_post = nullptr;
    this->d_J_v = nullptr;
    this->d_delta_mv = nullptr;
    this->d_delta_Sv = nullptr;
    this->d_delta_mz_mu = nullptr;
    this->d_delta_Sz_mu = nullptr;
    this->d_delta_mz_v2b = nullptr;
    this->d_delta_Sz_v2b = nullptr;
}

void NoiseStateGPU::compute_bytes(int n) { this->n_bytes = n * sizeof(float); }

void NoiseStateGPU::allocate_cuda_memory() {
    cudaMalloc(&d_ma_mu, n_bytes);
    cudaMalloc(&d_Sa_mu, n_bytes);
    cudaMalloc(&d_Sz_mu, n_bytes);
    cudaMalloc(&d_J_mu, n_bytes);
    cudaMalloc(&d_ma_v2b_prior, n_bytes);
    cudaMalloc(&d_Sa_v2b_prior, n_bytes);
    cudaMalloc(&d_Sa_v2_prior, n_bytes);
    cudaMalloc(&d_Cza_v2, n_bytes);
    cudaMalloc(&d_J_v2, n_bytes);
    cudaMalloc(&d_ma_v2_post, n_bytes);
    cudaMalloc(&d_Sa_v2_post, n_bytes);
    cudaMalloc(&d_J_v, n_bytes);
    cudaMalloc(&d_delta_mv, n_bytes);
    cudaMalloc(&d_delta_Sv, n_bytes);
    cudaMalloc(&d_delta_mz_mu, n_bytes);
    cudaMalloc(&d_delta_Sz_mu, n_bytes);
    cudaMalloc(&d_delta_mz_v2b, n_bytes);
    cudaMalloc(&d_delta_Sz_v2b, n_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for noise state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void NoiseStateGPU::copy_host_to_device(NoiseState &noise_state) {
    cudaMemcpy(d_ma_mu, noise_state.ma_mu.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa_mu, noise_state.Sa_mu.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sz_mu, noise_state.Sz_mu.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_J_mu, noise_state.J_mu.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma_v2b_prior, noise_state.ma_v2b_prior.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa_v2b_prior, noise_state.Sa_v2b_prior.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa_v2_prior, noise_state.Sa_v2_prior.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cza_v2, noise_state.Cza_v2.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_J_v2, noise_state.J_v2.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma_v2_post, noise_state.ma_v2_post.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa_v2_post, noise_state.Sa_v2_post.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_J_v, noise_state.J_v.data(), n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mv, noise_state.delta_mv.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sv, noise_state.delta_Sv.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mz_mu, noise_state.delta_mz_mu.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sz_mu, noise_state.delta_Sz_mu.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mz_v2b, noise_state.delta_mz_v2b.data(), n_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sz_v2b, noise_state.delta_Sz_v2b.data(), n_bytes,
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for noise state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void NoiseStateGPU::copy_device_to_host(NoiseState &noise_state) {
    cudaMemcpy(noise_state.ma_mu.data(), d_ma_mu, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.Sa_mu.data(), d_Sa_mu, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.Sz_mu.data(), d_Sz_mu, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.J_mu.data(), d_J_mu, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.ma_v2b_prior.data(), d_ma_v2b_prior, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.Sa_v2b_prior.data(), d_Sa_v2b_prior, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.Sa_v2_prior.data(), d_Sa_v2_prior, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.Cza_v2.data(), d_Cza_v2, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.J_v2.data(), d_J_v2, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.ma_v2_post.data(), d_ma_v2_post, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.Sa_v2_post.data(), d_Sa_v2_post, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.J_v.data(), d_J_v, n_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.delta_mv.data(), d_delta_mv, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.delta_Sv.data(), d_delta_Sv, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.delta_mz_mu.data(), d_delta_mz_mu, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.delta_Sz_mu.data(), d_delta_Sz_mu, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.delta_mz_v2b.data(), d_delta_mz_v2b, n_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(noise_state.delta_Sz_v2b.data(), d_delta_Sz_v2b, n_bytes,
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for noise state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
};

NoiseStateGPU::~NoiseStateGPU() {
    cudaFree(d_ma_mu);
    cudaFree(d_Sa_mu);
    cudaFree(d_Sz_mu);
    cudaFree(d_J_mu);
    cudaFree(d_ma_v2b_prior);
    cudaFree(d_Sa_v2b_prior);
    cudaFree(d_Sa_v2_prior);
    cudaFree(d_Cza_v2);
    cudaFree(d_J_v2);
    cudaFree(d_ma_v2_post);
    cudaFree(d_Sa_v2_post);
    cudaFree(d_J_v);
    cudaFree(d_delta_mv);
    cudaFree(d_delta_Sv);
    cudaFree(d_delta_mz_mu);
    cudaFree(d_delta_Sz_mu);
    cudaFree(d_delta_mz_v2b);
    cudaFree(d_delta_Sz_v2b);
};

////////////////////////
// STATE GPU
///////////////////////
StateGPU::StateGPU() {
    this->d_mz = nullptr;
    this->d_Sz = nullptr;
    this->d_ma = nullptr;
    this->d_Sa = nullptr;
    this->d_J = nullptr;
    this->d_msc = nullptr;
    this->d_Ssc = nullptr;
    this->d_mdsc = nullptr;
    this->d_Sdsc = nullptr;
    this->d_mra = nullptr;
    this->d_Sra = nullptr;
    this->d_mra_prev = nullptr;
    this->d_Sra_prev = nullptr;
    this->d_ms = nullptr;
    this->d_Ss = nullptr;
    this->d_SsTmp = nullptr;
    this->d_Sz_f = nullptr;
    this->d_Sa_f = nullptr;
    this->d_Sz_fp = nullptr;
    this->noise_state = NoiseStateGPU();
    this->derv_state = DerivativeStateGPU();
    this->lstm = LSTMStateGPU();
    this->cf_softmax = CfSoftmaxGPU();
}

void StateGPU::set_values(NetState &state, Network &net) {
    this->s_bytes = state.mz.size() * sizeof(float);
    this->sc_bytes = state.msc.size() * sizeof(float);
    this->dsc_bytes = state.mdsc.size() * sizeof(float);
    this->ra_bytes = state.mra.size() * sizeof(float);
    this->state_cpu = &state;
    if (net.is_full_cov) {
        // TODO: n_max_state is not correct
        this->max_full_cov_bytes =
            (net.n_max_state * (net.n_max_state + 1) / 2 * net.batch_size) *
            sizeof(float);
    } else {
        this->max_full_cov_bytes = 0;
    }

    this->mra_prev.assign(state.mra.begin(), state.mra.end());
    this->Sra_prev.assign(state.Sra.begin(), state.Sra.end());
    this->ms.resize(state.mra.size(), 0);
    this->Ss.resize(state.Sra.size(), 0);
    this->SsTmp.resize(state.Sra.size(), 0);

    // Noise state
    if (net.noise_type.compare("heteros") == 0 ||
        net.noise_type.compare("homosce") == 0) {
        this->noise_state.compute_bytes(net.n_y * net.batch_size);
    }

    // Derivative state
    if (net.collect_derivative) {
        int num_max_nodes = net.n_max_state / net.batch_size;
        this->derv_state.compute_bytes(net.n_state, num_max_nodes,
                                       net.batch_size);
    }

    // LSTM state
    if (net.num_max_lstm_states > 0) {
        this->lstm.set_values(this->state_cpu->lstm);
        this->lstm.compute_bytes(net.num_lstm_states, net.num_max_lstm_states);
    }

    // Closed-form softmax
    if (net.activations.back() == net.act_names.cf_softmax) {
        this->cf_softmax.set_values(this->state_cpu->cf_softmax);
    }
}

void StateGPU::allocate_cuda_memory() {
    cudaMalloc(&d_mz, s_bytes);
    cudaMalloc(&d_Sz, s_bytes);
    cudaMalloc(&d_ma, s_bytes);
    cudaMalloc(&d_Sa, s_bytes);
    cudaMalloc(&d_J, s_bytes);
    cudaMalloc(&d_msc, sc_bytes);
    cudaMalloc(&d_Ssc, sc_bytes);
    cudaMalloc(&d_mdsc, dsc_bytes);
    cudaMalloc(&d_Sdsc, dsc_bytes);
    cudaMalloc(&d_mra, ra_bytes);
    cudaMalloc(&d_Sra, ra_bytes);
    cudaMalloc(&d_mra_prev, ra_bytes);
    cudaMalloc(&d_Sra_prev, ra_bytes);
    cudaMalloc(&d_ms, ra_bytes);
    cudaMalloc(&d_Ss, ra_bytes);
    cudaMalloc(&d_SsTmp, ra_bytes);
    if (max_full_cov_bytes > 0) {
        cudaMalloc(&d_Sz_f, max_full_cov_bytes);
        cudaMalloc(&d_Sa_f, max_full_cov_bytes);
        cudaMalloc(&d_Sz_fp, max_full_cov_bytes);
    }
    // If the noise inference is disable, the default value for n_bytes is set
    // zero
    if (this->noise_state.n_bytes > 0) {
        this->noise_state.allocate_cuda_memory();
    }

    // Derivative state
    if (this->derv_state.n_state_bytes > 0) {
        this->derv_state.allocate_cuda_memory();
    }

    // LSTM state
    if (this->lstm.n_state_bytes > 0) {
        this->lstm.allocate_cuda_memory();
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for hidden states - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void StateGPU::copy_host_to_device() {
    // Initialize normalization parameters
    cudaMemcpy(d_mz, this->state_cpu->mz.data(), s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sz, this->state_cpu->Sz.data(), s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma, this->state_cpu->ma.data(), s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa, this->state_cpu->Sa.data(), s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, this->state_cpu->J.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msc, this->state_cpu->msc.data(), sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ssc, this->state_cpu->Ssc.data(), sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mdsc, this->state_cpu->mdsc.data(), dsc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sdsc, this->state_cpu->Sdsc.data(), dsc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mra, this->state_cpu->mra.data(), ra_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sra, this->state_cpu->Sra.data(), ra_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mra_prev, mra_prev.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sra_prev, Sra_prev.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms, ms.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ss, Ss.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SsTmp, SsTmp.data(), ra_bytes, cudaMemcpyHostToDevice);
    if (max_full_cov_bytes > 0) {
        cudaMemcpy(d_Sz_f, this->state_cpu->Sz_f.data(), max_full_cov_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sa_f, this->state_cpu->Sa_f.data(), max_full_cov_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sz_fp, this->state_cpu->Sz_fp.data(), max_full_cov_bytes,
                   cudaMemcpyHostToDevice);
    }

    // If the noise inference is disable, the default value for n_bytes is set
    // zero
    if (this->noise_state.n_bytes > 0) {
        this->noise_state.copy_host_to_device(this->state_cpu->noise_state);
    }

    // Derivative state
    if (this->derv_state.n_state_bytes > 0) {
        this->derv_state.copy_host_to_device(this->state_cpu->derv_state);
    }

    // LSTM state
    if (this->lstm.n_state_bytes > 0) {
        this->lstm.copy_host_to_device();
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to device for hidden states - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void StateGPU::copy_device_to_host() {
    cudaMemcpy(this->state_cpu->mz.data(), d_mz, s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->Sz.data(), d_Sz, s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->ma.data(), d_ma, s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->Sa.data(), d_Sa, s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->J.data(), d_J, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->msc.data(), d_msc, sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->Ssc.data(), d_Ssc, sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->mdsc.data(), d_mdsc, dsc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->Sdsc.data(), d_Sdsc, dsc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->mra.data(), d_mra, ra_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->state_cpu->Sra.data(), d_Sra, ra_bytes,
               cudaMemcpyDeviceToHost);
    // if (max_full_cov_bytes > 0) {
    //     cudaMemcpy(this->state_cpu->Sz_f.data(), d_Sz_f, max_full_cov_bytes,
    //                cudaMemcpyDeviceToHost);
    //     cudaMemcpy(this->state_cpu->Sa_f.data(), d_Sa_f, max_full_cov_bytes,
    //                cudaMemcpyDeviceToHost);
    //     cudaMemcpy(this->state_cpu->Sz_fp.data(), d_Sz_fp,
    //     max_full_cov_bytes,
    //                cudaMemcpyDeviceToHost);
    // }

    // If the noise inference is disable, the default value for n_bytes is set
    // zero
    if (this->noise_state.n_bytes > 0) {
        this->noise_state.copy_device_to_host(this->state_cpu->noise_state);
    }

    // Derivative state
    if (this->derv_state.n_state_bytes > 0) {
        this->derv_state.copy_device_to_host(this->state_cpu->derv_state);
    }

    // LSTM state
    if (this->lstm.n_state_bytes > 0) {
        this->lstm.copy_device_to_host();
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to host for hidden states - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

StateGPU::~StateGPU() {
    cudaFree(d_mz);
    cudaFree(d_Sz);
    cudaFree(d_ma);
    cudaFree(d_Sa);
    cudaFree(d_J);
    cudaFree(d_msc);
    cudaFree(d_Ssc);
    cudaFree(d_mdsc);
    cudaFree(d_Sdsc);
    cudaFree(d_mra);
    cudaFree(d_Sra);
    cudaFree(d_mra_prev);
    cudaFree(d_Sra_prev);
    cudaFree(d_ms);
    cudaFree(d_Ss);
    cudaFree(d_SsTmp);
    cudaFree(d_Sz_f);
    cudaFree(d_Sa_f);
}

////////////////////////
// DERIVATIVE STATE GPU
///////////////////////
DerivativeStateGPU::DerivativeStateGPU() {
    this->n_state_bytes = 0 * sizeof(float);
    this->n_tmp_bytes = 0 * sizeof(float);
    this->d_mda = nullptr;
    this->d_Sda = nullptr;
    this->d_md_node = nullptr;
    this->d_Sd_node = nullptr;
    this->d_Cdo_diwi = nullptr;
    this->d_md_layer = nullptr;
    this->d_Sd_layer = nullptr;
    this->d_md_layer_m = nullptr;
    this->d_Sd_layer_m = nullptr;
    this->d_md_layer_m_o = nullptr;
    this->d_Cdi_zi = nullptr;
    this->d_Cdo_zi = nullptr;
    this->d_Cld_zi = nullptr;
    this->d_Cld_zi_m = nullptr;
}

void DerivativeStateGPU::compute_bytes(int n_state, int n_max_nodes,
                                       int batch_size) {
    this->n_state_bytes = n_state * sizeof(float);
    this->n_tmp_bytes = n_max_nodes * n_max_nodes * batch_size * sizeof(float);
}

void DerivativeStateGPU::allocate_cuda_memory() {
    cudaMalloc(&d_mda, n_state_bytes);
    cudaMalloc(&d_Sda, n_state_bytes);
    cudaMalloc(&d_md_node, n_tmp_bytes);
    cudaMalloc(&d_Sd_node, n_tmp_bytes);
    cudaMalloc(&d_Cdo_diwi, n_tmp_bytes);
    cudaMalloc(&d_md_layer, n_state_bytes);
    cudaMalloc(&d_Sd_layer, n_state_bytes);
    cudaMalloc(&d_md_layer_m, n_tmp_bytes);
    cudaMalloc(&d_Sd_layer_m, n_tmp_bytes);
    cudaMalloc(&d_md_layer_m_o, n_tmp_bytes);
    cudaMalloc(&d_Cdi_zi, n_tmp_bytes);
    cudaMalloc(&d_Cdo_zi, n_tmp_bytes);
    cudaMalloc(&d_Cld_zi, n_state_bytes);
    cudaMalloc(&d_Cld_zi_m, n_tmp_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for derivative states - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void DerivativeStateGPU::copy_host_to_device(DerivativeState &derv_state) {
    cudaMemcpy(d_mda, derv_state.mda.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sda, derv_state.Sda.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_md_node, derv_state.md_node.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sd_node, derv_state.Sd_node.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cdo_diwi, derv_state.Cdo_diwi.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_md_layer, derv_state.md_layer.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sd_layer, derv_state.Sd_layer.data(), n_state_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_md_layer_m, derv_state.md_layer_m.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sd_layer_m, derv_state.Sd_layer_m.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_md_layer_m_o, derv_state.md_layer_m_o.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cdi_zi, derv_state.Cdi_zi.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cdo_zi, derv_state.Cdo_zi.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cld_zi_m, derv_state.Cld_zi_m.data(), n_tmp_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cld_zi, derv_state.Cld_zi.data(), n_state_bytes,
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for derivative state - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void DerivativeStateGPU::copy_device_to_host(DerivativeState &derv_state) {
    cudaMemcpy(derv_state.mda.data(), d_mda, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Sda.data(), d_Sda, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.md_node.data(), d_md_node, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Sd_node.data(), d_Sd_node, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Cdo_diwi.data(), d_Cdo_diwi, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.md_layer.data(), d_md_layer, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Sd_layer.data(), d_Sd_layer, n_state_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.md_layer_m.data(), d_md_layer_m, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Sd_layer_m.data(), d_Sd_layer_m, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.md_layer_m_o.data(), d_md_layer_m_o, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Cdi_zi.data(), d_Cdi_zi, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Cdo_zi.data(), d_Cdo_zi, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Cld_zi_m.data(), d_Cld_zi_m, n_tmp_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(derv_state.Cld_zi.data(), d_Cld_zi, n_state_bytes,
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for derivative states - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

DerivativeStateGPU::~DerivativeStateGPU() {
    cudaFree(d_mda);
    cudaFree(d_Sda);
    cudaFree(d_md_node);
    cudaFree(d_Sd_node);
    cudaFree(d_Cdo_diwi);
    cudaFree(d_md_layer);
    cudaFree(d_Sd_layer);
    cudaFree(d_md_layer_m);
    cudaFree(d_Sd_layer_m);
    cudaFree(d_md_layer_m_o);
    cudaFree(d_Cdi_zi);
    cudaFree(d_Cdo_zi);
    cudaFree(d_Cld_zi_m);
    cudaFree(d_Cld_zi);
}

////////////////////////
// Parameter GPU
///////////////////////
ParamGPU::ParamGPU() {
    this->d_mw = nullptr;
    this->d_Sw = nullptr;
    this->d_mb = nullptr;
    this->d_Sb = nullptr;
    this->d_mw_sc = nullptr;
    this->d_Sw_sc = nullptr;
    this->d_mb_sc = nullptr;
    this->d_Sb_sc = nullptr;
}

void ParamGPU::set_values(Param &theta) {
    this->w_bytes = theta.mw.size() * sizeof(float);
    this->b_bytes = theta.mb.size() * sizeof(float);
    this->w_sc_bytes = theta.mw_sc.size() * sizeof(float);
    this->b_sc_bytes = theta.mb_sc.size() * sizeof(float);
    this->theta_cpu = &theta;
}

void ParamGPU::allocate_cuda_memory() {
    cudaMalloc(&d_mw, w_bytes);
    cudaMalloc(&d_Sw, w_bytes);
    cudaMalloc(&d_mb, b_bytes);
    cudaMalloc(&d_Sb, b_bytes);
    cudaMalloc(&d_mw_sc, w_sc_bytes);
    cudaMalloc(&d_Sw_sc, w_sc_bytes);
    cudaMalloc(&d_mb_sc, b_sc_bytes);
    cudaMalloc(&d_Sb_sc, b_sc_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for parameters - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void ParamGPU::copy_host_to_device() {
    cudaMemcpy(d_mw, this->theta_cpu->mw.data(), w_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw, this->theta_cpu->Sw.data(), w_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb, this->theta_cpu->mb.data(), b_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sb, this->theta_cpu->Sb.data(), b_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mw_sc, this->theta_cpu->mw_sc.data(), w_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw_sc, this->theta_cpu->Sw_sc.data(), w_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb_sc, this->theta_cpu->mb_sc.data(), b_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sb_sc, this->theta_cpu->Sb_sc.data(), b_sc_bytes,
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for parameters - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void ParamGPU::copy_device_to_host() {
    cudaMemcpy(this->theta_cpu->mw.data(), d_mw, w_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->Sw.data(), d_Sw, w_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->mb.data(), d_mb, b_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->Sb.data(), d_Sb, b_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->mw_sc.data(), d_mw_sc, w_sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->Sw_sc.data(), d_Sw_sc, w_sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->mb_sc.data(), d_mb_sc, b_sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->theta_cpu->Sb_sc.data(), d_Sb_sc, b_sc_bytes,
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for parameters - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

ParamGPU::~ParamGPU() {
    cudaFree(d_mw);
    cudaFree(d_Sw);
    cudaFree(d_mb);
    cudaFree(d_Sb);
    cudaFree(d_mw_sc);
    cudaFree(d_Sw_sc);
    cudaFree(d_mb_sc);
    cudaFree(d_Sb_sc);
}

IndexGPU::IndexGPU() {
    this->d_Fmwa_1 = nullptr;
    this->d_Fmwa_2 = nullptr;
    this->d_FCzwa_1 = nullptr;
    this->d_FCzwa_2 = nullptr;
    this->d_Szz_ud = nullptr;
    this->d_pooling = nullptr;
    this->d_FCwz_2 = nullptr;
    this->d_Swz_ud = nullptr;
    this->d_Fmwa_2_sc = nullptr;
    this->d_FCzwa_1_sc = nullptr;
    this->d_FCzwa_2_sc = nullptr;
    this->d_Szz_ud_sc = nullptr;
}

void IndexGPU::set_values(IndexOut &idx) {
    this->Fmwa_1_bytes = idx.Fmwa_1.size() * sizeof(int);
    this->Fmwa_2_bytes = idx.Fmwa_2.size() * sizeof(int);
    this->FCzwa_1_bytes = idx.FCzwa_1.size() * sizeof(int);
    this->FCzwa_2_bytes = idx.FCzwa_2.size() * sizeof(int);
    this->Szz_ud_bytes = idx.Szz_ud.size() * sizeof(int);
    this->pooling_bytes = idx.pooling.size() * sizeof(int);
    this->FCwz_2_bytes = idx.FCwz_2.size() * sizeof(int);
    this->Swz_ud_bytes = idx.Swz_ud.size() * sizeof(int);
    this->Fmwa_2_sc_bytes = idx.Fmwa_2_sc.size() * sizeof(int);
    this->FCzwa_1_sc_bytes = idx.FCzwa_1_sc.size() * sizeof(int);
    this->FCzwa_2_sc_bytes = idx.FCzwa_2_sc.size() * sizeof(int);
    this->Szz_ud_sc_bytes = idx.Szz_ud_sc.size() * sizeof(int);
}

void IndexGPU::allocate_cuda_memory() {
    cudaMalloc(&d_Fmwa_1, Fmwa_1_bytes);
    cudaMalloc(&d_Fmwa_2, Fmwa_2_bytes);
    cudaMalloc(&d_FCzwa_1, FCzwa_1_bytes);
    cudaMalloc(&d_FCzwa_2, FCzwa_2_bytes);
    cudaMalloc(&d_Szz_ud, Szz_ud_bytes);
    cudaMalloc(&d_pooling, pooling_bytes);
    cudaMalloc(&d_FCwz_2, FCwz_2_bytes);
    cudaMalloc(&d_Swz_ud, Swz_ud_bytes);
    cudaMalloc(&d_Fmwa_2_sc, Fmwa_2_sc_bytes);
    cudaMalloc(&d_FCzwa_1_sc, FCzwa_1_sc_bytes);
    cudaMalloc(&d_FCzwa_2_sc, FCzwa_2_sc_bytes);
    cudaMalloc(&d_Szz_ud_sc, Szz_ud_sc_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg = "Failed to allocate CUDA memory for indices\n";
        std::cerr << error << ": " << err_msg;
    }
}

void IndexGPU::copy_host_to_device(IndexOut &idx) {
    cudaMemcpy(d_Fmwa_1, idx.Fmwa_1.data(), Fmwa_1_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Fmwa_2, idx.Fmwa_2.data(), Fmwa_2_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_FCzwa_1, idx.FCzwa_1.data(), FCzwa_1_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_FCzwa_2, idx.FCzwa_2.data(), FCzwa_2_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Szz_ud, idx.Szz_ud.data(), Szz_ud_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_pooling, idx.pooling.data(), pooling_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_FCwz_2, idx.FCwz_2.data(), FCwz_2_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Swz_ud, idx.Swz_ud.data(), Swz_ud_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Fmwa_2_sc, idx.Fmwa_2_sc.data(), Fmwa_2_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_FCzwa_1_sc, idx.FCzwa_1_sc.data(), FCzwa_1_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_FCzwa_2_sc, idx.FCzwa_2_sc.data(), FCzwa_2_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Szz_ud_sc, idx.Szz_ud_sc.data(), Szz_ud_sc_bytes,
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for indices - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

IndexGPU::~IndexGPU() {
    cudaFree(d_Fmwa_1);
    cudaFree(d_Fmwa_2);
    cudaFree(d_FCzwa_1);
    cudaFree(d_FCzwa_2);
    cudaFree(d_Szz_ud);
    cudaFree(d_pooling);
    cudaFree(d_FCwz_2);
    cudaFree(d_Swz_ud);
    cudaFree(d_Fmwa_2_sc);
    cudaFree(d_FCzwa_1_sc);
    cudaFree(d_FCzwa_2_sc);
    cudaFree(d_Szz_ud_sc);
}

//////////////////////////////
// DELTA STATE
//////////////////////////////
DeltaStateGPU::DeltaStateGPU() {
    this->d_delta_mz = nullptr;
    this->d_delta_Sz = nullptr;
    this->d_delta_mdsc = nullptr;
    this->d_delta_Sdsc = nullptr;
    this->d_delta_msc = nullptr;
    this->d_delta_Ssc = nullptr;
    this->d_delta_mzsc = nullptr;
    this->d_delta_Szsc = nullptr;
    this->d_dummy_m = nullptr;
    this->d_dummy_S = nullptr;
    this->d_delta_m = nullptr;
    this->d_delta_S = nullptr;
    this->d_delta_mx = nullptr;
    this->d_delta_Sx = nullptr;
}

void DeltaStateGPU::set_values(int s, int sc, int dsc, int max_n_s) {
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

    this->s_bytes = s * sizeof(float);
    this->sc_bytes = sc * sizeof(float);
    this->dsc_bytes = dsc * sizeof(float);
    this->max_n_s_bytes = max_n_s * sizeof(float);
}

void DeltaStateGPU::set_delta_softmax(int n) {
    this->delta_mu_y_check.resize(n, 0);
    this->delta_var_y_check.resize(n, 0);
    this->delta_mu_zy_check.resize(n, 0);
    this->delta_var_zy_check.resize(n, 0);
    this->softmax_bytes = n * sizeof(float);
}

void DeltaStateGPU::allocate_cuda_memory() {
    cudaMalloc(&d_delta_mz, max_n_s_bytes);
    cudaMalloc(&d_delta_Sz, max_n_s_bytes);
    cudaMalloc(&d_delta_mdsc, dsc_bytes);
    cudaMalloc(&d_delta_Sdsc, dsc_bytes);
    cudaMalloc(&d_delta_msc, sc_bytes);
    cudaMalloc(&d_delta_Ssc, sc_bytes);
    cudaMalloc(&d_delta_mzsc, max_n_s_bytes);
    cudaMalloc(&d_delta_Szsc, max_n_s_bytes);
    cudaMalloc(&d_dummy_m, max_n_s_bytes);
    cudaMalloc(&d_dummy_S, max_n_s_bytes);
    cudaMalloc(&d_delta_m, s_bytes);
    cudaMalloc(&d_delta_S, s_bytes);
    cudaMalloc(&d_delta_mx, dsc_bytes);
    cudaMalloc(&d_delta_Sx, dsc_bytes);

    if (this->softmax_bytes > 0) {
        cudaMalloc(&this->delta_mu_y_check, this->softmax_bytes);
        cudaMalloc(&this->delta_var_y_check, this->softmax_bytes);
        cudaMalloc(&this->delta_mu_zy_check, this->softmax_bytes);
        cudaMalloc(&this->delta_var_zy_check, this->softmax_bytes);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for delta state\n";
        std::cerr << error << ": " << err_msg;
    }
}

void DeltaStateGPU::copy_host_to_device() {
    cudaMemcpy(d_delta_mz, delta_mz.data(), max_n_s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sz, delta_Sz.data(), max_n_s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mdsc, delta_mdsc.data(), dsc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sdsc, delta_Sdsc.data(), dsc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_msc, delta_msc.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Ssc, delta_Ssc.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mzsc, delta_mzsc.data(), max_n_s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Szsc, delta_Szsc.data(), max_n_s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dummy_m, dummy_m.data(), max_n_s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_dummy_S, dummy_S.data(), max_n_s_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_m, delta_m.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_S, delta_S.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mx, delta_mx.data(), dsc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sx, delta_Sx.data(), dsc_bytes, cudaMemcpyHostToDevice);

    if (this->softmax_bytes > 0) {
        cudaMemcpy(this->d_delta_mu_y_check, this->d_delta_mu_y_check.data(),
                   this->softmax_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_delta_var_y_check, this->d_delta_var_y_check.data(),
                   this->softmax_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_delta_mu_zy_check, this->d_delta_mu_zy_check.data(),
                   this->softmax_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_delta_var_zy_check,
                   this->d_delta_var_zy_check.data(), this->softmax_bytes,
                   cudaMemcpyHostToDevice);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for delta state - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void DeltaStateGPU::copy_device_to_host() {
    cudaMemcpy(delta_mz.data(), d_delta_mz, max_n_s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sz.data(), d_delta_Sz, max_n_s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_mdsc.data(), d_delta_mdsc, dsc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sdsc.data(), d_delta_Sdsc, dsc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_msc.data(), d_delta_msc, sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Ssc.data(), d_delta_Ssc, sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_mzsc.data(), d_delta_mzsc, max_n_s_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Szsc.data(), d_delta_Szsc, max_n_s_bytes,
               cudaMemcpyDeviceToHost);

    cudaMemcpy(delta_m.data(), d_delta_m, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_S.data(), d_delta_S, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_mx.data(), d_delta_mx, dsc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sx.data(), d_delta_Sx, dsc_bytes, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for delta states - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

DeltaStateGPU::~DeltaStateGPU() {
    cudaFree(d_delta_mz);
    cudaFree(d_delta_Sz);
    cudaFree(d_delta_mdsc);
    cudaFree(d_delta_Sdsc);
    cudaFree(d_delta_msc);
    cudaFree(d_delta_Ssc);
    cudaFree(d_delta_mzsc);
    cudaFree(d_delta_Szsc);
    cudaFree(d_dummy_m);
    cudaFree(d_dummy_S);
    cudaFree(d_delta_m);
    cudaFree(d_delta_S);
    cudaFree(d_delta_mx);
    cudaFree(d_delta_Sx);
}

//////////////////////////////
// DELTA PARAM
//////////////////////////////
DeltaParamGPU::DeltaParamGPU() {
    this->d_delta_mw = nullptr;
    this->d_delta_Sw = nullptr;
    this->d_delta_mb = nullptr;
    this->d_delta_Sb = nullptr;
    this->d_delta_mw_sc = nullptr;
    this->d_delta_Sw_sc = nullptr;
    this->d_delta_mb_sc = nullptr;
    this->d_delta_Sb_sc = nullptr;
}

void DeltaParamGPU::set_values(int w, int b, int w_sc, int b_sc) {
    this->delta_mw.resize(w, 0);
    this->delta_Sw.resize(w, 0);
    this->delta_mb.resize(b, 0);
    this->delta_Sb.resize(b, 0);
    this->delta_mw_sc.resize(w_sc, 0);
    this->delta_Sw_sc.resize(w_sc, 0);
    this->delta_mb_sc.resize(b_sc, 0);
    this->delta_Sb_sc.resize(b_sc, 0);

    this->w_bytes = w * sizeof(float);
    this->b_bytes = b * sizeof(float);
    this->w_sc_bytes = w_sc * sizeof(float);
    this->b_sc_bytes = b_sc * sizeof(float);
}

void DeltaParamGPU::allocate_cuda_memory() {
    cudaMalloc(&d_delta_mw, w_bytes);
    cudaMalloc(&d_delta_Sw, w_bytes);
    cudaMalloc(&d_delta_mb, b_bytes);
    cudaMalloc(&d_delta_Sb, b_bytes);
    cudaMalloc(&d_delta_mw_sc, w_sc_bytes);
    cudaMalloc(&d_delta_Sw_sc, w_sc_bytes);
    cudaMalloc(&d_delta_mb_sc, b_sc_bytes);
    cudaMalloc(&d_delta_Sb_sc, b_sc_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for delta parameters - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void DeltaParamGPU::copy_host_to_device() {
    cudaMemcpy(d_delta_mw, delta_mw.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sw, delta_Sw.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mb, delta_mb.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sb, delta_Sb.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mw_sc, delta_mw_sc.data(), w_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sw_sc, delta_Sw_sc.data(), w_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_mb_sc, delta_mb_sc.data(), b_sc_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta_Sb_sc, delta_Sb_sc.data(), b_sc_bytes,
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for delta parameters\n";
        std::cerr << error << ": " << err_msg;
    }
}

void DeltaParamGPU::copy_device_to_host() {
    cudaMemcpy(delta_mw.data(), d_delta_mw, w_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sw.data(), d_delta_Sw, w_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_mb.data(), d_delta_mb, b_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sb.data(), d_delta_Sb, b_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_mw_sc.data(), d_delta_mw_sc, w_sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sw_sc.data(), d_delta_Sw_sc, w_sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_mb_sc.data(), d_delta_mb_sc, b_sc_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(delta_Sb_sc.data(), d_delta_Sb_sc, b_sc_bytes,
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for delta parameters - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

DeltaParamGPU::~DeltaParamGPU() {
    cudaFree(d_delta_mw);
    cudaFree(d_delta_Sw);
    cudaFree(d_delta_mb);
    cudaFree(d_delta_Sb);
    cudaFree(d_delta_mw_sc);
    cudaFree(d_delta_Sw_sc);
    cudaFree(d_delta_mb_sc);
    cudaFree(d_delta_Sb_sc);
}

///////////////////////////////
// INPUT
//////////////////////////////
InputGPU::InputGPU() {}

void InputGPU::set_values(Network &net) {
    this->id_bytes =
        net.batch_size * net.nodes.front() * net.input_seq_len * sizeof(float);
    if (net.is_full_cov) {
        this->id_f_bytes = (net.n_x * (net.n_x + 1)) / 2 * net.batch_size *
                           net.input_seq_len * sizeof(float);
    } else {
        this->id_f_bytes = 0;
    }

    this->d_x_batch = nullptr;
    this->d_Sx_batch = nullptr;
    this->d_Sx_f_batch = nullptr;
}

void InputGPU::allocate_cuda_memory() {
    cudaMalloc(&d_x_batch, id_bytes);
    cudaMalloc(&d_Sx_batch, id_bytes);
    if (id_f_bytes > 0) {
        cudaMalloc(&d_Sx_f_batch, id_f_bytes);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for inputs - data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void InputGPU::copy_host_to_device(std::vector<float> &x_batch,
                                   std::vector<float> &Sx_batch,
                                   std::vector<float> &Sx_f_batch) {
    cudaMemcpy(d_x_batch, x_batch.data(), id_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sx_batch, Sx_batch.data(), id_bytes, cudaMemcpyHostToDevice);
    if (id_f_bytes > 0) {
        cudaMemcpy(d_Sx_f_batch, Sx_f_batch.data(), id_f_bytes,
                   cudaMemcpyHostToDevice);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for inputs - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void InputGPU::copy_device_to_host(std::vector<float> &x_batch,
                                   std::vector<float> &Sx_batch,
                                   std::vector<float> &Sx_f_batch) {
    cudaMemcpy(x_batch.data(), d_x_batch, id_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sx_batch.data(), d_Sx_batch, id_bytes, cudaMemcpyDeviceToHost);
    if (id_f_bytes > 0) {
        cudaMemcpy(Sx_f_batch.data(), d_Sx_f_batch, id_f_bytes,
                   cudaMemcpyDeviceToHost);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for inputs - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

InputGPU::~InputGPU() {
    cudaFree(d_x_batch);
    cudaFree(d_Sx_batch);
    cudaFree(d_Sx_f_batch);
}

///////////////////////////////
// CONNECTOR INPUT GPU
//////////////////////////////
ConnectorInputGPU::ConnectorInputGPU(){};
ConnectorInputGPU::~ConnectorInputGPU() {
    cudaFree(d_ma);
    cudaFree(d_Sa);
    cudaFree(d_mz);
    cudaFree(d_Sz);
    cudaFree(d_J);
};
void ConnectorInputGPU::set_values(int input_size) {
    this->num_input_bytes = input_size * sizeof(float);
}

void ConnectorInputGPU::allocate_cuda_memory() {
    cudaMalloc(&d_ma, num_input_bytes);
    cudaMalloc(&d_Sa, num_input_bytes);
    cudaMalloc(&d_mz, num_input_bytes);
    cudaMalloc(&d_Sz, num_input_bytes);
    cudaMalloc(&d_J, num_input_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for  connected inputs - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void ConnectorInputGPU::copy_host_to_device(std::vector<float> &ma,
                                            std::vector<float> &Sa,
                                            std::vector<float> &mz,
                                            std::vector<float> &Sz,
                                            std::vector<float> &J) {
    cudaMemcpy(this->d_ma, ma.data(), num_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_Sa, Sa.data(), num_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mz, mz.data(), num_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_Sz, Sz.data(), num_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_J, J.data(), num_input_bytes, cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        if (error != cudaSuccess) {
            std::string err_msg =
                "Failed to make data transfer to device for connected inputs - "
                "data_transfer.cu\n";
            std::cerr << error << ": " << err_msg;
        }
    }
}

void ConnectorInputGPU::copy_device_to_host(std::vector<float> &ma,
                                            std::vector<float> &Sa,
                                            std::vector<float> &mz,
                                            std::vector<float> &Sz,
                                            std::vector<float> &J) {
    cudaMemcpy(ma.data(), this->d_ma, num_input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sa.data(), this->d_Sa, num_input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(mz.data(), this->d_mz, num_input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Sz.data(), this->d_Sz, num_input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(J.data(), this->d_J, num_input_bytes, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        if (error != cudaSuccess) {
            std::string err_msg =
                "Failed to make data transfer to host for connected inputs - "
                "data_transfer.cu\n";
            std::cerr << error << ": " << err_msg;
        }
    }
}

///////////////////////////////
// OUTPUT
//////////////////////////////
ObsGPU::ObsGPU(){};
void ObsGPU::set_values(int ny, int nye, int B) {
    this->od_bytes = B * ny * sizeof(float);
    this->ode_bytes = B * nye * sizeof(int);

    this->d_y_batch = nullptr;
    this->d_V_batch = nullptr;
    this->d_idx_ud_batch = nullptr;
}

void ObsGPU::allocate_cuda_memory() {
    cudaMalloc(&d_y_batch, od_bytes);
    cudaMalloc(&d_idx_ud_batch, ode_bytes);
    cudaMalloc(&d_V_batch, od_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for outputs - data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void ObsGPU::copy_host_to_device(std::vector<float> &y_batch,
                                 std::vector<int> &idx_ud_batch,
                                 std::vector<float> &V_batch) {
    cudaMemcpy(d_y_batch, y_batch.data(), od_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx_ud_batch, idx_ud_batch.data(), ode_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_batch, V_batch.data(), od_bytes, cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for outputs - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

void ObsGPU::copy_device_to_host(std::vector<float> &y_batch,
                                 std::vector<int> &idx_ud_batch,
                                 std::vector<float> &V_batch) {
    cudaMemcpy(y_batch.data(), d_y_batch, od_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(idx_ud_batch.data(), d_idx_ud_batch, ode_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(V_batch.data(), d_V_batch, od_bytes, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to host for outputs - "
            "data_transfer.cu\n";
        std::cerr << error << ": " << err_msg;
    }
}

ObsGPU::~ObsGPU() {
    cudaFree(d_y_batch);
    cudaFree(d_idx_ud_batch);
    cudaFree(d_V_batch);
}
