///////////////////////////////////////////////////////////////////////////////
// File:         data_transfer.cu
// Description:  Data transfer between CPU and GPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2022
// Updated:      June 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/data_transfer.cuh"

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
}
void StateGPU::set_values(NetState &state, Network &net) {
    this->s_bytes = state.mz.size() * sizeof(float);
    this->sc_bytes = state.msc.size() * sizeof(float);
    this->dsc_bytes = state.mdsc.size() * sizeof(float);
    this->ra_bytes = state.mra.size() * sizeof(float);
    if (net.is_full_cov) {
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

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for hidden states - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void StateGPU::copy_host_to_device(NetState &state) {
    // Initialize normalization parameters
    cudaMemcpy(d_mz, state.mz.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sz, state.Sz.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma, state.ma.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa, state.Sa.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, state.J.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msc, state.msc.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ssc, state.Ssc.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mdsc, state.mdsc.data(), dsc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sdsc, state.Sdsc.data(), dsc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mra, state.mra.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sra, state.Sra.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mra_prev, mra_prev.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sra_prev, Sra_prev.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms, ms.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ss, Ss.data(), ra_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SsTmp, SsTmp.data(), ra_bytes, cudaMemcpyHostToDevice);
    if (max_full_cov_bytes > 0) {
        cudaMemcpy(d_Sz_f, state.Sz_f.data(), max_full_cov_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sa_f, state.Sa_f.data(), max_full_cov_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sz_fp, state.Sz_fp.data(), max_full_cov_bytes,
                   cudaMemcpyHostToDevice);
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to device for hidden states - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void StateGPU::copy_device_to_host(NetState &state) {
    cudaMemcpy(state.mz.data(), d_mz, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.Sz.data(), d_Sz, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.ma.data(), d_ma, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.Sa.data(), d_Sa, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.J.data(), d_J, s_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.msc.data(), d_msc, sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.Ssc.data(), d_Ssc, sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.mdsc.data(), d_mdsc, dsc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.Sdsc.data(), d_Sdsc, dsc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.mra.data(), d_mra, ra_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(state.Sra.data(), d_Sra, ra_bytes, cudaMemcpyDeviceToHost);
    if (max_full_cov_bytes > 0) {
        cudaMemcpy(state.Sz_f.data(), d_Sz_f, max_full_cov_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(state.Sa_f.data(), d_Sa_f, max_full_cov_bytes,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(state.Sz_fp.data(), d_Sz_fp, max_full_cov_bytes,
                   cudaMemcpyDeviceToHost);
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

void ParamGPU::set_values(int w, int b, int w_sc, int b_sc) {
    this->w_bytes = w * sizeof(float);
    this->b_bytes = b * sizeof(float);
    this->w_sc_bytes = w_sc * sizeof(float);
    this->b_sc_bytes = b_sc * sizeof(float);
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

void ParamGPU::copy_host_to_device(Param &theta) {
    cudaMemcpy(d_mw, theta.mw.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw, theta.Sw.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb, theta.mb.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sb, theta.Sb.data(), b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mw_sc, theta.mw_sc.data(), w_sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sw_sc, theta.Sw_sc.data(), w_sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb_sc, theta.mb_sc.data(), b_sc_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sb_sc, theta.Sb_sc.data(), b_sc_bytes, cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for parameters - "
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

void ParamGPU::copy_device_to_host(Param &theta) {
    cudaMemcpy(theta.mw.data(), d_mw, w_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.Sw.data(), d_Sw, w_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.mb.data(), d_mb, b_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.Sb.data(), d_Sb, b_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.mw_sc.data(), d_mw_sc, w_sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.Sw_sc.data(), d_Sw_sc, w_sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.mb_sc.data(), d_mb_sc, b_sc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(theta.Sb_sc.data(), d_Sb_sc, b_sc_bytes, cudaMemcpyDeviceToHost);

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
        std::string err_msg = "Failed to allocate CUDA memory for indices";
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

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg = "Failed to allocate CUDA memory for delta state";
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

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data transfer to device for delta state - "
            "data_transfer.cu";
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
            "data_transfer.cu";
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
            "data_transfer.cu";
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
            "Failed to make data transfer to device for delta parameters";
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
            "data_transfer.cu";
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
InputGPU::InputGPU(Network &net) {
    id_bytes = net.batch_size * net.nodes.front() * sizeof(float);
    if (net.is_full_cov) {
        id_f_bytes = (net.nodes.front() * (net.nodes.front() + 1)) / 2 *
                     net.batch_size * sizeof(float);
    } else {
        id_f_bytes = 0;
    }

    d_x_batch = nullptr;
    d_Sx_batch = nullptr;
    d_Sx_f_batch = nullptr;
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
            "Failed to allocate CUDA memory for inputs - data_transfer.cu";
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
            "data_transfer.cu";
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
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

InputGPU::~InputGPU() {
    cudaFree(d_x_batch);
    cudaFree(d_Sx_batch);
    cudaFree(d_Sx_f_batch);
}

///////////////////////////////
// OUTPUT
//////////////////////////////
ObsGPU::ObsGPU(int ny, int nye, int B) {
    od_bytes = B * ny * sizeof(float);
    ode_bytes = B * nye * sizeof(int);

    d_y_batch = nullptr;
    d_V_batch = nullptr;
    d_idx_ud_batch = nullptr;
}

void ObsGPU::allocate_cuda_memory() {
    cudaMalloc(&d_y_batch, od_bytes);
    cudaMalloc(&d_idx_ud_batch, ode_bytes);
    cudaMalloc(&d_V_batch, od_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for outputs - data_transfer.cu";
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
            "data_transfer.cu";
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
            "data_transfer.cu";
        std::cerr << error << ": " << err_msg;
    }
}

ObsGPU::~ObsGPU() {
    cudaFree(d_y_batch);
    cudaFree(d_idx_ud_batch);
    cudaFree(d_V_batch);
}
