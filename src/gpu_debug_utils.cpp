///////////////////////////////////////////////////////////////////////////////
// File:         gpu_debug_utils.cpp
// Description:  Debug utils for GPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 22, 2022
// Updated:      May 22, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/gpu_debug_utils.h"

void save_inference_results(std::string &res_path, DeltaStateGPU &d_state_gpu,
                            Param &theta) {
    // Save results
    std::string delta_m_path = res_path + "1_delta_m";
    write_csv(delta_m_path, d_state_gpu.delta_m);

    std::string delta_S_path = res_path + "2_delta_S";
    write_csv(delta_S_path, d_state_gpu.delta_S);

    std::string delta_mx_path = res_path + "3_delta_mx";
    write_csv(delta_mx_path, d_state_gpu.delta_mx);

    std::string delta_Sx_path = res_path + "4_delta_Sx";
    write_csv(delta_Sx_path, d_state_gpu.delta_Sx);

    std::string mw_path = res_path + "5_mw.csv";
    write_csv(mw_path, theta.mw);
    std::string Sw_path = res_path + "6_Sw.csv";
    write_csv(Sw_path, theta.Sw);
    std::string mb_path = res_path + "7_mb.csv";
    write_csv(mb_path, theta.mb);
    std::string Sb_path = res_path + "8_Sb.csv";
    write_csv(Sb_path, theta.Sb);

    std::string mw_sc_path = res_path + "9_mw_sc.csv";
    write_csv(mw_sc_path, theta.mw_sc);
    std::string Sw_sc_path = res_path + "10_Sw_sc.csv";
    write_csv(Sw_sc_path, theta.Sw_sc);
    std::string mb_sc_path = res_path + "11_mb_sc.csv";
    write_csv(mb_sc_path, theta.mb_sc);
    std::string Sb_sc_path = res_path + "12_Sb_sc.csv";
    write_csv(Sb_sc_path, theta.Sb_sc);
}

void save_delta_param(std::string &res_path, DeltaParamGPU &d_param) {
    // Create the directory if not exists
    create_directory(res_path);

    // Common layers
    std::string d_mw_path = res_path + "1_d_mw.csv";
    write_csv(d_mw_path, d_param.delta_mw);

    std::string d_Sw_path = res_path + "2_d_Sw.csv";
    write_csv(d_Sw_path, d_param.delta_Sw);

    std::string d_mb_path = res_path + "3_d_mb.csv";
    write_csv(d_mb_path, d_param.delta_mb);

    std::string d_Sb_path = res_path + "4_d_Sb.csv";
    write_csv(d_Sb_path, d_param.delta_Sb);

    // Resnet layer
    std::string d_mw_sc_path = res_path + "5_d_mw_sc.csv";
    write_csv(d_mw_sc_path, d_param.delta_mw_sc);

    std::string d_Sw_sc_path = res_path + "6_d_Sw_sc.csv";
    write_csv(d_Sw_sc_path, d_param.delta_Sw_sc);

    std::string d_mb_sc_path = res_path + "7_d_mb_sc.csv";
    write_csv(d_mb_sc_path, d_param.delta_mb_sc);

    std::string d_Sb_sc_path = res_path + "8_d_Sb_sc.csv";
    write_csv(d_Sb_sc_path, d_param.delta_Sb_sc);
}
