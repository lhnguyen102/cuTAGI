///////////////////////////////////////////////////////////////////////////////
// File:         utils.cpp
// Description:  utils for TAGI package such saving and loading parameters
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 10, 2022
// Updated:      April 10, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/utils.h"

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
void save_idx(std::string &idx_path, IndexOut &idx) {
    // Save indices
    std::string FCzwa_1_path = idx_path + "1_FCzwa_1.csv";
    write_csv(FCzwa_1_path, idx.FCzwa_1);

    std::string FCzwa_2_path = idx_path + "2_FCzwa_2.csv";
    write_csv(FCzwa_2_path, idx.FCzwa_2);

    std::string Fmwa_1_path = idx_path + "3_Fmwa_1.csv";
    write_csv(Fmwa_1_path, idx.Fmwa_1);

    std::string Fmwa_2_path = idx_path + "4_Fmwa_2.csv";
    write_csv(Fmwa_2_path, idx.Fmwa_2);

    std::string pooling_path = idx_path + "5_pooling.csv";
    write_csv(pooling_path, idx.pooling);

    std::string Szz_ud_path = idx_path + "6_Szz_ud.csv";
    write_csv(Szz_ud_path, idx.Szz_ud);

    std::string FCzwa_1_sc_path = idx_path + "7_FCzwa_1_sc.csv";
    write_csv(FCzwa_1_sc_path, idx.FCzwa_1_sc);

    std::string FCzwa_2_sc_path = idx_path + "8_FCzwa_2_sc.csv";
    write_csv(FCzwa_2_sc_path, idx.FCzwa_2_sc);

    std::string Fmwa_2_sc_path = idx_path + "9_Fmwa_2_sc.csv";
    write_csv(Fmwa_2_sc_path, idx.Fmwa_2_sc);

    std::string Szz_ud_sc_path = idx_path + "10_Szz_ud_sc.csv";
    write_csv(Szz_ud_sc_path, idx.Szz_ud_sc);

    std::string Swz_ud_path = idx_path + "11_Swz_ud.csv";
    write_csv(Swz_ud_path, idx.Swz_ud);

    std::string FCwz_2_path = idx_path + "12_FCwz_2.csv";
    write_csv(FCwz_2_path, idx.FCwz_2);
}

void save_param(std::string &param_path, Param &theta) {
    std::string mw_path = param_path + "1_mw.csv";
    write_csv(mw_path, theta.mw);
    std::string Sw_path = param_path + "2_Sw.csv";
    write_csv(Sw_path, theta.Sw);
    std::string mb_path = param_path + "3_mb.csv";
    write_csv(mb_path, theta.mb);
    std::string Sb_path = param_path + "4_Sb.csv";
    write_csv(Sb_path, theta.Sb);

    std::string mw_sc_path = param_path + "5_mw_sc.csv";
    write_csv(mw_sc_path, theta.mw_sc);
    std::string Sw_sc_path = param_path + "6_Sw_sc.csv";
    write_csv(Sw_sc_path, theta.Sw_sc);
    std::string mb_sc_path = param_path + "7_mb_sc.csv";
    write_csv(mb_sc_path, theta.mb_sc);
    std::string Sb_sc_path = param_path + "8_Sb_sc.csv";
    write_csv(Sb_sc_path, theta.Sb_sc);
}

void load_net_param(std::string &model_name, std::string &net_name,
                    std::string &path, Param &theta)
/*Load saved parameter for network
Args:
    model_name: Name of the model
    net_name: Name of network
    path: Directory of the saved parameters
    theta: Parameters of network
*/
{
    // Common path
    std::string param_path = path + model_name + "_" + net_name + "_";

    // Path to parameters
    std::string mw_path = param_path + "mw.csv";
    std::string Sw_path = param_path + "Sw.csv";
    std::string mb_path = param_path + "mb.csv";
    std::string Sb_path = param_path + "Sb.csv";
    std::string mw_sc_path = param_path + "mw_sc.csv";
    std::string Sw_sc_path = param_path + "Sw_sc.csv";
    std::string mb_sc_path = param_path + "mb_sc.csv";
    std::string Sb_sc_path = param_path + "Sb_sc.csv";

    // Load parameters
    read_csv(mw_path, theta.mw);
    read_csv(Sw_path, theta.Sw);
    read_csv(mb_path, theta.mb);
    read_csv(Sb_path, theta.Sb);
    read_csv(mw_sc_path, theta.mw_sc);
    read_csv(Sw_sc_path, theta.Sw_sc);
    read_csv(mb_sc_path, theta.mb_sc);
    read_csv(Sb_sc_path, theta.Sb_sc);
}
void save_net_param(std::string &model_name, std::string &net_name,
                    std::string path, Param &theta)
/*Save parameters of network
Args:
    model_name: Name of the model
    net_name: Name of network
    path: Directory of the saved parameters
    theta: Parameters of network
 */
{
    // Common path
    std::string param_path = path + model_name + "_" + net_name + "_";

    // Path to parameters
    std::string mw_path = param_path + "mw.csv";
    std::string Sw_path = param_path + "Sw.csv";
    std::string mb_path = param_path + "mb.csv";
    std::string Sb_path = param_path + "Sb.csv";
    std::string mw_sc_path = param_path + "mw_sc.csv";
    std::string Sw_sc_path = param_path + "Sw_sc.csv";
    std::string mb_sc_path = param_path + "mb_sc.csv";
    std::string Sb_sc_path = param_path + "Sb_sc.csv";

    // Save parameters
    write_csv(mw_path, theta.mw);
    write_csv(Sw_path, theta.Sw);
    write_csv(mb_path, theta.mb);
    write_csv(Sb_path, theta.Sb);
    write_csv(mw_sc_path, theta.mw_sc);
    write_csv(Sw_sc_path, theta.Sw_sc);
    write_csv(mb_sc_path, theta.mb_sc);
    write_csv(Sb_sc_path, theta.Sb_sc);
}

void save_net_prop(std::string &param_path, std::string &idx_path, Param &theta,
                   IndexOut &idx) {
    // Save parameters to csv files
    save_param(param_path, theta);

    // Save indices
    save_idx(idx_path, idx);
}

void save_autoencoder_net_prop(Param &theta_e, Param &theta_d, IndexOut &idx_e,
                               IndexOut &idx_d, std::string &debug_path) {
    // Encoder
    std::string param_path_e = debug_path + "/saved_param_enc/";
    std::string idx_path_e = debug_path + "/saved_idx_enc/";
    save_net_prop(param_path_e, idx_path_e, theta_e, idx_e);

    // Decoder
    std::string param_path_d = debug_path + "/saved_param_dec/";
    std::string idx_path_d = debug_path + "/saved_idx_dec/";
    save_net_prop(param_path_d, idx_path_d, theta_d, idx_d);
}