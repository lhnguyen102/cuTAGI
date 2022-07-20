///////////////////////////////////////////////////////////////////////////////
// File:         utils.cpp
// Description:  utils for TAGI package such saving and loading parameters
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 10, 2022
// Updated:      July 20, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/utils.h"

void save_hidden_states(std::string &res_path, NetState &state) {
    // Create the directory if not exists
    create_directory(res_path);

    // Common layers
    std::string mz_path = res_path + "1_mz.csv";
    write_csv(mz_path, state.mz);

    std::string Sz_path = res_path + "2_Sz.csv";
    write_csv(Sz_path, state.Sz);

    std::string ma_path = res_path + "3_ma.csv";
    write_csv(ma_path, state.ma);

    std::string Sa_path = res_path + "4_Sa.csv";
    write_csv(Sa_path, state.Sa);

    std::string J_path = res_path + "5_J.csv";
    write_csv(J_path, state.J);

    // Resnet layer
    std::string msc_path = res_path + "6_msc.csv";
    write_csv(msc_path, state.msc);

    std::string Ssc_path = res_path + "7_Ssc.csv";
    write_csv(Ssc_path, state.Ssc);

    std::string mdsc_path = res_path + "8_mdsc.csv";
    write_csv(mdsc_path, state.mdsc);

    std::string Sdsc_path = res_path + "9_Sdsc.csv";
    write_csv(Sdsc_path, state.Sdsc);
}

void save_error_rate(std::string &res_path, std::vector<float> &error_rate,
                     std::string &suffix) {
    std::string error_rate_path = res_path + "error_rate_" + suffix + ".csv";
    write_csv(error_rate_path, error_rate);
}

void save_generated_images(std::string &res_path, std::vector<float> &imgs,
                           std::string &suffix) {
    /*Save outputs of neural network.*/
    std::string image_path = res_path + "generated_images_" + suffix + ".csv";
    write_csv(image_path, imgs);
}

void save_predictions(std::string &res_path, std::vector<float> &ma,
                      std::vector<float> &sa, std::string &suffix) {
    /*Save images that generated from neural network.*/
    std::string my_path = res_path + "y_" + suffix + ".csv";
    std::string Sy_path = res_path + "sy_" + suffix + ".csv";
    write_csv(my_path, ma);
    write_csv(Sy_path, sa);
}

void save_derivatives(std::string &res_path, std::vector<float> &md_layer,
                      std::vector<float> &Sd_layer, std::string &suffix) {
    /*Save images that generated from neural network.*/
    std::string my_path = res_path + "md_" + suffix + ".csv";
    std::string Sy_path = res_path + "Sd_" + suffix + ".csv";
    write_csv(my_path, md_layer);
    write_csv(Sy_path, Sd_layer);
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
    std::string mw_path = param_path + "1_mw.csv";
    std::string Sw_path = param_path + "2_Sw.csv";
    std::string mb_path = param_path + "3_mb.csv";
    std::string Sb_path = param_path + "4_Sb.csv";
    std::string mw_sc_path = param_path + "5_mw_sc.csv";
    std::string Sw_sc_path = param_path + "6_Sw_sc.csv";
    std::string mb_sc_path = param_path + "7_mb_sc.csv";
    std::string Sb_sc_path = param_path + "8_Sb_sc.csv";

    // Load parameters
    read_csv(mw_path, theta.mw, 1, false);
    read_csv(Sw_path, theta.Sw, 1, false);
    read_csv(mb_path, theta.mb, 1, false);
    read_csv(Sb_path, theta.Sb, 1, false);
    read_csv(mw_sc_path, theta.mw_sc, 1, false);
    read_csv(Sw_sc_path, theta.Sw_sc, 1, false);
    read_csv(mb_sc_path, theta.mb_sc, 1, false);
    read_csv(Sb_sc_path, theta.Sb_sc, 1, false);
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
    // Create directory
    create_directory(path);

    // Common path
    std::string param_path = path + model_name + "_" + net_name + "_";

    // Path to parameters
    std::string mw_path = param_path + "1_mw.csv";
    std::string Sw_path = param_path + "2_Sw.csv";
    std::string mb_path = param_path + "3_mb.csv";
    std::string Sb_path = param_path + "4_Sb.csv";
    std::string mw_sc_path = param_path + "5_mw_sc.csv";
    std::string Sw_sc_path = param_path + "6_Sw_sc.csv";
    std::string mb_sc_path = param_path + "7_mb_sc.csv";
    std::string Sb_sc_path = param_path + "8_Sb_sc.csv";

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