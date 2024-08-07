///////////////////////////////////////////////////////////////////////////////
// File:         utils.cpp
// Description:  utils for TAGI package such saving and loading parameters
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 10, 2022
// Updated:      July 20, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "../include/utils.h"

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
