///////////////////////////////////////////////////////////////////////////////
// File:         fc_layer_cpu.h
// Description:  Header file for CPU version offully-connected layer
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 12, 2023
// Updated:      April 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma onnce
#include <thread>
#include <vector>
void fc_mean_cpu(std::vector<float> &mw, std::vector<float> &mb,
                 std::vector<float> &ma, int w_pos, int b_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k, std::vector<float> &mz);

void fc_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                std::vector<float> &Sb, std::vector<float> &ma,
                std::vector<float> &Sa, int w_pos, int b_pos, int z_pos_in,
                int z_pos_out, int m, int n, int k, std::vector<float> &Sz);

void fc_full_cov_cpu(std::vector<float> &mw, std::vector<float> &Sa_f,
                     int w_pos, int no, int ni, int B,
                     std::vector<float> &Sz_fp);

void fc_full_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                     std::vector<float> &Sb, std::vector<float> &ma,
                     std::vector<float> &Sa, std::vector<float> &Sz_fp,
                     int w_pos, int b_pos, int z_pos_in, int z_pos_out, int no,
                     int ni, int B, std::vector<float> &Sz,
                     std::vector<float> &Sz_f);

void fc_delta_mz(std::vector<float> &mw, std::vector<float> &Sz,
                 std::vector<float> &J, std::vector<float> &delta_m, int w_pos,
                 int z_pos_in, int z_pos_out, int ni, int no, int B,
                 std::vector<float> &delta_mz);

void fc_delta_Sz(std::vector<float> &mw, std::vector<float> &Sz,
                 std::vector<float> &J, std::vector<float> &delta_S, int w_pos,
                 int z_pos_in, int z_pos_out, int ni, int no, int B,
                 std::vector<float> &delta_Sz);

void fc_delta_mw(std::vector<float> &Sw, std::vector<float> &ma,
                 std::vector<float> &delta_m, int w_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_mw);

void fc_delta_Sw(std::vector<float> &Sw, std::vector<float> &ma,
                 std::vector<float> &delta_S, int w_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_Sw);

void fc_delta_mb(std::vector<float> &C_bz, std::vector<float> &delta_m,
                 int b_pos, int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_mb);

void fc_delta_Sb(std::vector<float> &C_bz, std::vector<float> &delta_S,
                 int b_pos, int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_Sb);

void fc_full_var_multithreading(
    std::vector<float> &mw, std::vector<float> &Sw, std::vector<float> &Sb,
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &Sz_fp,
    int w_pos, int b_pos, int z_pos_in, int z_pos_out, int no, int ni, int B,
    unsigned int NUM_THREADS, std::vector<float> &Sz, std::vector<float> &Sz_f);

void fc_full_cov_multithreading(std::vector<float> &mw,
                                std::vector<float> &Sa_f, int w_pos, int no,
                                int ni, int B, unsigned int NUM_THREADS,
                                std::vector<float> &Sz_fp);

void fc_mean_var_multithreading(std::vector<float> &mw, std::vector<float> &Sw,
                                std::vector<float> &mb, std::vector<float> &Sb,
                                std::vector<float> &ma, std::vector<float> &Sa,
                                int w_pos, int b_pos, int z_pos_in,
                                int z_pos_out, int m, int n, int k,
                                unsigned int NUM_THREADS,
                                std::vector<float> &mz, std::vector<float> &Sz);

void fc_delta_mzSz_multithreading(std::vector<float> &mw,
                                  std::vector<float> &Sz, std::vector<float> &J,
                                  std::vector<float> &delta_m,
                                  std::vector<float> &delta_S, int w_pos,
                                  int z_pos_in, int z_pos_out, int ni, int no,
                                  int B, unsigned int NUM_THREADS,
                                  std::vector<float> &delta_mz,
                                  std::vector<float> &delta_Sz);

void fc_delta_w_multithreading(std::vector<float> &Sw, std::vector<float> &ma,
                               std::vector<float> &delta_m,
                               std::vector<float> &delta_S, int w_pos,
                               int z_pos_in, int z_pos_out, int m, int n, int k,
                               unsigned int NUM_THREADS,
                               std::vector<float> &delta_mw,
                               std::vector<float> &delta_Sw);

void fc_delta_b_multithreading(std::vector<float> &C_bz,
                               std::vector<float> &delta_m,
                               std::vector<float> &delta_S, int b_pos,
                               int z_pos_out, int m, int n, int k,
                               unsigned int NUM_THREADS,
                               std::vector<float> &delta_mb,
                               std::vector<float> &delta_Sb);