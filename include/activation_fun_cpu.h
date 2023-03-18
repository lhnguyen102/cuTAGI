///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun_cpu.h
// Description:  Header file for activation functions (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 11, 2022
// Updated:      March 18, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "common.h"
#include "struct_var.h"

void remax_cpu_v2(std::vector<float> &mz, std::vector<float> &Sz,
                  std::vector<float> &mu_m, std::vector<float> &var_m,
                  std::vector<float> &J_m, std::vector<float> &mu_log,
                  std::vector<float> &var_log, std::vector<float> &mu_sum,
                  std::vector<float> &var_sum, std::vector<float> &mu_logsum,
                  std::vector<float> &var_logsum,
                  std::vector<float> &cov_log_logsum, std::vector<float> &ma,
                  std::vector<float> &Sa, int z_pos, int z_remax_pos,
                  int z_sum_remax_pos, int no, int B, float omega_tol);

void compute_cov_m_a_check_cpu(std::vector<float> &var_log,
                               std::vector<float> &cov_log_logsum,
                               std::vector<float> &mu_a, int no, int B,
                               std::vector<float> &cov_m_a_check);

void compute_cov_m_a_cpu(std::vector<float> &cov_m_a_check,
                         std::vector<float> &mu_a, std::vector<float> &var_m,
                         std::vector<float> &var_z, std::vector<float> &J_m,
                         int z_pos, int no, int B, std::vector<float> &cov_a_m);

void no_act_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                         int zpos, int n, std::vector<float> &ma,
                         std::vector<float> &J, std::vector<float> &Sa);

void tanh_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa);

void sigmoid_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int n, std::vector<float> &ma,
                          std::vector<float> &J, std::vector<float> &Sa);

void relu_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa);

void softplus_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                           int zpos, int n, std::vector<float> &ma,
                           std::vector<float> &J, std::vector<float> &Sa);

void leakyrelu_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                            float alpha, int zpos, int n,
                            std::vector<float> &ma, std::vector<float> &J,
                            std::vector<float> &Sa);

void mixture_relu_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                      float omega_tol, int zpos, int z_pos, int start_idx,
                      int end_idx, std::vector<float> &ma,
                      std::vector<float> &J, std::vector<float> &Sa);

void mixture_tanh_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                      float omega_tol, int zpos, int start_idx, int end_idx,
                      std::vector<float> &ma, std::vector<float> &J,
                      std::vector<float> &Sa);

void mixture_sigmoid_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                         float omega_tol, int zpos, int start_idx, int end_idx,
                         std::vector<float> &ma, std::vector<float> &J,
                         std::vector<float> &Sa);

void exp_fun_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                 std::vector<float> &ma, std::vector<float> &Sa,
                 std::vector<float> &Cza);

void act_full_cov(std::vector<float> &Sz_f, std::vector<float> &J, int no,
                  int B, int z_pos_out, std::vector<float> &Sa_f);

void no_act_full_cov(std::vector<float> &Sz_f, int no, int B,
                     std::vector<float> &Sa_f);

void no_act_mean_var_multithreading(std::vector<float> &mz,
                                    std::vector<float> &Sz, int z_pos, int n,
                                    unsigned int NUM_THREADS,
                                    std::vector<float> &ma,
                                    std::vector<float> &J,
                                    std::vector<float> &Sa);

void tanh_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  unsigned int NUM_THREADS,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa);

void sigmoid_mean_var_multithreading(std::vector<float> &mz,
                                     std::vector<float> &Sz, int z_pos, int n,
                                     unsigned int NUM_THREADS,
                                     std::vector<float> &ma,
                                     std::vector<float> &J,
                                     std::vector<float> &Sa);

void no_act_full_cov_multithreading(std::vector<float> &Sz_f, int no, int B,
                                    unsigned int NUM_THREADS,
                                    std::vector<float> &Sa_f);

void relu_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  unsigned int NUM_THREADS,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa);

void softplus_mean_var_multithreading(std::vector<float> &mz,
                                      std::vector<float> &Sz, int z_pos, int n,
                                      unsigned int NUM_THREADS,
                                      std::vector<float> &ma,
                                      std::vector<float> &J,
                                      std::vector<float> &Sa);

void leakyrelu_mean_var_multithreading(
    std::vector<float> &mz, std::vector<float> &Sz, float alpha, int z_pos,
    int n, unsigned int NUM_THREADS, std::vector<float> &ma,
    std::vector<float> &J, std::vector<float> &Sa);

void mixture_relu_multithreading(std::vector<float> &mz, std::vector<float> &Sz,
                                 float omega_tol, int zpos, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &ma, std::vector<float> &J,
                                 std::vector<float> &Sa);

void mixture_relu_multithreading(std::vector<float> &mz, std::vector<float> &Sz,
                                 float omega_tol, int zpos, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &ma, std::vector<float> &J,
                                 std::vector<float> &Sa);

void mixture_tanh_multithreading(std::vector<float> &mz, std::vector<float> &Sz,
                                 float omega_tol, int zpos, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &ma, std::vector<float> &J,
                                 std::vector<float> &Sa);

void mixture_sigmoid_multithreading(std::vector<float> &mz,
                                    std::vector<float> &Sz, float omega_tol,
                                    int zpos, int n, unsigned int num_threads,
                                    std::vector<float> &ma,
                                    std::vector<float> &J,
                                    std::vector<float> &Sa);

void act_full_cov_multithreading(std::vector<float> &Sz_f,
                                 std::vector<float> &J, int no, int B,
                                 int z_pos_out, unsigned int NUM_THREADS,
                                 std::vector<float> &Sa_f);

void activate_hidden_states_cpu(Network &net, NetState &state, int j);

void exp_log_softmax_cpu(std::vector<float> &mz, std::vector<float> &vz,
                         std::vector<float> &me_check,
                         std::vector<float> &ve_check,
                         std::vector<float> &cov_z_e_check, float sigma_v,
                         int no, int B, int z_pos, std::vector<float> &ma,
                         std::vector<float> &va);
