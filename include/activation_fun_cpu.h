///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun_cpu.h
// Description:  Header file for activation functions (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 11, 2022
// Updated:      September 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "struct_var.h"

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

void act_full_cov_multithreading(std::vector<float> &Sz_f,
                                 std::vector<float> &J, int no, int B,
                                 int z_pos_out, unsigned int NUM_THREADS,
                                 std::vector<float> &Sa_f);

void activate_hidden_states(Network &net, NetState &state, int j);
