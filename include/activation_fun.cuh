///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cuh
// Description:  Header file for activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      February, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"

__global__ void noActMeanVar(float const *mz, float const *Sz, float *ma,
                             float *J, float *Sa, int zpos, int n);

__global__ void tanhMeanVar(float const *mz, float const *Sz, float *ma,
                            float *J, float *Sa, int zpos, int n);

__global__ void sigmoidMeanVar(float const *mz, float const *Sz, float *ma,
                               float *J, float *Sa, int zpos, int n);

__global__ void reluMeanVar(float const *mz, float const *Sz, float *ma,
                            float *J, float *Sa, int zpos, int n);

__global__ void softplusMeanVar(float const *mz, float const *Sz, float *ma,
                                float *J, float *Sa, int zpos, int n);

__global__ void leakyreluMeanVar(float const *mz, float const *Sz, float alpha,
                                 float *ma, float *J, float *Sa, int zpos,
                                 int n);

__global__ void mixture_relu(float const *mz, float const *Sz, float omega_tol,
                             int zpos, int n, float *ma, float *J, float *Sa);

__global__ void mixture_tanh(float const *mz, float const *Sz, float omega_tol,
                             int zpos, int n, float *ma, float *J, float *Sa);

__global__ void mixture_sigmoid(float const *mz, float const *Sz,
                                float omega_tol, int zpos, int n, float *ma,
                                float *J, float *Sa);

__global__ void exp_fun(float const *mz, float const *Sz, int n, float *ma,
                        float *Sa, float *Cza);

__global__ void compute_y_check(float const *mu_z, float const *var_z,
                                float const *mu_e_check,
                                float const *var_e_check,
                                float const *cov_z_e_check, int no, int B,
                                int z_pos, float *mu_y_check,
                                float *var_y_check);

__global__ void compute_cov_y_y_check(float const *mu_z, float const *var_z,
                                      float const *mu_e_check,
                                      float const *var_e_check,
                                      float const *cov_z_e_check, int no, int B,
                                      int z_pos, float *cov_y_y_check);

__global__ void compute_cov_z_y_check(float const *var_z,
                                      float const *cov_z_e_check, int no, int B,
                                      int z_pos, float *cov_z_y_check);

__global__ void actFullCov(float const *Szf, float const *J, int no, int B,
                           int zposOut, float *Saf);

__global__ void noActFullCov(float const *Szf, float *Saf, int Nf);

void activate_hidden_states(Network &net, StateGPU &state, int j);