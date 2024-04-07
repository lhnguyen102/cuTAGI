///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cuh
// Description:  Header file for activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      March 06, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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

__global__ void mixture_relu(float const *mz, float const *Sz,
                             int zpos, int n, float *ma, float *J, float *Sa);

__global__ void mixture_tanh(float const *mz, float const *Sz,
                             int zpos, int n, float *ma, float *J, float *Sa);

__global__ void mixture_sigmoid(float const *mz, float const *Sz,
                                int zpos, int n, float *ma,
                                float *J, float *Sa);

__global__ void exp_fun(float const *mz, float const *Sz, int n, float *ma,
                        float *Sa, float *Cza);

__global__ void actFullCov(float const *Szf, float const *J, int no, int B,
                           int zposOut, float *Saf);

__global__ void noActFullCov(float const *Szf, float *Saf, int Nf);

__global__ void compute_cov_m_a_check(float const *var_log,
                                      float const *cov_log_logsum,
                                      float const *mu_m, int no, int B,
                                      float *cov_m_a_check);

__global__ void compute_cov_m_a(float const *cov_m_a_check, float const *mu_a,
                                float const *var_m, float const *var_z,
                                float const *J_m, int z_pos, int no, int B,
                                float *cov_m_a);

void activate_hidden_states(Network &net, StateGPU &state, int j);