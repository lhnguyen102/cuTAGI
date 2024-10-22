#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"

////////////////////////////////////////////////////////////////////////////////
// FORWARD
////////////////////////////////////////////////////////////////////////////////
__global__ void convtranspose2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a, int const *widx,
    int const *aidx, int woho, int fo, int wihi, int fi, int ki, int rf,
    int batch_size, bool bias, float *mu_z, float *var_z);

template <typename T, size_t BLOCK_TILE, size_t SMEM_PADDING>
__global__ void convtranspose2d_fwd_mean_var_cuda_v1(
    const T *mu_w, const T *var_w, const T *mu_b, const T *var_b, const T *mu_a,
    const T *var_a, const int *widx, const int *aidx, int woho, int fo,
    int wihi, int fi, int ki, int rf, int batch_size, bool bias, float *mu_z,
    float *var_z);

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING, size_t PACK_SIZE>
__global__ void convtranspose2d_fwd_mean_var_cuda_v2(
    const T *mu_w, const T *var_w, const T *mu_b, const T *var_b, const T *mu_a,
    const T *var_a, const int *widx, const int *aidx, int woho, int fo,
    int wihi, int fi, int ki, int rf, int batch_size, bool bias, float *mu_z,
    float *var_z);

////////////////////////////////////////////////////////////////////////////////
// STATE BACKWARD
////////////////////////////////////////////////////////////////////////////////
__global__ void convtranspose2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    float const *delta_var_out, int const *widx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int rf, int batch_size, float *delta_mu,
    float *delta_var);

template <typename T, size_t BLOCK_TILE, size_t SMEM_PADDING>
__global__ void convtranspose2d_bwd_delta_z_cuda_v1(
    const T *mu_w, const T *jcb, const T *delta_mu_out, const T *delta_var_out,
    int const *widx, int const *zidx, int woho, int fo, int wihi, int fi,
    int ki, int rf, int batch_size, T *delta_mu, T *delta_var);

////////////////////////////////////////////////////////////////////////////////
// PARAM BACKWARD
////////////////////////////////////////////////////////////////////////////////
__global__ void convtranspose2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *delta_mu_out,
    float const *delta_var_out, int const *aidx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int batch_size, float *delta_mu_w,
    float *delta_var_w);

template <typename T, size_t BLOCK_TILE, size_t SMEM_PADDING>
__global__ void convtranspose2d_bwd_delta_w_cuda_v1(
    const T *var_w, const T *mu_a, const T *delta_mu_out,
    const T *delta_var_out, const int *aidx, const int *zidx, int woho, int fo,
    int wihi, int fi, int ki, int batch_size, T *delta_mu_w, T *delta_var_w);

__global__ void convtranspose2d_bwd_delta_b_cuda(
    float const *var_b, float const *delta_mu_out, float const *delta_var_out,
    int woho, int fo, int batch_size, float *delta_mu_b, float *delta_var_b);

template <typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y>
__global__ void convtranspose2d_bwd_delta_b_cuda_v1(
    const T *var_b, const T *delta_mu_out, const T *delta_var_out, int woho,
    int fo, int batch_size, T *delta_mu_b, T *delta_var_b);

template <typename T>
void convtranspose2d_bwd_delta_b_dual_sum_reduction(
    T *&var_b, T *&delta_mu_out, T *&delta_var_out, int batch_size, int woho,
    int fo, T *&buf_mu_in, T *&buf_var_in, T *&buf_mu_out, T *&buf_var_out,
    T *&delta_mu, T *&delta_var);