///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      March 10, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/convtranspose2d_layer_cuda.cuh"

__global__ void convtranspose2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a, int const *widx,
    int const *aidx, int woho, int fo, int wihi, int fi, int ki, int rf,
    int batch_size, bool bias, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < woho * fo && row < batch_size)  // k = woho * fo
    {
        float sum_mu = 0;
        float sum_var = 0;
        int aidx_tmp = 0;
        int widx_tmp = 0;
        int div_idx = col / woho;
        int mod_idx = col % woho;

        for (int i = 0; i < rf * fi; i++)  // n = ? * fi
        {
            int i_div_rf = i / rf;

            // minus 1 due to the index starting at 1
            widx_tmp = widx[mod_idx * rf + i % rf] + div_idx * ki * ki +
                       i_div_rf * ki * ki * fo - 1;

            aidx_tmp = aidx[mod_idx * rf + i % rf] + row * wihi * fi +
                       i_div_rf * wihi - 1;

            if (aidx_tmp + 1 < wihi * fi * batch_size + 1) {
                sum_mu += mu_w[widx_tmp] * mu_a[aidx_tmp];

                sum_var += (mu_w[widx_tmp] * mu_w[widx_tmp] + var_w[widx_tmp]) *
                               var_a[aidx_tmp] +
                           var_w[widx_tmp] * mu_a[aidx_tmp] * mu_a[aidx_tmp];
            }
        }

        mu_z[col + row * woho * fo] = sum_mu;
        var_z[col + row * woho * fo] = sum_var;
        if (bias) {
            mu_z[col + row * woho * fo] += mu_b[div_idx];
            var_z[col + row * woho * fo] += var_b[div_idx];
        }
    }
}

__global__ void convtranspose2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    float const *delta_var_out, int const *widx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int rf, int batch_size, float *delta_mu,
    float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int K = wihi * fi;

    if (col < K && row < batch_size)  // k = wihi * fi, m = B
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        int widx_tmp;
        int zidx_tmp;                      // updated index (idxSzzUd)
        for (int i = 0; i < rf * fo; i++)  // n = ki2 * fo
        {
            // minus 1 due to the index starting at 1
            widx_tmp = widx[(col % wihi) * ki * ki + i % rf] +
                       (i / rf) * ki * ki + (col / wihi) * ki * ki * fo - 1;

            // indices for deltaM
            zidx_tmp = zidx[(col % wihi) * ki * ki + i % rf] + (i / rf) * woho +
                       row * woho * fo - 1;
            if (zidx_tmp + 1 < woho * fo * batch_size + 1) {
                sum_mu += delta_mu_out[zidx_tmp] * mu_w[widx_tmp];
                sum_var +=
                    mu_w[widx_tmp] * delta_var_out[zidx_tmp] * mu_w[widx_tmp];
            }
        }
        // TODO: Double check the definition zposIn
        delta_mu[col + row * K] = sum_mu * jcb[col + row * K];
        delta_var[col + row * K] =
            sum_var * jcb[col + row * K] * jcb[col + row * K];
    }
}

__global__ void convtranspose2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *delta_mu_out,
    float const *delta_var_out, int const *aidx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/**/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int K = ki * ki * fo;
    int ki2 = ki * ki;
    if (col < K && row < fi)  // m = fi, k = ki2 * fo
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        int zidx_tmp;  // updated index
        int aidx_tmp;
        int col_div_ki2 = col / ki2;
        int col_mod_ki2 = col % ki2;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi * B
        {
            int i_div_wihi = i / wihi;
            int i_mod_wihi = i % wihi;

            // minus 1 due to the index starting at 1
            aidx_tmp = aidx[col_mod_ki2 * wihi + i_mod_wihi] + row * wihi +
                       i_div_wihi * wihi * fi - 1;

            zidx_tmp = zidx[col_mod_ki2 * wihi + i_mod_wihi] +
                       col_div_ki2 * woho + i_div_wihi * woho * fo - 1;

            if (aidx_tmp < wihi * fi * batch_size) {
                // minus 1 due to the index starting at 1
                sum_mu += mu_a[aidx_tmp] * delta_mu_out[zidx_tmp];
                sum_var +=
                    mu_a[aidx_tmp] * mu_a[aidx_tmp] * delta_var_out[zidx_tmp];
            }
        }

        delta_mu_w[col + row * K] = sum_mu * var_w[col + row * K];
        delta_var_w[col + row * K] =
            sum_var * var_w[col + row * K] * var_w[col + row * K];
    }
}

__global__ void convtranspose2d_bwd_delta_b_cuda(
    float const *var_b, float const *delta_mu_out, float const *delta_var_out,
    int woho, int fo, int batch_size, float *delta_mu_b, float *delta_var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < fo)  // k = fo, m = 1
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < woho * batch_size; i++)  // n = woho * B
        {
            int idx = col * woho + (i % woho) + (i / woho) * woho * fo;

            sum_mu += delta_mu_out[idx];
            sum_var += delta_var_out[idx];
        }

        delta_mu_b[col] = sum_mu * var_b[col];
        delta_var_b[col] = var_b[col] * sum_var * var_b[col];
    }
}
