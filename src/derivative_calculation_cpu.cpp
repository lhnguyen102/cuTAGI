///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calculation_cpu.cpp
// Description:  Calculate derivatives of neural networks
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 12, 2022
// Updated:      July 19, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/derivative_calculation_cpu.h"

void compute_node_derivative_mean_var_fc(
    std::vector<float> &mw, std::vector<float> &Sw, std::vector<float> &mda,
    std::vector<float> &Sda, int w_pos, int z_pos, int ni, int no, int B,
    std::vector<float> &md_node, std::vector<float> &Sd_node)
/* Compute derivatives for each node of fully-connected layer*/
{
    int k;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            k = (i / ni) * j * B;
            md_node[ni * B * j + i] = mw[k + w_pos] * mda[i + z_pos];
            Sd_node[ni * B * j + i] =
                Sw[k + w_pos] * Sda[i + z_pos] +
                Sw[k + w_pos] * mda[i + z_pos] * mda[i + z_pos] +
                Sda[i + z_pos] * mw[k + w_pos] * mw[k + w_pos];
        }
    }
}

void compute_cov_aw_aa_fc(std::vector<float> &mw, std::vector<float> &Sw,
                          std::vector<float> &J_o, std::vector<float> &ma_i,
                          std::vector<float> &Sa_i, int w_pos_i, int z_pos_i,
                          int z_pos_o, int ni, int no, int B,
                          std::vector<float> &Cao_wi,
                          std::vector<float> &Cao_ai)
/*Compute covriance between activation units of current layer and weights
   (Cao_wi) & activation units (Cao_ai) from the previous layer
*/
{
    int k, m;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            m = (i / ni) * no + j;
            k = (i / ni) * j * B;
            Cao_wi[ni * B * j + i] =
                Sw[k + w_pos_i] * ma_i[i + z_pos_i] * J_o[m + z_pos_o];
            Cao_ai[ni * B * j + i] =
                mw[k + w_pos_i] * Sa_i[i + z_pos_i] * J_o[m + z_pos_o];
        }
    }
}

void compute_cov_d_dw_fc(std::vector<float> &mda, std::vector<float> &ma,
                         std::vector<float> &Sa, std::vector<float> &J,
                         std::vector<float> &mw, std::vector<float> &Sw,
                         int act_i, int act_o, int w_pos_i, int z_pos_i,
                         int z_pos_o, int ni, int no, int B,
                         std::vector<float> &Cdo_diwi)
/*Compute covariance between the hidden state's derivatives of the next layer
   and the product of the hidden state's derivaitves & weights fromt the current
   layer
*/
{
    // TODO: Need to create a struct or enum for activation labels
    int k, m;
    float Cao_ai_tmp;
    if (act_i == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i / ni) * j * B;
                Cao_ai_tmp = Sw[k + w_pos_i] * ma[i + z_pos_i] * J[m + z_pos_o];
                Cdo_diwi[ni * B * j + i] =
                    (2 * Cao_ai_tmp * Cao_ai_tmp +
                     4 * Cao_ai_tmp * ma[i + z_pos_i] * ma[m + z_pos_o]) *
                    mw[k + w_pos_i];
            }
        }
    } else if (act_i == 2)  // sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i / ni) * j * B;
                Cao_ai_tmp = Sw[k + w_pos_i] * ma[i + z_pos_i] * J[m + z_pos_o];
                Cdo_diwi[ni * B * j + i] =
                    (Cao_ai_tmp - 2 * Cao_ai_tmp * ma[i + z_pos_i] -
                     2 * ma[m + z_pos_o] * Cao_ai_tmp +
                     2 * Cao_ai_tmp * Cao_ai_tmp +
                     4 * Cao_ai_tmp * ma[i + z_pos_i] * ma[m + z_pos_o]) *
                    mw[k + w_pos_i];
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdo_diwi[i] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i / ni) * j * B;
                Cdo_diwi[ni * B * j + i] +=
                    (-2.0f * ma[m + z_pos_o] * mw[k + w_pos_i] *
                     Sa[i + z_pos_i] * J[m + z_pos_o]) *
                    mda[ni * B * j + i + z_pos_i];
            }
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i / ni) * j * B;
                Cdo_diwi[ni * B * j + i] +=
                    (mw[k + w_pos_i] * Sa[i + z_pos_i] * J[m + z_pos_o]) *
                    mda[ni * B * j + i + z_pos_i];
            }
        }
    } else {
    }
}

// void compute_d_dw_fc(std::vector<float> &mda, std::vector<float> &mw,
//                      std::vector<float> &Cdo_wi, std::vector<float> &Cdo_di,
//                      int w_pos_i, int z_pos_i, int ni, int no, int B,
//                      std::vector<float> &Cdo_diwi)
// /*Compute covariance between the hidden state's derivatives of the next layer
//    and the product of the hidden state's derivaitves & weights fromt the
//    current layer*/
// {
//     int k;
//     for (int j = 0; j < no; j++) {
//         for (int i = 0; i < ni * B; i++) {
//             k = (i / ni) * j * B;
//             Cdo_diwi[ni * B * j + i] =
//                 Cdo_wi[ni * B * j + i] * mda[ni * B * j + i + z_pos_i] +
//                 Cdo_di[ni * B * j + i] * mw[k + w_pos_i];
//         }
//     }
// }

// void compute_cov_d_dw_dw_fc(std::vector<float> &md_n, std::vector<float>
// &mw_o,
//                             std::vector<float> &Cdo_diwi, int w_pos_o, int
//                             ni, int no, int nn, int B, std::vector<float>
//                             &Cdn_dowo_diwi)
// /*Compute covariance between the hidden state's derivatives of the l+2 layer
//    and
//     - product of product of hidden state's derivatives & the weights of the
//       next layer and,
//     - of product of hidden state's derivative & weights of the
//       current layers.
// See equation Equation (31) for further details.
// */
// {
//     int l, q;
//     for (int j = 0; j < B * ni; j++) {
//         for (int i = 0; i < no * nn; j++) {
//             l = (i % no) * B * ni + j;
//             q = (i % no) + nn * (j / ni);
//             Cdn_dowo_diwi[j + no * nn * i] =
//                 md_n[q] * mw_o[i + w_pos_o] * Cdo_diwi[l];
//         }
//     }
// }

void compute_layer_derivative_mean_var_fc(
    std::vector<float> &md_node, std::vector<float> &Sd_node,
    std::vector<float> &md_layer, std::vector<float> &Sd_layer,
    std::vector<float> &md_layer_m_o, std::vector<float> &mw_o,
    std::vector<float> &Cdo_diwi, int z_pos_o, int z_pos_n, int ni, int no,
    int nn, int B, std::vector<float> &md_layer_m,
    std::vector<float> &Sd_layer_m)
/*Compute the derivatives of output w.r.t layer's nodes

*NOTE:
    i -> input i.e., represent the current layer (l)
    o -> output i.e., represent the next layer (l + 1)
    n -> next layer after output i.e., represent layer (l + 2)
*/
{
    int m, l;
    float sum_mean, sum_cov, tmp_md, tmp_cov;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            // Cross covariance
            sum_mean = 0;
            sum_cov = 0;
            tmp_md = 0;
            tmp_cov = 0;
            for (int k = 0; k < nn; k++) {
                l = k + (i / ni) * no * nn + k * nn * j;
                tmp_md = md_layer_m_o[l];
                tmp_cov = md_layer[k + (i / ni) * nn + z_pos_n] * mw_o[k * no] *
                          Cdo_diwi[i + j * ni * B];
                sum_cov += Sd_node[i + j * ni * B] * tmp_md * tmp_md +
                           tmp_cov * tmp_cov +
                           2 * tmp_cov * tmp_md * md_node[i + j * ni * B];
                sum_mean += tmp_cov;
            }

            // Variance
            m = (i / ni) * no + j;
            md_layer_m[ni * B * j + i] =
                sum_mean + md_node[ni * B * j + i] * md_layer[m + z_pos_o];
            Sd_layer_m[ni * B * j + i] =
                sum_cov + Sd_node[ni * B * j + i] * Sd_layer[m + z_pos_o] +
                Sd_layer[m + z_pos_o] * md_node[ni * B * j + i] *
                    md_node[ni * B * j + i];
        }
    }
}

// void compute_cov_az(std::vector<float> &J, std::vector<float> &Sz,
//                     std::vector<float> &mw, int w_pos_i, int z_pos_i,
//                     int z_pos_o, int ni, int no, int B,
//                     std::vector<float> &Cai_zi, std::vector<float> &Cao_zi)
// /*Compute the covariance between activation units & hidden states*/
// {
//     int k, m;
//     for (int j = 0; j < no; j++) {
//         for (int i = 0; i < ni * B; i++) {
//             m = (i / ni) * no + j;
//             k = (i / ni) * j * B;
//             Cai_zi[ni * B * j + i] = J[i + z_pos_i] * Sz[i + z_pos_i];
//             Cao_zi[ni * B * j + i] =
//                 mw[k + w_pos_i] * J[i] * Sz[i] * J[m + z_pos_o];
//         }
//     }
// }

void compute_cov_dz(std::vector<float> &ma, std::vector<float> &J,
                    std::vector<float> &Sz, std::vector<float> &mw, int act_o,
                    int act_i, int w_pos_i, int z_pos_i, int z_pos_o, int ni,
                    int no, int B, std::vector<float> &Cdi_zi,
                    std::vector<float> &Cdo_zi)
/*Compute covariance between derivatives of the next and current layers and
   hidden states of the current layer*/
{
    // TODO: Need to create a struct or enum for activation labels
    int k, m;
    if (act_i == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                Cdi_zi[ni * B * j + i] =
                    -2.0f * ma[i + z_pos_i] * J[i + z_pos_i] * Sz[i + z_pos_i];
            }
        }
    } else if (act_i == 2)  // sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                Cdi_zi[ni * B * j + i] = (1.0f - 2.0f * ma[i + z_pos_i]) *
                                         J[i + z_pos_i] * Sz[i + z_pos_i];
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdi_zi[i] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i / ni) * j * B;
                Cdo_zi[ni * B * j + i] = -2.0f * ma[m + z_pos_o] *
                                         mw[k + w_pos_i] * J[i] * Sz[i] *
                                         J[m + z_pos_o];
            }
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cdo_zi[ni * B * j + i] = (1.0f - 2.0f * ma[m + z_pos_o]) *
                                         mw[k + w_pos_i] * J[i] * Sz[i] *
                                         J[m + z_pos_o];
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdo_zi[i] = 0.0f;
        }
    }
}

void compute_cov_last_current_layers(std::vector<float> &mw,
                                     std::vector<float> &md_layer,
                                     std::vector<float> &md_node,
                                     std::vector<float> &md_layer_m_o,
                                     std::vector<float> &Cdi_zi,
                                     std::vector<float> &Cdo_zi, int w_pos_i,
                                     int w_pos_o, int z_pos_n, int ni, int no,
                                     int nn, int B, std::vector<float> &Cld_zi)
/*Compute the covariance between last layer and current layer's  hidden states*/
{
    int l, q;
    float sum, tmp_md;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            sum = 0;
            for (int k = 0; k < nn; k++) {
                l = k + (i / ni) * no * nn + k * nn * j;
                q = (i / ni) * j * B + k * ni * no;
                tmp_md = md_layer_m_o[l];
                sum += tmp_md * Cdi_zi[i + j * ni * B] * mw[q + w_pos_i] +
                       Cdo_zi[i + j * ni * B] * md_node[i + j * ni * B] *
                           md_layer[k + (i / ni) * nn + z_pos_n] *
                           mw[k * no + w_pos_o];
            }
            Cld_zi[ni * B * j + i] = sum;
        }
    }
}

void compute_cov_last_last_minus_1_layers(std::vector<float> &mw,
                                          std::vector<float> &Cdi_zi,
                                          std::vector<float> &Cdo_zi,
                                          int w_pos_i, int ni, int no, int B,
                                          std::vector<float> &Cld_zi)
/*Compute the covariance between last layer and current layer's  hidden states*/
{
    int q;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            q = (i / ni) * j * B + ni * no;
            Cld_zi[ni * B * j + i] = Cdi_zi[i + j * ni * B] * mw[q + w_pos_i];
        }
    }
}

void copy_derivative_mean(std::vector<float> &md_layer_m, int ni, int no, int B,
                          std::vector<float> &md_layer_m_o)
/*Copy layer derivative mean from next layer to avoid overwritting it between
   layer b/c we only store the layer derivatives of the current layer*/
{
    for (int i = 0; i < ni * no * B; i++) {
        md_layer_m_o[i] = md_layer_m[i];
    }
}

void sum_derivatives(std::vector<float> &d_layer_m, int ni, int no, int B,
                     int z_pos, std::vector<float> &d_layer)
/*Sum the derivatives along the row (next layer) */
{
    float sum;
    for (int i = 0; i < B * ni; i++) {
        sum = 0;
        for (int j = 0; j < no; j++) {
            sum += d_layer_m[j * B * ni + i];
        }
        d_layer[i + z_pos] = sum;
    }
}

void compute_layer_derivative(Network &net, Param &theta, NetState &state,
                              int curr_layer)
/* Compute derivatives of output layer's hidden states w.r.t hidden states of
   the current layers*/
{
    // Initialization
    int ni = net.nodes[curr_layer];
    int no = net.nodes[curr_layer + 1];
    int nn = net.nodes[curr_layer + 2];
    int w_pos_i = net.w_pos[curr_layer];
    int w_pos_o = net.w_pos[curr_layer + 1];
    int z_pos_i = net.z_pos[curr_layer];
    int z_pos_o = net.z_pos[curr_layer + 1];
    int z_pos_n = net.z_pos[curr_layer + 2];
    int act_i = net.activations[curr_layer];
    int act_o = net.activations[curr_layer + 1];

    // Copy md_layer_m for next layer
    copy_derivative_mean(state.derv_state.md_layer_m, ni, no, net.batch_size,
                         state.derv_state.md_layer_m_o);

    // Compute node derivatives
    compute_node_derivative_mean_var_fc(
        theta.mw, theta.Sw, state.derv_state.mda, state.derv_state.Sda, w_pos_i,
        z_pos_i, ni, no, net.batch_size, state.derv_state.md_node,
        state.derv_state.Sd_node);

    // Compute cov(d+, dw)
    compute_cov_d_dw_fc(state.derv_state.mda, state.ma, state.Sa, state.J,
                        theta.mw, theta.Sw, act_i, act_o, w_pos_i, z_pos_i,
                        z_pos_o, ni, no, net.batch_size,
                        state.derv_state.Cdo_diwi);

    // Compute layer derivatives
    compute_layer_derivative_mean_var_fc(
        state.derv_state.md_node, state.derv_state.Sd_node,
        state.derv_state.md_layer, state.derv_state.Sd_layer,
        state.derv_state.md_layer_m_o, theta.mw, state.derv_state.Cdo_diwi,
        z_pos_o, z_pos_n, ni, no, nn, net.batch_size,
        state.derv_state.md_layer_m, state.derv_state.Sd_layer_m);

    sum_derivatives(state.derv_state.md_layer_m, ni, no, net.batch_size,
                    z_pos_i, state.derv_state.md_layer);
    sum_derivatives(state.derv_state.Sd_layer_m, ni, no, net.batch_size,
                    z_pos_i, state.derv_state.Sd_layer);

    // Compute cov(d+, z) & cov(d, z)
    compute_cov_dz(state.ma, state.J, state.Sz, theta.mw, act_o, act_i, w_pos_i,
                   z_pos_i, z_pos_o, ni, no, net.batch_size,
                   state.derv_state.Cdi_zi, state.derv_state.Cdo_zi);

    // Compute cov(d_output, z)
    compute_cov_last_current_layers(
        theta.mw, state.derv_state.md_layer, state.derv_state.md_layer_m_o,
        state.derv_state.md_layer_m, state.derv_state.Cdi_zi,
        state.derv_state.Cdo_zi, w_pos_i, w_pos_o, z_pos_n, ni, no, nn,
        net.batch_size, state.derv_state.Cld_zi_m);

    sum_derivatives(state.derv_state.Cld_zi_m, ni, no, net.batch_size, z_pos_i,
                    state.derv_state.Cld_zi);
}

void compute_last_minus_1_layer_derivative(Network &net, Param &theta,
                                           NetState &state, int curr_layer)
/* Compute derivatives of output layer's hidden states w.r.t hidden states of
   the current layers*/
{
    // Initialization
    int ni = net.nodes[curr_layer];
    int no = net.nodes[curr_layer + 1];
    int w_pos_i = net.w_pos[curr_layer];
    int w_pos_o = net.w_pos[curr_layer + 1];
    int z_pos_i = net.z_pos[curr_layer];
    int z_pos_o = net.z_pos[curr_layer + 1];
    int act_i = net.activations[curr_layer];
    int act_o = net.activations[curr_layer + 1];

    // Compute node derivatives
    compute_node_derivative_mean_var_fc(
        theta.mw, theta.Sw, state.derv_state.mda, state.derv_state.Sda, w_pos_i,
        z_pos_i, ni, no, net.batch_size, state.derv_state.md_node,
        state.derv_state.Sd_node);

    sum_derivatives(state.derv_state.md_node, ni, no, net.batch_size, z_pos_i,
                    state.derv_state.md_layer);
    sum_derivatives(state.derv_state.Sd_node, ni, no, net.batch_size, z_pos_i,
                    state.derv_state.Sd_layer);

    // Compute cov(d+, z) & cov(d, z)
    compute_cov_dz(state.ma, state.J, state.Sz, theta.mw, act_o, act_i, w_pos_i,
                   z_pos_i, z_pos_o, ni, no, net.batch_size,
                   state.derv_state.Cdi_zi, state.derv_state.Cdo_zi);

    // Compute cov(d_output, z)
    compute_cov_last_last_minus_1_layers(
        theta.mw, state.derv_state.Cdi_zi, state.derv_state.Cdo_zi, w_pos_i, ni,
        no, net.batch_size, state.derv_state.Cld_zi_m);

    sum_derivatives(state.derv_state.Cld_zi_m, ni, no, net.batch_size, z_pos_i,
                    state.derv_state.Cld_zi);
}

////////////////////////////////////////////////////////////////////////////////
// ACTIVATION DERIVATIVES
////////////////////////////////////////////////////////////////////////////////
void tanh_derivatives(std::vector<float> &ma, std::vector<float> &Sa,
                      std::vector<float> &J, int z_pos, int n,
                      std::vector<float> &mda, std::vector<float> &Sda)
/*Compute mean and variance for the derivatives*/
{
    for (int i = 0; i < n; i++) {
        mda[i + z_pos] = (1 - powf(ma[i + z_pos], 2) - Sa[i + z_pos]);
        Sda[i + z_pos] =
            (2 * Sa[i + z_pos] * (Sa[i + z_pos] + 2 * powf(ma[i + z_pos], 2)));
    }
}

void sigmoid_derivatives(std::vector<float> &ma, std::vector<float> &Sa,
                         std::vector<float> &J, int z_pos, int n,
                         std::vector<float> &mda, std::vector<float> &Sda) {
    for (int i = 0; i < n; i++) {
        mda[i + z_pos] = J[i + z_pos] - Sa[i + z_pos];
        Sda[i + z_pos] =
            Sa[i + z_pos] * (2 * Sa[i + z_pos] + 4 * powf(ma[i + z_pos], 2) -
                             4 * ma[i + z_pos] + 1);
    }
}

void relu_derivatives(std::vector<float> &mz, int z_pos, int n,
                      std::vector<float> &mda, std::vector<float> &Sda) {
    for (int i = 0; i < n; i++) {
        if (mz[i + z_pos] > 0) {
            mda[i + z_pos] = 1.0f;
        }
        Sda[i + z_pos] = 0.0f;
    }
}

void no_act_derivatives(int z_pos, int n, std::vector<float> &mda,
                        std::vector<float> &Sda) {
    for (int i = 0; i < n; i++) {
        mda[i + z_pos] = 1.0f;
        Sda[i + z_pos] = 0.0f;
    }
}

void compute_activation_derivatives(Network &net, NetState &state, int j) {
    int n = net.nodes[j] * net.batch_size;
    if (net.activations[j] == 1)  // tanh
    {
        tanh_derivatives(state.ma, state.Sa, state.J, net.z_pos[j], n,
                         state.derv_state.mda, state.derv_state.Sda);
    } else if (net.activations[j] == 2)  // Sigmoid
    {
        sigmoid_derivatives(state.ma, state.Sa, state.J, net.z_pos[j], n,
                            state.derv_state.mda, state.derv_state.Sda);
    } else if (net.activations[j] == 4)  // ReLU
    {
        relu_derivatives(state.mz, net.z_pos[j], n, state.derv_state.mda,
                         state.derv_state.Sda);
    } else if (net.activations[j] == 0)  // No activation
    {
        no_act_derivatives(net.z_pos[j], n, state.derv_state.mda,
                           state.derv_state.Sda);
    } else {
        throw std::invalid_argument(
            "Activation function is invalid --derivative_cpu.cpp");
    }
}

////////////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////////////
void compute_network_derivatives(Network &net, Param &theta, NetState &state,
                                 int l)
/*Compute derivative of ouput layer's hidden states w.r.t to the hidden states
   of the lth layer*/
{
    // Last layer
    int last_layer = net.layers.size() - 1;
    compute_last_minus_1_layer_derivative(net, theta, state, last_layer);

    for (int k = net.nodes.size() - 2; k >= 0; k--) {
        if (net.layers[k + 1] == net.layer_names.fc) {
            compute_layer_derivative(net, theta, state, k);
        }
    }
}