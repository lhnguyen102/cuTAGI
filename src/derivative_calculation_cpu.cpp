///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calculation_cpu.cpp
// Description:  Network properties
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 12, 2022
// Updated:      July 15, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/derivative_calculation_cpu.h"

void compute_node_derivative_mean_var_fc(
    std::vector<float> &mw, std::vector<float> &Sw, std::vector<float> &mda,
    std::vector<float> &Sda, int ni, int no, int B, std::vector<float> &md_node,
    std::vector<float> &Sd_node)
/* Compute derivatives for each node of fully-connected layer*/
{
    int k;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            k = (i / ni) * j * B;
            md_node[ni * B * j + i] = mw[k] * mda[i];
            Sd_node[ni * B * j + i] = Sw[k] * Sda[i] + Sw[k] * mda[i] * mda[i] +
                                      Sda[i] * mw[k] * mw[k];
        }
    }
}

void compute_cov_aw_aa_fc(std::vector<float> &mw, std::float<float> &Sw,
                          std::vector<float> &J_o, std::vector<float> &ma_i,
                          std::vector<float> &Sa_i, int ni, int no, int B,
                          std::vector<float> &Cao_wi,
                          std::vector<float> &Cao_ai)
/*Compute covraince between activation units of current layer and weights
   (Cao_wi) & activation units (Cao_ai) from the previous layer
*/
{
    int k, m;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            m = (i / ni) * no + j;
            k = (i / ni) * j * B;
            Cao_wi[ni * B * j + i] = Sw[k] * ma_i[i] * J_o[m];
            Cao_ai[ni * B * j + i] = mw[k] * Sa_i[i] * J_o[m];
        }
    }
}

void compute_cov_dd_fc(std::vector<float> &ma_o, std::vector<float> &ma_i,
                       std::vector<float> &Cao_wi, std::vector<float> &Cao_ai,
                       int act_o, int act_i, int ni, int no, int B,
                       std::vector<float> &Cdo_di)
/*Compute covariance between derivatives of current layer and weights &
   derivatives from previous layer */
{
    // TODO: Need to create a struct or enum for activation labels
    int m;
    float Cao_ai_tmp;
    if (act_i == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cao_ai_tmp = Cao_ai[ni * B * j + i];
                Cdo_di[ni * B * j + i] = 2 * Cao_ai_tmp * Cao_ai_tmp +
                                         4 * Cao_ai_tmp * ma_i[i] * ma_o[m];
            }
        }
    } else if (act_i == 2)  // sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cao_ai_tmp = Cao_ai[ni * B * j + i];
                Cdo_di[ni * B * j + i] = Cao_ai_tmp - 2 * Cao_ai_tmp * ma_i[i] -
                                         2 * ma_o[m] * Cao_ai_tmp +
                                         2 * Cao_ai_tmp * Cao_ai_tmp +
                                         4 * Cao_ai_tmp * ma_i[i] * ma_o[m];
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdo_di[i] = 0.0f;
        }
    }
}

void compute_cov_dw_fc(std::vector<float> &ma_o, std::vector<float> &ma_i,
                       std::vector<float> &Cao_wi, std::vector<float> &Cao_ai,
                       int act_o, int act_i, int ni, int no, int B,
                       std::vector<float> &Cdo_wi)
/*Compute covariance between derivatives of next layer and weights from
   current layer */
{
    // TODO: Need to create a struct or enum for activation labels
    int m;

    if (act_o == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cdo_w[ni * B * j + i] =
                    -2.0f * ma_o[m] * Cao_wi[ni * B * j + i];
            }
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cdo_w[ni * B * j + i] = Cao_wi[ni * B * j + i](1 - 2 * ma_o[m]);
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdo_wi[i] = 0.0f;
        }
    }
}

void compute_d_dw_fc(std::vector<float> &md, std::vector<float> &mw,
                     std::vector<float> &Cdo_wi, std::vector<float> &Cdo_di,
                     int ni, int no, int B, std::vector<float> &Cdo_diwi)
/*Compute covariance between the hidden state's derivatives of the next layer
   and the product of the hidden state's derivaitves & weights fromt the
   current layer*/
{
    int k;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            k = (i / ni) * j * B;
            Cdo_diwi[ni * B * j + i] =
                Cdo_wi[ni * B * j + i] * md[ni * B * j + i] +
                Cdo_di[ni * B * j + i] * mw[k];
        }
    }
}

void compute_mean_dn_wo_fc(std::vector<float> &md_n, std::vector<float> &mw_o,
                           int ni, int no, int nn, int B,
                           std::vector<float> &mdn_wo)
/*Compute mean for the product of hidden state's derivatives from l+2 layer &
   weights*/
{
    int p;
    for (int j = 0; j < B; j++) {
        for (int i = 0; i < no * nn; i++) {
            p = (j * nn) + (i / no);
            mdn_wo[j * no * nn + i] = md_n[p] * mw_o[j * no * nn + i];
        }
    }
}

void compute_cov_d_dw_dw_fc(std::vector<float> &md_n, std::vector<float> &mw_o,
                            std::vector<float> &Cdo_diwi, int ni, int no,
                            int nn, int B, std::vector<float> &Cdn_dowo_diwi)
/*Compute covariance between the hidden state's derivatives of the l+2 layer
   and
    - product of product of hidden state's derivatives & the weights of the
      next layer and,
    - of product of hidden state's derivative & weights of the
      current layers.
See equation Equation (31) for further details.
*/
{
    int l, q;
    for (int j = 0; j < B * ni; j++) {
        for (int i = 0; i < no * nn; j++) {
            l = (i % no) * B * ni + j;
            q = (i % no) + nn * (j / ni);
            Cdn_dowo_diwi[j + no * nn * i] = md_n[q] * mw_o[i] * Cdo_diwi[l];
        }
    }
}

void compute_layer_derivative_mean_var_fc(
    std::vector<float> &md_node, std::vector<float> &Sd_node,
    std::vector<float> &md_o, std::vector<float> &Sd_o,
    std::vector<float> &md_layer_m_o, std::vector<float> &Cdn_dowo_diwi, int ni,
    int no, int B, std::vector<float> &md_layer_m,
    std::vector<float> &Sd_layer_m)
/*Compute the layer derivative w.r.t nodes without considering the cross
   covariance & derivative of the next layer
*/
{
    int m, l;
    float sum_mean, sum_cov, tmp_md, tmp_cov;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            // Cross covariance
            sum = 0;
            for (int k = 0; k < nn; k++) {
                l = k + (i / ni) * no * nn + k * nn * j;
                tmp_md = md_layer_m_o[l];
                tmp_cov = Cdn_dowo_diwi[l];
                sum_cov += Sd_node[i + j * ni * B] * tmp_md * tmp_md +
                           tmp_cov * tmp_cov +
                           2 * tmp_cov * tmp_md * md_node[i + j * ni * B];
                sum_mean += tmp_cov;
            }

            // Variance
            m = (i / ni) * no + j;
            md_layer_m[ni * B * j + i] =
                sum_mean + md_node[ni * B * j + i] * md_o[m];
            Sd_layer_m[ni * B * j + i] =
                sum_cov + Sd_node[ni * B * j + i] * Sd_o[m] +
                Sd_o[m] * md_node[ni * B * j + i] * md_node[ni * B * j + i];
        }
    }
}

void reshape_md_layer_m(std::vector<float> &md_layer_m_o,
                        std::vector<float> &md_layer_m_reshaped)
/*Reshape the mean of the layer derivatives from the next layer that uses to
   compute the one at the current layer*/
{
    int l;
    for (int j = 0; j < B * ni; j++) {
        for (int i = 0; i < no * nn; i++) {
            l = (i / no) + (i % no) * nn + (j / ni) * no * nn;
            md_layer_m_reshaped[j + i * no * ni] = md_layer_m_o[l];
        }
    }
}

void compute_cov_az(std::vector<float> &Jo, std::vector<float> &J,
                    std::vector<float> &Sz, std::vector<float> &mw, int ni,
                    int no, int B, std::vector<float> &Cai_zi,
                    std::vector<float> &Cao_zi)
/*Compute the covariance between activation units & hidden states*/
{
    int k, m;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            m = (i / ni) * no + j;
            k = (i / ni) * j * B;
            Cai_zi[ni * B * j + i] = J[i] * Sz[i];
            Cao_zi[ni * B * j + i] = mw[k] * J[i] * Sz[i] * J_o[m];
        }
    }
}

void compute_cov_dz(std::vector<float> &ma_o, std::vector<float> &ma_i,
                    std::vector<float> &Cai_zi, std::vector<float> &Cao_zi,
                    int act_o, int act_i, int ni, int no, int B,
                    std::vector<float> &Cdi_zi, std::vector<float> &Cdo_zi)
/*Compute covariance between derivatives of the next and current layers and
   hidden states of the current layer*/
{
    // TODO: Need to create a struct or enum for activation labels
    int m;
    if (act_i == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                Cdi_zi[ni * B * j + i] =
                    -2.0f * ma_i[i] * Cai_zi[ni * B * j + i];
            }
        }
    } else if (act_i == 2)  // sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                Cdi_zi[ni * B * j + i] =
                    (1.0f - 2.0f * ma_i[i]) * Cai_zi[ni * B * j + i];
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
                Cdo_zi[ni * B * j + i] =
                    -2.0f * ma_o[m] * Cao_zi[ni * B * j + i];
            }
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cdo_zi[ni * B * j + i] =
                    (1.0f - 2.0f * ma_o[m]) * Cao_zi[ni * B * j + i];
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdo_zi[i] = 0.0f;
        }
    }
}

void compute_last_layer_cov(std::vector<float> &mw_o, std::vector<float> &mw,
                            std::vector<float> &md_n, std::vector<float> &mw_o,
                            std::vector<float> &md_node,
                            std::vector<float> &md_layer_m_o,
                            std::vector<float> &Cdi_zi,
                            std::vector<float> &Cdo_zi, int ni, int no, int no,
                            int B, std::vector<float> &Cld_zi)
/*Compute the covariance between last layer and the hidden states of the current
   layer*/
{
    int l, q;
    float sum;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            sum = 0;
            for (int k = 0; k < nn; k++) {
                l = k + (i / ni) * no * nn + k * nn * j;
                q = (i / ni) * j * B + k * ni * no;
                tmp_md = md_layer_m_o[l];
                sum += tmp_md * Cdi_zi[i + j * ni * B] * mw[q] +
                       Cdo_zi[i + j * ni * B] * md_node[i + j * ni * B] *
                           md_n[k + (i / ni) * nn] * mw_o[k * no];
            }
            Cld_zi[ni * B * j + i] = sum;
        }
    }
}
