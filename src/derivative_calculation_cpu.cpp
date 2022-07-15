///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calculation_cpu.cpp
// Description:  Network properties
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 12, 2022
// Updated:      July 13, 2022
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
                Cdo_di[ni * B * j + i] =
                    2 * Cao_ai_tmp * Cao_ai_tmp +
                    4 * Cao_ai_tmp * ma_i[ni * B * j + i] * ma_o[m];
            }
        }
    } else if (act_i == 2)  // sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cao_ai_tmp = Cao_ai[ni * B * j + i];
                Cdo_di[ni * B * j + i] =
                    Cao_ai_tmp - 2 * Cao_ai_tmp * ma_i[ni * B * j + i] -
                    2 * ma_o[m] * Cao_ai_tmp + 2 * Cao_ai_tmp * Cao_ai_tmp +
                    4 * Cao_ai_tmp * ma_i[ni * B * j + i] * ma_o[m];
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
/*Compute covariance between derivatives of current layer and weights  from
   previous layer */
{
    // TODO: Need to create a struct or enum for activation labels
    int m;

    if (act_o == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                Cdo_w[ni * B * j + i] = -2 * ma_o[m] * Cao_wi[ni * B * j + i];
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
/*Compute covariance between the hidden state's derivatives of the current layer
   and the product of the hidden state's derivaitves & weights fromt the
   previous layer*/
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
/*Compute mean for the product of hidden state's derivatives fromt next layer &
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

void compute_cov_d_dw_dw_fc(std::vector<float> &mdn_wo,
                            std::vector<float> &Cdo_diwi, int ni, int no,
                            int nn, int B, std::vector<float> &Cdn_dowo_diwi)
/*Compute covariance between the hidden state's derivatives of the next layer
   and product of product of hidden state's derivatives & the weights of the
   current layer and of product of hidden state's derivative & weights of the
   previous layers i.e., Cov(d+, dw * d-m-). See equation Equation (31) for
   further details*/
{
    int l, q;
    for (int j = 0; j < B * ni; j++) {
        for (int i = 0; i < no * nn; j++) {
            l = (i % no) * B * ni + j;
            p = (j / ni) * no * nn + i;
            Cdn_dowo_diwi[j + no * nn * i] = mdn_wo[p] * Cdo_diwi[l];
        }
    }
}

void compute_layer_derivative_mean_var_part_1_fc(
    std::vector<float> &md_node, std::vector<float> &Sd_node,
    std::vector<float> &md_o, std::vector<float> &Sd_o, int ni, int no, int B,
    std::vector<float> &md_layer_m, std::vector<float> &Sd_layer_m)
/*Compute the layer derivative w.r.t nodes without considering the cross
   covariance & derivative of the next layer
*/
{
    int m;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            m = (i / ni) * no + j;
            md_layer_m[ni * B * j + i] = md_node[ni * B * j + i] * md_o[m];
            Sd_layer_m[ni * B * j + i] =
                Sd_node[ni * B * j + i] * Sd_o[m] +
                Sd_o[m] * md_node[ni * B * j + i] * md_node[ni * B * j + i];
        }
    }
}

void compute_layer_derivative_mean_var_part_2_fc(
    std::vector<float> &Sd_node, std::vector<float> &Cdn_dowo_diwi, int ni,
    int no, int nn, int B, std::vector<float> &md_layer_m,
    std::vector<float> &Sd_layer_m)
/*Compute the layer derivative w.r.t nodes without considering the cross
   covariance & derivative of the next layer
*/
{}
