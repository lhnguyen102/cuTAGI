///////////////////////////////////////////////////////////////////////////////
// File:         indices.h
// Description:  Header file for indices for TAGI. Note that this is a header
// file for indices.cpp where the detailed code are written.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      April 20, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include "common.h"
#include "net_prop.h"
#include "struct_var.h"

////////////////////////////
// IMAGE CONSTRUCTION
////////////////////////////
std::vector<int> get_image(int w, int h);

std::vector<int> get_padded_image(int w, int h, int w_p, int h_p, int start_idx,
                                  int end_idx, int offset, int pad, int pad_idx,
                                  std::vector<int> &raw_img);

void get_padded_image_dim(int pad, int pad_type, int w, int h, int &w_p,
                          int &h_p, int &start_idx, int &end_idx, int &offset);

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, int, int>
image_construction(int w, int h, int pad, int pad_idx, int pad_type);

////////////////////////////
// RECEPTIVE FIELD
////////////////////////////
std::vector<int> get_ref_kernel(std::vector<int> &img, int kernel, int w);

std::vector<int> get_img_receptive_field(int wo, int ho, int wi, int stride,
                                         int kernel,
                                         std::vector<int> &ref_kernel);

std::vector<int> get_padded_img_receptive_field(std::vector<int> &padded_img,
                                                std::vector<int> &idx, int wo,
                                                int ho, int kernel);

std::vector<int> get_receptive_field(std::vector<int> &img,
                                     std::vector<int> padded_img, int kernel,
                                     int stride, int wo, int ho, int wi,
                                     int hi);

////////////////////////////
// IDEM SORT
////////////////////////////
std::vector<int> get_unique_idx(std::vector<int> &M, int pad_idx);

std::vector<int> hist_count(std::vector<int> &M, std::vector<int> &uM);

std::tuple<std::vector<int>, int> get_base_idx(std::vector<int> &N);

////////////////////////////
// REF SORT
////////////////////////////
std::vector<int> get_idx_from_base(std::vector<int> &base_idx,
                                   std::vector<int> &uM, int pad_idx,
                                   int w_base_idx, int h_base_idx);

std::vector<int> get_sorted_idx(std::vector<int> &v);

std::vector<int> look_up(std::vector<int> &v, int value);

std::tuple<std::vector<int>, std::vector<int>> get_sorted_reference(
    std::vector<int> &FCzz_idx, std::vector<int> &Fmwa_2_idx, int pad,
    int pad_idx);

RefIndexOut get_ref_idx(std::vector<int> &M, int pad, int pad_idx);

////////////////////////////
// GET INDICES FOR F * mwa
////////////////////////////
std::vector<int> repeat_vector_row(std::vector<int> &v, int num_copies);

std::vector<int> repeat_vector_col(std::vector<int> &v, int num_copies);

std::vector<int> assign_to_base_idx(std::vector<int> &base_idx,
                                    std::vector<int> &idx, int pad_idx,
                                    int w_base_idx, int h_base_idx);

std::vector<int> reorganize_idx_from_ref(std::vector<int> &M, int pad,
                                         std::vector<int> &pad_pos,
                                         int pad_idx_out,
                                         std::vector<int> &idx_ref,
                                         std::vector<int> &base_idx,
                                         int w_base_idx, int h_base_idx);

std::vector<int> get_FCzwa_1_idx(int kernel, int wo, int ho, int pad,
                                 std::vector<int> &pad_pos,
                                 std::vector<int> &idx_ref,
                                 std::vector<int> base_idx, int param_pad_idx,
                                 int w_base_idx, int h_base_idx);

std::vector<int> get_FCzwa_2_idx(std::vector<int> &Fmwa_2_idx, int pad,
                                 int pad_idx, std::vector<int> &idx_ref,
                                 std::vector<int> &base_idx, int w_base_idx,
                                 int h_base_idx);

std::vector<int> get_Szz_ud_idx(int kernel, int wo, int ho, int pad,
                                std::vector<int> &pad_pos,
                                std::vector<int> &idx_ref,
                                std::vector<int> &base_idx, int pad_idx,
                                int w_base_idx, int h_base_idx);

////////////////////////////////////
// INDICES FOR CONVOLUTIONAL LAYER
///////////////////////////////////
ConvIndexOut get_conv_idx(int kernel, int stride, int wi, int hi, int wo,
                          int ho, int pad, int pad_type, int pad_idx_in,
                          int pad_idx_out, int param_pad_idx);

////////////////////////////////////
// INDICES FOR CONVOLUTIONAL LAYER
///////////////////////////////////
PoolIndex get_pool_idx(int kernel, int stride, int wi, int hi, int wo, int ho,
                       int pad, int pad_type, int pad_idx_in, int pad_idx_out);

////////////////////////////////////////////
// INDICES FOR TRANSPOSE CONVOLUTIONAL LAYER
////////////////////////////////////////////
TconvIndexOut get_tconv_idx(int kernel, int wi, int hi, int wo, int ho,
                            int pad_idx_in, int pad_idx_out, int param_pad_idx,
                            ConvIndexOut &convIndex);

void tagi_idx(IndexOut &idx, Network &net);
void index_default(IndexOut &idx);
