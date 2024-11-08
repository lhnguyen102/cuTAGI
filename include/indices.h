#pragma once
#include "common.h"

struct RefIndexOut {
    std::vector<int> ref, base_idx, pad_pos;
    int w, h;
};
struct ConvIndexOut {
    std::vector<int> Fmwa_2_idx, FCzwa_1_idx, FCzwa_2_idx, Szz_ud_idx;
    int w, h;
};
struct PoolIndex {
    std::vector<int> pool_idx, Szz_ud_idx;
    int w, h;
};
struct TconvIndexOut {
    std::vector<int> FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx;
    int w_wz, h_wz, w_zz, h_zz;
};

struct IndexOut {
    /* Network's hidden states
       Args:
        Fmwa_1: Weight indices for mean product WA
        Fmwa_2: Activation indices for mean product WA
        FCzwa_1: Weight indices for covariance Z|WA
        FCzwa_2: Activation indices for covariance Z|WA
        Szz_ud: Next hidden state indices for covariance Z|Z+
        pooling: Pooling index
        FCwz_2: Activation indices for covariance W|Z+
        Swz_ud: Hidden state (Z+) indices for covariance Z|Z+

    NOTE*: The extension _sc means shortcut i.e. the same indices meaning for
    the residual network
    */
    std::vector<int> Fmwa_1, Fmwa_2, FCzwa_1, FCzwa_2, Szz_ud, pooling, FCwz_2,
        Swz_ud;
    std::vector<int> Fmwa_2_sc, FCzwa_1_sc, FCzwa_2_sc, Szz_ud_sc;
};

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

void index_default(IndexOut &idx);
