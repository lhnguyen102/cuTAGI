#include "../include/indices.h"

#include <algorithm>

#include "../include/custom_logger.h"

////////////////////////////
// IMAGE CONSTRUCTION
////////////////////////////
std::vector<int> get_image(int w, int h)
/*Create an image with indices from 0 to width * height

 * Args:
 *    w: Width of an image
 *    h: Height of an image
 *
 * Returns:
 *    img: Image associated with indices
 * */
{
    std::vector<int> raw_img(w * h);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            raw_img[r * w + c] = r * w + c + 1;
        }
    }
    return raw_img;
}

std::vector<int> get_padded_image(int w, int h, int w_p, int h_p, int start_idx,
                                  int end_idx, int offset, int pad, int pad_idx,
                                  std::vector<int> &raw_img)
/*Create an padded image associated with the raw image's indices
 * and padding index.

 * Args:
 *    w: Width of an image
 *    h: Height of an image
 *    w_p: Width of the padded image
 *    h_p: Height of the padded image
 *    start_idx: Start index to insert the raw image
 *    end_idx: End index to insert the raw image
 *    offset: Shift of raw image w.r.t the padded image
 *    pad: Number of padding
 *    pad_idx: Padding index i.e. width * height * filter * B + 1
 *
 * Returns:
 *    padded_img: Image with padding and each paddding is assigned to
 *    pad_idx
 *
 * */
{
    std::vector<int> padded_img(w_p * h_p);
    fill(padded_img.begin(), padded_img.end(), pad_idx);
    for (int r = start_idx; r < end_idx; r++) {
        for (int c = start_idx; c < end_idx; c++) {
            padded_img[r * w_p + c] = raw_img[(r - offset) * w + (c - offset)];
        }
    }
    return padded_img;
}

void get_padded_image_dim(int pad, int pad_type, int w, int h, int &w_p,
                          int &h_p, int &start_idx, int &end_idx, int &offset)
/*
 * Get dimensions for the padded image

 * Args:
 *    pad: Number of padding
 *    pad_type: Padding type 1: two sides, 2: 4 sides
 *    w: Width of the image
 *    h: Height of the image
 *
 * Returns:
 *    w_p: Width of the padded image
 *    h_p: Height of the padded image
 *    start_idx: Start index to insert the raw image
 *    end_idx: End index to insert the raw image
 *    offset: Shift of raw image w.r.t the padded image
 */
{
    if (pad > 0) {
        if (pad_type == 1) {
            start_idx = pad;
            end_idx = w + pad;
            offset = pad;
            w_p = w + 2 * pad;
            h_p = h + 2 * pad;
        } else if (pad_type == 2) {
            start_idx = 0;
            end_idx = w;
            offset = 0;
            w_p = w + pad;
            h_p = h + pad;
        }
    } else {
        start_idx = 0;
        end_idx = 0;
        offset = 0;
        w_p = w;
        h_p = h;
    }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>, int, int>
image_construction(int w, int h, int pad, int pad_idx, int pad_type)
/*
 * Construct the raw and padded image.

 * Args:
 *    w: width of an image
 *    h: height of an image
 *    pad: Number of pads
 *    pad_idx: Index of the pad i.e. numel(image) + 1
 *    pad_type: Padding type 1: two sides, 2: 4 sides
 *
 * Returns:
 *    raw_img: Raw image without padding
 *    img: Image with indices including padding
 *    padded_img: Image with padding and each paddding is assigned to pad_idx
 */
{
    int w_p, h_p, start_idx, end_idx, offset;
    std::vector<int> padded_img, raw_img, img;
    get_padded_image_dim(pad, pad_type, w, h, w_p, h_p, start_idx, end_idx,
                         offset);
    raw_img = get_image(w, h);
    img = get_image(w_p, h_p);
    if (pad > 0) {
        padded_img = get_padded_image(w, h, w_p, h_p, start_idx, end_idx,
                                      offset, pad, pad_idx, raw_img);
    } else {
        padded_img.assign(img.begin(), img.end());
    }

    return {raw_img, img, padded_img, w_p, h_p};
}

////////////////////////////
// RECEPTIVE FIELD
////////////////////////////
std::vector<int> get_ref_kernel(std::vector<int> &img, int kernel, int w)
/* Get first receptive field of an image

 * Args:
 *    img: Image associated with the indices
 *    kernel: Size of the receptive field
 *    w: Width of an image
 *
 * Returns:
 *    ref_kernel: First receptive field of an image
 **/
{
    std::vector<int> ref_kernel(kernel * kernel);
    for (int r = 0; r < kernel; r++) {
        for (int c = 0; c < kernel; c++) {
            ref_kernel[r * kernel + c] = img[r * w + c];
        }
    }
    return ref_kernel;
}

std::vector<int> get_img_receptive_field(int wo, int ho, int wi, int stride,
                                         int kernel,
                                         std::vector<int> &ref_kernel)
/*
 * Get receptive field associated with image indices.
 * Args:
 *    wo: Width of the next image
 *    ho: Height of the next image
 *    wi: Width of the current image including padding
 *    stride: Stride of the receptive field
 *    kernel: Size of the receptive field
 *    ref_kernel: First receptive field of an image
 *
 * Returns:
 *    idx: Receptive filed of the current image
 **/
{
    std::vector<int> idx(wo * ho * kernel * kernel);
    for (int rw = 0; rw < wo; rw++) {
        for (int cw = 0; cw < kernel * kernel; cw++) {
            idx[rw * kernel * kernel + cw] = ref_kernel[cw] + rw * stride;
        }
    }

    for (int rh = 1; rh < ho; rh++) {
        for (int ch = 0; ch < kernel * kernel * wo; ch++) {
            idx[rh * (kernel * kernel * wo) + ch] = idx[ch] + rh * stride * wi;
        }
    }

    return idx;
}

std::vector<int> get_padded_img_receptive_field(std::vector<int> &padded_img,
                                                std::vector<int> &idx, int wo,
                                                int ho, int kernel)
/*
 * Get receptive field associated with padded image indices
 *
 * Args:
 *    padded_img: Current image with pad index
 *    idx: Indices of the receptive field i.e. kernel size
 *    wo: Width of the next image
 *    ho: Height of the next image
 *    kernel: Size of the receptive field
 *
 * Returns:
 *   padded_idx: Receptive field associated with pad idx of the current image
 *
 * */
{
    std::vector<int> padded_idx(wo * ho * kernel * kernel);
    for (int r = 0; r < wo * ho; r++) {
        for (int c = 0; c < kernel * kernel; c++) {
            padded_idx[r * kernel * kernel + c] =
                padded_img[idx[r * kernel * kernel + c] - 1];
        }
    }
    return padded_idx;
}

std::vector<int> get_receptive_field(std::vector<int> &img,
                                     std::vector<int> padded_img, int kernel,
                                     int stride, int wo, int ho, int wi, int hi)
/*
 * Main functon for getting receptive field of the current image for next image
 *
 * Args:
 *    img: Current image without pad index
 *    padded_img: Current image with pad index
 *    stride: Stride of the receptive field
 *    kernel: Size of the receptive field
 *    wo: Width of the next image
 *    ho: Height of the next image
 *    wi: Width of the current image including padding
 *    hi: Width of the current image including padding
 *
 *  Returns:
 *    padded_idx: Receptive field associated with pad idx of the current image
 *
 **/
{
    // Check validity for recetive field

    if ((hi - kernel) / stride + 1 != wo && kernel != 1) {
        LOG(LogLevel::ERROR, "Invalid receptive field");
    }
    // Get reference kernel
    std::vector<int> ref_kernel = get_ref_kernel(img, kernel, wi);

    // Get receptive field for next image
    std::vector<int> idx =
        get_img_receptive_field(wo, ho, wi, stride, kernel, ref_kernel);

    // Get receptive field that includes padding
    std::vector<int> padded_idx =
        get_padded_img_receptive_field(padded_img, idx, wo, ho, kernel);

    return padded_idx;
}

////////////////////////////
// IDEM SORT
////////////////////////////
std::vector<int> get_unique_idx(std::vector<int> &M, int pad_idx)
/*Get unique indices of matrix M.

 *
 * Args:
 *    M: A matrix
 *    pad_idx: Padding index
 *
 * Returns:
 *    uM: Unique indices in the matrix M
 **/
{
    // Copy and sort matrix
    std::vector<int> uM;
    uM.assign(M.begin(), M.end());
    sort(uM.begin(), uM.end());

    // Get unique value from matrox
    std::vector<int>::iterator ip;
    ip = std::unique(uM.begin(), uM.end());
    uM.resize(distance(uM.begin(), ip));
    uM.erase(std::remove(uM.begin(), uM.end(), pad_idx), uM.end());

    return uM;
}

std::vector<int> hist_count(std::vector<int> &M, std::vector<int> &uM) {
    std::vector<int> N(uM.size());
    for (int i = 0; i < uM.size(); i++) {
        N[i] = count(M.begin(), M.end(), uM[i]);
    }
    return N;
}

std::tuple<std::vector<int>, int> get_base_idx(std::vector<int> &N) {
    int num_idx = N.size();
    int max_freq = *max_element(N.begin(), N.end());
    int num_elements = 0;
    std::vector<int> base_idx(num_idx * max_freq);
    for (int r = 0; r < num_idx; r++) {
        int freq = N[r];
        for (int c = 0; c < freq; c++) {
            base_idx[c * num_idx + r] = 1;
            num_elements += 1;
        }
    }

    return {base_idx, num_elements};
}

////////////////////////////
// REF SORT
////////////////////////////
std::vector<int> get_idx_from_base(std::vector<int> &base_idx,
                                   std::vector<int> &uM, int pad_idx,
                                   int w_base_idx, int h_base_idx)
/*
 * Get indices from the base index matrix.
 *
 * Args:
 *    base_idx: Binary index matrix for the receptive field
 *    uM: Unique index value in the receptive field
 *    pad_idx: Index of padding
 *    w_base_idx: Width of the binary index matrix
 *    h_base_idx: Height of the binary index matrix
 *
 * Returns:
 *    idx: Indices from base index matrix.
 **/
{
    std::vector<int> idx(w_base_idx * h_base_idx);
    for (int r = 0; r < h_base_idx; r++) {
        for (int c = 0; c < w_base_idx; c++) {
            idx[r * w_base_idx + c] = base_idx[r * w_base_idx + c] * uM[c];
        }
    }
    // Remove zero-index
    idx.erase(std::remove(idx.begin(), idx.end(), 0), idx.end());

    return idx;
}

std::vector<int> get_sorted_idx(std::vector<int> &v)
/*
 * Get sorted indices of a vector.
 *
 * Args:
 *    v: Vector of interest
 *
 * Returns:
 *    sorted_idx: Sorted indices
 **/
{
    // Create a pair vector containing values and indices for a vector
    std::vector<std::pair<int, int>> pv;

    // Insert the index in pair vector
    for (int i = 0; i < v.size(); i++) {
        pv.push_back(std::make_pair(v[i], i));
    }

    // Sort pair vector
    std::sort(pv.begin(), pv.end());

    // Get only sorted indices
    std::vector<int> sorted_idx(v.size());
    for (int i = 0; i < v.size(); i++) {
        sorted_idx[i] = pv[i].second;
    }

    return sorted_idx;
}

std::vector<int> look_up(std::vector<int> &v, int value) {
    std::vector<int> idx;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == value) {
            idx.push_back(i);
        }
    }
    return idx;
}

std::tuple<std::vector<int>, std::vector<int>> get_sorted_reference(
    std::vector<int> &FCzz_idx, std::vector<int> &Fmwa_2_idx, int pad,
    int pad_idx)
/*
 * Get the sorted reference indices.
 *
 * Args:
 *    FCzz_idx; Indices for the current hidden state given the previos one
 *    Fmwa_2_idx: Indices for receptive field
 *    pad: Number of padding
 *    pad_idx: Indice of padding
 *
 * Returns:
 *    sorted_ref: Sorted reference indices
 *    padding_idx: Padding indices of the receptive field indices
 **/
{
    // Sort the FCzz indices
    std::vector<int> FCzz_idx_ref = get_sorted_idx(FCzz_idx);
    std::vector<int> sorted_FCzz_idx = get_sorted_idx(FCzz_idx_ref);

    // Padding index
    std::vector<int> pad_pos, sorted_Fmwa_2;

    // Sort Fmwa indices
    if (pad > 0) {
        // Copy vector Fmwa_2
        std::vector<int> new_Fmwa_2;
        new_Fmwa_2.assign(Fmwa_2_idx.begin(), Fmwa_2_idx.end());

        // Removed the padding
        new_Fmwa_2.erase(
            std::remove(new_Fmwa_2.begin(), new_Fmwa_2.end(), pad_idx),
            new_Fmwa_2.end());

        // Sort Fmwa_2
        sorted_Fmwa_2 = get_sorted_idx(new_Fmwa_2);

        // Find the paddding pos in Fmwa_2
        pad_pos = look_up(Fmwa_2_idx, pad_idx);
    } else {
        sorted_Fmwa_2 = get_sorted_idx(Fmwa_2_idx);
    }

    // Get sorted reference
    std::vector<int> sorted_ref(sorted_Fmwa_2.size());
    for (int i = 0; i < sorted_Fmwa_2.size(); i++) {
        sorted_ref[i] = sorted_Fmwa_2[sorted_FCzz_idx[i]];
    }

    return {sorted_ref, pad_pos};
}

RefIndexOut get_ref_idx(std::vector<int> &M, int pad, int pad_idx)
/*
 * Get reference indices of a matrix M based on its unique values
 * */
{
    // Initialize pointers
    std::vector<int> uM, N, base_idx, ordered_idx, ref, pad_pos;
    int Ne, w, h;
    uM = get_unique_idx(M, pad_idx);
    N = hist_count(M, uM);

    // Get binary base index matrix
    std::tie(base_idx, Ne) = get_base_idx(N);
    w = N.size();
    h = *std::max_element(N.begin(), N.end());
    ordered_idx = get_idx_from_base(base_idx, uM, pad_idx, w, h);

    // Sort indices of receptive field
    std::tie(ref, pad_pos) = get_sorted_reference(ordered_idx, M, pad, pad_idx);

    return {ref, base_idx, pad_pos, w, h};
}

////////////////////////////
// GET INDICES FOR F * mwa
////////////////////////////
std::vector<int> repeat_vector_row(std::vector<int> &v, int num_copies)
/*
 * Copie a vector to n copies along the column.
 *
 * Args:
 *    v: Vector to be copied
 *    num_copies: Number of copies
 *
 * Returns:
 *    cv: Copied vector;
 * */

{
    std::vector<int> cv(v.size() * num_copies);
    for (int i = 0; i < num_copies; i++) {
        for (int j = 0; j < v.size(); j++) {
            cv[i * v.size() + j] = v[j];
        }
    }
    return cv;
}

std::vector<int> repeat_vector_col(std::vector<int> &v, int num_copies)
/*
 * Copie a vector to n copies along the row.
 *
 * Args:
 *    v: Vector to be copied
 *    num_copies: Number of copies
 *
 * Returns:
 *    rv: Copied vector;
 * */

{
    std::vector<int> rv(v.size() * num_copies);
    int N = v.size();
    for (int i = 0; i < num_copies; i++) {
        for (int j = 0; j < N; j++) {
            rv[j * num_copies + i] = v[j];
        }
    }
    return rv;
}

std::vector<int> assign_to_base_idx(std::vector<int> &base_idx,
                                    std::vector<int> &idx, int pad_idx,
                                    int w_base_idx, int h_base_idx)
/*
 * Assign indices to base index matrix.
 * Args:
 *    base_idx: Binary base index matrix
 *    idx: Index vector
 *    pad_idx: padding idex i.e max parameter index + 1
 *    w_base_idx: Width of the base index matrix
 *    h_base_idx: Height of the base index matrix
 *
 * Returns:
 *    idx_mat: Matice of index */
{
    std::vector<int> idx_mat(w_base_idx * h_base_idx);
    int count = 0;
    for (int r = 0; r < h_base_idx; r++) {
        for (int c = 0; c < w_base_idx; c++) {
            if (base_idx[r * w_base_idx + c] == 1) {
                idx_mat[r * w_base_idx + c] = idx[count];
                count += 1;
            } else {
                idx_mat[r * w_base_idx + c] = pad_idx;
            }
        }
    }
    return idx_mat;
}

std::vector<int> reorganize_idx_from_ref(std::vector<int> &M, int pad,
                                         std::vector<int> &pad_pos,
                                         int pad_idx_out,
                                         std::vector<int> &idx_ref,
                                         std::vector<int> &base_idx,
                                         int w_base_idx, int h_base_idx)
/*
 * Re-organize the index matrix as the reference indices
 *
 * Args:
 *    M: Original matrix of indices
 *    pad: Number of paddings
 *    pad_idx: Indices for padding of the current image
 *    base_idx: Binary base index matrix
 *    w_base_idx: Width of the base index matrix
 *    h_base_idx: Height of the base index matrix
 *
 * Returns:
 *    sorted_idx: Indices for z+ conditional on z*/
{
    std::vector<int> tmp;
    tmp.assign(M.begin(), M.end());
    if (pad > 0) {
        for (int i = 0; i < pad_pos.size(); i++) {
            tmp[pad_pos[i]] = -1;
        }
        tmp.erase(std::remove(tmp.begin(), tmp.end(), -1), tmp.end());
    }

    // Reorganize indices as reference indices
    if (idx_ref.size() != tmp.size()) {
        throw std::length_error(
            "Size of reference indices is not equal to size of original "
            "indices");
    }

    std::vector<int> tmp_2(idx_ref.size());
    for (int j = 0; j < idx_ref.size(); j++) {
        tmp_2[j] = tmp[idx_ref[j]];
    }

    // Assign indices to base index matrix
    std::vector<int> sorted_idx = assign_to_base_idx(
        base_idx, tmp_2, pad_idx_out, w_base_idx, h_base_idx);

    return sorted_idx;
}

std::vector<int> get_FCzwa_1_idx(int kernel, int wo, int ho, int pad,
                                 std::vector<int> &pad_pos,
                                 std::vector<int> &idx_ref,
                                 std::vector<int> base_idx, int param_pad_idx,
                                 int w_base_idx, int h_base_idx)
/*
 * Get indices for hidden states conditional on weight.
 *
 * Args:
 *    kernel: Kernel size
 *    wo: Widht of the next image
 *    ho: Height of the next image
 *    pad: Number of padding
 *    pad_pos: Position of padding
 *    base_idx: Binary base index matrix
 *    w_base_idx: Width of the base index matrix
 *    h_base_idx: Height of the base index matrix

 *
 * Returns:
 *    FCzwa_1_idx: Indices for hidden states conditional on weigh
 *    */
{
    std::vector<int> param = get_image(kernel * kernel, 1);
    std::vector<int> tmp = repeat_vector_row(param, wo * ho);

    std::vector<int> FCzwa_1_idx =
        reorganize_idx_from_ref(tmp, pad, pad_pos, param_pad_idx, idx_ref,
                                base_idx, w_base_idx, h_base_idx);

    return FCzwa_1_idx;
}

std::vector<int> get_FCzwa_2_idx(std::vector<int> &Fmwa_2_idx, int pad,
                                 int pad_idx, std::vector<int> &idx_ref,
                                 std::vector<int> &base_idx, int w_base_idx,
                                 int h_base_idx)
/*
 * Get indices for hidden states (z+) conditional on previous hidden
 * states (z).
 *
 * Args:
 *    Fmwa_2_idx: Indices for receptive field
 *    pad: Number of paddings
 *    pad_idx: Indices for padding of the current image
 *    base_idx: Binary base index matrix
 *    w_base_idx: Width of the base index matrix
 *    h_base_idx: Height of the base index matrix
 *
 * Returns:
 *    FCzwa_2_idx: Indices for z+ conditional on z
 *    */
{
    std::vector<int> tmp;
    tmp.assign(Fmwa_2_idx.begin(), Fmwa_2_idx.end());
    if (pad > 0) {
        tmp.erase(std::remove(tmp.begin(), tmp.end(), pad_idx), tmp.end());
    }

    // Reorganize indices as reference indices
    if (idx_ref.size() != tmp.size()) {
        LOG(LogLevel::ERROR,
            "Size of reference indices is not equal to size of FCzwa 2");
    }

    std::vector<int> tmp_2(idx_ref.size());
    for (int j = 0; j < idx_ref.size(); j++) {
        tmp_2[j] = tmp[idx_ref[j]];
    }

    // Assign indices to base index matrix
    std::vector<int> FCzwa_2_idx =
        assign_to_base_idx(base_idx, tmp_2, pad_idx, w_base_idx, h_base_idx);

    return FCzwa_2_idx;
}

std::vector<int> get_Szz_ud_idx(int kernel, int wo, int ho, int pad,
                                std::vector<int> &pad_pos,
                                std::vector<int> &idx_ref,
                                std::vector<int> &base_idx, int pad_idx,
                                int w_base_idx, int h_base_idx)
/*
 * Get indices for Szz.
 *
 * Args:
 *    kernel: kernel size
 *    wo: widht of the next image
 *    ho: height of the next image
 *    pad: Number of padding
 *    pad_idx: Index for padding for next image
 *    w_base_idx: Width of the base index matrix
 *    h_base_idx: Height of the base index matrix
 *
 * Returns:
 *    Szz_ud_idx:
 *    */
{
    std::vector<int> next_image = get_image(wo, ho);
    std::vector<int> tmp = repeat_vector_col(next_image, kernel * kernel);
    std::vector<int> Szz_ud_idx = reorganize_idx_from_ref(
        tmp, pad, pad_pos, pad_idx, idx_ref, base_idx, w_base_idx, h_base_idx);

    return Szz_ud_idx;
}

////////////////////////////////////
// INDICES FOR CONVOLUTIONAL LAYER
///////////////////////////////////
ConvIndexOut get_conv_idx(int kernel, int stride, int wi, int hi, int wo,
                          int ho, int pad, int pad_type, int pad_idx_in,
                          int pad_idx_out, int param_pad_idx)
/*
 * Get index matrices for convolutional layer.
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    wo: width of the output image
 *    ho: height of the output image
 *    pad: Number of padding
 *    pad_type: Type of paddings
 *    pad_idx_in: Padding index for the input image
 *    pad_idx_out: Index for the padding of the output image
 *    param_pad_idx: Index for the padding of the parameters
 *
 * Returns:
 *    FCzwa_1_idx: Index for the parameters sorted as the input hidden state
 *      ordering
 *    FCzwa_2_idx: Index for the receptive field indices sorted as the input
 *      hidden state ordering
 *    Szz_ud_idx: Index for the output hidden states sorted as the input
 *      hidden state ordering
 *    w: Width of three above-mentioned index matrix
 *    h: Height of three above_mentioned idex matrix
 * */
{
    // Initialize pointers
    std::vector<int> raw_img, img, padded_img, Fmwa_2_idx;
    std::vector<int> FCzwa_1_idx, FCzwa_2_idx, Szz_ud_idx, tmp;
    int w_p, h_p, num_elements;

    // Generate image indices
    std::tie(raw_img, img, padded_img, w_p, h_p) =
        image_construction(wi, hi, pad, pad_idx_in, pad_type);

    // Get indices for receptive field
    Fmwa_2_idx =
        get_receptive_field(img, padded_img, kernel, stride, wo, ho, w_p, h_p);

    // Get unique indices and its frequency of the receptive field
    auto Fmwa_2 = get_ref_idx(Fmwa_2_idx, pad, pad_idx_in);

    // Get indices for FCzwa 1
    tmp = get_FCzwa_1_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                          Fmwa_2.base_idx, param_pad_idx, Fmwa_2.w, Fmwa_2.h);

    FCzwa_1_idx = transpose_matrix(tmp, Fmwa_2.w, Fmwa_2.h);

    // Get indices for FCzwa 2
    FCzwa_2_idx = get_FCzwa_2_idx(Fmwa_2_idx, pad, pad_idx_in, Fmwa_2.ref,
                                  Fmwa_2.base_idx, Fmwa_2.w, Fmwa_2.h);

    // Get indices for Szz ud
    Szz_ud_idx =
        get_Szz_ud_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                       Fmwa_2.base_idx, pad_idx_out, Fmwa_2.w, Fmwa_2.h);

    return {Fmwa_2_idx, FCzwa_1_idx, FCzwa_2_idx,
            Szz_ud_idx, Fmwa_2.w,    Fmwa_2.h};
}

////////////////////////////////////
// INDICES FOR CONVOLUTIONAL LAYER
///////////////////////////////////
PoolIndex get_pool_idx(int kernel, int stride, int wi, int hi, int wo, int ho,
                       int pad, int pad_type, int pad_idx_in, int pad_idx_out) {
    // Initialize pointers
    std::vector<int> raw_img, img, padded_img, Fmwa_2_idx, tmp;
    std::vector<int> Szz_ud_idx;
    RefIndexOut Fmwa_2;
    int w_p, h_p;

    // Generate image indices
    std::tie(raw_img, img, padded_img, w_p, h_p) =
        image_construction(wi, hi, pad, pad_idx_in, pad_type);

    // Get indices for receptive field
    tmp =
        get_receptive_field(img, padded_img, kernel, stride, wo, ho, w_p, h_p);
    if (!(kernel == stride || (kernel == wi && stride == 1))) {
        // Get unique indices and its frequency of the receptive field
        Fmwa_2 = get_ref_idx(tmp, pad, pad_idx_in);

        // Get indices for Szz ud
        Szz_ud_idx =
            get_Szz_ud_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                           Fmwa_2.base_idx, pad_idx_out, Fmwa_2.w, Fmwa_2.h);
    }

    // NOTE THAT DOUBLE CHECK WHY WE NEED THE TRANSPOSE HEAR AND SIZE OF MATRIX
    Fmwa_2_idx = transpose_matrix(tmp, kernel * kernel, wo * ho);

    return {Fmwa_2_idx, Szz_ud_idx, Fmwa_2.w, Fmwa_2.h};
}
////////////////////////////////////////////
// INDICES FOR TRANSPOSE CONVOLUTIONAL LAYER
////////////////////////////////////////////
TconvIndexOut get_tconv_idx(int kernel, int wi, int hi, int wo, int ho,
                            int pad_idx_in, int pad_idx_out, int param_pad_idx,
                            ConvIndexOut &convIndex)
/*
 * Get index matrices for transpose convolutional layer.
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    wo: width of the output image
 *    ho: height of the output image
 *    pad: Number of padding
 *    pad_type: Type of paddings
 *    pad_idx_in: Padding index for the input image
 *    pad_idx_out: Index for the padding of the output image
 *    param_pad_idx: Index for the padding of the parameters
 *
 * Returns:
 *    FCwz_2_idx: Activation indices for covariance Z|WA
 *    Swz_ud_idx: Hidden state (Z+) indices for covariance Z|Z+
 *    FCzwa_1_idx: Weight indices for mean product WA
 *    Szz_ud_idx: Next hidden state indices for covariance Z|Z+
 *    w: Width of three above-mentioned index matrix
 *    h: Height of three above_mentioned idex matrix
 *
 * NOTE:
 *    In order to take advantages of convolutional layer. The input image in
 *    this function correspond to the output of the transpose conv. layer.
 *    */
{
    // Initialize pointers
    std::vector<int> FCwz_2_idx, Swz_ud_idx;
    std::vector<int> FCzwa_1_idx, Szz_ud_idx;
    std::vector<int> FCzwa_1_idx_t, FCzwa_2_idx_t, Szz_ud_idx_t, tmp_1, tmp_2,
        tmp_3, tmp_4;
    int pad = 1;

    // Transpose convolutional index matrix
    FCzwa_1_idx_t =
        transpose_matrix(convIndex.FCzwa_1_idx, convIndex.w, convIndex.h);
    FCzwa_2_idx_t =
        transpose_matrix(convIndex.FCzwa_2_idx, convIndex.w, convIndex.h);
    Szz_ud_idx_t =
        transpose_matrix(convIndex.Szz_ud_idx, convIndex.w, convIndex.h);

    ///////////////////////////////////////////
    /* Indices for FCwz 2 and Swz ud */
    // Get unique indices and its frequency of the receptive field
    auto FCwz_1 = get_ref_idx(convIndex.FCzwa_1_idx, pad, param_pad_idx);

    // Get indices for FCwz_2
    tmp_1 = reorganize_idx_from_ref(Szz_ud_idx_t, pad, FCwz_1.pad_pos,
                                    pad_idx_out, FCwz_1.ref, FCwz_1.base_idx,
                                    FCwz_1.w, FCwz_1.h);
    FCwz_2_idx = transpose_matrix(tmp_1, FCwz_1.w, FCwz_1.h);

    // Get indices for Swz ud
    tmp_2 = reorganize_idx_from_ref(FCzwa_2_idx_t, pad, FCwz_1.pad_pos,
                                    pad_idx_in, FCwz_1.ref, FCwz_1.base_idx,
                                    FCwz_1.w, FCwz_1.h);

    Swz_ud_idx = transpose_matrix(tmp_2, FCwz_1.w, FCwz_1.h);

    //////////////////////////////////////////
    /* Indices for FCzz 2 and Szz ud */
    // Get unique indices and its frequency of the receptive field
    auto Szz_ud = get_ref_idx(Szz_ud_idx_t, pad, pad_idx_out);

    // Get indices for FCwz_2
    tmp_3 = reorganize_idx_from_ref(convIndex.FCzwa_1_idx, pad, Szz_ud.pad_pos,
                                    param_pad_idx, Szz_ud.ref, Szz_ud.base_idx,
                                    Szz_ud.w, Szz_ud.h);

    FCzwa_1_idx = transpose_matrix(tmp_3, Szz_ud.w, Szz_ud.h);

    // Get indices for Szz ud
    tmp_4 = reorganize_idx_from_ref(FCzwa_2_idx_t, pad, Szz_ud.pad_pos,
                                    pad_idx_in, Szz_ud.ref, Szz_ud.base_idx,
                                    Szz_ud.w, Szz_ud.h);

    Szz_ud_idx = transpose_matrix(tmp_4, Szz_ud.w, Szz_ud.h);

    return {FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx,
            FCwz_1.w,   FCwz_1.h,   Szz_ud.w,    Szz_ud.h};
}

void index_default(IndexOut &idx) {
    if (idx.Fmwa_1.size() == 0) {
        idx.Fmwa_1.resize(1, 0);
    }
    if (idx.Fmwa_2.size() == 0) {
        idx.Fmwa_2.resize(1, 0);
    }
    if (idx.FCzwa_1.size() == 0) {
        idx.FCzwa_1.resize(1, 0);
    }
    if (idx.FCzwa_2.size() == 0) {
        idx.FCzwa_2.resize(1, 0);
    }
    if (idx.Szz_ud.size() == 0) {
        idx.Szz_ud.resize(1, 0);
    }
    if (idx.pooling.size() == 0) {
        idx.pooling.resize(1, 0);
    }
    if (idx.FCwz_2.size() == 0) {
        idx.FCwz_2.resize(1, 0);
    }
    if (idx.Swz_ud.size() == 0) {
        idx.Swz_ud.resize(1, 0);
    }

    if (idx.Fmwa_2_sc.size() == 0) {
        idx.Fmwa_2_sc.resize(1, 0);
    }
    if (idx.FCzwa_1_sc.size() == 0) {
        idx.FCzwa_1_sc.resize(1, 0);
    }
    if (idx.FCzwa_2_sc.size() == 0) {
        idx.FCzwa_2_sc.resize(1, 0);
    }
    if (idx.Szz_ud_sc.size() == 0) {
        idx.Szz_ud_sc.resize(1, 0);
    }
}
