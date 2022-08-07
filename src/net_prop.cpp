///////////////////////////////////////////////////////////////////////////////
// File:         net_prop.cpp
// Description:  Network properties
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 29, 2021
// Updated:      July 29, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/net_prop.h"

#include "../include/common.h"

/////////////////////////////
// NETWORK PROPERTIES
/////////////////////////////
std::tuple<int, int> compute_downsample_img_size(int kernel, int stride, int wi,
                                                 int hi, int pad, int pad_type)
/* compute the size of downsampling images i.e. reduction of image size
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: width of the input image
 *    hi: height of the input image
 *    pad: number of paddings
 *    pad_type: padding type
 *
 * returns:
 *    wo: width of the output image
 *    ho: height of the output image
 *    */
{
    int wo, ho, nom_w, nom_h;

    // Compute nominator of conv. formulation given a padding type
    if (pad_type == 1) {
        nom_w = wi - kernel + 2 * pad;
        nom_h = hi - kernel + 2 * pad;
    } else if (pad_type == 2) {
        nom_w = wi - kernel + pad;
        nom_h = hi - kernel + pad;
    } else {
        nom_w = wi - kernel;
        nom_h = hi - kernel;
    }

    // Check validity of the conv. hyper-parameters such as wi, hi, kernel,
    // stride

    if (nom_w % stride == 0 && nom_h % stride == 0) {
        wo = nom_w / stride + 1;
        ho = nom_h / stride + 1;
    } else {
        throw std::invalid_argument(
            "Input hyper-parameters for conv. layer are invalid ");
    }

    return {wo, ho};
}

std::tuple<int, int> compute_upsample_img_size(int kernel, int stride, int wi,
                                               int hi, int pad, int pad_type)
/* Compute the size of upsampling images i.e. increase of image size.
 *
 * Args:
 *    Kernel: size of the receptive field
 *    stride: Stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    pad: Number of paddings
 *    pad_type: Padding type
 *
 * Returns:
 *    wo: Width of the output image
 *    ho: Height of the output image
 *    */
{
    int wo, ho, nom_w, nom_h;
    // Compute nominator of tconv. formulation given a padding type
    if (pad_type == 1) {
        wo = stride * (wi - 1) + kernel - 2 * pad;
        ho = stride * (hi - 1) + kernel - 2 * pad;
        nom_w = wo - kernel + 2 * pad;
        nom_h = ho - kernel + 2 * pad;
    }

    // Check validity of the conv. hyper-parameters such as wi, hi, kernel,
    // stride
    else if (pad_type == 2) {
        wo = stride * (wi - 1) + kernel - pad;
        ho = stride * (hi - 1) + kernel - pad;
        nom_w = wo - kernel + pad;
        nom_h = ho - kernel + pad;
    }

    if (nom_w % stride != 0 || nom_h % stride != 0) {
        throw std::invalid_argument(
            "Input hyper-parameters for tconv. layer are invalid ");
    }

    return {wo, ho};
}

std::tuple<int, int> get_number_param_fc(int ni, int no, bool use_bias)
/* Get the number of parameters for full-connected layer.
 *
 * Args:
 *    ni: Number of input node
 *    no: Number of output node
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight parameters
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = ni * no;
    if (use_bias) {
        n_b = no;
    } else {
        n_b = 0;
    }

    return {n_w, n_b};
}

std::tuple<int, int> get_number_param_conv(int kernel, int fi, int fo,
                                           bool use_bias)
/* Get the number of parameters for conv. and tconv. layer.
 *
 * Args:
 *    kernel: Size of the receptive field
 *    fi: Number of filters for input image
 *    fo: Number of filters for output image
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = kernel * kernel * fi * fo;
    if (use_bias) {
        n_b = fo;
    } else {
        n_b = 0;
    }

    return {n_w, n_b};
}

std::tuple<int, int> get_number_param_norm(int n)
/*
 * Get number of parameters for the normalization layer given that the
 * previous is full-connected layer.
 *
 * Args:
 *    n: Either the number of input nodes (fc) or input filers (conv.).
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 * */
{
    int n_w, n_b;
    n_w = n;
    n_b = n;

    return {n_w, n_b};
}

void get_similar_layer(Network &net)
/* Label similar layer as indices so that we can avoid doubling the indices
 * in the case of conv., tconv., and pooling layers.
 *
 * Args:
 *    net: Network properties
 *
 * Returns:
 *    We update the attribute "similar_layers" in net
 *    */
{
    int num_layers = net.layers.size();
    int label = 0;
    for (int k = 0; k < net.layers.size(); k++) {
        net.similar_layers[k] = k;
    }

    for (int i = 0; i < num_layers; i++) {
        if (net.similar_layers[i] == i) {
            for (int j = 0; j < num_layers; j++) {
                if (net.widths[j] == net.widths[i] &&
                    net.heights[j] == net.heights[i] &&
                    net.kernels[j] == net.kernels[i] &&
                    net.strides[j] == net.strides[i] &&
                    net.filters[j] == net.filters[i] &&
                    net.pads[j] == net.pads[i] &&
                    net.pad_types[j] == net.pad_types[i] &&
                    net.layers[j] != net.layer_names.fc) {
                    net.similar_layers[j] = i;
                }
            }
        }
    }
}

void set_idx_to_similar_layer(std::vector<int> &similar_layers,
                              std::vector<int> &idx)
/* Set indext for each layer based on the similar features
 */
{
    for (int j = 0; j < similar_layers.size(); j++) {
        idx[j] = idx[similar_layers[j]];
    }
}

int get_first_shortcut_layer(std::vector<int> shortcut) {
    int first_shortcut = -1;
    for (int i = 0; i < shortcut.size(); i++) {
        if (shortcut[i] > 1) {
            first_shortcut = shortcut[i];
            break;
        }
    }
    return first_shortcut;
}

/////////////////////////////
// PARAMETER INITIALIZATION
/////////////////////////////
float he_init(float fan_in)

/* He initialization for neural networks. Further details can be found in
 * Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification. He et al., 2015.
 *
 * Args:
 *    fan_in: Number of input variables
 * Returns:
 *    scale: Standard deviation for weight distribution
 *
 *  */

{
    float scale = pow(1 / fan_in, 0.5);

    return scale;
}

float xavier_init(float fan_in, float fan_out)

/* Xavier initialization for neural networks. Further details can be found in
 *  Understanding the difficulty of training deep feedforward neural networks
 *  - Glorot, X. & Bengio, Y. (2010).
 *
 * Args:
 *    fan_in: Number of input variables
 *    fan_out: Number of output variables
 *
 * Returns:
 *    scale: Standard deviation for weight distribution
 *
 *  */

{
    float scale;
    scale = pow(2 / (fan_in + fan_out), 0.5);

    return scale;
}

std::tuple<std::vector<float>, std::vector<float>> gaussian_param_init(
    float scale, float gain, int N)
/* Parmeter initialization of TAGI neural networks.
 *
 * Args:
 *    scale: Standard deviation for weight distribution
 *    gain: Mutiplication factor
 *    N: Number of parameters
 *
 * Returns:
 *    m: Mean
 *    S: Variance
 *
 *  */
{
    // Initialize device
    std::random_device rd;

    // Mersenne twister PRNG - seed
    std::mt19937 gen(rd());

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);

    // Weights
    for (int i = 0; i < N; i++) {
        // Variance
        S[i] = gain * pow(scale, 2);

        // Get normal distribution
        std::normal_distribution<float> d(0.0f, scale);

        // Get sample for weights
        m[i] = d(gen);
    }

    return {m, S};
}

std::tuple<std::vector<float>, std::vector<float>> gaussian_param_init_ni(
    float scale, float gain, float noise_gain, int N)
/* Parmeter initialization of TAGI neural network including the noise's hidden
 * states
 *
 * Args:
 *    scale: Standard deviation for weight distribution
 *    gain: Mutiplication factor
 *    N: Number of parameters
 *
 * Returns:
 *    m: Mean
 *    S: Variance
 *
 *  */
{
    // Initialize device
    std::random_device rd;

    // Mersenne twister PRNG - seed
    std::mt19937 gen(rd());

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);

    // Weights
    for (int i = 0; i < N; i++) {
        // Variance for output and noise's hidden states
        if (i < N / 2) {
            S[i] = gain * pow(scale, 2);
        } else {
            S[i] = noise_gain * pow(scale, 2);
            scale = pow(S[i], 0.5);
            int a = 0;
        }

        // Get normal distribution
        std::normal_distribution<float> d(0.0f, scale);

        // Get sample for weights
        m[i] = d(gen);
    }

    return {m, S};
}

void get_net_props(Network &net)
/*
 * Get network properties based on the network architecture
 * provided by user.
 *
 * Args:
 *    net: Network architecture
 *    */
{
    bool use_bias;
    int num_layers = net.layers.size();
    std::vector<int> z_pos(num_layers, 0);
    std::vector<int> sc_pos(num_layers, 0);
    std::vector<int> ra_pos(num_layers, 0);
    net.n_state = net.nodes[0] * net.batch_size;
    net.n_max_state = net.nodes[0] * net.batch_size;
    net.n_ra = 0;
    net.n_state_sc = 0;
    net.init_sc = get_first_shortcut_layer(net.shortcuts);

    for (int j = 1; j < num_layers; j++) {
        // Biases are not used for this layer if the next two layer is
        // normalization layer
        if (j < num_layers - 2 && (net.layers[j] != net.layer_names.ln ||
                                   net.layers[j] != net.layer_names.bn)) {
            if (net.layers[j + 1] == net.layer_names.ln ||
                net.layers[j + 1] == net.layer_names.bn) {
                use_bias = false;
            } else {
                use_bias = true;
            }
        } else {
            use_bias = true;
        }

        // Full connected layer
        if (net.layers[j] == net.layer_names.fc) {
            // Compute number of weights and biases
            std::tie(net.num_weights[j], net.num_biases[j]) =
                get_number_param_fc(net.nodes[j - 1], net.nodes[j], use_bias);
        }
        // Convolutional layer
        else if (net.layers[j] == net.layer_names.conv) {
            // Compute the image size
            std::tie(net.widths[j], net.heights[j]) =
                compute_downsample_img_size(
                    net.kernels[j - 1], net.strides[j - 1], net.widths[j - 1],
                    net.heights[j - 1], net.pads[j - 1], net.pad_types[j - 1]);

            // Compute number of nodes
            net.nodes[j] = net.widths[j] * net.heights[j] * net.filters[j];

            // Compute number of weights and biases
            std::tie(net.num_weights[j], net.num_biases[j]) =
                get_number_param_conv(net.kernels[j - 1], net.filters[j - 1],
                                      net.filters[j], use_bias);
        }
        // Transpose convolutional layers
        else if (net.layers[j] == net.layer_names.tconv) {
            // Compute the image size
            std::tie(net.widths[j], net.heights[j]) = compute_upsample_img_size(
                net.kernels[j - 1], net.strides[j - 1], net.widths[j - 1],
                net.heights[j - 1], net.pads[j - 1], net.pad_types[j - 1]);

            // Compute number of nodes
            net.nodes[j] = net.widths[j] * net.heights[j] * net.filters[j];

            // Compute number of weights and biases
            std::tie(net.num_weights[j], net.num_biases[j]) =
                get_number_param_conv(net.kernels[j - 1], net.filters[j - 1],
                                      net.filters[j], use_bias);
        }
        // Pooling layer
        else if (net.layers[j] == net.layer_names.mp ||
                 net.layers[j] == net.layer_names.ap) {
            // Compute the image size
            std::tie(net.widths[j], net.heights[j]) =
                compute_downsample_img_size(
                    net.kernels[j - 1], net.strides[j - 1], net.widths[j - 1],
                    net.heights[j - 1], net.pads[j - 1], net.pad_types[j - 1]);

            // Compute number of nodes
            net.nodes[j] = net.widths[j] * net.heights[j] * net.filters[j];

        }
        // Layernorm layer
        else if (net.layers[j] == net.layer_names.ln) {
            // Compute the image size
            net.widths[j] = net.widths[j - 1];
            net.heights[j] = net.heights[j - 1];

            // Number of weights and biases
            if (net.layers[j - 1] == net.layer_names.fc) {
                std::tie(net.num_weights[j], net.num_biases[j]) =
                    get_number_param_norm(net.nodes[j - 1]);
            } else if (net.layers[j - 1] == net.layer_names.conv ||
                       net.layers[j - 1] == net.layer_names.tconv) {
                std::tie(net.num_weights[j], net.num_biases[j]) =
                    get_number_param_norm(net.filters[j - 1]);
            }

            // Compute number of nodes
            net.nodes[j] = net.widths[j] * net.heights[j] * net.filters[j];

            // Index position running average
            ra_pos[j] = net.batch_size;
            net.n_ra += net.batch_size;

        }

        // Batchnorm layer
        else if (net.layers[j] == net.layer_names.bn) {
            // Compute the image size
            net.widths[j] = net.widths[j - 1];
            net.heights[j] = net.heights[j - 1];

            // Number of weights and biases
            if (net.layers[j - 1] == net.layer_names.fc) {
                std::tie(net.num_weights[j], net.num_biases[j]) =
                    get_number_param_norm(net.nodes[j - 1]);

                // Index position running average
                ra_pos[j] = net.nodes[j];
                net.n_ra += net.nodes[j];
            } else if (net.layers[j - 1] == net.layer_names.conv ||
                       net.layers[j - 1] == net.layer_names.tconv) {
                std::tie(net.num_weights[j], net.num_biases[j]) =
                    get_number_param_norm(net.filters[j - 1]);

                // Index position running average
                ra_pos[j] = net.filters[j];
                net.n_ra += net.filters[j];
            }

            // Compute number of nodes
            net.nodes[j] = net.widths[j] * net.heights[j] * net.filters[j];

        } else {
            throw std::invalid_argument("Layer is not valid");
        }

        // Hidden state position
        z_pos[j] = net.batch_size * net.nodes[j - 1];
        if (net.shortcuts[j] > -1) {
            sc_pos[j + 1] = net.batch_size * net.nodes[j];
            net.n_state_sc += net.batch_size * net.nodes[j];
        }

        // Residual networks
        if (net.shortcuts[j] > -1 &&
            (net.filters[net.shortcuts[j]] != net.filters[j] ||
             net.widths[net.shortcuts[j]] != net.widths[j] ||
             net.heights[net.shortcuts[j]] != net.heights[j])) {
            use_bias = true;
            // Compute number of weights and biases
            std::tie(net.num_weights_sc[net.shortcuts[j]],
                     net.num_biases_sc[net.shortcuts[j]]) =
                get_number_param_conv(1, net.filters[net.shortcuts[j]],
                                      net.filters[j], use_bias);
        }

        // Count total number of elements for state vector
        net.n_state += net.nodes[j] * net.batch_size;
        if (net.nodes[j] * net.batch_size > net.n_max_state) {
            net.n_max_state = net.nodes[j] * net.batch_size;
        };

        // Compute overlap
        if (net.kernels[j - 1] == net.strides[j - 1] ||
            (net.kernels[j - 1] == net.widths[j - 1] &&
             net.strides[j - 1] == 1)) {
            net.overlap[j - 1] = 0;
        }
    }

    // Fist shortcut
    if (net.init_sc > -1) {
        net.n_state_sc += net.nodes[net.init_sc] * net.batch_size;
        sc_pos[net.init_sc + 1] = net.nodes[net.init_sc] * net.batch_size;
    }

    // Cumsum index position
    net.w_pos = cumsum(net.num_weights);
    net.b_pos = cumsum(net.num_biases);
    net.w_sc_pos = cumsum(net.num_weights_sc);
    net.b_sc_pos = cumsum(net.num_biases_sc);
    net.z_pos = cumsum(z_pos);
    net.sc_pos = cumsum(sc_pos);
    net.ra_pos = cumsum(ra_pos);
}

void net_default(Network &net)
/* Initialize network to default value
 *
 * Args:
 *    net: Network architecture
 **/
{
    int num_layers = net.layers.size();
    // Number of inputs & outputs
    if (net.noise_type.compare("heteros") == 0) {
        net.n_y = net.nodes.back() / 2;
    } else {
        net.n_y = net.nodes.back();
    }
    net.n_x = net.nodes.front();

    // Network architecture
    if (net.widths.size() == 0) {
        net.widths.resize(num_layers, 0);
    }
    if (net.heights.size() == 0) {
        net.heights.resize(num_layers, 0);
    }
    if (net.filters.size() == 0) {
        net.filters.resize(num_layers, 0);
    }
    if (net.kernels.size() == 0) {
        net.kernels.resize(num_layers, 1);
    }
    if (net.strides.size() == 0) {
        net.strides.resize(num_layers, 1);
    }
    if (net.pads.size() == 0) {
        net.pads.resize(num_layers, 1);
    }
    if (net.pad_types.size() == 0) {
        net.pad_types.resize(num_layers, 0);
    }
    if (net.shortcuts.size() == 0) {
        net.shortcuts.resize(num_layers, -1);
    }

    // Parameter's hyper-parameters for network
    net.num_weights.resize(num_layers, 0);
    net.num_biases.resize(num_layers, 0);
    net.num_weights_sc.resize(num_layers, 0);
    net.num_biases_sc.resize(num_layers, 0);
    net.similar_layers.resize(num_layers, 0);
    net.overlap.resize(num_layers, 1);

    if (net.gain_w.size() == 0) {
        net.gain_w.resize(num_layers, 1);
    }
    if (net.gain_b.size() == 0) {
        net.gain_b.resize(num_layers, 1);
    }
    net.w_pos.resize(num_layers, 0);
    net.b_pos.resize(num_layers, 0);
    net.w_sc_pos.resize(num_layers, 0);
    net.b_sc_pos.resize(num_layers, 0);
    net.row_zw.resize(num_layers, 0);
    net.col_z_ud.resize(num_layers, 0);
    net.row_w_sc.resize(num_layers, 0);
    net.col_z_sc.resize(num_layers, 0);

    if (net.sigma_v_min == 0) {
        net.sigma_v_min = net.sigma_v;
    }

    // Network's indices
    if (sizeof(net.nye) == 0) {
        net.nye = net.nodes.back();
    }
    if (net.nye != net.nodes.back() && net.nye > 0) {
        net.is_idx_ud = true;
    }
    if (net.Fmwa_1_col.size() == 0) {
        net.Fmwa_1_col.resize(num_layers, 0);
    }
    if (net.FCzwa_1_col.size() == 0) {
        net.FCzwa_1_col.resize(num_layers, 0);
    }
}

void initialize_derivative_state(Network &net, NetState &state) {
    int num_max_nodes = net.n_max_state / net.batch_size;
    state.derv_state.mda.resize(net.n_state, 1);
    state.derv_state.Sda.resize(net.n_state, 0);
    state.derv_state.md_node.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.Sd_node.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.Cdo_diwi.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.md_layer.resize(net.n_state, 1);
    state.derv_state.Sd_layer.resize(net.n_state, 0);
    state.derv_state.md_layer_m.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.Sd_layer_m.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.md_layer_m_o.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.Cdi_zi.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.Cdo_zi.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
    state.derv_state.Cld_zi.resize(net.n_state, 0);
    state.derv_state.Cld_zi_m.resize(
        num_max_nodes * num_max_nodes * net.batch_size, 0);
}

NetState initialize_net_states(Network &net) {
    NetState state;
    state.mz.resize(net.n_state, 0);      // Mean of hidden states
    state.Sz.resize(net.n_state, 1);      // Variance of hidden states
    state.ma.resize(net.n_state, 0);      // Mean of activation units
    state.Sa.resize(net.n_state, 1);      // Variance of activation units
    state.J.resize(net.n_state, 1);       // Diagonal Jacobian matrix
    state.msc.resize(net.n_state_sc, 0);  // Mean of identity's hidden states
    state.Ssc.resize(net.n_state_sc,
                     1);  // Variance of identity's hidden states
    state.mdsc.resize(net.n_state_sc, 0);  // Mean of residual
    state.Sdsc.resize(net.n_state_sc, 1);  // Variance of residual
    state.mra.resize(net.n_ra, 0);  // Mean of batch and layer normalization
    state.Sra.resize(net.n_ra, 1);  // Variance of batch and layer normalization

    // TODO: Is there a better way to initialize the full covariance matrix?
    if (net.is_full_cov) {
        int n = net.n_max_state / net.batch_size;
        state.Sz_f.resize((n * (n + 1) / 2) * net.batch_size, 0);
        state.Sa_f.resize((n * (n + 1) / 2) * net.batch_size, 0);
        state.Sz_fp.resize((n * (n + 1) / 2) * net.batch_size, 0);
    }

    // Noise inference
    if (net.noise_type.compare("homosce") == 0 ||
        net.noise_type.compare("heteros") == 0) {
        int n_noise = net.n_y * net.batch_size;
        state.noise_state.ma_mu.resize(n_noise, 0);
        state.noise_state.Sa_mu.resize(n_noise, 0);
        state.noise_state.Sz_mu.resize(n_noise, 0);
        state.noise_state.J_mu.resize(n_noise, 1);
        state.noise_state.ma_v2b_prior.resize(n_noise, 0);
        state.noise_state.Sa_v2b_prior.resize(n_noise, 0);
        state.noise_state.Sa_v2_prior.resize(n_noise, 0);
        state.noise_state.Cza_v2.resize(n_noise, 0);
        state.noise_state.J_v2.resize(n_noise, 1);
        state.noise_state.ma_v2_post.resize(n_noise, 0);
        state.noise_state.Sa_v2_post.resize(n_noise, 0);
        state.noise_state.J_v.resize(n_noise, 1);
        state.noise_state.delta_mv.resize(n_noise, 0);
        state.noise_state.delta_Sv.resize(n_noise, 0);
        state.noise_state.delta_mz_mu.resize(n_noise, 0);
        state.noise_state.delta_Sz_mu.resize(n_noise, 0);
        state.noise_state.delta_mz_v2b.resize(n_noise, 0);
        state.noise_state.delta_Sz_v2b.resize(n_noise, 0);
    }
    if (net.noise_type.compare("homosce") == 0) {
        set_homosce_noise_param(net.mu_v2b, net.sigma_v2b,
                                state.noise_state.ma_v2b_prior,
                                state.noise_state.Sa_v2b_prior);
    }

    // Derivative state
    if (net.collect_derivative) {
        initialize_derivative_state(net, state);
    }

    return state;
}

//////////////////////////////////////
// INITIALIZE PARAMETERS
/////////////////////////////////////
Param initialize_param(Network &net) {
    /*
     * Initialize weights and niases for Network
     *
     * Args:
     *    net: network architecture
     *
     * Returns:
     *    p: Mean and variance for parameters
     **/

    // Initialization
    int num_layers = net.layers.size();
    int tot_weights = sum(net.num_weights);
    int tot_biases = sum(net.num_biases);
    int tot_weights_sc = sum(net.num_weights_sc);
    int tot_biases_sc = sum(net.num_biases_sc);

    Param p;
    p.mw.resize(tot_weights, 0);
    p.Sw.resize(tot_weights, 0);
    p.mb.resize(tot_biases, 0);
    p.Sb.resize(tot_biases, 0);
    p.mw_sc.resize(tot_weights_sc, 0);
    p.Sw_sc.resize(tot_weights_sc, 0);
    p.mb_sc.resize(tot_biases_sc, 0);
    p.Sb_sc.resize(tot_biases_sc, 0);

    for (int j = 1; j < num_layers; j++) {
        float scale = 0;
        float fan_in = 1;
        float fan_out = 0;
        std::vector<float> mw_j;
        std::vector<float> Sw_j;
        std::vector<float> mb_j;
        std::vector<float> Sb_j;

        // Full-connected layer
        if (net.layers[j] == net.layer_names.fc) {
            fan_in = net.nodes[j - 1];
            fan_out = net.nodes[j];

            // Compute variance
            if (net.init_method.compare("Xavier") == 0 ||
                net.init_method.compare("xavier") == 0) {
                scale = xavier_init(fan_in, fan_out);
            } else {
                scale = he_init(fan_in);
            }

            // Weight
            if (net.num_weights[j] > 0) {
                if (net.noise_type.compare("heteros") == 0 &&
                    j == num_layers - 1) {
                    std::tie(mw_j, Sw_j) = gaussian_param_init_ni(
                        scale, net.gain_w[j], net.noise_gain,
                        net.num_weights[j]);
                } else {
                    std::tie(mw_j, Sw_j) = gaussian_param_init(
                        scale, net.gain_w[j], net.num_weights[j]);
                }
            }

            // Biases
            if (net.num_biases[j] > 0) {
                if (net.noise_type.compare("heteros") == 0 &&
                    j == num_layers - 1) {
                    std::tie(mb_j, Sb_j) = gaussian_param_init_ni(
                        scale, net.gain_b[j], net.noise_gain,
                        net.num_biases[j]);
                } else {
                    std::tie(mb_j, Sb_j) = gaussian_param_init(
                        scale, net.gain_b[j], net.num_biases[j]);
                }
            }
        }

        // Convolutional layer
        else if (net.layers[j] == net.layer_names.conv ||
                 net.layers[j] == net.layer_names.tconv) {
            fan_in = pow(net.kernels[j - 1], 2) * net.filters[j - 1];
            fan_out = pow(net.kernels[j - 1], 2) * net.filters[j];

            // Compute variance
            if (net.init_method.compare("Xavier") == 0 ||
                net.init_method.compare("xavier") == 0) {
                scale = xavier_init(fan_in, fan_out);
            } else {
                scale = he_init(fan_in);
            }

            // Weight
            if (net.num_weights[j] > 0) {
                std::tie(mw_j, Sw_j) = gaussian_param_init(scale, net.gain_w[j],
                                                           net.num_weights[j]);
            }

            // Biases
            if (net.num_biases[j] > 0) {
                std::tie(mb_j, Sb_j) = gaussian_param_init(scale, net.gain_b[j],
                                                           net.num_biases[j]);
            }
        }

        // Normalization layer
        else if (net.layers[j] == net.layer_names.bn ||
                 net.layers[j] == net.layer_names.ln) {
            // Weight
            if (net.num_weights[j] > 0) {
                mw_j.resize(net.num_weights[j], 1);
                Sw_j.resize(net.num_weights[j], 1);
            }

            // Biases
            if (net.num_biases[j] > 0) {
                mb_j.resize(net.num_biases[j], 0);
                Sb_j.resize(net.num_biases[j], 0.0001f);
            }
        }

        // Push to main vector
        if (net.num_weights[j] > 0) {
            push_back_with_idx(p.mw, mw_j, net.w_pos[j - 1]);
            push_back_with_idx(p.Sw, Sw_j, net.w_pos[j - 1]);
        }
        if (net.num_biases[j] > 0) {
            push_back_with_idx(p.mb, mb_j, net.b_pos[j - 1]);
            push_back_with_idx(p.Sb, Sb_j, net.b_pos[j - 1]);
        }

        // Residual net
        if (net.shortcuts[j] > -1 &&
            (net.filters[net.shortcuts[j]] != net.filters[j] ||
             net.widths[net.shortcuts[j]] != net.widths[j] ||
             net.heights[net.shortcuts[j]] != net.heights[j])) {
            std::vector<float> mw_sc_j;
            std::vector<float> Sw_sc_j;
            float scale_sc;
            float fan_in_sc = net.filters[net.shortcuts[j]];
            float fan_out_sc = net.filters[j];

            // Compute variance
            if (net.init_method.compare("Xavier") == 0 ||
                net.init_method.compare("xavier") == 0) {
                scale_sc = xavier_init(fan_in_sc, fan_out_sc);
            } else {
                scale_sc = he_init(fan_in_sc);
            }

            std::tie(mw_sc_j, Sw_sc_j) =
                gaussian_param_init(scale_sc, net.gain_w[net.shortcuts[j]],
                                    net.num_weights_sc[net.shortcuts[j]]);

            push_back_with_idx(p.mw_sc, mw_sc_j,
                               net.w_sc_pos[net.shortcuts[j] - 1]);
            push_back_with_idx(p.Sw_sc, Sw_sc_j,
                               net.w_sc_pos[net.shortcuts[j] - 1]);

            std::vector<float> mb_sc_j(net.num_biases_sc[net.shortcuts[j]], 0);
            std::vector<float> Sb_sc_j(net.num_biases_sc[net.shortcuts[j]],
                                       1e-6);
            push_back_with_idx(p.mb_sc, mb_sc_j,
                               net.b_sc_pos[net.shortcuts[j] - 1]);
            push_back_with_idx(p.Sb_sc, Sb_sc_j,
                               net.b_sc_pos[net.shortcuts[j] - 1]);
        }

        //    if (net.num_biases_sc[j] > 0)
        //    {
        //      std::vector<float> mb_sc_j(net.num_biases_sc[j], 0);
        //      std::vector<float> Sb_sc_j(net.num_biases_sc[j], 1e-6);
        //      push_back_with_idx(p.mb, mb_j, net.b_sc_pos[j - 1]);
        //      push_back_with_idx(p.Sb, Sb_j, net.b_sc_pos[j - 1]);
        //    }
    }

    return p;
}

void load_cfg(std::string net_file, Network &net)
/*
 * Load the user-speficied network archtecture
 *
 * Returns:
 *    net: Network architecture such as layers, nodes, activation function
 *      etc.
 **/
{
    // Dictionary for the cfg file
    std::string key_words[] = {"layers",         "nodes",
                               "kernels",        "strides",
                               "widths",         "heights",
                               "filters",        "pads",
                               "pad_types",      "shortcuts",
                               "activations",    "batch_size",
                               "sigma_v",        "decay_factor_sigma_v",
                               "sigma_v_min",    "sigma_x",
                               "init_method",    "is_full_cov",
                               "noise_type",     "mu_v2b",
                               "sigma_v2b",      "noise_gain",
                               "multithreading", "collect_derivative"};
    int num_keys = sizeof(key_words) / sizeof(key_words[0]);

    // Map strings
    std::map<std::string, std::vector<int> Network::*> fields = {
        {"layers", &Network::layers},
        {"nodes", &Network::nodes},
        {"kernels", &Network::kernels},
        {"strides", &Network::strides},
        {"widths", &Network::widths},
        {"heights", &Network::heights},
        {"filters", &Network::filters},
        {"pads", &Network::pads},
        {"pad_types", &Network::pad_types},
        {"shortcuts", &Network::shortcuts},
        {"activations", &Network::activations}};

    // Load cfg file
    std::string cfg_path = get_current_dir() + "/cfg/" + net_file;
    std::ifstream cfg_file(cfg_path);

    // Initialize pointers
    int d;
    float f;
    std::string si;
    std::string line;
    std::vector<int> v;

    // Open cfg file
    while (std::getline(cfg_file, line)) {
        // Remove white space between characters
        std::string::iterator end_pos =
            std::remove(line.begin(), line.end(), ' ');
        line.erase(end_pos, line.end());
        std::string::iterator tab_pos =
            std::remove(line.begin(), line.end(), '\t');
        line.erase(tab_pos, line.end());

        for (int k = 0; k < num_keys; k++) {
            // Key =  keyword + separator
            std::string key = key_words[k] + ":";

            // Find key in cfg file
            auto pos = line.find(key);
            if (pos == 0) {
                // Store data
                if (key_words[k] == "batch_size") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> d;
                        net.batch_size = d;
                    }
                } else if (key_words[k] == "sigma_v") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> f;
                        net.sigma_v = f;
                    }
                } else if (key_words[k] == "decay_factor_sigma_v") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> f;
                        net.decay_factor_sigma_v = f;
                    }
                } else if (key_words[k] == "sigma_v_min") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> f;
                        net.sigma_v_min = f;
                    }
                } else if (key_words[k] == "sigma_x") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> f;
                        net.sigma_x = f;
                    }
                } else if (key_words[k] == "init_method") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        net.init_method = si;
                    }
                } else if (key_words[k] == "is_full_cov") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        if (si.compare("true") == 0) {
                            net.is_full_cov = true;
                        } else if (si.compare("true") == 0) {
                            net.is_full_cov = false;
                        } else {
                            throw std::invalid_argument(
                                "Input must be true or false - is_full_cov");
                        }
                    }
                } else if (key_words[k] == "noise_type") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        net.noise_type = si;
                    }
                } else if (key_words[k] == "mu_v2b") {
                    std::stringstream ss(line.substr(pos + key.size() + 1));
                    std::vector<float> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);

                        if (iss >> f) {
                            v.push_back(f);
                        }
                    }
                    net.mu_v2b = v;
                } else if (key_words[k] == "sigma_v2b") {
                    std::stringstream ss(line.substr(pos + key.size() + 1));
                    std::vector<float> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> f) {
                            v.push_back(f);
                        }
                    }
                    net.sigma_v2b = v;
                } else if (key_words[k] == "noise_gain") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> f;
                        net.noise_gain = f;
                    }
                } else if (key_words[k] == "multithreading") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        if (si.compare("true") == 0) {
                            net.multithreading = true;
                        } else if (si.compare("false") == 0) {
                            net.multithreading = false;
                        } else {
                            throw std::invalid_argument(
                                "Input must be true or false - multithreading");
                        }
                    }
                } else if (key_words[k] == "collect_derivative") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        if (si.compare("true") == 0) {
                            net.collect_derivative = true;
                        } else if (si.compare("false") == 0) {
                            net.collect_derivative = false;
                        } else {
                            throw std::invalid_argument(
                                "Input must be true or false - "
                                "collect_derivative");
                        }
                    }
                } else {
                    std::stringstream ss(line.substr(pos + key.size() + 1));
                    std::vector<int> v;
                    while (ss.good()) {
                        // Remove comma between layers
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);

                        // If string is dtype d, store in a container v
                        if (iss >> d) {
                            v.push_back(d);
                        }
                    }
                    // Map to field of Network
                    net.*fields[key_words[k]] = v;
                }
                break;
            }
        }
    }
}

bool is_conv(std::vector<int> &layers, LayerLabel &layer_names)
/* Does network contain the convolutional layer?

Args:
    layers: All layer types of the network
    layer_names: Code name of each layer

Returns:
    bool
*/
{
    for (int i = 0; i < layers.size(); i++) {
        if (layers[i] == layer_names.conv) {
            return true;
        }
    }
    return false;
}

bool is_tconv(std::vector<int> &layers, LayerLabel &layer_names)
/* Does network contain the transpose convolutional layer layer?
 */
{
    for (int i = 0; i < layers.size(); i++) {
        if (layers[i] == layer_names.tconv) {
            return true;
        }
    }
    return false;
}

bool is_fc(std::vector<int> &layers, LayerLabel &layer_names)
/* Does network contain the fully-connected layer?
 */
{
    for (int i = 0; i < layers.size(); i++) {
        if (layers[i] == layer_names.fc) {
            return true;
        }
    }
    return false;
}

bool is_leakyrelu(std::vector<int> &activations)
/* Does network contain the leakyrely activation?
 */
{
    for (int i = 0; i < activations.size(); i++) {
        // TODO: Put label instead of integer for leakyrelu
        if (activations[i] == 6) {
            return true;
        }
    }
    return false;
}

void test_get_net_prop() {
    Network net;
    net.layers = {2, 2, 4, 2, 4, 2, 4, 1, 1};
    net.nodes = {3072, 0, 0, 0, 0, 0, 0, 64, 11};
    net.kernels = {5, 3, 5, 3, 5, 3, 1, 1, 1};
    net.strides = {1, 2, 1, 2, 1, 2, 0, 0, 0};
    net.filters = {3, 4, 4, 4, 4, 8, 8, 1, 1};
    net.pads = {2, 1, 2, 1, 2, 1, 0, 0, 0};
    net.pad_types = {1, 2, 1, 2, 1, 2, 0, 0, 0};
    net.activations = {0, 4, 0, 4, 0, 4, 0, 4, 0};
    net.widths = {32, 0, 0, 0, 0, 0, 0, 0, 0};
    net.heights = {32, 0, 0, 0, 0, 0, 0, 0, 0};
    net.shortcuts.resize(net.layers.size(), 0);
    net.num_weights.resize(net.layers.size(), 0);
    net.num_biases.resize(net.layers.size(), 0);
    net.num_weights_sc.resize(net.layers.size(), 0);
    net.num_biases_sc.resize(net.layers.size(), 0);
    net.similar_layers.resize(net.layers.size(), 0);
    net.w_pos.resize(net.layers.size(), 0);
    net.b_pos.resize(net.layers.size(), 0);
    net.w_sc_pos.resize(net.layers.size(), 0);
    net.b_sc_pos.resize(net.layers.size(), 0);

    get_net_props(net);
    get_similar_layer(net);
    std::cout << "Widths = " << std::endl;
    print_matrix(net.widths, 1, 9);

    std::cout << "Heights = " << std::endl;
    print_matrix(net.heights, 1, 9);

    std::cout << "Nodes = " << std::endl;
    print_matrix(net.nodes, 1, 9);

    std::cout << "Similar layers = " << std::endl;
    print_matrix(net.similar_layers, 1, 9);
}

void test_initialize_param() {
    Network net;
    Param p;
    net.layers = {2, 2, 4, 2, 4, 2, 4, 1, 1};
    net.nodes = {1024, 0, 0, 0, 0, 0, 0, 4, 11};
    net.kernels = {5, 3, 5, 3, 5, 3, 1, 1, 1};
    net.strides = {1, 2, 1, 2, 1, 2, 0, 0, 0};
    net.filters = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    net.pads = {2, 1, 2, 1, 2, 1, 0, 0, 0};
    net.pad_types = {1, 2, 1, 2, 1, 2, 0, 0, 0};
    net.activations = {0, 4, 0, 4, 0, 4, 0, 4, 0};
    net.widths = {32, 0, 0, 0, 0, 0, 0, 0, 0};
    net.heights = {32, 0, 0, 0, 0, 0, 0, 0, 0};
    net.shortcuts.resize(net.layers.size(), 0);
    net.num_weights.resize(net.layers.size(), 0);
    net.num_biases.resize(net.layers.size(), 0);
    net.num_weights_sc.resize(net.layers.size(), 0);
    net.num_biases_sc.resize(net.layers.size(), 0);
    net.similar_layers.resize(net.layers.size(), 0);
    net.init_method = "Xavier";
    net.gain_w.resize(net.layers.size(), 1);
    net.gain_b.resize(net.layers.size(), 1);

    get_net_props(net);
    get_similar_layer(net);
    p = initialize_param(net);

    std::cout << "Widths = " << std::endl;
    print_matrix(net.widths, 1, 9);

    std::cout << "Heights = " << std::endl;
    print_matrix(net.heights, 1, 9);

    std::cout << "Nodes = " << std::endl;
    print_matrix(net.num_weights, 1, 9);

    std::cout << "Similar layers = " << std::endl;
    print_matrix(net.similar_layers, 1, 9);
}
