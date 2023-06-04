///////////////////////////////////////////////////////////////////////////////
// File:         struct_var.h
// Description:  Header file for struct variable in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 20, 2022
// Updated:      June 04, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <string>
#include <vector>

// NETWORK PROPERTIES
struct LayerLabel {
    int fc = 1;      // Full-connected layer
    int conv = 2;    // Convolutional layer
    int tconv = 21;  // Transpose convolutional layer
    int mp = 3;      // Pooling layer
    int ap = 4;      // Average pooling layer
    int ln = 5;      // Layer normalization layer
    int bn = 6;      // Batch normalization layer
    int lstm = 7;    // LSTM layer
    int mha = 8;     // Multi-head self-attention layer
};

struct ActLabel {
    int no_act = 0;  // No activaivation
    int tanh = 1;
    int sigmoid = 2;
    int relu = 4;
    int softplus = 5;
    int leakyrelu = 6;
    int mrelu = 7;
    int mtanh = 8;
    int msigmoid = 9;
    int softmax = 10;
    int remax = 11;
    int hr_softmax = 12;
};

struct MultiHeadAttentionProp
/*Properties of multi-head self-attention

Args:
    num_heads: Number of attention heads
    time_step: Number of timesteps
    head_size: Size of attention head
 */
{
    std::vector<int> num_heads, timestep, head_size;
};

struct Network {
    /*Network properties
      Args:
        layers: A vector contains different layers of network architecture
        nodes: Number of hidden units
        kernels: Kernel size fo convolutional layer
        widths: Width of image
        heights: Heights of image
        filters: Number of filters i.e. depth of image for each layer
        activations: Activation function
        pads: Padding that applied to image
        pad_types: Type of padding
        num_weights: Number of weights
        num_biases: Number of biases
        num_weights_sc: Number of weights for residual network
        num_biases_sc: Number of biases for residual network
        similar_layers: Index of layers having the smilar indices
        layer_names: Label for each layer
        act_names: Label for each activation function
        init_method: Initalization method e.g. He and Xavier
        gain_w: Gain for weight parameters when initializing
        gain_b: Gain for biases parameters when initializing
        noise_gain : Gain fof biases parameters relating to noise's hidden
            states
        w_pos: Weights weight position for each layer in the vector of weights
        b_pos: Biases position for each layer in the vector of biases
        w_pos_sc: Weights position for each layer in the vector of weights for
                  resnet
        b_pos_sc: Biases position for each layer in the vector of biases
                  for resnet
        z_pos: Hidden state's position for each layers in the hidden state
              vector
        z_pos_lstm: Hidden state's position for LSTM layers in the hidden state
              vector
        sc_pos: Position of the shortcut's hidden state in the shortcut's hidden
                state vector
        ra_pos: Statistical mean and variance position for the
                normalization layer
        overlap: Binary vector that indicates if the kernel size creates an
                 overlap pass through images

        ***********************************************************************
        Fmwa_1_pos: Position of weight indices for mean product WA
        Fmwa_2_pos: Position of either hidden-states and activaition-units
                    indices when perform TAGI's forward pass
        FCzwa_1_pos: Position of weight indices for covariance Z|WA
        FCzwa_2_pos: Position of activation indices for covariance Z|WA
        Szz_ud_pos: Position of next hidden state indices for covariance Z|Z+
        FCwz_2_pos: Position of activation indices for covariance W|Z+
        Swz_ud_pos: Position of hidden state (Z+) indices for covariance Z|Z+
        row_zw: Number of rows of weight indices for covariance Z|WA
        col_z_ud: Number of columns of next hidden state indices for covariance
                  Z|Z+
        Fmwa_1_col: Number of columns of weight indices for mean product WA
        FCzwa_1_col:Number of column of weight indices for covariance Z|WA

        ***********************************************************************
        n_x: Number of inputs
        n_y: Number of outputs without the heteroscedastic noise
        batch_size: Number of batches of data
        n_state: Total number of states
        n_max_state: Maximum number of states amongst layers
        n_state_sc: Number of states for residual network
        n_ra: Number of statistical mean and variances for the normalization
        layer init_sc: First shortcut layer for residual network
        nye: Number of observation for hierarchical softmax
        last_backward_layer: Index of last layer whose hidden states are updated
        sigma_v: Observation noise
        decay_factor_sigma_v: Decaying factor for sigma v (default value: 0.99)
        sigma_v_min: Minimum value of observation noise (default value: 0.3)
        is_output_ud: Whether or not to update output layer
        is_idx_ud: Wheher or not to update only hidden units in the output
                   layers
        sigma_x: Input noise noise
        noise_type: homosce or heteros
        mu_v2b: Mean of the observation noise squared
        sigma_v2b: Standard deviation of the observation noise squared
        alpha: alpha in leakyrelu
        epsilon: Constant for normalization layer to avoid zero-division
        ra_mt: Momentum for the normalization layer
        multithreading: Whether or not to run parallel computing using multiple
            threads
        collect_derivative: Enable the derivative computation mode
        is_full_cov: Enable full covariance mode
        input_seq_len: Sequence lenth for lstm inputs
        input_seq_len: Sequence lenth for last layer's outputs
        seq_stride: Spacing between sequences for lstm layer
        num_lstm_states: Number of lstm hidden states for all layers
        num_max_lstm_states: Number of maximum lstm hidden states amongst layers
        num_cpu_threads: Number of threads for gpu
        num_gpu_threads: Number of threads for gpu
        min_operations: Minimal number of operations to trigger multithread
        device: cpu or cuda will be used to perform TAGI forward and backward
            passes.
        omega_tol: Tolerance for the mixture activation
        cap_factor: A hyper-parameter being used to compute the max value for
            each parameter update

    NOTE*: sc means the shortcut for residual network.
    */
    std::vector<int> layers, nodes, kernels, strides, widths, heights, filters,
        activations;
    std::vector<int> pads, pad_types, shortcuts, num_weights, num_biases;
    std::vector<int> num_weights_sc, num_biases_sc, similar_layers;
    LayerLabel layer_names;
    ActLabel act_names;
    std::string init_method = "Xavier";
    std::vector<float> gain_w, gain_b;
    std::vector<int> w_pos, b_pos, w_sc_pos, b_sc_pos;
    std::vector<int> z_pos, z_pos_lstm, sc_pos, ra_pos, overlap;

    // Position for each layer in index vector
    std::vector<int> Fmwa_1_pos, Fmwa_2_pos, FCzwa_1_pos, FCzwa_2_pos,
        Szz_ud_pos, pooling_pos;
    std::vector<int> FCwz_2_pos, Swz_ud_pos, Fmwa_2_sc_pos, FCzwa_1_sc_pos;
    std::vector<int> FCzwa_2_sc_pos, Szz_ud_sc_pos, row_zw, col_z_ud, row_w_sc;
    std::vector<int> col_z_sc;
    std::vector<int> Fmwa_1_col, FCzwa_1_col;

    int n_x, n_y;
    int batch_size = 1;
    int n_state = 1;
    int n_max_state = 1;
    int n_state_sc = 1;
    int n_ra = 1;
    int init_sc = -1;
    int nye = 0;
    int last_backward_layer = 1;
    float sigma_v = 0.0f;
    float sigma_x = 0.0f;
    std::string noise_type = "none";
    float noise_gain = 0.5f;
    std::vector<float> mu_v2b;
    std::vector<float> sigma_v2b;
    float alpha = 0.1f;
    float epsilon = 0.0001f;
    float ra_mt = 0.9f;
    bool is_output_ud = true;
    bool is_idx_ud = false;
    float decay_factor_sigma_v = 1.0f;
    float sigma_v_min = 0.0f;
    bool multithreading = true;
    bool is_full_cov = false;
    bool collect_derivative = false;
    int input_seq_len = 1;
    int output_seq_len = 0;
    int seq_stride = 0;
    int num_lstm_states = 0;
    int num_max_lstm_states = 0;
    unsigned int num_cpu_threads = 4;  // TODO: Automatic selection
    int num_gpu_threads = 16;
    int min_operations = 1000;
    std::string device = "cpu";
    float omega_tol = 0.0000001f;
    float cap_factor = 1.0f;
    MultiHeadAttentionProp* mha;

    Network() { mha = new MultiHeadAttentionProp; }
    ~Network() {
        delete mha;
        mha = nullptr;
    }
};

// NETWORK STATE
struct NoiseState {
    /* Observation noise's hidden states for the noise inference task

    Args:
        ma_mu: Mean of activation units for the output layer
        Sa_mu: Variance of activation units for the output layer
        Sz_mu: Variance of hidden states for the output layer
        J_mu: Jacobian matrix for the output layer
        ma_v2b_prior: Prior mean of activation units for the noise observation
        Sa_v2b_prior: Prior variance of activation units for the noise
            observation squared
        Cza_v2: Prior covariance between activation units and hidden states for
            the noise observation squared
        J_v2: Jacobian matrix for the noise observation
        ma_v2_post: Post mean of activation units for the noise observation
            squared
        Sa_v2_post: Post variance of activation units for the noise
            observation sqaured
        J_v: Jacobian matrix for the noise observation
        delta_mv: Updated values for mean of hidden states for the observation
             noise
        delta_Sv: Updated values for variance of hidden states for the
            observation noise
        delta_mz_mu: Updated values for mean of the hidden states for the
            outputs
        delta_Sz_mu: Updated values for variance of the hidden states for the
            outputs
        delta_mz_v2b: Updated values for mean of the hidden states for the
            observation noise squared
        delta_Sz_v2b: Updated values for variance of the hidden states for the
            observation noise squared
     */
    std::vector<float> ma_mu, Sa_mu, Sz_mu, J_mu, ma_v2b_prior, Sa_v2b_prior,
        Sa_v2_prior, Cza_v2, J_v2, ma_v2_post, Sa_v2_post, J_v, delta_mv,
        delta_Sv, delta_mz_mu, delta_Sz_mu, delta_mz_v2b, delta_Sz_v2b;
};

struct DerivativeState {
    /*Derivative's hidden states*/
    std::vector<float> mda, Sda, md_node, Sd_node, Cdo_diwi, md_layer, Sd_layer,
        md_layer_m, Sd_layer_m, md_layer_m_o, Cdi_zi, Cdo_zi, Cld_zi, Cld_zi_m;
};

struct LSTMState {
    /*Memory states for lstm network*/
    std::vector<float> mha, Sha, mf_ga, Sf_ga, Jf_ga, mi_ga, Si_ga, Ji_ga,
        mc_ga, Sc_ga, Jc_ga, mo_ga, So_ga, Jo_ga, mca, Sca, Jca, mc, Sc,
        mc_prev, Sc_prev, mh_prev, Sh_prev, Ci_c, Co_tanh_c;
};

struct Remax
/*Probablistic probability*/
{
    std::vector<float> mu_m, var_m, J_m, mu_log, var_log, mu_sum, var_sum,
        mu_logsum, var_logsum, cov_log_logsum, cov_m_a, cov_m_a_check;
    std::vector<int> z_pos, z_sum_pos;
};

struct MultiHeadAttentionState
/**
 * Multi-head self attention.
 *
 * Args:
 *     mu_k: Mean of keys (batch_size, num_heads, time_step, head_size).
 *     var_k: Variance of keys (batch_size, num_heads, time_step, head_size).
 *     mu_q: Mean of query (batch_size, num_heads, time_step, head_size).
 *     var_q: Variance of query (batch_size, num_heads, time_step, head_size).
 *     mu_v: Mean of value (batch_size, num_heads, time_step, head_size).
 *     var_v: Variance of value (batch_size, num_heads, time_step, head_size).
 *     num_heads: Number of attention heads.
 *     time_step: Time step.
 *     head_size: Size of attention heads.
 *     mu_att: Mean of attention (batch_size, num_heads, time_step, head_size).
 *     var_att: Variance of attention (batch_size, num_heads, time_step,
 * head_size).
 */
{
    Remax* remax;
    std::vector<float> mu_k, var_k, mu_q, var_q, mu_v, var_v, mu_att_score,
        var_att_score, mu_qk, var_qk, mu_mqk, var_mqk, J_mqk, mu_sv, var_sv,
        mu_out_proj, var_out_proj, J_out_proj, mu_in_proj, var_in_proj;
    std::vector<int> qkv_pos, att_pos, in_proj_pos;
    int buffer_size;
};

struct MultiHeadAttentionDelta {
    std::vector<float> delta_mu_att_score, delta_var_att_score, delta_mu_v,
        delta_var_v, delta_mu_q, delta_var_q, delta_mu_k, delta_var_k,
        delta_mu_out_proj, delta_var_out_proj, delta_mu_in_proj,
        delta_var_in_proj, delta_mu_buffer, delta_var_buffer, delta_mu_r,
        delta_var_r;
};

struct NetState {
    /* Network's hidden states

       Args:
        mz: Mean of hidden states
        Sz: Variance of hidden states
        ma: Mean of activation units
        Sa: Variance of activation units
        J: Jacobian matrix
        msc: Mean of the shortcut's hidden states
        Ssc: Variance of the short's hidden states
        mdsc: Mean of the delta shortcut's hidden states
        Sdsc: Variance of the delta short's hidden states
        Sz_f: Full covariance of hidden states
        Sa_f: Full covariance of activation units
        Sz_fp: Partially full covariance of hidden states
        mra: Statistical mean for the normalization layers
        Sra: Statistical variance for the normalization layers
    */
    std::vector<float> mz, Sz, ma, Sa, J, msc, Ssc, mdsc, Sdsc, Sz_f, Sa_f,
        Sz_fp;
    std::vector<float> mra, Sra;
    NoiseState noise_state;
    DerivativeState derv_state;
    LSTMState lstm;
    Remax remax;
    MultiHeadAttentionState* mha;

    NetState() { mha = new MultiHeadAttentionState; }
    ~NetState() {
        delete mha;
        mha = nullptr;
    }
};

// NETWORK PARAMETERS
struct Param {
    /* Network's hidden states
       Args:
        mw: Mean of weights
        Sw: Variance of weights
        mb: Mean of the biases
        Sb: Variance of the biases
        mw_sc: Mean of weights for residual network
        Sw_sc: Variance of weights for residual network
        mb_sc: Mean of biases for residual network
        Sb_sc: Variance of biases for residual network
    */

    std::vector<float> mw, Sw, mb, Sb, mw_sc, Sw_sc, mb_sc, Sb_sc;
};

// NETWORK INDICES
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

// NETWORK CONFIGURATION USER-SPECIFIED
struct NetConfig {
    std::vector<int> layers, nodes, kernels, strides, widths, heights, filters,
        pads, pad_types, shortcuts, activations;
    std::vector<float> mu_v2b, sigma_v2b;
    float sigma_v, sigma_v_min, sigma_x, decay_factor_sigma_v, noise_gain;
    int batch_size, input_seq_len, output_seq_len, seq_stride;
    bool multithreading = true, collect_derivative = false, is_full_cov = false;
    std::string init_method = "Xavier", noise_type = "none", device = "cpu";
};

// USER INPUT
struct UserInput {
    std::string model_name, net_name, task_name, data_name, encoder_net_name,
        decoder_net_name;
    std::string device = "cpu";
    int num_classes, num_epochs, num_train_data, num_test_data;
    bool load_param = false, debug = false;
    std::vector<float> mu, sigma;
    std::vector<std::string> x_train_dir, y_train_dir, x_test_dir, y_test_dir;
    std::vector<int> output_col;
    int num_features;
    bool data_norm = true;
};

// IMAGE DATA
struct ImageData {
    /* Image database's format
       Args:
        images: Images stored as a giant vector
        obs_label: Converted observation from the labels
        obs_idx: Observation indices assigned to each label
        labels: Raw labels

    NOTE*: In the context of TAGI, we uses the hierarchical softmax for the
    classification problem. So, we convert the raw lable for the label fictive
    observation.
    */
    std::vector<float> images;
    std::vector<float> obs_label;
    std::vector<int> obs_idx;
    std::vector<int> labels;
    int num_data, image_len, output_len;
};

// REGRESSION DATA

struct Dataloader {
    /* Regression-like database's format
    Args:
        x: Input data
        y: Output data
        mu_x: Sample mean of input data
        sigma_x: Sample standard deviation of input data
        mu_y: Sample mean of output data
        sigma_y: Sample standard deviation of output data
        num_data: Total number of data
        nx: Number of input features
        ny: Number of output features
     */
    std::vector<float> x, y, mu_x, sigma_x, mu_y, sigma_y;
    int num_data, nx, ny;
};

// HIERARCHICAL SOFTMAX
struct HrSoftmax {
    /* Hierarchical softmax
       Args:
        obs: A fictive observation \in [-1, 1]
        idx: Indices assigned to each label
        n_obs: Number of indices for each label
        len: Length of an observation e.g 10 labels -> len(obs) = 11
    */
    std::vector<float> obs;
    std::vector<int> idx;
    int n_obs, len;
};

// DIRECTORIES
struct SavePath {
    /* Different directories to store the results
       Args:
        curr_path: Current path of the workspacefolder
        saved_param_path: Directory to store the weights and biases
        saved_inference_path: Directory to store the inference results
        debug_path: Directory to store network's indices, parameters, inference
                    results that then uses to verify with test units.
    */
    std::string curr_path, saved_param_path, saved_inference_path, debug_path;
};

// FUNCTIONS
void init_multi_head_attention_states(MultiHeadAttentionState& mha_state,
                                      MultiHeadAttentionProp& mha_prop,
                                      int batch_size);

void init_multi_head_attention_delta_states(
    MultiHeadAttentionDelta& delta_mha_state, MultiHeadAttentionProp& mha_prop,
    int batch_size);
