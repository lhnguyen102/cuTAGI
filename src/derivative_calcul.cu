///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calcul.cu
// Description:  Calculate derivatives of neural networks
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 17, 2022
// Updated:      July 30, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/derivative_calcul.cuh"

__global__ void compute_node_derv_mean_var_fc(float const *mw, float const *Sw,
                                              float const *mda,
                                              float const *Sda, int w_pos,
                                              int z_pos, int ni, int no, int B,
                                              float *md_node, float *Sd_node)
/* Compute derivatives for each node for fully-connected layer

Args:
    mw: Mean of weights
    Sw: Variance of weights
    mda: Mean of activation derivatives w.r.t hidden states
    Sda: Variance of activation derivatives w.r.t hidden states
    w_pos: Weight position for this layer in the weight vector of network
    z_pos: Input-hidden-state position for this layer in the hidden-state
        vector of network
    ni: Number of hidden units for imputs
    no: Number of hidden units for outputs
    B: Batch size
    md_node: Derivative mean for each node
    Sd_node: Derivative variance for each node
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int k;
    if (row < no && col < ni * B) {
        k = (col % ni) + row * ni;
        md_node[ni * B * row + col] = mw[k + w_pos] * mda[col + z_pos];
        Sd_node[ni * B * row + col] =
            Sw[k + w_pos] * Sda[col + z_pos] +
            Sw[k + w_pos] * mda[col + z_pos] * mda[col + z_pos] +
            Sda[col + z_pos] * mw[k + w_pos] * mw[k + w_pos];
    }
}

__global__ void compute_cov_d_dw_fc(float const *mda, float const *ma,
                                    float const *Sa, float const *J,
                                    float const *mw, float const *Sw, int act_i,
                                    int act_o, int w_pos_i, int z_pos_i,
                                    int z_pos_o, int ni, int no, int B,
                                    float *Cdo_diwi)
/*Compute covariance between derivative and the product of derivaitves &
weights i.e., cov(d+, dw)

Args:
    mda: Mean of activation derivatives w.r.t hidden states
    ma: Mean of activation units
    mw: Mean of weights
    Sw: Variance of weights
    act_i: Activation function of the inputs
    act_o: Activation function of the outputs
    w_pos_i: Weight position for input in the weight vector of network
    z_pos_i: Input-hidden-state position for inputs in the hidden-state
        vector of network
    z_pos_o: Input-hidden-state position for outputs in the hidden-state
        vector of network
    ni: Number of hidden units for imputs
    no: Number of hidden units for outputs
    B: Batch size
    Cdo_widi: covariance(d+, dw)
*/
{
    // TODO: Need to create a struct or enum for activation labels
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k, m;
    float Cao_ai_tmp;
    if (act_i == 1)  // Tanh
    {
        if (row < no && col < ni * B) {
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cao_ai_tmp = mw[k + w_pos_i] * Sa[col + z_pos_i] * J[m + z_pos_o];
            Cdo_diwi[ni * B * row + col] =
                (2.0f * Cao_ai_tmp * Cao_ai_tmp +
                 4.0f * Cao_ai_tmp * ma[col + z_pos_i] * ma[m + z_pos_o]) *
                mw[k + w_pos_i];
        }
    } else if (act_i == 2)  // sigmoid
    {
        if (row < no && col < ni * B) {
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cao_ai_tmp = mw[k + w_pos_i] * Sa[col + z_pos_i] * J[m + z_pos_o];
            Cdo_diwi[ni * B * row + col] =
                (Cao_ai_tmp - 2.0f * Cao_ai_tmp * ma[col + z_pos_i] -
                 2.0f * ma[m + z_pos_o] * Cao_ai_tmp +
                 2.0f * Cao_ai_tmp * Cao_ai_tmp +
                 4.0f * Cao_ai_tmp * ma[col + z_pos_i] * ma[m + z_pos_o]) *
                mw[k + w_pos_i];
        }
    } else {
        if (row < no && col < ni * B) {
            Cdo_diwi[ni * B * row + col] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        if (row < no && col < ni * B) {
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_diwi[ni * B * row + col] +=
                (-2.0f * ma[m + z_pos_o] * Sw[k + w_pos_i] * ma[col + z_pos_i] *
                 J[m + z_pos_o]) *
                mda[col + z_pos_i];
        }
    } else if (act_o == 2)  // Sigmoid
    {
        if (row < no && col < ni * B) {
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_diwi[ni * B * row + col] +=
                (1.0f - 2.0f * ma[m + z_pos_o]) *
                (Sw[k + w_pos_i] * ma[col + z_pos_i] * J[m + z_pos_o]) *
                mda[col + z_pos_i];
        }
    }
}

__global__ void compute_layer_derv_mean_var_fc(
    float const *md_node, float const *Sd_node, float const *md_layer,
    float const *Sd_layer, float const *md_layer_m_o, float const *mw_o,
    float const *Cdo_diwi, int w_pos_o, int z_pos_o, int z_pos_n, int ni,
    int no, int nn, int B, float *md_layer_m, float *Sd_layer_m)
/*Compute the derivatives of output w.r.t layer's nodes

Args:
    md_node: Derivative mean for each node
    Sd_node: Derivative variance for each node
    md_layer: Layer derivative mean for the network
    Sd_layer: Layer derivative variance for the network
    md_layer_m_o: Layer derivative mean w/o summing over the node for
         output layer
    mw: Mean of weights
    Cdo_widi: covariance(d+, dw)
    w_pos_o: Weight position for output in the weight vector of network
    z_pos_o: Input-hidden-state position for output in the hidden-state
        vector of network
    z_pos_n: Input-hidden-state position for next layer in the hidden-state
        vector of network
    ni: Number of hidden units for inputs
    no: Number of hidden units for outputs
    nn: Number of hidden units for next layer
    B: Batch size
    md_layer_m_: Layer derivative mean w/o summing over the node for
        input layer
    Sd_layer_m_: Layer derivative mean w/o summing over the node for
        input layer

*NOTE:
    i -> input i.e., input layer (l)
    o -> output i.e., output layer (l + 1)
    n -> next layer after output layer i.e., layer (l + 2)
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int m, l;
    float sum_mean, sum_cov, tmp_md, tmp_cov;
    if (row < no && col < ni * B) {
        // Cross covariance
        sum_mean = 0;
        sum_cov = 0;
        tmp_md = 0;
        tmp_cov = 0;
        for (int k = 0; k < nn; k++) {
            l = k * no * B + (col / ni) * no + row;
            tmp_md = md_layer_m_o[l];
            tmp_cov = md_layer[k + (col / ni) * nn + z_pos_n] *
                      mw_o[row + k * no + w_pos_o] *
                      Cdo_diwi[col + row * ni * B];

            sum_cov += Sd_node[col + row * ni * B] * tmp_md * tmp_md +
                       tmp_cov * tmp_cov +
                       2 * tmp_cov * tmp_md * md_node[col + row * ni * B];
            sum_mean += tmp_cov;
        }

        // Variance
        m = (col / ni) * no + row;
        md_layer_m[ni * B * row + col] =
            sum_mean + md_node[ni * B * row + col] * md_layer[m + z_pos_o];
        Sd_layer_m[ni * B * row + col] =
            sum_cov + Sd_node[ni * B * row + col] * Sd_layer[m + z_pos_o] +
            Sd_layer[m + z_pos_o] * md_node[ni * B * row + col] *
                md_node[ni * B * row + col];
    }
}

__global__ void compute_cov_dz(float const *ma, float const *J, float const *Sz,
                               float const *mw, int act_o, int act_i,
                               int w_pos_i, int z_pos_i, int z_pos_o, int ni,
                               int no, int B, float *Cdi_zi, float *Cdo_zi)
/*Compute covariance between derivatives and hidden states

Args:
    ma: Mean of activation units
    J: Jacobian matrix
    Sz: Variance of hidden states
    mw: Mean of weights
    act_i: Activation function of the input
    act_o: Activation function of the output
    w_pos_i: Weight position for input in the weight vector of network
    z_pos_i: Input-hidden-state position for input in the hidden-state
        vector of network
    z_pos_o: Input-hidden-state position for output in the hidden-state
        vector of network
    ni: Number of hidden units for imputs
    no: Number of hidden units for outputs
    B: Batch size
    Cdi_zi: Covariance between derivative and hidden state of inputs
    Cdo_zi: Covariance between derivative of the ouputs and hidden state of
        inputs
*/
{
    // TODO: Need to create a struct or enum for activation labels
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int k, m;
    if (act_i == 1)  // Tanh
    {
        if (col < ni * B) {
            Cdi_zi[col] = -2.0f * ma[col + z_pos_i] * J[col + z_pos_i] *
                          Sz[col + z_pos_i];
        }

    } else if (act_i == 2)  // sigmoid
    {
        if (col < ni * B) {
            Cdi_zi[col] = (1.0f - 2.0f * ma[col + z_pos_i]) * J[col + z_pos_i] *
                          Sz[col + z_pos_i];
        }
    } else {
        if (col < ni * B) {
            Cdi_zi[col] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        if (row < no && col < ni * B) {
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_zi[ni * B * row + col] = -2.0f * ma[m + z_pos_o] *
                                         mw[k + w_pos_i] * J[col + z_pos_i] *
                                         Sz[col + z_pos_i] * J[m + z_pos_o];
        }
    }

    else if (act_o == 2)  // Sigmoid
    {
        if (row < no && col < ni * B) {
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_zi[ni * B * row + col] = (1.0f - 2.0f * ma[m + z_pos_o]) *
                                         mw[k + w_pos_i] * J[col + z_pos_i] *
                                         Sz[col + z_pos_i] * J[m + z_pos_o];
        }
    } else {
        if (row < no && col < ni * B) {
            Cdo_zi[ni * B * row + col] = 0.0f;
        }
    }
}

__global__ void compute_cov_last_current_layers(
    float const *mw, float const *md_layer, float const *md_node,
    float const *md_layer_m_o, float const *Cdi_zi, float const *Cdo_zi,
    int w_pos_i, int w_pos_o, int z_pos_n, int ni, int no, int nn, int B,
    float *Cld_zi_m)
/*Compute the covariance between final output and the hidden states

Args:
    mw: Mean of weights
    md_layer: Layer derivative mean for the network
    md_node: Derivative mean for each node
    md_layer_m_o: Layer derivative mean w/o summing over the node for
         output layer
    Cdi_zi: Covariance between derivative and hidden state of inputs
    Cdo_zi: Covariance between derivative of the ouputs and hidden state of
        inputs
    w_pos_i: Weight position for input in the weight vector of network
    w_pos_o: Weight position for output in the weight vector of network
    z_pos_n: Input-hidden-state position for l+2 layer in the hidden-state
        vector of network
    ni: Number of hidden units for imputs
    no: Number of hidden units for outputs
    no: Number of hidden units for l+2 layer
    B: Batch size
    Cdi_zi_m: Covariance between final output and the hidden states w/o
        summing over the node
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int l, q;
    float sum, tmp_md;
    if (row < no && col < ni * B) {
        sum = 0;
        for (int k = 0; k < nn; k++) {
            l = k * no * B + (col / ni) * no + row;
            q = (col % ni) + row * ni;
            tmp_md = md_layer_m_o[l];
            sum += tmp_md * Cdi_zi[col] * mw[q + w_pos_i] +
                   Cdo_zi[col + row * ni * B] * md_node[col + row * ni * B] *
                       md_layer[k + (col / ni) * nn + z_pos_n] *
                       mw[row + k * no + w_pos_o];
        }
        Cld_zi_m[ni * B * row + col] = sum;
    }
}

__global__ void compute_cov_last_last_minus_1_layers(
    float const *mw, float const *Cdi_zi, float const *Cdo_zi, int w_pos_i,
    int ni, int no, int B, float *Cld_zi)
/*Compute the covariance between last layer and current layer's  hidden states*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int q;
    if (row < no && col < ni * B) {
        q = (col % ni) + row * ni;
        Cld_zi[ni * B * row + col] =
            Cdi_zi[col + row * ni * B] * mw[q + w_pos_i];
    }
}

__global__ void copy_derivative(float const *md_layer_m, int ni, int no, int nn,
                                int B, float *md_layer_m_o)
/*Copy layer derivative mean from output layer to avoid overwritting it between
   layer b/c we only store the layer derivatives of the input layer*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni * no * B * nn) {
        md_layer_m_o[col] = md_layer_m[col];
    }
}

__global__ void sum_derv(float const *d_layer_m, int ni, int no, int B,
                         int z_pos, float *d_layer)
/*Sum the derivatives over the node (output layer) */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum;
    if (col < B * ni) {
        sum = 0;
        for (int j = 0; j < no; j++) {
            sum += d_layer_m[j * B * ni + col];
        }
        d_layer[col + z_pos] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// ACTIVATION DERIVATIVES
////////////////////////////////////////////////////////////////////////////////
__global__ void tanh_derv(float const *ma, float const *Sa, float const *J,
                          int z_pos, int n, float *mda, float *Sda)
/*Compute mean and variance for the derivatives*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        mda[col + z_pos] = (1 - powf(ma[col + z_pos], 2) - Sa[col + z_pos]);
        Sda[col + z_pos] = (2 * Sa[col + z_pos] *
                            (Sa[col + z_pos] + 2 * powf(ma[col + z_pos], 2)));
    }
}

__global__ void sigmoid_derv(float const *ma, float const *Sa, float const *J,
                             int z_pos, int n, float *mda, float *Sda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        mda[col + z_pos] = J[col + z_pos] - Sa[col + z_pos];
        Sda[col + z_pos] = Sa[col + z_pos] *
                           (2 * Sa[col + z_pos] + 4 * powf(ma[col + z_pos], 2) -
                            4 * ma[col + z_pos] + 1);
    }
}

__global__ void relu_derv(float const *mz, int z_pos, int n, float *mda,
                          float *Sda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        if (mz[col + z_pos] > 0) {
            mda[col + z_pos] = 1.0f;
        } else {
            mda[col + z_pos] = 0.0f;
        }
        Sda[col + z_pos] = 0.0f;
    }
}

__global__ void no_act_derv(int z_pos, int n, float *mda, float *Sda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        mda[col + z_pos] = 1.0f;
        Sda[col + z_pos] = 0.0f;
    }
}

void compute_layer_derv(Network &net, ParamGPU &theta, StateGPU &state,
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

    // Kernels
    // unsigned int blocks = (ni * no * net.batch_size + net.num_gpu_threads -
    // 1) /
    //                       net.num_gpu_threads;
    unsigned int copy_blocks =
        (ni * no * nn * net.batch_size + net.num_gpu_threads - 1) /
        net.num_gpu_threads;
    unsigned int grid_rows =
        (no + net.num_gpu_threads - 1) / net.num_gpu_threads;
    unsigned int grid_cols =
        (ni * net.batch_size + net.num_gpu_threads - 1) / net.num_gpu_threads;
    dim3 dim_grid(grid_cols, grid_rows);
    dim3 dim_block(net.num_gpu_threads, net.num_gpu_threads);

    copy_derivative<<<copy_blocks, net.num_gpu_threads>>>(
        state.derv_state.d_md_layer_m, ni, no, nn, net.batch_size,
        state.derv_state.d_md_layer_m_o);

    compute_node_derv_mean_var_fc<<<dim_grid, dim_block>>>(
        theta.d_mw, theta.d_Sw, state.derv_state.d_mda, state.derv_state.d_Sda,
        w_pos_i, z_pos_i, ni, no, net.batch_size, state.derv_state.d_md_node,
        state.derv_state.d_Sd_node);

    compute_cov_d_dw_fc<<<dim_grid, dim_block>>>(
        state.derv_state.d_mda, state.d_ma, state.d_Sa, state.d_J, theta.d_mw,
        theta.d_Sw, act_i, act_o, w_pos_i, z_pos_i, z_pos_o, ni, no,
        net.batch_size, state.derv_state.d_Cdo_diwi);

    compute_layer_derv_mean_var_fc<<<dim_grid, dim_block>>>(
        state.derv_state.d_md_node, state.derv_state.d_Sd_node,
        state.derv_state.d_md_layer, state.derv_state.d_Sd_layer,
        state.derv_state.d_md_layer_m_o, theta.d_mw,
        state.derv_state.d_Cdo_diwi, w_pos_o, z_pos_o, z_pos_n, ni, no, nn,
        net.batch_size, state.derv_state.d_md_layer_m,
        state.derv_state.d_Sd_layer_m);

    sum_derv<<<grid_cols, net.num_gpu_threads>>>(
        state.derv_state.d_md_layer_m, ni, no, net.batch_size, z_pos_i,
        state.derv_state.d_md_layer);
    sum_derv<<<grid_cols, net.num_gpu_threads>>>(
        state.derv_state.d_Sd_layer_m, ni, no, net.batch_size, z_pos_i,
        state.derv_state.d_Sd_layer);

    compute_cov_dz<<<dim_grid, dim_block>>>(
        state.d_ma, state.d_J, state.d_Sz, theta.d_mw, act_o, act_i, w_pos_i,
        z_pos_i, z_pos_o, ni, no, net.batch_size, state.derv_state.d_Cdi_zi,
        state.derv_state.d_Cdo_zi);

    compute_cov_last_current_layers<<<dim_grid, dim_block>>>(
        theta.d_mw, state.derv_state.d_md_layer, state.derv_state.d_md_node,
        state.derv_state.d_md_layer_m_o, state.derv_state.d_Cdi_zi,
        state.derv_state.d_Cdo_zi, w_pos_i, w_pos_o, z_pos_n, ni, no, nn,
        net.batch_size, state.derv_state.d_Cld_zi_m);

    sum_derv<<<grid_cols, net.num_gpu_threads>>>(
        state.derv_state.d_Cld_zi_m, ni, no, net.batch_size, z_pos_i,
        state.derv_state.d_Cld_zi);
}

void compute_last_layer_minus_1_derv(Network &net, ParamGPU &theta,
                                     StateGPU &state, int curr_layer)
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
    int nn = 1;

    // Kernels
    unsigned int blocks = (ni * no * net.batch_size + net.num_gpu_threads - 1) /
                          net.num_gpu_threads;
    unsigned int copy_blocks =
        (ni * no * nn * net.batch_size + net.num_gpu_threads - 1) /
        net.num_gpu_threads;
    unsigned int grid_rows =
        (no + net.num_gpu_threads - 1) / net.num_gpu_threads;
    unsigned int grid_cols =
        (ni * net.batch_size + net.num_gpu_threads - 1) / net.num_gpu_threads;
    dim3 dim_grid(grid_cols, grid_rows);
    dim3 dim_block(net.num_gpu_threads, net.num_gpu_threads);

    compute_node_derv_mean_var_fc<<<blocks, net.num_gpu_threads>>>(
        theta.d_mw, theta.d_Sw, state.derv_state.d_mda, state.derv_state.d_Sda,
        w_pos_i, z_pos_i, ni, no, net.batch_size, state.derv_state.d_md_node,
        state.derv_state.d_Sd_node);

    sum_derv<<<grid_cols, net.num_gpu_threads>>>(state.derv_state.d_md_node, ni,
                                                 no, net.batch_size, z_pos_i,
                                                 state.derv_state.d_md_layer);
    sum_derv<<<grid_cols, net.num_gpu_threads>>>(state.derv_state.d_Sd_node, ni,
                                                 no, net.batch_size, z_pos_i,
                                                 state.derv_state.d_Sd_layer);

    copy_derivative<<<copy_blocks, net.num_gpu_threads>>>(
        state.derv_state.d_md_node, ni, no, nn, net.batch_size,
        state.derv_state.d_md_layer_m);

    compute_cov_dz<<<dim_grid, dim_block>>>(
        state.d_ma, state.d_J, state.d_Sz, theta.d_mw, act_o, act_i, w_pos_i,
        z_pos_i, z_pos_o, ni, no, net.batch_size, state.derv_state.d_Cdi_zi,
        state.derv_state.d_Cdo_zi);

    compute_cov_last_last_minus_1_layers<<<dim_grid, dim_block>>>(
        theta.d_mw, state.derv_state.d_Cdi_zi, state.derv_state.d_Cdo_zi,
        w_pos_i, ni, no, net.batch_size, state.derv_state.d_Cld_zi_m);

    sum_derv<<<grid_cols, net.num_gpu_threads>>>(
        state.derv_state.d_Cld_zi_m, ni, no, net.batch_size, z_pos_i,
        state.derv_state.d_Cld_zi);
}

void compute_activation_derivatives(Network &net, StateGPU &state, int j) {
    int n = net.nodes[j] * net.batch_size;

    // Kernels
    unsigned int blocks = (n + net.num_gpu_threads - 1) / net.num_gpu_threads;
    if (net.activations[j] == 1)  // tanh
    {
        tanh_derv<<<blocks, net.num_gpu_threads>>>(
            state.d_ma, state.d_Sa, state.d_J, net.z_pos[j], n,
            state.derv_state.d_mda, state.derv_state.d_Sda);

    } else if (net.activations[j] == 2)  // Sigmoid
    {
        sigmoid_derv<<<blocks, net.num_gpu_threads>>>(
            state.d_ma, state.d_Sa, state.d_J, net.z_pos[j], n,
            state.derv_state.d_mda, state.derv_state.d_Sda);

    } else if (net.activations[j] == 4)  // ReLU
    {
        relu_derv<<<blocks, net.num_gpu_threads>>>(state.d_mz, net.z_pos[j], n,
                                                   state.derv_state.d_mda,
                                                   state.derv_state.d_Sda);

    } else if (net.activations[j] == 0)  // No activation
    {
        no_act_derv<<<blocks, net.num_gpu_threads>>>(
            net.z_pos[j], n, state.derv_state.d_mda, state.derv_state.d_Sda);

    } else {
        throw std::invalid_argument(
            "Activation function is invalid -- derivative_cpu.cpp");
    }
}

////////////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////////////
void compute_network_derivatives(Network &net, ParamGPU &theta, StateGPU &state,
                                 int l)
/*Compute derivative of ouput layer's hidden states w.r.t to the hidden states
   of the lth layer

  Args:
    net: Network architecture
    theta: Network's weights and biases
    state: Hidden states of network
*/
{
    // Last layer
    int last_layer = net.layers.size() - 2;
    compute_last_layer_minus_1_derv(net, theta, state, last_layer);

    // Other layers
    for (int k = net.nodes.size() - 3; k >= l; k--) {
        if (net.layers[k + 1] == net.layer_names.fc) {
            compute_layer_derv(net, theta, state, k);
        }
    }
}