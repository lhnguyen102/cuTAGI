#include "../include/derivative_calcul_cpu.h"

void compute_node_derv_mean_var_fc_cpu(
    std::vector<float> &mw, std::vector<float> &Sw, std::vector<float> &mda,
    std::vector<float> &Sda, int w_pos, int z_pos, int ni, int no, int B,
    std::vector<float> &md_node, std::vector<float> &Sd_node)
/* Compute derivatives for each node for fully-connected layer

Args:
    mw: Mean of weights
    Sw: Variance of weights
    mda: Mean of activation derivative w.r.t hidden states
    Sda: Variance of activation derivative w.r.t hidden states
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
    int k;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            k = (i % ni) + j * ni;
            md_node[ni * B * j + i] = mw[k + w_pos] * mda[i + z_pos];
            Sd_node[ni * B * j + i] =
                Sw[k + w_pos] * Sda[i + z_pos] +
                Sw[k + w_pos] * mda[i + z_pos] * mda[i + z_pos] +
                Sda[i + z_pos] * mw[k + w_pos] * mw[k + w_pos];
        }
    }
}

void compute_cov_d_dw_fc_cpu(std::vector<float> &mda, std::vector<float> &ma,
                             std::vector<float> &Sa, std::vector<float> &J,
                             std::vector<float> &mw, std::vector<float> &Sw,
                             int act_i, int act_o, int w_pos_i, int z_pos_i,
                             int z_pos_o, int ni, int no, int B,
                             std::vector<float> &Cdo_diwi)
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
    int k, m;
    float Cao_ai_tmp;
    if (act_i == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i % ni) + j * ni;
                Cao_ai_tmp = mw[k + w_pos_i] * Sa[i + z_pos_i] * J[m + z_pos_o];
                Cdo_diwi[ni * B * j + i] =
                    (2.0f * Cao_ai_tmp * Cao_ai_tmp +
                     4.0f * Cao_ai_tmp * ma[i + z_pos_i] * ma[m + z_pos_o]) *
                    mw[k + w_pos_i];
            }
        }
    } else if (act_i == 2)  // sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i % ni) + j * ni;
                Cao_ai_tmp = mw[k + w_pos_i] * Sa[i + z_pos_i] * J[m + z_pos_o];
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
                k = (i % ni) + j * ni;
                Cdo_diwi[ni * B * j + i] +=
                    (-2.0f * ma[m + z_pos_o] * Sw[k + w_pos_i] *
                     ma[i + z_pos_i] * J[m + z_pos_o]) *
                    mda[i + z_pos_i];
            }
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i % ni) + j * ni;
                Cdo_diwi[ni * B * j + i] +=
                    (1.0f - 2.0f * ma[m + z_pos_o]) *
                    (Sw[k + w_pos_i] * ma[i + z_pos_i] * J[m + z_pos_o]) *
                    mda[i + z_pos_i];
            }
        }
    } else {
    }
}

void compute_layer_derv_mean_var_fc_cpu(
    std::vector<float> &md_node, std::vector<float> &Sd_node,
    std::vector<float> &md_layer, std::vector<float> &Sd_layer,
    std::vector<float> &md_layer_m_o, std::vector<float> &mw_o,
    std::vector<float> &Cdo_diwi, int w_pos_o, int z_pos_o, int z_pos_n, int ni,
    int no, int nn, int B, std::vector<float> &md_layer_m,
    std::vector<float> &Sd_layer_m)
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
                l = k * no * B + (i / ni) * no + j;
                tmp_md = md_layer_m_o[l];
                tmp_cov = md_layer[k + (i / ni) * nn + z_pos_n] *
                          mw_o[j + k * no + w_pos_o] * Cdo_diwi[i + j * ni * B];

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

void compute_cov_dz_cpu(std::vector<float> &ma, std::vector<float> &J,
                        std::vector<float> &Sz, std::vector<float> &mw,
                        int act_o, int act_i, int w_pos_i, int z_pos_i,
                        int z_pos_o, int ni, int no, int B,
                        std::vector<float> &Cdi_zi, std::vector<float> &Cdo_zi)
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
    int k, m;
    if (act_i == 1)  // Tanh
    {
        for (int i = 0; i < ni * B; i++) {
            Cdi_zi[i] =
                -2.0f * ma[i + z_pos_i] * J[i + z_pos_i] * Sz[i + z_pos_i];
        }

    } else if (act_i == 2)  // sigmoid
    {
        for (int i = 0; i < ni * B; i++) {
            Cdi_zi[i] = (1.0f - 2.0f * ma[i + z_pos_i]) * J[i + z_pos_i] *
                        Sz[i + z_pos_i];
        }
    } else {
        for (int i = 0; i < ni * B; i++) {
            Cdi_zi[i] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i % ni) + j * ni;
                Cdo_zi[ni * B * j + i] = -2.0f * ma[m + z_pos_o] *
                                         mw[k + w_pos_i] * J[i + z_pos_i] *
                                         Sz[i + z_pos_i] * J[m + z_pos_o];
            }
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int j = 0; j < no; j++) {
            for (int i = 0; i < ni * B; i++) {
                m = (i / ni) * no + j;
                k = (i % ni) + j * ni;
                Cdo_zi[ni * B * j + i] = (1.0f - 2.0f * ma[m + z_pos_o]) *
                                         mw[k + w_pos_i] * J[i + z_pos_i] *
                                         Sz[i + z_pos_i] * J[m + z_pos_o];
            }
        }
    } else {
        for (int i = 0; i < no * ni * B; i++) {
            Cdo_zi[i] = 0.0f;
        }
    }
}

void compute_cov_last_current_layers_cpu(
    std::vector<float> &mw, std::vector<float> &md_layer,
    std::vector<float> &md_node, std::vector<float> &md_layer_m_o,
    std::vector<float> &Cdi_zi, std::vector<float> &Cdo_zi, int w_pos_i,
    int w_pos_o, int z_pos_n, int ni, int no, int nn, int B,
    std::vector<float> &Cld_zi_m)
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
    int l, q;
    float sum, tmp_md;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            sum = 0;
            for (int k = 0; k < nn; k++) {
                l = k * no * B + (i / ni) * no + j;
                q = (i % ni) + j * ni;
                tmp_md = md_layer_m_o[l];
                sum += tmp_md * Cdi_zi[i] * mw[q + w_pos_i] +
                       Cdo_zi[i + j * ni * B] * md_node[i + j * ni * B] *
                           md_layer[k + (i / ni) * nn + z_pos_n] *
                           mw[j + k * no + w_pos_o];
            }
            Cld_zi_m[ni * B * j + i] = sum;
        }
    }
}

void compute_cov_last_last_minus_1_layers_cpu(std::vector<float> &mw,
                                              std::vector<float> &Cdi_zi,
                                              std::vector<float> &Cdo_zi,
                                              int w_pos_i, int ni, int no,
                                              int B, std::vector<float> &Cld_zi)
/*Compute the covariance between last layer and current layer's  hidden states*/
{
    int q;
    for (int j = 0; j < no; j++) {
        for (int i = 0; i < ni * B; i++) {
            q = (i % ni) + j * ni;
            Cld_zi[ni * B * j + i] = Cdi_zi[i + j * ni * B] * mw[q + w_pos_i];
        }
    }
}

void copy_derv_cpu(std::vector<float> &md_layer_m, int ni, int no, int nn,
                   int B, std::vector<float> &md_layer_m_o)
/*Copy layer derivative mean from output layer to avoid overwritting it between
   layer b/c we only store the layer derivatives of the input layer*/
{
    for (int i = 0; i < ni * no * B * nn; i++) {
        md_layer_m_o[i] = md_layer_m[i];
    }
}

void sum_derv_cpu(std::vector<float> &d_layer_m, int ni, int no, int B,
                  int z_pos, std::vector<float> &d_layer)
/*Sum the derivatives over the node (output layer) */
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
////////////////////////////////////////////////////////////////////////////////
// ACTIVATION DERIVATIVES
////////////////////////////////////////////////////////////////////////////////
void tanh_derv_cpu(std::vector<float> &ma, std::vector<float> &Sa,
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

void sigmoid_derv_cpu(std::vector<float> &ma, std::vector<float> &Sa,
                      std::vector<float> &J, int z_pos, int n,
                      std::vector<float> &mda, std::vector<float> &Sda) {
    for (int i = 0; i < n; i++) {
        mda[i + z_pos] = J[i + z_pos] - Sa[i + z_pos];
        Sda[i + z_pos] =
            Sa[i + z_pos] * (2 * Sa[i + z_pos] + 4 * powf(ma[i + z_pos], 2) -
                             4 * ma[i + z_pos] + 1);
    }
}

void relu_derv_cpu(std::vector<float> &mz, int z_pos, int n,
                   std::vector<float> &mda, std::vector<float> &Sda) {
    for (int i = 0; i < n; i++) {
        if (mz[i + z_pos] > 0) {
            mda[i + z_pos] = 1.0f;
        } else {
            mda[i + z_pos] = 0.0f;
        }
        Sda[i + z_pos] = 0.0f;
    }
}

void no_act_derv_cpu(int z_pos, int n, std::vector<float> &mda,
                     std::vector<float> &Sda) {
    for (int i = 0; i < n; i++) {
        mda[i + z_pos] = 1.0f;
        Sda[i + z_pos] = 0.0f;
    }
}

/////////////////////////////////////////////////////////////////////////////
/// MULTITHREADING VERSION
/////////////////////////////////////////////////////////////////////////////
void node_derv_mean_var_fc_worker(
    std::vector<float> &mw, std::vector<float> &Sw, std::vector<float> &mda,
    std::vector<float> &Sda, int w_pos, int z_pos, int ni, int B, int start_idx,
    int end_idx, std::vector<float> &md_node, std::vector<float> &Sd_node) {
    int k;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (ni * B);
        int col = i % (ni * B);
        k = (col % ni) + row * ni;
        md_node[ni * B * row + col] = mw[k + w_pos] * mda[col + z_pos];
        Sd_node[ni * B * row + col] =
            Sw[k + w_pos] * Sda[col + z_pos] +
            Sw[k + w_pos] * mda[col + z_pos] * mda[col + z_pos] +
            Sda[col + z_pos] * mw[k + w_pos] * mw[k + w_pos];
    }
}

void compute_node_derv_mean_var_fc_mt(
    std::vector<float> &mw, std::vector<float> &Sw, std::vector<float> &mda,
    std::vector<float> &Sda, int w_pos, int z_pos, int ni, int no, int B,
    unsigned int num_threads, std::vector<float> &md_node,
    std::vector<float> &Sd_node)
/*Multithread for computing the node derivative's mean and variance*/
{
    const int tot_ops = ni * no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(node_derv_mean_var_fc_worker, std::ref(mw),
                                 std::ref(Sw), std::ref(mda), std::ref(Sda),
                                 w_pos, z_pos, ni, B, start_idx, end_idx,
                                 std::ref(md_node), std::ref(Sd_node));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void cov_d_dw_fc_worker(std::vector<float> &mda, std::vector<float> &ma,
                        std::vector<float> &Sa, std::vector<float> &J,
                        std::vector<float> &mw, std::vector<float> &Sw,
                        int act_i, int act_o, int w_pos_i, int z_pos_i,
                        int z_pos_o, int ni, int no, int B, int start_idx,
                        int end_idx, std::vector<float> &Cdo_diwi)
/* Worker for computing covariance cov(d+, dw)*/
{
    int k, m;
    float Cao_ai_tmp;
    if (act_i == 1)  // Tanh
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cao_ai_tmp = mw[k + w_pos_i] * Sa[col + z_pos_i] * J[m + z_pos_o];
            Cdo_diwi[ni * B * row + col] =
                (2.0f * Cao_ai_tmp * Cao_ai_tmp +
                 4.0f * Cao_ai_tmp * ma[col + z_pos_i] * ma[m + z_pos_o]) *
                mw[k + w_pos_i];
        }
    } else if (act_i == 2)  // Sigmoid
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cao_ai_tmp = mw[k + w_pos_i] * Sa[col + z_pos_i] * J[m + z_pos_o];
            Cdo_diwi[ni * B * row + col] =
                (Cao_ai_tmp - 2 * Cao_ai_tmp * ma[col + z_pos_i] -
                 2.0f * ma[m + z_pos_o] * Cao_ai_tmp +
                 2.0f * Cao_ai_tmp * Cao_ai_tmp +
                 4.0f * Cao_ai_tmp * ma[col + z_pos_i] * ma[m + z_pos_o]) *
                mw[k + w_pos_i];
        }
    } else {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            Cdo_diwi[ni * B * row + col] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_diwi[ni * B * row + col] +=
                (-2.0f * ma[m + z_pos_o] * Sw[k + w_pos_i] * ma[col + z_pos_i] *
                 J[m + z_pos_o]) *
                mda[col + z_pos_i];
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_diwi[ni * B * row + col] +=
                (1.0f - 2.0f * ma[m + z_pos_o]) *
                (Sw[k + w_pos_i] * ma[col + z_pos_i] * J[m + z_pos_o]) *
                mda[col + z_pos_i];
        }
    }
}

void compute_cov_d_dw_fc_mt(std::vector<float> &mda, std::vector<float> &ma,
                            std::vector<float> &Sa, std::vector<float> &J,
                            std::vector<float> &mw, std::vector<float> &Sw,
                            int act_i, int act_o, int w_pos_i, int z_pos_i,
                            int z_pos_o, int ni, int no, int B,
                            unsigned int num_threads,
                            std::vector<float> &Cdo_diwi)
/*Multithread version for computing the node derivative's mean and variance*/
{
    const int tot_ops = ni * no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(cov_d_dw_fc_worker, std::ref(mda), std::ref(ma),
                        std::ref(Sa), std::ref(J), std::ref(mw), std::ref(Sw),
                        act_i, act_o, w_pos_i, z_pos_i, z_pos_o, ni, no, B,
                        start_idx, end_idx, std::ref(Cdo_diwi));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void layer_derv_mean_var_fc_worker(
    std::vector<float> &md_node, std::vector<float> &Sd_node,
    std::vector<float> &md_layer, std::vector<float> &Sd_layer,
    std::vector<float> &md_layer_m_o, std::vector<float> &mw_o,
    std::vector<float> &Cdo_diwi, int w_pos_o, int z_pos_o, int z_pos_n, int ni,
    int no, int nn, int B, int start_idx, int end_idx,
    std::vector<float> &md_layer_m, std::vector<float> &Sd_layer_m)
/*Worker for computing the layer derivative*/
{
    int m, l;
    float sum_mean, sum_cov, tmp_md, tmp_cov;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (ni * B);
        int col = i % (ni * B);

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

void compute_layer_derv_mean_var_fc_mt(
    std::vector<float> &md_node, std::vector<float> &Sd_node,
    std::vector<float> &md_layer, std::vector<float> &Sd_layer,
    std::vector<float> &md_layer_m_o, std::vector<float> &mw_o,
    std::vector<float> &Cdo_diwi, int w_pos_o, int z_pos_o, int z_pos_n, int ni,
    int no, int nn, int B, unsigned int num_threads,
    std::vector<float> &md_layer_m, std::vector<float> &Sd_layer_m)
/*Multithreading version for computing the layer derivative*/

{
    const int tot_ops = ni * no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            layer_derv_mean_var_fc_worker, std::ref(md_node), std::ref(Sd_node),
            std::ref(md_layer), std::ref(Sd_layer), std::ref(md_layer_m_o),
            std::ref(mw_o), std::ref(Cdo_diwi), w_pos_o, z_pos_o, z_pos_n, ni,
            no, nn, B, start_idx, end_idx, std::ref(md_layer_m),
            std::ref(Sd_layer_m));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void cov_dz_fc_worker(std::vector<float> &ma, std::vector<float> &J,
                      std::vector<float> &Sz, std::vector<float> &mw, int act_i,
                      int act_o, int w_pos_i, int z_pos_i, int z_pos_o, int ni,
                      int no, int B, int start_idx, int end_idx,
                      std::vector<float> &Cdi_zi, std::vector<float> &Cdo_zi)
/*Worker for computing covariance between derivartives and hidden states*/
{
    int k, m;
    if (act_i == 1)  // Tanh
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            Cdi_zi[ni * B * row + col] = -2.0f *
                                         ma[ni * B * row + col + z_pos_i] *
                                         J[ni * B * row + col + z_pos_i] *
                                         Sz[ni * B * row + col + z_pos_i];
        }

    } else if (act_i == 2)  // sigmoid
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            Cdi_zi[ni * B * row + col] =
                (1.0f - 2.0f * ma[ni * B * row + col + z_pos_i]) *
                J[ni * B * row + col + z_pos_i] *
                Sz[ni * B * row + col + z_pos_i];
        }
    } else {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            Cdi_zi[ni * B * row + col] = 0.0f;
        }
    }

    if (act_o == 1)  // Tanh
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_zi[ni * B * row + col] = -2.0f * ma[m + z_pos_o] *
                                         mw[k + w_pos_i] * J[col + z_pos_i] *
                                         Sz[col + z_pos_i] * J[m + z_pos_o];
        }
    } else if (act_o == 2)  // Sigmoid
    {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            m = (col / ni) * no + row;
            k = (col % ni) + row * ni;
            Cdo_zi[ni * B * row + col] = (1.0f - 2.0f * ma[m + z_pos_o]) *
                                         mw[k + w_pos_i] * J[col + z_pos_i] *
                                         Sz[col + z_pos_i] * J[m + z_pos_o];
        }
    } else {
        for (int i = start_idx; i < end_idx; i++) {
            int row = i / (ni * B);
            int col = i % (ni * B);
            Cdo_zi[ni * B * row + col] = 0.0f;
        }
    }
}
void compute_cov_dz_fc_mt(std::vector<float> &ma, std::vector<float> &J,
                          std::vector<float> &Sz, std::vector<float> &mw,
                          int act_o, int act_i, int w_pos_i, int z_pos_i,
                          int z_pos_o, int ni, int no, int B,
                          unsigned int num_threads, std::vector<float> &Cdi_zi,
                          std::vector<float> &Cdo_zi)
/*Multithreading for computing covariance between derivatives and hidden
   states*/
{
    const int tot_ops = ni * no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            cov_dz_fc_worker, std::ref(ma), std::ref(J), std::ref(Sz),
            std::ref(mw), act_i, act_o, w_pos_i, z_pos_i, z_pos_o, ni, no, B,
            start_idx, end_idx, std::ref(Cdi_zi), std::ref(Cdo_zi));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void cov_last_current_layers_worker(
    std::vector<float> &mw, std::vector<float> &md_layer,
    std::vector<float> &md_node, std::vector<float> &md_layer_m_o,
    std::vector<float> &Cdi_zi, std::vector<float> &Cdo_zi, int w_pos_i,
    int w_pos_o, int z_pos_n, int ni, int no, int nn, int B, int start_idx,
    int end_idx, std::vector<float> &Cld_zi_m)
/*Worker for computing the covariance between final output and the hidden
   states*/
{
    int l, q;
    float sum, tmp_md;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (ni * B);
        int col = i % (ni * B);
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

void compute_cov_last_current_layers_mt(
    std::vector<float> &mw, std::vector<float> &md_layer,
    std::vector<float> &md_node, std::vector<float> &md_layer_m_o,
    std::vector<float> &Cdi_zi, std::vector<float> &Cdo_zi, int w_pos_i,
    int w_pos_o, int z_pos_n, int ni, int no, int nn, int B,
    unsigned int num_threads, std::vector<float> &Cld_zi_m)
/*Multithread version for computing the covariance between final output and the
   hidden states*/
{
    const int tot_ops = ni * no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            cov_last_current_layers_worker, std::ref(mw), std::ref(md_layer),
            std::ref(md_node), std::ref(md_layer_m_o), std::ref(Cdi_zi),
            std::ref(Cdo_zi), w_pos_i, w_pos_o, z_pos_n, ni, no, nn, B,
            start_idx, end_idx, std::ref(Cld_zi_m));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void cov_last_layer_minus_1_worker(std::vector<float> &mw,
                                   std::vector<float> &Cdi_zi,
                                   std::vector<float> &Cdo_zi, int w_pos_i,
                                   int ni, int no, int B, int start_idx,
                                   int end_idx, std::vector<float> &Cld_zi)
/*Compute the covariance between last layer and current layer's  hidden states*/
{
    int q;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / (ni * B);
        int col = i % (ni * B);
        q = (col % ni) + row * ni;
        Cld_zi[ni * B * row + col] =
            Cdi_zi[col + row * ni * B] * mw[q + w_pos_i];
    }
}

void compute_cov_last_layer_minus_1_fc_mt(std::vector<float> &mw,
                                          std::vector<float> &Cdi_zi,
                                          std::vector<float> &Cdo_zi,
                                          int w_pos_i, int ni, int no, int B,
                                          unsigned int num_threads,
                                          std::vector<float> &Cld_zi)
/*Multithreading version for computing the covariance between last layer and
   current layer's  hidden states*/
{
    const int tot_ops = ni * no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(cov_last_layer_minus_1_worker, std::ref(mw),
                        std::ref(Cdi_zi), std::ref(Cdo_zi), w_pos_i, ni, no, B,
                        start_idx, end_idx, std::ref(Cld_zi));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void copy_derv_worker(std::vector<float> &md_layer_m, int ni, int no, int nn,
                      int B, int start_idx, int end_idx,
                      std::vector<float> &md_layer_m_o)
/*Worker for coping layer derivative mean from output layer to avoid
   overwritting it between layer*/
{
    for (int i = start_idx; i < end_idx; i++) {
        md_layer_m_o[i] = md_layer_m[i];
    }
}

void copy_derv_mt(std::vector<float> &md_layer_m, int ni, int no, int nn, int B,
                  unsigned int num_threads, std::vector<float> &md_layer_m_o)
/*Multithread version for coping layer derivative mean from output layer to
   avoid overwritting it between layer*/
{
    const int tot_ops = ni * no * B * nn;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(copy_derv_worker, std::ref(md_layer_m), ni, no, nn, B,
                        start_idx, end_idx, std::ref(md_layer_m_o));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void sum_derv_worker(std::vector<float> &d_layer_m, int ni, int no, int B,
                     int z_pos, int start_idx, int end_idx,
                     std::vector<float> &d_layer)
/*Worker for summing the derivatives over the node (output layer)*/
{
    float sum;
    for (int i = start_idx; i < end_idx; i++) {
        sum = 0;
        for (int j = 0; j < no; j++) {
            sum += d_layer_m[j * B * ni + i];
        }
        d_layer[i + z_pos] = sum;
    }
}

void sum_derv_mt(std::vector<float> &d_layer_m, int ni, int no, int B,
                 int z_pos, unsigned int num_threads,
                 std::vector<float> &d_layer)
/*Multithread version for summing the derivatives over the node (output layer)*/
{
    const int tot_ops = ni * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(sum_derv_worker, std::ref(d_layer_m), ni, no, B, z_pos,
                        start_idx, end_idx, std::ref(d_layer));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void tanh_derv_worker(std::vector<float> &ma, std::vector<float> &Sa,
                      std::vector<float> &J, int z_pos, int start_idx,
                      int end_idx, std::vector<float> &mda,
                      std::vector<float> &Sda)
/*Worker for computing mean and variance for the derivatives*/
{
    for (int i = start_idx; i < end_idx; i++) {
        mda[i + z_pos] = (1 - powf(ma[i + z_pos], 2) - Sa[i + z_pos]);
        Sda[i + z_pos] =
            (2 * Sa[i + z_pos] * (Sa[i + z_pos] + 2 * powf(ma[i + z_pos], 2)));
    }
}

void tanh_derv_mt(std::vector<float> &ma, std::vector<float> &Sa,
                  std::vector<float> &J, int z_pos, int n,
                  unsigned int num_threads, std::vector<float> &mda,
                  std::vector<float> &Sda)
/*Multithread versiong for computing mean and variance for the derivatives*/
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(tanh_derv_worker, std::ref(ma), std::ref(Sa),
                                 std::ref(J), z_pos, start_idx, end_idx,
                                 std::ref(mda), std::ref(Sda));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void sigmoid_derv_worker(std::vector<float> &ma, std::vector<float> &Sa,
                         std::vector<float> &J, int z_pos, int start_idx,
                         int end_idx, std::vector<float> &mda,
                         std::vector<float> &Sda) {
    for (int i = start_idx; i < end_idx; i++) {
        mda[i + z_pos] = J[i + z_pos] - Sa[i + z_pos];
        Sda[i + z_pos] =
            Sa[i + z_pos] * (2 * Sa[i + z_pos] + 4 * powf(ma[i + z_pos], 2) -
                             4 * ma[i + z_pos] + 1);
    }
}

void sigmoid_derv_mt(std::vector<float> &ma, std::vector<float> &Sa,
                     std::vector<float> &J, int z_pos, int n,
                     unsigned int num_threads, std::vector<float> &mda,
                     std::vector<float> &Sda) {
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(sigmoid_derv_worker, std::ref(ma),
                                 std::ref(Sa), std::ref(J), z_pos, start_idx,
                                 end_idx, std::ref(mda), std::ref(Sda));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void relu_derv_worker(std::vector<float> &mz, int z_pos, int start_idx,
                      int end_idx, std::vector<float> &mda,
                      std::vector<float> &Sda) {
    for (int i = start_idx; i < end_idx; i++) {
        if (mz[i + z_pos] > 0) {
            mda[i + z_pos] = 1.0f;
        } else {
            mda[i + z_pos] = 0.0f;
        }
        Sda[i + z_pos] = 0.0f;
    }
}

void relu_derv_mt(std::vector<float> &mz, int z_pos, int n,
                  unsigned int num_threads, std::vector<float> &mda,
                  std::vector<float> &Sda) {
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(relu_derv_worker, std::ref(mz), z_pos, start_idx,
                        end_idx, std::ref(mda), std::ref(Sda));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void no_act_derv_worker(int z_pos, int start_idx, int end_idx,
                        std::vector<float> &mda, std::vector<float> &Sda) {
    for (int i = start_idx; i < end_idx; i++) {
        mda[i + z_pos] = 1.0f;
        Sda[i + z_pos] = 0.0f;
    }
}

void no_act_derv_mt(int z_pos, int n, unsigned int num_threads,
                    std::vector<float> &mda, std::vector<float> &Sda) {
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(no_act_derv_worker, z_pos, start_idx, end_idx,
                                 std::ref(mda), std::ref(Sda));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void compute_layer_derv_fc_cpu(Network &net, Param &theta, NetState &state,
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

    if (net.multithreading) {
        // Copy md_layer_m for next layer
        copy_derv_mt(state.derv_state.md_layer_m, ni, no, nn, net.batch_size,
                     net.num_cpu_threads, state.derv_state.md_layer_m_o);

        // Compute node derivatives
        compute_node_derv_mean_var_fc_mt(
            theta.mw, theta.Sw, state.derv_state.mda, state.derv_state.Sda,
            w_pos_i, z_pos_i, ni, no, net.batch_size, net.num_cpu_threads,
            state.derv_state.md_node, state.derv_state.Sd_node);

        // Compute cov(d+, dw)
        compute_cov_d_dw_fc_mt(
            state.derv_state.mda, state.ma, state.Sa, state.J, theta.mw,
            theta.Sw, act_i, act_o, w_pos_i, z_pos_i, z_pos_o, ni, no,
            net.batch_size, net.num_cpu_threads, state.derv_state.Cdo_diwi);

        // Compute layer derivatives
        compute_layer_derv_mean_var_fc_mt(
            state.derv_state.md_node, state.derv_state.Sd_node,
            state.derv_state.md_layer, state.derv_state.Sd_layer,
            state.derv_state.md_layer_m_o, theta.mw, state.derv_state.Cdo_diwi,
            w_pos_o, z_pos_o, z_pos_n, ni, no, nn, net.batch_size,
            net.num_cpu_threads, state.derv_state.md_layer_m,
            state.derv_state.Sd_layer_m);

        sum_derv_mt(state.derv_state.md_layer_m, ni, no, net.batch_size,
                    z_pos_i, net.num_cpu_threads, state.derv_state.md_layer);
        sum_derv_mt(state.derv_state.Sd_layer_m, ni, no, net.batch_size,
                    z_pos_i, net.num_cpu_threads, state.derv_state.Sd_layer);

        // Compute cov(d+, z) & cov(d, z)
        compute_cov_dz_fc_mt(state.ma, state.J, state.Sz, theta.mw, act_o,
                             act_i, w_pos_i, z_pos_i, z_pos_o, ni, no,
                             net.batch_size, net.num_cpu_threads,
                             state.derv_state.Cdi_zi, state.derv_state.Cdo_zi);

        // Compute cov(d_output, z)
        compute_cov_last_current_layers_mt(
            theta.mw, state.derv_state.md_layer, state.derv_state.md_node,
            state.derv_state.md_layer_m_o, state.derv_state.Cdi_zi,
            state.derv_state.Cdo_zi, w_pos_i, w_pos_o, z_pos_n, ni, no, nn,
            net.batch_size, net.num_cpu_threads, state.derv_state.Cld_zi_m);

        sum_derv_mt(state.derv_state.Cld_zi_m, ni, no, net.batch_size, z_pos_i,
                    net.num_cpu_threads, state.derv_state.Cld_zi);
    } else {
        copy_derv_cpu(state.derv_state.md_layer_m, ni, no, nn, net.batch_size,
                      state.derv_state.md_layer_m_o);

        compute_node_derv_mean_var_fc_cpu(
            theta.mw, theta.Sw, state.derv_state.mda, state.derv_state.Sda,
            w_pos_i, z_pos_i, ni, no, net.batch_size, state.derv_state.md_node,
            state.derv_state.Sd_node);

        compute_cov_d_dw_fc_cpu(state.derv_state.mda, state.ma, state.Sa,
                                state.J, theta.mw, theta.Sw, act_i, act_o,
                                w_pos_i, z_pos_i, z_pos_o, ni, no,
                                net.batch_size, state.derv_state.Cdo_diwi);

        compute_layer_derv_mean_var_fc_cpu(
            state.derv_state.md_node, state.derv_state.Sd_node,
            state.derv_state.md_layer, state.derv_state.Sd_layer,
            state.derv_state.md_layer_m_o, theta.mw, state.derv_state.Cdo_diwi,
            w_pos_o, z_pos_o, z_pos_n, ni, no, nn, net.batch_size,
            state.derv_state.md_layer_m, state.derv_state.Sd_layer_m);

        sum_derv_cpu(state.derv_state.md_layer_m, ni, no, net.batch_size,
                     z_pos_i, state.derv_state.md_layer);
        sum_derv_cpu(state.derv_state.Sd_layer_m, ni, no, net.batch_size,
                     z_pos_i, state.derv_state.Sd_layer);

        compute_cov_dz_cpu(state.ma, state.J, state.Sz, theta.mw, act_o, act_i,
                           w_pos_i, z_pos_i, z_pos_o, ni, no, net.batch_size,
                           state.derv_state.Cdi_zi, state.derv_state.Cdo_zi);

        compute_cov_last_current_layers_cpu(
            theta.mw, state.derv_state.md_layer, state.derv_state.md_node,
            state.derv_state.md_layer_m_o, state.derv_state.Cdi_zi,
            state.derv_state.Cdo_zi, w_pos_i, w_pos_o, z_pos_n, ni, no, nn,
            net.batch_size, state.derv_state.Cld_zi_m);

        sum_derv_cpu(state.derv_state.Cld_zi_m, ni, no, net.batch_size, z_pos_i,
                     state.derv_state.Cld_zi);
    }
}

void compute_last_minus_1_layer_derv_fc_cpu(Network &net, Param &theta,
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
    int nn = 1;

    if (net.multithreading) {  // Compute node derivatives
        compute_node_derv_mean_var_fc_mt(
            theta.mw, theta.Sw, state.derv_state.mda, state.derv_state.Sda,
            w_pos_i, z_pos_i, ni, no, net.batch_size, net.num_cpu_threads,
            state.derv_state.md_node, state.derv_state.Sd_node);

        sum_derv_mt(state.derv_state.md_node, ni, no, net.batch_size, z_pos_i,
                    net.num_cpu_threads, state.derv_state.md_layer);
        sum_derv_mt(state.derv_state.Sd_node, ni, no, net.batch_size, z_pos_i,
                    net.num_cpu_threads, state.derv_state.Sd_layer);

        // Copy md_layer_m for next layer
        copy_derv_mt(state.derv_state.md_node, ni, no, nn, net.batch_size,
                     net.num_cpu_threads, state.derv_state.md_layer_m);

        // Compute cov(d+, z) & cov(d, z)
        compute_cov_dz_fc_mt(state.ma, state.J, state.Sz, theta.mw, act_o,
                             act_i, w_pos_i, z_pos_i, z_pos_o, ni, no,
                             net.batch_size, net.num_cpu_threads,
                             state.derv_state.Cdi_zi, state.derv_state.Cdo_zi);

        // Compute cov(d_output, z)
        compute_cov_last_layer_minus_1_fc_mt(
            theta.mw, state.derv_state.Cdi_zi, state.derv_state.Cdo_zi, w_pos_i,
            ni, no, net.batch_size, net.num_cpu_threads,
            state.derv_state.Cld_zi_m);

        sum_derv_mt(state.derv_state.Cld_zi_m, ni, no, net.batch_size, z_pos_i,
                    net.num_cpu_threads, state.derv_state.Cld_zi);
    } else {
        compute_node_derv_mean_var_fc_cpu(
            theta.mw, theta.Sw, state.derv_state.mda, state.derv_state.Sda,
            w_pos_i, z_pos_i, ni, no, net.batch_size, state.derv_state.md_node,
            state.derv_state.Sd_node);

        sum_derv_cpu(state.derv_state.md_node, ni, no, net.batch_size, z_pos_i,
                     state.derv_state.md_layer);
        sum_derv_cpu(state.derv_state.Sd_node, ni, no, net.batch_size, z_pos_i,
                     state.derv_state.Sd_layer);

        copy_derv_cpu(state.derv_state.md_node, ni, no, nn, net.batch_size,
                      state.derv_state.md_layer_m);

        compute_cov_dz_cpu(state.ma, state.J, state.Sz, theta.mw, act_o, act_i,
                           w_pos_i, z_pos_i, z_pos_o, ni, no, net.batch_size,
                           state.derv_state.Cdi_zi, state.derv_state.Cdo_zi);

        compute_cov_last_last_minus_1_layers_cpu(
            theta.mw, state.derv_state.Cdi_zi, state.derv_state.Cdo_zi, w_pos_i,
            ni, no, net.batch_size, state.derv_state.Cld_zi_m);

        sum_derv_cpu(state.derv_state.Cld_zi_m, ni, no, net.batch_size, z_pos_i,
                     state.derv_state.Cld_zi);
    }
}

void compute_activation_derivatives_cpu(Network &net, NetState &state, int j) {
    int n = net.nodes[j] * net.batch_size;
    if (net.activations[j] == 1)  // tanh
    {
        if (net.multithreading) {
            tanh_derv_mt(state.ma, state.Sa, state.J, net.z_pos[j], n,
                         net.num_cpu_threads, state.derv_state.mda,
                         state.derv_state.Sda);
        } else {
            tanh_derv_cpu(state.ma, state.Sa, state.J, net.z_pos[j], n,
                          state.derv_state.mda, state.derv_state.Sda);
        }
    } else if (net.activations[j] == 2)  // Sigmoid
    {
        if (net.multithreading) {
            sigmoid_derv_mt(state.ma, state.Sa, state.J, net.z_pos[j], n,
                            net.num_cpu_threads, state.derv_state.mda,
                            state.derv_state.Sda);
        } else {
            sigmoid_derv_cpu(state.ma, state.Sa, state.J, net.z_pos[j], n,
                             state.derv_state.mda, state.derv_state.Sda);
        }
    } else if (net.activations[j] == 4)  // ReLU
    {
        if (net.multithreading) {
            relu_derv_mt(state.mz, net.z_pos[j], n, net.num_cpu_threads,
                         state.derv_state.mda, state.derv_state.Sda);
        } else {
            relu_derv_cpu(state.mz, net.z_pos[j], n, state.derv_state.mda,
                          state.derv_state.Sda);
        }
    } else if (net.activations[j] == 0)  // No activation
    {
        if (net.multithreading) {
            no_act_derv_mt(net.z_pos[j], n, net.num_cpu_threads,
                           state.derv_state.mda, state.derv_state.Sda);
        } else {
            no_act_derv_cpu(net.z_pos[j], n, state.derv_state.mda,
                            state.derv_state.Sda);
        }
    } else {
        throw std::invalid_argument(
            "Activation function is invalid -- derivative_cpu.cpp");
    }
}

////////////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////////////
void compute_network_derivatives_cpu(Network &net, Param &theta,
                                     NetState &state, int l)
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
    compute_last_minus_1_layer_derv_fc_cpu(net, theta, state, last_layer);

    // Other layers
    for (int k = net.nodes.size() - 3; k >= l; k--) {
        if (net.layers[k + 1] == net.layer_names.fc) {
            compute_layer_derv_fc_cpu(net, theta, state, k);
        }
    }
}