#pragma once
#include "base_layer.h"

struct Pool2dIndex {
    std::vector<int> pool_idx, z_ud_idx;
    int w, h;
};

Pool2dIndex get_pool_index(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out);

class AvgPool2d : public BaseLayer {
   public:
    size_t kernel_size = 0;
    int stride = 0;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> pool_idx, z_ud_idx;
    size_t row_zw = 0, col_z_ud = 0;
    bool overlap = true;

    AvgPool2d(size_t kernel_size, int stride = -1, int padding = 0,
              int padding_type = 0);

    virtual ~AvgPool2d();

    // Delete copy constructor and copy assignment
    AvgPool2d(const AvgPool2d &) = delete;
    AvgPool2d &operator=(const AvgPool2d &) = delete;

    // Optionally implement move constructor and move assignment
    AvgPool2d(AvgPool2d &&) = default;
    AvgPool2d &operator=(AvgPool2d &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    using BaseLayer::storing_states_for_training;
    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void preinit_layer() override;

   protected:
    void lazy_index_init();
};

////////////////////////////////////////////////////////////////////////////////
// Pool2d Backward and Forward
////////////////////////////////////////////////////////////////////////////////
void avgpool2d_fwd_overlapped_mean_var(const std::vector<float> &mu_a,
                                       const std::vector<float> &var_a,
                                       const std::vector<int> &a_idx, int woho,
                                       int wihi, int ki, int k, int pad_idx,
                                       int start_chunk, int end_chunk,
                                       std::vector<float> &mu_z,
                                       std::vector<float> &var_z);

void avgpool2d_fwd_mean_var(const std::vector<float> &mu_a,
                            const std::vector<float> &var_a,
                            const std::vector<int> a_idx, int woho, int wihi,
                            int ki, int k, int start_chunk, int end_chunk,
                            std::vector<float> &mu_z,
                            std::vector<float> &var_z);

void avgpool2d_bwd_overlapped_delta_z(
    const std::vector<float> &jcb, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &z_ud_idx,
    int woho, int wihi, int ki, int n, int k, int pad_idx, int start_chunk,
    int end_chunk, std::vector<float> &delta_mu, std::vector<float> &delta_var);

void avgpool2d_bwd_delta_z(const std::vector<float> &jcb,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out, int wo,
                           int ki, int k, int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var);

void avgpool2d_fwd_overlapped_mean_var_mp(const std::vector<float> &mu_a,
                                          const std::vector<float> &var_a,
                                          const std::vector<int> &a_idx,
                                          int woho, int wihi, int ki, int k,
                                          int pad_idx, unsigned int num_threads,
                                          std::vector<float> &mu_z,
                                          std::vector<float> &var_z);

void avgpool2d_fwd_mean_var_mp(const std::vector<float> &mu_a,
                               const std::vector<float> &var_a,
                               const std::vector<int> a_idx, int woho, int wihi,
                               int ki, int k, unsigned int num_threads,
                               std::vector<float> &mu_z,
                               std::vector<float> &var_z);

void avgpool2d_bwd_overlapped_delta_z_mp(
    const std::vector<float> &jcb, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &z_ud_idx,
    int woho, int wihi, int ki, int n, int k, int pad_idx,
    unsigned int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var);

void avgpool2d_bwd_delta_z_mp(const std::vector<float> &jcb,
                              const std::vector<float> &delta_mu_out,
                              const std::vector<float> &delta_var_out, int wo,
                              int ki, int k, unsigned int num_threads,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var);