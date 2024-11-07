#include "../include/config.h"
#include "../include/cuda_error_checking.cuh"
#include "../include/custom_logger.h"
#include "../include/data_struct_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// Hidden States
////////////////////////////////////////////////////////////////////////////////
HiddenStateCuda::HiddenStateCuda(size_t size, size_t block_size)
    : BaseHiddenStates(size, block_size)
/*
 */
{
    // Allocate data on gpu device
    this->allocate_memory();
}

HiddenStateCuda::HiddenStateCuda() : BaseHiddenStates() {}

HiddenStateCuda::~HiddenStateCuda()
/*
Free GPU memory using cudaFree
*/
{
    this->deallocate_memory();
}
void HiddenStateCuda::deallocate_memory() {
    cudaFree(this->d_mu_a);
    cudaFree(this->d_var_a);
    cudaFree(this->d_jcb);

    // Reset pointers to nullptr to avoid dangling pointers
    this->d_mu_a = nullptr;
    this->d_var_a = nullptr;
    this->d_jcb = nullptr;
}

void HiddenStateCuda::set_input_x(const std::vector<float> &mu_x,
                                  const std::vector<float> &var_x,
                                  const size_t block_size)
/*
 */
{
    size_t data_size = mu_x.size();
    this->actual_size = data_size / block_size;
    this->block_size = block_size;

    for (int i = 0; i < data_size; i++) {
        this->mu_a[i] = mu_x[i];
        this->jcb[i] = 1.0f;
    }
    if (var_x.size() == data_size) {
        for (int i = 0; i < data_size; i++) {
            this->var_a[i] = var_x[i];
        }
    } else {
        for (int i = 0; i < data_size; i++) {
            this->var_a[i] = 0.0f;
        }
    }
    this->chunks_to_device(data_size);
}

void HiddenStateCuda::allocate_memory() {
    // Check if already allocated, and deallocate if necessary
    if (this->d_mu_a != nullptr || this->d_var_a != nullptr ||
        this->d_jcb != nullptr) {
        this->deallocate_memory();
    }
    // Allocate memory on the GPU using cudaMalloc
    CHECK_CUDA_ERROR(cudaMalloc((void **)&this->d_mu_a, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&this->d_var_a, size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&this->d_jcb, size * sizeof(float)));

    // TODO: Jacobian needs to be intialized at 1.0. Zeros to mu_a and var_a?
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
};

void HiddenStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_a, this->var_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();
}

void HiddenStateCuda::chunks_to_device(const size_t chunk_size)
/*
 */
{
    assert(chunk_size <= this->size);

    cudaMemcpy(this->d_mu_a, this->mu_a.data(), chunk_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_a, this->var_a.data(), chunk_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), chunk_size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void HiddenStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->mu_a.data(), this->d_mu_a,
               this->mu_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_a.data(), this->d_var_a,
               this->var_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb.data(), this->d_jcb, this->jcb.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void HiddenStateCuda::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    // NOTE: hidden state is used as buffer, so we only care the max size of
    // hidden states in the total network in order to store the data. We only
    // reallocate the pointer if the new size is greater than the current size.
    if (new_size > this->size) {
        cudaDeviceSynchronize();

        this->size = new_size;
        this->mu_a.resize(this->size, 0.0f);
        this->var_a.resize(this->size, 0.0f);
        this->jcb.resize(this->size, 1.0f);

        this->deallocate_memory();
        this->allocate_memory();
    }
    // The actual size and block size need to be updated because these sizes
    // will be required between layers during forward pass
    this->block_size = new_block_size;
    this->actual_size = new_size / new_block_size;
}

void HiddenStateCuda::swap(BaseHiddenStates &other) {
    HiddenStateCuda *cu_other = dynamic_cast<HiddenStateCuda *>(&other);
    if (cu_other) {
        BaseHiddenStates::swap(other);
        std::swap(this->d_mu_a, cu_other->d_mu_a);
        std::swap(this->d_var_a, cu_other->d_var_a);
        std::swap(this->d_jcb, cu_other->d_jcb);
    } else {
        LOG(LogLevel::ERROR, "Swap input invalid.");
    }
}

void HiddenStateCuda::copy_from(const BaseHiddenStates &source, int num_data)
/*
 */
{
    if (num_data == -1) {
        num_data = std::min(this->size, source.size);
    }

    const HiddenStateCuda *cu_source =
        dynamic_cast<const HiddenStateCuda *>(&source);

    if (!cu_source) {
        LOG(LogLevel::ERROR, "Invalid source.");
    }

    cudaMemcpy(this->d_mu_a, cu_source->d_mu_a, num_data * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->d_var_a, cu_source->d_var_a, num_data * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->d_jcb, cu_source->d_jcb, num_data * sizeof(float),
               cudaMemcpyDeviceToDevice);

    CHECK_LAST_CUDA_ERROR();

    this->block_size = source.block_size;
    this->actual_size = source.actual_size;
    this->width = source.width;
    this->height = source.height;
    this->depth = source.depth;
}

////////////////////////////////////////////////////////////////////////////////
// Delta Hidden States
////////////////////////////////////////////////////////////////////////////////
DeltaStateCuda::DeltaStateCuda(size_t size, size_t block_size)
    : BaseDeltaStates(size, block_size)
/*
 */
{
    // Allocate data on gpu device
    this->allocate_memory();
}

DeltaStateCuda::DeltaStateCuda() : BaseDeltaStates() {}

DeltaStateCuda::~DeltaStateCuda()
/*
 */
{
    this->deallocate_memory();
}

void DeltaStateCuda::deallocate_memory() {
    cudaFree(this->d_delta_mu);
    cudaFree(this->d_delta_var);

    CHECK_LAST_CUDA_ERROR();

    // Reset pointers to nullptr to avoid dangling pointers
    this->d_delta_mu = nullptr;
    this->d_delta_var = nullptr;
}

void DeltaStateCuda::allocate_memory()
/*
 */
{
    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc(&this->d_delta_mu, size * sizeof(float));
    cudaMalloc(&this->d_delta_var, size * sizeof(float));

    CHECK_LAST_CUDA_ERROR();
}

void DeltaStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_delta_mu, this->delta_mu.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_delta_var, this->delta_var.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();
}

void DeltaStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->delta_mu.data(), this->d_delta_mu,
               this->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->delta_var.data(), this->d_delta_var,
               this->size * sizeof(float), cudaMemcpyDeviceToHost);
}

void DeltaStateCuda::reset_zeros() {
    cudaMemset(d_delta_mu, 0, sizeof(float) * size);
    cudaMemset(d_delta_var, 0, sizeof(float) * size);
}

void DeltaStateCuda::copy_from(const BaseDeltaStates &source, int num_data)
/*
 */
{
    if (num_data == -1) {
        num_data = this->size;
    }

    const DeltaStateCuda *cu_source =
        dynamic_cast<const DeltaStateCuda *>(&source);

    if (!cu_source) {
        LOG(LogLevel::ERROR, "Invalid source.");
    }

    cudaMemcpy(this->d_delta_mu, cu_source->d_delta_mu,
               num_data * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(this->d_delta_var, cu_source->d_delta_var,
               num_data * sizeof(float), cudaMemcpyDeviceToDevice);

    CHECK_LAST_CUDA_ERROR();

    this->block_size = source.block_size;
}

void DeltaStateCuda::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    // Same as HiddenStateCuda
    if (new_size > this->size) {
        cudaDeviceSynchronize();
        this->size = new_size;
        this->reset_zeros();
        this->deallocate_memory();
        this->allocate_memory();
    }
    this->block_size = new_block_size;
    this->actual_size = new_size / new_block_size;
}

void DeltaStateCuda::swap(BaseDeltaStates &other) {
    DeltaStateCuda *cu_other = dynamic_cast<DeltaStateCuda *>(&other);
    if (cu_other) {
        BaseDeltaStates::swap(other);
        std::swap(this->d_delta_mu, cu_other->d_delta_mu);
        std::swap(this->d_delta_var, cu_other->d_delta_var);
    } else {
        LOG(LogLevel::ERROR, "Swap input invalid.");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Temporary Hidden States
////////////////////////////////////////////////////////////////////////////////
TempStateCuda::TempStateCuda(size_t size, size_t block_size)
    : BaseTempStates(size, block_size)
/*
 */
{
    // Allocate memory on the GPU using cudaMalloc
    this->allocate_memory();
}

TempStateCuda::TempStateCuda() : BaseTempStates() {}

TempStateCuda::~TempStateCuda()
/*
 */
{
    this->deallocate_memory();
}

void TempStateCuda::deallocate_memory()
/*
 */
{
    cudaFree(this->d_tmp_1);
    cudaFree(this->d_tmp_2);

    CHECK_LAST_CUDA_ERROR();

    this->d_tmp_1 = nullptr;
    this->d_tmp_2 = nullptr;
}

void TempStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_tmp_1, this->tmp_1.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_tmp_2, this->tmp_2.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void TempStateCuda::allocate_memory()
/*
 */
{
    cudaMalloc(&this->d_tmp_1, size * sizeof(float));
    cudaMalloc(&this->d_tmp_2, size * sizeof(float));
}

void TempStateCuda::to_host() {
    cudaMemcpy(this->tmp_1.data(), this->d_tmp_1, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->tmp_2.data(), this->d_tmp_2, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}

void TempStateCuda::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    if (new_size > this->size) {
        cudaDeviceSynchronize();
        this->size = new_size;
        this->tmp_1.resize(this->size, 0.0f);
        this->tmp_2.resize(this->size, 0.0f);

        this->deallocate_memory();
        this->allocate_memory();
    }
    this->size = new_size;
    this->block_size = new_block_size;
}

////////////////////////////////////////////////////////////////////////////////
// Backward States
////////////////////////////////////////////////////////////////////////////////

BackwardStateCuda::BackwardStateCuda() {}
BackwardStateCuda::~BackwardStateCuda()
/*
 */
{
    this->deallocate_memory();
}

void BackwardStateCuda::deallocate_memory()
/*
 */
{
    cudaFree(this->d_mu_a);
    cudaFree(this->d_jcb);
    this->d_mu_a = nullptr;
    this->d_jcb = nullptr;
}

void BackwardStateCuda::allocate_memory()
/*
 */
{
    if (this->d_mu_a != nullptr || this->d_jcb != nullptr) {
        this->deallocate_memory();
    }
    this->mu_a.resize(this->size, 0.0f);
    this->jcb.resize(this->size, 1.0f);
    cudaMalloc(&this->d_mu_a, this->size * sizeof(float));
    cudaMalloc(&this->d_jcb, this->size * sizeof(float));
    this->to_device();
    CHECK_LAST_CUDA_ERROR();
}

void BackwardStateCuda::to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_a, this->mu_a.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_jcb, this->jcb.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
}

void BackwardStateCuda::to_host()
/*
 */
{
    cudaMemcpy(this->mu_a.data(), this->d_mu_a, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb.data(), this->d_jcb, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}

void BackwardStateCuda::set_size(size_t new_size)
/*
 */
{
    if (new_size > this->size) {
        cudaDeviceSynchronize();
        this->size = new_size;
        this->mu_a.resize(new_size, 0.0);
        this->jcb.resize(new_size, 1.0f);

        this->deallocate_memory();
        this->allocate_memory();
    }
}

////////////////////////////////////////////////////////////////////////////////
// Observation
////////////////////////////////////////////////////////////////////////////////
ObservationCuda::ObservationCuda() {}
ObservationCuda::~ObservationCuda()
/*
 */
{
    this->deallocate_memory();
}

void ObservationCuda::deallocate_memory()
/*
 */
{
    cudaFree(d_mu_obs);
    cudaFree(d_var_obs);
    cudaFree(d_selected_idx);

    CHECK_LAST_CUDA_ERROR();

    this->d_mu_obs = nullptr;
    this->d_var_obs = nullptr;
    this->d_selected_idx = nullptr;
}

void ObservationCuda::allocate_memory() {
    cudaMalloc(&this->d_mu_obs, this->size * sizeof(float));
    cudaMalloc(&this->d_var_obs, this->size * sizeof(float));

    if (this->idx_size != 0) {
        cudaMalloc(&this->d_selected_idx, this->idx_size * sizeof(int));
    }

    CHECK_LAST_CUDA_ERROR();
}

void ObservationCuda::to_device() {
    cudaMemcpy(this->d_mu_obs, this->mu_obs.data(), this->size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_obs, this->var_obs.data(),
               this->size * sizeof(float), cudaMemcpyHostToDevice);
    if (this->idx_size != 0) {
        cudaMemcpy(this->d_selected_idx, this->selected_idx.data(),
                   this->size * sizeof(int), cudaMemcpyHostToDevice);
    }

    CHECK_LAST_CUDA_ERROR();
}

void ObservationCuda::to_host() {
    cudaMemcpy(this->mu_obs.data(), this->d_mu_obs, this->size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_obs.data(), this->d_var_obs,
               this->size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->selected_idx.data(), this->d_selected_idx,
               this->size * sizeof(int), cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}
void ObservationCuda::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    if (new_size > this->size) {
        cudaDeviceSynchronize();
        this->size = size;

        this->deallocate_memory();
        this->allocate_memory();
    }
    this->block_size = new_block_size;
    this->actual_size = new_size / new_block_size;
}

////////////////////////////////////////////////////////////////////////////////
// LSTM states
////////////////////////////////////////////////////////////////////////////////
LSTMStateCuda::LSTMStateCuda() {}
LSTMStateCuda::LSTMStateCuda(size_t num_states, size_t num_inputs)
    : BaseLSTMStates(num_states, num_inputs)
/*
 */
{
    this->allocate_memory();
}

LSTMStateCuda::~LSTMStateCuda()
/*
 */
{
    this->deallocate_memory();
}

void LSTMStateCuda::deallocate_memory()
/*
 */
{
    cudaFree(d_mu_ha);
    d_mu_ha = nullptr;
    cudaFree(d_var_ha);
    d_var_ha = nullptr;
    cudaFree(d_mu_f_ga);
    d_mu_f_ga = nullptr;
    cudaFree(d_var_f_ga);
    d_var_f_ga = nullptr;
    cudaFree(d_jcb_f_ga);
    d_jcb_f_ga = nullptr;
    cudaFree(d_mu_i_ga);
    d_mu_i_ga = nullptr;
    cudaFree(d_var_i_ga);
    d_var_i_ga = nullptr;
    cudaFree(d_jcb_i_ga);
    d_jcb_i_ga = nullptr;
    cudaFree(d_mu_c_ga);
    d_mu_c_ga = nullptr;
    cudaFree(d_var_c_ga);
    d_var_c_ga = nullptr;
    cudaFree(d_jcb_c_ga);
    d_jcb_c_ga = nullptr;
    cudaFree(d_mu_o_ga);
    d_mu_o_ga = nullptr;
    cudaFree(d_var_o_ga);
    d_var_o_ga = nullptr;
    cudaFree(d_jcb_o_ga);
    d_jcb_o_ga = nullptr;
    cudaFree(d_mu_ca);
    d_mu_ca = nullptr;
    cudaFree(d_var_ca);
    d_var_ca = nullptr;
    cudaFree(d_jcb_ca);
    d_jcb_ca = nullptr;
    cudaFree(d_mu_c);
    d_mu_c = nullptr;
    cudaFree(d_var_c);
    d_var_c = nullptr;
    cudaFree(d_mu_c_prev);
    d_mu_c_prev = nullptr;
    cudaFree(d_var_c_prev);
    d_var_c_prev = nullptr;
    cudaFree(d_mu_h_prev);
    d_mu_h_prev = nullptr;
    cudaFree(d_var_h_prev);
    d_var_h_prev = nullptr;
    cudaFree(d_cov_i_c);
    d_cov_i_c = nullptr;
    cudaFree(d_cov_o_tanh_c);
    d_cov_o_tanh_c = nullptr;

    // Prior for hidden and cell states
    cudaFree(d_mu_c_prior);
    d_mu_c_prior = nullptr;
    cudaFree(d_var_c_prior);
    d_var_c_prior = nullptr;
    cudaFree(d_mu_h_prior);
    d_mu_h_prior = nullptr;
    cudaFree(d_var_h_prior);
    d_var_h_prior = nullptr;

    CHECK_LAST_CUDA_ERROR();
}

void LSTMStateCuda::set_num_states(size_t num_states, size_t num_inputs)
/*
 */
{
    this->num_states = num_states;
    this->num_inputs = num_inputs;
    this->reset_zeros();

    this->deallocate_memory();
    this->allocate_memory();
}

void LSTMStateCuda::allocate_memory()
/*
 */
{
    size_t size =
        ((num_states + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE * sizeof(float);
    size_t size_ha = ((num_states + num_inputs + PACK_SIZE - 1) / PACK_SIZE) *
                     PACK_SIZE * sizeof(float);

    cudaMalloc((void **)&d_mu_ha, size_ha);
    cudaMalloc((void **)&d_var_ha, size_ha);

    cudaMalloc((void **)&d_mu_f_ga, size);
    cudaMalloc((void **)&d_var_f_ga, size);
    cudaMalloc((void **)&d_jcb_f_ga, size);

    cudaMalloc((void **)&d_mu_i_ga, size);
    cudaMalloc((void **)&d_var_i_ga, size);
    cudaMalloc((void **)&d_jcb_i_ga, size);

    cudaMalloc((void **)&d_mu_c_ga, size);
    cudaMalloc((void **)&d_var_c_ga, size);
    cudaMalloc((void **)&d_jcb_c_ga, size);

    cudaMalloc((void **)&d_mu_o_ga, size);
    cudaMalloc((void **)&d_var_o_ga, size);
    cudaMalloc((void **)&d_jcb_o_ga, size);

    cudaMalloc((void **)&d_mu_ca, size);
    cudaMalloc((void **)&d_var_ca, size);
    cudaMalloc((void **)&d_jcb_ca, size);

    cudaMalloc((void **)&d_mu_c, size);
    cudaMalloc((void **)&d_var_c, size);

    cudaMalloc((void **)&d_mu_c_prev, size);
    cudaMalloc((void **)&d_var_c_prev, size);

    cudaMalloc((void **)&d_mu_h_prev, size);
    cudaMalloc((void **)&d_var_h_prev, size);

    cudaMalloc((void **)&d_cov_i_c, size);
    cudaMalloc((void **)&d_cov_o_tanh_c, size);

    cudaMalloc((void **)&d_mu_c_prior, size);
    cudaMalloc((void **)&d_var_c_prior, size);

    cudaMalloc((void **)&d_mu_h_prior, size);
    cudaMalloc((void **)&d_var_h_prior, size);

    // TODO: do we need to clear out all data
    this->to_device();
}

void LSTMStateCuda::to_device() {
    // Copy mu_ha and var_ha
    cudaMemcpy(d_mu_ha, this->mu_ha.data(),
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_ha, this->var_ha.data(),
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyHostToDevice);

    // Copy mu_f_ga and var_f_ga
    cudaMemcpy(d_mu_f_ga, this->mu_f_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_f_ga, this->var_f_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_f_ga, this->jcb_f_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_i_ga, var_i_ga, and jcb_i_ga
    cudaMemcpy(d_mu_i_ga, this->mu_i_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_i_ga, this->var_i_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_i_ga, this->jcb_i_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_c_ga, var_c_ga, and jcb_c_ga
    cudaMemcpy(d_mu_c_ga, this->mu_c_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c_ga, this->var_c_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_c_ga, this->jcb_c_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_o_ga, var_o_ga, and jcb_o_ga
    cudaMemcpy(d_mu_o_ga, this->mu_o_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_o_ga, this->var_o_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_o_ga, this->jcb_o_ga.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_ca, var_ca, and jcb_ca
    cudaMemcpy(d_mu_ca, this->mu_ca.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_ca, this->var_ca.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_jcb_ca, this->jcb_ca.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);

    // Copy mu_c, var_c, mu_c_prev, and var_c_prev
    cudaMemcpy(d_mu_c, this->mu_c.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c, this->var_c.data(), this->num_states * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_mu_c_prev, this->mu_c_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c_prev, this->var_c_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy mu_h_prev and var_h_prev
    cudaMemcpy(d_mu_h_prev, this->mu_h_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_h_prev, this->var_h_prev.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Copy cov_i_c and cov_o_tanh_c
    cudaMemcpy(d_cov_i_c, this->cov_i_c.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cov_o_tanh_c, this->cov_o_tanh_c.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    // Prior for cell and hidden states
    cudaMemcpy(d_mu_c_prior, this->mu_c_prior.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_c_prior, this->var_c_prior.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mu_h_prior, this->mu_h_prior.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_var_h_prior, this->var_h_prior.data(),
               this->num_states * sizeof(float), cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();
}

void LSTMStateCuda::to_host()
/*
 */
{
    // Copy back mu_ha and var_ha
    cudaMemcpy(this->mu_ha.data(), d_mu_ha,
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ha.data(), d_var_ha,
               (this->num_states + this->num_inputs) * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Copy back mu_f_ga and var_f_ga
    cudaMemcpy(this->mu_f_ga.data(), d_mu_f_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_f_ga.data(), d_var_f_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back jcb_f_ga
    cudaMemcpy(this->jcb_f_ga.data(), d_jcb_f_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_i_ga, var_i_ga, and jcb_i_ga
    cudaMemcpy(this->mu_i_ga.data(), d_mu_i_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_i_ga.data(), d_var_i_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_i_ga.data(), d_jcb_i_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_c_ga, var_c_ga, and jcb_c_ga
    cudaMemcpy(this->mu_c_ga.data(), d_mu_c_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c_ga.data(), d_var_c_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_c_ga.data(), d_jcb_c_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_o_ga, var_o_ga, and jcb_o_ga
    cudaMemcpy(this->mu_o_ga.data(), d_mu_o_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_o_ga.data(), d_var_o_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_o_ga.data(), d_jcb_o_ga,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_ca, var_ca, and jcb_ca
    cudaMemcpy(this->mu_ca.data(), d_mu_ca, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ca.data(), d_var_ca, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->jcb_ca.data(), d_jcb_ca, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Copy back mu_c, var_c, mu_c_prev, and var_c_prev
    cudaMemcpy(this->mu_c.data(), d_mu_c, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c.data(), d_var_c, this->num_states * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_c_prev.data(), d_mu_c_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c_prev.data(), d_var_c_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back mu_h_prev and var_h_prev
    cudaMemcpy(this->mu_h_prev.data(), d_mu_h_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_h_prev.data(), d_var_h_prev,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy back cov_i_c and cov_o_tanh_c
    cudaMemcpy(this->cov_i_c.data(), d_cov_i_c,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->cov_o_tanh_c.data(), d_cov_o_tanh_c,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    // Prior for cell and hidden states
    cudaMemcpy(this->mu_c_prior.data(), d_mu_c_prior,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_c_prior.data(), d_var_c_prior,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_h_prior.data(), d_mu_h_prior,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_h_prior.data(), d_var_h_prior,
               this->num_states * sizeof(float), cudaMemcpyDeviceToHost);

    CHECK_LAST_CUDA_ERROR();
}