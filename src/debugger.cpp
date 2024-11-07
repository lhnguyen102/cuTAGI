#include "../include/debugger.h"

#include <memory>
#include <vector>

#include "../include/data_struct.h"
#ifdef USE_CUDA
#include "../include/base_layer_cuda.cuh"
#include "../include/data_struct_cuda.cuh"
#endif

ModelDebugger::ModelDebugger(Sequential &test_model, Sequential &ref_model)
    : cpu_output_updater("cpu"), cuda_output_updater("cuda") {
    this->test_model = test_model;
    this->ref_model = ref_model;
}

ModelDebugger::~ModelDebugger() {}

void ModelDebugger::lazy_init(int batch_size, int z_buffer_size)
/*
 */
{
    if (test_model.device.compare("cpu") == 0) {
        test_output_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        test_input_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        test_temp_states =
            std::make_shared<BaseTempStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (test_model.device.compare("cuda") == 0) {
        test_output_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        test_input_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        test_temp_states =
            std::make_shared<TempStateCuda>(z_buffer_size, batch_size);
    }
#endif

    if (ref_model.device.compare("cpu") == 0) {
        ref_output_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        ref_input_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        ref_temp_states =
            std::make_shared<BaseTempStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        ref_output_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        ref_input_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        ref_temp_states =
            std::make_shared<TempStateCuda>(z_buffer_size, batch_size);
    }
#endif

    if (test_model.device.compare("cpu") == 0) {
        this->test_output_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
        this->test_input_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (test_model.device.compare("cuda") == 0) {
        this->test_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        this->test_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
#endif

    if (ref_model.device.compare("cpu") == 0) {
        ref_output_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
        ref_input_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
    }
#ifdef USE_CUDA
    else if (ref_model.device.compare("cuda") == 0) {
        ref_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        ref_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
#endif
}

void ModelDebugger::debug_forward(const std::vector<float> &mu_x,
                                  const std::vector<float> &var_x)
/*
 */
{
    int batch_size = mu_x.size() / test_model.layers.front()->input_size;
    int z_buffer_size = batch_size * test_model.z_buffer_size;

    this->lazy_init(batch_size, z_buffer_size);

    // Merge input data to the input buffer
    this->test_input_z_buffer->set_input_x(mu_x, var_x, batch_size);
    this->ref_input_z_buffer->set_input_x(mu_x, var_x, batch_size);

    int num_layers = this->test_model.layers.size();

    for (int i = 0; i < num_layers; i++) {
        auto *test_current_layer = this->test_model.layers[i].get();
        auto *ref_current_layer = this->ref_model.layers[i].get();

        test_current_layer->forward(*this->test_input_z_buffer,
                                    *this->test_output_z_buffer,
                                    *this->test_temp_states);

        ref_current_layer->forward(*this->ref_input_z_buffer,
                                   *this->ref_output_z_buffer,
                                   *this->ref_temp_states);

        // Copy to host for gpu model
#ifdef USE_CUDA
        if (this->test_model.device.compare("cuda") == 0) {
            HiddenStateCuda *test_output_z_buffer_cu =
                dynamic_cast<HiddenStateCuda *>(
                    this->test_output_z_buffer.get());
            test_output_z_buffer_cu->to_host();
        }
        if (this->ref_model.device.compare("cuda") == 0) {
            HiddenStateCuda *ref_output_z_buffer_cu =
                dynamic_cast<HiddenStateCuda *>(
                    this->ref_output_z_buffer.get());
            ref_output_z_buffer_cu->to_host();
        }
#endif

        // Test here
        for (int j = 0; j < test_current_layer->output_size * batch_size; j++) {
            if (this->test_output_z_buffer->mu_a[j] !=
                this->ref_output_z_buffer->mu_a[j]) {
                auto layer_name = test_current_layer->get_layer_name();
                std::cout << "Layer name: " << layer_name << " " << "Layer no "
                          << i << "\n"
                          << std::endl;
                // std::vector<float> test_mu_ra, test_var_ra, ref_mu_ra,
                //     ref_var_ra;

                // std::tie(test_mu_ra, test_var_ra) =
                //     test_current_layer->get_running_mean_var();

                // std::tie(ref_mu_ra, ref_var_ra) =
                //     test_current_layer->get_running_mean_var();

                int check = 1;
                break;
            }
        }
        std::swap(this->test_input_z_buffer, this->test_output_z_buffer);
        std::swap(this->ref_input_z_buffer, this->ref_output_z_buffer);
    }
    // Output buffer is considered as the final output of network
    std::swap(this->test_output_z_buffer, this->test_input_z_buffer);
    std::swap(this->ref_output_z_buffer, this->ref_input_z_buffer);
    int check = 1;
}

void ModelDebugger::debug_backward(std::vector<float> &y_batch,
                                   std::vector<float> &var_obs,
                                   std::vector<int> &idx_ud_batch)
/*
 */
{
    int batch_size = this->ref_output_z_buffer->block_size;
    // Output layer
    if (idx_ud_batch.size() != 0) {
        if (this->test_model.device.compare("cpu") == 0) {
            this->cpu_output_updater.update_using_indices(
                *this->test_output_z_buffer, y_batch, var_obs, idx_ud_batch,
                *this->test_input_delta_z_buffer);
        }
#ifdef USE_CUDA
        else {
            this->cuda_output_updater.update_using_indices(
                *this->test_output_z_buffer, y_batch, var_obs, idx_ud_batch,
                *this->test_input_delta_z_buffer);
        }
#endif

        if (this->ref_model.device.compare("cpu") == 0) {
            this->cpu_output_updater.update_using_indices(
                *this->ref_output_z_buffer, y_batch, var_obs, idx_ud_batch,
                *this->ref_input_delta_z_buffer);
        }
#ifdef USE_CUDA
        else {
            this->cuda_output_updater.update_using_indices(
                *this->ref_output_z_buffer, y_batch, var_obs, idx_ud_batch,
                *this->ref_input_delta_z_buffer);
        }
#endif
    } else {
        if (this->test_model.device.compare("cpu") == 0) {
            this->cpu_output_updater.update(*this->test_output_z_buffer,
                                            y_batch, var_obs,
                                            *this->test_input_delta_z_buffer);
        }
#ifdef USE_CUDA
        else {
            this->cuda_output_updater.update(*this->test_output_z_buffer,
                                             y_batch, var_obs,
                                             *this->test_input_delta_z_buffer);
        }
#endif

        if (this->ref_model.device.compare("cpu") == 0) {
            this->cpu_output_updater.update(*this->ref_output_z_buffer, y_batch,
                                            var_obs,
                                            *this->ref_input_delta_z_buffer);
        }
#ifdef USE_CUDA
        else {
            this->cuda_output_updater.update(*this->ref_output_z_buffer,
                                             y_batch, var_obs,
                                             *this->ref_input_delta_z_buffer);
        }
#endif
    }

    int num_layers = test_model.layers.size();

    for (int i = num_layers - 1; i > 0; i--) {
        auto *test_current_layer = test_model.layers[i].get();
        auto *ref_current_layer = ref_model.layers[i].get();

        // // Backward pass for parameters and hidden states
        // if (test_model.param_update) {
        //     test_current_layer->param_backward(*test_current_layer->bwd_states,
        //                                        *test_input_delta_z_buffer,
        //                                        *test_temp_states);

        //     ref_current_layer->param_backward(*ref_current_layer->bwd_states,
        //                                       *ref_input_delta_z_buffer,
        //                                       *ref_temp_states);
        // }

        // // Backward pass for hidden states
        // test_current_layer->state_backward(
        //     *test_current_layer->bwd_states, *test_input_delta_z_buffer,
        //     *test_output_delta_z_buffer, *test_temp_states);

        // ref_current_layer->state_backward(
        //     *ref_current_layer->bwd_states, *ref_input_delta_z_buffer,
        //     *ref_output_delta_z_buffer, *ref_temp_states);

        // Copy to host for gpu model
#ifdef USE_CUDA
        if (this->test_model.device.compare("cuda") == 0) {
            DeltaStateCuda *test_output_delta_z_buffer_cu =
                dynamic_cast<DeltaStateCuda *>(
                    this->test_output_delta_z_buffer.get());
            test_output_delta_z_buffer_cu->to_host();

            BaseLayerCuda *test_current_layer_cu =
                dynamic_cast<BaseLayerCuda *>(test_current_layer);
            test_current_layer_cu->params_to_host();
            test_current_layer_cu->delta_params_to_host();
        }
        if (this->ref_model.device.compare("cuda") == 0) {
            DeltaStateCuda *ref_output_delta_z_buffer_cu =
                dynamic_cast<DeltaStateCuda *>(
                    this->ref_output_delta_z_buffer.get());
            ref_output_delta_z_buffer_cu->to_host();

            BaseLayerCuda *ref_current_layer_cu =
                dynamic_cast<BaseLayerCuda *>(ref_current_layer);
            ref_current_layer_cu->params_to_host();
            ref_current_layer_cu->delta_params_to_host();
        }
#endif

        // Test here
        if (test_current_layer->get_layer_type() != LayerType::Activation) {
            auto layer_name = test_current_layer->get_layer_name();
            for (int j = 0; j < test_current_layer->input_size * batch_size;
                 j++) {
                if (this->test_output_delta_z_buffer->delta_var[j] !=
                    this->ref_output_delta_z_buffer->delta_var[j]) {
                    std::cout << "Layer name: " << layer_name << " "
                              << "Hidden states " << " " << "Layer no " << i
                              << "\n"
                              << std::endl;

                    int check = 0;
                    break;
                }
            }
            for (int k = 0; k < test_current_layer->delta_mu_w.size(); k++) {
                if (test_current_layer->delta_var_w[k] !=
                    ref_current_layer->delta_var_w[k]) {
                    std::cout << "Layer name: " << layer_name << " "
                              << "Weight " << " " << "Layer no " << i << "\n"
                              << std::endl;

                    int check = 0;
                    break;
                }
            }

            // for (int k = 0; k < test_current_layer->delta_mu_b.size();
            // k++) {
            //     if (test_current_layer->delta_mu_b[k] !=
            //         ref_current_layer->delta_mu_b[k]) {
            //         std::cout << "Layer name: " << layer_name << " "
            //                   << "Bias "
            //                   << " "
            //                   << "Layer no " << i << "\n"
            //                   << std::endl;

            //         int check = 0;
            //         break;
            //     }
            // }
        }

        // Pass new input data for next iteration
        if (test_current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(test_input_delta_z_buffer, test_output_delta_z_buffer);
        }

        if (ref_current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(ref_input_delta_z_buffer, ref_output_delta_z_buffer);
        }
    }

    // // Parameter update for input layer
    // if (test_model.param_update) {
    //     test_model.layers[0]->param_backward(*test_model.layers[0]->bwd_states,
    //                                          *test_input_delta_z_buffer,
    //                                          *test_temp_states);
    // }

    // // State update for input layer
    // if (test_model.input_state_update) {
    //     test_model.layers[0]->state_backward(
    //         *test_model.layers[0]->bwd_states, *test_input_delta_z_buffer,
    //         *test_output_delta_z_buffer, *test_temp_states);
    // }

    // if (ref_model.param_update) {
    //     ref_model.layers[0]->param_backward(*ref_model.layers[0]->bwd_states,
    //                                         *ref_input_delta_z_buffer,
    //                                         *ref_temp_states);
    // }

    // if (ref_model.input_state_update) {
    //     ref_model.layers[0]->state_backward(
    //         *ref_model.layers[0]->bwd_states, *ref_input_delta_z_buffer,
    //         *ref_output_delta_z_buffer, *ref_temp_states);
    // }
}

#ifdef USE_CUDA
CrossValidator::CrossValidator(Sequential &test_model, TagiNetwork *ref_model,
                               std::string &param_prefix)
    : cpu_output_updater("cpu"),
      cuda_output_updater("cuda")
//
{
    this->test_model = test_model;
    this->ref_model = ref_model;
    this->test_model.load_csv(param_prefix);
}

CrossValidator::~CrossValidator() {}

void CrossValidator::lazy_init(int batch_size, int z_buffer_size)
/*
 */
{
    if (test_model.device.compare("cpu") == 0) {
        test_output_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        test_input_z_buffer =
            std::make_shared<BaseHiddenStates>(z_buffer_size, batch_size);
        test_temp_states =
            std::make_shared<BaseTempStates>(z_buffer_size, batch_size);
    } else if (test_model.device.compare("cuda") == 0) {
        test_output_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        test_input_z_buffer =
            std::make_shared<HiddenStateCuda>(z_buffer_size, batch_size);
        test_temp_states =
            std::make_shared<TempStateCuda>(z_buffer_size, batch_size);
    }

    if (test_model.device.compare("cpu") == 0) {
        this->test_output_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
        this->test_input_delta_z_buffer =
            std::make_shared<BaseDeltaStates>(z_buffer_size, batch_size);
    } else if (test_model.device.compare("cuda") == 0) {
        this->test_output_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
        this->test_input_delta_z_buffer =
            std::make_shared<DeltaStateCuda>(z_buffer_size, batch_size);
    }
}

void CrossValidator::validate_forward(const std::vector<float> &mu_x,
                                      const std::vector<float> &var_x)
/*
 */
{
    // Ref Model i.e., older version
    std::vector<float> Sx_f_batch;
    std::vector<float> x_batch = mu_x;
    std::vector<float> Sx_batch(mu_x.size(), 0);
    this->ref_model->prop.ra_mt = 0.9;
    this->ref_model->feed_forward(x_batch, Sx_batch, Sx_f_batch);
    this->ref_model->state_gpu.copy_device_to_host();
    this->ref_model->get_network_outputs();
    if (this->ref_model->prop.batch_size == 1 &&
        this->ref_model->prop.input_seq_len == 1) {
        save_prev_states(this->ref_model->prop, this->ref_model->state_gpu);
    }

    // Test Model
    int batch_size = mu_x.size() / test_model.layers.front()->input_size;
    int z_buffer_size = batch_size * test_model.z_buffer_size;

    this->lazy_init(batch_size, z_buffer_size);

    // Merge input data to the input buffer
    this->test_input_z_buffer->set_input_x(mu_x, var_x, batch_size);
    int num_layers = this->test_model.layers.size();

    for (int i = 0; i < num_layers; i++) {
        auto *test_current_layer = this->test_model.layers[i].get();

        test_current_layer->forward(*this->test_input_z_buffer,
                                    *this->test_output_z_buffer,
                                    *this->test_temp_states);

        // Copy to host for gpu model
        if (this->test_model.device.compare("cuda") == 0) {
            HiddenStateCuda *test_output_z_buffer_cu =
                dynamic_cast<HiddenStateCuda *>(
                    this->test_output_z_buffer.get());
            test_output_z_buffer_cu->to_host();
        }

        std::swap(this->test_input_z_buffer, this->test_output_z_buffer);
    }
    // Output buffer is considered as the final output of network
    std::swap(this->test_output_z_buffer, this->test_input_z_buffer);
    int check = 0;
}

void CrossValidator::validate_backward(std::vector<float> &y_batch,
                                       std::vector<float> &var_obs,
                                       std::vector<int> &idx_ud_batch)
/*
 */
{
    // Ref Model i.e., older version
    std::vector<int> idx_ud_batch_2(
        this->ref_model->prop.nye * this->ref_model->prop.batch_size, 0);
    this->ref_model->state_feed_backward(y_batch, var_obs, idx_ud_batch_2);
    this->ref_model->param_feed_backward();
    this->ref_model->d_state_gpu.copy_device_to_host();
    this->ref_model->d_theta_gpu.copy_device_to_host();

    int batch_size = this->test_output_z_buffer->block_size;
    // Output layer
    if (idx_ud_batch.size() > 0) {
        if (this->test_model.device.compare("cpu") == 0) {
            this->cpu_output_updater.update_using_indices(
                *this->test_output_z_buffer, y_batch, var_obs, idx_ud_batch,
                *this->test_input_delta_z_buffer);
        } else {
            this->cuda_output_updater.update_using_indices(
                *this->test_output_z_buffer, y_batch, var_obs, idx_ud_batch,
                *this->test_input_delta_z_buffer);
        }
    } else {
        if (this->test_model.device.compare("cpu") == 0) {
            this->cpu_output_updater.update(*this->test_output_z_buffer,
                                            y_batch, var_obs,
                                            *this->test_input_delta_z_buffer);
        } else {
            this->cuda_output_updater.update(*this->test_output_z_buffer,
                                             y_batch, var_obs,
                                             *this->test_input_delta_z_buffer);
        }
    }

    int num_layers = test_model.layers.size();

    for (int i = num_layers - 1; i > 0; i--) {
        auto *test_current_layer = test_model.layers[i].get();

        // // Backward pass for parameters and hidden states
        // if (test_model.param_update) {
        //     test_current_layer->param_backward(*test_current_layer->bwd_states,
        //                                        *test_input_delta_z_buffer,
        //                                        *test_temp_states);
        // }

        // // Backward pass for hidden states
        // test_current_layer->state_backward(
        //     *test_current_layer->bwd_states, *test_input_delta_z_buffer,
        //     *test_output_delta_z_buffer, *test_temp_states);

        // Copy to host for gpu model
        if (this->test_model.device.compare("cuda") == 0) {
            DeltaStateCuda *test_output_delta_z_buffer_cu =
                dynamic_cast<DeltaStateCuda *>(
                    this->test_output_delta_z_buffer.get());
            test_output_delta_z_buffer_cu->to_host();

            BaseLayerCuda *test_current_layer_cu =
                dynamic_cast<BaseLayerCuda *>(test_current_layer);
            test_current_layer_cu->params_to_host();
            test_current_layer_cu->delta_params_to_host();
        }

        // Pass new input data for next iteration
        if (test_current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(test_input_delta_z_buffer, test_output_delta_z_buffer);
        }
    }

    // // Parameter update for input layer
    // if (test_model.param_update) {
    //     test_model.layers[0]->param_backward(*test_model.layers[0]->bwd_states,
    //                                          *test_input_delta_z_buffer,
    //                                          *test_temp_states);
    // }

    // // State update for input layer
    // if (test_model.input_state_update) {
    //     test_model.layers[0]->state_backward(
    //         *test_model.layers[0]->bwd_states, *test_input_delta_z_buffer,
    //         *test_output_delta_z_buffer, *test_temp_states);
    // }
}
#endif