#pragma once
#include <math.h>

#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "data_struct.h"

void compute_delta_z_output(std::vector<float> &mu_a, std::vector<float> &var_a,
                            std::vector<float> &jcb, std::vector<float> &obs,
                            std::vector<float> &var_obs, int start_chunk,
                            int end_chunk, std::vector<float> &delta_mu,
                            std::vector<float> &delta_var);

void compute_delta_z_output_mp(std::vector<float> &mu_a,
                               std::vector<float> &var_a,
                               std::vector<float> &jcb, std::vector<float> &obs,
                               std::vector<float> &var_v, int n,
                               unsigned int num_threads,
                               std::vector<float> &delta_mu,
                               std::vector<float> &delta_var);

void compute_selected_delta_z_output_mp(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_obs, std::vector<int> &selected_idx, int n_obs,
    int n_enc, int n, unsigned int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var);

void compute_selected_delta_z_output(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_obs, std::vector<int> &selected_idx, int n_obs,
    int n_enc, int start_chunk, int end_chunk, std::vector<float> &delta_mu,
    std::vector<float> &delta_var);

void compute_delta_z_heteros(std::vector<float> &mu_a,
                             std::vector<float> &var_a, std::vector<float> &jcb,
                             std::vector<float> &obs, int start_chunk,
                             int end_chunk, std::vector<float> &delta_mu,
                             std::vector<float> &delta_var);

void compute_delta_z_heteros_mp(std::vector<float> &mu_a,
                                std::vector<float> &var_a,
                                std::vector<float> &jcb,
                                std::vector<float> &obs, int n,
                                unsigned int num_threads,
                                std::vector<float> &delta_mu,
                                std::vector<float> &delta_var);

////////////////////////////////////////////////////////////////////////////////
// Base Output Updater
////////////////////////////////////////////////////////////////////////////////

class BaseOutputUpdater {
   public:
    BaseOutputUpdater();
    ~BaseOutputUpdater();

    virtual void update_output_delta_z(BaseHiddenStates &output_states,
                                       BaseObservation &obs,
                                       BaseDeltaStates &delta_states);

    virtual void update_selected_output_delta_z(BaseHiddenStates &output_states,
                                                BaseObservation &obs,
                                                BaseDeltaStates &delta_states);

    virtual void update_output_delta_z_heteros(BaseHiddenStates &output_states,
                                               BaseObservation &obs,
                                               BaseDeltaStates &delta_states);

    virtual std::string get_name() const { return "BaseOutputUpdater"; };
};

////////////////////////////////////////////////////////////////////////////////
// Output Updater
////////////////////////////////////////////////////////////////////////////////
class OutputUpdater {
   public:
    std::shared_ptr<BaseOutputUpdater> updater;
    std::shared_ptr<BaseObservation> obs;
    std::string device = "cpu";

    OutputUpdater(const std::string model_device);
    OutputUpdater();

    ~OutputUpdater();

    void update(BaseHiddenStates &output_states, std::vector<float> &mu_obs,
                std::vector<float> &var_obs, BaseDeltaStates &delta_states);

    void update_using_indices(BaseHiddenStates &output_states,
                              std::vector<float> &mu_obs,
                              std::vector<float> &var_obs,
                              std::vector<int> &selected_idx,
                              BaseDeltaStates &delta_states);

    void update_heteros(BaseHiddenStates &output_states,
                        std::vector<float> &mu_obs,
                        BaseDeltaStates &delta_states);
};
