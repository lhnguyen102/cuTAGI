///////////////////////////////////////////////////////////////////////////////
// File:         utils.h
// Description:  Header file for utils
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 10, 2022
// Updated:      July 19, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include <vector>

#include "common.h"
#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

void save_error_rate(std::string &res_path, std::vector<float> &error_rate,
                     std::string &suffix);

void save_generated_images(std::string &res_path, std::vector<float> &imgs,
                           std::string &suffix);

void save_hidden_states(std::string &res_path, NetState &state);

void save_idx(std::string &idx_path, IndexOut &idx);

void save_param(std::string &param_path, Param &theta);

void load_net_param(std::string &model_name, std::string &net_name,
                    std::string &path, Param &theta);

void save_net_param(std::string &model_name, std::string &net_name,
                    std::string path, Param &theta);

void save_net_prop(std::string &param_path, std::string &idx_path, Param &theta,
                   IndexOut &idx);

void save_autoencoder_net_prop(Param &theta_e, Param &theta_d, IndexOut &idx_e,
                               IndexOut &idx_d, std::string &debug_path);

void save_predictions(std::string &res_path, std::vector<float> &ma,
                      std::vector<float> &sa, std::string &suffix);

void save_derivatives(std::string &res_path, std::vector<float> &md_layer,
                      std::vector<float> &Sd_layer, std::string &suffix);
