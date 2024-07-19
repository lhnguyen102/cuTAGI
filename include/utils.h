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

void save_error_rate(std::string &res_path, std::vector<float> &error_rate,
                     std::string &suffix);

void save_generated_images(std::string &res_path, std::vector<float> &imgs,
                           std::string &suffix);

void save_predictions(std::string &res_path, std::vector<float> &ma,
                      std::vector<float> &sa, std::string &suffix);

void save_derivatives(std::string &res_path, std::vector<float> &md_layer,
                      std::vector<float> &Sd_layer, std::string &suffix);
