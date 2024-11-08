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
