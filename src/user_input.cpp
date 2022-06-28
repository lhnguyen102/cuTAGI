///////////////////////////////////////////////////////////////////////////////
// File:         user_input.cpp
// Description:  Load user input
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 05, 2022
// Updated:      May 29, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/user_input.h"

UserInput load_userinput(std::string &user_input_file)
/*
 * Load the user-speficied input

 **/
{
    // Dictionary for the cfg file
    std::string key_words[] = {"model_name",
                               "net_name",
                               "task_name",
                               "data_name",
                               "encoder_net_name",
                               "decoder_net_name",
                               "device",
                               "load_param",
                               "num_epochs",
                               "num_classes",
                               "num_train_data",
                               "num_test_data",
                               "mu",
                               "sigma",
                               "x_train_dir",
                               "y_train_dir",
                               "x_test_dir",
                               "y_test_dir",
                               "debug"};
    int num_keys = sizeof(key_words) / sizeof(key_words[0]);

    // Load user input file
    std::string cfg_path = get_current_dir() + "/cfg/" + user_input_file;
    std::ifstream cfg_file(cfg_path);

    // Initialize pointers
    UserInput user_input;
    int d;
    float f;
    bool b;
    std::string si;
    std::string line;

    // Open user input file
    while (std::getline(cfg_file, line)) {
        // TODO: Define a comment line. Remove white space between characters
        std::string::iterator end_pos =
            std::remove(line.begin(), line.end(), ' ');
        line.erase(end_pos, line.end());
        std::string::iterator tab_pos =
            std::remove(line.begin(), line.end(), '\t');
        line.erase(tab_pos, line.end());

        for (int k = 0; k < num_keys; k++) {
            // Key =  keyword + separator
            std::string key = key_words[k] + ":";

            // Find key in user input file
            // NOTE pos equal to zero means that the key matches the entire word
            auto pos = line.find(key);
            if (pos == 0) {
                // Store data
                if (key_words[k] == "model_name") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        user_input.model_name = si;
                    }
                } else if (key_words[k] == "net_name") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        user_input.net_name = si;
                    }
                } else if (key_words[k] == "task_name") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        user_input.task_name = si;
                    }
                } else if (key_words[k] == "data_name") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        user_input.data_name = si;
                    }
                } else if (key_words[k] == "num_classes") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> d;
                        user_input.num_classes = d;
                    }
                } else if (key_words[k] == "num_epochs") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> d;
                        user_input.num_epochs = d;
                    }
                } else if (key_words[k] == "num_train_data") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> d;
                        user_input.num_train_data = d;
                    }
                } else if (key_words[k] == "num_test_data") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> d;
                        user_input.num_test_data = d;
                    }
                } else if (key_words[k].compare("mu") == 0) {
                    std::stringstream ss(line.substr(pos + key.size()));
                    std::vector<float> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> f) {
                            v.push_back(f);
                        }
                    }
                    user_input.mu = v;
                } else if (key_words[k].compare("sigma") == 0) {
                    std::stringstream ss(line.substr(pos + key.size()));
                    std::vector<float> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> f) {
                            v.push_back(f);
                        }
                    }
                    user_input.sigma = v;
                } else if (key_words[k].compare("x_train_dir") == 0) {
                    std::stringstream ss(line.substr(pos + key.size()));
                    std::vector<std::string> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> si) {
                            v.push_back(si);
                        }
                    }
                    user_input.x_train_dir = v;
                } else if (key_words[k] == "y_train_dir") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    std::vector<std::string> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> si) {
                            v.push_back(si);
                        }
                    }
                    user_input.y_train_dir = v;
                } else if (key_words[k].compare("x_test_dir") == 0) {
                    std::stringstream ss(line.substr(pos + key.size()));
                    std::vector<std::string> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> si) {
                            v.push_back(si);
                        }
                    }
                    user_input.x_test_dir = v;
                } else if (key_words[k] == "y_test_dir") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    std::vector<std::string> v;
                    while (ss.good()) {
                        std::string tmp;
                        std::getline(ss, tmp, ',');
                        std::stringstream iss(tmp);
                        if (iss >> si) {
                            v.push_back(si);
                        }
                    }
                    user_input.y_test_dir = v;
                } else if (key_words[k] == "encoder_net_name") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        user_input.encoder_net_name = si;
                    }
                } else if (key_words[k] == "decoder_net_name") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        user_input.decoder_net_name = si;
                    }
                } else if (key_words[k] == "device") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        if (si.compare("cpu") == 0) {
                            user_input.device = "cpu";
                        } else if (si.compare("cuda") == 0) {
                            user_input.device = "cuda";
                        } else {
                            throw std::invalid_argument(
                                "Device can be either cuda or cpu - "
                                "user_input.cpp");
                        }
                    }
                } else if (key_words[k] == "load_param") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        if (si.compare("true") == 0) {
                            user_input.load_param = true;
                        }
                    }
                }

                else if (key_words[k] == "debug") {
                    std::stringstream ss(line.substr(pos + key.size()));
                    if (ss.good()) {
                        ss >> si;
                        if (si.compare("true") == 0) {
                            user_input.debug = true;
                        }
                    }
                }
                break;
            }
        }
    }
    return user_input;
}