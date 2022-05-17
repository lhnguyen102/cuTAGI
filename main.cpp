///////////////////////////////////////////////////////////////////////////////
// File:         main.cpp
// Description:  API for c++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      April 27, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>

#include "include/struct_var.h"
#include "include/task.cuh"
#include "include/user_input.h"

int main(int argc, char *argv[]) {
    // User input file
    std::string user_input_file;
    if (argc > 1) {
        user_input_file = argv[1];
    } else {
        throw std::invalid_argument(
            "User need to provide user input file -> see README");
    }

    // Default path
    SavePath path;
    path.curr_path = get_current_dir();
    path.saved_param_path = path.curr_path + "/saved_param/";
    path.debug_path = path.curr_path + "/debug_data/";
    path.saved_inference_path = path.curr_path + "/saved_results/";

    // Run task
    task_command(user_input_file, path);
    return 0;
}
