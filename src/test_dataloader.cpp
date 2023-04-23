///////////////////////////////////////////////////////////////////////////////
// File:         test_dataloader.cpp
// Description:  Testing unit for dataloader
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 21, 2022
// Updated:      August 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#include "../include/test_unit.h"
int main() {
    // Data
    std::vector<float> data = {1, 9,  17, 2, 10, 18, 3, 11, 19, 4, 12, 20,
                               5, 13, 21, 6, 14, 22, 7, 15, 23, 8, 16, 24};
    std::vector<float> inputs, outputs;
    std::vector<int> output_col = {0, 2};
    int num_input_ts = 4;
    int num_output_ts = 2;
    int num_features = 3;
    int stride = 1;
    int num_samples =
        (data.size() / num_features - num_input_ts - num_output_ts) / stride +
        1;
    int num_outputs = output_col.size();
    std::vector<float> input_data(num_input_ts * num_features * num_samples);
    std::vector<float> output_data(num_output_ts * num_outputs * num_samples);

    // Get inputs and outputs
    create_rolling_windows(data, output_col, num_input_ts, num_output_ts,
                           num_features, stride, input_data, output_data);

    // Print matrix
    std::cout << "inputs\n" << std::endl;
    print_matrix(inputs, num_features, num_samples * num_input_ts);
    std::cout << "outputs\n" << std::endl;
    print_matrix(outputs, output_col.size(), num_samples * num_output_ts);

    return 0;
}
