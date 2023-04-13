///////////////////////////////////////////////////////////////////////////////
// File:         test_gpu.cu
// Description:  Main script to test the GPU implementation of cuTAGI
// Authors:      Florensa, Miquel, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      April 13, 2023
// Contact:      miquelflorensa11@gmail.com, luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_gpu.cuh"

const int NUM_TESTS = 9;

int test_gpu(std::vector<std::string>& user_input_options,
             int num_tests_passed_cpu) {
    std::string reinizialize_test_outputs = "";
    std::string test_architecture = "";
    std::string date = "";
    bool single_test = true;

    if (user_input_options.size() == 1 &&
        (user_input_options[0] == "-h" || user_input_options[0] == "--help")) {
        // Help message have already been showed in test_cpu.cpp
        return -1;
    } else if (user_input_options.size() > 0 && user_input_options.size() < 3) {
        if (user_input_options[0] == "-reset") {
            if (user_input_options.size() == 1) {
                reinizialize_test_outputs = "all";
            } else {
                // Check if the architecture is valid
                check_valid_input_architecture(user_input_options[1]);

                reinizialize_test_outputs = user_input_options[1];
            }
        } else {
            // Check if the architecture is valid
            check_valid_input_architecture(user_input_options[0]);

            test_architecture = user_input_options[0];
        }
        std::time_t t = std::time(0);  // get time now
        std::tm* now = std::localtime(&t);
        std::string year = std::to_string(now->tm_year + 1900);
        std::string month = std::to_string(now->tm_mon + 1);
        if (month.size() == 1) month = "0" + month;
        std::string day = std::to_string(now->tm_mday);
        if (day.size() == 1) day = "0" + day;

        date = year + "_" + month + "_" + day;

    } else if (user_input_options.size() == 0) {
        test_architecture = "all";
        single_test = false;
    } else if (user_input_options.size() > 1) {
        std::cout << "Too many arguments" << std::endl;
        return -1;
    }

    // Read last test dates
    std::vector<std::string> test_dates = read_dates();

    // Index of the current test
    int test_num;

    ////////////////////////////
    //      PERFORM TESTS     //
    ////////////////////////////

    if (test_architecture.size() > 0) {
        int num_test_passed = num_tests_passed_cpu;

        // Perform test on GPU for the classification task
        if (test_architecture == "all" || test_architecture == "cnn") {
            test_num = 6;  // CNN

            bool test_result =
                test_cnn_gpu(false, test_dates[test_num], "cnn", "mnist");

            print_test_results(single_test, test_result, NUM_TESTS, test_num,
                               "CNN");

            if (test_result) num_test_passed++;
        }

        // Perform test on GPU with batch normalization for classification task
        if (test_architecture == "all" ||
            test_architecture == "cnn_batch_norm") {
            test_num = 7;  // CNN batch norm.

            bool test_result = test_cnn_batch_norm_gpu(
                false, test_dates[test_num], "cnn_batch_norm", "mnist");

            print_test_results(single_test, test_result, NUM_TESTS, test_num,
                               "CNN batch normalization");

            if (test_result) num_test_passed++;
        }

        // Perform test on GPU with autoencoder for image generation
        if (test_architecture == "all" || test_architecture == "autoencoder") {
            test_num = 8;  // Autoencoder

            bool test_result = test_autoencoder_gpu(false, test_dates[test_num],
                                                    "autoencoder", "mnist");

            print_test_results(single_test, test_result, NUM_TESTS, test_num,
                               "Autoencoder");

            if (test_result) num_test_passed++;
        }

        // Number of tests passed
        if (test_architecture == "all") {
            std::cout << std::endl;
            std::cout << "--------------------SUMMARY--------------------"
                      << std::endl;
            std::cout << "Passed tests: [" << num_test_passed << "/"
                      << NUM_TESTS << "]" << std::endl;
            return num_test_passed;
        }
        return -1;
    }

    ///////////////////////////////
    // REINIZIALIZE TEST OUTPUTS //
    ///////////////////////////////

    if (reinizialize_test_outputs.size() > 0) {
        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "cnn") {
            // Reinizialize test outputs for classification task
            std::cout << "Reinizializing CNN test outputs" << std::endl;

            test_cnn_gpu(true, date, "cnn", "mnist");

            test_num = 6;  // CNN

            // Update de last date of the test
            write_dates(test_dates, test_num, date);
            test_dates[test_num] = date;
        }

        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "cnn_batch_norm") {
            // Reinizialize test outputs for classification task
            std::cout << "Reinizializing CNN batch norm. test outputs"
                      << std::endl;

            test_cnn_batch_norm_gpu(true, date, "cnn_batch_norm", "mnist");

            test_num = 7;  // CNN

            // Update de last date of the test
            write_dates(test_dates, test_num, date);
            test_dates[test_num] = date;
        }

        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "autoencoder") {
            // Reinizialize test outputs for for image generation
            std::cout << "Reinizializing Autoencoder test outputs" << std::endl;

            test_autoencoder_gpu(true, date, "autoencoder", "mnist");

            test_num = 8;  // Autoencoder

            // Update de last date of the test
            write_dates(test_dates, test_num, date);
            test_dates[test_num] = date;
        }
        return 0;
    }

    return -1;
}
