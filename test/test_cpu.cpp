///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.cpp
// Description:  Main script to test the CPU implementation of cuTAGI
// Authors:      Florensa, Miquel, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      April 13, 2023
// Contact:      miquelflorensa11@gmail.com, luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_cpu.h"

const int NUM_TESTS_CPU = 6;
const int NUM_TESTS_GPU = 3;
const std::vector<std::string> AVAILABLE_ARCHITECTURES = {
    "fnn",      "fnn_heteros", "fnn_full_cov",   "fnn_derivatives", "lstm",
    "act_func", "cnn",         "cnn_batch_norm", "autoencoder"};

std::vector<std::string> read_dates() {
    std::ifstream file("test/data/last_dates.csv");
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    std::istringstream iss(line);
    std::string value;
    std::vector<std::string> dates;

    while (std::getline(iss, value, ',')) {
        dates.push_back(value);
    }

    file.close();

    return dates;
}

void write_dates(std::vector<std::string> dates, int column, std::string date) {
    std::ofstream file("test/data/last_dates.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    for (int i = 0; i < AVAILABLE_ARCHITECTURES.size(); i++) {
        file << AVAILABLE_ARCHITECTURES[i];
        if (i != AVAILABLE_ARCHITECTURES.size() - 1) {
            file << ",";
        }
    }
    file << std::endl;

    for (int i = 0; i < dates.size(); i++) {
        if (i == column) {
            file << date;
        } else {
            file << dates[i];
        }
        if (i != dates.size() - 1) {
            file << ",";
        }
    }
    file << std::endl;

    file.close();
}

void check_valid_input_architecture(std::string test_architecture) {
    if (std::find(AVAILABLE_ARCHITECTURES.begin(),
                  AVAILABLE_ARCHITECTURES.end(),
                  test_architecture) == AVAILABLE_ARCHITECTURES.end()) {
        std::cerr << "Error: Invalid architecture name." << std::endl;
        std::cerr << "build/main test -h to see available architectures."
                  << std::endl;
        exit(1);
    }
}

void print_test_results(bool single_test, bool test_passed, int num_tests,
                        int test_num, std::string arch_name,
                        std::chrono::milliseconds run_time) {
    if (!single_test) {
        int percentage = round((100.0 / num_tests) * (test_num + 1));
        std::cout << "[";
        if (percentage < 100) std::cout << " ";
        if (percentage < 10) std::cout << " ";
        std::cout << percentage << "%] ";
    }
    int num_spaces = 35;
    arch_name.append(":");

    if (test_passed) {
        std::cout << std::setw(num_spaces) << std::left << arch_name;
        std::cout << "\033[32;1mPASS\033[0m";
    } else {
        std::cout << std::setw(num_spaces) << std::left << arch_name;
        std::cout << "\033[31;1mFAIL\033[0m";
    }

    std::cout << "  (took " << run_time.count() << "ms)" << std::endl;
}

int test_cpu(std::vector<std::string>& user_input_options,
             bool compute_gpu_tests,
             std::chrono::steady_clock::time_point test_start) {
    std::string reinizialize_test_outputs = "";
    std::string test_architecture = "";
    std::string date = "";
    bool single_test = true;

    if (user_input_options.size() == 1 &&
        (user_input_options[0] == "-h" || user_input_options[0] == "--help")) {
        int num_spaces = 35;

        std::cout << "Usage: build/main [options]" << std::endl;
        std::cout << "Options:" << std::endl;

        std::cout << std::setw(num_spaces) << std::left << "test"
                  << "Perform tests on all architectures" << std::endl;

        std::cout << std::setw(num_spaces) << std::left
                  << "test <architecture-name>"
                  << "Run one specific test" << std::endl;

        std::cout << std::setw(num_spaces) << std::left << "test -reset (all)"
                  << "Reinizialize all test references" << std::endl;

        std::cout << std::setw(num_spaces) << std::left
                  << "test -reset <architecture-name>"
                  << "Reinizialize one specific test reference" << std::endl;

        std::cout << std::endl;

        std::cout << "Available architectures: [";
        for (int i = 0; i < AVAILABLE_ARCHITECTURES.size(); i++) {
            std::cout << AVAILABLE_ARCHITECTURES[i];
            if (i != AVAILABLE_ARCHITECTURES.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        return -1;
    } else if (user_input_options.size() > 0 && user_input_options.size() < 3) {
        if (user_input_options[0] == "-reset") {
            if (user_input_options.size() == 1) {
                reinizialize_test_outputs = "all";
            } else if (user_input_options.size() == 2 &&
                       user_input_options[1] == "all") {
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
        int num_tests;
        std::string device;

        if (!compute_gpu_tests) {
            num_tests = NUM_TESTS_CPU;
            device = "CPU";
        } else {
            num_tests = NUM_TESTS_CPU + NUM_TESTS_GPU;
            device = "GPU";
        }

        std::cout << "=========================================" << std::endl;
        std::cout << "            TAGI Unit Test               " << std::endl;
        std::cout << "=========================================" << std::endl;

        int num_test_passed = 0;

        std::cout << "Performing " << test_architecture << " tests on "
                  << device << " device" << std::endl;

        // Perform test on CPU for the FNN architecture
        if (test_architecture == "all" || test_architecture == "fnn") {
            test_num = 0;  // FNN

            auto start = std::chrono::steady_clock::now();
            bool test_result =
                test_fnn_cpu(false, test_dates[test_num], "fnn", "1D") &&
                test_fnn_cpu(false, test_dates[test_num], "fnn",
                             "Boston_housing");

            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            print_test_results(single_test, test_result, num_tests, test_num,
                               "FNN", run_time);

            if (test_result) num_test_passed++;
        }

        // Perform test on CPU for the FNN architecture with heteroscedastic
        // noise
        if (test_architecture == "all" || test_architecture == "fnn_heteros") {
            test_num = 1;  // FNN heteroscedastic noise

            auto start = std::chrono::steady_clock::now();
            bool test_result =
                test_fnn_heteros_cpu(false, test_dates[test_num], "fnn_heteros",
                                     "1D_noise_inferance");

            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            print_test_results(single_test, test_result, num_tests, test_num,
                               "FNN heteroscedastic", run_time);

            if (test_result) num_test_passed++;
        }

        // Perform test on CPU for the FNN architecture with full covariance
        if (test_architecture == "all" || test_architecture == "fnn_full_cov") {
            test_num = 2;  // FNN full covariance

            auto start = std::chrono::steady_clock::now();
            bool test_result = test_fnn_full_cov_cpu(
                false, test_dates[test_num], "fnn_full_cov", "1D_full_cov");

            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            print_test_results(single_test, test_result, num_tests, test_num,
                               "FNN full covariance", run_time);

            if (test_result) num_test_passed++;
        }

        // Perform test on CPU for the FNN architecture for estimating
        // derivatives of the input layer
        if (test_architecture == "all" ||
            test_architecture == "fnn_derivatives") {
            test_num = 3;  // FNN derivatives

            auto start = std::chrono::steady_clock::now();
            bool test_result =
                test_fnn_derivatives_cpu(false, test_dates[test_num],
                                         "fnn_derivatives", "1D_derivatives");

            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            print_test_results(single_test, test_result, num_tests, test_num,
                               "FNN derivatives", run_time);

            if (test_result) num_test_passed++;
        }

        // Perform test on CPU for the series forecasting task
        if (test_architecture == "all" || test_architecture == "lstm") {
            test_num = 4;  // LSTM

            auto start = std::chrono::steady_clock::now();
            bool test_result = test_lstm_cpu(false, test_dates[test_num],
                                             "lstm", "time_series");

            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            print_test_results(single_test, test_result, num_tests, test_num,
                               "LSTM", run_time);

            if (test_result) num_test_passed++;
        }

        // Perform test on CPU for checking activation functions
        if (test_architecture == "all" || test_architecture == "act_func") {
            test_num = 5;  // Activation Functions

            auto start = std::chrono::steady_clock::now();
            bool test_result = test_act_func_cpu(false, test_dates[test_num],
                                                 "act_func", "Boston_housing");

            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                      start);
            print_test_results(single_test, test_result, num_tests, test_num,
                               "Activation functions", run_time);

            if (test_result) num_test_passed++;
        }

        auto end = std::chrono::steady_clock::now();
        auto run_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end - test_start)
                            .count();

        // Number of tests passed
        if (test_architecture == "all" && !compute_gpu_tests) {
            std::cout << std::endl;
            std::cout << "-----------------SUMMARY-----------------"
                      << std::endl;
            std::cout << "Total tests: " << num_tests << std::endl;
            std::cout << "Passed: " << num_test_passed << std::endl;
            std::cout << "Failed: " << num_tests - num_test_passed << std::endl;
            std::cout << "Total time taken: " << run_time << "ms" << std::endl;
            std::cout << "========================================="
                      << std::endl;
            return num_test_passed;
        }
        return num_test_passed;
    }

    ///////////////////////////////
    // REINIZIALIZE TEST OUTPUTS //
    ///////////////////////////////

    if (reinizialize_test_outputs.size() > 0) {
        std::string answer;

        std::cout
            << "Are you sure you want to recompute the tests references on " +
                   reinizialize_test_outputs + " architecture/s? (yes/no): ";
        std::cin >> answer;

        if (answer == "Y" || answer == "y" || answer == "yes" ||
            answer == "Yes") {
            if (reinizialize_test_outputs == "all" ||
                reinizialize_test_outputs == "fnn") {
                // Reinizialize test outputs for the FNN architecture
                std::cout << "Reinizializing FNN test outputs" << std::endl;

                test_fnn_cpu(true, date, "fnn", "1D");
                test_fnn_cpu(true, date, "fnn", "Boston_housing");

                test_num = 0;  // FNN

                // Update de last date of the test
                write_dates(test_dates, test_num, date);
                test_dates[test_num] = date;
            }

            if (reinizialize_test_outputs == "all" ||
                reinizialize_test_outputs == "fnn_heteros") {
                // Reinizialize test outputs for the FNN architecture with
                // heteroscedastic noise
                std::cout
                    << "Reinizializing FNN heteroscedastic noise test outputs"
                    << std::endl;

                test_fnn_heteros_cpu(true, date, "fnn_heteros",
                                     "1D_noise_inferance");

                test_num = 1;  // FNN heteroscedastic noise

                // Update de last date of the test
                write_dates(test_dates, test_num, date);
                test_dates[test_num] = date;
            }

            if (reinizialize_test_outputs == "all" ||
                reinizialize_test_outputs == "fnn_full_cov") {
                // Reinizialize test outputs for the FNN architecture with
                // full covariance
                std::cout << "Reinizializing FNN full covariance test outputs"
                          << std::endl;

                test_fnn_full_cov_cpu(true, date, "fnn_full_cov",
                                      "1D_full_cov");

                test_num = 2;  // FNN full covariance

                // Update de last date of the test
                write_dates(test_dates, test_num, date);
                test_dates[test_num] = date;
            }

            if (reinizialize_test_outputs == "all" ||
                reinizialize_test_outputs == "fnn_derivatives") {
                // Reinizialize test outputs for the FNN architecture for
                // estimating derivatives of the input layer
                std::cout << "Reinizializing FNN derivatives test outputs"
                          << std::endl;

                test_fnn_derivatives_cpu(true, date, "fnn_derivatives",
                                         "1D_derivatives");

                test_num = 3;  // FNN derivatives

                // Update de last date of the test
                write_dates(test_dates, test_num, date);
                test_dates[test_num] = date;
            }

            if (reinizialize_test_outputs == "all" ||
                reinizialize_test_outputs == "lstm") {
                // Reinizialize test outputs for the series forecasting task
                std::cout << "Reinizializing LSTM test outputs" << std::endl;

                test_lstm_cpu(true, date, "lstm", "time_series");

                test_num = 4;  // LSTM

                // Update de last date of the test
                write_dates(test_dates, test_num, date);
                test_dates[test_num] = date;
            }

            if (reinizialize_test_outputs == "all" ||
                reinizialize_test_outputs == "act_func") {
                // Reinizialize test outputs for activations function tests
                std::cout << "Reinizializing Activations function test outputs"
                          << std::endl;

                test_act_func_cpu(true, date, "act_func", "Boston_housing");

                test_num = 5;  // Activation functions

                // Update de last date of the test
                write_dates(test_dates, test_num, date);
                test_dates[test_num] = date;
            }
            return 0;
        } else {
            // Reinizialization Aborted
            return -2;
        }
    }

    return -1;
}
