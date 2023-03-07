///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.cpp
// Description:  Main script to test the CPU implementation of cuTAGI
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_cpu.h"

/**
 * @brief Read the last dates of the tests
 *
 * @return std::vector<std::string> vector with the last dates of the tests
 */
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

/**
 * @brief Write the last dates of the tests
 *
 * @param dates vector with the last dates of the tests
 * @param column column to change
 * @param date new current date
 */
void write_dates(std::vector<std::string> dates, int column, std::string date) {
    std::ofstream file("test/data/last_dates.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    file << "fnn,fnn_heteros,fnn_full_cov,fnn_derivates,cnn,cnn_batch_norm,"
            "autoencoder,lstm,cnn_resnet"
         << std::endl;

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

void test_cpu(std::vector<std::string>& user_input_options) {
    std::string reinizialize_test_outputs = "";
    std::string test_architecture = "";
    std::string date = "";

    if (user_input_options.size() == 1 &&
        (user_input_options[0] == "-h" || user_input_options[0] == "--help")) {
        int num_spaces = 35;

        std::cout << "Usage: build/main [options]" << std::endl;
        std::cout << "Options:" << std::endl;

        std::cout << std::setw(num_spaces) << std::left << "test"
                  << "Perform tests on all architectures" << std::endl;

        std::cout << std::setw(num_spaces) << std::left
                  << "test [architecture-name]"
                  << "Run one specific test" << std::endl;

        std::cout << std::setw(num_spaces) << std::left << "test -reset all"
                  << "Reinizialize all test references" << std::endl;

        std::cout << std::setw(num_spaces) << std::left
                  << "test -reset <architecture-name>"
                  << "Reinizialize one specific test reference" << std::endl;

        std::cout << std::endl;

        std::cout << "Available architectures: [fnn, fnn_heteros, "
                     "fnn_full_cov, "
                     "fnn_derivates, cnn, cnn_batch_norm, autoencoder, lstm, "
                     "cnn_resnet, lstm, cnn_resnet]"
                  << std::endl;
        return;
    } else if (user_input_options.size() > 0 && user_input_options.size() < 3) {
        if (user_input_options[0] == "-reset") {
            if (user_input_options.size() == 1) {
                reinizialize_test_outputs = "all";
            } else {
                std::vector<std::string> possible_architectures = {
                    "all",
                    "fnn",
                    "fnn_heteros",
                    "fnn_full_cov",
                    "fnn_derivates",
                    "cnn",
                    "cnn_batch_norm",
                    "autoencoder",
                    "lstm",
                    "cnn_resnet"};

                // Check if the architecture is valid
                if (std::find(possible_architectures.begin(),
                              possible_architectures.end(),
                              user_input_options[1]) ==
                    possible_architectures.end()) {
                    throw std::runtime_error(
                        "This architecture is not valid. test -h for help.");
                }

                reinizialize_test_outputs = user_input_options[1];
            }
        } else {
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
    } else if (user_input_options.size() > 1) {
        std::cout << "Too many arguments" << std::endl;
        return;
    }

    // Read last test dates
    std::vector<std::string> test_dates = read_dates();

    // Index of the current test
    int test_num;

    ////////////////////////////
    //      PERFORM TESTS     //
    ////////////////////////////

    if (test_architecture.size() > 0) {
        int num_tests = 9;

        std::cout << "Performing " << test_architecture << " tests"
                  << std::endl;

        // Perform test on CPU for the FNN architecture
        if (test_architecture == "all" || test_architecture == "fnn") {
            test_num = 0;  // FNN

            if (test_fnn_cpu(false, test_dates[test_num], "fnn", "1D") &&
                test_fnn_cpu(false, test_dates[test_num], "fnn",
                             "Boston_housing")) {
                std::cout << "[ " << floor((100 / num_tests) * (test_num + 1))
                          << "%] "
                          << "\033[32;1mFNN tests passed\033[0m" << std::endl;
            } else {
                std::cout << "[ " << floor((100 / num_tests) * (test_num + 1))
                          << "%] "
                          << "\033[31;1mFNN tests failed\033[0m" << std::endl;
            }
        }

        // Perform test on CPU for the FNN architecture with heteroscedastic
        // noise
        if (test_architecture == "all" || test_architecture == "fnn_heteros") {
            test_num = 1;  // FNN heteroscedastic noise

            if (test_fnn_heteros_cpu(false, test_dates[test_num], "fnn_heteros",
                                     "1D_noise_inferance")) {
                std::cout << "[ " << floor((100 / num_tests) * (test_num + 1))
                          << "%] "
                          << "\033[32;1mFNN heteroscedastic tests passed\033[0m"
                          << std::endl;
            } else {
                std::cout << "[ " << floor((100 / num_tests) * (test_num + 1))
                          << "%] "
                          << "\033[31;1mFNN heteroscedastic tests failed\033[0m"
                          << std::endl;
            }
        }

        // Perform test on CPU for the FNN architecture with full covariance
        if (test_architecture == "all" || test_architecture == "fnn_full_cov") {
            test_num = 2;  // FNN full covariance

            if (test_fnn_full_cov_cpu(false, test_dates[test_num],
                                      "fnn_full_cov", "1D_full_cov")) {
                std::cout << "[ " << floor((100 / num_tests) * (test_num + 1))
                          << "%] "
                          << "\033[32;1mFNN full covariance tests passed\033[0m"
                          << std::endl;
            } else {
                std::cout << "[ " << floor((100 / num_tests) * (test_num + 1))
                          << "%] "
                          << "\033[31;1mFNN full covariance tests failed\033[0m"
                          << std::endl;
            }
        }
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
        }
    }
}
