#include "test_cpu.h"

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
    return dates;
}

void write_dates(std::vector<std::string> dates, int column, std::string date) {
    std::ofstream file("test/data/last_dates.csv");
    file << "fnn,fnn_hetero,fnn_full_cov,fnn_derivates,cnn,cnn_batch_norm,"
            "autoencoder,lstm,cnn_resnet,lstm,cnn_resnet"
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
}

void test_cpu(std::vector<std::string>& user_input_options) {
    std::string reinizialize_test_outputs = "";
    std::string date = "";

    if (user_input_options.size() == 1 &&
        (user_input_options[0] == "-h" || user_input_options[0] == "--help")) {
        std::cout << "Perform Tests:                    build/main test"
                  << std::endl;
        std::cout << "Reinizialize all test outputs:    build/main test -all"
                  << std::endl;
        std::cout << "Reinizialize one specific output: build/main test "
                     "[architecture-name]"
                  << std::endl;
        std::cout << "Available architectures: fnn, fnn_hetero, "
                     "fnn_full_cov, "
                     "fnn_derivates, cnn, cnn_batch_norm, autoencoder, lstm, "
                     "cnn_resnet, lstm, cnn_resnet"
                  << std::endl;
        return;
    } else if (user_input_options.size() == 1) {
        if (user_input_options[0] == "-all") {
            reinizialize_test_outputs = "all";
        } else {
            reinizialize_test_outputs = user_input_options[0];
        }
        std::time_t t = std::time(0);  // get time now
        std::tm* now = std::localtime(&t);
        std::string year = std::to_string(now->tm_year + 1900);
        std::string month = std::to_string(now->tm_mon + 1);
        if (month.size() == 1) month = "0" + month;
        std::string day = std::to_string(now->tm_mday);
        if (day.size() == 1) day = "0" + day;

        date = year + "_" + month + "_" + day;

    } else if (user_input_options.size() > 1) {
        std::cout << "Too many arguments" << std::endl;
        return;
    }

    std::vector<std::string> test_dates = read_dates();

    // #########################
    //      PERFORM TESTS
    // #########################
    if (user_input_options.size() == 0) {
        // test_fnn_cpu(false, test_dates[0], "fnn", "Boston_housing");
        if (test_fnn_cpu(false, test_dates[0], "fnn", "1D") &&
            test_fnn_cpu(false, test_dates[0], "fnn", "Boston_housing")) {
            std::cout << "[  " << floor((100 / 11) * 1) << "%] "
                      << "FNN tests passed" << std::endl;
        } else {
            std::cout << "[  " << floor((100 / 11) * 1) << "%] "
                      << "FNN tests failed" << std::endl;
        }

        if (test_fnn_heteroscedastic_cpu(false, test_dates[1], "fnn_heteros",
                                         "1D") &&
            test_fnn_heteroscedastic_cpu(false, test_dates[1], "fnn_heteros",
                                         "Boston_housing")) {
            std::cout << "[ " << floor((100 / 11) * 2) << "%] "
                      << "FNN heteroscedastic tests passed" << std::endl;
        } else {
            std::cout << "[ " << floor((100 / 11) * 2) << "%] "
                      << "FNN heteroscedastic tests failed" << std::endl;
        }

        if (test_fnn_full_cov_cpu(false, test_dates[2], "fnn_full_cov", "1D") &&
            test_fnn_full_cov_cpu(false, test_dates[2], "fnn_full_cov",
                                  "Boston_housing")) {
            std::cout << "[ " << floor((100 / 11) * 3) << "%] "
                      << "FNN full covariance tests passed" << std::endl;
        } else {
            std::cout << "[ " << floor((100 / 11) * 3) << "%] "
                      << "FNN full covariance tests failed" << std::endl;
        }

        if (test_fnn_derivatives_cpu(false, test_dates[3], "fnn_derivatives",
                                     "1D")) {
            // test_fnn_derivatives_cpu(false, test_dates[3], "fnn_derivatives",
            //                          "Boston_housing")) {
            std::cout << "[ " << floor((100 / 11) * 4) << "%] "
                      << "FNN derivatives tests passed" << std::endl;
        } else {
            std::cout << "[ " << floor((100 / 11) * 4) << "%] "
                      << "FNN derivatives tests failed" << std::endl;
        }
    }

    // test_fnn_heteroscedastic_cpu();

    // test_fnn_full_cov_cpu();

    // test_fnn_derivatives_cpu();

    // test_cnn_cpu();

    // test_cnn_batch_norm_cpu();

    // test_autoencoder_cpu();

    // test_lstm_cpu();

    // test_cnn_resnet_cpu();

    // #########################
    // REINIZIALIZE TEST OUTPUTS
    // #########################
    if (user_input_options.size() == 1) {
        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "fnn") {
            // Perform test on CPU for the different architectures
            std::cout << "Reinizializing FNN test outputs" << std::endl;
            // test_fnn_cpu(true, date, "fnn", "Boston_housing");
            test_fnn_cpu(true, date, "fnn", "1D");
            test_fnn_cpu(true, date, "fnn", "Boston_housing");

            write_dates(test_dates, 0, date);
        }

        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "fnn_heteros") {
            std::cout << "Reinizializing FNN heteroscedastic test outputs"
                      << std::endl;
            test_fnn_heteroscedastic_cpu(true, date, "fnn_heteros", "1D");
            test_fnn_heteroscedastic_cpu(true, date, "fnn_heteros",
                                         "Boston_housing");
            write_dates(test_dates, 1, date);
        }

        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "fnn_full_cov") {
            std::cout << "Reinizializing FNN full covariance test outputs"
                      << std::endl;
            test_fnn_full_cov_cpu(true, date, "fnn_full_cov", "1D");
            test_fnn_full_cov_cpu(true, date, "fnn_full_cov", "Boston_housing");
            write_dates(test_dates, 2, date);
        }

        if (reinizialize_test_outputs == "all" ||
            reinizialize_test_outputs == "fnn_derivates") {
            std::cout << "Reinizializing FNN derivatives test outputs"
                      << std::endl;
            test_fnn_derivatives_cpu(true, date, "fnn_derivatives", "1D");
            // test_fnn_derivatives_cpu(true, date, "fnn_derivatives",
            //                          "Boston_housing");
            write_dates(test_dates, 3, date);
        }
    }
}
