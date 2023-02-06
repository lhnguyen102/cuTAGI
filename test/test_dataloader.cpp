#include "test_dataloader.h"

/**
 * @brief Compare two csv files.
 *
 * @param[in] file1 the first file to compare
 * @param[in] file2 the second file to compare
 */
bool compare_csv_files(const std::string &file1, const std::string &file2) {
    std::ifstream f1(file1);
    std::ifstream f2(file2);

    if (!f1.is_open() || !f2.is_open()) {
        std::cout << "Error opening one of the files." << std::endl;
        return false;
    }

    std::string line1, line2;
    int lineNumber = 1;

    while (std::getline(f1, line1) && std::getline(f2, line2)) {
        if (line1 != line2) {
            std::cout << "Files differ at line " << lineNumber << std::endl;
            std::cout << "File 1: " << line1 << std::endl;
            std::cout << "File 2: " << line2 << std::endl;
            return false;
        }
        lineNumber++;
    }

    if (std::getline(f1, line1) || std::getline(f2, line2)) {
        std::cout << "Files have different number of lines." << std::endl;
        return false;
    }

    f1.close();
    f2.close();

    return true;
}

/**
 * @brief Train the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 */
Dataloader train_data(std::string problem, TagiNetworkCPU &net,
                      std::string data_path, bool normalize) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_train_data;

    if (problem == "Boston_housing") {
        x_dir = data_path + "/x_train.csv";
        y_dir = data_path + "/y_train.csv";
        num_train_data = 455;
    } else if (problem == "1D") {
        num_train_data = 20;
        x_dir = data_path + "/x_train.csv";
        y_dir = data_path + "/y_train.csv";
    } else {
        num_train_data = 400;
        x_dir = data_path + "/x_train.csv";
        y_dir = data_path + "/y_train.csv";
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    // Train data

    // Initialize the mu and sigma
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    return get_dataloader(x_path, y_path, mu_x, sigma_x, mu_y, sigma_y,
                          num_train_data, net.prop.n_x, net.prop.n_y,
                          normalize);
}

/**
 * @brief Test the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 * @param train_db contains the training data
 */
Dataloader test_data(std::string problem, TagiNetworkCPU &net,
                     std::string data_path, Dataloader &train_db,
                     bool normalize) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_test_data;

    if (problem == "Boston_housing") {
        x_dir = data_path + "/x_test.csv";
        y_dir = data_path + "/y_test.csv";
        num_test_data = 51;
    } else if (problem == "1D") {
        num_test_data = 100;
        x_dir = data_path + "/x_test.csv";
        y_dir = data_path + "/y_test.csv";
    } else {
        num_test_data = 200;
        x_dir = data_path + "/x_test.csv";
        y_dir = data_path + "/y_test.csv";
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    // Test data
    return get_dataloader(x_path, y_path, train_db.mu_x, train_db.sigma_x,
                          train_db.mu_y, train_db.sigma_y, num_test_data,
                          net.prop.n_x, net.prop.n_y, normalize);
}