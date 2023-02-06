#include "test_utils.h"

template <typename T>
void read_params(std::string filename, T &theta) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for reading." << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line);
    int row = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        int column = 0;
        while (std::getline(iss, value, ',')) {
            if (column == 0 && !value.empty()) theta.mw[row] = std::stod(value);
            if (column == 1 && !value.empty()) theta.Sw[row] = std::stod(value);
            if (column == 2 && !value.empty()) theta.mb[row] = std::stod(value);
            if (column == 3 && !value.empty()) theta.Sb[row] = std::stod(value);
            if (column == 4 && !value.empty())
                theta.mw_sc[row] = std::stod(value);
            if (column == 5 && !value.empty())
                theta.Sw_sc[row] = std::stod(value);
            if (column == 6 && !value.empty())
                theta.mb_sc[row] = std::stod(value);
            if (column == 7 && !value.empty())
                theta.Sb_sc[row] = std::stod(value);
            column++;
        }
        row++;
    }

    file.close();
}

template <typename T>
void write_params(std::string filename, T &theta) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    file << "mw,Sw,mb,Sb,mw_sc,Sw_sc,mb_sc,Sb_sc" << std::endl;

    int rows =
        std::max({theta.mw.size(), theta.Sw.size(), theta.mb.size(),
                  theta.Sb.size(), theta.mw_sc.size(), theta.Sw_sc.size(),
                  theta.mb_sc.size(), theta.Sb_sc.size()});
    for (int i = 0; i < rows; i++) {
        if (i < theta.mw.size())
            file << theta.mw[i] << ",";
        else
            file << ",";

        if (i < theta.Sw.size())
            file << theta.Sw[i] << ",";
        else
            file << ",";

        if (i < theta.mb.size())
            file << theta.mb[i] << ",";
        else
            file << ",";

        if (i < theta.Sb.size())
            file << theta.Sb[i] << ",";
        else
            file << ",";

        if (i < theta.mw_sc.size())
            file << theta.mw_sc[i] << ",";
        else
            file << ",";

        if (i < theta.Sw_sc.size())
            file << theta.Sw_sc[i] << ",";
        else
            file << ",";

        if (i < theta.mb_sc.size())
            file << theta.mb_sc[i] << ",";
        else
            file << ",";

        if (i < theta.Sb_sc.size())
            file << theta.Sb_sc[i] << std::endl;
        else
            file << std::endl;
    }

    file.close();
}

template <typename T>
void write_forward_hidden_states(std::string filename, T &net_state) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    file << "mz,Sz,ma,Sa,J" << std::endl;

    int rows =
        std::max({net_state.mz.size(), net_state.Sz.size(), net_state.ma.size(),
                  net_state.Sa.size(), net_state.J.size()});
    for (int i = 0; i < rows; i++) {
        if (i < net_state.mz.size())
            file << net_state.mz[i] << ",";
        else
            file << ",";

        if (i < net_state.Sz.size())
            file << net_state.Sz[i] << ",";
        else
            file << ",";

        if (i < net_state.ma.size())
            file << net_state.ma[i] << ",";
        else
            file << ",";

        if (i < net_state.Sa.size())
            file << net_state.Sa[i] << ",";
        else
            file << ",";

        if (i < net_state.J.size())
            file << net_state.J[i] << std::endl;
        else
            file << std::endl;
    }

    file.close();
}
template <typename T>
void write_backward_hidden_states(std::string filename, T &tagi_net,
                                  int num_layers) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    int rows = 0;
    std::vector<std::vector<float>> mean, var;
    mean.clear();
    var.clear();
    for (int i = 0; i < num_layers; i++) {
        mean.push_back(std::get<0>(tagi_net.get_inovation_mean_var(i)));
        var.push_back(std::get<1>(tagi_net.get_inovation_mean_var(i)));
        if (mean[i].size() > rows) rows = mean[i].size();
        if (var[i].size() > rows) rows = var[i].size();
        file << "mean_l" << i + 1 << ",var_l" << i + 1 << ",";

        if (i == num_layers - 1) file << std::endl;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < num_layers; j++) {
            if (i < mean[j].size())
                file << mean[j][i] << ",";
            else
                file << ",";

            if (i < var[j].size() && j != num_layers - 1)
                file << var[j][i] << ",";
            else if (i < var[j].size() && j == num_layers - 1)
                file << var[j][i];
        }
        file << std::endl;
    }
}