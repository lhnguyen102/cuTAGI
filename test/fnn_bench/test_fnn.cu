#include "test_fnn.cuh"

void test_fnn() {
    std::cout << "\033[1;31mThis is red text\033[0m" << std::endl;
    Network net;

    net.layers = {1, 1, 1, 1};
    net.nodes = {13, 10, 15, 1};
    net.activations = {0, 0, 0, 0};
    net.batch_size = 5;

    net.is_idx_ud = true;
    TagiNetwork tagi_net(net);

    SavePath path;
    path.curr_path = get_current_dir();
    path.saved_param_path = path.curr_path + "/saved_param/";

    std::vector<std::string> x_train_dir;
    x_train_dir.push_back(path.curr_path +
                          "/data/UCI/Boston_housing/x_train.csv");

    std::vector<std::string> y_train_dir;
    y_train_dir.push_back(path.curr_path +
                          "/data/UCI/Boston_housing/y_train.csv");

    std::vector<std::string> x_test_dir;
    x_test_dir.push_back(path.curr_path +
                         "/data/UCI/Boston_housing/x_test.csv");

    std::vector<std::string> y_test_dir;
    y_test_dir.push_back(path.curr_path +
                         "/data/UCI/Boston_housing/y_test.csv");

    std::string model_name = "1D";
    std::string net_name = "test_fnn_bench_net_2";

    // save_net_param(model_name, net_name, path.curr_path + "/test/data1/",
    //               tagi_net.theta);

    // Load_net_param
    // load_net_param(model_name, net_name, path.saved_param_path,
    // tagi_net.theta);

    read_params(
        path.curr_path +
            "/test/data/2023_01_26_init_param_fnn_bench_Boston_housing.csv",
        tagi_net.theta);

    // Train data
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    auto train_db =
        get_dataloader(x_train_dir, y_train_dir, mu_x, sigma_x, mu_y, sigma_y,
                       455, tagi_net.prop.n_x, tagi_net.prop.n_y, true);

    // Test data
    auto test_db = get_dataloader(
        x_test_dir, y_test_dir, train_db.mu_x, train_db.sigma_x, train_db.mu_y,
        train_db.sigma_y, 51, tagi_net.prop.n_x, tagi_net.prop.n_y, true);

    // Training
    bool train_mode = true;
    regression(tagi_net, train_db, 50, path, train_mode, false);

    train_mode = false;
    regression(tagi_net, test_db, 50, path, train_mode, false);

    // save_net_param(model_name, net_name, path.curr_path + "/test/data2/",
    //               tagi_net.theta);

    std::cout << "test_fnn end" << std::endl;

    // write_params(
    //     path.curr_path +
    //         "/test/data/2023_01_26_opt_param_fnn_bench_Boston_housing.csv",
    //     tagi_net.theta.mw, tagi_net.theta.Sw, tagi_net.theta.mb,
    //     tagi_net.theta.Sb, tagi_net.theta.mw_sc, tagi_net.theta.Sw_sc,
    //     tagi_net.theta.mb_sc, tagi_net.theta.Sb_sc, false);

    write_params(
        path.curr_path +
            "/test/data/2023_01_26_opt_param_fnn_bench_Boston_housing.csv",
        tagi_net.theta);

    std::string filename =
        path.curr_path +
        "/test/data/"
        "2023_01_26_forward_hidden_states_fnn_bench_Boston_housing.csv";

    write_forward_hidden_states(filename, tagi_net.state);

    std::string filename_backward =
        path.curr_path +
        "/test/data/"
        "2023_01_26_backward_hidden_states_fnn_bench_Boston_housing.csv";

    write_backward_hidden_states(filename_backward, tagi_net,
                                 net.layers.size() - 2);
}