// #include <gtest/gtest.h>

// #include <chrono>
// #include <cstdlib>
// #include <memory>
// #include <random>
// #include <stdexcept>
// #include <string>
// #include <vector>

// #include "../../include/activation.h"
// #include "../../include/base_output_updater.h"
// #include "../../include/batchnorm_layer.h"
// #include "../../include/conv2d_layer.h"
// #include "../../include/custom_logger.h"
// #include "../../include/data_struct.h"
// #include "../../include/dataloader.h"
// #include "../../include/distributed.h"
// #include "../../include/linear_layer.h"
// #include "../../include/pooling_layer.h"
// #include "../../include/sequential.h"

// extern bool g_gpu_enabled;

// void distributed_mnist_test_runner(DistributedSequential &model,
//                                    float &avg_error_output) {
//     //////////////////////////////////////////////////////////////////////
//     // Data preprocessing
//     //////////////////////////////////////////////////////////////////////
//     std::vector<std::string> x_train_paths = {
//         "./data/mnist/train-images-idx3-ubyte"};
//     std::vector<std::string> y_train_paths = {
//         "./data/mnist/train-labels-idx1-ubyte"};

//     std::string data_name = "mnist";
//     std::vector<float> mu = {0.1309};
//     std::vector<float> sigma = {2.0f};
//     int num_train_data = 60000;
//     int num_classes = 10;
//     int width = 28;
//     int height = 28;
//     int channel = 1;
//     int n_x = width * height;
//     int n_y = 11;

//     auto train_db = get_images_v2(data_name, x_train_paths, y_train_paths, mu,
//                                   sigma, num_train_data, num_classes, width,
//                                   height, channel, true);

//     ////////////////////////////////////////////////////////////////////////////
//     // Training
//     ////////////////////////////////////////////////////////////////////////////
//     OutputUpdater output_updater("cuda");

//     unsigned seed = 42;
//     std::default_random_engine seed_e(seed);
//     int batch_size = 32;  // Increased batch size for distributed training
//     float sigma_obs = 1.0;

//     // Get local batch size based on world size
//     int local_batch_size = batch_size / model.config.world_size;
//     int local_start_idx = model.config.rank * local_batch_size;

//     std::vector<float> x_batch(local_batch_size * n_x, 0.0f);
//     std::vector<float> var_obs(local_batch_size * train_db.output_len,
//                                pow(sigma_obs, 2));
//     std::vector<float> y_batch(local_batch_size * train_db.output_len, 0.0f);
//     std::vector<int> batch_idx(local_batch_size);
//     std::vector<int> idx_ud_batch(train_db.output_len * local_batch_size, 0);
//     std::vector<int> label_batch(local_batch_size, 0);
//     std::vector<float> mu_a_output(local_batch_size * n_y, 0);
//     std::vector<float> var_a_output(local_batch_size * n_y, 0);
//     auto data_idx = create_range(train_db.num_data);

//     int mt_idx = 0;
//     std::vector<int> error_rate(train_db.num_data, 0);
//     std::vector<int> error_rate_batch;
//     std::vector<float> prob_class_batch;

//     std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

//     // Training loop
//     for (int i = 0; i < 100; i++) {
//         // Get local batch for this rank
//         for (int j = 0; j < local_batch_size; j++) {
//             int global_idx = i * batch_size + local_start_idx + j;
//             batch_idx[j] = data_idx[global_idx % train_db.num_data];
//         }

//         get_batch_images_labels(train_db, batch_idx, local_batch_size, 0,
//                                 x_batch, y_batch, idx_ud_batch, label_batch);

//         model.forward(x_batch);
//         output_updater.update_using_indices(*model.model->output_z_buffer,
//                                             y_batch, var_obs, idx_ud_batch,
//                                             *model.model->input_delta_z_buffer);
//         model.backward();
//         model.step();

//         // Extract outputs for error calculation
//         for (int j = 0; j < local_batch_size * n_y; j++) {
//             mu_a_output[j] = model.model->output_z_buffer->mu_a[j];
//             var_a_output[j] = model.model->output_z_buffer->var_a[j];
//         }

//         std::tie(error_rate_batch, prob_class_batch) =
//             get_error(mu_a_output, var_a_output, label_batch, num_classes,
//                       local_batch_size);

//         mt_idx = i * local_batch_size;
//         update_vector(error_rate, error_rate_batch, mt_idx, 1);
//     }

//     int curr_idx = mt_idx + local_batch_size;
//     avg_error_output = compute_average_error_rate(error_rate, curr_idx, 100);
// }

// class DistributedTest : public ::testing::Test {
//    protected:
//     void SetUp() override {
//         // Check for MNIST data files and download if needed
//         const std::string x_train_path = "./data/mnist/train-images-idx3-ubyte";
//         const std::string y_train_path = "./data/mnist/train-labels-idx1-ubyte";

//         if (!file_exists(x_train_path) || !file_exists(y_train_path)) {
//             std::cout << "MNIST training data files are missing. Downloading..."
//                       << std::endl;
//             if (!download_mnist_data()) {
//                 std::cerr << "Failed to download MNIST data files."
//                           << std::endl;
//             }
//         }
//     }

//     bool file_exists(const std::string &path) {
//         std::ifstream file(path);
//         return file.good();
//     }

//     bool download_mnist_data() {
//         if (system("mkdir -p ./data/mnist") != 0) {
//             std::cerr << "Failed to create directory ./data/mnist" << std::endl;
//             return false;
//         }

//         std::string base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/";
//         std::vector<std::string> files = {"train-images-idx3-ubyte",
//                                           "train-labels-idx1-ubyte"};

//         for (const auto &file : files) {
//             std::string cmd = "curl --fail -o ./data/mnist/" + file + ".gz " +
//                               base_url + file + ".gz";
//             if (system(cmd.c_str()) != 0) {
//                 std::cerr << "Failed to download " << file << std::endl;
//                 return false;
//             }
//             cmd = "gunzip -f ./data/mnist/" + file + ".gz";
//             if (system(cmd.c_str()) != 0) {
//                 std::cerr << "Failed to extract " << file << std::endl;
//                 return false;
//             }
//         }
//         return true;
//     }
// };

// #ifdef USE_CUDA
// TEST_F(DistributedTest, NCCLDistributedTraining) {
//     if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

//     // Initialize MPI for NCCL ID broadcast
// #ifdef USE_MPI
//     int initialized;
//     MPI_Initialized(&initialized);
//     if (!initialized) {
//         int provided;
//         MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
//     }

//     int rank, world_size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     if (world_size != 2) {
//         GTEST_SKIP() << "This test requires exactly 2 MPI processes";
//     }

//     // Create base model
//     auto base_model = std::make_shared<Sequential>(
//         Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(8), ReLU(),
//         AvgPool2d(3, 2), Conv2d(8, 8, 5, false), BatchNorm2d(8), ReLU(),
//         AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));

//     // Configure distributed training with NCCL
//     std::vector<int> device_ids = {0, 1};  // Using GPUs 0 and 1
//     DistributedConfig config(device_ids, "nccl", rank, world_size);

//     // Create distributed model
//     DistributedSequential dist_model(base_model, config);

//     float avg_error;
//     float threshold = 0.5;
//     distributed_mnist_test_runner(dist_model, avg_error);

//     // Only check the result on rank 0
//     if (rank == 0) {
//         EXPECT_LT(avg_error, threshold)
//             << "Error rate is higher than threshold";
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
// #else
//     GTEST_SKIP() << "MPI support is required for this test";
// #endif
// }
// #endif