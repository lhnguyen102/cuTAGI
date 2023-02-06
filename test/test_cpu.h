#include <chrono>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>

#include "fnn/test_fnn_cpu.h"
#include "fnn_full_cov/test_fnn_full_cov_cpu.h"
#include "fnn_heteros/test_fnn_heteroscedastic_cpu.h"
// #include "fnn_derivatives/test_fnn_derivatives_cpu.h"
// #include "cnn/test_cnn_cpu.h"
// #include "cnn_batch_norm/test_cnn_batch_norm_cpu.h"
// #include "autoencoder/test_autoencoder_cpu.h"
// #include "lstm/test_lstm_cpu.h"

void test_cpu(std::vector<std::string> &user_input_options);