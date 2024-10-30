#pragma once
#include <string>

#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/sequential.h"

void mnist_test_runner(Sequential &model, float &avg_error_output);

void sin_signal_lstm_test_runner(Sequential &model, int input_seq_len,
                                 float &mse, float &log_lik);

void sin_signal_smoother_test_runner(Sequential &model, int input_seq_len,
                                     int num_features, float &mse,
                                     float &log_lik);

void heteros_test_runner(Sequential &model, float &mse, float &log_lik);