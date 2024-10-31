#pragma once
#include <string>

#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/sequential.h"

void sin_signal_lstm_test_runner(Sequential &model, int input_seq_len,
                                 float &mse, float &log_lik);

void sin_signal_smoother_test_runner(Sequential &model, int input_seq_len,
                                     int num_features, float &mse,
                                     float &log_lik);
