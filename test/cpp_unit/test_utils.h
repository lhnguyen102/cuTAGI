#pragma once
#include <string>

#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/sequential.h"

void fnn_mnist_test(Sequential& model, float threshold,
                    float& avg_error_output);