#include "../include/config.h"
#include "../include/cuda_error_checking.cuh"
#include "../include/custom_logger.h"
#include "../include/norm_layer.h"
#include "../include/norm_layer_cuda.cuh"

#define WARP_SIZE 32
