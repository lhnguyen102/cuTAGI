#pragma once
#include <vector>

#include "data_struct.h"

// obs_to_class (cost.h) for a batch of rows on the device. mz and Sz hold
// rows x hs.len values, P receives rows x n_classes class probabilities.
void obs_to_class_cuda(const float *mz, const float *Sz, int rows,
                       const HRCSoftmax &hs, int n_classes,
                       const std::vector<float> &level_scale,
                       const std::vector<float> &level_offset, float *P);
