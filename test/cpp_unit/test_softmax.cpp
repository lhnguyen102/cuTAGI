#include <gtest/gtest.h>

#include <vector>

#include "../../include/activation.h"
#include "../../include/data_struct.h"

TEST(Softmax, EachRowSumsToOne) {
    int batch_size = 2, num_classes = 3;
    int size = batch_size * num_classes;
    BaseHiddenStates input_states, output_states;
    input_states.mu_a = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 4.0f};
    input_states.var_a.assign(size, 0.1f);
    input_states.jcb.assign(size, 1.0f);
    input_states.block_size = batch_size;
    input_states.actual_size = num_classes;
    input_states.size = size;
    output_states.mu_a.resize(size);
    output_states.var_a.resize(size);
    output_states.jcb.resize(size);
    BaseTempStates temp_states;

    Softmax softmax;
    softmax.forward(input_states, output_states, temp_states);

    for (int b = 0; b < batch_size; b++) {
        float row_sum = 0.0f;
        for (int k = 0; k < num_classes; k++) {
            row_sum += output_states.mu_a[b * num_classes + k];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "row " << b;
    }
}
