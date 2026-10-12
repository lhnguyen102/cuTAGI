#include <gtest/gtest.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>

#include "../../include/data_struct_cuda.cuh"

extern bool g_gpu_enabled;

TEST(DeltaStateCuda, GrowingSizeLeavesNoCudaError) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    cudaGetLastError();
    DeltaStateCuda delta_states(4, 1);
    delta_states.set_size(1 << 20, 1);
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
#endif
