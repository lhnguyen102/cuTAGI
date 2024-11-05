#include <gtest/gtest.h>

#include <string>

bool g_gpu_enabled = true;

int main(int argc, char **argv) {
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") {
            g_gpu_enabled = false;
            std::cout << "GPU mode enabled.\n";
        }
    }

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
