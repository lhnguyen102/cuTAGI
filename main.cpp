///////////////////////////////////////////////////////////////////////////////
// This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>

#include <iostream>
#include <string>
#include <vector>

#include "include/custom_logger.h"
#

int main(int argc, char* argv[]) {
    // User input file
    std::string user_input_file;
    std::vector<std::string> user_input_options;
    if (argc == 0) {
        LOG(LogLevel::ERROR,
            "User need to provide user input file -> see README");
    } else {
        user_input_file = argv[1];
        for (int i = 2; i < argc; i++) {
            user_input_options.push_back(argv[i]);
        }
    }

    return 0;
}
