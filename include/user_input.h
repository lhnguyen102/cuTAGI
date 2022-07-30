///////////////////////////////////////////////////////////////////////////////
// File:         user_input.h
// Description:  Load user input
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 05, 2022
// Updated:      July 24, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "struct_var.h"

UserInput load_userinput(std::string &user_input_file);