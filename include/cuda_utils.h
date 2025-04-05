#pragma once

#include <string>
#include <vector>

bool is_cuda_available();

int get_cuda_device_count();

int get_current_cuda_device();

bool set_cuda_device(int device_index);

bool is_cuda_device_available(int device_index);

bool is_nccl_available();

std::string get_cuda_device_properties(int device_index);

bool get_cuda_device_memory(int device_index, size_t& free_memory,
                            size_t& total_memory);
