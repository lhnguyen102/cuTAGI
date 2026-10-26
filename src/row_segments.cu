#include <algorithm>
#include <cub/cub.cuh>
#include <vector>

#include "../include/cuda_error_checking.cuh"
#include "../include/row_segments.cuh"

RowSegments::~RowSegments() {
    cudaFree(this->d_row);
    cudaFree(this->d_item_id);
    cudaFree(this->d_sorted_row);
    cudaFree(this->d_sorted_item);
    cudaFree(this->d_touched_row);
    cudaFree(this->d_segment_count);
    cudaFree(this->d_segment_offset);
    cudaFree(this->d_num_touched);
    cudaFree(this->d_sort_temp);
}

void RowSegments::reserve(int num_items)
/*
 */
{
    if (num_items <= this->capacity) {
        return;
    }
    for (int **buf : {&this->d_row, &this->d_item_id, &this->d_sorted_row,
                      &this->d_sorted_item, &this->d_touched_row,
                      &this->d_segment_count, &this->d_segment_offset}) {
        cudaFree(*buf);
        CHECK_CUDA_ERROR(cudaMalloc((void **)buf, num_items * sizeof(int)));
    }
    if (this->d_num_touched == nullptr) {
        CHECK_CUDA_ERROR(
            cudaMalloc((void **)&this->d_num_touched, sizeof(int)));
    }
    std::vector<int> iota(num_items);
    for (int i = 0; i < num_items; i++) iota[i] = i;
    cudaMemcpy(this->d_item_id, iota.data(), num_items * sizeof(int),
               cudaMemcpyHostToDevice);

    size_t sort_bytes = 0, encode_bytes = 0, scan_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, this->d_row,
                                    this->d_sorted_row, this->d_item_id,
                                    this->d_sorted_item, num_items);
    cub::DeviceRunLengthEncode::Encode(
        nullptr, encode_bytes, this->d_sorted_row, this->d_touched_row,
        this->d_segment_count, this->d_num_touched, num_items);
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, this->d_segment_count,
                                  this->d_segment_offset, num_items);
    this->sort_temp_bytes =
        std::max(sort_bytes, std::max(encode_bytes, scan_bytes));
    cudaFree(this->d_sort_temp);
    CHECK_CUDA_ERROR(cudaMalloc(&this->d_sort_temp, this->sort_temp_bytes));
    this->capacity = num_items;
}

void RowSegments::group(int num_items)
/*
 */
{
    cub::DeviceRadixSort::SortPairs(
        this->d_sort_temp, this->sort_temp_bytes, this->d_row,
        this->d_sorted_row, this->d_item_id, this->d_sorted_item, num_items);
    cub::DeviceRunLengthEncode::Encode(this->d_sort_temp, this->sort_temp_bytes,
                                       this->d_sorted_row, this->d_touched_row,
                                       this->d_segment_count,
                                       this->d_num_touched, num_items);
    cub::DeviceScan::ExclusiveSum(this->d_sort_temp, this->sort_temp_bytes,
                                  this->d_segment_count, this->d_segment_offset,
                                  num_items);
    cudaMemcpy(&this->num_touched, this->d_num_touched, sizeof(int),
               cudaMemcpyDeviceToHost);
}
