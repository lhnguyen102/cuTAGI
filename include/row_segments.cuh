#pragma once
#include <cstddef>

// Items keyed by a row id, stably sorted by row and run-length encoded into
// one segment per touched row. Items inside a segment keep their input order,
// so a sequential reduction over a segment is deterministic.
class RowSegments {
   public:
    int *d_row = nullptr;  // row id of each item, written by the caller
    int *d_sorted_item =
        nullptr;  // input position of each item, grouped by row
    int *d_touched_row = nullptr;  // distinct rows, ascending
    int *d_segment_offset = nullptr;
    int *d_segment_count = nullptr;
    int num_touched = 0;

    RowSegments() = default;
    ~RowSegments();
    RowSegments(const RowSegments &) = delete;
    RowSegments &operator=(const RowSegments &) = delete;

    // Grows the device buffers to hold num_items; keeps them otherwise.
    void reserve(int num_items);
    // Groups the first num_items entries of d_row and sets num_touched.
    void group(int num_items);

   private:
    int *d_item_id = nullptr;
    int *d_sorted_row = nullptr;
    int *d_num_touched = nullptr;
    void *d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    int capacity = 0;
};
