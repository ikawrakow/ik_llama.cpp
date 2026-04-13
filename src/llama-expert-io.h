#pragma once

#include <cstddef>
#include <vector>

struct llama_file_range {
    size_t first = 0;
    size_t last  = 0;

    bool empty() const {
        return first >= last;
    }
};

struct llama_expert_tensor_index {
    size_t deferred_bytes = 0;
    size_t dense_bytes = 0;

    std::vector<std::vector<llama_file_range>> file_ranges;

    bool empty() const {
        return deferred_bytes == 0;
    }
};
