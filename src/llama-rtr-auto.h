#pragma once

#include <cstdint>
#include <limits>

struct llama_rtr_auto_peak_inputs {
    uint64_t model_bytes;
    uint64_t repack_workspace_bytes;
    uint64_t max_read_buffer_bytes;
    uint64_t n_load_workers;
    uint64_t cuda_staging_bytes;
    bool     use_cuda_staging;
};

// Returns false rather than wrapping on any arithmetic overflow.
static inline bool llama_rtr_auto_peak_bytes(
        const llama_rtr_auto_peak_inputs & input,
        uint64_t & peak_bytes) {
    const uint64_t max = std::numeric_limits<uint64_t>::max();
    if (input.n_load_workers != 0 && input.max_read_buffer_bytes > max / input.n_load_workers) {
        return false;
    }
    const uint64_t read_bytes = input.n_load_workers * input.max_read_buffer_bytes;
    uint64_t staging_bytes = 0;
    if (input.use_cuda_staging) {
        if (input.n_load_workers != 0 && input.cuda_staging_bytes > max / input.n_load_workers) {
            return false;
        }
        staging_bytes = input.n_load_workers * input.cuda_staging_bytes;
    }
    if (input.model_bytes > max - input.repack_workspace_bytes) {
        return false;
    }
    const uint64_t with_workspace = input.model_bytes + input.repack_workspace_bytes;
    if (with_workspace > max - read_bytes) {
        return false;
    }
    const uint64_t with_reads = with_workspace + read_bytes;
    if (with_reads > max - staging_bytes) {
        return false;
    }
    peak_bytes = with_reads + staging_bytes;
    return true;
}
