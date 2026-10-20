#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <memory>
#include <vector>

struct llama_model;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer_type;
struct ggml_moe_provider;

class llama_moe_cache {
public:
    llama_moe_cache(const llama_model & model,
            const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts,
            size_t size, bool async_mode = false);
    ~llama_moe_cache();

    ggml_backend * backend(int32_t il) const;
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;
    ggml_tensor * get_experts(const ggml_tensor * tensor) const;
    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst,
            ggml_cgraph * graph, ggml_backend_sched_copy_phase phase);

    const ggml_moe_provider * provider() const;
    bool has_eligible_layers() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
