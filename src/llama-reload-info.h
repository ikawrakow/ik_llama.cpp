#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <sys/stat.h>
#include <fstream>

struct llama_model;
struct llama_model_loader;

struct tensor_reload_source {
    std::string   path;
    size_t        data_offset   = 0;
    size_t        nbytes        = 0;
    int64_t       last_mtime    = 0;
    int64_t       last_mtime_ns = 0;

    ggml_backend_buffer_t original_buffer = nullptr;
    void                * original_data   = nullptr;
    ggml_type             original_type     = GGML_TYPE_COUNT;
    size_t                original_nbytes   = 0;
    int64_t               original_ne[GGML_MAX_DIMS];
    size_t                original_nb[GGML_MAX_DIMS];

    struct split_info {
        int64_t ne[GGML_MAX_DIMS];
        size_t  nb[GGML_MAX_DIMS];
        void  * data;
        ggml_backend_buffer_t buffer;
        struct ggml_tensor * tensor = nullptr;
    };
    std::vector<split_info> original_splits;

    std::vector<std::string> sibling_names;
    ggml_split_tensor_t    * original_extra = nullptr;

    enum class reload_state {
        UNINITIALIZED,
        ON_ORIGINAL,
        DETACHED,
        FALLBACK_CPU
    };
    reload_state state = reload_state::UNINITIALIZED;
};

struct reload_info {
    std::unordered_map<std::string, tensor_reload_source> tensor_reload_sources;
    std::atomic<bool> reload_snapshots_done{false};

    reload_info(const llama_model_loader & ml);

    bool reload_tensor(const char * name, llama_model & model);
    bool reload_changed_tensors(llama_model & model);
    void snapshot_all_reload_tensors(llama_model & model);
};
