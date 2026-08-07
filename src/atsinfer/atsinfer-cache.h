#ifndef ATSINFER_CACHE_H
#define ATSINFER_CACHE_H

#include "atsinfer-placement.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

enum class ATSInferResidency {
    CPU_ONLY,
    GPU_ONLY,
    CPU_AND_GPU
};

struct atsinfer_tensor_state {
    std::string tensor_name;
    size_t size_bytes = 0;
    void * host_ptr = nullptr;
    void * device_ptr = nullptr;
    ATSInferResidency residency = ATSInferResidency::CPU_ONLY;
    bool is_static = false;       // True if placed on GPU during static model loading
    bool is_pinned = false;       // Locked in GPU memory
    bool is_dirty = false;
    uint64_t last_used_timestamp = 0;
    int ref_count = 0;            // In-flight compute reference count
    int layer_id = -1;
    bool is_moe_expert = false;
};

enum class ATSInferEvictionPolicy {
    LRU,
    LAYER_DISTANCE,
    MOE_EXPERT_PRIORITY
};

class ATSInferTensorCache {
public:
    ATSInferTensorCache(size_t total_vram_budget_bytes = 0);
    ~ATSInferTensorCache();

    void set_eviction_policy(ATSInferEvictionPolicy policy);

    bool register_tensor(const std::string & name, size_t size_bytes, int layer_id, bool is_moe, bool is_static, ATSInferResidency initial_residency);

    // Reserve space for dynamic GPU promotion; triggers eviction if needed
    bool reserve_gpu_space(const std::string & tensor_name, size_t size_bytes, int current_layer_id, std::vector<std::string> & evicted_tensors);

    bool pin_tensor(const std::string & name);
    bool unpin_tensor(const std::string & name);

    void acquire_ref(const std::string & name);
    void release_ref(const std::string & name);

    void update_usage(const std::string & name, uint64_t timestamp);

    atsinfer_tensor_state * get_tensor_state(const std::string & name);

    size_t get_used_vram_bytes() const;
    size_t get_free_vram_bytes() const;
    size_t get_total_vram_budget() const;

private:
    std::string select_eviction_candidate(int current_layer_id);

    mutable std::mutex lock;
    size_t total_budget;
    size_t currently_allocated_gpu_bytes;
    ATSInferEvictionPolicy policy;
    std::unordered_map<std::string, atsinfer_tensor_state> tensors;
};

#endif // ATSINFER_CACHE_H
