#include "atsinfer-cache.h"
#include <algorithm>
#include <cmath>
#include <limits>

ATSInferTensorCache::ATSInferTensorCache(size_t total_vram_budget_bytes)
    : total_budget(total_vram_budget_bytes),
      currently_allocated_gpu_bytes(0),
      policy(ATSInferEvictionPolicy::LRU) {
}

ATSInferTensorCache::~ATSInferTensorCache() {
}

void ATSInferTensorCache::set_eviction_policy(ATSInferEvictionPolicy new_policy) {
    std::lock_guard<std::mutex> guard(lock);
    policy = new_policy;
}

bool ATSInferTensorCache::register_tensor(
    const std::string & name,
    size_t size_bytes,
    int layer_id,
    bool is_moe,
    bool is_static,
    ATSInferResidency initial_residency) {

    std::lock_guard<std::mutex> guard(lock);
    atsinfer_tensor_state state;
    state.tensor_name = name;
    state.size_bytes = size_bytes;
    state.layer_id = layer_id;
    state.is_moe_expert = is_moe;
    state.is_static = is_static;
    state.residency = initial_residency;

    if (initial_residency == ATSInferResidency::GPU_ONLY || initial_residency == ATSInferResidency::CPU_AND_GPU) {
        currently_allocated_gpu_bytes += size_bytes;
    }

    tensors[name] = state;
    return true;
}

bool ATSInferTensorCache::reserve_gpu_space(
    const std::string & tensor_name,
    size_t size_bytes,
    int current_layer_id,
    std::vector<std::string> & evicted_tensors) {

    std::lock_guard<std::mutex> guard(lock);
    evicted_tensors.clear();

    if (total_budget > 0 && size_bytes > total_budget) {
        return false; // Fits nowhere
    }

    auto it = tensors.find(tensor_name);
    if (it != tensors.end()) {
        if (it->second.residency == ATSInferResidency::GPU_ONLY || it->second.residency == ATSInferResidency::CPU_AND_GPU) {
            return true; // Already resident
        }
    }

    while (total_budget > 0 && (currently_allocated_gpu_bytes + size_bytes > total_budget)) {
        std::string victim = select_eviction_candidate(current_layer_id);
        if (victim.empty()) {
            return false; // Cannot evict anything (all pinned or in-flight)
        }

        auto & victim_state = tensors[victim];
        victim_state.residency = ATSInferResidency::CPU_ONLY;
        if (currently_allocated_gpu_bytes >= victim_state.size_bytes) {
            currently_allocated_gpu_bytes -= victim_state.size_bytes;
        } else {
            currently_allocated_gpu_bytes = 0;
        }

        evicted_tensors.push_back(victim);
    }

    currently_allocated_gpu_bytes += size_bytes;
    if (it != tensors.end()) {
        it->second.residency = ATSInferResidency::CPU_AND_GPU;
    }

    return true;
}

bool ATSInferTensorCache::pin_tensor(const std::string & name) {
    std::lock_guard<std::mutex> guard(lock);
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        it->second.is_pinned = true;
        return true;
    }
    return false;
}

bool ATSInferTensorCache::unpin_tensor(const std::string & name) {
    std::lock_guard<std::mutex> guard(lock);
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        it->second.is_pinned = false;
        return true;
    }
    return false;
}

void ATSInferTensorCache::acquire_ref(const std::string & name) {
    std::lock_guard<std::mutex> guard(lock);
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        it->second.ref_count++;
    }
}

void ATSInferTensorCache::release_ref(const std::string & name) {
    std::lock_guard<std::mutex> guard(lock);
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        if (it->second.ref_count > 0) {
            it->second.ref_count--;
        }
    }
}

void ATSInferTensorCache::update_usage(const std::string & name, uint64_t timestamp) {
    std::lock_guard<std::mutex> guard(lock);
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        it->second.last_used_timestamp = timestamp;
    }
}

atsinfer_tensor_state * ATSInferTensorCache::get_tensor_state(const std::string & name) {
    std::lock_guard<std::mutex> guard(lock);
    auto it = tensors.find(name);
    if (it != tensors.end()) {
        return &it->second;
    }
    return nullptr;
}

size_t ATSInferTensorCache::get_used_vram_bytes() const {
    std::lock_guard<std::mutex> guard(lock);
    return currently_allocated_gpu_bytes;
}

size_t ATSInferTensorCache::get_free_vram_bytes() const {
    std::lock_guard<std::mutex> guard(lock);
    if (total_budget > currently_allocated_gpu_bytes) {
        return total_budget - currently_allocated_gpu_bytes;
    }
    return 0;
}

size_t ATSInferTensorCache::get_total_vram_budget() const {
    std::lock_guard<std::mutex> guard(lock);
    return total_budget;
}

std::string ATSInferTensorCache::select_eviction_candidate(int current_layer_id) {
    std::string candidate = "";
    uint64_t oldest_timestamp = std::numeric_limits<uint64_t>::max();
    int max_distance = -1;

    for (const auto & kv : tensors) {
        const auto & state = kv.second;

        // Skip non-GPU, static, pinned, or in-flight tensors
        if (state.residency == ATSInferResidency::CPU_ONLY) continue;
        if (state.is_static || state.is_pinned || state.ref_count > 0) continue;

        if (policy == ATSInferEvictionPolicy::LRU) {
            if (state.last_used_timestamp < oldest_timestamp) {
                oldest_timestamp = state.last_used_timestamp;
                candidate = kv.first;
            }
        } else if (policy == ATSInferEvictionPolicy::LAYER_DISTANCE) {
            int dist = (state.layer_id >= 0 && current_layer_id >= 0) ? std::abs(state.layer_id - current_layer_id) : 0;
            if (dist > max_distance) {
                max_distance = dist;
                candidate = kv.first;
            }
        } else if (policy == ATSInferEvictionPolicy::MOE_EXPERT_PRIORITY) {
            // Prefer evicting inactive MoE experts first (using LRU order among experts)
            if (state.is_moe_expert) {
                if (state.last_used_timestamp < oldest_timestamp) {
                    oldest_timestamp = state.last_used_timestamp;
                    candidate = kv.first;
                }
            } else if (candidate.empty()) {
                if (state.last_used_timestamp < oldest_timestamp) {
                    oldest_timestamp = state.last_used_timestamp;
                    candidate = kv.first;
                }
            }
        }
    }

    return candidate;
}
