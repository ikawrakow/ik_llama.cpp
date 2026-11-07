#pragma once

#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace llama_moe_cache_policy_detail {

enum class kind { lru, classlru };

inline bool valid(enum llama_moe_cache_policy policy) {
    return policy == LLAMA_MOE_CACHE_POLICY_LRU || policy == LLAMA_MOE_CACHE_POLICY_CLASSLRU;
}

struct state {
    static constexpr int n_class = 8;
    static constexpr float half_life = 128.0f;

    struct fill { int32_t expert; int32_t slot; };

    int32_t n_expert = 0;
    int32_t n_slots = 0;
    kind policy = kind::lru;
    std::vector<int32_t *> slot_map;
    std::vector<int32_t> key_of;
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    std::vector<uint32_t> seen;
    std::vector<int32_t> uniq;
    uint32_t seen_gen = 0;

    int32_t head = -1;
    int32_t tail = -1;
    std::vector<int8_t> list_of;
    std::array<int32_t, n_class> class_head;
    std::array<int32_t, n_class> class_tail;
    std::vector<float> crf;
    std::vector<int64_t> last_use;
    std::vector<int64_t> clock;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, kind policy) {
        GGML_ASSERT(n_layer >= 0 && n_expert > 0 && n_slots >= 0);
        this->n_expert = n_expert;
        this->n_slots = n_slots;
        this->policy = policy;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        seen.assign(n_expert, 0);
        if (policy == kind::lru) {
            head = n_slots ? 0 : -1;
            tail = n_slots - 1;
            return;
        }
        list_of.assign(n_slots, 0);
        class_head.fill(-1);
        class_tail.fill(-1);
        if (n_slots > 0) {
            class_head[0] = 0;
            class_tail[0] = n_slots - 1;
        }
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
    }

    void touch_lru(int32_t s) {
        if (s == tail) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        prev[next[s]] = prev[s];
        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    void move_class(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == class_tail[from]) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else class_head[from] = next[s];
        if (next[s] >= 0) prev[next[s]] = prev[s]; else class_tail[from] = prev[s];
        prev[s] = class_tail[c];
        next[s] = -1;
        if (class_tail[c] >= 0) next[class_tail[c]] = s; else class_head[c] = s;
        class_tail[c] = s;
        list_of[s] = (int8_t) c;
    }

    int32_t make_key(int32_t il, int32_t expert) const { return il*n_expert + expert; }
    int32_t key_layer(int32_t key) const { return key/n_expert; }
    int32_t key_expert(int32_t key) const { return key%n_expert; }

    int class_of(int32_t key) const {
        GGML_ASSERT(std::isfinite(crf[key]) && crf[key] >= 1.0f);
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/half_life);
    }

    void use(int32_t il, int32_t key) {
        crf[key] = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
        GGML_ASSERT(std::isfinite(crf[key]) && crf[key] >= 1.0f);
    }

    bool is_selected(int32_t il, int32_t key) const {
        return key >= 0 && key_layer(key) == il && seen[key_expert(key)] == seen_gen;
    }

    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = class_head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) return s;
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t best = -1;
        double best_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) continue;
            const int32_t key = key_of[s];
            if (key < 0) return s;
            const int64_t age = clock[key_layer(key)] - last_use[key];
            const double score = std::ldexp((double) age, -c);
            if (score > best_score) {
                best = s;
                best_score = score;
            }
        }
        GGML_ASSERT(best >= 0);
        return best;
    }

    bool plan_lru(int32_t il, const int32_t * ids, size_t n_ids,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = 0;
        n_eviction = 0;
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch_lru(slots[e]);
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = head;
            if (key_of[s] >= 0) {
                const int32_t old = key_of[s];
                slot_map[old / n_expert][old % n_expert] = -1;
                ++n_eviction;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch_lru(s);
            fills.push_back({ e, s });
        }
        return true;
    }

    bool plan_classlru(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = n_eviction = 0;
        if (il < 0 || (size_t) il >= slot_map.size() || !slot_map[il] || !ids ||
                n_tokens <= 0 || n_ids == 0 || (uint64_t) n_tokens > n_ids ||
                n_ids % (size_t) n_tokens != 0) return false;
        for (size_t i = 0; i < n_ids; ++i) {
            if (ids[i] < 0 || ids[i] >= n_expert) return false;
        }
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] >= 0) {
                move_class(slots[e], class_of(key));
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                const int32_t old = key_of[s];
                slot_map[key_layer(old)][key_expert(old)] = -1;
                ++n_eviction;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e] = s;
            move_class(s, class_of(key));
            fills.push_back({e, s});
        }
        clock[il] += n_tokens;
        return true;
    }

    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens,
            std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        if (policy == kind::lru) return plan_lru(il, ids, n_ids, fills, n_hit, n_eviction);
        return plan_classlru(il, ids, n_ids, n_tokens, fills, n_hit, n_eviction);
    }
};

inline bool recover_selected_tokens(const ggml_tensor * flattened, int64_t n_expert_used,
        int64_t n_expert, int64_t & n_tokens) {
    if (!flattened || flattened->op != GGML_OP_RESHAPE || !flattened->src[0]) return false;
    const ggml_tensor * source = flattened->src[0];
    if (n_expert_used <= 0 || n_expert <= 0 || n_expert_used > n_expert ||
            source->ne[0] <= 0 || source->ne[0] > n_expert || source->ne[1] <= 0 ||
            source->ne[2] != 1 || source->ne[3] != 1 || flattened->ne[0] != ggml_nelements(source) ||
            flattened->ne[1] != 1 || flattened->ne[2] != 1 || flattened->ne[3] != 1 ||
            ggml_nelements(flattened) != ggml_nelements(source)) return false;
    n_tokens = source->ne[1];
    return true;
}

inline bool slot_map_eligible(int64_t n_tokens, int64_t n_expert_used,
        int64_t slot_entries, int64_t n_slots, int64_t max_batch) {
    if (n_tokens <= 0 || n_tokens > max_batch || n_expert_used <= 0 ||
            slot_entries < 0 || n_slots < 0 || n_tokens > INT64_MAX/n_expert_used) return false;
    return std::min<int64_t>(n_tokens*n_expert_used, slot_entries) <= n_slots;
}

} // namespace llama_moe_cache_policy_detail
