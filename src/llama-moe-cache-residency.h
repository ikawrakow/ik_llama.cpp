#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

class llama_moe_cache_slots {
public:
    enum class slot_state : uint8_t { free, copying, ready, retired };

    struct key {
        int32_t layer = -1;
        int32_t expert = -1;
        bool operator==(const key & other) const {
            return layer == other.layer && expert == other.expert;
        }
    };

    struct pin {
        size_t slot = 0;
        key owner;
        uint64_t generation = 0;
    };

    struct snapshot {
        slot_state state = slot_state::free;
        key owner;
        uint64_t generation = 0;
        uint32_t pins = 0;
        int32_t slot = -1;
    };

    struct assignment {
        key owner;
        int32_t slot = -1;
    };

    struct miss_observation {
        slot_state state = slot_state::free;
        uint8_t demand = 0;
        bool has_free_slot = false;
        bool incremented = false;
    };

    llama_moe_cache_slots(size_t n_slots, size_t n_layers, size_t n_experts,
            bool blocking_scratch = true, bool async_admission = false)
            : slots_(n_slots), n_layers_(n_layers), n_experts_(n_experts),
              owner_slots_(n_layers*n_experts, -1),
              demand_(async_admission ? n_layers*n_experts : 0, 0),
              seen_(blocking_scratch ? n_experts : 0) {
        if (blocking_scratch) unique_.reserve(n_experts);
        for (size_t slot = 0; slot < n_slots; ++slot) {
            slots_[slot].prev = slot == 0 ? -1 : (int32_t) slot - 1;
            slots_[slot].next = slot + 1 == n_slots ? -1 : (int32_t) slot + 1;
            slots_[slot].in_lru = true;
        }
        lru_head_ = n_slots ? 0 : -1;
        lru_tail_ = n_slots ? (int32_t) n_slots - 1 : -1;
    }

    bool reserve(size_t slot, key owner, uint64_t & generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!can_reserve(slot, owner)) return false;
        reserve_locked(slot, owner, generation);
        return true;
    }

    bool reserve_lowest_free(key owner, int32_t & slot, uint64_t & generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(owner) || owner_slots_[index(owner)] >= 0 || owner_exists(owner, slots_.size())) return false;
        for (size_t candidate = 0; candidate < slots_.size(); ++candidate) {
            if (slots_[candidate].state != slot_state::free) continue;
            reserve_locked(candidate, owner, generation);
            slot = (int32_t) candidate;
            return true;
        }
        return false;
    }

    bool replace_oldest_ready(key new_owner, const int32_t * excluded, size_t n_excluded,
            assignment & victim, uint64_t & generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(new_owner) || owner_slots_[index(new_owner)] >= 0 || owner_exists(new_owner, slots_.size())) return false;
        const int32_t slot = oldest_eligible(false, excluded, n_excluded);
        if (slot < 0) return false;
        victim = { slots_[slot].owner, slot };
        replace_locked((size_t) slot, new_owner, generation);
        return true;
    }

    bool complete(size_t slot, key owner, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (matches(slot, owner, generation, slot_state::retired) && slots_[slot].copy_pending) {
            make_free(slot);
            return false;
        }
        if (!matches(slot, owner, generation, slot_state::copying) || owner_slots_[index(owner)] != (int32_t) slot) return false;
        slots_[slot].state = slot_state::ready;
        slots_[slot].copy_pending = false;
        if (!demand_.empty()) demand_[index(owner)] = 0;
        lru_insert_mru((int32_t) slot);
        return true;
    }

    bool fail(size_t slot, key owner, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!matches(slot, owner, generation, slot_state::copying) &&
                !(matches(slot, owner, generation, slot_state::retired) && slots_[slot].copy_pending)) return false;
        make_free(slot);
        return true;
    }

    bool cancel(size_t slot, key owner, uint64_t generation) {
        return fail(slot, owner, generation);
    }

    bool retire(size_t slot, key owner, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!matches(slot, owner, generation, slot_state::ready) &&
                !matches(slot, owner, generation, slot_state::copying)) return false;
        slot_record & record = slots_[slot];
        const bool copying = record.state == slot_state::copying;
        if (owner_slots_[index(owner)] == (int32_t) slot) owner_slots_[index(owner)] = -1;
        lru_remove((int32_t) slot);
        record.state = slot_state::retired;
        record.copy_pending = copying;
        if (record.pins == 0 && !copying) make_free(slot);
        return true;
    }

    miss_observation observe_miss(key owner) {
        std::lock_guard<std::mutex> lock(mutex_);
        miss_observation result;
        if (!valid(owner) || demand_.empty()) return result;
        const int32_t slot = owner_slots_[index(owner)];
        if (slot >= 0) {
            result.state = slots_[slot].state;
            if (result.state == slot_state::ready) {
                demand_[index(owner)] = 0;
                result.demand = 0;
                return result;
            }
            if (result.state == slot_state::copying) {
                result.demand = demand_[index(owner)];
                return result;
            }
        }
        uint8_t & demand = demand_[index(owner)];
        if (demand < 8) {
            ++demand;
            result.incremented = true;
        }
        result.demand = demand;
        for (const slot_record & record : slots_) result.has_free_slot |= record.state == slot_state::free;
        return result;
    }

    void rollback_miss_demand(key owner, bool incremented) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(owner) || demand_.empty()) return;
        const int32_t slot = owner_slots_[index(owner)];
        if (slot >= 0 && slots_[slot].state == slot_state::ready) {
            demand_[index(owner)] = 0;
        } else if (incremented && demand_[index(owner)] != 0) {
            --demand_[index(owner)];
        }
    }

    bool plan_blocking(int32_t layer, const int32_t * ids, size_t n_ids,
            std::vector<assignment> & fills, std::vector<assignment> & evicted,
            size_t & n_hit) {
        std::lock_guard<std::mutex> lock(mutex_);
        fills.clear();
        evicted.clear();
        n_hit = 0;
        if (layer < 0 || (size_t) layer >= n_layers_ || (n_ids && !ids) || seen_.size() != n_experts_) return false;
        if (++seen_generation_ == 0) {
            std::fill(seen_.begin(), seen_.end(), 0);
            seen_generation_ = 1;
        }
        unique_.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && (size_t) ids[i] < n_experts_);
            if (seen_[ids[i]] != seen_generation_) {
                seen_[ids[i]] = seen_generation_;
                unique_.push_back(ids[i]);
            }
        }
        if (unique_.size() > slots_.size()) return false;
        for (int32_t expert : unique_) {
            const int32_t slot = owner_slots_[index({ layer, expert })];
            if (slot >= 0 && slots_[slot].state == slot_state::ready) {
                lru_touch(slot);
                ++n_hit;
            }
        }
        std::sort(unique_.begin(), unique_.end());
        for (int32_t expert : unique_) {
            const key owner{ layer, expert };
            if (owner_slots_[index(owner)] >= 0) continue;
            const int32_t slot = oldest_eligible(true, nullptr, 0);
            if (slot < 0) return false;
            slot_record & record = slots_[slot];
            if (record.state == slot_state::ready) {
                const key old = record.owner;
                owner_slots_[index(old)] = -1;
                evicted.push_back({ old, slot });
            } else if (record.state != slot_state::free) {
                return false;
            }
            record.generation = next_generation(record.generation);
            record.owner = owner;
            record.state = slot_state::ready;
            record.copy_pending = false;
            record.pins = 0;
            owner_slots_[index(owner)] = slot;
            lru_touch(slot);
            fills.push_back({ owner, slot });
        }
        return true;
    }

    bool pin_rows(const key * owners, size_t n_rows, int32_t * row_slots,
            pin * pins, size_t pin_capacity, size_t & n_pins) {
        std::lock_guard<std::mutex> lock(mutex_);
        n_pins = 0;
        if ((n_rows && (!owners || !row_slots)) || pin_capacity < n_rows || (n_rows && !pins)) return false;
        for (size_t row = 0; row < n_rows; ++row) {
            row_slots[row] = -1;
            if (!valid(owners[row])) continue;
            const int32_t slot = owner_slots_[index(owners[row])];
            if (slot < 0 || (size_t) slot >= slots_.size()) continue;
            slot_record & record = slots_[slot];
            if (record.state != slot_state::ready || !(record.owner == owners[row])) continue;
            row_slots[row] = slot;
            bool already_pinned = false;
            for (size_t i = 0; i < n_pins; ++i) {
                if (pins[i].slot == (size_t) slot && pins[i].generation == record.generation && pins[i].owner == owners[row]) {
                    already_pinned = true;
                    break;
                }
            }
            if (!already_pinned) {
                if (record.pins == UINT32_MAX) {
                    release_prefix(pins, n_pins);
                    n_pins = 0;
                    return false;
                }
                ++record.pins;
                pins[n_pins++] = { (size_t) slot, record.owner, record.generation };
                lru_touch(slot);
            }
        }
        return true;
    }

    bool release_pins(const pin * pins, size_t n_pins) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (n_pins && !pins) return false;
        for (size_t i = 0; i < n_pins; ++i) {
            if (pins[i].slot >= slots_.size()) return false;
            const slot_record & record = slots_[pins[i].slot];
            if (!(record.owner == pins[i].owner) || record.generation != pins[i].generation ||
                    record.pins == 0 || (record.state != slot_state::ready && record.state != slot_state::retired)) return false;
            for (size_t j = 0; j < i; ++j) if (pins[j].slot == pins[i].slot) return false;
        }
        for (size_t i = 0; i < n_pins; ++i) {
            slot_record & record = slots_[pins[i].slot];
            --record.pins;
            if (record.state == slot_state::retired && record.pins == 0) make_free(pins[i].slot);
        }
        return true;
    }

    snapshot find(key owner) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(owner)) return {};
        const int32_t slot = owner_slots_[index(owner)];
        if (slot < 0 || (size_t) slot >= slots_.size()) return {};
        const slot_record & record = slots_[slot];
        return record.owner == owner ? snapshot{ record.state, record.owner, record.generation, record.pins, slot } : snapshot{};
    }

    size_t live_pins() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t result = 0;
        for (const slot_record & record : slots_) result += record.pins;
        return result;
    }

private:
    struct slot_record {
        slot_state state = slot_state::free;
        key owner;
        uint64_t generation = 0;
        uint32_t pins = 0;
        int32_t prev = -1, next = -1;
        bool copy_pending = false, in_lru = false;
    };

    // This lock owns slot identity, READY lookup, demand and replacement order.
    // Callers release it before backend copies, waits, CPU execution or collection.
    mutable std::mutex mutex_;
    std::vector<slot_record> slots_;
    size_t n_layers_, n_experts_;
    std::vector<int32_t> owner_slots_;
    std::vector<uint8_t> demand_;
    std::vector<uint32_t> seen_;
    std::vector<int32_t> unique_;
    uint32_t seen_generation_ = 0;
    int32_t lru_head_ = -1, lru_tail_ = -1;

    bool valid(key owner) const {
        return owner.layer >= 0 && (size_t) owner.layer < n_layers_ && owner.expert >= 0 &&
                (size_t) owner.expert < n_experts_;
    }
    size_t index(key owner) const { return (size_t) owner.layer*n_experts_ + (size_t) owner.expert; }
    bool owner_exists(key owner, size_t except) const {
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i != except && slots_[i].state != slot_state::free && slots_[i].owner == owner) return true;
        }
        return false;
    }
    int32_t oldest_eligible(bool allow_free, const int32_t * excluded, size_t n_excluded) const {
        if (allow_free) {
            for (int32_t slot = lru_head_; slot >= 0; slot = slots_[slot].next) {
                bool skip = false;
                for (size_t i = 0; excluded && i < n_excluded; ++i) skip |= excluded[i] == slot;
                if (!skip && slots_[slot].state == slot_state::free) return slot;
            }
        }
        for (int32_t slot = lru_head_; slot >= 0; slot = slots_[slot].next) {
            bool skip = false;
            for (size_t i = 0; excluded && i < n_excluded; ++i) skip |= excluded[i] == slot;
            const slot_record & record = slots_[slot];
            if (skip) continue;
            if (record.state == slot_state::ready && record.pins == 0 &&
                    owner_slots_[index(record.owner)] == slot) return slot;
        }
        return -1;
    }
    bool can_reserve(size_t slot, key owner) const {
        return slot < slots_.size() && valid(owner) && owner_slots_[index(owner)] < 0 &&
                slots_[slot].state == slot_state::free && !owner_exists(owner, slot);
    }
    void reserve_locked(size_t slot, key owner, uint64_t & generation) {
        slot_record & record = slots_[slot];
        lru_remove((int32_t) slot);
        record.generation = next_generation(record.generation);
        record.owner = owner;
        record.pins = 0;
        record.copy_pending = true;
        record.state = slot_state::copying;
        owner_slots_[index(owner)] = (int32_t) slot;
        generation = record.generation;
    }
    bool matches(size_t slot, key owner, uint64_t generation, slot_state state) const {
        return slot < slots_.size() && slots_[slot].state == state && slots_[slot].owner == owner &&
                slots_[slot].generation == generation;
    }
    static uint64_t next_generation(uint64_t generation) {
        ++generation;
        return generation == 0 ? 1 : generation;
    }
    void lru_remove(int32_t slot) {
        if (slot < 0 || (size_t) slot >= slots_.size() || !slots_[slot].in_lru) return;
        slot_record & record = slots_[slot];
        if (record.prev >= 0) slots_[record.prev].next = record.next; else lru_head_ = record.next;
        if (record.next >= 0) slots_[record.next].prev = record.prev; else lru_tail_ = record.prev;
        record.prev = record.next = -1;
        record.in_lru = false;
    }
    void lru_insert_mru(int32_t slot) {
        lru_remove(slot);
        slot_record & record = slots_[slot];
        record.prev = lru_tail_;
        record.next = -1;
        if (lru_tail_ >= 0) slots_[lru_tail_].next = slot; else lru_head_ = slot;
        lru_tail_ = slot;
        record.in_lru = true;
    }
    void lru_touch(int32_t slot) {
        if (slot >= 0 && (size_t) slot < slots_.size() && slots_[slot].in_lru && lru_tail_ != slot) lru_insert_mru(slot);
    }
    void replace_locked(size_t slot, key owner, uint64_t & generation) {
        slot_record & record = slots_[slot];
        owner_slots_[index(record.owner)] = -1;
        lru_remove((int32_t) slot);
        record.generation = next_generation(record.generation);
        record.owner = owner;
        record.pins = 0;
        record.copy_pending = true;
        record.state = slot_state::copying;
        owner_slots_[index(owner)] = (int32_t) slot;
        generation = record.generation;
    }
    void make_free(size_t slot) {
        slot_record & record = slots_[slot];
        if (valid(record.owner) && owner_slots_[index(record.owner)] == (int32_t) slot) owner_slots_[index(record.owner)] = -1;
        record.state = slot_state::free;
        record.owner = {};
        record.pins = 0;
        record.copy_pending = false;
        lru_insert_mru((int32_t) slot);
    }
    void release_prefix(const pin * pins, size_t count) {
        for (size_t i = 0; i < count; ++i) --slots_[pins[i].slot].pins;
    }
};
