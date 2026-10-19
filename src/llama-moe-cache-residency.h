#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

class llama_moe_cache_residency {
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

    llama_moe_cache_residency(size_t n_slots, size_t n_layers, size_t n_experts)
            : slots_(n_slots), n_layers_(n_layers), n_experts_(n_experts),
              ready_(n_layers * n_experts, -1) {}

    bool reserve(size_t slot, key owner, uint64_t & generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= slots_.size() || !valid(owner) || slots_[slot].state != slot_state::free ||
                ready_[index(owner)] >= 0) return false;
        for (const slot_record & existing : slots_) {
            if (existing.state != slot_state::free && existing.owner == owner) return false;
        }
        slot_record & record = slots_[slot];
        record.generation = next_generation(record.generation);
        record.owner = owner;
        record.pins = 0;
        record.copy_pending = true;
        record.state = slot_state::copying;
        generation = record.generation;
        return true;
    }

    // Atomically remove a zero-pin READY owner and reserve the same slot for
    // the next generation. There is no observable FREE interval.
    bool replace_ready(size_t slot, key expected_owner, uint64_t expected_generation,
            key new_owner, uint64_t & generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(expected_owner) || !valid(new_owner) || expected_owner == new_owner ||
                !matches(slot, expected_owner, expected_generation, slot_state::ready) ||
                slots_[slot].pins != 0 || ready_[index(expected_owner)] != (int32_t) slot ||
                ready_[index(new_owner)] >= 0) return false;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (i != slot && slots_[i].state != slot_state::free && slots_[i].owner == new_owner) return false;
        }
        ready_[index(expected_owner)] = -1;
        slot_record & record = slots_[slot];
        record.generation = next_generation(record.generation);
        record.owner = new_owner;
        record.pins = 0;
        record.copy_pending = true;
        record.state = slot_state::copying;
        generation = record.generation;
        return true;
    }

    bool complete(size_t slot, key owner, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (matches(slot, owner, generation, slot_state::retired) && slots_[slot].copy_pending) {
            make_free(slots_[slot]);
            return false;
        }
        if (!matches(slot, owner, generation, slot_state::copying) || ready_[index(owner)] >= 0) return false;
        slots_[slot].state = slot_state::ready;
        slots_[slot].copy_pending = false;
        ready_[index(owner)] = (int32_t) slot;
        return true;
    }

    bool fail(size_t slot, key owner, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!matches(slot, owner, generation, slot_state::copying) &&
                !(matches(slot, owner, generation, slot_state::retired) && slots_[slot].copy_pending)) return false;
        make_free(slots_[slot]);
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
        if (record.state == slot_state::ready && valid(record.owner) && ready_[index(record.owner)] == (int32_t) slot) {
            ready_[index(record.owner)] = -1;
        }
        record.state = slot_state::retired;
        record.copy_pending = copying;
        if (record.pins == 0 && !copying) make_free(record);
        return true;
    }

    bool reusable(size_t slot) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slot < slots_.size() && slots_[slot].state == slot_state::free;
    }

    // The whole request is pinned under one lock, then execution proceeds without holding it.
    bool pin_rows(const key * owners, size_t n_rows, int32_t * row_slots, uint64_t * row_generations,
            pin * pins, size_t pin_capacity, size_t & n_pins) {
        std::lock_guard<std::mutex> lock(mutex_);
        n_pins = 0;
        if ((n_rows && (!owners || !row_slots || !row_generations)) || pin_capacity < n_rows || (n_rows && !pins)) return false;
        for (size_t row = 0; row < n_rows; ++row) {
            row_slots[row] = -1;
            row_generations[row] = 0;
            if (!valid(owners[row])) continue;
            const int32_t slot = ready_[index(owners[row])];
            if (slot < 0 || (size_t) slot >= slots_.size()) continue;
            slot_record & record = slots_[slot];
            if (record.state != slot_state::ready || !(record.owner == owners[row])) continue;
            row_slots[row] = slot;
            row_generations[row] = record.generation;
            bool already_pinned = false;
            for (size_t i = 0; i < n_pins; ++i) {
                if (pins[i].slot == (size_t) slot && pins[i].generation == record.generation &&
                        pins[i].owner == owners[row]) {
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
            if (record.state == slot_state::retired && record.pins == 0) make_free(record);
        }
        return true;
    }

    snapshot inspect(size_t slot) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (slot >= slots_.size()) return {};
        const slot_record & record = slots_[slot];
        return { record.state, record.owner, record.generation, record.pins, (int32_t) slot };
    }

    snapshot find(key owner) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!valid(owner)) return {};
        for (size_t slot = 0; slot < slots_.size(); ++slot) {
            const slot_record & record = slots_[slot];
            if (record.state != slot_state::free && record.owner == owner) {
                return { record.state, record.owner, record.generation, record.pins, (int32_t) slot };
            }
        }
        return {};
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
        bool copy_pending = false;
    };

    // Acquire one group lock at a time; backend work and CPU kernels run unlocked.
    mutable std::mutex mutex_;
    std::vector<slot_record> slots_;
    size_t n_layers_;
    size_t n_experts_;
    std::vector<int32_t> ready_;

    bool valid(key owner) const {
        return owner.layer >= 0 && (size_t) owner.layer < n_layers_ && owner.expert >= 0 &&
                (size_t) owner.expert < n_experts_;
    }
    size_t index(key owner) const { return (size_t) owner.layer*n_experts_ + (size_t) owner.expert; }
    bool matches(size_t slot, key owner, uint64_t generation, slot_state state) const {
        return slot < slots_.size() && slots_[slot].state == state && slots_[slot].owner == owner &&
                slots_[slot].generation == generation;
    }
    static uint64_t next_generation(uint64_t generation) {
        ++generation;
        return generation == 0 ? 1 : generation;
    }
    static void make_free(slot_record & record) {
        record.state = slot_state::free;
        record.owner = {};
        record.pins = 0;
        record.copy_pending = false;
    }
    void release_prefix(const pin * pins, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            slot_record & record = slots_[pins[i].slot];
            --record.pins;
        }
    }
};
