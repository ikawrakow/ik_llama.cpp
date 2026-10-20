#include "llama-moe-cache-residency.h"

#include <cstdio>
#include <vector>

using slots = llama_moe_cache_slots;
using key = slots::key;

static bool ready(slots & cache, key owner, int32_t slot) {
    uint64_t generation = 0;
    return cache.reserve(slot, owner, generation) && cache.complete(slot, owner, generation);
}

int main() {
    std::vector<slots::assignment> fills, evicted;
    size_t hits = 0;
    slots blocking(3, 2, 4);
    const int32_t first[] = { 0, 1 };
    if (!blocking.plan_blocking(0, first, 2, fills, evicted, hits) ||
            fills.size() != 2 || fills[0].slot != 0 || fills[1].slot != 1 || !evicted.empty()) return 1;
    const int32_t third[] = { 2 };
    if (!blocking.plan_blocking(1, third, 1, fills, evicted, hits) ||
            fills.size() != 1 || fills[0].slot != 2 || !evicted.empty()) return 2;

    slots ordered_blocking(3, 2, 4), ordered_async(3, 2, 4);
    const int32_t all[] = { 0, 1, 2 };
    if (!ordered_blocking.plan_blocking(0, all, 3, fills, evicted, hits)) return 3;
    if (!ready(ordered_async, {0, 0}, 0) || !ready(ordered_async, {0, 1}, 1) ||
            !ready(ordered_async, {0, 2}, 2)) return 4;
    const int32_t touch[] = { 0 };
    const key hit_owner[] = { { 0, 0 } };
    if (!ordered_blocking.plan_blocking(0, touch, 1, fills, evicted, hits) || hits != 1) return 5;
    int32_t row_slot = -1;
    slots::pin pin{};
    size_t n_pins = 0;
    if (!ordered_async.pin_rows(hit_owner, 1, &row_slot, &pin, 1, n_pins) || n_pins != 1 ||
            !ordered_async.release_pins(&pin, n_pins)) return 6;
    uint64_t generation = 0;
    slots::assignment victim;
    if (!ordered_blocking.plan_blocking(1, third, 1, fills, evicted, hits) || evicted.size() != 1 ||
            !ordered_async.replace_oldest_ready({1, 3}, nullptr, 0, victim, generation) ||
            evicted[0].owner.expert != 1 || victim.owner.expert != 1 || evicted[0].slot != victim.slot) return 7;

    slots protected_slots(3, 1, 4);
    if (!ready(protected_slots, {0, 0}, 0) || !ready(protected_slots, {0, 1}, 1)) return 8;
    uint64_t copying_generation = 0;
    if (!protected_slots.reserve(2, {0, 2}, copying_generation) ||
            !protected_slots.pin_rows(hit_owner, 1, &row_slot, &pin, 1, n_pins)) return 9;
    if (!protected_slots.replace_oldest_ready({0, 3}, nullptr, 0, victim, generation) || victim.slot != 1 ||
            !protected_slots.release_pins(&pin, n_pins)) return 10;

    slots free_order(3, 1, 4);
    uint64_t free_generation = 0;
    if (!free_order.reserve(0, {0, 0}, free_generation) || !free_order.cancel(0, {0, 0}, free_generation) ||
            !free_order.reserve(1, {0, 1}, free_generation) || !free_order.cancel(1, {0, 1}, free_generation)) return 14;
    int32_t free_slot = -1;
    if (!free_order.reserve_lowest_free({0, 2}, free_slot, free_generation) || free_slot != 0) return 15;

    slots stale(1, 1, 4);
    uint64_t old_generation = 0, new_generation = 0;
    if (!stale.reserve(0, {0, 0}, old_generation) || !stale.cancel(0, {0, 0}, old_generation) ||
            !stale.reserve(0, {0, 1}, new_generation) || stale.complete(0, {0, 0}, old_generation) ||
            !stale.complete(0, {0, 1}, new_generation) || stale.find({0, 0}).slot >= 0 ||
            stale.find({0, 1}).state != slots::slot_state::ready) return 11;

    slots wrapping(2, 1, 4);
    const int32_t seed[] = { 0, 1, 1 };
    if (!wrapping.plan_blocking(0, seed, 3, fills, evicted, hits) || fills.size() != 2) return 12;
    for (uint32_t i = 0; i < 300; ++i) {
        const int32_t duplicate[] = { 0, 2 + (int32_t) (i % 2), 2 + (int32_t) (i % 2) };
        if (!wrapping.plan_blocking(0, duplicate, 3, fills, evicted, hits) ||
                hits != 1 || fills.size() != 1 || evicted.size() != 1) return 13;
    }
    std::puts("MoE cache slot manager passed");
    return 0;
}
