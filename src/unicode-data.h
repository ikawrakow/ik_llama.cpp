#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_lowercase;
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> unicode_map_uppercase;
extern const std::vector<range_nfd> unicode_ranges_nfd;

// canonical normalization tables (used by the Raon-1 tokenizer)
struct unicode_cpt_value {
    uint32_t cpt;
    uint32_t value;
};

struct unicode_decomp_entry {
    uint32_t cpt;
    uint32_t first;
    uint32_t second;
};

struct unicode_comp_entry {
    uint64_t key;
    uint32_t value;
};

extern const unicode_cpt_value unicode_combining_class[];
extern const size_t unicode_combining_class_count;
extern const unicode_decomp_entry unicode_decomposition[];
extern const size_t unicode_decomposition_count;
extern const unicode_comp_entry unicode_composition[];
extern const size_t unicode_composition_count;
