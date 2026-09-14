#pragma once

// DeepSeek-V4.1 engram: host-side n-gram hashing + embedding-table row gather.
//
// All hash constants (token map, multipliers, primes, offsets, pad id) come from the
// GGUF via llama_hparams — they are generated at conversion time and must never be
// regenerated at runtime (the hashing has to be bit-exact with training).
//
// Reference: v41_ref/inference/engram.py (NgramHashState). Text-only for now; the
// DEAD sentinel is reserved for image spans once vision lands.

#include "llama-hparams.h"

#include <cstdint>
#include <vector>

struct ggml_tensor;

#define LLAMA_ENGRAM_DEAD (-1)

struct llama_engram {
    // compressed token id per absolute sequence position (LLAMA_ENGRAM_DEAD for dead spans)
    std::vector<int32_t> compressed;
    uint32_t           size = 0; // highest position pushed so far

    // map and store tokens at absolute positions [start_pos, start_pos + n)
    void push_tokens(const llama_hparams & hparams, const int32_t * tokens, uint32_t n, uint32_t start_pos);

    // row ids of the n-grams (sizes 2..max_ngram_size) ending at positions [pos, pos + n),
    // for engram layer slot `slot` (index into hparams.engram_layer_ids, NOT the layer id).
    // out receives n * ((max_ngram_size - 1) * engram_n_heads) row ids, column-major per
    // position as (ngram_size - 2) * n_heads + head — the layout the engram wkv projection
    // expects after flattening.
    void hash_positions(const llama_hparams & hparams, uint32_t slot, uint32_t pos, uint32_t n, uint64_t * out) const;

    // dequantize rows of a host-resident quantized table (the engram_embd tensors) into
    // out [n_rows * table->ne[0]] f32. The table must live in a host buffer (mmap is fine);
    // row ids are bounds-checked against table->ne[1].
    static void gather_rows(const ggml_tensor * table, const uint64_t * row_ids, int64_t n_rows, float * out);
};
