#include "llama-engram.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstring>

void llama_engram::push_tokens(const llama_hparams & hparams, const int32_t * tokens, uint32_t n, uint32_t start_pos) {
    if (hparams.engram_layer_count == 0 || n == 0) {
        return;
    }
    if (start_pos + n > compressed.size()) {
        compressed.resize(start_pos + n);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t tok = tokens[i];
        GGML_ASSERT(tok >= 0 && (uint32_t) tok < hparams.engram_token_map_size);
        compressed[start_pos + i] = hparams.engram_token_map[tok];
    }
    size = std::max(size, start_pos + n);
}

void llama_engram::hash_positions(const llama_hparams & hparams, uint32_t slot, uint32_t pos, uint32_t n, uint64_t * out) const {
    GGML_ASSERT(slot < hparams.engram_layer_count);
    const uint32_t max_ng  = hparams.engram_max_ngram_size;
    const uint32_t n_heads = hparams.engram_n_heads;
    const uint32_t n_cols  = (max_ng - 1) * n_heads;

    const uint64_t   pad    = hparams.engram_pad_id;
    const uint64_t * mults  = hparams.engram_multipliers.data() + slot * max_ng;
    const uint64_t * primes = hparams.engram_primes.data()      + slot * n_cols;
    const uint64_t * offs   = hparams.engram_offsets.data()     + slot * n_cols;

    for (uint32_t t = 0; t < n; ++t) {
        const uint32_t p = pos + t;
        GGML_ASSERT(p < size);
        // one lookback at a time, exactly like the reference: the running XOR after step i
        // is the hash of the (i+1)-gram; lookbacks stop at the sequence start and at dead
        // tokens, and once blocked every longer lookback is padded too
        uint64_t rolling = 0;
        bool     blocked = false;
        for (uint32_t shift = 0; shift < max_ng; ++shift) {
            const int32_t src  = p >= shift ? compressed[p - shift] : 0;
            blocked = blocked || p < shift || src == LLAMA_ENGRAM_DEAD;
            const uint64_t tok  = blocked ? pad : (uint64_t) (uint32_t) src;
            // compressed ids < 2^17 and multipliers < 2^63/vocab/2 by construction, so the
            // product fits int64 and the rolling XOR stays non-negative
            const uint64_t prod = tok * mults[shift];
            if (shift == 0) {
                rolling = prod;
                continue;
            }
            rolling ^= prod;
            const uint32_t col0 = (shift - 1) * n_heads;
            for (uint32_t h = 0; h < n_heads; ++h) {
                out[(size_t) t * n_cols + col0 + h] = rolling % primes[col0 + h] + offs[col0 + h];
            }
        }
    }
}

void llama_engram::gather_rows(const ggml_tensor * table, const uint64_t * row_ids, int64_t n_rows, float * out) {
    const ggml_type_traits_t traits = ggml_internal_get_type_traits(table->type);
    GGML_ASSERT(traits.to_float != nullptr);
    GGML_ASSERT(table->data != nullptr);
    const int64_t key_len  = table->ne[0];
    const size_t  row_size = ggml_row_size(table->type, key_len);
    const char  * base     = (const char *) table->data;
    for (int64_t r = 0; r < n_rows; ++r) {
        // the reference (ParallelEngramEmbedding, model.py:312-321) zero-masks lookups
        // outside the table instead of failing; on the real model hash ids are always in
        // range (offsets are cumulative), but a small-table fixture exercises this path
        if (row_ids[r] >= (uint64_t) table->ne[1]) {
            memset(out + r * key_len, 0, key_len * sizeof(float));
            continue;
        }
        traits.to_float(base + row_ids[r] * row_size, out + r * key_len, key_len);
    }
}
