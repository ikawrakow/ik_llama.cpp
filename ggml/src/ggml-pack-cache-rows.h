#pragma once

#include "ggml.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__cplusplus)
#define GGML_PACK_CACHE_ROWS_ALIGNOF(type) alignof(type)
#else
#define GGML_PACK_CACHE_ROWS_ALIGNOF(type) _Alignof(type)
#endif

static inline size_t ggml_pack_cache_rows_dst_alignment(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return GGML_PACK_CACHE_ROWS_ALIGNOF(float);
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
            // Q8_0 blocks begin with an fp16 scale and inherit its alignment.
            return GGML_PACK_CACHE_ROWS_ALIGNOF(ggml_fp16_t);
        default:
            return 0;
    }
}

static inline bool ggml_pack_cache_rows_source_layout_is_valid(const struct ggml_tensor * src) {
    if (src == NULL || src->type != GGML_TYPE_F32 ||
        src->ne[0] <= 0 || src->ne[1] <= 0 || src->ne[2] != 1 || src->ne[3] != 1 ||
        (uint64_t) src->ne[0] > (uint64_t) (SIZE_MAX/sizeof(float))) {
        return false;
    }

    const size_t alignment = GGML_PACK_CACHE_ROWS_ALIGNOF(float);
    const size_t row_bytes = (size_t) src->ne[0]*sizeof(float);
    const size_t max_span = (size_t) PTRDIFF_MAX;
    if (src->nb[0] != sizeof(float) || src->nb[1] < row_bytes ||
        src->nb[1] % alignment != 0 || src->view_offs % alignment != 0 ||
        (src->view_src == NULL && src->view_offs != 0) ||
        (src->data != NULL && (uintptr_t) src->data % alignment != 0) ||
        row_bytes > max_span) {
        return false;
    }

    const size_t row_extent = (size_t) (src->ne[1] - 1);
    if (row_extent > 0 && src->nb[1] > (max_span - row_bytes)/row_extent) {
        return false;
    }
    const size_t span_bytes = row_bytes + row_extent*src->nb[1];
    if (src->view_src != NULL) {
        const size_t root_bytes = ggml_nbytes(src->view_src);
        if (src->view_offs > max_span - span_bytes ||
            src->view_offs > root_bytes || span_bytes > root_bytes - src->view_offs) {
            return false;
        }
        if ((src->view_src->data == NULL) != (src->data == NULL) ||
            (src->data != NULL && src->data != (const char *) src->view_src->data + src->view_offs)) {
            return false;
        }
    }
    return true;
}

static inline bool ggml_pack_cache_rows_layout_is_valid(
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * cache,
        size_t                     dst_offset) {
    if (!ggml_pack_cache_rows_source_layout_is_valid(a) ||
        !ggml_pack_cache_rows_source_layout_is_valid(b) || cache == NULL ||
        (cache->type != GGML_TYPE_F32 &&
         cache->type != GGML_TYPE_F16 &&
         cache->type != GGML_TYPE_Q8_0) ||
        a->ne[1] != b->ne[1] || a->ne[0] > INT64_MAX - b->ne[0] ||
        cache->ne[0] != a->ne[0] + b->ne[0] ||
        cache->ne[1] <= 0 || cache->ne[2] != 1 || cache->ne[3] != 1 ||
        cache->view_src != NULL || cache->view_offs != 0 || a == cache || b == cache ||
        a->view_src == cache || b->view_src == cache) {
        return false;
    }

    const int64_t block_size = ggml_blck_size(cache->type);
    if (block_size <= 0 || cache->ne[0] % block_size != 0 ||
        (cache->type == GGML_TYPE_Q8_0 &&
         (a->ne[0] % block_size != 0 || b->ne[0] % block_size != 0))) {
        return false;
    }

    const size_t alignment = ggml_pack_cache_rows_dst_alignment(cache->type);
    const size_t row_bytes = ggml_row_size(cache->type, cache->ne[0]);
    const size_t max_span = (size_t) PTRDIFF_MAX;
    if (alignment == 0 || cache->nb[0] != ggml_type_size(cache->type) ||
        cache->nb[1] < row_bytes || cache->nb[1] % alignment != 0 ||
        cache->view_offs % alignment != 0 || dst_offset % alignment != 0 ||
        (cache->data != NULL && (uintptr_t) cache->data % alignment != 0) ||
        cache->nb[1] == 0 || dst_offset % cache->nb[1] != 0 || row_bytes > max_span) {
        return false;
    }

    const size_t dst_row = dst_offset/cache->nb[1];
    if (dst_row >= (size_t) cache->ne[1] ||
        (size_t) a->ne[1] > (size_t) cache->ne[1] - dst_row) {
        return false;
    }

    const size_t row_extent = (size_t) (a->ne[1] - 1);
    if (row_extent > 0 && cache->nb[1] > (max_span - row_bytes)/row_extent) {
        return false;
    }
    const size_t span_bytes = row_bytes + row_extent*cache->nb[1];
    const size_t root_extent = (size_t) (cache->ne[1] - 1);
    if (root_extent > 0 && cache->nb[1] > (max_span - row_bytes)/root_extent) {
        return false;
    }
    const size_t root_bytes = row_bytes + root_extent*cache->nb[1];
    if (dst_offset > max_span - span_bytes ||
        dst_offset > root_bytes || span_bytes > root_bytes - dst_offset) {
        return false;
    }
    return true;
}

static inline bool ggml_pack_cache_rows_op_is_valid(const struct ggml_tensor * op) {
    if (op == NULL || op->op != GGML_OP_PACK_CACHE_ROWS ||
        op->src[0] == NULL || op->src[1] == NULL || op->src[2] == NULL) {
        return false;
    }
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];
    const struct ggml_tensor * cache = op->src[2];
    if (!ggml_pack_cache_rows_layout_is_valid(a, b, cache, op->view_offs) ||
        op->type != cache->type || op->view_src != cache ||
        op->ne[0] != cache->ne[0] || op->ne[1] != a->ne[1] ||
        op->ne[2] != 1 || op->ne[3] != 1 ||
        op->nb[0] != cache->nb[0] || op->nb[1] != cache->nb[1]) {
        return false;
    }

    uint64_t encoded_offset = 0;
    memcpy(&encoded_offset, op->op_params, sizeof(encoded_offset));
    const void * expected_data = cache->data == NULL ? NULL :
        (const char *) cache->data + op->view_offs;
    return encoded_offset == op->view_offs && op->data == expected_data;
}

#undef GGML_PACK_CACHE_ROWS_ALIGNOF
