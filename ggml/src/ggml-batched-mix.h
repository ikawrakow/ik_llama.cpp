#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Shared CPU/CUDA contract for tensors consumed or produced by
// GGML_OP_BATCHED_MIX. Higher strides are arbitrary, but every effective F32
// address must remain element-aligned and the full byte span must be both
// ptrdiff_t-representable and contained by the root view source.
static inline bool ggml_batched_mix_f32_span_bytes(
        const struct ggml_tensor * tensor, size_t * span_out) {
    if (tensor == NULL || tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) {
        return false;
    }

    const size_t max_span = (size_t) PTRDIFF_MAX;
    size_t span = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return false;
        }
        const size_t extent = (size_t) (tensor->ne[i] - 1);
        if (extent > 0 && tensor->nb[i] > (max_span - span)/extent) {
            return false;
        }
        span += extent*tensor->nb[i];
    }

    *span_out = span;
    return true;
}

static inline bool ggml_batched_mix_f32_layout_is_valid(const struct ggml_tensor * tensor) {
    size_t span = 0;
    if (!ggml_batched_mix_f32_span_bytes(tensor, &span)) {
        return false;
    }
    if (tensor->view_offs % sizeof(float) != 0 ||
            (tensor->data != NULL && (uintptr_t) tensor->data % sizeof(float) != 0)) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 1 && tensor->nb[i] % sizeof(float) != 0) {
            return false;
        }
    }

    if (tensor->view_src == NULL) {
        return tensor->view_offs == 0;
    }

    size_t source_span = 0;
    if (!ggml_batched_mix_f32_span_bytes(tensor->view_src, &source_span) ||
            tensor->view_src->view_src != NULL || tensor->view_offs > source_span ||
            span > source_span - tensor->view_offs) {
        return false;
    }
    return tensor->view_offs <= (size_t) PTRDIFF_MAX - span;
}
