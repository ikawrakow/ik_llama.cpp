#pragma once

#include "ggml.h"

#include <algorithm>
#include <vector>

struct dsv4_mask_view {
    ggml_tensor * mask;
    int64_t n_kv, n_tokens, n_stream;
    ggml_tensor * view;
};

using dsv4_mask_view_cache = std::vector<dsv4_mask_view>;

// Cache graph tensors, not their contents. The scheduler still copies updated inputs on every execution.
inline ggml_tensor * dsv4_get_mask_view(
        ggml_context * ctx,
        dsv4_mask_view_cache & cache,
        ggml_tensor * mask,
        int64_t n_kv,
        int64_t n_tokens,
        int64_t n_stream) {
    GGML_ASSERT(mask != nullptr && (mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32));
    GGML_ASSERT(n_kv > 0 && n_tokens > 0);
    n_stream = std::max<int64_t>(1, n_stream);
    GGML_ASSERT(n_tokens % n_stream == 0);

    if (n_stream == 1 && ggml_is_matrix(mask) && ggml_is_contiguous(mask) &&
            mask->ne[0] == n_kv && mask->ne[1] == n_tokens) {
        return mask;
    }

    // Only a handful of layouts per graph.
    for (const auto & entry : cache) {
        if (entry.mask == mask && entry.n_kv == n_kv && entry.n_tokens == n_tokens && entry.n_stream == n_stream) {
            return entry.view;
        }
    }

    auto base = ggml_view_2d(ctx, mask, n_kv, n_tokens, mask->nb[1], 0);
    if (!ggml_is_contiguous(base)) {
        base = ggml_cont(ctx, base);
        ggml_format_name(base, "%s_cont", mask->name);
    }
    auto view = base;
    if (n_stream > 1) {
        const int64_t n_tokens_stream = n_tokens/n_stream;
        view = ggml_view_4d(ctx, base, n_kv, n_tokens_stream, 1, n_stream,
                base->nb[1], base->nb[1]*n_tokens_stream, base->nb[1]*n_tokens_stream, 0);
    }
    ggml_format_name(view, "%s_view", mask->name);
    cache.push_back({mask, n_kv, n_tokens, n_stream, view});
    return view;
}
