#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-build-context.h"
#include "../llama-dsv4.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>

static ggml_tensor * dsv4_hc_mean_for_capture(ggml_context * ctx, ggml_tensor * x) {
    GGML_ASSERT(x != nullptr && x->ne[1] > 0);

    ggml_tensor * mean = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t hc_row = 1; hc_row < x->ne[1]; ++hc_row) {
        mean = ggml_add(ctx, mean,
                ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], hc_row * x->nb[1]));
    }
    return ggml_scale(ctx, mean, 1.0f / (float) x->ne[1]);
}

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_concat_named(
        ggml_context * ctx,
        ggml_tensor  * a,
        ggml_tensor  * b,
        int            dim,
        const char   * name) {
    ggml_tensor * r = ggml_concat(ctx, a, b, dim);
    ggml_set_name(r, name);
    return r;
}

// Concat along dim 2 (the KV dimension), multi-stream aware: with ne[3] > 1
// the CUDA backend's type-generic concat fast paths require ne[3] == 1 and the
// general path is F32-only, so concat per stream (dim 2) and re-assemble (dim 3).
static ggml_tensor * dsv4_concat_kv(
        ggml_context * ctx,
        ggml_tensor  * a,
        ggml_tensor  * b) {
    if (a->ne[3] <= 1) {
        return ggml_concat(ctx, a, b, 2);
    }
    GGML_ASSERT(a->ne[3] == b->ne[3]);
    ggml_tensor * result = nullptr;
    for (int64_t s = 0; s < a->ne[3]; ++s) {
        ggml_tensor * a_s = ggml_view_4d(ctx, a, a->ne[0], a->ne[1], a->ne[2], 1,
                a->nb[1], a->nb[2], a->nb[3], s*a->nb[3]);
        ggml_tensor * b_s = ggml_view_4d(ctx, b, b->ne[0], b->ne[1], b->ne[2], 1,
                b->nb[1], b->nb[2], b->nb[3], s*b->nb[3]);
        ggml_tensor * c_s = dsv4_concat_named(ctx, a_s, b_s, 2, "dsv4_concat_kv_s");
        result = result == nullptr ? c_s : dsv4_concat_named(ctx, result, c_s, 3, "dsv4_concat_kv");
    }
    return result;
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

static ggml_tensor * dsv4_new_i32_input(ggml_context * ctx, ggml_tensor ** dst, int64_t n, const char * name) {
    *dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, std::max<int64_t>(1, n));
    ggml_set_input(*dst);
    ggml_set_name(*dst, name);
    return *dst;
}

static ggml_tensor * dsv4_new_i64_input(ggml_context * ctx, ggml_tensor ** dst, int64_t n, const char * name) {
    *dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, std::max<int64_t>(1, n));
    ggml_set_input(*dst);
    ggml_set_name(*dst, name);
    return *dst;
}

static ggml_tensor * dsv4_new_mask_input(ggml_context * ctx, ggml_tensor ** dst, int64_t n_kv, int64_t n_tokens, const char * name,
        ggml_type mask_type) {
    //*dst = ggml_new_tensor_2d(ctx, mask_type, std::max<int64_t>(1, n_kv), GGML_PAD(std::max<int64_t>(1, n_tokens), GGML_KQ_MASK_PAD));
    *dst = ggml_new_tensor_2d(ctx, mask_type, std::max<int64_t>(1, n_kv), std::max<int64_t>(1, n_tokens));
    ggml_set_input(*dst);
    ggml_set_name(*dst, name);
    return *dst;
}

static void dsv4_build_plan_inputs(
        ggml_context * ctx,
        llama_context::dsv4_runtime::comp_inputs & inputs,
        const llama_context::dsv4_runtime::comp_plan & plan,
        const char * tag,
        int64_t n_tokens,
        bool create_mask = true, bool flash_attn = true,
        uint32_t cand_block = 0, uint32_t cand_topk = 0) {
    dsv4_new_i32_input(ctx, &inputs.state_pos, (int64_t) plan.state_pos.size(), (std::string(tag) + "_state_pos").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_persist_src_idxs, (int64_t) plan.state_persist_src_idxs.size(), (std::string(tag) + "_persist_src").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_persist_dst_idxs, (int64_t) plan.state_persist_dst_idxs.size(), (std::string(tag) + "_persist_dst").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_read_idxs, (int64_t) plan.state_read_idxs.size(), (std::string(tag) + "_state_read").c_str());
    dsv4_new_i64_input(ctx, &inputs.state_write_idxs, (int64_t) plan.state_write_idxs.size(), (std::string(tag) + "_state_write").c_str());
    inputs.state_write_idxs_lid = nullptr;
    if (!plan.state_write_idxs_lid.empty()) {
        dsv4_new_i64_input(ctx, &inputs.state_write_idxs_lid, (int64_t) plan.state_write_idxs_lid.size(), (std::string(tag) + "_state_write_lid").c_str());
    }
    dsv4_new_i32_input(ctx, &inputs.state_write_pos, (int64_t) plan.state_write_pos.size(), (std::string(tag) + "_write_pos").c_str());
    if (create_mask) {
        auto type = flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        dsv4_new_mask_input(ctx, &inputs.kq_mask, std::max<int64_t>(1, plan.n_kv), n_tokens, (std::string(tag) + "_kq_mask").c_str(), type);
    } else {
        inputs.kq_mask = nullptr;
    }
    inputs.cand_pin = nullptr;
    if (create_mask && cand_block > 0 && plan.n_kv > 0 &&
            plan.n_kv % cand_block == 0 && plan.n_kv/cand_block > cand_topk) {
        const int64_t n_stream = std::max<int64_t>(1, plan.n_stream);
        inputs.cand_pin = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                plan.n_kv/cand_block, std::max<int64_t>(1, n_tokens/n_stream), 1, n_stream);
        ggml_set_input(inputs.cand_pin);
        ggml_set_name(inputs.cand_pin, (std::string(tag) + "_cand_pin").c_str());
    }
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, ggml_tensor **append_row, bool neg_inf) {
    if (*append_row == nullptr) {
        ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
        row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
        row = ggml_reshape_2d(ctx, row, t->ne[0], 1);
        *append_row = row;
    }
    return dsv4_concat_named(ctx, t, *append_row, 1, "dsv4_append_zero_row");
}

static ggml_tensor * dsv4_cache_view_2d(
        ggml_context * ctx,
        ggml_tensor  * cache,
        int64_t        dim0,
        int64_t        dim1) {
    return ggml_view_2d(ctx, cache, dim0, dim1, ggml_row_size(cache->type, dim0), 0);
}

static ggml_tensor * dsv4_build_mask_stream_view(
        ggml_context * ctx,
        ggml_tensor  * mask,
        int64_t        n_stream,
        int64_t        n_tokens) {
    if (n_stream <= 1) {
        return mask;
    }

    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tokens_stream = n_tokens/n_stream;
    return ggml_view_4d(ctx, mask, mask->ne[0], n_tokens_stream, 1, n_stream,
            mask->nb[1], mask->nb[1]*n_tokens_stream, mask->nb[1]*n_tokens_stream, 0);
}

static ggml_tensor * dsv4_build_raw_mask_view(
        ggml_context * ctx,
        ggml_tensor  * mask,
        ggml_tensor  * raw_k_read_idxs,
        int64_t        n_kv,
        int64_t        n_tokens,
        int64_t        n_stream,
        const llm_build_cb & cb, int il) {
    const int64_t n_tokens_stream = n_stream > 0 ? n_tokens/n_stream : n_tokens;
    const int64_t n_rows_stream = GGML_PAD(n_kv, 256);

    if (raw_k_read_idxs == nullptr) {
        auto base = ggml_view_2d(ctx, mask, n_kv, n_tokens, mask->nb[1], 0);
        if (!ggml_is_contiguous(base)) {
            base = ggml_cont(ctx, base);
            cb(base, "mask_base", il);
        }
        return n_stream == 1 ? base : dsv4_build_mask_stream_view(ctx, base, n_stream, n_tokens);
        //auto base = mask->ne[0] == n_kv && mask->ne[1] == n_tokens ? mask
        //          : ggml_cont(ctx, ggml_view_2d(ctx, mask, n_kv, n_tokens, mask->nb[1], 0));
        //return n_stream == 1 ? base : dsv4_build_mask_stream_view(ctx, base, n_stream, n_tokens);
        ////ggml_tensor * base = ggml_cont(ctx, ggml_view_2d(ctx, mask, n_kv, n_tokens, mask->nb[1], 0));
        ////return dsv4_build_mask_stream_view(ctx, base, n_stream, n_tokens);
    }

    if (n_stream <= 0 || n_tokens % n_stream != 0 || raw_k_read_idxs->ne[0] < n_rows_stream*n_stream) {
        ggml_tensor * base = ggml_cont(ctx, ggml_view_2d(ctx, mask, n_kv, n_tokens, mask->nb[1], 0));
        cb(base, "mask_base1", il);
        return dsv4_build_mask_stream_view(ctx, base, std::max<int64_t>(1, n_stream), n_tokens);
    }

    if (n_stream == 1 && mask->ne[0] == raw_k_read_idxs->ne[0]) {
        return mask;
    }

    printf("%s: Oops(%s). mask is %ld x %ld x %ld x %ld. n_stream = %ld, n_tokens = %ld, raw_k_read_idxs = %ld x %ld x %ld x %ld\n",
            __func__, mask->name, mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3], n_stream, n_tokens,
            raw_k_read_idxs->ne[0], raw_k_read_idxs->ne[1], raw_k_read_idxs->ne[2], raw_k_read_idxs->ne[3]);

    ggml_tensor * mask_t = ggml_cont(ctx, ggml_transpose(ctx, mask));
    ggml_tensor * result = nullptr;
    for (int64_t s = 0; s < n_stream; ++s) {
        ggml_tensor * idxs = ggml_view_1d(ctx, raw_k_read_idxs, n_kv,
                s*n_rows_stream*ggml_element_size(raw_k_read_idxs));
        ggml_tensor * mask_s = ggml_view_2d(ctx, mask_t, n_tokens_stream, mask->ne[0], mask_t->nb[1],
                s*n_tokens_stream*mask_t->nb[0]);
        ggml_tensor * rows = ggml_get_rows(ctx, mask_s, idxs);
        ggml_tensor * stream = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_transpose(ctx, rows)),
                n_kv, n_tokens_stream, 1, 1);
        result = result == nullptr ? stream : ggml_concat(ctx, result, stream, 3);
        cb(result, "raw_mask_view", s);
    }
    return result;
}

static ggml_tensor * dsv4_pad_raw_k_to(
        ggml_context * ctx,
        ggml_tensor  * raw_k,
        int64_t        n_kv_target) {
    const int64_t n_kv_cur = raw_k->ne[2];
    if (n_kv_target <= n_kv_cur) {
        return raw_k;
    }

    const int64_t n_pad = n_kv_target - n_kv_cur;
    ggml_tensor * row0 = ggml_view_4d(ctx, raw_k,
            raw_k->ne[0], raw_k->ne[1], 1, raw_k->ne[3],
            raw_k->nb[1], raw_k->nb[2], raw_k->nb[3], 0);
    ggml_tensor * zero_row = ggml_cont(ctx, row0);
    if (zero_row->type != GGML_TYPE_F32) {
        zero_row = ggml_cast(ctx, zero_row, GGML_TYPE_F32);
    }
    zero_row = ggml_scale(ctx, zero_row, 0.0f);
    if (raw_k->type != zero_row->type && !ggml_is_quantized(raw_k->type)) {
        zero_row = ggml_cast(ctx, zero_row, raw_k->type);
    }
    ggml_tensor * zeros = ggml_repeat_4d(ctx, zero_row, zero_row->ne[0], zero_row->ne[1], n_pad, zero_row->ne[3]);
    return dsv4_concat_named(ctx, raw_k, zeros, 2, "dsv4_raw_k_pad");
}

static ggml_tensor * dsv4_pad_raw_mask_to(
        ggml_context * ctx,
        ggml_tensor  * raw_mask,
        int64_t        n_kv_target,
        int64_t        n_tokens) {
    const int64_t n_kv_cur = raw_mask->ne[0];
    if (n_kv_target <= n_kv_cur) {
        return raw_mask;
    }

    printf("%s: Oops, padding mask\n", __func__);

    const int64_t n_pad = n_kv_target - n_kv_cur;
    GGML_UNUSED(n_tokens);
    ggml_tensor * pad = ggml_new_tensor_4d(ctx, raw_mask->type, n_pad, raw_mask->ne[1], raw_mask->ne[2], raw_mask->ne[3]);
    pad = ggml_fill(ctx, pad, -INFINITY);
    return dsv4_concat_named(ctx, raw_mask, pad, 0, "dsv4_raw_mask_pad");
}

static ggml_tensor * dsv4_pad_mask_tokens(
        ggml_context * ctx,
        ggml_tensor  * mask,
        int64_t        n_tokens) {
    const int64_t n_stream = std::max<int64_t>(1, mask->ne[3]);
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tokens_pad = GGML_PAD(n_tokens/n_stream, GGML_KQ_MASK_PAD);
    if (mask->ne[1] >= n_tokens_pad) {
        return mask;
    }

    if (n_stream > 1) {
        // Pad each stream separately and re-assemble along dim 3. This keeps
        // every concat on the contiguous fast paths supported by the CUDA
        // backend for any tensor type (dim 1 with ne[2]*ne[3] == 1, and dim 3),
        // whereas a single multi-stream concat along dim 1 falls back to the
        // F32-only general path and fails for f16 flash-attention masks.
        ggml_tensor * result = nullptr;
        for (int64_t s = 0; s < n_stream; ++s) {
            ggml_tensor * mask_s = ggml_view_4d(ctx, mask, mask->ne[0], mask->ne[1], 1, 1,
                    mask->nb[1], mask->nb[2], mask->nb[3], s*mask->nb[3]);
            ggml_tensor * pad_s = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], n_tokens_pad - mask->ne[1], 1, 1);
            pad_s = ggml_fill(ctx, pad_s, -INFINITY);
            ggml_tensor * padded_s = dsv4_concat_named(ctx, mask_s, pad_s, 1, "dsv4_mask_tokens_pad_s");
            result = result == nullptr ? padded_s : dsv4_concat_named(ctx, result, padded_s, 3, "dsv4_mask_tokens_pad");
        }
        return result;
    }

    ggml_tensor * pad = ggml_new_tensor_4d(ctx, mask->type, mask->ne[0], n_tokens_pad - mask->ne[1], mask->ne[2], mask->ne[3]);
    pad = ggml_fill(ctx, pad, -INFINITY);
    auto new_mask = dsv4_concat_named(ctx, mask, pad, 1, "dsv4_mask_tokens_pad");

    //printf("%s: Oops: padding mask %s from %ld x %ld x %ld x %ld to %ld x %ld x %ld x %ld\n", __func__, mask->name,
    //        mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3],
    //        new_mask->ne[0], new_mask->ne[1], new_mask->ne[2], new_mask->ne[3]);
    return new_mask;
}

static ggml_tensor * dsv4_cache_view_3d(
        ggml_context * ctx,
        ggml_tensor  * cache,
        int64_t        n_embd_head,
        int64_t        n_kv) {
    return ggml_view_3d(ctx, cache,
            n_embd_head, 1, n_kv,
            ggml_row_size(cache->type, n_embd_head),
            ggml_row_size(cache->type, n_embd_head),
            0);
}

static ggml_tensor * dsv4_slice_1d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        offset,
        int64_t        size) {
    return ggml_view_1d(ctx, t, size, offset*ggml_element_size(t));
}

static ggml_tensor * dsv4_require_f32_rows(
        ggml_context * ctx,
        ggml_tensor  * t) {
    if (t == nullptr || t->type == GGML_TYPE_F32) {
        return t;
    }

    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static ggml_tensor * dsv4_cache_stream_view_3d(
        ggml_context * ctx,
        ggml_tensor  * cache,
        int64_t        n_embd_head,
        int64_t        n_kv,
        int64_t        kv_size,
        int64_t        stream) {
    const size_t offset = (size_t) ggml_row_size(cache->type, n_embd_head) * (size_t) kv_size * (size_t) stream;
    return ggml_view_3d(ctx, cache,
            n_embd_head, 1, n_kv,
            ggml_row_size(cache->type, n_embd_head),
            ggml_row_size(cache->type, n_embd_head),
            offset);
}

static ggml_tensor * dsv4_cache_stream_view_4d(
        ggml_context * ctx,
        ggml_tensor  * cache,
        int64_t        n_embd_head,
        int64_t        n_kv,
        int64_t        kv_size,
        int64_t        s0,
        int64_t        n_stream) {
    const size_t row_size = ggml_row_size(cache->type, n_embd_head);
    const size_t offset = row_size*(size_t) kv_size*(size_t) s0;
    return ggml_view_4d(ctx, cache,
            n_embd_head, 1, n_kv, n_stream,
            row_size,
            row_size,
            row_size*(size_t) kv_size,
            offset);
}

static ggml_tensor * dsv4_raw_get_k(
        llama_context * lctx,
        ggml_context  * ctx,
        ggml_tensor   * cache,
        ggml_tensor   * raw_k_read_idxs,
        int64_t         n_embd_head, const llm_build_cb & cb, [[maybe_unused]] int il) {
    if (cache == nullptr) {
        return nullptr;
    }

    const auto & raw = lctx->dsv4.raw;
    const int64_t n_kv_visible = raw.n_kv;
    if (n_kv_visible <= 0) {
        return nullptr;
    }

    // Keep the visible row count for masks, but expose a stable 256-row cache
    // view to attention.
    const int64_t n_kv = std::max<int64_t>(256, GGML_PAD(n_kv_visible, 256));

    const auto & sinfo = raw.sinfo_read;
    const int64_t n_stream = (int64_t) sinfo.n_stream();
    if (n_stream <= 0) {
        return nullptr;
    }

    const int64_t n_embd_gqa = cache->ne[0];
    GGML_ASSERT(n_embd_head > 0);
    GGML_ASSERT(n_embd_gqa % n_embd_head == 0);

    const int64_t n_head_kv = n_embd_gqa/n_embd_head;

    if (n_stream == 1 && lctx->kv_self.n == raw_k_read_idxs->ne[0]) {
        return ggml_view_3d(ctx, cache, n_embd_head, n_head_kv, n_kv,
                ggml_row_size(cache->type, n_embd_head),
                ggml_row_size(cache->type, n_embd_head)*n_head_kv, 0);
    }

    GGML_ASSERT(raw_k_read_idxs != nullptr);
    GGML_ASSERT(raw_k_read_idxs->ne[0] >= n_kv*n_stream);

    // Gather controller-owned slots into the attention layout.
    ggml_tensor * cache_2d = dsv4_cache_view_2d(ctx, cache, n_embd_gqa, cache->ne[1]);
    ggml_tensor * idxs = raw_k_read_idxs->type == GGML_TYPE_I32
            ? raw_k_read_idxs : ggml_cast(ctx, raw_k_read_idxs, GGML_TYPE_I32);
    ggml_tensor * rows = ggml_get_rows(ctx, cache_2d, idxs);
    cb(rows, "raw_k", il);
    if (ggml_is_quantized(cache->type) && rows->type != GGML_TYPE_F32) {
        rows = ggml_cast(ctx, rows, GGML_TYPE_F32);
    } else if (rows->type != cache->type && !ggml_is_quantized(cache->type)) {
        rows = ggml_cast(ctx, rows, cache->type);
    }

    ggml_tensor * raw_k = ggml_reshape_4d(ctx, rows, n_embd_head, n_head_kv, n_kv, n_stream);

    return raw_k;
}

static ggml_tensor * dsv4_raw_cpy_k(
        llama_context * lctx,
        ggml_context  * ctx,
        ggml_tensor   * cache,
        ggml_tensor   * k_cur,
        ggml_tensor   * raw_k_write_src_idxs,
        ggml_tensor   * raw_k_write_idxs,
        ggml_cgraph   * gf,
        int64_t         n_embd_head,
        const llm_build_cb & cb,
        int64_t         il) {
    if (cache == nullptr || k_cur == nullptr || raw_k_write_idxs == nullptr || raw_k_write_src_idxs == nullptr) {
        return nullptr;
    }

    GGML_ASSERT(2*il + 1 < (int64_t) lctx->cache_copies.size());
    GGML_ASSERT(k_cur->ne[1] == 1);

    ggml_tensor * cache_2d = dsv4_cache_view_2d(ctx, cache, n_embd_head, cache->ne[1]);
    ggml_tensor * cur_2d = ggml_view_2d(ctx, k_cur, n_embd_head, k_cur->ne[2], k_cur->nb[2], 0);
    ggml_tensor * write = nullptr;

    const auto & sinfo = lctx->dsv4.raw.sinfo_write;
    if (sinfo.n_stream() <= 1 && cur_2d->ne[1] == raw_k_write_idxs->ne[0]) {
        cur_2d = dsv4_require_f32_rows(ctx, cur_2d);
        write = ggml_set_rows(ctx, cache_2d, cur_2d, raw_k_write_idxs);
    } else if (sinfo.n_stream() <= 1) {
        ggml_tensor * src_idxs = raw_k_write_src_idxs->type == GGML_TYPE_I32 ? raw_k_write_src_idxs : ggml_cast(ctx, raw_k_write_src_idxs, GGML_TYPE_I32);
        ggml_tensor * cur_sel = ggml_get_rows(ctx, cur_2d, src_idxs);
        cb(cur_sel, "sel", il);
        cur_sel = dsv4_require_f32_rows(ctx, cur_sel);
        write = ggml_set_rows(ctx, cache_2d, cur_sel, raw_k_write_idxs);
    } else {
        const int64_t n_fanout = (int64_t) sinfo.size()*(int64_t) sinfo.n_stream();

        GGML_ASSERT(sinfo.n_stream() > 1);
        GGML_ASSERT(raw_k_write_idxs->ne[0] == n_fanout);
        GGML_ASSERT(raw_k_write_src_idxs->ne[0] == n_fanout);

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            ggml_tensor * src_idxs_s = ggml_view_1d(ctx, raw_k_write_src_idxs, sinfo.size(),
                    s*sinfo.size()*ggml_element_size(raw_k_write_src_idxs));
            ggml_tensor * k_idxs_s = ggml_view_1d(ctx, raw_k_write_idxs, sinfo.size(), s*sinfo.size()*ggml_element_size(raw_k_write_idxs));
            ggml_tensor * cur_sel = ggml_get_rows(ctx, cur_2d, src_idxs_s);
            cb(cur_sel, "sel", il);
            ggml_tensor * cur_f32 = dsv4_require_f32_rows(ctx, cur_sel);
            ggml_tensor * cur = ggml_set_rows(ctx, cache_2d, cur_f32, k_idxs_s);
            if (write == nullptr) {
                write = cur;
            } else {
                ggml_build_forward_expand(gf, cur);
            }
        }
    }

    lctx->cache_copies[2*il + 0].cpy = write;
    lctx->cache_copies[2*il + 0].step = ggml_row_size(cache->type, n_embd_head);
    ggml_build_forward_expand(gf, write);

    return write;
}

static ggml_tensor * dsv4_comp_get_k(
        ggml_context * ctx,
        ggml_tensor  * cache,
        const llama_context::dsv4_runtime::comp_context & comp,
        int64_t        n_embd_head,
        int64_t        kv_size) {
    const int64_t n_kv = comp.n_kv;
    if (cache == nullptr || n_kv <= 0) {
        return nullptr;
    }

    if (comp.sinfo.n_stream() == 0) {
        return ggml_reshape_4d(ctx, dsv4_cache_view_3d(ctx, cache, n_embd_head, n_kv), n_embd_head, 1, n_kv, 1);
    }

    return dsv4_cache_stream_view_4d(ctx, cache, n_embd_head, n_kv, kv_size, comp.sinfo.s0, (int64_t) comp.sinfo.n_stream());
}

static ggml_tensor * dsv4_comp_cpy_k(
        ggml_context * ctx,
        ggml_tensor  * cache,
        ggml_tensor  * cur,
        ggml_tensor  * idxs,
        int64_t        n_embd_head) {
    ggml_tensor * cache_2d = dsv4_cache_view_2d(ctx, cache, n_embd_head, cache->ne[1]);
    cur = dsv4_require_f32_rows(ctx, cur);
    return ggml_set_rows(ctx, cache_2d, cur, idxs);
}

static ggml_tensor * dsv4_comp_state_cpy(
        ggml_context * ctx,
        ggml_tensor  * cache,
        ggml_tensor  * cur,
        ggml_tensor  * idxs) {
    cur = dsv4_require_f32_rows(ctx, cur);
    return ggml_set_rows(ctx, cache, cur, idxs);
}

static ggml_tensor * dsv4_repeat_streams(ggml_context * ctx, ggml_tensor * t, int64_t n_stream) {
    if (t->ne[3] == n_stream) {
        return t;
    }

    GGML_ASSERT(t->ne[3] == 1);
    return ggml_repeat_4d(ctx, t, t->ne[0], t->ne[1], t->ne[2], n_stream);
}

static ggml_tensor * dsv4_build_attn(
        ggml_context * ctx,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * kq_mask,
        ggml_tensor  * sinks,
        float          kq_scale,
        const llm_build_cb & cb,
        int            il,
        int            n_compressed,
        ggml_cgraph  * gf) {
    const bool v_trans = v->nb[1] > v->nb[2];
    const int64_t n_stream = k->ne[3];

    // Multi-stream (parallel sequences) is handled by slicing per stream and
    // processing each stream independently. This applies to the flash attention
    // path as well: the DSV4 cache/controller is non-unified, so a multi-stream
    // mask (ne[3] == n_stream) cannot be routed through ggml_flash_attn_ext,
    // which requires mask->ne[3] == 1.
    if (n_stream > 1) {
        GGML_ASSERT(q->ne[2] % n_stream == 0);
        const int64_t n_tokens_stream = q->ne[2]/n_stream;
        ggml_tensor * result = nullptr;

        for (int64_t s = 0; s < n_stream; ++s) {
            ggml_tensor * q_s = ggml_view_3d(ctx, q, q->ne[0], q->ne[1], n_tokens_stream,
                    q->nb[1], q->nb[2], s*n_tokens_stream*q->nb[2]);
            ggml_tensor * k_s = ggml_view_4d(ctx, k, k->ne[0], k->ne[1], k->ne[2], 1,
                    k->nb[1], k->nb[2], k->nb[3], s*k->nb[3]);
            ggml_tensor * v_s = ggml_view_4d(ctx, v, v->ne[0], v->ne[1], v->ne[2], 1,
                    v->nb[1], v->nb[2], v->nb[3], s*v->nb[3]);
            ggml_tensor * mask_s = kq_mask;
            if (ggml_is_matrix(kq_mask)) {
                mask_s = ggml_view_2d(ctx, kq_mask, kq_mask->ne[0], n_tokens_stream,
                        kq_mask->nb[1], s*n_tokens_stream*kq_mask->nb[1]);
            } else {
                mask_s = ggml_view_2d(ctx, kq_mask, kq_mask->ne[0], kq_mask->ne[1],
                        kq_mask->nb[1], s*kq_mask->nb[3]);
            }

            ggml_tensor * cur_s = dsv4_build_attn(ctx, hparams, cparams,
                    q_s, k_s, v_s, mask_s, sinks, kq_scale, cb, il, n_compressed, gf);
            result = result == nullptr ? cur_s : ggml_concat(ctx, result, cur_s, 1);
        }
        return result;
    }

    q = ggml_view_4d(ctx, q, q->ne[0], q->ne[1], q->ne[2] / n_stream, n_stream,
            q->nb[1], q->nb[2], q->nb[3] / n_stream, 0);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // The DSV4 cache/controller is non-unified. Keep the eligibility rule
    // explicit so a future unified cache cannot route multi-stream masks
    // through Flash Attention accidentally.
    constexpr bool kv_unified = false;
    const bool use_flash_attn = cparams.flash_attn &&
            (!kv_unified || kq_mask->ne[3] == 1);
    if (use_flash_attn) {

        if (v_trans) {
            v = ggml_transpose(ctx, v);
        }

        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx, v, GGML_TYPE_F16);
        }

        if (kq_mask->type == GGML_TYPE_F32) {
            kq_mask = ggml_cast(ctx, kq_mask, GGML_TYPE_F16);
        }

        ggml_tensor * selected = nullptr;
        if (n_compressed > 0) {
            int n_compressed_padded = GGML_PAD(n_compressed, 256);
            if (n_compressed_padded < kq_mask->ne[0]) {
                selected = ggml_mask_to_index(ctx, kq_mask, n_compressed_padded);
                cb(selected, "mask_to_idx", il);
                ggml_build_forward_expand(gf, selected);
            }
        }

        ggml_tensor * cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, "fattn", il);
        // DSV4 uses the generic CPU FA path here for numerical correctness.
        if (selected) {
            cur->src[5] = selected;
        } else {
            cur->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
        }
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        ggml_build_forward_expand(gf, cur);
        return ggml_reshape_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    cb(kq, "kq", il);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    if (kq->type != GGML_TYPE_F32) {
        kq = ggml_cast(ctx, kq, GGML_TYPE_F32);
    }

    if (hparams.attn_soft_cap) {
        kq = ggml_scale(ctx, kq, 1.0f / hparams.f_attn_logit_softcapping);
        kq = ggml_tanh(ctx, kq);
        kq = ggml_scale(ctx, kq, hparams.f_attn_logit_softcapping);
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
    } else {
        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
    }
    cb(kq, "kq_soft_max", il);

    if (!v_trans) {
        v = ggml_cont(ctx, ggml_transpose(ctx, v));
        cb(v, "v_cont", il);
    }

    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);
    cb(kqv, "kqv", il);
    ggml_tensor * cur = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
}

static ggml_tensor * build_hc_pre(
        ggml_context * ctx0,
        llm_build_context & llm,
        const llama_hparams & hparams,
        int64_t n_embd,
        float norm_rms_eps,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post_out,
        ggml_tensor ** comb_out,
        const llm_build_cb & cb, int il,
        ggml_tensor * pre_in = nullptr,      // V4.1: collapse with the mix the previous sublayer produced
        ggml_tensor ** pre_out = nullptr) {  // V4.1: hand this sublayer's mix to the next one
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t nt         = x->ne[2];

    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx0, x);
    }
    auto flat = ggml_reshape_2d(ctx0, x, n_embd * hc, nt);
    auto normed = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    cb(normed, "hc_pre", il);
    auto mixes  = ggml_mul_mat(ctx0, hc_fn, normed);
    cb(mixes, "hc_pre_mixes", il);

    auto all = ggml_hc_pre(ctx0, mixes, hc_scale, hc_base, hc, hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps);

    auto pre  = ggml_view_2d(ctx0, all, hc, nt, hc*sizeof(float), 0);
    auto post = ggml_view_2d(ctx0, all, hc, nt, hc*sizeof(float), hc*nt*sizeof(float));
    auto comb = ggml_view_3d(ctx0, all, hc, hc, nt, hc*sizeof(float), hc*hc*sizeof(float), 2*hc*nt*sizeof(float));

    *post_out = post;
    *comb_out = comb;
    if (pre_out) {
        *pre_out = pre;
    }

    return llm.build_mhc_weighted_sum(x, pre_in ? pre_in : pre, n_embd, hc);
}

static ggml_tensor * build_hc_head(
        ggml_context * ctx0,
        llm_build_context & llm,
        const llama_hparams & hparams,
        int64_t n_embd,
        float norm_rms_eps,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) {
    const int64_t hc     = hparams.dsv4_hc_mult;

    ggml_tensor * mixes = llm.build_mhc_pre_projection(x, hc_fn, nullptr,
            n_embd, hc, norm_rms_eps, false);
    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);

    return llm.build_mhc_weighted_sum(x, pre, n_embd, hc);
}

static ggml_tensor * build_compressed_kv_from_state(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        int il,
        const char * tag,
        bool overlap,
        ggml_tensor ** pre_rope = nullptr) {
    const int64_t n_embd_head_rope = llm.hparams.n_rot;
    const int64_t n_blocks = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs != nullptr);

    // type 0 pools two overlapping groups (V4's CSA), type 1 one disjoint group per row
    int type = overlap ? 0 : 1;
    ggml_tensor * comp = ggml_ds4_comp(ctx0, kv_state, score_state, state_read_idxs, ratio, type);

    llm.cb(comp, tag, il);

    comp = llm.llm_build_norm(ctx0, comp, llm.hparams, norm, nullptr, LLM_NORM_RMS, llm.cb, il);
    llm.cb(comp, tag, il);

    // DeepSeek-V4.1 derives its index keys from the pooled latent before it is rotated
    if (pre_rope) {
        *pre_rope = comp;
    }

    comp = ggml_reshape_3d(ctx0, comp, n_embd_head, 1, n_blocks);
    comp = ggml_rope_ext_inplace(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, llm.rope_type, llm.n_ctx_orig,
            llm.hparams.dsv4_compress_rope_base, llm.freq_scale, llm.ext_factor,
            dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    comp->op_params[15] = 1;
    llm.cb(comp, tag, il);

    return comp;
}

static ggml_tensor * build_top_k_mask(
        ggml_context * ctx0,
        ggml_tensor * kq_mask,
        ggml_tensor * top_k) {
    if (!ggml_is_contiguous(kq_mask)) {
        kq_mask = ggml_cont(ctx0, kq_mask);
    }
    if (top_k->ne[0] <= kq_mask->ne[0] && top_k->ne[1] <= kq_mask->ne[1] && top_k->ne[2] == kq_mask->ne[2] && top_k->ne[3] == kq_mask->ne[3]) {
        return ggml_indexer_mask(ctx0, kq_mask, top_k);
    }
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    //ggml_tensor * kq_mask_top_k = ggml_blend(ctx0, kq_mask_all, top_k, 0.0f);
    // Fo siome reason the above is not faster than this
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);
    return ggml_add(ctx0, kq_mask_top_k, kq_mask);
}

static ggml_tensor * dsv4_build_lid_top_k_shared(
        ggml_context * ctx0,
        ggml_tensor * indexer_k,
        ggml_tensor * indexer_q,
        ggml_tensor * indexer_weights,
        ggml_tensor * indexer_mask,
        int n_top_k, const llm_build_cb & cb) {
    const int64_t n_stream = indexer_k->ne[3];
    const int64_t n_tokens = indexer_q->ne[1];

    if (n_stream <= 0 || indexer_k->ne[2] != 1 || indexer_q->ne[3] != n_stream ||
            indexer_weights->ne[3] != n_stream || indexer_mask->ne[1] < n_tokens ||
            indexer_mask->ne[3] < n_stream) {
        return nullptr;
    }

    ggml_tensor * selected = nullptr;
    for (int64_t s = 0; s < n_stream; ++s) {
        ggml_tensor * k = ggml_view_2d(ctx0, indexer_k,
                indexer_k->ne[0], indexer_k->ne[1], indexer_k->nb[1], s*indexer_k->nb[3]);
        ggml_tensor * q = ggml_view_3d(ctx0, indexer_q,
                indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2],
                indexer_q->nb[1], indexer_q->nb[2], s*indexer_q->nb[3]);
        q = ggml_permute(ctx0, q, 0, 2, 1, 3);

        ggml_tensor * w = ggml_view_2d(ctx0, indexer_weights,
                indexer_weights->ne[0], indexer_weights->ne[1], indexer_weights->nb[1],
                s*indexer_weights->nb[3]);
        ggml_tensor * mask = ggml_view_2d(ctx0, indexer_mask,
                indexer_mask->ne[0], n_tokens, indexer_mask->nb[1],
                s*n_tokens*indexer_mask->nb[1]);

        ggml_tensor * cur = ggml_indexer_topk(ctx0, k, q, w, mask, nullptr,
                GGML_UNARY_OP_RELU, n_top_k);
        if (selected) {
            selected = ggml_concat(ctx0, selected, cur, 1);
            cb(selected, "top_k", s);
        } else {
            selected = cur;
        }
        //selected = selected == nullptr ? cur : ggml_concat(ctx0, selected, cur, 1);
    }

    return selected == nullptr ? nullptr : ggml_cont(ctx0, selected);
}

static ggml_tensor * dsv4_build_lid_top_k(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il, ggml_cgraph * gf, const llm_build_cb & cb,
        ggml_tensor * lid_cache = nullptr,
        const llama_context::dsv4_runtime::comp_context * lid_ctx_in = nullptr,
        int64_t n_lid_in = -1,
        ggml_tensor * lid_kq_mask = nullptr) {
    const auto & hparams = llm.hparams;
    const auto & layer = llm.model.layers[il];
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot;
    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_tokens = cur->ne[1];
    // under shared streams the index keys live on the key owner, under the reader's plan
    if (!lid_cache)   lid_cache   = llm.lctx.dsv4.cache.lid_k[il];
    const auto & lid_ctx = lid_ctx_in ? *lid_ctx_in : llm.lctx.dsv4.lid_ctx;
    if (!lid_kq_mask) lid_kq_mask = llm.lctx.dsv4.inputs.csa.kq_mask;
    const int64_t n_lid = n_lid_in > 0 ? n_lid_in : llm.lctx.dsv4.lid_plan.n_kv;
    const int hadamard_block = llama_model::hadamard_size((int) n_embd_indexer_head);

    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(hadamard_block > 0);
    GGML_ASSERT(n_embd_indexer_head % hadamard_block == 0);

    ggml_tensor * indexer_q = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_attn_q_b, qr);
    llm.cb(indexer_q, "lid_q", il);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);

    indexer_q = ggml_rope_ext_inplace(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            llm.rope_type, llm.n_ctx_orig,
            hparams.dsv4_compress_rope_base, llm.freq_scale,
            llm.ext_factor, dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    indexer_q->op_params[15] = 1;
    llm.cb(indexer_q, "indexer_q", il);
    GGML_ASSERT(indexer_q->ne[0] % hadamard_block == 0);
    indexer_q = ggml_hadamard(ctx0, indexer_q, hadamard_block);
    llm.cb(indexer_q, "lid_q_hadamard", il);

    ggml_tensor * indexer_weights = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_proj, cur);
    llm.cb(indexer_weights, "lid_weights", il);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / std::sqrt(float(n_embd_indexer_head * n_indexer_head)));

    ggml_tensor * indexer_k = dsv4_comp_get_k(ctx0,
            lid_cache,
            lid_ctx,
            n_embd_indexer_head,
            lid_cache->ne[1]/std::max<uint32_t>(1, llm.lctx.dsv4.cache.n_stream));
    llm.cb(indexer_k, "lid_k", il);

    const int64_t n_stream = std::max<int64_t>(1, indexer_k->ne[3]);
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2] / n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3] / n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1] / n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2] / n_stream, indexer_weights->nb[3] / n_stream, 0);

    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    llm.cb(indexer_q, "lid_q_stream", il);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
    llm.cb(indexer_k, "lid_k_stream", il);

    GGML_ASSERT(lid_kq_mask != nullptr);
    ggml_tensor * lid_mask = dsv4_build_raw_mask_view(ctx0,
            lid_kq_mask, nullptr, n_lid, n_tokens, n_stream, cb, il);
    const uint32_t n_top_k = (uint32_t) std::min<int64_t>(n_lid, hparams.indexer_top_k);
    if (llm.cparams.fused_idx_topk && n_lid > n_top_k) {
        if (ggml_tensor * selected = dsv4_build_lid_top_k_shared(ctx0,
                    indexer_k, indexer_q, indexer_weights, lid_mask, (int) n_top_k, cb)) {
            if (selected) {
                ggml_build_forward_expand(gf, selected);
                llm.cb(selected, "lid_top_k", il);
                return selected;
            }
        }
    }

    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
    llm.cb(indexer_kq, "lid_kq", il);

    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
    llm.cb(indexer_kq, "lid_kq_perm", il);

    ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
    indexer_score = ggml_sum_rows(ctx0, indexer_score);
    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
    llm.cb(indexer_score, "lid_score", il);

    indexer_score = ggml_add(ctx0, indexer_score, lid_mask);
    llm.cb(indexer_score, "lid_score_masked", il);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    llm.cb(top_k, "lid_top_k", il);

    return top_k;
}

static void ds4_build_comp(ggml_tensor * cur, llm_build_context & llm, ggml_context * ctx0,
        llama_context::dsv4_runtime::comp_inputs & inputs,
        llama_context::dsv4_runtime::comp_plan & plan,
        ggml_tensor * comp_wkv, ggml_tensor * comp_wgate, ggml_tensor * comp_ape, ggml_tensor * norm,
        ggml_tensor * cache_state, ggml_tensor * cache_score, ggml_tensor * cache_k,
        ggml_tensor ** append_state, ggml_tensor ** append_score,
        int head_size, int il, bool do_hadamard, const std::string & tag, ggml_cgraph * gf, bool is_hca,
        ggml_tensor ** pre_rope_out = nullptr) {

    ggml_tensor * state_kv = llm.llm_build_lora_mm(llm.lctx, ctx0, comp_wkv, cur);
    llm.cb(state_kv, (tag + "_state_kv").c_str(), il);
    // at ratio 1 the file carries no gate, and a softmax over one element is 1 regardless
    ggml_tensor * state_score = comp_wgate ? llm.llm_build_lora_mm(llm.lctx, ctx0, comp_wgate, cur) : state_kv;
    llm.cb(state_score, (tag + "_state_score").c_str(), il);
    if (comp_ape) {
        ggml_tensor * ape_rows = ggml_get_rows(ctx0, comp_ape, inputs.state_pos);
        llm.cb(ape_rows, (tag + "_ape").c_str(), il);
        state_score = ggml_add(ctx0, state_score, ape_rows);
    }
    ggml_tensor * dep = nullptr;

    if (append_state) {
        state_kv = dsv4_append_zero_row(ctx0, state_kv, append_state, false);
    }
    if (append_score) {
        state_score = dsv4_append_zero_row(ctx0, state_score, append_score, true);
    }

    if (inputs.state_write_idxs != nullptr && plan.state_write_idxs.size() > 0) {
        ggml_tensor * source_kv = dsv4_concat_named(ctx0, cache_state, state_kv, 1, (tag + "_source_kv").c_str());
        ggml_tensor * source_score = dsv4_concat_named(ctx0, cache_score, state_score, 1, (tag + "_source_score").c_str());
        const auto & hp = llm.model.hparams;
        const uint32_t ratio = is_hca ? hp.dsv4_hca_ratio : hp.dsv4_csa_ratio;
        const bool  overlap  = is_hca ? hp.dsv4_hca_overlap : hp.dsv4_csa_overlap;
        ggml_tensor * comp = build_compressed_kv_from_state(ctx0, llm,
                                           source_kv, source_score,
                                           inputs.state_read_idxs,
                                           inputs.state_write_pos,
                                           norm, ratio, head_size, il,
                                           (tag + "_state_compress").c_str(), overlap, pre_rope_out);
        if (do_hadamard) {
            const int hadamard_block = llama_model::hadamard_size(head_size);
            GGML_ASSERT(hadamard_block > 0);
            GGML_ASSERT(comp->ne[0] % hadamard_block == 0);
            comp = ggml_hadamard(ctx0, comp, hadamard_block);
            llm.cb(comp, (tag + "_state_compress_hadamard").c_str(), il);
        }
        ggml_tensor * comp_2d = ggml_reshape_2d(ctx0, comp, head_size, inputs.state_write_idxs->ne[0]);
        ggml_tensor * write = dsv4_comp_cpy_k(ctx0, cache_k, comp_2d, inputs.state_write_idxs, head_size);
        ggml_build_forward_expand(gf, write);
        llm.cb(write, (tag + "_k_write").c_str(), il);
        dep = comp;
    }

    if (dep) {
        ggml_build_forward_expand(gf, dep);
    }
    ggml_tensor * persist_kv = ggml_get_rows(ctx0, state_kv, inputs.state_persist_src_idxs);
    llm.cb(persist_kv, (tag + "_persist_kv").c_str(), il);
    ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inputs.state_persist_src_idxs);
    llm.cb(persist_score, (tag + "_persist_score").c_str(), il);
    ggml_tensor * state_kv_write = dsv4_comp_state_cpy(ctx0, cache_state, persist_kv, inputs.state_persist_dst_idxs);
    ggml_tensor * state_score_write = dsv4_comp_state_cpy(ctx0, cache_score, persist_score, inputs.state_persist_dst_idxs);
    ggml_build_forward_expand(gf, state_kv_write);
    ggml_build_forward_expand(gf, state_score_write);
    llm.cb(state_kv_write, (tag + "_k_state_persist").c_str(), il);
    llm.cb(state_score_write, (tag + "_score_state_persist").c_str(), il);
}

// Gather the hashed rows of a token, project them into one key per hyper-connection stream
// plus a shared value, gate the value with the normalized key.query dot product and add it
// into every stream. Row ids come from llama_set_engram_rows; the table is only ever read
// through get_rows, so it stays in the file mapping.
static ggml_tensor * ds4_build_engram(ggml_context * ctx0, llm_build_context & llm, ggml_tensor * x, int il, int eg) {
    const auto & model   = llm.model;
    const auto & hparams = model.hparams;
    const auto & layer   = model.layers[il];
    const auto & cb      = llm.cb;
    auto & lctx = llm.lctx;

    const int64_t n_embd  = llm.n_embd;
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc*n_embd;
    const int64_t nt      = x->ne[2];
    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    if ((int) lctx.inp_engram_rows.size() < eg + 1) lctx.inp_engram_rows.resize(eg + 1, nullptr);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_cols*nt);
    ggml_set_input(rows);
    ggml_format_name(rows, "engram_rows-%d", il);
    lctx.inp_engram_rows[eg] = rows;

    ggml_tensor * emb = ggml_get_rows(ctx0, layer.engram_embd, rows);          // [key_len, n_cols*nt] f32
    emb = ggml_reshape_2d(ctx0, emb, n_cols*key_len, nt);
    cb(emb, "engram_embd", il);

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = llm.llm_build_lora_mm(lctx, ctx0, layer.engram_wkv, emb);  // [n_embd*(hc+1), nt]
    cb(kv, "engram_kv", il);
    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // the gate scales reach ggml_mul, which takes f32; the file may carry them quantized
    auto as_f32 = [&](ggml_tensor * w, const char * name) -> ggml_tensor * {
        if (w->type == GGML_TYPE_F32) return ggml_reshape_2d(ctx0, w, hc_dim, 1);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, w->ne[1]);
        ggml_set_input(ids);
        ggml_format_name(ids, "%s_ids-%d", name, il);
        lctx.inp_engram_gate_ids.push_back(ids);
        return ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, w, ids), hc_dim, 1);
    };
    // normalized per (token, hc copy) over n_embd
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };
    ggml_tensor * k = grouped_norm(key, as_f32(layer.engram_k, "engram_k"));
    ggml_tensor * q = grouped_norm(ggml_cont(ctx0, x), as_f32(layer.engram_q, "engram_q"));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));                 // [1, hc, nt]
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));
    // signed square root before the sigmoid, as in the training kernel
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);
    ggml_tensor * out = ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
    cb(out, "engram_out", il);
    return out;
}

static ggml_tensor * ds4_attention(ggml_cgraph * gf, ggml_context * ctx0, llm_build_context & llm, ggml_tensor * inpL,
        ggml_tensor ** append_csa_state, ggml_tensor ** append_csa_score,
        ggml_tensor ** append_lid_state, ggml_tensor ** append_lid_score,
        ggml_tensor * inp_pos, ggml_tensor * KQ_mask, ggml_tensor * KQ_mask_swa_win, int il,
        ggml_tensor ** comp_kv_ptr, ggml_tensor ** comp_mask_ptr,
        ggml_tensor * hc_pre_in = nullptr, ggml_tensor ** hc_pre_out = nullptr) {

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    const auto & model = llm.model;
    const auto & layer = model.layers[il];
    const auto & hparams = model.hparams;
    const auto & cparams = llm.cparams;
    const auto & cb    = llm.cb;

    auto & lctx    = llm.lctx;
    auto & kv_self = llm.kv_self;

    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t hc = hparams.dsv4_hc_mult;

    const auto n_tokens = llm.n_tokens;
    const auto n_head   = llm.n_head;
    const auto n_kv     = llm.n_kv;

    ggml_tensor * cur = build_hc_pre(ctx0, llm, hparams, llm.n_embd, hparams.f_norm_rms_eps, inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, llm.cb, il, hc_pre_in, hc_pre_out);
    llm.cb(cur, "hc_attn_pre", il);

    cur = llm.llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, llm.cb, il);
    cb(cur, "attn_norm", il);

    ggml_tensor * qr = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = llm.llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "qr_norm", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    const bool use_compress_rope = ratio != 0;
    const float freq_base_l = use_compress_rope ? hparams.dsv4_compress_rope_base : llm.freq_base;
    const float freq_scale_l = use_compress_rope ? llm.freq_scale : 1.0f;
    const float ext_factor_l = use_compress_rope ? llm.ext_factor : 0.0f;
    const float attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l = use_compress_rope ? llm.beta_fast : 0.0f;
    const float beta_slow_l = use_compress_rope ? llm.beta_slow : 0.0f;
    const int32_t n_ctx_orig_l = use_compress_rope ? llm.n_ctx_orig : 0;

    auto build_rope = [&] (int nhead, ggml_tensor * qin, ggml_tensor * wq, ggml_tensor * norm, const std::string & tag) {
        auto q = llm.llm_build_lora_mm(llm.lctx, ctx0, wq, qin);
        cb(q, (tag + "_b").c_str(), il);
        q = ggml_reshape_2d(ctx0, q, n_embd_head, nhead * n_tokens);
        // a bare rms_norm when the layer has no weight for it
        if (norm != nullptr || hparams.dsv4_q_head_norm) {
            q = llm.llm_build_norm(ctx0, q, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(q, (tag + "_norm").c_str(), il);
        }
        q = ggml_reshape_3d(ctx0, q, n_embd_head, nhead, n_tokens);
        q = ggml_rope_ext_inplace(ctx0, q, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        q->op_params[15] = 1;
        cb(q, (tag + "_rope").c_str(), il);
        return q;
    };

    auto q = build_rope(n_head, qr, layer.wq_b, nullptr, "q");

    auto kv = build_rope(1, cur, layer.wkv_latent, layer.attn_kv_norm, "kv");

    if (cparams.k_cache_hadamard) {
        if (int block_size = lctx.model.hadamard_size_k(il); block_size > 0) {
            q = ggml_hadamard(ctx0, q, block_size);
            kv = ggml_hadamard(ctx0, kv, block_size);
            cb(q, "q_hadamard", il);
            cb(kv, "kv_hadamard", il);
        }
    }
    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

    const bool shared = hparams.dsv4_shared_streams;
    const bool is_a = ratio != 0 && ratio == hparams.dsv4_csa_ratio;
    const bool is_b = ratio != 0 && !is_a && ratio == hparams.dsv4_hca_ratio;
    auto & slot_in   = is_a ? lctx.dsv4.inputs.csa : lctx.dsv4.inputs.hca;
    auto & slot_plan = is_a ? lctx.dsv4.csa_plan   : lctx.dsv4.hca_plan;
    auto & slot_ctx  = is_a ? lctx.dsv4.csa_ctx    : lctx.dsv4.hca_ctx;
    ggml_tensor * slot_k = is_a ? lctx.dsv4.cache.csa_k[il] : lctx.dsv4.cache.hca_k[il];
    const bool slot_ready = (is_a || is_b) && slot_in.state_pos != nullptr && slot_plan.state_pos.size() > 0;

    if (shared && slot_ready && hparams.dsv41_is_kv_source(il)) {
        // only a source pools: readers alias its slot cache, the key owner keys off the same latent
        ggml_tensor * pre_rope = nullptr;
        ds4_build_comp(cur, llm, ctx0, slot_in, slot_plan,
                layer.attn_comp_wkv, layer.attn_comp_wgate,
                nullptr, layer.attn_comp_norm,
                is_a ? lctx.dsv4.cache.csa_state_kv[il]    : lctx.dsv4.cache.hca_state_kv[il],
                is_a ? lctx.dsv4.cache.csa_state_score[il] : lctx.dsv4.cache.hca_state_score[il],
                slot_k, nullptr, nullptr,
                n_embd_head, il, false, is_a ? "csa" : "hca", gf, !is_a, &pre_rope);

        if (hparams.dsv41_owns_index_k(il) && pre_rope != nullptr && lctx.dsv4.cache.lid_k[il] != nullptr) {
            const int64_t n_idx  = hparams.indexer_head_size;
            const int64_t n_blk  = pre_rope->ne[1];
            const int hb = llama_model::hadamard_size((int) n_idx);
            GGML_ASSERT(hb > 0 && n_idx % hb == 0);
            ggml_tensor * idx_k = llm.llm_build_lora_mm(lctx, ctx0, layer.indexer_attn_k, pre_rope);
            idx_k = llm.llm_build_norm(ctx0, idx_k, hparams, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, cb, il);
            idx_k = ggml_reshape_3d(ctx0, idx_k, n_idx, 1, n_blk);
            idx_k = ggml_rope_ext_inplace(ctx0, idx_k, slot_in.state_write_pos, nullptr, n_embd_head_rope, llm.rope_type, llm.n_ctx_orig,
                    hparams.dsv4_compress_rope_base, llm.freq_scale, llm.ext_factor,
                    dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
            idx_k->op_params[15] = 1;
            // the indexer query is rotated the same way in dsv4_build_lid_top_k
            idx_k = ggml_hadamard(ctx0, idx_k, hb);
            cb(idx_k, "lid_k_new", il);
            ggml_tensor * write = dsv4_comp_cpy_k(ctx0, lctx.dsv4.cache.lid_k[il], ggml_reshape_2d(ctx0, idx_k, n_idx, n_blk),
                    slot_in.state_write_idxs, n_idx);
            ggml_build_forward_expand(gf, write);
            cb(write, "lid_k_write", il);
        }
    }

    if (!shared && ratio == hparams.dsv4_csa_ratio &&
            lctx.dsv4.inputs.csa.state_pos != nullptr &&
            lctx.dsv4.csa_plan.state_pos.size() > 0) {

        ds4_build_comp(cur, llm, ctx0, lctx.dsv4.inputs.csa, lctx.dsv4.csa_plan,
                layer.attn_comp_wkv, layer.attn_comp_wgate,
                layer.attn_comp_ape, layer.attn_comp_norm,
                lctx.dsv4.cache.csa_state_kv[il], lctx.dsv4.cache.csa_state_score[il], lctx.dsv4.cache.csa_k[il],
                append_csa_state, append_csa_score,
                n_embd_head, il, false, "csa", gf, false);


        ds4_build_comp(cur, llm, ctx0, lctx.dsv4.inputs.lid, lctx.dsv4.lid_plan,
                layer.indexer_comp_wkv, layer.indexer_comp_wgate,
                layer.indexer_comp_ape, layer.indexer_comp_norm,
                lctx.dsv4.cache.lid_state_kv[il], lctx.dsv4.cache.lid_state_score[il], lctx.dsv4.cache.lid_k[il],
                append_lid_state, append_lid_score,
                hparams.indexer_head_size, il, true, "lid", gf, false);

    }

    if (!shared && ratio == hparams.dsv4_hca_ratio &&
            lctx.dsv4.inputs.hca.state_pos != nullptr &&
            lctx.dsv4.hca_plan.state_pos.size() > 0) {

        ds4_build_comp(cur, llm, ctx0, lctx.dsv4.inputs.hca, lctx.dsv4.hca_plan,
                layer.attn_comp_wkv, layer.attn_comp_wgate,
                layer.attn_comp_ape, layer.attn_comp_norm,
                lctx.dsv4.cache.hca_state_kv[il], lctx.dsv4.cache.hca_state_score[il], lctx.dsv4.cache.hca_k[il],
                nullptr, nullptr,
                n_embd_head, il, false, "hca", gf, true);

    }

    const bool raw_compacted = kv_self.is_compacted((int) il);
    // the raw plan indexes compacted rows, so a dense layer in a compacted context (an MTP
    // companion's NextN layer) is cell-addressed and must bypass the plan on store and read alike
    const bool plan_addressed = !kv_self.any_compacted() || raw_compacted;

    ggml_tensor * raw_k_write = nullptr;
    if (plan_addressed && hparams.n_head_kv(il) == 1 && lctx.dsv4.inputs.raw_k_write_idxs != nullptr) {
        raw_k_write = dsv4_raw_cpy_k(&lctx, ctx0, kv_self.k_l[il], kv,
                lctx.dsv4.inputs.raw_k_write_src_idxs, lctx.dsv4.inputs.raw_k_write_idxs, gf, n_embd_head, cb, il);
        if (raw_k_write != nullptr) {
            cb(raw_k_write, "dsv4_raw_k_write", il);
        }
    }
    if (raw_k_write == nullptr) {
        llm.llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, kv, nullptr, n_tokens, llm.kv_head, cb, il);
    }
    if (il < (int64_t) kv_self.v_l.size() && kv_self.v_l[il] != nullptr) {
        llm.llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, nullptr, kv, n_tokens, llm.kv_head, cb, il);
    }

    ggml_tensor * const read_idxs = plan_addressed ? lctx.dsv4.inputs.raw_k_read_idxs : nullptr;

    ggml_tensor * raw_k = nullptr;
    ggml_tensor * raw_mask = nullptr;
    if (raw_compacted) {
        // live window rows [win_off, win_off + w_view); KQ_mask_swa_win is that view's column-exact mask
        GGML_ASSERT(hparams.n_head_kv(il) == 1 && KQ_mask_swa_win != nullptr && lctx.swa_window_view.active);
        const size_t row_size = ggml_row_size(kv_self.k_l[il]->type, n_embd_head);
        raw_k = ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head, 1, lctx.swa_window_view.w_view,
                row_size, row_size, row_size*(size_t) lctx.swa_window_view.win_off);
        raw_mask = KQ_mask_swa_win;
    } else if (hparams.n_head_kv(il) == 1 && read_idxs != nullptr) {
        raw_k = dsv4_raw_get_k(&lctx, ctx0, kv_self.k_l[il], read_idxs, n_embd_head, cb, il);
    }
    if (raw_k == nullptr) {
        raw_k = ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head, hparams.n_head_kv(il), n_kv,
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head),
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head) * hparams.n_head_kv(il),
                0);
    }
    cb(raw_k, "raw_k", il);

    const int64_t raw_kq_n_kv = raw_compacted ? lctx.swa_window_view.w_view
        : raw_k != nullptr && read_idxs != nullptr && lctx.dsv4.raw.n_kv > 0
        ? lctx.dsv4.raw.n_kv
        : (raw_k != nullptr ? raw_k->ne[2] * raw_k->ne[3] : n_kv);
    const int64_t raw_attn_n_kv = raw_compacted ? lctx.swa_window_view.w_view
        : raw_kq_n_kv > 0 ? std::max<int64_t>(256, GGML_PAD(raw_kq_n_kv, 256)) : raw_kq_n_kv;
    if (!raw_compacted && raw_k->ne[3] == 1) {
        raw_k = dsv4_pad_raw_k_to(ctx0, raw_k, raw_attn_n_kv);
    }
    if (raw_mask == nullptr) {
        raw_mask = dsv4_build_raw_mask_view(ctx0, KQ_mask,
                read_idxs, raw_kq_n_kv, n_tokens, raw_k->ne[3], cb, il);
        cb(raw_mask, "raw_mask_view", il);
        raw_mask = dsv4_pad_mask_tokens(ctx0, raw_mask, n_tokens);
        raw_mask = dsv4_pad_raw_mask_to(ctx0, raw_mask, raw_attn_n_kv, n_tokens);
    }
    cb(raw_mask, "dsv4_raw_mask_padded", il);
    ggml_tensor * attn = nullptr;

    if (hparams.n_swa > 0 && !raw_compacted) {
        constexpr int k_fa_chunk = 256;
        int n_swa = hparams.n_swa;
        int ntokens = std::max(k_fa_chunk, int(q->ne[2]));
        int nton = k_fa_chunk*((ntokens + n_swa + k_fa_chunk - 1)/k_fa_chunk);
        int first = raw_k->ne[2] - nton;
        if (first > 0) {
            raw_k = ggml_view_4d(ctx0, raw_k, raw_k->ne[0], raw_k->ne[1], nton, raw_k->ne[3],
                    raw_k->nb[1], raw_k->nb[2], raw_k->nb[3], raw_k->nb[2]*first);
            raw_mask = ggml_view_4d(ctx0, raw_mask, nton, raw_mask->ne[1], raw_mask->ne[2], raw_mask->ne[3],
                    raw_mask->nb[1], raw_mask->nb[2], raw_mask->nb[3], raw_mask->nb[0]*first);
            raw_mask = ggml_cont(ctx0, raw_mask);
        }
    }

    auto build_the_attn = [&] (ggml_tensor * raw_k, ggml_tensor * raw_mask, ggml_tensor * extra_mask,
            ggml_tensor * cache, const auto & extra_ctx,
            const std::string & tag, int n_swa_eff) {
        auto n_stream = std::max<uint32_t>(1, lctx.dsv4.cache.n_stream);
        auto extra_k = cache;
        if (extra_k->ne[1] > 1) {
            extra_k = dsv4_comp_get_k(ctx0, cache, extra_ctx, n_embd_head, cache->ne[1]/n_stream);
            cb(extra_k, "extra_k", il);
        }
        if (extra_k->ne[3] > 1 && extra_mask != nullptr && extra_mask->ne[3] == 1) {
            // The K side is multi-stream, but the mask was built as a flat
            // [n_kv, n_tokens] plan mask (e.g. CSA/HCA plans where the indexer
            // top-k path is skipped). Bring it into the same per-stream layout
            // as the raw mask ([n_kv, n_tokens/n_stream, 1, n_stream]).
            extra_mask = dsv4_build_mask_stream_view(ctx0, extra_mask, extra_k->ne[3], n_tokens);
        }
        if (cparams.flash_attn) {
            extra_mask = dsv4_pad_mask_tokens(ctx0, extra_mask, n_tokens);
        }
        raw_k = dsv4_repeat_streams(ctx0, raw_k, extra_k->ne[3]);
        if (!cparams.flash_attn && !raw_compacted) {
            raw_mask = dsv4_build_raw_mask_view(ctx0, KQ_mask,
                    read_idxs, raw_kq_n_kv, n_tokens, extra_k->ne[3], cb, il);
            raw_mask = dsv4_pad_raw_mask_to(ctx0, raw_mask, raw_attn_n_kv, n_tokens);
        }
        if (!cparams.flash_attn && raw_compacted && raw_mask->ne[1] != n_tokens) {
            // non-FA masks are unpadded; drop the win mask's token padding before the concat
            raw_mask = ggml_view_2d(ctx0, raw_mask, raw_mask->ne[0], n_tokens, raw_mask->nb[1], 0);
        }
        if (cparams.flash_attn && extra_mask->type != GGML_TYPE_F16) {
            extra_mask = ggml_cast(ctx0, extra_mask, GGML_TYPE_F16);
        }
        if (raw_mask->type != extra_mask->type) {
            raw_mask = ggml_cast(ctx0, raw_mask, extra_mask->type);
        }
        if (raw_k->type != extra_k->type) {
            extra_k = ggml_cast(ctx0, extra_k, raw_k->type);
            cb(extra_k, (tag + "_k_cast").c_str(), il);
        }
        ggml_tensor * k_all = dsv4_concat_kv(ctx0, raw_k, extra_k);
        ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, extra_mask, 0);
        cb(extra_k, (tag + "_k").c_str(), il);
        cb(k_all, (tag + "_k_all").c_str(), il);
        cb(kq_mask, (tag + "_kq_mask").c_str(), il);

        auto attn = dsv4_build_attn(ctx0, hparams, cparams, q, k_all, k_all, kq_mask,
                model.layers[il].attn_sinks, kq_scale, cb, il, n_swa_eff, gf);
        return attn;
    };

    auto num_streams = [] (const auto & comp) {
        int n_stream = comp.sinfo.n_stream();
        return std::max(1, n_stream);
    };

    if (shared && (is_a || is_b) &&
            slot_in.kq_mask != nullptr &&
            slot_plan.n_kv > 0 &&
            slot_k != nullptr &&
            !cparams.k_cache_hadamard) {
        // the index source of this stream picks the rows; the layers after it reuse that pick
        ggml_tensor *& shared_top_k = is_a ? lctx.dsv4.top_k_a : lctx.dsv4.top_k_b;
        auto comp_mask = slot_in.kq_mask;
        auto comp_kv   = slot_k;
        bool have_new_top_k = false;
        if (hparams.dsv41_is_index_source(il) && lctx.dsv4.cache.lid_k[il] != nullptr &&
                hparams.indexer_top_k < slot_in.kq_mask->ne[0]) {
            shared_top_k = dsv4_build_lid_top_k(ctx0, llm, qr, cur, inp_pos, il, gf, cb,
                    lctx.dsv4.cache.lid_k[il], &slot_ctx, slot_plan.n_kv, slot_in.kq_mask);
            have_new_top_k = true;
        }
        int n_eff = hparams.n_swa + (int) slot_plan.n_kv;
        if (shared_top_k) {
            if (have_new_top_k || hparams.dsv41_kv_source[il] != hparams.dsv41_kv_source[hparams.dsv41_topk_source[il]]) {
                if (n_tokens == 1) {
                    comp_kv = ggml_get_rows_ext(ctx0, comp_kv, shared_top_k, true, false);
                    comp_kv = ggml_reshape_3d(ctx0, comp_kv, comp_kv->ne[0], 1, comp_kv->ne[1]);
                    cb(comp_kv, "comp_kv_getrows", il);
                    comp_mask = ggml_get_rows_ext(ctx0, comp_mask, shared_top_k, true, true);
                    *comp_kv_ptr = comp_kv;
                    *comp_mask_ptr = comp_mask;
                } else {
                    comp_mask = build_top_k_mask(ctx0, dsv4_build_raw_mask_view(ctx0, slot_in.kq_mask, nullptr,
                                slot_plan.n_kv, n_tokens, num_streams(slot_ctx), cb, il), shared_top_k);
                    *comp_mask_ptr = comp_mask;
                    cb(comp_mask, "comp_mask", il);
                }
            } else {
                GGML_ASSERT(*comp_mask_ptr);
                if (n_tokens == 1) {
                    GGML_ASSERT(*comp_kv_ptr);
                    comp_kv = *comp_kv_ptr;
                    comp_mask = *comp_mask_ptr;
                } else {
                    comp_mask = *comp_mask_ptr;
                }
            }
            n_eff = hparams.n_swa + (int) shared_top_k->ne[0];
        }
        attn = build_the_attn(raw_k, raw_mask, comp_mask, comp_kv, slot_ctx, is_a ? "csa" : "hca", n_eff);
        cb(attn, "attn_shared", il);
    } else if (!shared && ratio == hparams.dsv4_csa_ratio &&
            lctx.dsv4.inputs.csa.kq_mask != nullptr &&
            lctx.dsv4.csa_plan.n_kv > 0 &&
            lctx.dsv4.lid_plan.n_kv > 0 &&
            !cparams.k_cache_hadamard) {
        auto csa_mask = lctx.dsv4.inputs.csa.kq_mask;
        auto csa_kv   = lctx.dsv4.cache.csa_k[il];
        if (hparams.indexer_top_k < lctx.dsv4.inputs.csa.kq_mask->ne[0]) {
            auto top_k = dsv4_build_lid_top_k(ctx0, llm, qr, cur, inp_pos, il, gf, cb);
            if (n_tokens == 1) {
                // When we are dealing with a single token, we can just use ggml_get_rows_ext to get the
                // selected rows from the CSA cache and setup the corresponding mask. This makes the
                // raw_kv and csa_kv concetenation much less expensive for long context.
                csa_kv = ggml_get_rows_ext(ctx0, csa_kv, top_k, true, false);
                csa_kv = ggml_reshape_3d(ctx0, csa_kv, csa_kv->ne[0], 1, csa_kv->ne[1]);
                cb(csa_kv, "csa_kv_getrows", il);
                csa_mask = ggml_get_rows_ext(ctx0, csa_mask, top_k, true, true);
            } else {
                csa_mask = build_top_k_mask(ctx0, dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.csa.kq_mask, nullptr,
                            lctx.dsv4.csa_plan.n_kv, n_tokens, num_streams(lctx.dsv4.csa_ctx), cb, il), top_k);
                cb(csa_mask, "csa_mask", il);
            }
        }
        int n_csa = hparams.n_swa + hparams.indexer_top_k;
        attn = build_the_attn(raw_k, raw_mask, csa_mask, csa_kv, lctx.dsv4.csa_ctx, "csa", n_csa);
        cb(attn, "attn_csa", il);
    } else if (!shared && ratio == hparams.dsv4_hca_ratio &&
            lctx.dsv4.inputs.hca.kq_mask != nullptr &&
            lctx.dsv4.hca_plan.n_kv > 0 &&
            std::any_of(lctx.dsv4.hca_plan.n_visible.begin(), lctx.dsv4.hca_plan.n_visible.end(),
                [](int32_t n_visible) { return n_visible > 0; }) &&
            !cparams.k_cache_hadamard) {
        ggml_tensor * hca_mask = dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.hca.kq_mask, nullptr,
                lctx.dsv4.hca_plan.n_kv, n_tokens, num_streams(lctx.dsv4.hca_ctx), cb, il);
        int n_hca = hparams.n_swa + (n_kv + hparams.dsv4_hca_ratio - 1)/hparams.dsv4_hca_ratio;
        attn = build_the_attn(raw_k, raw_mask, hca_mask, lctx.dsv4.cache.hca_k[il], lctx.dsv4.hca_ctx, "hca", n_hca);
        cb(attn, "attn_hca", il);
    } else {
        attn = dsv4_build_attn(ctx0, hparams, cparams, q, raw_k, raw_k, raw_mask, model.layers[il].attn_sinks, kq_scale, cb, il, -1, gf);
        cb(attn, "attn_raw", il);
    }
    ggml_build_forward_expand(gf, attn);

    attn = ggml_reshape_3d(ctx0, attn, n_embd_head, n_head, n_tokens);
    attn = ggml_rope_ext_inplace(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    attn->op = GGML_OP_ROPE_BACK;
    attn->op_params[15] = 1;
    cb(attn, "attn", il);

    const int64_t o_group_dim = layer.wo_a->ne[0];
    const int64_t n_groups = (n_head * n_embd_head) / o_group_dim;
    const int64_t o_lora_rank = layer.wo_b->ne[0] / n_groups;

    GGML_ASSERT((n_head * n_embd_head) % o_group_dim == 0);
    GGML_ASSERT(layer.wo_b->ne[0] % n_groups == 0);

    attn = ggml_reshape_3d(ctx0, attn, o_group_dim, n_groups, n_tokens);
    attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);

    ggml_tensor * oa = ggml_mul_mat(ctx0,
            ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups),
            attn);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    if (n_tokens == 1) {
        oa = ggml_reshape_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
    } else {
        oa = ggml_cont_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
    }

    cur = llm.llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
    cb(cur, "attn_out", il);

    inpL = llm.build_mhc_post(cur, post, residual, comb, llm.n_embd, hc, true);
    cb(inpL, "hc_attn_post", il);

    return inpL;

}

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    const bool is_mtp = lctx.cparams.mtp_op_type != MTP_OP_NONE;

    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(0));
    GGML_ASSERT(n_embd_head_nope > 0);

    dsv4_new_i32_input(ctx0, &lctx.dsv4.inputs.raw_k_write_src_idxs, (int64_t) lctx.dsv4.raw.write_src_idxs.size(), "dsv4_raw_k_write_src_idxs");
    dsv4_new_i32_input(ctx0, &lctx.dsv4.inputs.raw_k_write_idxs, (int64_t) lctx.dsv4.raw.write_dst_idxs.size(), "dsv4_raw_k_write_idxs");
    dsv4_new_i32_input(ctx0, &lctx.dsv4.inputs.raw_k_read_idxs, (int64_t) lctx.dsv4.raw.read_dst_idxs.size(), "dsv4_raw_k_read_idxs");
    dsv4_build_plan_inputs(ctx0, lctx.dsv4.inputs.csa, lctx.dsv4.csa_plan, "dsv4_csa", n_tokens, true, lctx.cparams.flash_attn);
    dsv4_build_plan_inputs(ctx0, lctx.dsv4.inputs.hca, lctx.dsv4.hca_plan, "dsv4_hca", n_tokens, true, lctx.cparams.flash_attn);
    dsv4_build_plan_inputs(ctx0, lctx.dsv4.inputs.lid, lctx.dsv4.lid_plan, "dsv4_lid", n_tokens, false, lctx.cparams.flash_attn);

    ggml_tensor * inp_pos = build_inp_pos();
    // build only the mask the graph consumes; an input tensor without a consumer is never allocated
    ggml_tensor * KQ_mask = nullptr;
    ggml_tensor * KQ_mask_swa_win = nullptr;
    const int n_layer_begin = is_mtp ? n_layer - hparams.nextn_predict_layers : 0;
    const int n_layer_end   = is_mtp ? n_layer : n_layer - hparams.nextn_predict_layers;

    if (kv_self.any_compacted()) {
        // an MTP companion walks only its dense NextN layer while its other layers compact
        bool walked_compacted = false, walked_dense = false;
        for (int il = n_layer_begin; il < n_layer_end; ++il) {
            (kv_self.is_compacted(il) ? walked_compacted : walked_dense) = true;
        }
        if (walked_compacted) {
            bool KQ_mask_swa_windowed = false;
            KQ_mask_swa_win = build_swa_mask_for_graph(hparams.n_swa, /* compacted = */ true, &KQ_mask_swa_windowed);
            GGML_ASSERT(KQ_mask_swa_windowed && KQ_mask_swa_win != nullptr);
        }
        if (walked_dense) {
            KQ_mask = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
        }
    } else {
        KQ_mask = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
    }
    ggml_tensor * inpL = nullptr;

    ggml_tensor * append_csa_state = nullptr;
    ggml_tensor * append_csa_score = nullptr;
    ggml_tensor * append_lid_state = nullptr;
    ggml_tensor * append_lid_score = nullptr;

    if (is_mtp) {
        GGML_ASSERT(model.mtp && hparams.nextn_predict_layers == 1);
        GGML_ASSERT(n_layer > hparams.nextn_predict_layers);

        const int64_t n_hidden = n_embd * hc;
        ggml_tensor * hidden_state = build_inp_mtp_states(n_hidden);

        ggml_tensor * tok_embd = build_inp_embd_mtp(model.tok_embd);
        const int il_mtp = n_layer - hparams.nextn_predict_layers;
        const auto & mtp_layer = model.layers[il_mtp];

        hidden_state = ggml_reshape_2d(ctx0, hidden_state, n_embd, hc * n_tokens);
        tok_embd = ggml_reshape_3d(ctx0, tok_embd, n_embd, 1, n_tokens);
        tok_embd = ggml_repeat_4d(ctx0, tok_embd, n_embd, hc, n_tokens, 1);
        tok_embd = ggml_reshape_2d(ctx0, tok_embd, n_embd, hc * n_tokens);
        inpL = build_mtp_input(mtp_layer, hidden_state, tok_embd, il_mtp);
        GGML_ASSERT(inpL->ne[0] == n_embd);
        GGML_ASSERT(inpL->ne[1] == hc * n_tokens);
        GGML_ASSERT(inpL->ne[2] == 1);
        GGML_ASSERT(inpL->ne[3] == 1);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, hc, n_tokens);
        cb(inpL, "mtp_eh_proj", il_mtp);
    } else {
        ggml_tensor * inp = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
        inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
        cb(inpL, "hc_init", -1);
    }

    lctx.dsv4.top_k_a = nullptr;
    lctx.dsv4.top_k_b = nullptr;

    // lagged hyper-connections: a sublayer computes the mix the next one collapses with; layer 0
    // gets a one-hot that selects the first copy, and the last FFN's mix collapses the output
    const bool hc_lag = hparams.dsv4_hc_lag;
    ggml_tensor * hc_pre_mix = nullptr;
    if (hc_lag) {
        GGML_ASSERT(!is_mtp && "V4.1 MTP graph is not implemented");
        hc_pre_mix = ggml_concat(ctx0,
                ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
                ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
        cb(hc_pre_mix, "hc_pre_init", -1);
    }

    ggml_tensor * comp_kv   = nullptr;
    ggml_tensor * comp_mask = nullptr;

    for (int il = n_layer_begin; il < n_layer_end; ++il) {

        if (model.layers[il].engram_embd) {
            const int eg = hparams.engram_index(il);
            GGML_ASSERT(eg >= 0 && "engram tensor on a layer the metadata does not list");
            inpL = ds4_build_engram(ctx0, *this, inpL, il, eg);
        }

        ggml_tensor * hc_attn_pre = nullptr;
        auto cur = ds4_attention(gf, ctx0, *this, inpL,
                             &append_csa_state, &append_csa_score,
                             &append_lid_state, &append_lid_score,
                             inp_pos, KQ_mask, KQ_mask_swa_win, il,
                             &comp_kv, &comp_mask,
                             hc_pre_mix, hc_lag ? &hc_attn_pre : nullptr);
        inpL = cur;

        ggml_tensor *post, *comb;
        auto residual = inpL;
        cur = build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, cb, il,
                hc_attn_pre, hc_lag ? &hc_pre_mix : nullptr);
        cb(cur, "hc_ffn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        } else {
            // DSV4 uses separate up and gate expert tensors. Do not silently
            // select the fork-only merged gate path for another GGUF.
            //GGML_ASSERT(model.layers[il].ffn_up_gate_exps == nullptr &&
            //        "merged DSV4 MoE gate tensors use an unsupported layout");
            ggml_tensor * selected_experts = nullptr;
            ggml_tensor * exp_probs_b = model.layers[il].ffn_exp_probs_b;
            const bool is_media = lctx.inp_embd != nullptr;
            if (is_media && model.layers[il].ffn_exp_probs_b_vl != nullptr) {
                // image tokens route through the vision bias instead of the hash path
                exp_probs_b = model.layers[il].ffn_exp_probs_b_vl;
            } else if ((uint32_t) il < hparams.dsv4_hash_layer_count && lctx.inp_tokens != nullptr) {
                selected_experts = ggml_get_rows(ctx0, model.layers[il].ffn_gate_tid2eid, lctx.inp_tokens);
                cb(selected_experts, "hashed_exps", il);
                exp_probs_b = nullptr;
            } else if (is_media && (uint32_t) il < hparams.dsv4_hash_layer_count) {
                // media batch without the vision bias: no token ids for the hash-layer gather; fall back to generic expert routing (degraded, but valid).
                static std::atomic_bool warned_media_hash_fallback{false};
                bool expect = false;
                if (warned_media_hash_fallback.compare_exchange_strong(expect, true)) {
                    LLAMA_LOG_WARN("%s: hash layers lack the vision expert-routing bias (exp_probs_b_vl); image tokens fall back to generic expert routing there. Re-export the model with blk.N.exp_probs_b_vl.bias to restore vision expert routing on hash layers\n", __func__);
                }
            }

            // Hash layers carry an explicit fixed-width expert map. During
            // warmup the generic graph reserves all experts, but this input
            // still contains only the model's active expert IDs.
            const int64_t moe_n_expert_used = selected_experts != nullptr
                    ? selected_experts->ne[0]
                    : n_expert_used;

            auto moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        nullptr,
                        model.layers[il].ffn_up_exps,
                        nullptr,
                        model.layers[il].ffn_gate_exps,
                        nullptr,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        exp_probs_b,
                        n_expert, moe_n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps, nullptr, nullptr, nullptr,
                        selected_experts);

            ggml_build_forward_expand(gf, moe_out);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            ggml_build_forward_expand(gf, cur);
        }

        cb(cur, "ffn_out", il);

        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hc, true);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        if (lctx.dflash.capture) {
            ggml_tensor * capture = dsv4_hc_mean_for_capture(ctx0, inpL);
            cb(capture, "l_out", il);
            ggml_build_forward_expand(gf, capture);
        } else {
            cb(inpL, "l_out", il);
        }
    }

    if (is_mtp) {
        const int il_mtp = n_layer - hparams.nextn_predict_layers;
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
        cb(h_nextn, "result_mtp_embd", -1);
        ggml_set_output(h_nextn);
        ggml_build_forward_expand(gf, h_nextn);

        inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);
        ggml_tensor * out = build_hc_head(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
        cb(out, "mtp_hc_head", -1);

        ggml_tensor * head_norm = model.layers[il_mtp].nextn.shared_head_norm
                ? model.layers[il_mtp].nextn.shared_head_norm : model.output_norm;
        GGML_ASSERT(head_norm != nullptr);
        out = llm_build_norm(ctx0, out, hparams, head_norm, nullptr, LLM_NORM_RMS, cb, -1);
        cb(out, "mtp_shared_head_norm", -1);

        out = build_output(lctx, ctx0, out, model.output, nullptr, cb);
        cb(out, "result_output", -1);
        ggml_build_forward_expand(gf, out);
        return gf;
    }

    if (lctx.cparams.mtp && (hparams.nextn_predict_layers > 0 || llm_arch_is_dsv4(model.arch))) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
        cb(h_nextn, "result_mtp_embd", -1);
        ggml_set_output(h_nextn);
        ggml_build_forward_expand(gf, h_nextn);
    }

    if (n_outputs != n_tokens) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
        if (hc_pre_mix) {
            hc_pre_mix = ggml_get_rows(ctx0, hc_pre_mix, inp_out_ids);
        }
    }

    ggml_tensor * out = nullptr;
    if (hc_lag) {
        // the last FFN's mix is the one nothing consumed: it collapses the copies at the output
        out = build_mhc_weighted_sum(inpL, hc_pre_mix, n_embd, hc);
        cb(out, "hc_out", -1);
    } else {
        out = build_hc_head(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.hc_head_fn,
                model.hc_head_scale,
                model.hc_head_base);
        cb(out, "hc_head", -1);
    }

    if (model.output_norm != nullptr) {
        out = llm_build_norm(ctx0, out, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
        cb(out, "result_norm", -1);
        out = build_output(lctx, ctx0, out, model.output, nullptr, cb);
    } else {
        out = build_output(lctx, ctx0, out, model.output, nullptr, cb);
    }
    cb(out, "result_output", -1);

    ggml_build_forward_expand(gf, out);

    return gf;
}

ggml_cgraph * llm_build_context::build_dflash_dsv4() {
    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t ctx_len = lctx.dflash.visible_cross_ctx > 0
            ? (int64_t) lctx.dflash.visible_cross_ctx
            : std::max<int64_t>(1, (int64_t) cparams.n_ctx - (int64_t) hparams.dflash_block_size);
    const int64_t n_kv_total = GGML_PAD(ctx_len + n_tokens, flash_attn ? 256 : 32);
    const ggml_type mask_type = flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!lctx.ensure_dflash_kv_cache_tensors((int32_t) ctx_len)) {
        LLAMA_LOG_ERROR("%s: failed to initialize DSV4 DFlash K/V cache\n", __func__);
        return nullptr;
    }

    GGML_ASSERT(n_embd_head_nope > 0);
    GGML_ASSERT(hparams.n_head_kv() == 1);
    GGML_ASSERT(model.dflash_fc != nullptr);
    GGML_ASSERT(model.dflash_hidden_norm != nullptr);
    // like the body: no learned output head, and the hyper-connection mixes lag by one sublayer
    const bool hc_lag = hparams.dflash_dsv41;
    GGML_ASSERT((model.hc_head_fn != nullptr && model.hc_head_base != nullptr && model.hc_head_scale != nullptr) || hc_lag);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes((int) std::max<int64_t>(n_tokens, ctx_len)) + 48 * n_layer, false);

    lctx.dflash.inputs.kq_mask = nullptr;
    lctx.dflash.kv.kq_mask_tensor = nullptr;
    lctx.dflash.inputs.kq_mask_swa = ggml_new_tensor_2d(ctx0, mask_type, n_kv_total, GGML_PAD(n_tokens, GGML_KQ_MASK_PAD));
    lctx.dflash.kv.kq_mask_swa_tensor = lctx.dflash.inputs.kq_mask_swa;
    ggml_set_input(lctx.dflash.inputs.kq_mask_swa);
    cb(lctx.dflash.inputs.kq_mask_swa, "dsv4_dflash_kq_mask_swa", -1);

    lctx.dflash.kv.draft_tail_rows_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(lctx.dflash.kv.draft_tail_rows_tensor);
    cb(lctx.dflash.kv.draft_tail_rows_tensor, "dflash_draft_tail_rows", -1);

    ggml_tensor * tok_embd = model.tok_embd;
    GGML_ASSERT(tok_embd != nullptr);
    ggml_tensor * inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, tok_embd, cb);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hparams.dsv4_hc_mult, n_tokens, 1);
    cb(inpL, "dsv4_dflash_hc_init", -1);

    auto build_attention = [&](int il, ggml_tensor * input, ggml_tensor * inp_pos) {
        const auto & layer = model.layers[il];
        ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, layer.wq_a, input);
        qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
        ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq_b, qr);
        if (!hc_lag) {
            q = ggml_reshape_2d(ctx0, q, n_embd_head, n_head * n_tokens);
            q = ggml_rms_norm(ctx0, q, hparams.f_norm_rms_eps);
        }
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
        q = ggml_rope_ext_inplace(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q->op_params[15] = 1;
        cb(q, "dsv4_dflash_q", il);

        ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, layer.wkv_latent, input);
        kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, cb, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

        kv = ggml_rope_ext_inplace(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        kv->op_params[15] = 1;
        kv = ggml_permute(ctx0, kv, 0, 2, 1, 3);
        cb(kv, "dsv4_dflash_kv", il);

        ggml_tensor * Kcur = ggml_set_rows(ctx0, lctx.dflash.kv.k_ctx_cache[il], kv,
                lctx.dflash.kv.draft_tail_rows_tensor);
        ggml_tensor * Vcur = ggml_set_rows(ctx0, lctx.dflash.kv.v_ctx_cache[il], kv,
                lctx.dflash.kv.draft_tail_rows_tensor);
        cb(Kcur, "dsv4_dflash_k_set_tail", il);
        cb(Vcur, "dsv4_dflash_v_set_tail", il);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);
        Kcur = ggml_view_3d(ctx0, lctx.dflash.kv.k_ctx_cache[il],
                lctx.dflash.kv.k_ctx_cache[il]->ne[0], n_kv_total,
                lctx.dflash.kv.k_ctx_cache[il]->ne[2],
                lctx.dflash.kv.k_ctx_cache[il]->nb[1],
                lctx.dflash.kv.k_ctx_cache[il]->nb[2], 0);
        Vcur = ggml_view_3d(ctx0, lctx.dflash.kv.v_ctx_cache[il],
                lctx.dflash.kv.v_ctx_cache[il]->ne[0], n_kv_total,
                lctx.dflash.kv.v_ctx_cache[il]->ne[2],
                lctx.dflash.kv.v_ctx_cache[il]->nb[1],
                lctx.dflash.kv.v_ctx_cache[il]->nb[2], 0);
        if (Kcur->type == GGML_TYPE_F32) { Kcur = ggml_cast(ctx0, Kcur, GGML_TYPE_F16); }
        if (Vcur->type == GGML_TYPE_F32) { Vcur = ggml_cast(ctx0, Vcur, GGML_TYPE_F16); }

        ggml_tensor * q_attn = ggml_permute(ctx0, q, 0, 2, 1, 3);
        ggml_tensor * mask = lctx.dflash.inputs.kq_mask_swa;
        ggml_tensor * attn = ggml_flash_attn_ext(ctx0, q_attn, Kcur, Vcur, mask,
                1.0f / std::sqrt((float) n_embd_head), hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        if (layer.attn_sinks) {
            ggml_flash_attn_ext_add_sinks(attn, layer.attn_sinks);
        }
        cb(attn, "dsv4_dflash_attn", il);
        ggml_build_forward_expand(gf, attn);

        attn = ggml_reshape_3d(ctx0, attn, n_embd_head, n_head, n_tokens);
        attn = ggml_rope_ext_inplace(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        attn->op = GGML_OP_ROPE_BACK;
        attn->op_params[15] = 1;

        const int64_t o_group_dim = layer.wo_a->ne[0];
        const int64_t n_groups = (n_head * n_embd_head) / o_group_dim;
        const int64_t o_lora_rank = layer.wo_b->ne[0] / n_groups;
        attn = ggml_reshape_3d(ctx0, attn, o_group_dim, n_groups, n_tokens);
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);
        ggml_tensor * oa = ggml_mul_mat(ctx0,
                ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), attn);
        oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
        oa = ggml_cont_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
        return llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
    };

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * hc_pre_mix = nullptr;
    if (hc_lag) {
        // layer 0 collapses with a one-hot selecting the first copy
        hc_pre_mix = ggml_concat(ctx0,
                ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens), 1.0f),
                ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.dsv4_hc_mult - 1, n_tokens), 0.0f), 0);
        cb(hc_pre_mix, "dsv4_dflash_hc_pre_init", -1);
    }
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;
        ggml_tensor * hc_attn_pre = nullptr;
        ggml_tensor * cur = build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, cb, il,
                hc_pre_mix, hc_lag ? &hc_attn_pre : nullptr);
        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cur = build_attention(il, cur, inp_pos);
        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hparams.dsv4_hc_mult, true);

        residual = inpL;
        cur = build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, cb, il,
                hc_attn_pre, hc_lag ? &hc_pre_mix : nullptr);
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        ggml_tensor * moe = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp, nullptr,
                layer.ffn_up_exps, nullptr,
                layer.ffn_gate_exps, nullptr,
                layer.ffn_down_exps, nullptr,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm, true, hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, false, layer.ffn_up_gate_exps, nullptr, nullptr, nullptr, nullptr);
        ggml_tensor * shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, false, false);
        cur = ggml_add(ctx0, moe, shexp);
        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hparams.dsv4_hc_mult, true);
    }

    ggml_tensor * out = nullptr;
    if (hc_lag) {
        out = build_mhc_weighted_sum(inpL, hc_pre_mix, n_embd, hparams.dsv4_hc_mult);
        cb(out, "dsv4_dflash_hc_out", -1);
    } else {
        out = build_hc_head(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    }
    out = llm_build_norm(ctx0, out, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
    out = build_output(lctx, ctx0, out, model.output, nullptr, cb);
    if (lctx.dflash.dspark) {
        cb(out, "dflash_base_result_output", -1);
    } else {
        cb(out, "result_output", -1);
    }
    ggml_build_forward_expand(gf, out);

    lctx.dflash.draft_tokens_tensor = nullptr;
    ggml_tensor * draft_tokens = nullptr;
    ggml_tensor * result = out;
    if (lctx.dflash.dspark) {
        result = build_dspark_logits(*this, out, lctx.inp_tokens, &draft_tokens);
        cb(result, "result_output", -1);
    } else {
        draft_tokens = ggml_argmax(ctx0, result);
    }
    ggml_set_name(draft_tokens, "draft_argmax");
    ggml_build_forward_expand(gf, result);
    ggml_build_forward_expand(gf, draft_tokens);
    lctx.dflash.draft_tokens_tensor = draft_tokens;
    return gf;
}




// ---- V4.1 separate-arch implementation: selected at runtime by V41_SEPARATE, alongside the in-threaded path ----


static ggml_tensor * dsv4_build_candidate_mask(
        ggml_context * ctx0,
        ggml_tensor * block_score,
        ggml_tensor * cand_pin,
        uint32_t cand_topk_blocks,
        const llm_build_cb & cb, int il);
static ggml_tensor * dsv4_expand_cand_keep(
        ggml_context * ctx0, ggml_tensor * block_keep, int64_t n_pos,
        const llm_build_cb & cb, int il);
static ggml_tensor * ds4_attention_v41(ggml_cgraph * gf, ggml_context * ctx0, llm_build_context & llm, ggml_tensor * inpL,
        ggml_tensor ** append_csa_state, ggml_tensor ** append_csa_score,
        ggml_tensor * inp_pos, ggml_tensor * KQ_mask, ggml_tensor * KQ_mask_swa_win, int il,
        ggml_tensor ** topk_carry = nullptr, ggml_tensor * hc_pre_in = nullptr, ggml_tensor ** hc_pre_out = nullptr,
        ggml_tensor ** cand_carry = nullptr);
static ggml_tensor * dsv4_build_lid_top_k_v41(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il, ggml_cgraph * gf, const llm_build_cb & cb,
        ggml_tensor * kq_mask = nullptr,
        ggml_tensor * cand_pin = nullptr, ggml_tensor ** cand_carry = nullptr);
static ggml_tensor * dsv4_build_v41_index_key(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * latent,
        ggml_tensor * comp_pos,
        ggml_tensor * write_idxs,
        int il);

static ggml_tensor * ds4_attention_v41(ggml_cgraph * gf, ggml_context * ctx0, llm_build_context & llm, ggml_tensor * inpL,
        ggml_tensor ** append_csa_state, ggml_tensor ** append_csa_score,
        ggml_tensor * inp_pos, ggml_tensor * KQ_mask, ggml_tensor * KQ_mask_swa_win, int il,
        ggml_tensor ** topk_carry, ggml_tensor * hc_pre_in, ggml_tensor ** hc_pre_out,
        ggml_tensor ** cand_carry) {

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    const auto & model = llm.model;
    const auto & layer = model.layers[il];
    const auto & hparams = model.hparams;
    const auto & cparams = llm.cparams;
    const auto & cb    = llm.cb;

    auto & lctx    = llm.lctx;
    auto & kv_self = llm.kv_self;

    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t hc = hparams.dsv4_hc_mult;

    const auto n_tokens = llm.n_tokens;
    const auto n_head   = llm.n_head;
    const auto n_kv     = llm.n_kv;

    ggml_tensor * cur = build_hc_pre(ctx0, llm, hparams, llm.n_embd, hparams.f_norm_rms_eps, inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, llm.cb, il, hc_pre_in, hc_pre_out);
    llm.cb(cur, "hc_attn_pre", il);

    cur = llm.llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, llm.cb, il);
    cb(cur, "attn_norm", il);

    ggml_tensor * qr = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = llm.llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "qr_norm", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    const bool use_compress_rope = ratio != 0;
    const float freq_base_l = use_compress_rope ? hparams.dsv4_compress_rope_base : llm.freq_base;
    const float freq_scale_l = use_compress_rope ? llm.freq_scale : 1.0f;
    const float ext_factor_l = use_compress_rope ? llm.ext_factor : 0.0f;
    const float attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l = use_compress_rope ? llm.beta_fast : 0.0f;
    const float beta_slow_l = use_compress_rope ? llm.beta_slow : 0.0f;
    const int32_t n_ctx_orig_l = use_compress_rope ? llm.n_ctx_orig : 0;

    auto build_rope = [&] (int nhead, ggml_tensor * qin, ggml_tensor * wq, ggml_tensor * norm, const std::string & tag) {
        auto q = llm.llm_build_lora_mm(llm.lctx, ctx0, wq, qin);
        cb(q, (tag + "_b").c_str(), il);
        q = ggml_reshape_2d(ctx0, q, n_embd_head, nhead * n_tokens);
        if (tag == "kv") {
            q = llm.llm_build_norm(ctx0, q, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);
            cb(q, (tag + "_norm").c_str(), il);
        }
        q = ggml_reshape_3d(ctx0, q, n_embd_head, nhead, n_tokens);
        q = ggml_rope_ext_inplace(ctx0, q, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        q->op_params[15] = 1;
        cb(q, (tag + "_rope").c_str(), il);
        return q;
    };

    auto q = build_rope(n_head, qr, layer.wq_b, nullptr, "q");

    auto kv = build_rope(1, cur, layer.wkv_latent, layer.attn_kv_norm, "kv");

    if (cparams.k_cache_hadamard) {
        if (int block_size = lctx.model.hadamard_size_k(il); block_size > 0) {
            q = ggml_hadamard(ctx0, q, block_size);
            kv = ggml_hadamard(ctx0, kv, block_size);
            cb(q, "q_hadamard", il);
            cb(kv, "kv_hadamard", il);
        }
    }
    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

    const bool is_csa_kv_source = hparams.dsv41_is_kv_source(il);
    if (ratio == hparams.dsv4_csa_ratio && is_csa_kv_source &&
            lctx.dsv4.inputs.csa.state_pos != nullptr &&
            lctx.dsv4.csa_plan.state_pos.size() > 0) {

        ggml_tensor * csa_latent = nullptr;
        ds4_build_comp(cur, llm, ctx0, lctx.dsv4.inputs.csa, lctx.dsv4.csa_plan,
                layer.attn_comp_wkv, layer.attn_comp_wgate,
                layer.attn_comp_ape, layer.attn_comp_norm,
                lctx.dsv4.cache.csa_state_kv[il], lctx.dsv4.cache.csa_state_score[il], lctx.dsv4.cache.csa_k[il],
                append_csa_state, append_csa_score,
                n_embd_head, il, false, "csa", gf, false, &csa_latent);

        if (layer.indexer_attn_k != nullptr &&
                lctx.dsv4.inputs.csa.state_write_idxs_lid != nullptr &&
                lctx.dsv4.csa_plan.state_write_idxs_lid.size() > 0) {
            ggml_tensor * ik = dsv4_build_v41_index_key(ctx0, llm, csa_latent,
                    lctx.dsv4.inputs.csa.state_write_pos,
                    lctx.dsv4.inputs.csa.state_write_idxs_lid, il);
            ggml_build_forward_expand(gf, ik);
            cb(ik, "v41_index_k_write", il);
        }

    }

    const bool is_hca_kv_source = hparams.dsv41_is_kv_source(il);
    if (ratio == hparams.dsv4_hca_ratio && is_hca_kv_source &&
            lctx.dsv4.inputs.hca.state_pos != nullptr &&
            lctx.dsv4.hca_plan.state_pos.size() > 0) {

        ggml_tensor * hca_latent = nullptr;
        ds4_build_comp(cur, llm, ctx0, lctx.dsv4.inputs.hca, lctx.dsv4.hca_plan,
                layer.attn_comp_wkv, layer.attn_comp_wgate,
                layer.attn_comp_ape, layer.attn_comp_norm,
                lctx.dsv4.cache.hca_state_kv[il], lctx.dsv4.cache.hca_state_score[il], lctx.dsv4.cache.hca_k[il],
                nullptr, nullptr,
                n_embd_head, il, false, "hca", gf, true, &hca_latent);

        if (layer.indexer_attn_k != nullptr &&
                lctx.dsv4.inputs.hca.state_write_idxs_lid != nullptr &&
                lctx.dsv4.hca_plan.state_write_idxs_lid.size() > 0) {
            ggml_tensor * ik = dsv4_build_v41_index_key(ctx0, llm, hca_latent,
                    lctx.dsv4.inputs.hca.state_write_pos,
                    lctx.dsv4.inputs.hca.state_write_idxs_lid, il);
            ggml_build_forward_expand(gf, ik);
            cb(ik, "v41_index_k_write", il);
        }

    }

    const bool raw_compacted = kv_self.is_compacted((int) il);
    const bool plan_addressed = !kv_self.any_compacted() || raw_compacted;

    ggml_tensor * raw_k_write = nullptr;
    if (plan_addressed && hparams.n_head_kv(il) == 1 && lctx.dsv4.inputs.raw_k_write_idxs != nullptr) {
        raw_k_write = dsv4_raw_cpy_k(&lctx, ctx0, kv_self.k_l[il], kv,
                lctx.dsv4.inputs.raw_k_write_src_idxs, lctx.dsv4.inputs.raw_k_write_idxs, gf, n_embd_head, cb, il);
        if (raw_k_write != nullptr) {
            cb(raw_k_write, "dsv4_raw_k_write", il);
        }
    }
    if (raw_k_write == nullptr) {
        llm.llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, kv, nullptr, n_tokens, llm.kv_head, cb, il);
    }
    if (il < (int64_t) kv_self.v_l.size() && kv_self.v_l[il] != nullptr) {
        llm.llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, nullptr, kv, n_tokens, llm.kv_head, cb, il);
    }

    ggml_tensor * const read_idxs = plan_addressed ? lctx.dsv4.inputs.raw_k_read_idxs : nullptr;

    ggml_tensor * raw_k = nullptr;
    ggml_tensor * raw_mask = nullptr;
    if (raw_compacted) {
        GGML_ASSERT(hparams.n_head_kv(il) == 1 && KQ_mask_swa_win != nullptr && lctx.swa_window_view.active);
        const size_t row_size = ggml_row_size(kv_self.k_l[il]->type, n_embd_head);
        raw_k = ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head, 1, lctx.swa_window_view.w_view,
                row_size, row_size, row_size*(size_t) lctx.swa_window_view.win_off);
        raw_mask = KQ_mask_swa_win;
    } else if (hparams.n_head_kv(il) == 1 && read_idxs != nullptr) {
        raw_k = dsv4_raw_get_k(&lctx, ctx0, kv_self.k_l[il], read_idxs, n_embd_head, cb, il);
    }
    if (raw_k == nullptr) {
        raw_k = ggml_view_3d(ctx0, kv_self.k_l[il],
                n_embd_head, hparams.n_head_kv(il), n_kv,
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head),
                ggml_row_size(kv_self.k_l[il]->type, n_embd_head) * hparams.n_head_kv(il),
                0);
    }
    cb(raw_k, "raw_k", il);

    const int64_t raw_kq_n_kv = raw_compacted ? lctx.swa_window_view.w_view
        : raw_k != nullptr && read_idxs != nullptr && lctx.dsv4.raw.n_kv > 0
        ? lctx.dsv4.raw.n_kv
        : (raw_k != nullptr ? raw_k->ne[2] * raw_k->ne[3] : n_kv);
    const int64_t raw_attn_n_kv = raw_compacted ? lctx.swa_window_view.w_view
        : raw_kq_n_kv > 0 ? std::max<int64_t>(256, GGML_PAD(raw_kq_n_kv, 256)) : raw_kq_n_kv;
    if (!raw_compacted && raw_k->ne[3] == 1) {
        raw_k = dsv4_pad_raw_k_to(ctx0, raw_k, raw_attn_n_kv);
    }
    if (raw_mask == nullptr) {
        raw_mask = dsv4_build_raw_mask_view(ctx0, KQ_mask,
                read_idxs, raw_kq_n_kv, n_tokens, raw_k->ne[3], cb, il);
        cb(raw_mask, "raw_mask_view", il);
        raw_mask = dsv4_pad_mask_tokens(ctx0, raw_mask, n_tokens);
        raw_mask = dsv4_pad_raw_mask_to(ctx0, raw_mask, raw_attn_n_kv, n_tokens);
    }
    cb(raw_mask, "dsv4_raw_mask_padded", il);
    ggml_tensor * attn = nullptr;

    if (hparams.n_swa > 0 && !raw_compacted) {
        constexpr int k_fa_chunk = 256;
        int n_swa = hparams.n_swa;
        int ntokens = std::max(k_fa_chunk, int(q->ne[2]));
        int nton = k_fa_chunk*((ntokens + n_swa + k_fa_chunk - 1)/k_fa_chunk);
        int first = raw_k->ne[2] - nton;
        const int64_t n_valid = lctx.dsv4.raw.n_kv > 0 ? lctx.dsv4.raw.n_kv : raw_k->ne[2];

        const int64_t first_min = std::max<int64_t>(0, n_valid - (int64_t) q->ne[2] - (int64_t) n_swa);
        if (first > first_min) { first = (int) first_min; }
        if (first > 0) {
            raw_k = ggml_view_4d(ctx0, raw_k, raw_k->ne[0], raw_k->ne[1], nton, raw_k->ne[3],
                    raw_k->nb[1], raw_k->nb[2], raw_k->nb[3], raw_k->nb[2]*first);
            raw_mask = ggml_view_4d(ctx0, raw_mask, nton, raw_mask->ne[1], raw_mask->ne[2], raw_mask->ne[3],
                    raw_mask->nb[1], raw_mask->nb[2], raw_mask->nb[3], raw_mask->nb[0]*first);
            raw_mask = ggml_cont(ctx0, raw_mask);
        }
    }

    auto build_the_attn = [&] (ggml_tensor * raw_k, ggml_tensor * raw_mask, ggml_tensor * extra_mask,
            ggml_tensor * cache, const auto & extra_ctx,
            const std::string & tag, int n_swa_eff) {
        auto n_stream = std::max<uint32_t>(1, lctx.dsv4.cache.n_stream);
        auto extra_k = cache;
        if (extra_k->ne[1] > 1) {
            extra_k = dsv4_comp_get_k(ctx0, cache, extra_ctx, n_embd_head, cache->ne[1]/n_stream);
            cb(extra_k, "extra_k", il);
        }
        if (cparams.flash_attn) {
            extra_mask = dsv4_pad_mask_tokens(ctx0, extra_mask, n_tokens);
        }
        raw_k = dsv4_repeat_streams(ctx0, raw_k, extra_k->ne[3]);
        if (!cparams.flash_attn && !raw_compacted) {
            raw_mask = dsv4_build_raw_mask_view(ctx0, KQ_mask,
                    read_idxs, raw_kq_n_kv, n_tokens, extra_k->ne[3], cb, il);
            raw_mask = dsv4_pad_raw_mask_to(ctx0, raw_mask, raw_attn_n_kv, n_tokens);
        }
        if (!cparams.flash_attn && raw_compacted && raw_mask->ne[1] != n_tokens) {
            raw_mask = ggml_view_2d(ctx0, raw_mask, raw_mask->ne[0], n_tokens, raw_mask->nb[1], 0);
        }
        if (cparams.flash_attn && extra_mask->type != GGML_TYPE_F16) {
            extra_mask = ggml_cast(ctx0, extra_mask, GGML_TYPE_F16);
        }
        if (raw_mask->type != extra_mask->type) {
            raw_mask = ggml_cast(ctx0, raw_mask, extra_mask->type);
        }
        if (raw_k->type != extra_k->type) {
            extra_k = ggml_cast(ctx0, extra_k, raw_k->type);
            cb(extra_k, (tag + "_k_cast").c_str(), il);
        }
        ggml_tensor * k_all = ggml_concat(ctx0, raw_k, extra_k, 2);
        ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, extra_mask, 0);
        cb(extra_k, (tag + "_k").c_str(), il);
        cb(k_all, (tag + "_k_all").c_str(), il);
        cb(kq_mask, (tag + "_kq_mask").c_str(), il);

        auto attn = dsv4_build_attn(ctx0, hparams, cparams, q, k_all, k_all, kq_mask,
                model.layers[il].attn_sinks, kq_scale, cb, il, n_swa_eff, gf);
        return attn;
    };

    auto num_streams = [] (const auto & comp) {
        int n_stream = comp.sinfo.n_stream();
        return std::max(1, n_stream);
    };

    if (ratio == hparams.dsv4_csa_ratio &&
            lctx.dsv4.inputs.csa.kq_mask != nullptr &&
            lctx.dsv4.csa_plan.n_kv > 0 &&
            lctx.dsv4.lid_plan.n_kv > 0 &&
            !cparams.k_cache_hadamard) {
        auto csa_mask = lctx.dsv4.inputs.csa.kq_mask;
        const int32_t csa_src_il = hparams.dsv41_kv_source[il];
        auto csa_kv   = csa_src_il >= 0 ? lctx.dsv4.cache.csa_k[csa_src_il] : nullptr;
        if (hparams.indexer_top_k < lctx.dsv4.inputs.csa.kq_mask->ne[0]) {
            ggml_tensor * top_k = nullptr;
            if (!hparams.dsv41_is_index_source(il)) {
                GGML_ASSERT(topk_carry && *topk_carry && "DSV4: layer reuses a top-k no index source produced");
                top_k = *topk_carry;
            } else {
                top_k = dsv4_build_lid_top_k_v41(ctx0, llm, qr, cur, inp_pos, il, gf, cb,
                        nullptr, lctx.dsv4.inputs.csa.cand_pin, cand_carry);
                if (topk_carry) { *topk_carry = top_k; }
            }
            if (n_tokens == 1) {
                csa_kv = ggml_get_rows_ext(ctx0, csa_kv, top_k, true, false);
                csa_kv = ggml_reshape_3d(ctx0, csa_kv, csa_kv->ne[0], 1, csa_kv->ne[1]);
                cb(csa_kv, "csa_kv_getrows", il);
                csa_mask = ggml_get_rows_ext(ctx0, csa_mask, top_k, true, true);
            } else {
                csa_mask = build_top_k_mask(ctx0, dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.csa.kq_mask, nullptr,
                            lctx.dsv4.csa_plan.n_kv, n_tokens, num_streams(lctx.dsv4.csa_ctx), cb, il), top_k);
                cb(csa_mask, "csa_mask", il);
            }
        }
        int n_csa = hparams.n_swa + hparams.indexer_top_k;
        attn = build_the_attn(raw_k, raw_mask, csa_mask, csa_kv, lctx.dsv4.csa_ctx, "csa", n_csa);
        cb(attn, "attn_csa", il);
    } else if (ratio == hparams.dsv4_hca_ratio &&
            lctx.dsv4.inputs.hca.kq_mask != nullptr &&
            lctx.dsv4.hca_plan.n_kv > 0 &&
            std::any_of(lctx.dsv4.hca_plan.n_visible.begin(), lctx.dsv4.hca_plan.n_visible.end(),
                [](int32_t n_visible) { return n_visible > 0; }) &&
            !cparams.k_cache_hadamard) {
        ggml_tensor * hca_mask = dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.hca.kq_mask, nullptr,
                lctx.dsv4.hca_plan.n_kv, n_tokens, num_streams(lctx.dsv4.hca_ctx), cb, il);
        ggml_tensor * hca_kv = lctx.dsv4.cache.hca_k[il];
        const int32_t hca_src_il = hparams.dsv41_kv_source[il];
        if (hca_src_il >= 0) { hca_kv = lctx.dsv4.cache.hca_k[hca_src_il]; }
        if (hparams.indexer_top_k < lctx.dsv4.inputs.hca.kq_mask->ne[0]) {
            ggml_tensor * top_k = nullptr;
            if (!hparams.dsv41_is_index_source(il)) {
                GGML_ASSERT(topk_carry && *topk_carry && "DSV4: layer reuses a top-k no index source produced");
                top_k = *topk_carry;
            } else {
                top_k = dsv4_build_lid_top_k_v41(ctx0, llm, qr, cur, inp_pos, il, gf, cb, lctx.dsv4.inputs.hca.kq_mask,
                        lctx.dsv4.inputs.hca.cand_pin, cand_carry);
                if (topk_carry) { *topk_carry = top_k; }
            }
            if (n_tokens == 1) {
                hca_kv = ggml_get_rows_ext(ctx0, hca_kv, top_k, true, false);
                hca_kv = ggml_reshape_3d(ctx0, hca_kv, hca_kv->ne[0], 1, hca_kv->ne[1]);
                cb(hca_kv, "hca_kv_getrows", il);
                hca_mask = ggml_get_rows_ext(ctx0, hca_mask, top_k, true, true);
            } else {
                hca_mask = build_top_k_mask(ctx0, dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.hca.kq_mask, nullptr,
                            lctx.dsv4.hca_plan.n_kv, n_tokens, num_streams(lctx.dsv4.hca_ctx), cb, il), top_k);
                cb(hca_mask, "hca_mask", il);
            }
        }
        int n_hca = hparams.n_swa + hparams.indexer_top_k;
        attn = build_the_attn(raw_k, raw_mask, hca_mask, hca_kv, lctx.dsv4.hca_ctx, "hca", n_hca);
        cb(attn, "attn_hca", il);
    } else {
        attn = dsv4_build_attn(ctx0, hparams, cparams, q, raw_k, raw_k, raw_mask, model.layers[il].attn_sinks, kq_scale, cb, il, -1, gf);
        cb(attn, "attn_raw", il);
    }
    ggml_build_forward_expand(gf, attn);

    attn = ggml_reshape_3d(ctx0, attn, n_embd_head, n_head, n_tokens);
    attn = ggml_rope_ext_inplace(ctx0, attn, inp_pos, nullptr, n_embd_head_rope, llm.rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    attn->op = GGML_OP_ROPE_BACK;
    attn->op_params[15] = 1;
    cb(attn, "attn", il);

    const int64_t o_group_dim = layer.wo_a->ne[0];
    const int64_t n_groups = (n_head * n_embd_head) / o_group_dim;
    const int64_t o_lora_rank = layer.wo_b->ne[0] / n_groups;

    GGML_ASSERT((n_head * n_embd_head) % o_group_dim == 0);
    GGML_ASSERT(layer.wo_b->ne[0] % n_groups == 0);

    attn = ggml_reshape_3d(ctx0, attn, o_group_dim, n_groups, n_tokens);
    attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);

    ggml_tensor * oa = ggml_mul_mat(ctx0,
            ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups),
            attn);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    if (n_tokens == 1) {
        oa = ggml_reshape_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
    } else {
        oa = ggml_cont_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
    }

    cur = llm.llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
    cb(cur, "attn_out", il);

    inpL = llm.build_mhc_post(cur, post, residual, comb, llm.n_embd, hc, true);
    cb(inpL, "hc_attn_post", il);

    return inpL;

}

static ggml_tensor * dsv4_build_lid_top_k_v41(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il, ggml_cgraph * gf, const llm_build_cb & cb,
        ggml_tensor * kq_mask,
        ggml_tensor * cand_pin, ggml_tensor ** cand_carry) {
    const auto & hparams = llm.hparams;
    const auto & layer = llm.model.layers[il];
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot;
    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_tokens = cur->ne[1];
    const int hadamard_block = llama_model::hadamard_size((int) n_embd_indexer_head);

    if (kq_mask == nullptr) { kq_mask = llm.lctx.dsv4.inputs.csa.kq_mask; }
    GGML_ASSERT(kq_mask != nullptr);
    const int64_t n_lid = kq_mask->ne[0];

    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(hadamard_block > 0);
    GGML_ASSERT(n_embd_indexer_head % hadamard_block == 0);

    ggml_tensor * indexer_q = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_attn_q_b, qr);
    llm.cb(indexer_q, "lid_q", il);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);

    indexer_q = ggml_rope_ext_inplace(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            llm.rope_type, llm.n_ctx_orig,
            hparams.dsv4_compress_rope_base, llm.freq_scale,
            llm.ext_factor, dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    indexer_q->op_params[15] = 1;
    llm.cb(indexer_q, "indexer_q", il);

    const int32_t lid_src_il = llm.hparams.dsv41_kv_source[il];
    GGML_ASSERT(lid_src_il >= 0 && "DSV4: layer reads index keys with no source");
    ggml_tensor * lid_cache = llm.lctx.dsv4.cache.lid_k[lid_src_il];

    ggml_tensor * indexer_weights = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_proj, cur);
    llm.cb(indexer_weights, "lid_weights", il);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / std::sqrt(float(n_embd_indexer_head * n_indexer_head)));

    ggml_tensor * indexer_k = nullptr;
    indexer_k = dsv4_comp_get_k(ctx0,
            lid_cache,
            llm.lctx.dsv4.lid_ctx,
            n_embd_indexer_head,
            lid_cache->ne[1]/std::max<uint32_t>(1, llm.lctx.dsv4.cache.n_stream));
    GGML_ASSERT(indexer_k != nullptr);
    int64_t n_lid_use = n_lid;
    if (n_lid_use > indexer_k->ne[2]) {
        LLAMA_LOG_WARN("%s: LID size mismatch: group n_lid=%lld > lid cache read=%lld (clamping)\n",
                __func__, (long long) n_lid_use, (long long) indexer_k->ne[2]);
        n_lid_use = indexer_k->ne[2];
    }
    if (n_lid_use < indexer_k->ne[2]) {
        indexer_k = ggml_view_4d(ctx0, indexer_k,
                indexer_k->ne[0], indexer_k->ne[1], n_lid_use, indexer_k->ne[3],
                indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    }
    llm.cb(indexer_k, "lid_k", il);

    const int64_t n_stream = std::max<int64_t>(1, indexer_k->ne[3]);
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2] / n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3] / n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1] / n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2] / n_stream, indexer_weights->nb[3] / n_stream, 0);

    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    llm.cb(indexer_q, "lid_q_stream", il);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
    llm.cb(indexer_k, "lid_k_stream", il);

    ggml_tensor * lid_mask = dsv4_build_raw_mask_view(ctx0,
            kq_mask, nullptr, n_lid, n_tokens, n_stream, cb, il);

    const bool is_cand_source = cand_carry != nullptr && cand_pin != nullptr &&
            hparams.dsv4_candidate_source_layer >= 0 && il == hparams.dsv4_candidate_source_layer;
    if (cand_carry != nullptr && *cand_carry != nullptr && !is_cand_source) {
        lid_mask = ggml_add(ctx0, lid_mask, dsv4_expand_cand_keep(ctx0, *cand_carry, n_lid, cb, il));
        llm.cb(lid_mask, "lid_mask_cand", il);
    }

    const uint32_t n_top_k = (uint32_t) std::min<int64_t>(n_lid, hparams.indexer_top_k);
    if (!is_cand_source && llm.cparams.fused_idx_topk && n_lid > n_top_k) {
        if (ggml_tensor * selected = dsv4_build_lid_top_k_shared(ctx0,
                    indexer_k, indexer_q, indexer_weights, lid_mask, (int) n_top_k, cb)) {
            if (selected) {
                ggml_build_forward_expand(gf, selected);
                llm.cb(selected, "lid_top_k", il);
                return selected;
            }
        }
    }

    const int64_t n_tok_dim = indexer_q->ne[1];
    const int64_t idx_chunk = llama_dsv4_idx_score_chunk(n_lid, n_indexer_head, n_stream);
    const int64_t tok_chunk = idx_chunk > 0 && n_tok_dim > idx_chunk ? idx_chunk : n_tok_dim;

    const int64_t n_cand_blocks = is_cand_source ? cand_pin->ne[0] : 0;
    const int64_t cand_block    = n_cand_blocks > 0 ? n_lid/n_cand_blocks : 0;
    GGML_ASSERT(!is_cand_source || (n_cand_blocks > 0 && n_lid % n_cand_blocks == 0));

    auto dsv4_build_score_chunk = [&](int64_t c0, int64_t tc) {
        auto q_c = ggml_cont(ctx0, ggml_view_4d(ctx0, indexer_q,
                indexer_q->ne[0], tc, indexer_q->ne[2], indexer_q->ne[3],
                indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3], (size_t) c0*indexer_q->nb[1]));
        auto w_c = ggml_view_4d(ctx0, indexer_weights,
                indexer_weights->ne[0], tc, indexer_weights->ne[2], indexer_weights->ne[3],
                indexer_weights->nb[1], indexer_weights->nb[2], indexer_weights->nb[3],
                (size_t) c0*indexer_weights->nb[1]);
        auto kq_c = ggml_mul_mat(ctx0, indexer_k, q_c);
        kq_c = ggml_cont(ctx0, ggml_permute(ctx0, kq_c, 2, 1, 0, 3));
        auto sc_c = ggml_relu(ctx0, kq_c);
        sc_c = ggml_mul(ctx0, sc_c, w_c);
        sc_c = ggml_sum_rows(ctx0, sc_c);
        sc_c = ggml_cont(ctx0, ggml_permute(ctx0, sc_c, 2, 1, 0, 3));
        auto m_c = ggml_view_4d(ctx0, lid_mask,
                lid_mask->ne[0], tc, lid_mask->ne[2], lid_mask->ne[3],
                lid_mask->nb[1], lid_mask->nb[2], lid_mask->nb[3], (size_t) c0*lid_mask->nb[1]);
        return ggml_add(ctx0, sc_c, m_c);
    };
    ggml_tensor * top_k = nullptr;
    ggml_tensor * cand_block_score = nullptr;
    for (int64_t c0 = 0; c0 < n_tok_dim; c0 += tok_chunk) {
        const int64_t tc = std::min(tok_chunk, n_tok_dim - c0);
        auto sc_c = dsv4_build_score_chunk(c0, tc);
        llm.cb(sc_c, "lid_score_chunk", il);
        if (is_cand_source) {
            auto bs_c = ggml_pool_2d(ctx0, sc_c, GGML_OP_POOL_MAX, (int) cand_block, 1, (int) cand_block, 1, 0, 0);
            cand_block_score = cand_block_score == nullptr ? bs_c : ggml_concat(ctx0, cand_block_score, bs_c, 1);
        }
        auto tk_c = ggml_cont(ctx0, ggml_top_k(ctx0, sc_c, (int) n_top_k));
        top_k = top_k == nullptr ? tk_c : ggml_concat(ctx0, top_k, tk_c, 1);
    }

    if (is_cand_source) {
        *cand_carry = dsv4_build_candidate_mask(ctx0, cand_block_score, cand_pin,
                hparams.dsv4_candidate_topk_blocks, cb, il);
    }

    llm.cb(top_k, "lid_top_k", il);

    return top_k;
}

static ggml_tensor * dsv4_build_v41_index_key(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * latent,
        ggml_tensor * comp_pos,
        ggml_tensor * write_idxs,
        int il) {
    const auto & hparams = llm.hparams;
    const auto & layer = llm.model.layers[il];
    const int64_t n_embd_ik      = hparams.indexer_head_size;
    const int64_t n_embd_ik_rope = hparams.n_rot;
    const int64_t n_blocks       = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(layer.indexer_attn_k && layer.indexer_k_norm);
    GGML_ASSERT(n_embd_ik >= n_embd_ik_rope);
    GGML_ASSERT(n_blocks > 0);

    ggml_tensor * k = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_attn_k, latent);
    k = llm.llm_build_norm(ctx0, k, hparams, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, llm.cb, il);
    k = ggml_reshape_3d(ctx0, k, n_embd_ik, 1, n_blocks);
    llm.cb(k, "v41_index_k", il);
    k = ggml_rope_ext_inplace(ctx0, k, comp_pos, nullptr, n_embd_ik_rope, llm.rope_type, llm.n_ctx_orig,
            hparams.dsv4_compress_rope_base, llm.freq_scale, llm.ext_factor,
            dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    k->op_params[15] = 1;
    llm.cb(k, "v41_index_k_rot", il);

    ggml_tensor * k_2d = ggml_reshape_2d(ctx0, k, n_embd_ik, n_blocks);
    return dsv4_comp_cpy_k(ctx0, llm.lctx.dsv4.cache.lid_k[il], k_2d, write_idxs, n_embd_ik);
}

ggml_cgraph * llm_build_context::build_deepseek41() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t n_embd_head = hparams.n_embd_head_k(0);
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(0));
    GGML_ASSERT(n_embd_head_nope > 0);

    dsv4_new_i32_input(ctx0, &lctx.dsv4.inputs.raw_k_write_src_idxs, (int64_t) lctx.dsv4.raw.write_src_idxs.size(), "dsv4_raw_k_write_src_idxs");
    dsv4_new_i32_input(ctx0, &lctx.dsv4.inputs.raw_k_write_idxs, (int64_t) lctx.dsv4.raw.write_dst_idxs.size(), "dsv4_raw_k_write_idxs");
    dsv4_new_i32_input(ctx0, &lctx.dsv4.inputs.raw_k_read_idxs, (int64_t) lctx.dsv4.raw.read_dst_idxs.size(), "dsv4_raw_k_read_idxs");
    dsv4_build_plan_inputs(ctx0, lctx.dsv4.inputs.csa, lctx.dsv4.csa_plan, "dsv4_csa", n_tokens, true, lctx.cparams.flash_attn,
            hparams.dsv4_candidate_block_size, hparams.dsv4_candidate_topk_blocks);
    dsv4_build_plan_inputs(ctx0, lctx.dsv4.inputs.hca, lctx.dsv4.hca_plan, "dsv4_hca", n_tokens, true, lctx.cparams.flash_attn,
            hparams.dsv4_candidate_block_size, hparams.dsv4_candidate_topk_blocks);

    ggml_tensor * inp_pos = build_inp_pos();


    const int n_layer_end = n_layer - hparams.nextn_predict_layers;

    ggml_tensor * KQ_mask = nullptr;
    ggml_tensor * KQ_mask_swa_win = nullptr;
    if (kv_self.any_compacted()) {
        bool walked_compacted = false, walked_dense = false;
        for (int il = 0; il < n_layer_end; ++il) {
            (kv_self.is_compacted(il) ? walked_compacted : walked_dense) = true;
        }
        if (walked_compacted) {
            bool KQ_mask_swa_windowed = false;
            KQ_mask_swa_win = build_swa_mask_for_graph(hparams.n_swa, /* compacted = */ true, &KQ_mask_swa_windowed);
            GGML_ASSERT(KQ_mask_swa_windowed && KQ_mask_swa_win != nullptr);
        }
        if (walked_dense) {
            KQ_mask = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
        }
    } else {
        KQ_mask = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
    }

    ggml_tensor * append_csa_state = nullptr;
    ggml_tensor * append_csa_score = nullptr;

    ggml_tensor * inpL = nullptr;
    ggml_tensor * inp = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    ggml_tensor * topk_carry = nullptr;

    ggml_tensor * cand_carry_v = nullptr;
    ggml_tensor ** cand_carry = &cand_carry_v;

    ggml_tensor * hc_pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(hc_pre_mix, "hc_pre_init", -1);


    for (int il = 0; il < n_layer_end; ++il) {
        auto & layer = model.layers[il];

        if (layer.engram_embd != nullptr) {
            const int eg = hparams.engram_index(il);
            GGML_ASSERT(eg >= 0 && "engram tensor on a layer the metadata does not list");
            inpL = ds4_build_engram(ctx0, *this, inpL, il, eg);
        }

        if (lctx.dflash.capture) {
            ggml_tensor * capture = dsv4_hc_mean_for_capture(ctx0, inpL);
            cb(capture, "l_out", il);
            ggml_build_forward_expand(gf, capture);
        } else {
            cb(inpL, "l_out", il);
        }

        ggml_tensor * hc_attn_pre = nullptr;
        inpL = ds4_attention_v41(gf, ctx0, *this, inpL,
                &append_csa_state, &append_csa_score,
                inp_pos, KQ_mask, KQ_mask_swa_win, il, &topk_carry, hc_pre_mix, &hc_attn_pre, cand_carry);

        // ---- MoE FFN ----
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;
        ggml_tensor * cur = build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, cb, il, hc_attn_pre, &hc_pre_mix);
        cb(cur, "hc_ffn_pre", il);
        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        const bool is_media = lctx.inp_embd != nullptr;
        ggml_tensor * exp_probs_b = (is_media && layer.ffn_exp_probs_b_vl != nullptr)
            ? layer.ffn_exp_probs_b_vl
            : layer.ffn_exp_probs_b;
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp, nullptr,
                layer.ffn_up_exps, nullptr,
                layer.ffn_gate_exps, nullptr,
                layer.ffn_down_exps, nullptr,
                exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                true, hparams.expert_weights_scale,
                (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, false, layer.ffn_up_gate_exps, nullptr, nullptr, nullptr,
                selected_experts);
        ggml_build_forward_expand(gf, moe_out);
        ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hc, true);
        cb(inpL, "hc_ffn_post", il);
    }
    ggml_tensor * inp_out_ids = nullptr;
    if (n_outputs != n_tokens) {
        inp_out_ids = build_inp_out_ids();
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    GGML_ASSERT(hc_pre_mix != nullptr && "DEEPSEEK41 hc: no mix carried to the final collapse");
    ggml_tensor * head_mix = hc_pre_mix;
    if (head_mix->ne[1] != inpL->ne[2]) {
        GGML_ASSERT(inp_out_ids != nullptr);
        head_mix = ggml_get_rows(ctx0, head_mix, inp_out_ids);
    }
    ggml_tensor * out = build_mhc_weighted_sum(inpL, head_mix, n_embd, hc);
    cb(out, "hc_head", -1);
    if (model.output_norm != nullptr) {
        out = llm_build_norm(ctx0, out, hparams, model.output_norm, nullptr, LLM_NORM_RMS, cb, -1);
        cb(out, "result_norm", -1);
    }
    out = build_output(lctx, ctx0, out, model.output, nullptr, cb);
    cb(out, "result_output", -1);
    ggml_build_forward_expand(gf, out);

    return gf;
}

static ggml_tensor * dsv4_build_candidate_mask(
        ggml_context * ctx0,
        ggml_tensor * block_score,
        ggml_tensor * cand_pin,
        uint32_t cand_topk_blocks,
        const llm_build_cb & cb, int il) {
    if (cand_pin == nullptr || block_score == nullptr) {
        return nullptr;
    }

    const int64_t n_blocks = cand_pin->ne[0];

    GGML_ASSERT(n_blocks > 0 && block_score->ne[0] == n_blocks);

    ggml_tensor * bs = block_score;
    cb(bs, "cand_block_score", il);

    bs = ggml_add(ctx0, bs, cand_pin);
    cb(bs, "cand_block_score_pin", il);

    const int64_t k = std::min<int64_t>((int64_t) cand_topk_blocks, n_blocks);
    ggml_tensor * top = ggml_cont(ctx0, ggml_top_k(ctx0, bs, (int) k));
    cb(top, "cand_block_top_k", il);

    ggml_tensor * keep = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blocks, bs->ne[1], bs->ne[3]);
    keep = ggml_fill(ctx0, keep, -INFINITY);

    ggml_tensor * top3 = ggml_view_4d(ctx0, top, top->ne[0], top->ne[1], top->ne[3], 1,
            top->nb[1], top->nb[2], top->ne[3]*top->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top3->ne[0], top3->ne[1], top3->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    keep = ggml_set_rows(ctx0, keep, zeros, top3);
    keep = ggml_view_4d(ctx0, keep, keep->ne[1], keep->ne[2], 1, keep->ne[3],
            keep->nb[2], keep->nb[3], keep->nb[3], 0);
    cb(keep, "cand_keep", il);

    return keep;
}

static ggml_tensor * dsv4_expand_cand_keep(
        ggml_context * ctx0,
        ggml_tensor * block_keep,
        int64_t n_pos,
        const llm_build_cb & cb, int il) {
    const int64_t n_blocks = block_keep->ne[0];
    GGML_ASSERT(n_blocks > 0 && n_pos % n_blocks == 0);
    const int64_t block = n_pos/n_blocks;

    ggml_tensor * keep = ggml_reshape_4d(ctx0, block_keep, 1, n_blocks, block_keep->ne[1], block_keep->ne[3]);
    keep = ggml_repeat_4d(ctx0, keep, block, n_blocks, keep->ne[2], keep->ne[3]);
    keep = ggml_reshape_4d(ctx0, keep, n_pos, keep->ne[2], 1, keep->ne[3]);
    cb(keep, "cand_mask", il);

    return keep;
}
