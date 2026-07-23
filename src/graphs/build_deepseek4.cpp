#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-build-context.h"
#include "../llama-dsv4.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

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
        bool create_mask = true, bool flash_attn = true) {
    //printf("%s(%s): n_tokens = %ld\n", __func__, tag, n_tokens);
    dsv4_new_i32_input(ctx, &inputs.state_pos, (int64_t) plan.state_pos.size(), (std::string(tag) + "_state_pos").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_persist_src_idxs, (int64_t) plan.state_persist_src_idxs.size(), (std::string(tag) + "_persist_src").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_persist_dst_idxs, (int64_t) plan.state_persist_dst_idxs.size(), (std::string(tag) + "_persist_dst").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_read_idxs, (int64_t) plan.state_read_idxs.size(), (std::string(tag) + "_state_read").c_str());
    dsv4_new_i64_input(ctx, &inputs.state_write_idxs, (int64_t) plan.state_write_idxs.size(), (std::string(tag) + "_state_write").c_str());
    dsv4_new_i32_input(ctx, &inputs.state_write_pos, (int64_t) plan.state_write_pos.size(), (std::string(tag) + "_write_pos").c_str());
    if (create_mask) {
        auto type = flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
        dsv4_new_mask_input(ctx, &inputs.kq_mask, std::max<int64_t>(1, plan.n_kv), n_tokens, (std::string(tag) + "_kq_mask").c_str(), type);
    } else {
        inputs.kq_mask = nullptr;
    }
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);
    return dsv4_concat_named(ctx, t, row, 1, "dsv4_append_zero_row");
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
        //printf("%s(Oops): %d, %d, %d\n", __func__, n_stream <= 0, n_tokens % n_stream != 0, raw_k_read_idxs->ne[0] < n_rows_stream*n_stream);
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

    //printf("Oops: padding KV cache\n");

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

static ggml_tensor * dsv4_cache_read_f32(ggml_context * ctx, ggml_tensor * t) {
    return t != nullptr && ggml_is_quantized(t->type) ? ggml_cast(ctx, t, GGML_TYPE_F32) : t;
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
        return dsv4_cache_read_f32(ctx, ggml_reshape_4d(ctx, dsv4_cache_view_3d(ctx, cache, n_embd_head, n_kv), n_embd_head, 1, n_kv, 1));
    }

    return dsv4_cache_read_f32(ctx, dsv4_cache_stream_view_4d(ctx, cache, n_embd_head, n_kv, kv_size, comp.sinfo.s0, (int64_t) comp.sinfo.n_stream()));
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

static ggml_tensor * dsv4_build_kq_zero_bias(
        ggml_context * ctx,
        const llama_cparams & cparams,
        ggml_tensor * kq_mask,
        int64_t n_head) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(n_head);

    if (!cparams.flash_attn || kq_mask->ne[3] == 1) {
        return nullptr;
    }

    // The zero-bias fallback is only needed for unified multi-stream KV.
    // The DSV4 cache/controller is non-unified, so keep the direct FA path.
    return nullptr;
}

static ggml_tensor * dsv4_build_attn(
        ggml_context * ctx,
        const llama_hparams & hparams,
        const llama_cparams & cparams,
        ggml_tensor  * q,
        ggml_tensor  * k,
        ggml_tensor  * v,
        ggml_tensor  * kq_b,
        ggml_tensor  * kq_mask,
        ggml_tensor  * sinks,
        float          kq_scale,
        const llm_build_cb & cb,
        int            il,
        int            n_compressed,
        ggml_cgraph  * gf) {
    const bool v_trans = v->nb[1] > v->nb[2];
    const int64_t n_stream = k->ne[3];

    if (!cparams.flash_attn && n_stream > 1) {
        GGML_ASSERT(kq_b == nullptr);
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
                    q_s, k_s, v_s, nullptr, mask_s, sinks, kq_scale, cb, il, n_compressed, gf);
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
            (!kv_unified || kq_mask->ne[3] == 1) &&
            kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

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
                //if (q->ne[1] == 1) {
                //    selected = ggml_view_1d(ctx, selected, selected->ne[0], 0);
                //    kq_mask = ggml_view_1d(ctx, kq_mask, kq_mask->ne[0], 0);
                //    kq_mask = ggml_reshape_2d(ctx, kq_mask, 1, kq_mask->ne[0]);
                //    kq_mask = ggml_get_rows(ctx, kq_mask, selected);
                //    kq_mask = ggml_reshape_1d(ctx, kq_mask, kq_mask->ne[1]);
                //    k = ggml_get_rows(ctx, k, selected);
                //    auto kq = ggml_mul_mat(ctx, k, q);
                //    if (kq_b != nullptr) {
                //        kq = ggml_add(ctx, kq, kq_b);
                //    }
                //    kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);
                //    ggml_soft_max_add_sinks(kq, sinks);
                //    v = ggml_cont(ctx, ggml_transpose(ctx, k));
                //    auto kqv = ggml_mul_mat(ctx, v, kq);
                //    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
                //    kqv = ggml_reshape_2d(ctx, kqv, kqv->ne[0]*kqv->ne[1], kqv->ne[2]*kqv->ne[3]);
                //    return kqv;
                //}
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

    if (kq_b != nullptr) {
        kq = ggml_add(ctx, kq, kq_b);
        cb(kq, "kq_plus_kq_b", il);
    }

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

static ggml_tensor * build_hc_sinkhorn(
        ggml_context * ctx0,
        const llama_hparams & hparams,
        ggml_tensor * comb) {
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);
    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
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
        const llm_build_cb & cb, int il) {
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

    ////ggml_tensor * mixes = llm.build_mhc_pre_projection(x, hc_fn, nullptr,
    ////        n_embd, hc, norm_rms_eps, false);

    ////printf("hc_scale: %ld x %ld x %ld x %ld, hc_base: %ld x %ld x %ld x %ld\n",
    ////        hc_scale->ne[0], hc_scale->ne[1], hc_scale->ne[2], hc_scale->ne[3],
    ////        hc_base->ne[0], hc_base->ne[1], hc_base->ne[2], hc_base->ne[3]);
    //ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    //ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);
    //ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);

    //ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    //ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);
    //ggml_tensor * base_comb = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    //ggml_tensor * pre = ggml_cont(ctx0, dsv4_view_2d(ctx0, mixes, hc, nt, 0));
    //pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    //pre = ggml_sigmoid(ctx0, pre);
    //pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);

    //auto post = ggml_cont(ctx0, dsv4_view_2d(ctx0, mixes, hc, nt, hc));
    //post = dsv4_hc_affine(ctx0, post, scale_post, base_post);
    //post = ggml_sigmoid(ctx0, post);
    //post = ggml_scale(ctx0, post, 2.0f);

    //auto comb = ggml_cont(ctx0, dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc));
    //comb = dsv4_hc_affine(ctx0, comb, scale_comb, base_comb);
    //comb = ggml_sinkhorn(ctx0, comb, hc, hparams.dsv4_hc_sinkhorn_iters, hparams.dsv4_hc_eps, false);
    ////*comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    ////*comb = build_hc_sinkhorn(ctx0, hparams, *comb);
    //printf("pre: %ld x %ld x %ld x %ld, post: %ld x %ld x %ld x %ld, comb: %ld x %ld x %ld x %ld\n",
    //        pre->ne[0], pre->ne[1], pre->ne[2], pre->ne[3], post->ne[0], post->ne[1], post->ne[2], post->ne[3],
    //        comb->ne[0], comb->ne[1], comb->ne[2], comb->ne[3]);

    *post_out = post;
    *comb_out = comb;

    return llm.build_mhc_weighted_sum(x, pre, n_embd, hc);
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

static ggml_tensor * build_hca_compressed_kv_from_state(
        ggml_context * ctx0,
        llm_build_context & llm,
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        int il) {
    const int64_t n_embd_head_rope = llm.hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs != nullptr);

    auto comp = ggml_ds4_comp(ctx0, kv_state, score_state, state_read_idxs, llama_context::dsv4_runtime::HCA_RATIO, 1);

    llm.cb(comp, "hca_comp_merge", il);

    comp = llm.llm_build_norm(ctx0, comp, llm.hparams, norm, nullptr, LLM_NORM_RMS, llm.cb, il);
    llm.cb(comp, "hca_comp_norm", il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));
    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, llm.rope_type, llm.n_ctx_orig,
            llm.hparams.dsv4_compress_rope_base, llm.freq_scale, llm.ext_factor,
            dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    llm.cb(comp, "hca_comp_out", il);

    return comp;
}

static ggml_tensor * build_overlap_compressed_kv_from_state(
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
        const char * tag) {
    const int64_t n_embd_head_rope = llm.hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs != nullptr);

    // TODO: remove this. With a specialized op we can store -1 into the index for negative positions
    //       and then set the appropriate values (0 or inf) in the kernel.
    kv_state = dsv4_append_zero_row(ctx0, kv_state, false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    ggml_tensor * comp = ggml_ds4_comp(ctx0, kv_state, score_state, state_read_idxs, ratio, 0);

    llm.cb(comp, tag, il);

    comp = llm.llm_build_norm(ctx0, comp, llm.hparams, norm, nullptr, LLM_NORM_RMS, llm.cb, il);
    llm.cb(comp, tag, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));
    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, llm.rope_type, llm.n_ctx_orig,
            llm.hparams.dsv4_compress_rope_base, llm.freq_scale, llm.ext_factor,
            dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
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

        ggml_tensor * cur = ggml_indexer_topk(ctx0, k, q, w, mask,
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
        int il, ggml_cgraph * gf, const llm_build_cb & cb) {
    const auto & hparams = llm.hparams;
    const auto & layer = llm.model.layers[il];
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot;
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_lid = llm.lctx.dsv4.lid_plan.n_kv;
    const int hadamard_block = llama_model::hadamard_size((int) n_embd_indexer_head);

    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(hadamard_block > 0);
    GGML_ASSERT(n_embd_indexer_head % hadamard_block == 0);

    ggml_tensor * indexer_q = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_attn_q_b, qr);
    llm.cb(indexer_q, "lid_q", il);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, n_tokens);

    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
            0);
    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
            ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));
    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
            llm.rope_type, llm.n_ctx_orig,
            hparams.dsv4_compress_rope_base, llm.freq_scale,
            llm.ext_factor, dsv4_rope_attn_factor(llm.freq_scale, llm.ext_factor), llm.beta_fast, llm.beta_slow);
    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    llm.cb(indexer_q, "indexer_q", il);
    GGML_ASSERT(indexer_q->ne[0] % hadamard_block == 0);
    indexer_q = ggml_hadamard(ctx0, indexer_q, hadamard_block);
    llm.cb(indexer_q, "lid_q_hadamard", il);

    ggml_tensor * indexer_weights = llm.llm_build_lora_mm(llm.lctx, ctx0, layer.indexer_proj, cur);
    llm.cb(indexer_weights, "lid_weights", il);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / std::sqrt(float(n_embd_indexer_head * n_indexer_head)));

    ggml_tensor * indexer_k = dsv4_comp_get_k(ctx0,
            llm.lctx.dsv4.cache.lid_k[il],
            llm.lctx.dsv4.lid_ctx,
            n_embd_indexer_head,
            llm.lctx.dsv4.cache.lid_k[il]->ne[1]/std::max<uint32_t>(1, llm.lctx.dsv4.cache.n_stream));
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

    GGML_ASSERT(llm.lctx.dsv4.inputs.csa.kq_mask != nullptr);
    ggml_tensor * lid_mask = dsv4_build_raw_mask_view(ctx0,
            llm.lctx.dsv4.inputs.csa.kq_mask, nullptr, n_lid, n_tokens, n_stream, cb, il);
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

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    if (lctx.cparams.mtp_op_type != MTP_OP_NONE) {
        GGML_ABORT("DeepSeek4 MTP execution is not implemented");
    }

    //printf("================================================================= %s\n", __func__);

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

    ggml_tensor * inp = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : build_inp_KQ_mask();
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * cur = build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, cb, il);
        cb(cur, "hc_attn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_a, cur);
        cb(qr, "qr", il);

        qr = llm_build_norm(ctx0, qr, hparams, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(qr, "qr_norm", il);

        const int64_t ratio = hparams.dsv4_compress_ratios[il];
        const bool use_compress_rope = ratio != 0;
        const float freq_base_l = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
        const float freq_scale_l = use_compress_rope ? freq_scale : 1.0f;
        const float ext_factor_l = use_compress_rope ? ext_factor : 0.0f;
        const float attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
        const float beta_fast_l = use_compress_rope ? beta_fast : 0.0f;
        const float beta_slow_l = use_compress_rope ? beta_slow : 0.0f;
        const int32_t n_ctx_orig_l = use_compress_rope ? n_ctx_orig : 0;

        ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_b, qr);
        cb(q, "q_b", il);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, hparams.f_norm_rms_eps);
        cb(q, "q_b", il);

        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head),
                ggml_row_size(q->type, n_embd_head) * n_head,
                0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head),
                ggml_row_size(q->type, n_embd_head) * n_head,
                ggml_row_size(q->type, n_embd_head_nope));
        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(q_pe, "q_pe", il);
        q = ggml_concat(ctx0, q_nope, q_pe, 0);
        cb(q, "q", il);

        ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, model.layers[il].wkv_latent, cur);
        cb(kv, "wkv", il);
        kv = llm_build_norm(ctx0, kv, hparams, model.layers[il].attn_kv_norm, nullptr, LLM_NORM_RMS, cb, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);
        cb(kv, "kv_norm", il);

        ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_tokens,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                0);
        ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_tokens,
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head),
                ggml_row_size(kv->type, n_embd_head_nope));
        kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(kv_pe, "kv_pe", il);
        kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
        cb(kv, "kv", il);

        if (cparams.k_cache_hadamard) {
            if (int block_size = lctx.model.hadamard_size_k(il); block_size > 0) {
                q = ggml_hadamard(ctx0, q, block_size);
                kv = ggml_hadamard(ctx0, kv, block_size);
                cb(q, "q_hadamard", il);
                cb(kv, "kv_hadamard", il);
            }
        }
        cb(kv, "dsv4_raw_k_before_write", il);
        const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

        ggml_tensor * hca_state_kv = nullptr;
        ggml_tensor * hca_state_score = nullptr;
        if (ratio == llama_context::dsv4_runtime::HCA_RATIO && lctx.dsv4.inputs.hca.state_pos != nullptr && lctx.dsv4.hca_plan.state_pos.size() > 0) {
            hca_state_kv = llm_build_lora_mm(lctx, ctx0, model.layers[il].attn_comp_wkv, cur);
            cb(hca_state_kv, "hca_state_kv", il);
            hca_state_score = llm_build_lora_mm(lctx, ctx0, model.layers[il].attn_comp_wgate, cur);
            cb(hca_state_score, "hca_state_score", il);
            ggml_tensor * ape_rows = ggml_get_rows(ctx0, model.layers[il].attn_comp_ape, lctx.dsv4.inputs.hca.state_pos);
            cb(ape_rows, "ape", il);
            hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
            cb(hca_state_kv, "hca_state_kv", il);
            cb(hca_state_score, "hca_state_score", il);
        }

        if (ratio == llama_context::dsv4_runtime::CSA_RATIO && lctx.dsv4.inputs.csa.state_pos != nullptr && lctx.dsv4.csa_plan.state_pos.size() > 0) {
            ggml_tensor * csa_state_kv = llm_build_lora_mm(lctx, ctx0, model.layers[il].attn_comp_wkv, cur);
            cb(csa_state_kv, "csa_state_kv", il);
            ggml_tensor * csa_state_score = llm_build_lora_mm(lctx, ctx0, model.layers[il].attn_comp_wgate, cur);
            cb(csa_state_score, "csa_state_score", il);
            ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, model.layers[il].attn_comp_ape, lctx.dsv4.inputs.csa.state_pos);
            cb(csa_ape_rows, "csa_ape", il);
            csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
            ggml_tensor * csa_dep = nullptr;

            if (lctx.dsv4.inputs.csa.state_write_idxs != nullptr && lctx.dsv4.csa_plan.state_write_idxs.size() > 0) {
                ggml_tensor * csa_source_kv = dsv4_concat_named(ctx0, lctx.dsv4.cache.csa_state_kv[il], csa_state_kv, 1, "dsv4_csa_source_kv");
                ggml_tensor * csa_source_score = dsv4_concat_named(ctx0, lctx.dsv4.cache.csa_state_score[il], csa_state_score, 1, "dsv4_csa_source_score");
                ggml_tensor * csa_comp = build_overlap_compressed_kv_from_state(
                        ctx0, *this,
                        csa_source_kv, csa_source_score,
                        lctx.dsv4.inputs.csa.state_read_idxs,
                        lctx.dsv4.inputs.csa.state_write_pos,
                        model.layers[il].attn_comp_norm,
                        llama_context::dsv4_runtime::CSA_RATIO,
                        n_embd_head,
                        il,
                        "csa_state_compress");
                ggml_tensor * csa_comp_2d = ggml_reshape_2d(ctx0, csa_comp, n_embd_head, lctx.dsv4.inputs.csa.state_write_idxs->ne[0]);
                ggml_tensor * csa_write = dsv4_comp_cpy_k(ctx0, lctx.dsv4.cache.csa_k[il], csa_comp_2d, lctx.dsv4.inputs.csa.state_write_idxs, n_embd_head);
                ggml_build_forward_expand(gf, csa_write);
                cb(csa_write, "dsv4_csa_k_write", il);
                csa_dep = csa_comp;
            }

            if (csa_dep) {
                ggml_build_forward_expand(gf, csa_dep);
            }
            ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, lctx.dsv4.inputs.csa.state_persist_src_idxs);
            cb(csa_persist_kv, "csa_persist_kv", il);
            ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, lctx.dsv4.inputs.csa.state_persist_src_idxs);
            cb(csa_persist_score, "csa_persist_score", il);
            ggml_tensor * csa_state_kv_write = dsv4_comp_state_cpy(ctx0, lctx.dsv4.cache.csa_state_kv[il], csa_persist_kv, lctx.dsv4.inputs.csa.state_persist_dst_idxs);
            ggml_tensor * csa_state_score_write = dsv4_comp_state_cpy(ctx0, lctx.dsv4.cache.csa_state_score[il], csa_persist_score, lctx.dsv4.inputs.csa.state_persist_dst_idxs);
            ggml_build_forward_expand(gf, csa_state_kv_write);
            ggml_build_forward_expand(gf, csa_state_score_write);
            cb(csa_state_kv_write, "dsv4_csa_k_state_persist", il);
            cb(csa_state_score_write, "dsv4_csa_score_state_persist", il);

            ggml_tensor * lid_state_kv = llm_build_lora_mm(lctx, ctx0, model.layers[il].indexer_comp_wkv, cur);
            cb(lid_state_kv, "lid_state_kv", il);
            ggml_tensor * lid_state_score = llm_build_lora_mm(lctx, ctx0, model.layers[il].indexer_comp_wgate, cur);
            cb(lid_state_score, "lid_state_score", il);
            ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, model.layers[il].indexer_comp_ape, lctx.dsv4.inputs.lid.state_pos);
            cb(lid_ape_rows, "lid_ape", il);
            lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
            ggml_tensor * lid_dep = nullptr;

            if (lctx.dsv4.inputs.lid.state_write_idxs != nullptr && lctx.dsv4.lid_plan.state_write_idxs.size() > 0) {
                ggml_tensor * lid_source_kv = dsv4_concat_named(ctx0, lctx.dsv4.cache.lid_state_kv[il], lid_state_kv, 1, "dsv4_lid_source_kv");
                ggml_tensor * lid_source_score = dsv4_concat_named(ctx0, lctx.dsv4.cache.lid_state_score[il], lid_state_score, 1, "dsv4_lid_source_score");
                ggml_tensor * lid_comp = build_overlap_compressed_kv_from_state(
                        ctx0, *this,
                        lid_source_kv, lid_source_score,
                        lctx.dsv4.inputs.lid.state_read_idxs,
                        lctx.dsv4.inputs.lid.state_write_pos,
                        model.layers[il].indexer_comp_norm,
                        llama_context::dsv4_runtime::CSA_RATIO,
                        hparams.indexer_head_size,
                        il,
                        "lid_state_compress");
                const int hadamard_block = llama_model::hadamard_size((int) hparams.indexer_head_size);
                GGML_ASSERT(hadamard_block > 0);
                GGML_ASSERT(lid_comp->ne[0] % hadamard_block == 0);
                lid_comp = ggml_hadamard(ctx0, lid_comp, hadamard_block);
                cb(lid_comp, "lid_state_compress_hadamard", il);
                ggml_tensor * lid_comp_2d = ggml_reshape_2d(ctx0, lid_comp, hparams.indexer_head_size, lctx.dsv4.inputs.lid.state_write_idxs->ne[0]);
                ggml_tensor * lid_write = dsv4_comp_cpy_k(ctx0, lctx.dsv4.cache.lid_k[il], lid_comp_2d, lctx.dsv4.inputs.lid.state_write_idxs, hparams.indexer_head_size);
                ggml_build_forward_expand(gf, lid_write);
                cb(lid_write, "dsv4_lid_k_write", il);
                lid_dep = lid_comp;
            }

            if (lid_dep) {
                ggml_build_forward_expand(gf, lid_dep);
            }
            ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, lctx.dsv4.inputs.lid.state_persist_src_idxs);
            cb(lid_persist_kv, "lid_persist_kv", il);
            ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, lctx.dsv4.inputs.lid.state_persist_src_idxs);
            cb(lid_persist_score, "lid_persist_score", il);
            ggml_tensor * lid_state_kv_write = dsv4_comp_state_cpy(ctx0, lctx.dsv4.cache.lid_state_kv[il], lid_persist_kv, lctx.dsv4.inputs.lid.state_persist_dst_idxs);
            ggml_tensor * lid_state_score_write = dsv4_comp_state_cpy(ctx0, lctx.dsv4.cache.lid_state_score[il], lid_persist_score, lctx.dsv4.inputs.lid.state_persist_dst_idxs);
            ggml_build_forward_expand(gf, lid_state_kv_write);
            ggml_build_forward_expand(gf, lid_state_score_write);
            cb(lid_state_kv_write, "dsv4_lid_k_state_persist", il);
            cb(lid_state_score_write, "dsv4_lid_score_state_persist", il);
        }

        if (ratio == llama_context::dsv4_runtime::HCA_RATIO && hca_state_kv != nullptr && hca_state_score != nullptr) {
            ggml_tensor * hca_dep = nullptr;
            if (lctx.dsv4.inputs.hca.state_write_idxs != nullptr && lctx.dsv4.hca_plan.state_write_idxs.size() > 0) {
                ggml_tensor * hca_source_kv = dsv4_concat_named(ctx0, lctx.dsv4.cache.hca_state_kv[il], hca_state_kv, 1, "dsv4_hca_source_kv");
                ggml_tensor * hca_source_score = dsv4_concat_named(ctx0, lctx.dsv4.cache.hca_state_score[il], hca_state_score, 1, "dsv4_hca_source_score");
                ggml_tensor * hca_comp = build_hca_compressed_kv_from_state(
                        ctx0, *this,
                        hca_source_kv, hca_source_score,
                        lctx.dsv4.inputs.hca.state_read_idxs,
                        lctx.dsv4.inputs.hca.state_write_pos,
                        model.layers[il].attn_comp_norm,
                        n_embd_head,
                        il);
                ggml_tensor * hca_comp_2d = ggml_reshape_2d(ctx0, hca_comp, n_embd_head, lctx.dsv4.inputs.hca.state_write_idxs->ne[0]);
                ggml_tensor * hca_write = dsv4_comp_cpy_k(ctx0, lctx.dsv4.cache.hca_k[il], hca_comp_2d, lctx.dsv4.inputs.hca.state_write_idxs, n_embd_head);
                ggml_build_forward_expand(gf, hca_write);
                cb(hca_write, "dsv4_hca_k_write", il);
                hca_dep = hca_comp;
            }

            if (hca_dep) {
                ggml_build_forward_expand(gf, hca_dep);
            }
            ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, lctx.dsv4.inputs.hca.state_persist_src_idxs);
            ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, lctx.dsv4.inputs.hca.state_persist_src_idxs);
            cb(hca_persist_kv, "hca_persist_kv", il);
            cb(hca_persist_score, "hca_persist_score", il);
            ggml_tensor * hca_state_kv_write = dsv4_comp_state_cpy(ctx0, lctx.dsv4.cache.hca_state_kv[il], hca_persist_kv, lctx.dsv4.inputs.hca.state_persist_dst_idxs);
            ggml_tensor * hca_state_score_write = dsv4_comp_state_cpy(ctx0, lctx.dsv4.cache.hca_state_score[il], hca_persist_score, lctx.dsv4.inputs.hca.state_persist_dst_idxs);
            ggml_build_forward_expand(gf, hca_state_kv_write);
            ggml_build_forward_expand(gf, hca_state_score_write);
            cb(hca_state_kv_write, "dsv4_hca_k_state_persist", il);
            cb(hca_state_score_write, "dsv4_hca_score_state_persist", il);
        }

        ggml_tensor * raw_k_write = nullptr;
        if (hparams.n_head_kv(il) == 1 && lctx.dsv4.inputs.raw_k_write_idxs != nullptr) {
            raw_k_write = dsv4_raw_cpy_k(&lctx, ctx0, kv_self.k_l[il], kv, lctx.dsv4.inputs.raw_k_write_src_idxs, lctx.dsv4.inputs.raw_k_write_idxs, gf, n_embd_head, cb, il);
            if (raw_k_write != nullptr) {
                cb(raw_k_write, "dsv4_raw_k_write", il);
            }
        }
        if (raw_k_write == nullptr) {
            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, kv, nullptr, n_tokens, kv_head, cb, il);
        }
        if (il < (int64_t) kv_self.v_l.size() && kv_self.v_l[il] != nullptr) {
            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf, nullptr, kv, n_tokens, kv_head, cb, il);
        }

        ggml_tensor * raw_k = nullptr;
        if (hparams.n_head_kv(il) == 1 && lctx.dsv4.inputs.raw_k_read_idxs != nullptr) {
            raw_k = dsv4_raw_get_k(&lctx, ctx0, kv_self.k_l[il], lctx.dsv4.inputs.raw_k_read_idxs, n_embd_head, cb, il);
        }
        if (raw_k == nullptr) {
            raw_k = ggml_view_3d(ctx0, kv_self.k_l[il],
                    n_embd_head, hparams.n_head_kv(il), n_kv,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head) * hparams.n_head_kv(il),
                    0);
        }
        cb(raw_k, "raw_k", il);

        const int64_t raw_kq_n_kv = raw_k != nullptr && lctx.dsv4.raw.n_kv > 0
                ? lctx.dsv4.raw.n_kv
                : (raw_k != nullptr ? raw_k->ne[2] * raw_k->ne[3] : n_kv);
        const int64_t raw_attn_n_kv = raw_kq_n_kv > 0 ? std::max<int64_t>(256, GGML_PAD(raw_kq_n_kv, 256)) : raw_kq_n_kv;
        if (raw_k != nullptr && raw_k->ne[3] == 1) {
            raw_k = dsv4_pad_raw_k_to(ctx0, raw_k, raw_attn_n_kv);
        }
        ggml_tensor * raw_mask = dsv4_build_raw_mask_view(ctx0, KQ_mask,
                lctx.dsv4.inputs.raw_k_read_idxs, raw_kq_n_kv, n_tokens, raw_k->ne[3], cb, il);
        cb(raw_mask, "raw_mask_view", il);
        raw_mask = dsv4_pad_mask_tokens(ctx0, raw_mask, n_tokens);
        raw_mask = dsv4_pad_raw_mask_to(ctx0, raw_mask, raw_attn_n_kv, n_tokens);
        cb(raw_mask, "dsv4_raw_mask_padded", il);
        ggml_tensor * attn = nullptr;

        if (ratio == llama_context::dsv4_runtime::CSA_RATIO &&
                lctx.dsv4.inputs.csa.kq_mask != nullptr &&
                lctx.dsv4.csa_plan.n_kv > 0 &&
                lctx.dsv4.lid_plan.n_kv > 0 &&
                !cparams.k_cache_hadamard) {
            ggml_tensor * csa_k = dsv4_comp_get_k(ctx0,
                    lctx.dsv4.cache.csa_k[il],
                    lctx.dsv4.csa_ctx,
                    n_embd_head,
                    lctx.dsv4.cache.csa_k[il]->ne[1]/std::max<uint32_t>(1, lctx.dsv4.cache.n_stream));
            ggml_tensor * top_k = dsv4_build_lid_top_k(ctx0, *this, qr, cur, inp_pos, il, gf, cb);
            ggml_tensor * csa_mask = build_top_k_mask(ctx0,
                    dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.csa.kq_mask, nullptr,
                            lctx.dsv4.csa_plan.n_kv, n_tokens, csa_k->ne[3], cb, il),
                    top_k);
            cb(csa_mask, "csa_mask", il);
            const bool use_fattn = cparams.flash_attn;
            if (use_fattn) {
                csa_mask = dsv4_pad_mask_tokens(ctx0, csa_mask, n_tokens);
            }
            raw_k = dsv4_repeat_streams(ctx0, raw_k, csa_k->ne[3]);
            if (!use_fattn) {
                raw_mask = dsv4_build_raw_mask_view(ctx0, KQ_mask,
                        lctx.dsv4.inputs.raw_k_read_idxs, raw_kq_n_kv, n_tokens, csa_k->ne[3], cb, il);
                raw_mask = dsv4_pad_raw_mask_to(ctx0, raw_mask, raw_attn_n_kv, n_tokens);
            }
            if (use_fattn && csa_mask->type != GGML_TYPE_F16) {
                csa_mask = ggml_cast(ctx0, csa_mask, GGML_TYPE_F16);
            }
            if (raw_mask->type != csa_mask->type) {
                raw_mask = ggml_cast(ctx0, raw_mask, csa_mask->type);
            }
            {
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
                }
            }
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
            //printf("k_all: %ld x %ld x %ld x %ld, raw_k: %ld x %ld x %ld x %ld, csa_k = %ld x %ld x %ld x %ld, q = %ld x %ld x %ld x %ld\n",
            //        k_all->ne[0], k_all->ne[1], k_all->ne[2], k_all->ne[3],
            //        raw_k->ne[0], raw_k->ne[1], raw_k->ne[2], raw_k->ne[3],
            //        csa_k->ne[0], csa_k->ne[1], csa_k->ne[2], csa_k->ne[3],
            //        q->ne[0], q->ne[1], q->ne[2], q->ne[3]);
            ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
            ggml_tensor * kq_b = dsv4_build_kq_zero_bias(ctx0, cparams, kq_mask, q->ne[1]);
            cb(csa_k, "csa_k", il);
            cb(k_all, "csa_k_all", il);
            cb(kq_mask, "csa_kq_mask", il);
            int n_csa = hparams.n_swa + hparams.indexer_top_k;
            attn = dsv4_build_attn(ctx0, hparams, cparams, q, k_all, k_all, kq_b, kq_mask, model.layers[il].attn_sinks, kq_scale, cb, il, n_csa, gf);
            cb(attn, "attn_csa", il);
        } else if (ratio == llama_context::dsv4_runtime::HCA_RATIO &&
                lctx.dsv4.inputs.hca.kq_mask != nullptr &&
                lctx.dsv4.hca_plan.n_kv > 0 &&
                std::any_of(lctx.dsv4.hca_plan.n_visible.begin(), lctx.dsv4.hca_plan.n_visible.end(),
                        [](int32_t n_visible) { return n_visible > 0; }) &&
                !cparams.k_cache_hadamard) {
            ggml_tensor * hca_k = dsv4_comp_get_k(ctx0,
                    lctx.dsv4.cache.hca_k[il],
                    lctx.dsv4.hca_ctx,
                    n_embd_head,
                    lctx.dsv4.cache.hca_k[il]->ne[1]/std::max<uint32_t>(1, lctx.dsv4.cache.n_stream));
            const bool use_fattn = cparams.flash_attn;
            ggml_tensor * hca_mask = dsv4_build_raw_mask_view(ctx0, lctx.dsv4.inputs.hca.kq_mask, nullptr,
                    lctx.dsv4.hca_plan.n_kv, n_tokens, hca_k->ne[3], cb, il);
            hca_mask = dsv4_pad_mask_tokens(ctx0, hca_mask, n_tokens);
            if (use_fattn && hca_mask->type != GGML_TYPE_F16) {
                hca_mask = ggml_cast(ctx0, hca_mask, GGML_TYPE_F16);
            }
            raw_k = dsv4_repeat_streams(ctx0, raw_k, hca_k->ne[3]);
            if (raw_mask->type != hca_mask->type) {
                raw_mask = ggml_cast(ctx0, raw_mask, hca_mask->type);
            }
            {
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
                }
            }
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
            ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
            ggml_tensor * kq_b = dsv4_build_kq_zero_bias(ctx0, cparams, kq_mask, q->ne[1]);
            cb(hca_k, "hca_k", il);
            cb(k_all, "hca_k_all", il);
            cb(kq_mask, "hca_kq_mask", il);
            int n_hca = (n_kv + llama_context::dsv4_runtime::HCA_RATIO - 1)/llama_context::dsv4_runtime::HCA_RATIO;
            n_hca += hparams.n_swa;
            attn = dsv4_build_attn(ctx0, hparams, cparams, q, k_all, k_all, kq_b, kq_mask, model.layers[il].attn_sinks, kq_scale, cb, il, n_hca, gf);
            cb(attn, "attn_hca", il);
        } else {
            ggml_tensor * kq_b = dsv4_build_kq_zero_bias(ctx0, cparams, raw_mask, q->ne[1]);
            attn = dsv4_build_attn(ctx0, hparams, cparams, q, raw_k, raw_k, kq_b, raw_mask, model.layers[il].attn_sinks, kq_scale, cb, il, -1, gf);
            cb(attn, "attn_raw", il);
        }

        attn = ggml_reshape_3d(ctx0, attn, n_embd_head, n_head, n_tokens);
        ggml_tensor * attn_nope = ggml_view_3d(ctx0, attn, n_embd_head_nope, n_head, n_tokens,
                ggml_row_size(attn->type, n_embd_head),
                ggml_row_size(attn->type, n_embd_head) * n_head,
                0);
        ggml_tensor * attn_pe = ggml_view_3d(ctx0, attn, n_embd_head_rope, n_head, n_tokens,
                ggml_row_size(attn->type, n_embd_head),
                ggml_row_size(attn->type, n_embd_head) * n_head,
                ggml_row_size(attn->type, n_embd_head_nope));
        attn_pe = ggml_rope_back(ctx0, attn_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        cb(attn_pe, "attn_derope", il);
        attn = ggml_concat(ctx0, attn_nope, attn_pe, 0);
        cb(attn, "attn", il);

        const int64_t o_group_dim = model.layers[il].wo_a->ne[0];
        const int64_t n_groups = (n_head * n_embd_head) / o_group_dim;
        const int64_t o_lora_rank = model.layers[il].wo_b->ne[0] / n_groups;

        GGML_ASSERT((n_head * n_embd_head) % o_group_dim == 0);
        GGML_ASSERT(model.layers[il].wo_b->ne[0] % n_groups == 0);

        attn = ggml_reshape_3d(ctx0, attn, o_group_dim, n_groups, n_tokens);
        attn = ggml_permute(ctx0, attn, 0, 2, 1, 3);

        ggml_tensor * oa = ggml_mul_mat(ctx0,
                ggml_reshape_3d(ctx0, model.layers[il].wo_a, model.layers[il].wo_a->ne[0], o_lora_rank, n_groups),
                attn);
        cb(oa, "attn_wo_a", il);
        oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
        if (n_tokens == 1) {
            oa = ggml_reshape_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
        } else {
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank * n_groups, n_tokens);
        }

        cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_b, oa);
        cb(cur, "attn_out", il);

        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hc, true);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
                inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, cb, il);
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
            GGML_ASSERT(model.layers[il].ffn_up_gate_exps == nullptr &&
                    "merged DSV4 MoE gate tensors use an unsupported layout");
            ggml_tensor * selected_experts = nullptr;
            ggml_tensor * exp_probs_b = model.layers[il].ffn_exp_probs_b;
            if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
                selected_experts = ggml_get_rows(ctx0, model.layers[il].ffn_gate_tid2eid, lctx.inp_tokens);
                cb(selected_experts, "hashed_exps", il);
                exp_probs_b = nullptr;
            }

            // Hash layers carry an explicit fixed-width expert map. During
            // warmup the generic graph reserves all experts, but this input
            // still contains only the model's active expert IDs.
            const int64_t moe_n_expert_used = selected_experts != nullptr
                    ? selected_experts->ne[0]
                    : n_expert_used;

            const int64_t dsv4_n_stream = std::max<int64_t>(1, lctx.dsv4.csa_ctx.graph_n_stream);
            // Wide packed DSV4 fused/IQK MoE diverges above 1024 total tokens.
            // Evaluate each active stream independently to preserve packed parity.
            constexpr int64_t dsv4_moe_max_tokens = 1024;

            auto build_dsv4_moe = [&](ggml_tensor * moe_cur,
                                      ggml_tensor * moe_exp_probs_b,
                                      ggml_tensor * moe_selected_experts) {
                return llm_build_moe_ffn(ctx0, lctx, moe_cur,
                        model.layers[il].ffn_gate_inp,
                        nullptr,
                        model.layers[il].ffn_up_exps,
                        nullptr,
                        model.layers[il].ffn_gate_exps,
                        nullptr,
                        model.layers[il].ffn_down_exps,
                        nullptr,
                        moe_exp_probs_b,
                        n_expert, moe_n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        true, hparams.expert_weights_scale,
                        (enum llm_expert_gating_func_type) hparams.expert_gating_func,
                        cb, il, gf, false, model.layers[il].ffn_up_gate_exps, nullptr, nullptr, nullptr,
                        moe_selected_experts);
            };

            ggml_tensor * moe_out = nullptr;
            if (dsv4_n_stream > 1 && cur->ne[1] > dsv4_moe_max_tokens &&
                    cur->ne[1] % dsv4_n_stream == 0) {
                const int64_t n_tokens_stream = cur->ne[1]/dsv4_n_stream;
                auto stream_view = [&](ggml_tensor * tensor, int64_t stream) {
                    if (tensor == nullptr || tensor->ne[1] != cur->ne[1]) {
                        return tensor;
                    }
                    return ggml_view_2d(ctx0, tensor, tensor->ne[0], n_tokens_stream,
                            tensor->nb[1], stream*n_tokens_stream*tensor->nb[1]);
                };

                for (int64_t stream = 0; stream < dsv4_n_stream; ++stream) {
                    ggml_tensor * stream_result = build_dsv4_moe(
                            stream_view(cur, stream),
                            stream_view(exp_probs_b, stream),
                            stream_view(selected_experts, stream));
                    moe_out = moe_out == nullptr ? stream_result : ggml_concat(ctx0, moe_out, stream_result, 1);
                }
            } else {
                moe_out = build_dsv4_moe(cur, exp_probs_b, selected_experts);
            }
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
        }

        cb(cur, "ffn_out", il);

        inpL = build_mhc_post(cur, post, residual, comb, n_embd, hc, true);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_out", il);
    }

    if (n_outputs != n_tokens) {
        ggml_tensor * inp_out_ids = build_inp_out_ids();
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    ggml_tensor * out = build_hc_head(ctx0, *this, hparams, n_embd, hparams.f_norm_rms_eps,
            inpL,
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base);
    cb(out, "hc_head", -1);

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
