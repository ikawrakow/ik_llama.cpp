#include "llama-dsv41.h"

#include <random>

#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// DeepSeek-V4.1 (CSA2) host-side graph-input preparation.
// Cloned from llama-dsv4.cpp. M3 step 1: raw sliding-window K context. Step 2: engram
// n-gram cache. Step 3: the r2 (ratio 2) and r1 (ratio 1) compressed-KV plans, both
// non-overlap (type-1 ggml_ds4_comp semantics), the comp_k caches and f32 partial-group
// state rings at kv_source layers, and their reset. Step 4: the lid_k indexer-key cache
// at index-key owners (V4.1 has no LID plan — the indexer reuses the shared compressor
// latent, so no lid plans or state rings exist).

static bool dsv41_batch_has_coupled(const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static bool dsv41_token_has_seq(const llama_batch & batch, int32_t i, llama_seq_id seq_id) {
    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.seq_id[i] == nullptr) {
        return seq_id == 0;
    }

    for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
        if (batch.seq_id[i][s] == seq_id) {
            return true;
        }
    }

    return false;
}

static std::vector<llama_seq_id> dsv41_batch_unique_seq_ids(const llama_batch & batch) {
    std::vector<llama_seq_id> seq_ids;
    std::unordered_set<llama_seq_id> seen;

    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        seq_ids.push_back(0);
        return seq_ids;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            continue;
        }

        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = batch.seq_id[i][s];
            if (seen.insert(seq_id).second) {
                seq_ids.push_back(seq_id);
            }
        }
    }

    if (seq_ids.empty()) {
        seq_ids.push_back(0);
    }

    return seq_ids;
}

static int64_t dsv41_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (n_stream <= 1) {
        return 0;
    }

    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        LLAMA_LOG_ERROR("%s: DSV41 seq_id %d is outside stream range %u\n", __func__, seq_id, n_stream);
        return -1;
    }

    return (int64_t) seq_id*size;
}

static std::vector<llama_seq_id> dsv41_build_stream_seq_ids(
        const llama_batch & batch,
        uint32_t n_stream) {
    if (n_stream <= 1) {
        return { 0 };
    }

    const std::vector<llama_seq_id> seq_ids = dsv41_batch_unique_seq_ids(batch);
    if (seq_ids.size() <= 1 || dsv41_batch_has_coupled(batch)) {
        return { seq_ids.empty() ? 0 : seq_ids.front() };
    }

    return seq_ids;
}

static llama_context::dsv41_runtime::slot_info dsv41_build_raw_sinfo(
        const llama_batch & batch,
        uint32_t n_stream) {
    llama_context::dsv41_runtime::slot_info sinfo;

    const std::vector<llama_seq_id> seq_ids = dsv41_build_stream_seq_ids(batch, n_stream);
    const int64_t graph_n_stream = (int64_t) seq_ids.size();
    bool have_stream = false;

    sinfo.s0 = INT_MAX;
    sinfo.s1 = 0;
    sinfo.resize((size_t) std::max<int64_t>(1, graph_n_stream));
    for (int64_t s = 0; s < graph_n_stream; ++s) {
        const llama_seq_id seq_id = seq_ids[(size_t) s];
        const int64_t strm = dsv41_stream_offset(n_stream, seq_id, 1);
        if (strm < 0) {
            continue;
        }
        sinfo.strm[(size_t) s] = (llama_seq_id) strm;
        sinfo.idxs[(size_t) s].assign(1, 0);
        sinfo.s0 = std::min<int32_t>(sinfo.s0, (int32_t) strm);
        sinfo.s1 = std::max<int32_t>(sinfo.s1, (int32_t) strm);
        have_stream = true;
    }

    if (!have_stream) {
        sinfo.resize(1);
        sinfo.strm[0] = 0;
        sinfo.idxs[0].assign(1, 0);
        sinfo.s0 = 0;
        sinfo.s1 = 0;
    }

    if (n_stream > 1 && sinfo.s1 - sinfo.s0 + 1 != (int32_t) sinfo.n_stream()) {
        LLAMA_LOG_ERROR("%s: DSV41 raw streams are not contiguous in batch\n", __func__);
    }

    return sinfo;
}

static llama_context::dsv41_runtime::slot_info dsv41_build_raw_read_sinfo(
        const llama_context::dsv41_runtime::slot_info & sinfo_write,
        const llama_batch & batch,
        uint32_t n_stream) {
    if (!dsv41_batch_has_coupled(batch)) {
        return sinfo_write;
    }

    const llama_seq_id seq_id =
            (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_tokens > 0 && batch.n_seq_id[0] > 0 && batch.seq_id[0] != nullptr)
            ? batch.seq_id[0][0]
            : 0;
    const int64_t strm = dsv41_stream_offset(n_stream, seq_id, 1);
    if (strm < 0) {
        return {};
    }

    size_t i_stream = 0;
    for (; i_stream < sinfo_write.n_stream(); ++i_stream) {
        if ((int64_t) sinfo_write.strm[i_stream] == strm) {
            break;
        }
    }
    if (i_stream == sinfo_write.n_stream()) {
        LLAMA_LOG_ERROR("%s: DSV41 raw write stream not found for coupled read\n", __func__);
        return {};
    }

    llama_context::dsv41_runtime::slot_info sinfo;
    sinfo.resize(1);
    sinfo.strm[0] = sinfo_write.strm[i_stream];
    sinfo.idxs[0] = sinfo_write.idxs[i_stream];
    sinfo.s0 = (int32_t) strm;
    sinfo.s1 = sinfo.s0;

    return sinfo;
}

static bool dsv41_validate_batch_seq_ids(
        const llama_context & lctx,
        const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return true;
    }

    const uint32_t n_stream = std::max<uint32_t>(1, lctx.cparams.n_seq_max);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            LLAMA_LOG_ERROR("%s: DSV41 token %d is missing seq_id ownership\n", __func__, i);
            return false;
        }

        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = batch.seq_id[i][s];
            if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
                LLAMA_LOG_ERROR("%s: DSV41 token %d seq_id %d is outside n_seq_max=%u\n",
                        __func__, i, seq_id, n_stream);
                return false;
            }
        }
    }

    return true;
}

static bool dsv41_build_raw_context(
        const llama_context & lctx,
        const llama_batch & batch,
        llama_context::dsv41_runtime::raw_context & raw) {
    raw = {};
    const uint32_t n_stream = std::max<uint32_t>(1, lctx.cparams.n_seq_max);
    const std::vector<llama_seq_id> write_seq_ids = dsv41_build_stream_seq_ids(batch, n_stream);
    raw.sinfo_write = dsv41_build_raw_sinfo(batch, n_stream);
    raw.sinfo_read  = dsv41_build_raw_read_sinfo(raw.sinfo_write, batch, n_stream);
    raw.graph_n_stream = (int64_t) raw.sinfo_write.n_stream();
    std::vector<llama_seq_id> read_seq_ids = write_seq_ids;

    if (dsv41_batch_has_coupled(batch)) {
        const llama_seq_id coupled_seq_id =
                (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_tokens > 0 && batch.n_seq_id[0] > 0 && batch.seq_id[0] != nullptr)
                ? batch.seq_id[0][0]
                : 0;
        read_seq_ids.assign(1, coupled_seq_id);
    }

    if (batch.n_tokens <= 0) {
        return true;
    }

    const llama_kv_cache & kv = lctx.kv_self;
    if (kv.head + batch.n_tokens > (int32_t) kv.size) {
        LLAMA_LOG_ERROR("%s: DSV41 raw write slots [%d, %d) are outside kv cache size %u\n",
                __func__, kv.head, kv.head + batch.n_tokens, kv.size);
        return false;
    }

    raw.write_counts.push_back(batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t slot = kv.head + i;
        const llama_kv_cell & cell = kv.cells[(size_t) slot];

        if (batch.pos != nullptr && cell.pos != batch.pos[i]) {
            LLAMA_LOG_ERROR("%s: DSV41 raw write slot %d pos mismatch: cell=%d batch=%d\n",
                    __func__, slot, cell.pos, batch.pos[i]);
            return false;
        }

        raw.write_src_idxs.push_back(i);
        raw.write_dst_idxs.push_back(slot);
    }

    raw.n_kv = 0;

    for (size_t s = 0; s < raw.sinfo_read.n_stream(); ++s) {
        const llama_seq_id seq_id = read_seq_ids[s];
        raw.sinfo_read.idxs[s].clear();
        int32_t count = 0;
        for (uint32_t slot = 0; slot < kv.size; ++slot) {
            const llama_kv_cell & cell = kv.cells[slot];
            if (cell.is_empty() || cell.pos < 0) {
                continue;
            }
            if (!cell.has_seq_id(seq_id)) {
                continue;
            }
            raw.sinfo_read.idxs[s].push_back(slot);
            raw.read_dst_idxs.push_back((int32_t) slot);
            ++count;
        }
        raw.read_counts.push_back(count);
        raw.n_kv = std::max<int64_t>(raw.n_kv, count);
    }

    if (raw.read_counts.empty()) {
        raw.read_counts.push_back(0);
    }

    for (size_t s = 0; s < raw.sinfo_write.n_stream(); ++s) {
        const llama_seq_id seq_id = write_seq_ids[s];
        raw.sinfo_write.idxs[s].clear();
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (!dsv41_token_has_seq(batch, i, seq_id)) {
                continue;
            }
            raw.sinfo_write.idxs[s].push_back((uint32_t) (kv.head + i));
        }
    }

    if (raw.sinfo_write.n_stream() > 1) {
        std::vector<int32_t> write_src_idxs;
        std::vector<int32_t> write_dst_idxs;
        const size_t rows_per_stream = raw.sinfo_write.size();
        for (size_t s = 0; s < raw.sinfo_write.n_stream(); ++s) {
            if (raw.sinfo_write.idxs[s].size() != rows_per_stream) {
                LLAMA_LOG_ERROR("%s: DSV41 packed batch has unequal raw-write rows per stream\n", __func__);
                return false;
            }

            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (dsv41_token_has_seq(batch, i, write_seq_ids[s])) {
                    write_src_idxs.push_back(i);
                }
            }

            for (uint32_t slot : raw.sinfo_write.idxs[s]) {
                write_dst_idxs.push_back((int32_t) slot);
            }
        }

        raw.write_src_idxs = std::move(write_src_idxs);
        raw.write_dst_idxs = std::move(write_dst_idxs);
    }

    // The graph exposes a rectangular raw-key view. Repeat the last valid row
    // for shorter streams; the corresponding mask entries remain -INFINITY.
    // This preserves the logical visibility while allowing one get_rows op to
    // serve all streams.
    if (raw.n_kv > 0) {
        raw.read_dst_idxs.clear();
        const size_t read_rows = GGML_PAD((size_t) raw.n_kv, 256u);
        for (size_t s = 0; s < raw.sinfo_read.n_stream(); ++s) {
            const auto & rows = raw.sinfo_read.idxs[s];
            for (uint32_t slot : rows) {
                raw.read_dst_idxs.push_back((int32_t) slot);
            }

            const int32_t pad = rows.empty() ? 0 : (int32_t) rows.back();
            for (size_t i = rows.size(); i < read_rows; ++i) {
                raw.read_dst_idxs.push_back(pad);
            }
        }
    }

    return true;
}

// --- M3 step 3: compressed-KV (CSA2) plan machinery, cloned from llama-dsv4.cpp with
// CSA_RATIO=4/HCA_RATIO=128 replaced by R2=2/R1=1, both non-overlap, and no LID plan ---

static bool dsv41_cache_type_supported(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0;
}

static bool dsv41_validate_cache_type(ggml_type type, int64_t width, const char * name) {
    if (!dsv41_cache_type_supported(type)) {
        LLAMA_LOG_ERROR("%s: unsupported DSV41 %s cache type %s\n", __func__, name, ggml_type_name(type));
        return false;
    }
    if (ggml_is_quantized(type) && width % ggml_blck_size(type) != 0) {
        LLAMA_LOG_ERROR("%s: DSV41 %s cache width %d is not aligned to %d elements for %s\n",
                __func__, name, (int)width, (int)ggml_blck_size(type), ggml_type_name(type));
        return false;
    }
    return true;
}

static ggml_backend_buffer_type_t dsv41_layer_buft(const llama_context & lctx, int32_t il) {
    if (il >= 0 && il < (int32_t) lctx.model.buft_layer.size() && lctx.model.buft_layer[il].buft != nullptr) {
        return lctx.model.buft_layer[il].buft;
    }

    if (il >= 0 && il < (int32_t) lctx.model.layers.size()) {
        const ggml_tensor * ref = lctx.model.layers[il].attn_comp_wkv;
        if (ref == nullptr) {
            ref = lctx.model.layers[il].wq_a;
        }
        if (ref != nullptr && ref->buffer != nullptr) {
            return ggml_backend_buffer_get_type(ref->buffer);
        }
    }

    return llama_default_buffer_type_cpu(true);
}

static uint32_t dsv41_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

static void dsv41_batch_shape(
        const llama_batch & batch,
        uint32_t & n_seqs,
        uint32_t & n_seq_tokens) {
    n_seqs = 1;
    n_seq_tokens = (uint32_t) std::max(1, batch.n_tokens);

    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return;
    }

    std::map<llama_seq_id, uint32_t> counts;
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr) {
            return;
        }

        counts[batch.seq_id[i][0]]++;
    }

    if (counts.empty()) {
        return;
    }

    const uint32_t seq_tokens = counts.begin()->second;
    for (const auto & [_, count] : counts) {
        if (count != seq_tokens) {
            return;
        }
    }

    n_seqs = (uint32_t) counts.size();
    n_seq_tokens = std::max<uint32_t>(1, seq_tokens);
}

static int64_t dsv41_comp_graph_n_stream(const llama_batch & batch, uint32_t n_stream) {
    if (n_stream <= 1) {
        return 1;
    }

    const std::vector<llama_seq_id> seq_ids = dsv41_batch_unique_seq_ids(batch);
    if (seq_ids.size() <= 1 || dsv41_batch_has_coupled(batch)) {
        return 1;
    }

    return (int64_t) seq_ids.size();
}

static llama_context::dsv41_runtime::comp_context dsv41_build_comp_context(
        const llama_batch & batch,
        uint32_t n_stream,
        int64_t n_kv) {
    llama_context::dsv41_runtime::comp_context ctx;
    ctx.sinfo = dsv41_build_raw_sinfo(batch, n_stream);
    ctx.graph_n_stream = dsv41_comp_graph_n_stream(batch, n_stream);
    ctx.n_kv = n_kv;
    return ctx;
}

static llama_context::dsv41_runtime::comp_plan dsv41_build_reserve_comp_plan(
        const llama_batch & batch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    llama_context::dsv41_runtime::comp_plan plan;
    plan.n_visible.resize((size_t) batch.n_tokens, (int32_t) kv_size);
    plan.n_stream = dsv41_comp_graph_n_stream(batch, n_stream);
    plan.n_kv = kv_size;

    if (batch.n_tokens == 0) {
        return plan;
    }

    uint32_t n_seqs = 1;
    uint32_t n_seq_tokens = 1;
    dsv41_batch_shape(batch, n_seqs, n_seq_tokens);

    plan.n_visible.assign((size_t) batch.n_tokens, 0);

    const uint64_t n_blocks_u64 = (uint64_t) n_seqs*((n_seq_tokens + ratio - 1)/ratio);
    const size_t n_blocks = (size_t) std::max<uint64_t>(1, n_blocks_u64);
    GGML_ASSERT((uint64_t) n_blocks == std::max<uint64_t>(1, n_blocks_u64));
    const uint64_t state_rows = (uint64_t) state_size*(uint64_t) n_stream;
    const size_t n_persist = (size_t) std::min<uint64_t>((uint64_t) batch.n_tokens, state_rows);

    plan.state_pos.resize((size_t) batch.n_tokens);
    plan.state_delta_src_idxs.resize((size_t) batch.n_tokens);
    plan.state_delta_dst_idxs.resize((size_t) batch.n_tokens);
    plan.state_persist_src_idxs.resize(n_persist);
    plan.state_persist_dst_idxs.resize(n_persist);
    plan.state_read_idxs.resize((overlap ? 2u : 1u)*ratio*n_blocks);
    plan.state_write_idxs.resize(n_blocks);
    plan.state_write_pos.resize(n_blocks);

    return plan;
}

static bool dsv41_validate_comp_plan(
        const char * tag,
        const llama_batch & batch,
        const llama_context::dsv41_runtime::comp_plan & plan,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    const int64_t max_state_read_idx = (int64_t) state_size*n_stream + batch.n_tokens + (overlap ? 0 : -1);

    if (plan.n_visible.size() != (size_t) std::max(0, batch.n_tokens)) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan n_visible size mismatch: got=%zu expected=%d\n",
                __func__, tag, plan.n_visible.size(), std::max(0, batch.n_tokens));
        return false;
    }

    if (plan.state_pos.size() > (size_t) std::max(0, batch.n_tokens)) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan state_pos size %zu exceeds batch tokens %d\n",
                __func__, tag, plan.state_pos.size(), batch.n_tokens);
        return false;
    }

    if (plan.state_delta_src_idxs.size() != plan.state_pos.size() ||
        plan.state_delta_dst_idxs.size() != plan.state_pos.size()) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan delta src/dst sizes %zu/%zu mismatch state_pos size %zu\n",
                __func__, tag, plan.state_delta_src_idxs.size(), plan.state_delta_dst_idxs.size(), plan.state_pos.size());
        return false;
    }

    if (plan.state_persist_src_idxs.size() != plan.state_persist_dst_idxs.size()) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan persist src/dst sizes %zu/%zu mismatch\n",
                __func__, tag, plan.state_persist_src_idxs.size(), plan.state_persist_dst_idxs.size());
        return false;
    }

    if (plan.state_write_idxs.size() != plan.state_write_pos.size()) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan write idx/pos sizes %zu/%zu mismatch\n",
                __func__, tag, plan.state_write_idxs.size(), plan.state_write_pos.size());
        return false;
    }

    if (plan.state_read_idxs.size() != (overlap ? 2u : 1u)*(size_t) ratio*plan.state_write_idxs.size()) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan read size %zu does not match %u x writes %zu\n",
                __func__, tag, plan.state_read_idxs.size(), ratio, plan.state_write_idxs.size());
        return false;
    }

    for (size_t i = 0; i < plan.state_pos.size(); ++i) {
        const int64_t pos = plan.state_pos[i];
        if (pos < 0 || pos >= (int64_t) ratio) {
            LLAMA_LOG_ERROR("%s: DSV41 %s state_pos[%zu]=%lld outside ratio=%u\n",
                    __func__, tag, i, (long long) pos, ratio);
            return false;
        }

        const int64_t src = plan.state_delta_src_idxs[i];
        const int64_t dst = plan.state_delta_dst_idxs[i];
        if (src < 0 || src >= batch.n_tokens || dst < 0 || (uint32_t) dst >= state_size*n_stream) {
            LLAMA_LOG_ERROR("%s: DSV41 %s delta row[%zu] src=%lld dst=%lld is outside the batch/state ring\n",
                    __func__, tag, i, (long long) src, (long long) dst);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_persist_src_idxs.size(); ++i) {
        const int64_t src = plan.state_persist_src_idxs[i];
        const int64_t dst = plan.state_persist_dst_idxs[i];
        if (src < 0 || src >= batch.n_tokens) {
            LLAMA_LOG_ERROR("%s: DSV41 %s persist src[%zu]=%lld outside current batch rows=%d\n",
                    __func__, tag, i, (long long) src, batch.n_tokens);
            return false;
        }
        if (dst < 0 || (uint32_t) dst >= state_size*n_stream) {
            LLAMA_LOG_ERROR("%s: DSV41 %s persist dst[%zu]=%lld outside state_size*n_stream=%u\n",
                    __func__, tag, i, (long long) dst, state_size*n_stream);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_read_idxs.size(); ++i) {
        const int64_t idx = plan.state_read_idxs[i];
        if (idx < 0 || idx > max_state_read_idx) {
            LLAMA_LOG_ERROR("%s: DSV41 %s read idx[%zu]=%lld outside max source row=%lld\n",
                    __func__, tag, i, (long long) idx, (long long) max_state_read_idx);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_write_idxs.size(); ++i) {
        const int64_t idx = plan.state_write_idxs[i];
        if (idx < 0 || (uint32_t) idx >= kv_size*n_stream) {
            LLAMA_LOG_ERROR("%s: DSV41 %s write idx[%zu]=%lld outside kv_size*n_stream=%u\n",
                    __func__, tag, i, (long long) idx, kv_size*n_stream);
            return false;
        }
    }

    if (plan.n_kv == 0 || (uint32_t) plan.n_kv > kv_size) {
        LLAMA_LOG_ERROR("%s: DSV41 %s plan n_kv=%lld outside kv_size=%u\n",
                __func__, tag, (long long) plan.n_kv, kv_size);
        return false;
    }

    return true;
}

static llama_context::dsv41_runtime::comp_plan dsv41_build_comp_plan(
        const llama_batch & batch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    llama_context::dsv41_runtime::comp_plan plan;
    plan.n_visible.resize((size_t) batch.n_tokens);
    plan.n_stream = dsv41_comp_graph_n_stream(batch, n_stream);

    if (n_stream <= 1 && dsv41_batch_unique_seq_ids(batch).size() > 1) {
        LLAMA_LOG_ERROR("%s: DSV41 single compressed stream cannot serve multiple sequences\n", __func__);
        return plan;
    }

    const int64_t state_rows = (int64_t) state_size*n_stream;

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;
    std::map<std::pair<llama_seq_id, llama_pos>, int32_t> curr_token_idx_map;

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t n_token_seqs =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.seq_id[i] != nullptr
                ? batch.n_seq_id[i]
                : 1;
        for (int32_t s = 0; s < n_token_seqs; ++s) {
            const llama_seq_id seq_id =
                    batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.seq_id[i] != nullptr
                    ? batch.seq_id[i][s]
                    : 0;
            curr_token_idx_map[std::make_pair(seq_id, batch.pos[i])] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            return (int32_t) (state_rows + batch.n_tokens);
        }

        const auto it = curr_token_idx_map.find(std::make_pair(seq_id, pos));
        if (it != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + it->second);
        }

        const int64_t stream_off = dsv41_stream_offset(n_stream, seq_id, state_size);
        GGML_ASSERT(stream_off >= 0);
        return (int32_t) (stream_off + pos%state_size);
    };

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_pos pos = batch.pos[i];
        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const llama_seq_id delta_seq_id =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.seq_id[i] != nullptr && batch.n_seq_id[i] > 0
                ? batch.seq_id[i][0]
                : 0;
        plan.state_delta_src_idxs.push_back(i);
        plan.state_delta_dst_idxs.push_back((int32_t) (
                dsv41_stream_offset(n_stream, delta_seq_id, state_size) + pos%state_size));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[(size_t) i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        const int32_t n_token_seqs =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.seq_id[i] != nullptr
                ? batch.n_seq_id[i]
                : 1;
        for (int32_t s = 0; s < n_token_seqs; ++s) {
            const llama_seq_id seq_id =
                    batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.seq_id[i] != nullptr
                    ? batch.seq_id[i][s]
                    : 0;
            const int64_t stream_off = dsv41_stream_offset(n_stream, seq_id, state_size);
            const int32_t state_idx = (int32_t) (stream_off + pos%state_size);
            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(), [state_idx](const persist_row & row) {
                return row.dst == state_idx;
            });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, i, pos });
            } else if (pos > it->pos) {
                it->src = i;
                it->pos = pos;
            }

            if ((pos + 1) % ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - ratio;
            const int64_t cache_off = dsv41_stream_offset(n_stream, seq_id, kv_size);
            plan.state_write_idxs.push_back(cache_off + pos/ratio);
            plan.state_write_pos.push_back((int32_t) source_start);

            if (overlap) {
                const llama_pos prev_start = source_start - ratio;
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + j));
                }
            }
        }
    }

    // a ratio-2 batch without a completed group writes a dummy row at kv_size-1 to keep
    // graph shapes stable; the row stays outside every token's n_visible
    if (ratio == llama_context::dsv41_runtime::R2_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty()) {
        const llama_seq_id seq_id0 =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[0] > 0 && batch.seq_id[0] != nullptr
                ? batch.seq_id[0][0]
                : 0;
        const uint32_t source_idx = (uint32_t) state_source_idx(seq_id0, batch.pos[0]);
        const int64_t cache_off = std::max<int64_t>(0, dsv41_stream_offset(n_stream, seq_id0, kv_size));
        plan.state_write_idxs.push_back(cache_off + (int64_t) kv_size - 1);
        plan.state_write_pos.push_back(0);

        if (overlap) {
            for (uint32_t j = 0; j < ratio; ++j) {
                overlap_prev_reads.push_back(source_idx);
                overlap_cur_reads.push_back(source_idx);
            }
        } else {
            for (uint32_t j = 0; j < ratio; ++j) {
                plan.state_read_idxs.push_back(source_idx);
            }
        }
    }

    if (overlap) {
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(), overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(), overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    plan.n_kv = GGML_PAD(plan.n_kv, 256u);

    std::sort(persist_rows.begin(), persist_rows.end(), [](const persist_row & a, const persist_row & b) {
        return a.dst < b.dst;
    });

    for (const persist_row & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }

    if (plan.n_kv == 0) {
        plan.n_kv = GGML_PAD(1, 256u);
    }

    return plan;
}

static void dsv41_set_mask_tensor(
        ggml_tensor * tensor,
        const llama_context::dsv41_runtime::comp_plan & plan,
        int32_t n_tokens) {
    if (tensor == nullptr) {
        return;
    }

    if (tensor->buffer == nullptr) {
        return;
    }

    const int64_t width = tensor->ne[0];
    const int64_t height = tensor->ne[1];
    auto type = tensor->type;
    GGML_ASSERT(type == GGML_TYPE_F16 || type == GGML_TYPE_F32);

    if (type == GGML_TYPE_F16) {
        auto h_inf = ggml_fp32_to_fp16(-INFINITY);
        auto h_zero = ggml_fp32_to_fp16(0.0f);
        std::vector<ggml_fp16_t> storage((size_t) width*height, h_inf);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const int32_t n_visible = i < (int32_t) plan.n_visible.size() ? plan.n_visible[(size_t) i] : 0;
            for (int32_t j = 0; j < n_visible && j < width; ++j) {
                storage[(size_t) i*width + j] = h_zero;
            }
        }
        ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size()*sizeof(ggml_fp16_t));
    } else {
        std::vector<float> storage((size_t) width*height, -INFINITY);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const int32_t n_visible = i < (int32_t) plan.n_visible.size() ? plan.n_visible[(size_t) i] : 0;
            for (int32_t j = 0; j < n_visible && j < width; ++j) {
                storage[(size_t) i*width + j] = 0.0f;
            }
        }
        ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size()*sizeof(float));
    }
}

bool llama_context::ensure_dsv41_cache_tensors() {
    const int32_t n_layer = model.hparams.n_layer;
    const int64_t n_embd_head = model.hparams.n_embd_head_k(0);
    const uint32_t n_stream = std::max<uint32_t>(1, cparams.n_seq_max);

    if (!dsv41_validate_cache_type(kv_self.type_k, n_embd_head, "compressed")) {
        return false;
    }

    // M3 step 4: the indexer key cache lives at index-key owners (V4: LID cache at
    // every ratio-4 layer, fed by a second compressor; V4.1 derives the keys from the
    // shared compressor latent instead)
    if (model.hparams.indexer_head_size > 0 &&
        !dsv41_validate_cache_type(cparams.idx_type_k, model.hparams.indexer_head_size, "indexer")) {
        return false;
    }

    if (dsv41.cache.cache_ctx != nullptr &&
        (int32_t) dsv41.cache.comp_k.size() == n_layer &&
        dsv41.cache.n_stream == n_stream) {
        return true;
    }

    free_dsv41_cache_tensors();

    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (16 * std::max(1, n_layer)) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    dsv41.cache.cache_ctx = ggml_init(params);
    if (dsv41.cache.cache_ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to allocate DSV41 cache context\n", __func__);
        return false;
    }

    auto & cache = dsv41.cache;
    cache.n_stream = n_stream;
    cache.comp_k.resize((size_t) n_layer, nullptr);
    cache.comp_state_kv.resize((size_t) n_layer, nullptr);
    cache.comp_state_score.resize((size_t) n_layer, nullptr);
    cache.lid_k.resize((size_t) n_layer, nullptr);

    auto alloc_tensor = [&](ggml_tensor * tensor, ggml_backend_buffer_type_t buft) -> bool {
        const size_t tensor_bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, tensor_bytes);
        if (buf == nullptr) {
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_tensor_alloc(buf, tensor, ggml_backend_buffer_get_base(buf));
        ggml_backend_buffer_clear(buf, 0);
        cache.cache_bufs.push_back(buf);
        return true;
    };

    for (int32_t il = 0; il < n_layer; ++il) {
        if (!model.hparams.dsv41_is_kv_source((uint32_t) il)) {
            continue;
        }

        const uint32_t ratio = model.hparams.dsv4_compress_ratios[(size_t) il];
        if (ratio != dsv41_runtime::R2_RATIO && ratio != dsv41_runtime::R1_RATIO) {
            LLAMA_LOG_ERROR("%s: DSV41 kv_source layer %d has unsupported compress ratio %u\n", __func__, il, ratio);
            free_dsv41_cache_tensors();
            return false;
        }

        const uint32_t comp_kv = GGML_PAD(dsv41_comp_size(cparams.n_ctx, ratio), 256u);
        ggml_backend_buffer_type_t buft = dsv41_layer_buft(*this, il);

        cache.comp_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, kv_self.type_k, n_embd_head, comp_kv*n_stream, 1);
        cache.comp_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, n_embd_head, ratio*n_stream);
        cache.comp_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, n_embd_head, ratio*n_stream);

        if (!alloc_tensor(cache.comp_k[(size_t) il], buft) ||
            !alloc_tensor(cache.comp_state_kv[(size_t) il], buft) ||
            !alloc_tensor(cache.comp_state_score[(size_t) il], buft)) {
            LLAMA_LOG_ERROR("%s: failed to allocate DSV41 compressor buffers for layer %d\n", __func__, il);
            free_dsv41_cache_tensors();
            return false;
        }
    }

    // M3 step 4: indexer key cache at index-key owners. Owners are kv_sources (the keys
    // are derived from that layer's compressor latent), so the row layout follows the
    // layer's own ratio — same row count as its comp_k, but indexer-head wide and in
    // the indexer cache type (V4: lid_k at every CSA layer, cparams.idx_type_k).
    for (int32_t il = 0; il < n_layer; ++il) {
        if (!model.hparams.dsv41_owns_index_k((uint32_t) il)) {
            continue;
        }

        const uint32_t ratio = model.hparams.dsv4_compress_ratios[(size_t) il];
        const uint32_t lid_kv = GGML_PAD(dsv41_comp_size(cparams.n_ctx, ratio), 256u);
        ggml_backend_buffer_type_t buft = dsv41_layer_buft(*this, il);

        cache.lid_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, cparams.idx_type_k,
                model.hparams.indexer_head_size, lid_kv*n_stream, 1);

        if (!alloc_tensor(cache.lid_k[(size_t) il], buft)) {
            LLAMA_LOG_ERROR("%s: failed to allocate DSV41 indexer key buffer for layer %d\n", __func__, il);
            free_dsv41_cache_tensors();
            return false;
        }
    }

    auto bytes = [](const auto & tensors) {
        size_t total = 0;
        for (const ggml_tensor * tensor : tensors) {
            if (tensor != nullptr) {
                total += ggml_nbytes(tensor);
            }
        }
        return total;
    };

    const size_t comp_k_bytes = bytes(cache.comp_k);
    const size_t state_bytes = bytes(cache.comp_state_kv) + bytes(cache.comp_state_score);
    const size_t lid_k_bytes = bytes(cache.lid_k);

    LLAMA_LOG_INFO("%s: DSV41 cache: comp K=%7.2f MiB (%s), states=%7.2f MiB, lid K=%7.2f MiB (%s), total=%7.2f MiB, streams=%u\n",
            __func__,
            (float) comp_k_bytes / (1024.0f * 1024.0f), ggml_type_name(kv_self.type_k),
            (float) state_bytes / (1024.0f * 1024.0f),
            (float) lid_k_bytes / (1024.0f * 1024.0f), ggml_type_name(cparams.idx_type_k),
            (float) (comp_k_bytes + state_bytes + lid_k_bytes) / (1024.0f * 1024.0f),
            n_stream);

    return true;
}

void llama_context::free_dsv41_cache_tensors() {
    auto release_vector = [](auto & v) {
        using vec_type = std::decay_t<decltype(v)>;
        vec_type().swap(v);
    };

    for (ggml_backend_buffer_t buf : dsv41.cache.cache_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    release_vector(dsv41.cache.cache_bufs);
    release_vector(dsv41.cache.comp_k);
    release_vector(dsv41.cache.comp_state_kv);
    release_vector(dsv41.cache.comp_state_score);
    release_vector(dsv41.cache.lid_k);
    dsv41.cache.n_stream = 1;
    if (dsv41.cache.cache_ctx != nullptr) {
        ggml_free(dsv41.cache.cache_ctx);
        dsv41.cache.cache_ctx = nullptr;
    }
}

template<typename T>
static void dsv41_set_input_tensor(ggml_tensor * tensor, const std::vector<T> & values) {
    if (tensor == nullptr || tensor->buffer == nullptr || values.empty()) {
        return;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(T));
}

// M3 step 2 (plan B10): push the ubatch tokens into the engram n-gram cache, deduping by
// absolute position so a re-decode of the same batch never double-pushes (engram.size is
// the count of contiguous positions pushed so far; tokens must arrive in non-decreasing
// pos order). Single-sequence only for M3.
static void dsv41_engram_push_tokens(llama_context & lctx, const llama_batch & batch) {
    const auto & hparams = lctx.model.hparams;
    auto & engram = lctx.dsv41.engram;

    if (batch.token == nullptr) {
        throw std::runtime_error("DSV41 engram: embedding-input batches are not supported yet");
    }
    if (batch.pos == nullptr) {
        throw std::runtime_error("DSV41 engram: batch positions are required");
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id != nullptr &&
            (batch.n_seq_id[i] != 1 || batch.seq_id == nullptr || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != 0)) {
            throw std::runtime_error("DSV41 engram: multi-sequence batches are not supported yet");
        }
        if (i > 0 && batch.pos[i] < batch.pos[i - 1]) {
            throw std::runtime_error("DSV41 engram: batch positions must be non-decreasing");
        }
    }

    const uint32_t first_pos = (uint32_t) batch.pos[0];
    const uint32_t n_tokens  = (uint32_t) batch.n_tokens;

    // positions [first_pos, min(first_pos + n, size)) were pushed by an earlier ubatch
    // (engram.size is the count of contiguous positions pushed so far, so only positions
    // at or past it are unseen)
    uint32_t skip = first_pos < engram.size ? std::min(engram.size - first_pos, n_tokens) : 0;
    if (first_pos + skip != engram.size && skip < n_tokens) {
        throw std::runtime_error("DSV41 engram: position gap — the n-gram cache is missing history");
    }
    if (skip < n_tokens) {
        engram.push_tokens(hparams, batch.token + skip, n_tokens - skip, first_pos + skip);
    }
}

// hash + gather the engram rows for the ubatch into the F32 graph inputs
// (dsv41_engram_rows{s}); runs after dsv41_engram_push_tokens so the cache also covers
// the n-gram lookback into previous ubatches
static void dsv41_engram_set_inputs(llama_context & lctx, const llama_batch & batch) {
    const auto & hparams = lctx.model.hparams;

    const uint32_t n_cols  = (hparams.engram_max_ngram_size - 1)*hparams.engram_n_heads;
    const uint32_t key_len = hparams.engram_key_length;
    const uint32_t first_pos = (uint32_t) batch.pos[0];
    const int64_t  n_rows = (int64_t) n_cols*batch.n_tokens;

    lctx.dsv41.engram_ids.resize((size_t) n_rows);
    lctx.dsv41.engram_gather.resize((size_t) n_rows*key_len);

    for (uint32_t s = 0; s < hparams.engram_layer_count; ++s) {
        lctx.dsv41.engram.hash_positions(hparams, s, first_pos, batch.n_tokens, lctx.dsv41.engram_ids.data());

        const int32_t il = (int32_t) hparams.engram_layer_ids[s];
        const ggml_tensor * table = lctx.model.layers[il].engram_embd;
        GGML_ASSERT(table != nullptr && table->ne[0] == (int64_t) key_len);
        // the table lives in the host-side ctx_input buffer and must never enter the
        // compute graph — only the gathered F32 rows do
        GGML_ASSERT(ggml_backend_buffer_is_host(table->buffer));

        // reference ParallelEngramEmbedding (model.py:312-321) zero-masks lookups
        // outside the table; gather_rows implements that (rows with id >= ne[1]
        // gather as zeros), so the raw hash ids go in unmodified. The tiny model's
        // hash constants target the real ~384M-row layout while its table is much
        // smaller, so nearly every id masks to zero there; for the real model
        // ids < ne[1] by construction and the mask is a no-op.
        llama_engram::gather_rows(table, lctx.dsv41.engram_ids.data(), n_rows, lctx.dsv41.engram_gather.data());
        dsv41_set_input_tensor(lctx.dsv41.inputs.engram_rows[s], lctx.dsv41.engram_gather);
    }
}

bool llama_prepare_dsv41_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan) {
    if (lctx.model.arch != LLM_ARCH_DEEPSEEK41) {
        return true;
    }

    if (!dsv41_validate_batch_seq_ids(lctx, batch)) {
        return false;
    }

    if (!lctx.ensure_dsv41_cache_tensors()) {
        return false;
    }

    const uint32_t cache_n_stream = std::max<uint32_t>(1, lctx.dsv41.cache.n_stream);
    const uint32_t r2_kv_size = GGML_PAD(dsv41_comp_size(lctx.cparams.n_ctx, llama_context::dsv41_runtime::R2_RATIO), 256u);
    const uint32_t r1_kv_size = GGML_PAD(dsv41_comp_size(lctx.cparams.n_ctx, llama_context::dsv41_runtime::R1_RATIO), 256u);

    const auto build_plan = [&](uint32_t ratio, uint32_t kv_size) {
        // both V4.1 plans are non-overlap; the state ring holds ratio rows per stream
        return reserve_plan
            ? dsv41_build_reserve_comp_plan(batch, ratio, false, ratio, kv_size, cache_n_stream)
            : dsv41_build_comp_plan(batch, ratio, false, ratio, kv_size, cache_n_stream);
    };

    // M3 step 1: raw sliding-window context. The reserve pass needs no raw
    // indices (the graph only reads their sizes for input shapes).
    lctx.dsv41.raw = {};
    if (!reserve_plan && !dsv41_build_raw_context(lctx, batch, lctx.dsv41.raw)) {
        return false;
    }

    // M3 step 3: the r2/r1 compressed plans (the dummy-write special case is keyed on
    // the ratio-2 plan inside the builder)
    lctx.dsv41.r2_plan = build_plan(llama_context::dsv41_runtime::R2_RATIO, r2_kv_size);
    lctx.dsv41.r1_plan = build_plan(llama_context::dsv41_runtime::R1_RATIO, r1_kv_size);
    lctx.dsv41.r2_ctx = dsv41_build_comp_context(batch, cache_n_stream, lctx.dsv41.r2_plan.n_kv);
    lctx.dsv41.r1_ctx = dsv41_build_comp_context(batch, cache_n_stream, lctx.dsv41.r1_plan.n_kv);

    if (!dsv41_validate_comp_plan("r2", batch, lctx.dsv41.r2_plan, llama_context::dsv41_runtime::R2_RATIO, false,
                llama_context::dsv41_runtime::R2_RATIO, r2_kv_size, cache_n_stream) ||
        !dsv41_validate_comp_plan("r1", batch, lctx.dsv41.r1_plan, llama_context::dsv41_runtime::R1_RATIO, false,
                llama_context::dsv41_runtime::R1_RATIO, r1_kv_size, cache_n_stream)) {
        return false;
    }

    // M3 step 2: engram n-gram cache. The reserve pass uses a synthetic batch, so it
    // never touches the cache; both real passes push (the dedupe by absolute position
    // makes the second push of the same ubatch a no-op).
    if (!reserve_plan && batch.n_tokens > 0 && lctx.model.hparams.engram_layer_count > 0) {
        dsv41_engram_push_tokens(lctx, batch);
    }

    if (!set_tensors) {
        return true;
    }

    dsv41_set_input_tensor(lctx.dsv41.inputs.raw_k_write_src_idxs, lctx.dsv41.raw.write_src_idxs);
    dsv41_set_input_tensor(lctx.dsv41.inputs.raw_k_write_idxs, lctx.dsv41.raw.write_dst_idxs);
    dsv41_set_input_tensor(lctx.dsv41.inputs.raw_k_read_idxs, lctx.dsv41.raw.read_dst_idxs);

    auto set_comp = [&](llama_context::dsv41_runtime::comp_inputs & inputs,
                        const llama_context::dsv41_runtime::comp_plan & plan) {
        dsv41_set_input_tensor(inputs.state_pos, plan.state_pos);
        dsv41_set_input_tensor(inputs.state_persist_src_idxs, plan.state_persist_src_idxs);
        dsv41_set_input_tensor(inputs.state_persist_dst_idxs, plan.state_persist_dst_idxs);
        dsv41_set_input_tensor(inputs.state_read_idxs, plan.state_read_idxs);
        dsv41_set_input_tensor(inputs.state_write_idxs, plan.state_write_idxs);
        dsv41_set_input_tensor(inputs.state_write_pos, plan.state_write_pos);
        dsv41_set_mask_tensor(inputs.kq_mask, plan, batch.n_tokens);
    };

    set_comp(lctx.dsv41.inputs.r2, lctx.dsv41.r2_plan);
    set_comp(lctx.dsv41.inputs.r1, lctx.dsv41.r1_plan);

    if (batch.n_tokens > 0 && lctx.model.hparams.engram_layer_count > 0) {
        dsv41_engram_set_inputs(lctx, batch);
    }

    return true;
}

const std::vector<float> & llama_dsv41_debug_engram_gather(const llama_context * ctx) {
    return ctx->dsv41.engram_gather;
}

uint32_t llama_dsv41_debug_engram_size(const llama_context * ctx) {
    return ctx->dsv41.engram.size;
}

const std::vector<int32_t> & llama_dsv41_debug_r2_write_pos(const llama_context * ctx) {
    return ctx->dsv41.r2_plan.state_write_pos;
}

const std::vector<int64_t> & llama_dsv41_debug_r2_write_idxs(const llama_context * ctx) {
    return ctx->dsv41.r2_plan.state_write_idxs;
}

// test-only debug access: per-token compressed visibility of the last prepared
// ubatch's r2/r1 plans (drives the causal compressed masks)
const std::vector<int32_t> & llama_dsv41_debug_r2_n_visible(const llama_context * ctx) {
    return ctx->dsv41.r2_plan.n_visible;
}

const std::vector<int32_t> & llama_dsv41_debug_r1_n_visible(const llama_context * ctx) {
    return ctx->dsv41.r1_plan.n_visible;
}

size_t llama_dsv41_debug_cache_bytes(const llama_context * ctx) {
    size_t total = 0;
    for (ggml_backend_buffer_t buf : ctx->dsv41.cache.cache_bufs) {
        if (buf != nullptr) {
            total += ggml_backend_buffer_get_size(buf);
        }
    }
    return total;
}

void llama_reset_dsv41_state(llama_context * ctx, int32_t seq_id) {
    if (ctx == nullptr) {
        return;
    }

    const uint32_t n_stream = std::max<uint32_t>(1, ctx->dsv41.cache.n_stream);
    if (seq_id >= (llama_seq_id) n_stream) {
        LLAMA_LOG_ERROR("%s: DSV41 seq_id %d is outside stream range %u\n", __func__, seq_id, n_stream);
        return;
    }

    // M3 step 2: the engram n-gram cache is single-sequence, so every reset clears it
    // fully (a per-seq clear of sequence 0 is the whole cache)
    ctx->dsv41.engram.compressed.clear();
    ctx->dsv41.engram.size = 0;

    if (seq_id < 0) {
        for (ggml_backend_buffer_t buf : ctx->dsv41.cache.cache_bufs) {
            ggml_backend_buffer_clear(buf, 0);
        }
        return;
    }

    // per-seq reset: clear only this stream's rows of the compressed caches and the
    // partial-group state rings (mirror llama_reset_dsv4_state)
    auto clear_tensor = [seq_id, n_stream](ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return;
        }

        GGML_ASSERT(tensor->ne[1] % n_stream == 0);
        const size_t row_bytes = tensor->nb[1];
        const size_t rows_per_stream = (size_t) tensor->ne[1] / n_stream;
        const size_t offset = (size_t) seq_id * rows_per_stream * row_bytes;
        const size_t bytes = rows_per_stream * row_bytes;
        std::vector<uint8_t> zeros(bytes, 0);
        ggml_backend_tensor_set(tensor, zeros.data(), offset, bytes);
    };

    for (ggml_tensor * tensor : ctx->dsv41.cache.comp_k) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv41.cache.comp_state_kv) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv41.cache.comp_state_score) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv41.cache.lid_k) clear_tensor(tensor);
}
