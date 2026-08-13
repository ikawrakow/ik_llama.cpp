#include "llama-dsv4.h"

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
#include <type_traits>
#include <unordered_set>

static bool dsv4_cache_type_supported(ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0;
}

// Per-step capture is limited to the eight-row CSA/LID ring.
// TODO: Expand to a larger number
static constexpr int DSV4_PER_STEP_MAX_STATE_ROWS = 8;

static bool dsv4_validate_cache_type(ggml_type type, int64_t width, const char * name) {
    if (!dsv4_cache_type_supported(type)) {
        LLAMA_LOG_ERROR("%s: unsupported DSV4 %s cache type %s\n", __func__, name, ggml_type_name(type));
        return false;
    }
    if (ggml_is_quantized(type) && width % ggml_blck_size(type) != 0) {
        LLAMA_LOG_ERROR("%s: DSV4 %s cache width %d is not aligned to %d elements for %s\n",
                __func__, name, (int)width, (int)ggml_blck_size(type), ggml_type_name(type));
        return false;
    }
    return true;
}

static ggml_backend_buffer_type_t llama_dsv4_layer_buft(const llama_context & lctx, int32_t il) {
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

static uint32_t dsv4_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

static bool dsv4_validate_csa_lid_visibility(
        const llama_context & lctx,
        uint32_t csa_kv_size,
        uint32_t lid_kv_size) {
    const auto & csa_plan = lctx.dsv4.csa_plan;
    const auto & lid_plan = lctx.dsv4.lid_plan;
    const auto & csa_ctx = lctx.dsv4.csa_ctx;
    const auto & lid_ctx = lctx.dsv4.lid_ctx;

    if (csa_kv_size != lid_kv_size ||
            csa_plan.n_stream != lid_plan.n_stream ||
            csa_plan.n_kv != lid_plan.n_kv ||
            csa_plan.n_visible != lid_plan.n_visible ||
            csa_ctx.graph_n_stream != lid_ctx.graph_n_stream ||
            csa_ctx.n_kv != lid_ctx.n_kv ||
            csa_ctx.sinfo.strm != lid_ctx.sinfo.strm ||
            csa_ctx.sinfo.idxs != lid_ctx.sinfo.idxs ||
            csa_ctx.sinfo.s0 != lid_ctx.sinfo.s0 ||
            csa_ctx.sinfo.s1 != lid_ctx.sinfo.s1) {
        LLAMA_LOG_ERROR("%s: DSV4 CSA/LID visibility contracts differ\n", __func__);
        return false;
    }

    return true;
}

static void dsv4_batch_shape(
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

static bool dsv4_batch_has_coupled(const llama_batch & batch) {
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

static bool dsv4_token_has_seq(const llama_batch & batch, int32_t i, llama_seq_id seq_id) {
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

static std::vector<llama_seq_id> dsv4_batch_unique_seq_ids(const llama_batch & batch) {
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

static int64_t dsv4_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (n_stream <= 1) {
        return 0;
    }

    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        LLAMA_LOG_ERROR("%s: DSV4 seq_id %d is outside stream range %u\n", __func__, seq_id, n_stream);
        return -1;
    }

    return (int64_t) seq_id*size;
}

static int64_t dsv4_comp_graph_n_stream(const llama_batch & batch, uint32_t n_stream) {
    if (n_stream <= 1) {
        return 1;
    }

    const std::vector<llama_seq_id> seq_ids = dsv4_batch_unique_seq_ids(batch);
    if (seq_ids.size() <= 1 || dsv4_batch_has_coupled(batch)) {
        return 1;
    }

    return (int64_t) seq_ids.size();
}

static std::vector<llama_seq_id> dsv4_build_stream_seq_ids(
        const llama_batch & batch,
        uint32_t n_stream) {
    if (n_stream <= 1) {
        return { 0 };
    }

    const std::vector<llama_seq_id> seq_ids = dsv4_batch_unique_seq_ids(batch);
    if (seq_ids.size() <= 1 || dsv4_batch_has_coupled(batch)) {
        return { seq_ids.empty() ? 0 : seq_ids.front() };
    }

    return seq_ids;
}

static llama_context::dsv4_runtime::slot_info dsv4_build_comp_sinfo(
        const llama_batch & batch,
        uint32_t n_stream) {
    llama_context::dsv4_runtime::slot_info sinfo;

    const std::vector<llama_seq_id> seq_ids = dsv4_build_stream_seq_ids(batch, n_stream);
    const int64_t graph_n_stream = (int64_t) seq_ids.size();
    bool have_stream = false;

    sinfo.s0 = INT_MAX;
    sinfo.s1 = 0;
    sinfo.resize((size_t) std::max<int64_t>(1, graph_n_stream));
    for (int64_t s = 0; s < graph_n_stream; ++s) {
        const llama_seq_id seq_id = seq_ids[(size_t) s];
        const int64_t strm = dsv4_stream_offset(n_stream, seq_id, 1);
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
        LLAMA_LOG_ERROR("%s: DSV4 compressed streams are not contiguous in batch\n", __func__);
    }

    return sinfo;
}

static llama_context::dsv4_runtime::slot_info dsv4_build_raw_read_sinfo(
        const llama_context::dsv4_runtime::slot_info & sinfo_write,
        const llama_batch & batch,
        uint32_t n_stream) {
    if (!dsv4_batch_has_coupled(batch)) {
        return sinfo_write;
    }

    const llama_seq_id seq_id =
            (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_tokens > 0 && batch.n_seq_id[0] > 0 && batch.seq_id[0] != nullptr)
            ? batch.seq_id[0][0]
            : 0;
    const int64_t strm = dsv4_stream_offset(n_stream, seq_id, 1);
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
        LLAMA_LOG_ERROR("%s: DSV4 raw write stream not found for coupled read\n", __func__);
        return {};
    }

    llama_context::dsv4_runtime::slot_info sinfo;
    sinfo.resize(1);
    sinfo.strm[0] = sinfo_write.strm[i_stream];
    sinfo.idxs[0] = sinfo_write.idxs[i_stream];
    sinfo.s0 = (int32_t) strm;
    sinfo.s1 = sinfo.s0;

    return sinfo;
}

static bool dsv4_validate_batch_seq_ids(
        const llama_context & lctx,
        const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return true;
    }

    const uint32_t n_stream = std::max<uint32_t>(1, lctx.cparams.n_seq_max);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            LLAMA_LOG_ERROR("%s: DSV4 token %d is missing seq_id ownership\n", __func__, i);
            return false;
        }

        for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = batch.seq_id[i][s];
            if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
                LLAMA_LOG_ERROR("%s: DSV4 token %d seq_id %d is outside n_seq_max=%u\n",
                        __func__, i, seq_id, n_stream);
                return false;
            }
        }
    }

    return true;
}

static bool dsv4_build_raw_context(
        const llama_context & lctx,
        const llama_batch & batch,
        llama_context::dsv4_runtime::raw_context & raw) {
    raw = {};
    const uint32_t n_stream = std::max<uint32_t>(1, lctx.cparams.n_seq_max);
    const std::vector<llama_seq_id> write_seq_ids = dsv4_build_stream_seq_ids(batch, n_stream);
    raw.sinfo_write = dsv4_build_comp_sinfo(batch, n_stream);
    raw.sinfo_read  = dsv4_build_raw_read_sinfo(raw.sinfo_write, batch, n_stream);
    raw.graph_n_stream = (int64_t) raw.sinfo_write.n_stream();
    std::vector<llama_seq_id> read_seq_ids = write_seq_ids;

    if (dsv4_batch_has_coupled(batch)) {
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
        LLAMA_LOG_ERROR("%s: DSV4 raw write slots [%d, %d) are outside kv cache size %u\n",
                __func__, kv.head, kv.head + batch.n_tokens, kv.size);
        return false;
    }

    // compacted layers address raw K rows through [sinks | window] geometry rather than by cell
    const bool compacted = kv.any_compacted();
    if (compacted) {
        if (kv.head_swa + (uint32_t) batch.n_tokens > kv.size_swa) {
            LLAMA_LOG_ERROR("%s: DSV4 compacted raw write rows [%u, %u) are outside size_swa %u\n",
                    __func__, kv.head_swa, kv.head_swa + (uint32_t) batch.n_tokens, kv.size_swa);
            return false;
        }
        if (batch.pos != nullptr && batch.n_tokens > 0 &&
            kv.pos_base_swa + (llama_pos) (kv.head_swa - kv.sink_rows) != batch.pos[0]) {
            LLAMA_LOG_ERROR("%s: DSV4 compacted write row %u disagrees with batch position %d (base %d)\n",
                    __func__, kv.head_swa, batch.pos[0], kv.pos_base_swa);
            return false;
        }
    }

    raw.write_counts.push_back(batch.n_tokens);
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t slot = kv.head + i;
        const llama_kv_cell & cell = kv.cells[(size_t) slot];

        if (batch.pos != nullptr && cell.pos != batch.pos[i]) {
            LLAMA_LOG_ERROR("%s: DSV4 raw write slot %d pos mismatch: cell=%d batch=%d\n",
                    __func__, slot, cell.pos, batch.pos[i]);
            return false;
        }

        raw.write_src_idxs.push_back(i);
        raw.write_dst_idxs.push_back(compacted ? (int32_t) kv.head_swa + i : slot);
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
            if (compacted && cell.pos < kv.pos_base_swa) {
                // rows before the window base were overwritten by compaction
                continue;
            }
            const uint32_t row = compacted
                ? kv.sink_rows + (uint32_t) (cell.pos - kv.pos_base_swa) : slot;
            raw.sinfo_read.idxs[s].push_back(row);
            raw.read_dst_idxs.push_back((int32_t) row);
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
            if (!dsv4_token_has_seq(batch, i, seq_id)) {
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
                LLAMA_LOG_ERROR("%s: DSV4 packed batch has unequal raw-write rows per stream\n", __func__);
                return false;
            }

            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (dsv4_token_has_seq(batch, i, write_seq_ids[s])) {
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

static llama_context::dsv4_runtime::comp_context dsv4_build_comp_context(
        const llama_batch & batch,
        uint32_t n_stream,
        int64_t n_kv) {
    llama_context::dsv4_runtime::comp_context ctx;
    ctx.sinfo = dsv4_build_comp_sinfo(batch, n_stream);
    ctx.graph_n_stream = dsv4_comp_graph_n_stream(batch, n_stream);
    ctx.n_kv = n_kv;
    return ctx;
}

static llama_context::dsv4_runtime::comp_plan dsv4_build_reserve_comp_plan(
        const llama_batch & batch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    llama_context::dsv4_runtime::comp_plan plan;
    plan.n_visible.resize((size_t) batch.n_tokens, (int32_t) kv_size);
    plan.n_stream = dsv4_comp_graph_n_stream(batch, n_stream);
    plan.n_kv = kv_size;

    if (batch.n_tokens == 0) {
        return plan;
    }

    uint32_t n_seqs = 1;
    uint32_t n_seq_tokens = 1;
    dsv4_batch_shape(batch, n_seqs, n_seq_tokens);

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

static uint32_t dsv4_cache_kv_size(const std::vector<ggml_tensor *> & tensors) {
    for (ggml_tensor * tensor : tensors) {
        if (tensor != nullptr) {
            return (uint32_t) tensor->ne[1];
        }
    }

    return 0;
}

static uint32_t dsv4_cache_state_size(const std::vector<ggml_tensor *> & tensors) {
    for (ggml_tensor * tensor : tensors) {
        if (tensor != nullptr) {
            return (uint32_t) tensor->ne[1];
        }
    }

    return 0;
}

static bool dsv4_validate_comp_plan(
        const char * tag,
        const llama_batch & batch,
        const llama_context::dsv4_runtime::comp_plan & plan,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    const int64_t max_state_read_idx = (int64_t) state_size*n_stream + batch.n_tokens + (overlap ? 0 : -1);

    if (plan.n_visible.size() != (size_t) std::max(0, batch.n_tokens)) {
        LLAMA_LOG_ERROR("%s: DSV4 %s plan n_visible size mismatch: got=%zu expected=%d\n",
                __func__, tag, plan.n_visible.size(), std::max(0, batch.n_tokens));
        return false;
    }

    if (plan.state_pos.size() > (size_t) std::max(0, batch.n_tokens)) {
        LLAMA_LOG_ERROR("%s: DSV4 %s plan has too many state_pos rows: %zu > %d\n",
                __func__, tag, plan.state_pos.size(), std::max(0, batch.n_tokens));
        return false;
    }

    if (plan.state_delta_src_idxs.size() != plan.state_pos.size() ||
        plan.state_delta_dst_idxs.size() != plan.state_pos.size()) {
        LLAMA_LOG_ERROR("%s: DSV4 %s delta row metadata mismatch: state=%zu src=%zu dst=%zu\n",
                __func__, tag, plan.state_pos.size(), plan.state_delta_src_idxs.size(),
                plan.state_delta_dst_idxs.size());
        return false;
    }

    if (plan.state_persist_src_idxs.size() != plan.state_persist_dst_idxs.size()) {
        LLAMA_LOG_ERROR("%s: DSV4 %s persist idx size mismatch: src=%zu dst=%zu\n",
                __func__, tag, plan.state_persist_src_idxs.size(), plan.state_persist_dst_idxs.size());
        return false;
    }

    if (plan.state_write_idxs.size() != plan.state_write_pos.size()) {
        LLAMA_LOG_ERROR("%s: DSV4 %s write idx size mismatch: idxs=%zu pos=%zu\n",
                __func__, tag, plan.state_write_idxs.size(), plan.state_write_pos.size());
        return false;
    }

    for (size_t i = 0; i < plan.n_visible.size(); ++i) {
        const int32_t n_visible = plan.n_visible[i];
        if (n_visible < 0 || (uint32_t) n_visible > kv_size) {
            LLAMA_LOG_ERROR("%s: DSV4 %s n_visible[%zu]=%d exceeds kv_size=%u\n",
                    __func__, tag, i, n_visible, kv_size);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_pos.size(); ++i) {
        const int64_t pos = plan.state_pos[i];
        if (pos < 0 || pos >= (int64_t) ratio) {
            LLAMA_LOG_ERROR("%s: DSV4 %s state_pos[%zu]=%lld outside ratio=%u\n",
                    __func__, tag, i, (long long) pos, ratio);
            return false;
        }

        const int64_t src = plan.state_delta_src_idxs[i];
        const int64_t dst = plan.state_delta_dst_idxs[i];
        if (src < 0 || src >= batch.n_tokens || dst < 0 || (uint32_t) dst >= state_size*n_stream) {
            LLAMA_LOG_ERROR("%s: DSV4 %s delta row[%zu] src=%lld dst=%lld is outside the batch/state ring\n",
                    __func__, tag, i, (long long) src, (long long) dst);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_persist_src_idxs.size(); ++i) {
        const int64_t src = plan.state_persist_src_idxs[i];
        const int64_t dst = plan.state_persist_dst_idxs[i];
        if (src < 0 || src >= batch.n_tokens) {
            LLAMA_LOG_ERROR("%s: DSV4 %s persist src[%zu]=%lld outside current batch rows=%d\n",
                    __func__, tag, i, (long long) src, batch.n_tokens);
            return false;
        }
        if (dst < 0 || (uint32_t) dst >= state_size*n_stream) {
            LLAMA_LOG_ERROR("%s: DSV4 %s persist dst[%zu]=%lld outside state_size*n_stream=%u\n",
                    __func__, tag, i, (long long) dst, state_size*n_stream);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_read_idxs.size(); ++i) {
        const int64_t idx = plan.state_read_idxs[i];
        if (idx < 0 || idx > max_state_read_idx) {
            LLAMA_LOG_ERROR("%s: DSV4 %s read idx[%zu]=%lld outside max source row=%lld\n",
                    __func__, tag, i, (long long) idx, (long long) max_state_read_idx);
            return false;
        }
    }

    for (size_t i = 0; i < plan.state_write_idxs.size(); ++i) {
        const int64_t idx = plan.state_write_idxs[i];
        if (idx < 0 || (uint32_t) idx >= kv_size*n_stream) {
            LLAMA_LOG_ERROR("%s: DSV4 %s write idx[%zu]=%lld outside kv_size*n_stream=%u\n",
                    __func__, tag, i, (long long) idx, kv_size*n_stream);
            return false;
        }
    }

    if (plan.n_kv == 0 || (uint32_t) plan.n_kv > kv_size) {
        LLAMA_LOG_ERROR("%s: DSV4 %s plan n_kv=%lld outside kv_size=%u\n",
                __func__, tag, (long long) plan.n_kv, kv_size);
        return false;
    }

    return true;
}

static llama_context::dsv4_runtime::comp_plan dsv4_build_comp_plan(
        const llama_batch & batch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream) {
    llama_context::dsv4_runtime::comp_plan plan;
    plan.n_visible.resize((size_t) batch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(batch, n_stream);

    if (n_stream <= 1 && dsv4_batch_unique_seq_ids(batch).size() > 1) {
        LLAMA_LOG_ERROR("%s: DSV4 single compressed stream cannot serve multiple sequences\n", __func__);
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

        const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
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
                dsv4_stream_offset(n_stream, delta_seq_id, state_size) + pos%state_size));

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
            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
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
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);
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

    if (ratio == llama_context::dsv4_runtime::CSA_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty()) {
        const llama_seq_id seq_id0 =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[0] > 0 && batch.seq_id[0] != nullptr
                ? batch.seq_id[0][0]
                : 0;
        const uint32_t source_idx = (uint32_t) state_source_idx(seq_id0, batch.pos[0]);
        const int64_t cache_off = std::max<int64_t>(0, dsv4_stream_offset(n_stream, seq_id0, kv_size));
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

template<typename T>
static void dsv4_set_input_tensor(ggml_tensor * tensor, const std::vector<T> & values) {
    if (tensor == nullptr || tensor->buffer == nullptr || values.empty()) {
        return;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(T));
}

static void dsv4_set_mask_tensor(
        ggml_tensor * tensor,
        const llama_context::dsv4_runtime::comp_plan & plan,
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

    //printf("%s: preparing mask %s of type %s with %ld x %ld entries\n", __func__, tensor->name, ggml_type_name(type), tensor->ne[0], tensor->ne[1]);
    if (type == GGML_TYPE_F16) {
        auto h_inf = ggml_fp32_to_fp16(-INFINITY);
        auto h_zero = ggml_fp32_to_fp16(0.0f);
        std::vector<ggml_fp16_t> storage((size_t) width*height, h_inf);
        for (int32_t i = 0; i < n_tokens; ++i) {
            const int32_t n_visible = i < (int32_t) plan.n_visible.size() ? plan.n_visible[(size_t) i] : 0;
            //if (i == 0) printf("    n_visible = %d\n", n_visible);
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

bool llama_context::ensure_dsv4_cache_tensors() {
    const int32_t n_layer = model.hparams.n_layer;
    const int64_t n_embd_head = model.hparams.n_embd_head_k(0);
    const int64_t n_indexer_head = model.hparams.indexer_head_size;
    const uint32_t n_stream = std::max<uint32_t>(1, cparams.n_seq_max);
    const uint32_t csa_kv = GGML_PAD(dsv4_comp_size(cparams.n_ctx, dsv4_runtime::CSA_RATIO), 256u);
    const uint32_t hca_kv = GGML_PAD(dsv4_comp_size(cparams.n_ctx, dsv4_runtime::HCA_RATIO), 256u);

    if (!dsv4_validate_cache_type(kv_self.type_k, n_embd_head, "raw/CSA/HCA") ||
        !dsv4_validate_cache_type(cparams.idx_type_k, n_indexer_head, "LID")) {
        return false;
    }

    if (dsv4.cache.cache_ctx != nullptr &&
        (int32_t) dsv4.cache.csa_k.size() == n_layer &&
        dsv4.cache.n_stream == n_stream) {
        return true;
    }

    free_dsv4_cache_tensors();

    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (16 * std::max(1, n_layer)) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    dsv4.cache.cache_ctx = ggml_init(params);
    if (dsv4.cache.cache_ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to allocate DSV4 cache context\n", __func__);
        return false;
    }

    auto & cache = dsv4.cache;
    cache.n_stream = n_stream;
    cache.csa_k.resize((size_t) n_layer, nullptr);
    cache.hca_k.resize((size_t) n_layer, nullptr);
    cache.lid_k.resize((size_t) n_layer, nullptr);
    cache.csa_state_kv.resize((size_t) n_layer, nullptr);
    cache.csa_state_score.resize((size_t) n_layer, nullptr);
    cache.hca_state_kv.resize((size_t) n_layer, nullptr);
    cache.hca_state_score.resize((size_t) n_layer, nullptr);
    cache.lid_state_kv.resize((size_t) n_layer, nullptr);
    cache.lid_state_score.resize((size_t) n_layer, nullptr);

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
        const uint32_t ratio = model.hparams.dsv4_compress_ratios[(size_t) il];
        ggml_backend_buffer_type_t buft = llama_dsv4_layer_buft(*this, il);

        if (ratio == dsv4_runtime::CSA_RATIO) {
            cache.csa_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, kv_self.type_k, n_embd_head, csa_kv*n_stream, 1);
            cache.lid_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, cparams.idx_type_k, n_indexer_head, csa_kv*n_stream, 1);
            cache.csa_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_embd_head, 2*dsv4_runtime::CSA_RATIO*n_stream);
            cache.csa_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_embd_head, 2*dsv4_runtime::CSA_RATIO*n_stream);
            cache.lid_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_indexer_head, 2*dsv4_runtime::CSA_RATIO*n_stream);
            cache.lid_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_indexer_head, 2*dsv4_runtime::CSA_RATIO*n_stream);

            if (!alloc_tensor(cache.csa_k[(size_t) il], buft) ||
                !alloc_tensor(cache.lid_k[(size_t) il], buft) ||
                !alloc_tensor(cache.csa_state_kv[(size_t) il], buft) ||
                !alloc_tensor(cache.csa_state_score[(size_t) il], buft) ||
                !alloc_tensor(cache.lid_state_kv[(size_t) il], buft) ||
                !alloc_tensor(cache.lid_state_score[(size_t) il], buft)) {
                LLAMA_LOG_ERROR("%s: failed to allocate DSV4 CSA/LID buffers for layer %d\n", __func__, il);
                free_dsv4_cache_tensors();
                return false;
            }
        } else if (ratio == dsv4_runtime::HCA_RATIO) {
            cache.hca_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, kv_self.type_k, n_embd_head, hca_kv*n_stream, 1);
            cache.hca_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, n_embd_head, dsv4_runtime::HCA_RATIO*n_stream);
            cache.hca_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, n_embd_head, dsv4_runtime::HCA_RATIO*n_stream);

            if (!alloc_tensor(cache.hca_k[(size_t) il], buft) ||
                !alloc_tensor(cache.hca_state_kv[(size_t) il], buft) ||
                !alloc_tensor(cache.hca_state_score[(size_t) il], buft)) {
                LLAMA_LOG_ERROR("%s: failed to allocate DSV4 HCA buffers for layer %d\n", __func__, il);
                free_dsv4_cache_tensors();
                return false;
            }
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

    const size_t csa_k_bytes = bytes(cache.csa_k);
    const size_t hca_k_bytes = bytes(cache.hca_k);
    const size_t lid_k_bytes = bytes(cache.lid_k);
    const size_t csa_state_bytes = bytes(cache.csa_state_kv) + bytes(cache.csa_state_score);
    const size_t hca_state_bytes = bytes(cache.hca_state_kv) + bytes(cache.hca_state_score);
    const size_t lid_state_bytes = bytes(cache.lid_state_kv) + bytes(cache.lid_state_score);

    LLAMA_LOG_INFO("%s: DSV4 cache: CSA K=%7.2f MiB (%s), HCA K=%7.2f MiB (%s), LID K=%7.2f MiB (%s), states=%7.2f MiB, total=%7.2f MiB, streams=%u\n",
            __func__,
            (float) csa_k_bytes / (1024.0f * 1024.0f), ggml_type_name(kv_self.type_k),
            (float) hca_k_bytes / (1024.0f * 1024.0f), ggml_type_name(kv_self.type_k),
            (float) lid_k_bytes / (1024.0f * 1024.0f), ggml_type_name(cparams.idx_type_k),
            (float) (csa_state_bytes + hca_state_bytes + lid_state_bytes) / (1024.0f * 1024.0f),
            (float) (csa_k_bytes + hca_k_bytes + lid_k_bytes + csa_state_bytes + hca_state_bytes + lid_state_bytes) / (1024.0f * 1024.0f),
            n_stream);

    return true;
}

void llama_context::free_dsv4_cache_tensors() {
    auto release_vector = [](auto & v) {
        using vec_type = std::decay_t<decltype(v)>;
        vec_type().swap(v);
    };

    for (ggml_backend_buffer_t buf : dsv4.cache.cache_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    release_vector(dsv4.cache.cache_bufs);
    release_vector(dsv4.cache.csa_k);
    release_vector(dsv4.cache.hca_k);
    release_vector(dsv4.cache.lid_k);
    release_vector(dsv4.cache.csa_state_kv);
    release_vector(dsv4.cache.csa_state_score);
    release_vector(dsv4.cache.hca_state_kv);
    release_vector(dsv4.cache.hca_state_score);
    release_vector(dsv4.cache.lid_state_kv);
    release_vector(dsv4.cache.lid_state_score);
    dsv4.cache.n_stream = 1;
    if (dsv4.cache.cache_ctx != nullptr) {
        ggml_free(dsv4.cache.cache_ctx);
        dsv4.cache.cache_ctx = nullptr;
    }
}

void llama_reset_dsv4_state(llama_context * ctx, int32_t seq_id) {
    if (ctx == nullptr) {
        return;
    }

    const uint32_t n_stream = std::max<uint32_t>(1, ctx->dsv4.cache.n_stream);
    if (seq_id >= (llama_seq_id) n_stream) {
        LLAMA_LOG_ERROR("%s: DSV4 seq_id %d is outside stream range %u\n", __func__, seq_id, n_stream);
        return;
    }

    if (seq_id < 0) {
        for (ggml_backend_buffer_t buf : ctx->dsv4.cache.cache_bufs) {
            ggml_backend_buffer_clear(buf, 0);
        }
        return;
    }

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

    for (ggml_tensor * tensor : ctx->dsv4.cache.csa_k) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.hca_k) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.lid_k) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.csa_state_kv) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.csa_state_score) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.hca_state_kv) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.hca_state_score) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.lid_state_kv) clear_tensor(tensor);
    for (ggml_tensor * tensor : ctx->dsv4.cache.lid_state_score) clear_tensor(tensor);
}

static std::vector<ggml_tensor *> dsv4_state_tensors(const llama_context & ctx) {
    std::vector<ggml_tensor *> tensors;
    const auto append = [&tensors](const std::vector<ggml_tensor *> & group) {
        for (ggml_tensor * tensor : group) {
            if (tensor != nullptr) {
                tensors.push_back(tensor);
            }
        }
    };

    append(ctx.dsv4.cache.csa_state_kv);
    append(ctx.dsv4.cache.csa_state_score);
    append(ctx.dsv4.cache.hca_state_kv);
    append(ctx.dsv4.cache.hca_state_score);
    append(ctx.dsv4.cache.lid_state_kv);
    append(ctx.dsv4.cache.lid_state_score);
    return tensors;
}

void llama_kv_cache::gpu_checkpoint::release_dsv4_per_step() {
    for (ggml_context * shadow_ctx : dsv4_per_step_shadow_ctxs) {
        ggml_free(shadow_ctx);
    }
    for (ggml_backend_buffer_t buffer : dsv4_per_step_shadow_bufs) {
        ggml_backend_buffer_free(buffer);
    }
    dsv4_per_step_shadow_ctxs.clear();
    dsv4_per_step_shadow_bufs.clear();
    dsv4_per_step_state.clear();
    dsv4_per_step_state_shadow.clear();
    dsv4_per_step_delta.clear();
    dsv4_per_step_csa_src.clear();
    dsv4_per_step_csa_dst.clear();
    dsv4_per_step_hca_src.clear();
    dsv4_per_step_hca_dst.clear();
    dsv4_per_step_lid_src.clear();
    dsv4_per_step_lid_dst.clear();
    dsv4_per_step_allocated = false;
    dsv4_per_step_saved = false;
    dsv4_per_step_max_tokens = 0;
    dsv4_per_step_base_bytes = 0;
    dsv4_per_step_delta_bytes = 0;
}

void llama_kv_cache::gpu_checkpoint::release_dsv4_snapshot() {
    for (ggml_context * shadow_ctx : dsv4_shadow_ctxs) {
        ggml_free(shadow_ctx);
    }
    for (ggml_backend_buffer_t buffer : dsv4_shadow_bufs) {
        ggml_backend_buffer_free(buffer);
    }
    dsv4_shadow_ctxs.clear();
    dsv4_shadow_bufs.clear();
    dsv4_state_data.clear();
    dsv4_state_shadow.clear();
    dsv4_shadow_allocated = false;
    dsv4_shadow_saved = false;
}

static bool dsv4_per_step_alloc(llama_context & ctx, int max_tokens) {
    auto & ckpt = ctx.kv_self.ckpt;
    const auto states = dsv4_state_tensors(ctx);
    if (states.empty() || max_tokens <= 0 || max_tokens > DSV4_PER_STEP_MAX_STATE_ROWS) {
        if (max_tokens > DSV4_PER_STEP_MAX_STATE_ROWS) {
            LLAMA_LOG_WARN("%s: DSV4 per-step supports at most %d verification rows; requested %d\n",
                    __func__, DSV4_PER_STEP_MAX_STATE_ROWS, max_tokens);
        }
        return false;
    }
    if (ckpt.dsv4_per_step_allocated && ckpt.dsv4_per_step_max_tokens >= max_tokens &&
        ckpt.dsv4_per_step_state.size() == states.size()) {
        return true;
    }

    ctx.kv_self.ckpt.release_dsv4_per_step();
    ckpt.dsv4_per_step_state = states;
    ckpt.dsv4_per_step_state_shadow.assign(states.size(), nullptr);
    ckpt.dsv4_per_step_delta.assign(states.size(), nullptr);

    struct entry {
        size_t index;
        ggml_tensor * source;
    };
    std::map<ggml_backend_buffer_type_t, std::vector<entry>> entries_by_buft;
    for (size_t i = 0; i < states.size(); ++i) {
        ggml_tensor * source = states[i];
        if (source == nullptr || source->buffer == nullptr) {
            ctx.kv_self.ckpt.release_dsv4_per_step();
            return false;
        }
        entries_by_buft[ggml_backend_buffer_get_type(source->buffer)].push_back({ i, source });
        ckpt.dsv4_per_step_base_bytes += ggml_nbytes(source);
        ckpt.dsv4_per_step_delta_bytes += ggml_row_size(source->type, source->ne[0]) * (size_t) max_tokens;
    }

    for (auto & [buft, entries] : entries_by_buft) {
        ggml_init_params params = {
            /*.mem_size   =*/ entries.size() * 3 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * graph_ctx = ggml_init(params);
        if (graph_ctx == nullptr) {
            ctx.kv_self.ckpt.release_dsv4_per_step();
            return false;
        }

        for (const entry & item : entries) {
            ggml_tensor * shadow = ggml_dup_tensor(graph_ctx, item.source);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                shadow->nb[d] = item.source->nb[d];
            }
            ggml_format_name(shadow, "dsv4_per_step_base_%zu", item.index);

            ggml_tensor * delta = ggml_new_tensor_2d(graph_ctx, item.source->type,
                    item.source->ne[0], max_tokens);
            ggml_format_name(delta, "dsv4_per_step_delta_%zu", item.index);
            ckpt.dsv4_per_step_state_shadow[item.index] = shadow;
            ckpt.dsv4_per_step_delta[item.index] = delta;
        }

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(graph_ctx, buft);
        if (buffer == nullptr) {
            ggml_free(graph_ctx);
            ctx.kv_self.ckpt.release_dsv4_per_step();
            return false;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_buffer_clear(buffer, 0);
        ckpt.dsv4_per_step_shadow_ctxs.push_back(graph_ctx);
        ckpt.dsv4_per_step_shadow_bufs.push_back(buffer);
    }

    ckpt.dsv4_per_step_max_tokens = max_tokens;
    ckpt.dsv4_per_step_allocated = true;
    LLAMA_LOG_INFO("%s: DSV4 per-step base=%8.2f MiB delta=%8.2f MiB max_tokens=%d\n",
            __func__, ckpt.dsv4_per_step_base_bytes / (1024.0 * 1024.0),
            ckpt.dsv4_per_step_delta_bytes / (1024.0 * 1024.0), max_tokens);
    return true;
}

static bool dsv4_per_step_copy_base(llama_context & ctx, bool restore) {
    auto & ckpt = ctx.kv_self.ckpt;
    if (!ckpt.dsv4_per_step_allocated || ckpt.dsv4_per_step_state.size() != ckpt.dsv4_per_step_state_shadow.size()) {
        return false;
    }

    std::vector<ggml_backend_t> backends;
    for (size_t i = 0; i < ckpt.dsv4_per_step_state.size(); ++i) {
        ggml_tensor * state = ckpt.dsv4_per_step_state[i];
        ggml_tensor * shadow = ckpt.dsv4_per_step_state_shadow[i];
        ggml_backend_t backend = state != nullptr
                ? ggml_backend_sched_get_tensor_backend(ctx.sched, state)
                : nullptr;
        if (state == nullptr || shadow == nullptr) {
            return false;
        }
        if (backend == nullptr) {
            if (state->buffer == nullptr || shadow->buffer == nullptr) {
                return false;
            }
            ggml_backend_tensor_copy(restore ? shadow : state, restore ? state : shadow);
            continue;
        }
        if (restore) {
            ggml_backend_tensor_copy_async(backend, backend, shadow, state);
        } else {
            ggml_backend_tensor_copy_async(backend, backend, state, shadow);
        }
        if (std::find(backends.begin(), backends.end(), backend) == backends.end()) {
            backends.push_back(backend);
        }
    }
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
    return true;
}

static bool dsv4_per_step_capture_group(
        llama_context & ctx,
        const std::vector<ggml_tensor *> & states,
        const llama_context::dsv4_runtime::comp_plan & plan) {
    auto & ckpt = ctx.kv_self.ckpt;
    if (plan.state_delta_src_idxs.size() != plan.state_delta_dst_idxs.size() ||
        plan.state_delta_src_idxs.size() > (size_t) ckpt.dsv4_per_step_max_tokens) {
        return false;
    }

    for (ggml_tensor * state : states) {
        if (state == nullptr) {
            continue;
        }
        ggml_tensor * delta = llama_dsv4_spec_ckpt_delta(&ctx, state);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx.sched, state);
        if (delta == nullptr || backend == nullptr || delta->ne[0] != state->ne[0]) {
            return false;
        }

        for (size_t row = 0; row < plan.state_delta_src_idxs.size(); ++row) {
            const int32_t src_idx = plan.state_delta_src_idxs[row];
            const int32_t dst_idx = plan.state_delta_dst_idxs[row];
            if (src_idx < 0 || (uint64_t) src_idx >= (uint64_t) delta->ne[1] ||
                dst_idx < 0 || (uint64_t) dst_idx >= (uint64_t) state->ne[1]) {
                return false;
            }

            ggml_tensor src_view = *state;
            ggml_tensor dst_view = *delta;
            src_view.ne[1] = src_view.ne[2] = src_view.ne[3] = 1;
            dst_view.ne[1] = dst_view.ne[2] = dst_view.ne[3] = 1;
            src_view.nb[2] = src_view.nb[3] = src_view.nb[1];
            dst_view.nb[2] = dst_view.nb[3] = dst_view.nb[1];
            src_view.data = (char *) state->data + (size_t) dst_idx * state->nb[1];
            dst_view.data = (char *) delta->data + (size_t) src_idx * delta->nb[1];
            src_view.view_src = nullptr;
            dst_view.view_src = nullptr;
            src_view.view_offs = 0;
            dst_view.view_offs = 0;
            ggml_backend_tensor_copy_async(backend, backend, &src_view, &dst_view);
        }
    }

    return true;
}

bool llama_dsv4_spec_ckpt_capture_rows(llama_context * ctx) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_DEEPSEEK4) {
        return true;
    }

    const auto & ckpt = ctx->kv_self.ckpt;
    if (ckpt.selected_spec_mode != LLAMA_SPEC_CKPT_PER_STEP ||
        !ckpt.dsv4_per_step_allocated || !ckpt.dsv4_per_step_saved) {
        return true;
    }

    const bool ok =
        dsv4_per_step_capture_group(*ctx, ctx->dsv4.cache.csa_state_kv,    ctx->dsv4.csa_plan) &&
        dsv4_per_step_capture_group(*ctx, ctx->dsv4.cache.csa_state_score, ctx->dsv4.csa_plan) &&
        dsv4_per_step_capture_group(*ctx, ctx->dsv4.cache.hca_state_kv,    ctx->dsv4.hca_plan) &&
        dsv4_per_step_capture_group(*ctx, ctx->dsv4.cache.hca_state_score, ctx->dsv4.hca_plan) &&
        dsv4_per_step_capture_group(*ctx, ctx->dsv4.cache.lid_state_kv,    ctx->dsv4.lid_plan) &&
        dsv4_per_step_capture_group(*ctx, ctx->dsv4.cache.lid_state_score, ctx->dsv4.lid_plan);
    if (!ok) {
        LLAMA_LOG_ERROR("%s: failed to queue DSV4 per-step compressor-state row capture\n", __func__);
    }
    return ok;
}

static bool dsv4_spec_ckpt_alloc_gpu(
        llama_context & ctx,
        const std::vector<ggml_tensor *> & tensors) {
    auto & ckpt = ctx.kv_self.ckpt;
    if (ckpt.dsv4_shadow_allocated) {
        return ckpt.dsv4_state_shadow.size() == tensors.size();
    }

    struct tensor_entry {
        size_t index;
        ggml_tensor * source;
    };
    std::map<ggml_backend_buffer_type_t, std::vector<tensor_entry>> entries_by_buft;
    const auto release_partial = [&]() {
        ckpt.release_dsv4_snapshot();
    };

    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }
        if (tensor->buffer == nullptr) {
            return false;
        }
        entries_by_buft[ggml_backend_buffer_get_type(tensor->buffer)].push_back({ i, tensor });
    }

    ckpt.dsv4_state_shadow.assign(tensors.size(), nullptr);
    for (auto & [buft, entries] : entries_by_buft) {
        ggml_init_params params = {
            /*.mem_size   =*/ entries.size() * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * shadow_ctx = ggml_init(params);
        if (shadow_ctx == nullptr) {
            release_partial();
            return false;
        }

        for (const auto & entry : entries) {
            ggml_tensor * shadow = ggml_dup_tensor(shadow_ctx, entry.source);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                shadow->nb[d] = entry.source->nb[d];
            }
            ggml_format_name(shadow, "dsv4_spec_shadow_%zu", entry.index);
            ckpt.dsv4_state_shadow[entry.index] = shadow;
        }

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(shadow_ctx, buft);
        if (buffer == nullptr) {
            ggml_free(shadow_ctx);
            release_partial();
            return false;
        }
        ggml_backend_buffer_clear(buffer, 0);
        LLAMA_LOG_INFO("%s: %10s DSV4 speculative shadow buffer = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buffer),
                ggml_backend_buffer_get_size(buffer) / 1024.0 / 1024.0);
        ckpt.dsv4_shadow_ctxs.push_back(shadow_ctx);
        ckpt.dsv4_shadow_bufs.push_back(buffer);
    }

    ckpt.dsv4_shadow_allocated = true;
    return true;
}

static bool dsv4_spec_ckpt_copy_gpu(
        llama_context & ctx,
        const std::vector<ggml_tensor *> & tensors,
        bool restore) {
    auto & ckpt = ctx.kv_self.ckpt;
    if (!ckpt.dsv4_shadow_allocated || ckpt.dsv4_state_shadow.size() != tensors.size()) {
        return false;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor * tensor = tensors[i];
        ggml_tensor * shadow = ckpt.dsv4_state_shadow[i];
        if (tensor == nullptr || shadow == nullptr) {
            continue;
        }

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx.sched, tensor);
        if (backend == nullptr) {
            return false;
        }
        if (restore) {
            ggml_backend_tensor_copy_async(backend, backend, shadow, tensor);
        } else {
            ggml_backend_tensor_copy_async(backend, backend, tensor, shadow);
        }
    }
    return true;
}

bool llama_dsv4_spec_ckpt_prepare(llama_context * ctx, int mode, int max_tokens) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_DEEPSEEK4) {
        return true;
    }

    if (mode == LLAMA_SPEC_CKPT_PER_STEP) {
        return dsv4_per_step_alloc(*ctx, max_tokens);
    }
    if (mode == LLAMA_SPEC_CKPT_GPU_FALLBACK) {
        return dsv4_spec_ckpt_alloc_gpu(*ctx, dsv4_state_tensors(*ctx));
    }
    return true;
}

bool llama_dsv4_spec_ckpt_save(llama_context * ctx, bool use_gpu) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_DEEPSEEK4) {
        return true;
    }

    if (ctx->kv_self.ckpt.selected_spec_mode == LLAMA_SPEC_CKPT_PER_STEP) {
        auto & ckpt = ctx->kv_self.ckpt;
        ckpt.dsv4_per_step_saved = false;
        ckpt.dsv4_per_step_csa_src.clear();
        ckpt.dsv4_per_step_csa_dst.clear();
        ckpt.dsv4_per_step_hca_src.clear();
        ckpt.dsv4_per_step_hca_dst.clear();
        ckpt.dsv4_per_step_lid_src.clear();
        ckpt.dsv4_per_step_lid_dst.clear();
        if (!use_gpu || !dsv4_per_step_copy_base(*ctx, false)) {
            LLAMA_LOG_ERROR("%s: failed to save DSV4 per-step compressor-state base\n", __func__);
            return false;
        }
        ckpt.dsv4_per_step_saved = true;
        return true;
    }

    const auto tensors = dsv4_state_tensors(*ctx);
    ctx->kv_self.ckpt.dsv4_shadow_saved = false;
    if (use_gpu) {
        if (!dsv4_spec_ckpt_alloc_gpu(*ctx, tensors) || !dsv4_spec_ckpt_copy_gpu(*ctx, tensors, false)) {
            LLAMA_LOG_ERROR("%s: failed to save DSV4 gpu-fallback checkpoint; explicit GPU mode will not downgrade to CPU\n", __func__);
            return false;
        }
        ctx->kv_self.ckpt.dsv4_state_data.clear();
        ctx->kv_self.ckpt.dsv4_shadow_saved = true;
        return true;
    }

    auto & saved = ctx->kv_self.ckpt.dsv4_state_data;
    saved.clear();
    for (ggml_tensor * tensor : tensors) {
        if (tensor == nullptr) {
            saved.emplace_back();
            continue;
        }

        const size_t nbytes = ggml_nbytes(tensor);
        saved.emplace_back(nbytes);
        ggml_backend_tensor_get(tensor, saved.back().data(), 0, nbytes);
    }

    return true;
}

static enum llama_spec_ckpt_restore_result dsv4_per_step_restore_rows(
        llama_context & ctx,
        const std::vector<ggml_tensor *> & states,
        size_t delta_offset,
        const std::vector<ggml_tensor *> & deltas,
        const std::vector<int32_t> & src_idxs,
        const std::vector<int32_t> & dst_idxs,
        int accepted_step,
        std::vector<ggml_backend_t> & backends) {
    auto & ckpt = ctx.kv_self.ckpt;
    if (src_idxs.size() != dst_idxs.size() || src_idxs.size() > (size_t) ckpt.dsv4_per_step_max_tokens ||
        delta_offset > deltas.size() || states.size() > deltas.size() - delta_offset) {
        LLAMA_LOG_ERROR("%s: invalid DSV4 per-step row restore: states=%zu delta_offset=%zu deltas=%zu src=%zu dst=%zu max=%d\n",
                __func__, states.size(), delta_offset, deltas.size(), src_idxs.size(), dst_idxs.size(),
                ckpt.dsv4_per_step_max_tokens);
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }

    for (size_t i = 0; i < states.size(); ++i) {
        ggml_tensor * state = states[i];
        ggml_tensor * delta = deltas[delta_offset + i];
        if (state == nullptr || delta == nullptr) {
            return LLAMA_SPEC_CKPT_RESTORE_FAILED;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx.sched, state);
        if (backend == nullptr && (state->buffer == nullptr || delta->buffer == nullptr)) {
            return LLAMA_SPEC_CKPT_RESTORE_FAILED;
        }

        for (size_t row = 0; row < src_idxs.size(); ++row) {
            if (src_idxs[row] > accepted_step) {
                continue;
            }
            // Reject invalid mappings instead of leaving stale compressor state.
            if (src_idxs[row] < 0 || dst_idxs[row] < 0 ||
                (uint64_t) dst_idxs[row] >= (uint64_t) state->ne[1]) {
                LLAMA_LOG_ERROR("%s: invalid visible DSV4 state row src=%d dst=%d accepted_step=%d state_rows=%lld\n",
                        __func__, src_idxs[row], dst_idxs[row], accepted_step, (long long) state->ne[1]);
                return LLAMA_SPEC_CKPT_RESTORE_FAILED;
            }
            ggml_tensor src_view = *delta;
            ggml_tensor dst_view = *state;
            src_view.ne[1] = src_view.ne[2] = src_view.ne[3] = 1;
            dst_view.ne[1] = dst_view.ne[2] = dst_view.ne[3] = 1;
            src_view.nb[2] = src_view.nb[3] = src_view.nb[1];
            dst_view.nb[2] = dst_view.nb[3] = dst_view.nb[1];
            src_view.data = (char *) delta->data + (size_t) src_idxs[row] * delta->nb[1];
            dst_view.data = (char *) state->data + (size_t) dst_idxs[row] * state->nb[1];
            src_view.view_src = nullptr;
            dst_view.view_src = nullptr;
            src_view.view_offs = 0;
            dst_view.view_offs = 0;
            if (backend != nullptr) {
                ggml_backend_tensor_copy_async(backend, backend, &src_view, &dst_view);
            } else {
                ggml_backend_tensor_copy(&src_view, &dst_view);
            }
        }
        if (backend != nullptr && std::find(backends.begin(), backends.end(), backend) == backends.end()) {
            backends.push_back(backend);
        }
    }
    return LLAMA_SPEC_CKPT_RESTORE_DIRECT;
}

enum llama_spec_ckpt_restore_result llama_dsv4_spec_ckpt_restore(llama_context * ctx, bool use_gpu, int accepted_step) {
    if (ctx == nullptr || ctx->model.arch != LLM_ARCH_DEEPSEEK4) {
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }

    auto & ckpt = ctx->kv_self.ckpt;
    if (ckpt.selected_spec_mode == LLAMA_SPEC_CKPT_PER_STEP) {
        if (!ckpt.dsv4_per_step_saved || !dsv4_per_step_copy_base(*ctx, true)) {
            LLAMA_LOG_ERROR("%s: failed to restore DSV4 per-step compressor-state base\n", __func__);
            return LLAMA_SPEC_CKPT_RESTORE_FAILED;
        }

        const auto compact = [](const std::vector<ggml_tensor *> & source) {
            std::vector<ggml_tensor *> result;
            for (ggml_tensor * tensor : source) {
                if (tensor != nullptr) {
                    result.push_back(tensor);
                }
            }
            return result;
        };
        const auto csa_kv = compact(ctx->dsv4.cache.csa_state_kv);
        const auto csa_score = compact(ctx->dsv4.cache.csa_state_score);
        const auto hca_kv = compact(ctx->dsv4.cache.hca_state_kv);
        const auto hca_score = compact(ctx->dsv4.cache.hca_state_score);
        const auto lid_kv = compact(ctx->dsv4.cache.lid_state_kv);
        const auto lid_score = compact(ctx->dsv4.cache.lid_state_score);
        const size_t csa_kv_off = 0;
        const size_t csa_score_off = csa_kv_off + csa_kv.size();
        const size_t hca_kv_off = csa_score_off + csa_score.size();
        const size_t hca_score_off = hca_kv_off + hca_kv.size();
        const size_t lid_kv_off = hca_score_off + hca_score.size();
        const size_t lid_score_off = lid_kv_off + lid_kv.size();
        if (ckpt.dsv4_per_step_delta.size() != lid_score_off + lid_score.size()) {
            LLAMA_LOG_ERROR("%s: DSV4 per-step delta tensor layout mismatch\n", __func__);
            return LLAMA_SPEC_CKPT_RESTORE_FAILED;
        }

        std::vector<ggml_backend_t> backends;
        const auto restore_group = [&](const std::vector<ggml_tensor *> & states, size_t offset) {
            return dsv4_per_step_restore_rows(*ctx, states, offset, ckpt.dsv4_per_step_delta,
                    offset == csa_kv_off || offset == csa_score_off ? ckpt.dsv4_per_step_csa_src :
                    offset == hca_kv_off || offset == hca_score_off ? ckpt.dsv4_per_step_hca_src : ckpt.dsv4_per_step_lid_src,
                    offset == csa_kv_off || offset == csa_score_off ? ckpt.dsv4_per_step_csa_dst :
                    offset == hca_kv_off || offset == hca_score_off ? ckpt.dsv4_per_step_hca_dst : ckpt.dsv4_per_step_lid_dst,
                    accepted_step, backends);
        };

        if (restore_group(csa_kv, csa_kv_off) == LLAMA_SPEC_CKPT_RESTORE_FAILED ||
            restore_group(csa_score, csa_score_off) == LLAMA_SPEC_CKPT_RESTORE_FAILED ||
            restore_group(hca_kv, hca_kv_off) == LLAMA_SPEC_CKPT_RESTORE_FAILED ||
            restore_group(hca_score, hca_score_off) == LLAMA_SPEC_CKPT_RESTORE_FAILED ||
            restore_group(lid_kv, lid_kv_off) == LLAMA_SPEC_CKPT_RESTORE_FAILED ||
            restore_group(lid_score, lid_score_off) == LLAMA_SPEC_CKPT_RESTORE_FAILED) {
            return LLAMA_SPEC_CKPT_RESTORE_FAILED;
        }
        for (ggml_backend_t backend : backends) {
            ggml_backend_synchronize(backend);
        }
        return LLAMA_SPEC_CKPT_RESTORE_DIRECT;
    }

    const auto tensors = dsv4_state_tensors(*ctx);
    if (use_gpu && ctx->kv_self.ckpt.dsv4_shadow_saved) {
        return dsv4_spec_ckpt_copy_gpu(*ctx, tensors, true)
            ? LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED
            : LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }

    const auto & saved = ctx->kv_self.ckpt.dsv4_state_data;
    if (saved.size() != tensors.size()) {
        LLAMA_LOG_ERROR("%s: DSV4 checkpoint tensor count mismatch: saved=%zu current=%zu\n",
                __func__, saved.size(), tensors.size());
        return LLAMA_SPEC_CKPT_RESTORE_FAILED;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        ggml_tensor * tensor = tensors[i];
        if (tensor == nullptr) {
            if (!saved[i].empty()) {
                LLAMA_LOG_ERROR("%s: DSV4 checkpoint null tensor %zu has saved data\n", __func__, i);
                return LLAMA_SPEC_CKPT_RESTORE_FAILED;
            }
            continue;
        }
        if (saved[i].size() != ggml_nbytes(tensor)) {
            LLAMA_LOG_ERROR("%s: DSV4 checkpoint tensor %zu size mismatch\n", __func__, i);
            return LLAMA_SPEC_CKPT_RESTORE_FAILED;
        }
        if (!saved[i].empty()) {
            ggml_backend_tensor_set(tensor, saved[i].data(), 0, saved[i].size());
        }
    }

    return LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED;
}

ggml_tensor * llama_dsv4_spec_ckpt_delta(llama_context * ctx, ggml_tensor * state_tensor) {
    if (ctx == nullptr || state_tensor == nullptr ||
        ctx->kv_self.ckpt.selected_spec_mode != LLAMA_SPEC_CKPT_PER_STEP ||
        !ctx->kv_self.ckpt.dsv4_per_step_allocated) {
        return nullptr;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    for (size_t i = 0; i < ckpt.dsv4_per_step_state.size(); ++i) {
        if (ckpt.dsv4_per_step_state[i] == state_tensor) {
            return ckpt.dsv4_per_step_delta[i];
        }
    }
    return nullptr;
}

void llama_dsv4_spec_ckpt_record_plan(llama_context * ctx) {
    if (ctx == nullptr || ctx->kv_self.ckpt.selected_spec_mode != LLAMA_SPEC_CKPT_PER_STEP) {
        return;
    }
    auto & ckpt = ctx->kv_self.ckpt;
    ckpt.dsv4_per_step_csa_src = ctx->dsv4.csa_plan.state_delta_src_idxs;
    ckpt.dsv4_per_step_csa_dst = ctx->dsv4.csa_plan.state_delta_dst_idxs;
    ckpt.dsv4_per_step_hca_src = ctx->dsv4.hca_plan.state_delta_src_idxs;
    ckpt.dsv4_per_step_hca_dst = ctx->dsv4.hca_plan.state_delta_dst_idxs;
    ckpt.dsv4_per_step_lid_src = ctx->dsv4.lid_plan.state_delta_src_idxs;
    ckpt.dsv4_per_step_lid_dst = ctx->dsv4.lid_plan.state_delta_dst_idxs;
}

void llama_dsv4_spec_ckpt_discard(llama_context * ctx) {
    if (ctx != nullptr) {
        ctx->kv_self.ckpt.dsv4_state_data.clear();
        ctx->kv_self.ckpt.dsv4_shadow_saved = false;
        ctx->kv_self.ckpt.dsv4_per_step_saved = false;
        ctx->kv_self.ckpt.dsv4_per_step_csa_src.clear();
        ctx->kv_self.ckpt.dsv4_per_step_csa_dst.clear();
        ctx->kv_self.ckpt.dsv4_per_step_hca_src.clear();
        ctx->kv_self.ckpt.dsv4_per_step_hca_dst.clear();
        ctx->kv_self.ckpt.dsv4_per_step_lid_src.clear();
        ctx->kv_self.ckpt.dsv4_per_step_lid_dst.clear();
    }
}

bool llama_prepare_dsv4_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan) {
    if (lctx.model.arch != LLM_ARCH_DEEPSEEK4) {
        return true;
    }

    if (!dsv4_validate_batch_seq_ids(lctx, batch)) {
        return false;
    }

    // Standalone companions contain only the predictor block, skip target state planning.
    const bool is_dsv4_mtp = lctx.model.mtp &&
        lctx.cparams.mtp_op_type != MTP_OP_NONE &&
        lctx.model.hparams.nextn_predict_layers > 0 &&
        lctx.model.hparams.dsv4_compress_ratios[(size_t) (lctx.model.hparams.n_layer - lctx.model.hparams.nextn_predict_layers)] == 0;
    if (is_dsv4_mtp) {
        lctx.dsv4.raw = {};
        if (!reserve_plan && !dsv4_build_raw_context(lctx, batch, lctx.dsv4.raw)) {
            return false;
        }
        lctx.dsv4.csa_plan = {};
        lctx.dsv4.hca_plan = {};
        lctx.dsv4.lid_plan = {};
        lctx.dsv4.csa_ctx = {};
        lctx.dsv4.hca_ctx = {};
        lctx.dsv4.lid_ctx = {};

        if (set_tensors) {
            dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_write_src_idxs, lctx.dsv4.raw.write_src_idxs);
            dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_write_idxs, lctx.dsv4.raw.write_dst_idxs);
            dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_read_idxs, lctx.dsv4.raw.read_dst_idxs);
        }
        return true;
    }

    if (!lctx.ensure_dsv4_cache_tensors()) {
        return false;
    }

    const uint32_t cache_n_stream = std::max<uint32_t>(1, lctx.dsv4.cache.n_stream);
    const uint32_t csa_kv_size = dsv4_cache_kv_size(lctx.dsv4.cache.csa_k)/cache_n_stream;
    const uint32_t hca_kv_size = dsv4_cache_kv_size(lctx.dsv4.cache.hca_k)/cache_n_stream;
    const uint32_t lid_kv_size = dsv4_cache_kv_size(lctx.dsv4.cache.lid_k)/cache_n_stream;
    const uint32_t csa_state_size = dsv4_cache_state_size(lctx.dsv4.cache.csa_state_kv)/cache_n_stream;
    const uint32_t hca_state_size = dsv4_cache_state_size(lctx.dsv4.cache.hca_state_kv)/cache_n_stream;
    const uint32_t lid_state_size = dsv4_cache_state_size(lctx.dsv4.cache.lid_state_kv)/cache_n_stream;

    const auto build_plan = [&](uint32_t ratio, bool overlap, uint32_t state_size, uint32_t kv_size, uint32_t n_stream) {
        return reserve_plan
            ? dsv4_build_reserve_comp_plan(batch, ratio, overlap, state_size, kv_size, n_stream)
            : dsv4_build_comp_plan(batch, ratio, overlap, state_size, kv_size, n_stream);
    };

    lctx.dsv4.raw = {};
    if (!reserve_plan && !dsv4_build_raw_context(lctx, batch, lctx.dsv4.raw)) {
        return false;
    }

    //auto tim1 = ggml_time_us();
    lctx.dsv4.csa_plan = build_plan(llama_context::dsv4_runtime::CSA_RATIO, true, csa_state_size, csa_kv_size, cache_n_stream);
    lctx.dsv4.hca_plan = build_plan(llama_context::dsv4_runtime::HCA_RATIO, false, hca_state_size, hca_kv_size, cache_n_stream);
    lctx.dsv4.lid_plan = build_plan(llama_context::dsv4_runtime::CSA_RATIO, true, lid_state_size, lid_kv_size, cache_n_stream);
    lctx.dsv4.csa_ctx = dsv4_build_comp_context(batch, cache_n_stream, lctx.dsv4.csa_plan.n_kv);
    lctx.dsv4.hca_ctx = dsv4_build_comp_context(batch, cache_n_stream, lctx.dsv4.hca_plan.n_kv);
    lctx.dsv4.lid_ctx = dsv4_build_comp_context(batch, cache_n_stream, lctx.dsv4.lid_plan.n_kv);
    //auto tim2 = ggml_time_us();
    //fprintf(stderr, "%s: %ld us to buils plans\n", __func__, tim2-tim1);

    if (!dsv4_validate_comp_plan("csa", batch, lctx.dsv4.csa_plan, llama_context::dsv4_runtime::CSA_RATIO, true, csa_state_size, csa_kv_size, cache_n_stream) ||
        !dsv4_validate_comp_plan("hca", batch, lctx.dsv4.hca_plan, llama_context::dsv4_runtime::HCA_RATIO, false, hca_state_size, hca_kv_size, cache_n_stream) ||
        !dsv4_validate_comp_plan("lid", batch, lctx.dsv4.lid_plan, llama_context::dsv4_runtime::CSA_RATIO, true, lid_state_size, lid_kv_size, cache_n_stream) ||
        !dsv4_validate_csa_lid_visibility(lctx, csa_kv_size, lid_kv_size)) {
        return false;
    }

    if (!set_tensors) {
        return true;
    }

    //tim1 = ggml_time_us();

    dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_write_src_idxs, lctx.dsv4.raw.write_src_idxs);
    dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_write_idxs, lctx.dsv4.raw.write_dst_idxs);
    dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_read_idxs, lctx.dsv4.raw.read_dst_idxs);

    auto set_comp = [&](llama_context::dsv4_runtime::comp_inputs & inputs, llama_context::dsv4_runtime::comp_plan & plan, bool set_mask) {
        dsv4_set_input_tensor(inputs.state_pos, plan.state_pos);
        dsv4_set_input_tensor(inputs.state_persist_src_idxs, plan.state_persist_src_idxs);
        dsv4_set_input_tensor(inputs.state_persist_dst_idxs, plan.state_persist_dst_idxs);
        dsv4_set_input_tensor(inputs.state_read_idxs, plan.state_read_idxs);
        dsv4_set_input_tensor(inputs.state_write_idxs, plan.state_write_idxs);
        dsv4_set_input_tensor(inputs.state_write_pos, plan.state_write_pos);
        if (set_mask) {
            dsv4_set_mask_tensor(inputs.kq_mask, plan, batch.n_tokens);
        }
    };

    set_comp(lctx.dsv4.inputs.csa, lctx.dsv4.csa_plan, true);
    set_comp(lctx.dsv4.inputs.hca, lctx.dsv4.hca_plan, true);
    set_comp(lctx.dsv4.inputs.lid, lctx.dsv4.lid_plan, false);
    llama_dsv4_spec_ckpt_record_plan(&lctx);

    //tim2 = ggml_time_us();
    //fprintf(stderr, "%s: setting input tensors took %ld us\n", __func__, tim2 - tim1);

    return true;
}
