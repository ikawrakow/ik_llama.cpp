#include "llama-dsv4.h"

#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace {

bool dsv4_env_flag_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && *env != '\0' &&
            std::strcmp(env, "0") != 0 &&
            std::strcmp(env, "false") != 0 &&
            std::strcmp(env, "off") != 0;
}

std::string dsv4_trace_path() {
    const char * env = std::getenv("LLAMA_DSV4_TRACE");
    if (env == nullptr || *env == '\0') {
        return std::string();
    }

    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0) {
        const char * override_path = std::getenv("LLAMA_DSV4_TRACE_PATH");
        return override_path != nullptr && *override_path != '\0'
            ? override_path
            : "/tmp/llama_dsv4_trace.jsonl";
    }

    return env;
}

std::mutex & dsv4_trace_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::string dsv4_json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

template<typename T>
void dsv4_json_append_array(std::ostringstream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

void dsv4_trace_emit_tokens(
        const llama_context & lctx,
        const llama_batch & batch,
        const std::vector<int32_t> & raw_k_write_idxs,
        const std::vector<int32_t> & raw_k_read_idxs) {
    std::ostringstream out;
    out << "{\"event\":\"tokens\",\"n_tokens\":" << batch.n_tokens
        << ",\"kv_head\":" << lctx.kv_self.head
        << ",\"raw_k_write_idxs\":";
    dsv4_json_append_array(out, raw_k_write_idxs);
    out << ",\"raw_k_read_idxs\":";
    dsv4_json_append_array(out, raw_k_read_idxs);
    out << ",\"positions\":[";
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (i != 0) {
            out << ',';
        }
        const llama_pos pos = batch.pos != nullptr
                ? batch.pos[i]
                : batch.all_pos_0 + i*batch.all_pos_1;
        out << pos;
    }
    out << "]";
    if (batch.token != nullptr) {
        out << ",\"tokens\":[";
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (i != 0) {
                out << ',';
            }
            out << batch.token[i];
        }
        out << "]";
    }
    out << "}";
    llama_dsv4_trace_jsonl(out.str());
}

void dsv4_trace_emit_plan(
        const char * tag,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        const llama_context::dsv4_runtime::comp_plan & plan) {
    std::ostringstream out;
    out << "{\"event\":\"plan\",\"tag\":\"" << dsv4_json_escape(tag)
        << "\",\"ratio\":" << ratio
        << ",\"overlap\":" << (overlap ? "true" : "false")
        << ",\"state_size\":" << state_size
        << ",\"kv_size\":" << kv_size
        << ",\"n_stream\":" << plan.n_stream
        << ",\"n_kv\":" << plan.n_kv
        << ",\"state_pos\":";
    dsv4_json_append_array(out, plan.state_pos);
    out << ",\"state_persist_src_idxs\":";
    dsv4_json_append_array(out, plan.state_persist_src_idxs);
    out << ",\"state_persist_dst_idxs\":";
    dsv4_json_append_array(out, plan.state_persist_dst_idxs);
    out << ",\"state_read_idxs\":";
    dsv4_json_append_array(out, plan.state_read_idxs);
    out << ",\"state_write_idxs\":";
    dsv4_json_append_array(out, plan.state_write_idxs);
    out << ",\"state_write_pos\":";
    dsv4_json_append_array(out, plan.state_write_pos);
    out << ",\"n_visible\":";
    dsv4_json_append_array(out, plan.n_visible);
    out << "}";
    llama_dsv4_trace_jsonl(out.str());
}

} // namespace

bool llama_dsv4_trace_enabled() {
    return dsv4_env_flag_enabled("LLAMA_DSV4_TRACE");
}

void llama_dsv4_trace_jsonl(const std::string & record) {
    if (!llama_dsv4_trace_enabled()) {
        return;
    }

    const std::string path = dsv4_trace_path();
    if (path.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(dsv4_trace_mutex());

    if (path == "-") {
        std::fputs(record.c_str(), stderr);
        std::fputc('\n', stderr);
        return;
    }

    std::error_code ec;
    const std::filesystem::path fs_path(path);
    if (fs_path.has_parent_path()) {
        std::filesystem::create_directories(fs_path.parent_path(), ec);
    }

    FILE * fp = std::fopen(path.c_str(), "ab");
    if (fp == nullptr) {
        return;
    }

    std::fwrite(record.data(), 1, record.size(), fp);
    std::fputc('\n', fp);
    std::fclose(fp);
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

static int64_t dsv4_k_rot_size(int64_t n_embd) {
    if (n_embd < 64 || n_embd % 64 != 0) {
        return 0;
    }

    int64_t n_rot = 64;
    do {
        n_rot *= 2;
    } while (n_embd % n_rot == 0);

    return n_rot / 2;
}

static void dsv4_build_hadamard_matrix(std::vector<float> & storage, int64_t n_rot) {
    if (n_rot <= 0) {
        storage.clear();
        return;
    }

    const size_t n_elem = (size_t) n_rot*(size_t) n_rot;
    if (storage.size() == n_elem) {
        return;
    }

    storage.assign(n_elem, 0.0f);
    storage[0] = 1.0f;

    for (int64_t size = 1; size < n_rot; size *= 2) {
        for (int64_t row = 0; row < size; ++row) {
            const float * src = storage.data() + row*n_rot;
            float * dst_top = storage.data() + row*n_rot + size;
            float * dst_bottom = storage.data() + (row + size)*n_rot;
            float * dst_bottom_right = dst_bottom + size;

            for (int64_t col = 0; col < size; ++col) {
                const float v = src[col];
                dst_top[col] = v;
                dst_bottom[col] = v;
                dst_bottom_right[col] = -v;
            }
        }
    }

    const float scale = 1.0f / std::sqrt((float) n_rot);
    for (float & v : storage) {
        v *= scale;
    }
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

static bool dsv4_batch_single_stream_compatible(const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return true;
    }

    llama_seq_id seq0 = -1;
    bool seen = false;

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr) {
            return false;
        }

        const llama_seq_id seq_id = batch.seq_id[i][0];
        if (!seen) {
            seq0 = seq_id;
            seen = true;
        } else if (seq_id != seq0) {
            return false;
        }
    }

    return true;
}

static llama_seq_id dsv4_batch_single_stream_seq_id(const llama_batch & batch) {
    if (batch.n_tokens <= 0 || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return 0;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] > 0 && batch.seq_id[i] != nullptr) {
            return batch.seq_id[i][0];
        }
    }

    return 0;
}

static bool dsv4_validate_single_stream_raw_layout(
        const llama_context & lctx,
        const llama_batch   & batch) {
    const llama_kv_cache & kv = lctx.kv_self;
    const llama_seq_id seq_id = dsv4_batch_single_stream_seq_id(batch);

    uint32_t live_prefix = 0;
    while (live_prefix < kv.size) {
        const llama_kv_cell & cell = kv.cells[live_prefix];
        if (cell.pos < 0 || cell.is_empty()) {
            break;
        }
        ++live_prefix;
    }

    if (live_prefix != kv.used) {
        LLAMA_LOG_ERROR("%s: DSV4 single-stream path requires a contiguous live raw-K prefix, but live_prefix=%u while kv.used=%u\n",
                __func__, live_prefix, kv.used);
        return false;
    }

    if (kv.n < live_prefix) {
        LLAMA_LOG_ERROR("%s: DSV4 raw read span %u is smaller than live raw prefix %u\n",
                __func__, kv.n, live_prefix);
        return false;
    }

    for (uint32_t i = 0; i < live_prefix; ++i) {
        const llama_kv_cell & cell = kv.cells[i];
        if (!cell.has_seq_id(seq_id)) {
            LLAMA_LOG_ERROR("%s: DSV4 raw prefix cell %u is missing active seq_id %d\n",
                    __func__, i, seq_id);
            return false;
        }
    }

    return true;
}

static bool dsv4_build_raw_write_idxs(
        const llama_context & lctx,
        const llama_batch   & batch,
        std::vector<int32_t> & raw_k_write_idxs) {
    raw_k_write_idxs.clear();
    raw_k_write_idxs.reserve((size_t) std::max(0, batch.n_tokens));

    if (batch.n_tokens <= 0) {
        return true;
    }

    const llama_kv_cache & kv = lctx.kv_self;
    if (kv.head < 0 || kv.head + batch.n_tokens > (int32_t) kv.size) {
        LLAMA_LOG_ERROR("%s: DSV4 raw write slots [%d, %d) are outside kv cache size %u\n",
                __func__, kv.head, kv.head + batch.n_tokens, kv.size);
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const int32_t slot = kv.head + i;
        const llama_kv_cell & cell = kv.cells[(size_t) slot];

        if (cell.pos != batch.pos[i]) {
            LLAMA_LOG_ERROR("%s: DSV4 raw write slot %d pos mismatch: cell=%d batch=%d\n",
                    __func__, slot, cell.pos, batch.pos[i]);
            return false;
        }

        if (batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[i] > 0 && batch.seq_id[i] != nullptr) {
            for (int32_t s = 0; s < batch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = batch.seq_id[i][s];
                if (!cell.has_seq_id(seq_id)) {
                    LLAMA_LOG_ERROR("%s: DSV4 raw write slot %d is missing seq_id %d\n",
                            __func__, slot, seq_id);
                    return false;
                }
            }
        }

        raw_k_write_idxs.push_back(slot);
    }

    return true;
}

static bool dsv4_build_raw_read_idxs(
        const llama_context & lctx,
        std::vector<int32_t> & raw_k_read_idxs) {
    raw_k_read_idxs.clear();

    const int64_t n_raw_read = lctx.kv_self.n;
    if (n_raw_read < 0 || n_raw_read > (int64_t) lctx.kv_self.size) {
        LLAMA_LOG_ERROR("%s: DSV4 raw read span %lld is outside kv cache size %u\n",
                __func__, (long long) n_raw_read, lctx.kv_self.size);
        return false;
    }

    raw_k_read_idxs.reserve((size_t) n_raw_read);
    for (int64_t i = 0; i < n_raw_read; ++i) {
        raw_k_read_idxs.push_back((int32_t) i);
    }

    return true;
}

static int64_t dsv4_graph_n_stream(const llama_batch & batch) {
    GGML_UNUSED(batch);
    return 1;
}

static llama_context::dsv4_runtime::comp_plan dsv4_build_reserve_comp_plan(
        const llama_batch & batch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size) {
    llama_context::dsv4_runtime::comp_plan plan;
    plan.n_visible.resize((size_t) batch.n_tokens, (int32_t) kv_size);
    plan.n_stream = dsv4_graph_n_stream(batch);
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
    const uint64_t state_rows = (uint64_t) state_size*(uint64_t) plan.n_stream;
    const size_t n_persist = (size_t) std::min<uint64_t>((uint64_t) batch.n_tokens, state_rows);

    plan.state_pos.resize((size_t) batch.n_tokens);
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
        uint32_t kv_size) {
    const int64_t max_state_read_idx = (int64_t) state_size + batch.n_tokens + (overlap ? 0 : -1);

    if (plan.n_stream != 1) {
        LLAMA_LOG_ERROR("%s: DSV4 %s plan expected single-stream branch path, got n_stream=%lld\n",
                __func__, tag, (long long) plan.n_stream);
        return false;
    }

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
    }

    for (size_t i = 0; i < plan.state_persist_src_idxs.size(); ++i) {
        const int64_t src = plan.state_persist_src_idxs[i];
        const int64_t dst = plan.state_persist_dst_idxs[i];
        if (src < 0 || src >= batch.n_tokens) {
            LLAMA_LOG_ERROR("%s: DSV4 %s persist src[%zu]=%lld outside current batch rows=%d\n",
                    __func__, tag, i, (long long) src, batch.n_tokens);
            return false;
        }
        if (dst < 0 || (uint32_t) dst >= state_size) {
            LLAMA_LOG_ERROR("%s: DSV4 %s persist dst[%zu]=%lld outside state_size=%u\n",
                    __func__, tag, i, (long long) dst, state_size);
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
        if (idx < 0 || (uint32_t) idx >= kv_size) {
            LLAMA_LOG_ERROR("%s: DSV4 %s write idx[%zu]=%lld outside kv_size=%u\n",
                    __func__, tag, i, (long long) idx, kv_size);
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
        uint32_t kv_size) {
    llama_context::dsv4_runtime::comp_plan plan;
    plan.n_visible.resize((size_t) batch.n_tokens);
    plan.n_stream = dsv4_graph_n_stream(batch);

    const int64_t state_rows = (int64_t) state_size;

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
        const llama_seq_id seq_id =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[i] > 0 && batch.seq_id[i] != nullptr
                ? batch.seq_id[i][0]
                : 0;
        curr_token_idx_map[std::make_pair(seq_id, batch.pos[i])] = i;
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            return (int32_t) (state_rows + batch.n_tokens);
        }

        const auto it = curr_token_idx_map.find(std::make_pair(seq_id, pos));
        if (it != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + it->second);
        }

        return (int32_t) (pos%state_size);
    };

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        const llama_pos pos = batch.pos[i];
        const llama_seq_id seq_id =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[i] > 0 && batch.seq_id[i] != nullptr
                ? batch.seq_id[i][0]
                : 0;
        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[(size_t) i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        const int32_t state_idx = (int32_t) (pos%state_size);
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
        plan.state_write_idxs.push_back(pos/ratio);
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

    if (ratio == llama_context::dsv4_runtime::CSA_RATIO && plan.state_write_idxs.empty() && !plan.state_pos.empty()) {
        const llama_seq_id seq_id0 =
                batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.n_seq_id[0] > 0 && batch.seq_id[0] != nullptr
                ? batch.seq_id[0][0]
                : 0;
        const uint32_t source_idx = (uint32_t) state_source_idx(seq_id0, batch.pos[0]);
        plan.state_write_idxs.push_back((int64_t) kv_size - 1);
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
        std::vector<float> & storage,
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
    storage.assign((size_t) width*height, -INFINITY);

    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t n_visible = i < (int32_t) plan.n_visible.size() ? plan.n_visible[(size_t) i] : 0;
        for (int32_t j = 0; j < n_visible && j < width; ++j) {
            storage[(size_t) i*width + j] = 0.0f;
        }
    }

    ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size()*sizeof(float));
}

bool llama_context::ensure_dsv4_cache_tensors() {
    const int32_t n_layer = model.hparams.n_layer;
    const int64_t n_embd_head = model.hparams.n_embd_head_k(0);
    const int64_t n_indexer_head = model.hparams.indexer_head_size;
    const uint32_t csa_kv = GGML_PAD(dsv4_comp_size(cparams.n_ctx, dsv4_runtime::CSA_RATIO), 256u);
    const uint32_t hca_kv = GGML_PAD(dsv4_comp_size(cparams.n_ctx, dsv4_runtime::HCA_RATIO), 256u);

    if (dsv4.cache.cache_ctx != nullptr &&
        (int32_t) dsv4.cache.csa_k.size() == n_layer) {
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
            cache.csa_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, kv_self.type_k, n_embd_head, csa_kv, 1);
            cache.lid_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, kv_self.type_k, n_indexer_head, csa_kv, 1);
            cache.csa_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_embd_head, 2*dsv4_runtime::CSA_RATIO);
            cache.csa_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_embd_head, 2*dsv4_runtime::CSA_RATIO);
            cache.lid_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_indexer_head, 2*dsv4_runtime::CSA_RATIO);
            cache.lid_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, 2*n_indexer_head, 2*dsv4_runtime::CSA_RATIO);

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
            cache.hca_k[(size_t) il] = ggml_new_tensor_3d(cache.cache_ctx, kv_self.type_k, n_embd_head, hca_kv, 1);
            cache.hca_state_kv[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, n_embd_head, dsv4_runtime::HCA_RATIO);
            cache.hca_state_score[(size_t) il] = ggml_new_tensor_2d(cache.cache_ctx, GGML_TYPE_F32, n_embd_head, dsv4_runtime::HCA_RATIO);

            if (!alloc_tensor(cache.hca_k[(size_t) il], buft) ||
                !alloc_tensor(cache.hca_state_kv[(size_t) il], buft) ||
                !alloc_tensor(cache.hca_state_score[(size_t) il], buft)) {
                LLAMA_LOG_ERROR("%s: failed to allocate DSV4 HCA buffers for layer %d\n", __func__, il);
                free_dsv4_cache_tensors();
                return false;
            }
        }
    }

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
    if (dsv4.cache.cache_ctx != nullptr) {
        ggml_free(dsv4.cache.cache_ctx);
        dsv4.cache.cache_ctx = nullptr;
    }
}

void llama_reset_dsv4_state(llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    for (ggml_backend_buffer_t buf : ctx->dsv4.cache.cache_bufs) {
        ggml_backend_buffer_clear(buf, 0);
    }
}

bool llama_prepare_dsv4_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan) {
    if (lctx.model.arch != LLM_ARCH_DEEPSEEK4) {
        return true;
    }

    if (!dsv4_batch_single_stream_compatible(batch)) {
        LLAMA_LOG_ERROR("%s: DeepSeek-V4 branch path does not yet support multi-sequence or coupled-sequence batches in its fork-native planner\n", __func__);
        return false;
    }

    if (!dsv4_validate_single_stream_raw_layout(lctx, batch)) {
        return false;
    }

    if (!lctx.ensure_dsv4_cache_tensors()) {
        return false;
    }

    const uint32_t csa_kv_size = dsv4_cache_kv_size(lctx.dsv4.cache.csa_k);
    const uint32_t hca_kv_size = dsv4_cache_kv_size(lctx.dsv4.cache.hca_k);
    const uint32_t lid_kv_size = dsv4_cache_kv_size(lctx.dsv4.cache.lid_k);
    const uint32_t csa_state_size = dsv4_cache_state_size(lctx.dsv4.cache.csa_state_kv);
    const uint32_t hca_state_size = dsv4_cache_state_size(lctx.dsv4.cache.hca_state_kv);
    const uint32_t lid_state_size = dsv4_cache_state_size(lctx.dsv4.cache.lid_state_kv);

    const auto build_plan = [&](uint32_t ratio, bool overlap, uint32_t state_size, uint32_t kv_size) {
        return reserve_plan
            ? dsv4_build_reserve_comp_plan(batch, ratio, overlap, state_size, kv_size)
            : dsv4_build_comp_plan(batch, ratio, overlap, state_size, kv_size);
    };

    lctx.dsv4.csa_plan = build_plan(llama_context::dsv4_runtime::CSA_RATIO, true, csa_state_size, csa_kv_size);
    lctx.dsv4.hca_plan = build_plan(llama_context::dsv4_runtime::HCA_RATIO, false, hca_state_size, hca_kv_size);
    lctx.dsv4.lid_plan = build_plan(llama_context::dsv4_runtime::CSA_RATIO, true, lid_state_size, lid_kv_size);

    if (!dsv4_validate_comp_plan("csa", batch, lctx.dsv4.csa_plan, llama_context::dsv4_runtime::CSA_RATIO, true, csa_state_size, csa_kv_size) ||
        !dsv4_validate_comp_plan("hca", batch, lctx.dsv4.hca_plan, llama_context::dsv4_runtime::HCA_RATIO, false, hca_state_size, hca_kv_size) ||
        !dsv4_validate_comp_plan("lid", batch, lctx.dsv4.lid_plan, llama_context::dsv4_runtime::CSA_RATIO, true, lid_state_size, lid_kv_size)) {
        return false;
    }

    if (!set_tensors) {
        if (llama_dsv4_trace_enabled()) {
            if (!reserve_plan) {
                std::vector<int32_t> raw_k_write_idxs;
                std::vector<int32_t> raw_k_read_idxs;
                if (dsv4_build_raw_write_idxs(lctx, batch, raw_k_write_idxs) &&
                    dsv4_build_raw_read_idxs(lctx, raw_k_read_idxs)) {
                    dsv4_trace_emit_tokens(lctx, batch, raw_k_write_idxs, raw_k_read_idxs);
                }
            }
            dsv4_trace_emit_plan("csa", llama_context::dsv4_runtime::CSA_RATIO, true, csa_state_size, csa_kv_size, lctx.dsv4.csa_plan);
            dsv4_trace_emit_plan("hca", llama_context::dsv4_runtime::HCA_RATIO, false, hca_state_size, hca_kv_size, lctx.dsv4.hca_plan);
            dsv4_trace_emit_plan("lid", llama_context::dsv4_runtime::CSA_RATIO, true, lid_state_size, lid_kv_size, lctx.dsv4.lid_plan);
        }
        return true;
    }

    std::vector<int32_t> raw_k_write_idxs;
    if (!dsv4_build_raw_write_idxs(lctx, batch, raw_k_write_idxs)) {
        return false;
    }

    std::vector<int32_t> raw_k_read_idxs;
    if (!dsv4_build_raw_read_idxs(lctx, raw_k_read_idxs)) {
        return false;
    }

    if (llama_dsv4_trace_enabled()) {
        dsv4_trace_emit_tokens(lctx, batch, raw_k_write_idxs, raw_k_read_idxs);
    }

    dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_write_idxs, raw_k_write_idxs);
    dsv4_set_input_tensor(lctx.dsv4.inputs.raw_k_read_idxs, raw_k_read_idxs);

    auto set_comp = [&](llama_context::dsv4_runtime::comp_inputs & inputs, llama_context::dsv4_runtime::comp_plan & plan, std::vector<float> & mask_data) {
        dsv4_set_input_tensor(inputs.state_pos, plan.state_pos);
        dsv4_set_input_tensor(inputs.state_persist_src_idxs, plan.state_persist_src_idxs);
        dsv4_set_input_tensor(inputs.state_persist_dst_idxs, plan.state_persist_dst_idxs);
        dsv4_set_input_tensor(inputs.state_read_idxs, plan.state_read_idxs);
        dsv4_set_input_tensor(inputs.state_write_idxs, plan.state_write_idxs);
        dsv4_set_input_tensor(inputs.state_write_pos, plan.state_write_pos);
        dsv4_set_mask_tensor(inputs.kq_mask, mask_data, plan, batch.n_tokens);
    };

    set_comp(lctx.dsv4.inputs.csa, lctx.dsv4.csa_plan, lctx.dsv4.csa_mask_data);
    set_comp(lctx.dsv4.inputs.hca, lctx.dsv4.hca_plan, lctx.dsv4.hca_mask_data);
    set_comp(lctx.dsv4.inputs.lid, lctx.dsv4.lid_plan, lctx.dsv4.lid_mask_data);

    if (lctx.dsv4.inputs.lid.k_rot != nullptr && lctx.dsv4.inputs.lid.k_rot->buffer != nullptr) {
        const int64_t n_rot = dsv4_k_rot_size(lctx.model.hparams.indexer_head_size);
        dsv4_build_hadamard_matrix(lctx.dsv4.lid_k_rot_data, n_rot);
        if (!lctx.dsv4.lid_k_rot_data.empty()) {
            ggml_backend_tensor_set(lctx.dsv4.inputs.lid.k_rot, lctx.dsv4.lid_k_rot_data.data(), 0, lctx.dsv4.lid_k_rot_data.size()*sizeof(float));
        }
    }

    return true;
}
