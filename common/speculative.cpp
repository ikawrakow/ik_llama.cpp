#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

void llama_set_mtp_target_context(struct llama_context * ctx, struct llama_context * target_ctx);

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_DFLASH,
    COMMON_SPECULATIVE_TYPE_MTP,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_SUFFIX
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH},
    {"mtp",           COMMON_SPECULATIVE_TYPE_MTP},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX}
};

struct common_speculative_config {
    common_speculative_stage_params stage;
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(
            const common_speculative_stage_params & s,
            const common_params_speculative & p = common_params_speculative{})
        : stage(s), type(s.type), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

static bool common_speculative_are_dflash_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft)) {
        LOG_DBG("%s: DFlash draft model vocab type must match the target model\n", __func__);
        return false;
    }

    const bool add_bos_tgt = llama_vocab_get_add_bos(vocab_tgt);
    const bool add_bos_dft = llama_vocab_get_add_bos(vocab_dft);
    const bool add_eos_tgt = llama_vocab_get_add_eos(vocab_tgt);
    const bool add_eos_dft = llama_vocab_get_add_eos(vocab_dft);
    const llama_token bos_tgt = llama_vocab_bos(vocab_tgt);
    const llama_token bos_dft = llama_vocab_bos(vocab_dft);
    const llama_token eos_tgt = llama_vocab_eos(vocab_tgt);
    const llama_token eos_dft = llama_vocab_eos(vocab_dft);

    if (add_bos_tgt != add_bos_dft || add_eos_tgt != add_eos_dft ||
        (add_bos_tgt && bos_tgt != bos_dft) ||
        (add_eos_tgt && eos_tgt != eos_dft)) {
        LOG_DBG("%s: DFlash draft special tokens must match the target model (add_bos=%d/%d add_eos=%d/%d bos=%d/%d eos=%d/%d)\n",
                __func__,
                (int) add_bos_tgt,
                (int) add_bos_dft,
                (int) add_eos_tgt,
                (int) add_eos_dft,
                (int) bos_tgt,
                (int) bos_dft,
                (int) eos_tgt,
                (int) eos_dft);
        return false;
    }

    const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
    const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
    const int vocab_diff  = n_vocab_tgt > n_vocab_dft
        ? n_vocab_tgt - n_vocab_dft
        : n_vocab_dft - n_vocab_tgt;

    if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
        LOG_DBG("%s: DFlash draft vocab size differs too much from the target model (%d vs %d)\n",
                __func__, n_vocab_dft, n_vocab_tgt);
        return false;
    }

    for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
        const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
        const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

        if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
            LOG_DBG("%s: DFlash draft token %d differs - target '%s', draft '%s'\n", __func__, i,
                    common_token_to_piece(vocab_tgt, i).c_str(),
                    common_token_to_piece(vocab_dft, i).c_str());
            return false;
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_pos draft_base_pos,
            llama_seq_id draft_seq_id,
            llama_tokens & result) {
        GGML_UNUSED(draft_base_pos);
        GGML_UNUSED(draft_seq_id);
        draft(params, prompt_tgt, id_last, result);
    }

    virtual void accept(uint16_t n_accepted) = 0;
};

struct common_speculative_state_mtp;
struct common_speculative_state_dflash;

static void dflash_contract_log_append(
    const common_speculative_state_dflash & state,
    llama_seq_id seq_id,
    const std::vector<llama_pos> & new_positions);
static void dflash_contract_log_draft(
    const common_speculative_state_dflash & state,
    int32_t n_keep,
    size_t result_size);

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec);
static const common_speculative_state_mtp * common_speculative_get_mtp_state(const common_speculative * spec);
static common_speculative_state_dflash * common_speculative_get_dflash_state(common_speculative * spec);
static const common_speculative_state_dflash * common_speculative_get_dflash_state(const common_speculative * spec);
static int32_t common_speculative_feature_width(const common_speculative * spec);
static void dflash_materialize_target_window_features(common_speculative_state_dflash & state);
static void dflash_ring_reset_rows(common_speculative_state_dflash & state, const float * rows, int32_t n_rows);
static void dflash_append_target_features(
    common_speculative_state_dflash & state,
    const float * feature_rows,
    int32_t n_rows);
static void dflash_clear_target_features(common_speculative_state_dflash & state);
static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state);

static std::vector<llama_token> mtp_speculative_gen_draft(
    common_speculative_state_mtp & state,
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    llama_token id_last,
    llama_pos n_past,
    llama_seq_id seq_id,
    bool constant_draft_positions = false);

static int32_t mtp_update_kv_cache(struct llama_context * ctx, const llama_batch & batch, bool is_prompt_warmup);

static bool dflash_contract_log_enabled() {
    const char * env = std::getenv("IK_DFLASH_CONTRACT_LOG");
    if (env == nullptr || *env == '\0') {
        return false;
    }

    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "off") != 0;
}

static bool dflash_use_kv_cache_experiment() {
    const char * env = std::getenv("IK_DFLASH_KV_CACHE");
    if (env == nullptr || *env == '\0') {
        return false;
    }

    return std::strcmp(env, "0") != 0 &&
           std::strcmp(env, "false") != 0 &&
           std::strcmp(env, "off") != 0;
}

template <typename T>
static std::string dflash_contract_format_values(
        const std::vector<T> & values,
        size_t edge_count = 4) {
    std::ostringstream oss;
    oss << '[';
    if (values.empty()) {
        oss << ']';
        return oss.str();
    }

    const size_t head = std::min(edge_count, values.size());
    for (size_t i = 0; i < head; ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << values[i];
    }

    if (values.size() > edge_count * 2) {
        oss << ",...,";
        for (size_t i = values.size() - edge_count; i < values.size(); ++i) {
            if (i > values.size() - edge_count) {
                oss << ',';
            }
            oss << values[i];
        }
    } else {
        for (size_t i = head; i < values.size(); ++i) {
            oss << ',' << values[i];
        }
    }

    oss << ']';
    return oss.str();
}

struct dflash_contract_pos_summary {
    llama_pos first = -1;
    llama_pos last = -1;
    int32_t gap_count = 0;
    int32_t nonmono_count = 0;
};

static dflash_contract_pos_summary dflash_contract_summarize_positions(
        const std::vector<llama_pos> & positions) {
    dflash_contract_pos_summary summary;
    if (positions.empty()) {
        return summary;
    }

    summary.first = positions.front();
    summary.last = positions.back();
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] <= positions[i - 1]) {
            summary.nonmono_count++;
        } else if (positions[i] != positions[i - 1] + 1) {
            summary.gap_count++;
        }
    }

    return summary;
}

struct mtp_last_embd {
    std::vector<float> embd;
    float prob = 0.0f;
    int   last_id = -1;
};

struct common_speculative_state_mtp : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_mtp = nullptr;
    common_sampler * smpl;
    // For Gemma 4 external MTP assistant: draft positions are held constant
    bool constant_draft_positions = false;
    int n_embd = 0;
    std::unordered_map<llama_seq_id, std::vector<float>> target_hidden_by_seq;
    std::unordered_map<llama_seq_id, mtp_last_embd> draft_cache_by_seq;

    common_speculative_state_mtp(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_mtp,
            bool constant_draft_positions = false)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_mtp(ctx_mtp)
        , constant_draft_positions(constant_draft_positions)
    {
        struct common_params_sampling sparams;
        sparams.samplers_sequence = {
            llama_sampler_type::DIST,
        };
        smpl = common_sampler_init(llama_get_model(ctx_mtp), sparams);
        llama_set_mtp_target_context(ctx_mtp, ctx_tgt);
        n_embd = llama_mtp_state_n_embd(ctx_mtp);

        LOG_INF("%s: MTP context ready (n_ctx=%d, constant_draft_positions=%s)\n", __func__,
                llama_n_ctx(ctx_mtp), constant_draft_positions ? "true" : "false");
    }

    ~common_speculative_state_mtp() override {
        common_sampler_free(smpl);
        if (ctx_mtp) {
            llama_free(ctx_mtp);
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        target_hidden_by_seq.clear();
        draft_cache_by_seq.clear();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        draft(params, prompt_tgt, id_last, -1, 0, result);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_pos draft_base_pos,
            llama_seq_id seq_id,
            llama_tokens & result) override {

        const llama_pos mtp_pos_max = llama_kv_cache_seq_pos_max(ctx_mtp, seq_id);
        const bool has_draft_base_pos = draft_base_pos >= 0;
        // Prefer the target slot position when the caller has it. Gemma4 external MTP reads
        // the target KV cache directly, so ctx_mtp's own KV position is not authoritative.
        const llama_pos n_past = has_draft_base_pos
            ? draft_base_pos
            : (mtp_pos_max >= 0 ? mtp_pos_max + 1 : (llama_pos) prompt_tgt.size());

        if (!has_draft_base_pos && !prompt_tgt.empty() && mtp_pos_max < (llama_pos)prompt_tgt.size() - 1) {
            LOG_WRN("%s: MTP context not fully warmed up: pos_max = %d, expected = %d\n",
                    __func__, (int)mtp_pos_max, (int)prompt_tgt.size() - 1);
        }
        if (has_draft_base_pos && !constant_draft_positions && mtp_pos_max < n_past - 1) {
            LOG_WRN("%s: MTP context not fully warmed up: pos_max = %d, expected >= %d\n",
                    __func__, (int)mtp_pos_max, (int)n_past - 1);
        }

        llama_context * ctx = ctx_mtp;

        const auto hidden_it = target_hidden_by_seq.find(seq_id);
        if (hidden_it == target_hidden_by_seq.end() || (int) hidden_it->second.size() != n_embd) {
            LOG_WRN("%s: missing target hidden state for seq_id %d\n", __func__, (int) seq_id);
            result.clear();
            return;
        }

        if (!llama_set_draft_input_hidden_state_copy(ctx, hidden_it->second.data(), hidden_it->second.size())) {
            result.clear();
            return;
        }

        result = mtp_speculative_gen_draft(
            *this,
            smpl,
            ctx,
            params.n_max,
            params.p_min,
            id_last,
            n_past,
            seq_id,
            constant_draft_positions
        );
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};

#include "speculative-impl.h"

static bool common_speculative_capture_target_features(
        common_speculative * spec,
        const common_speculative_feature_view & features);

static bool common_speculative_feature_view_from_hidden_rows(
        const std::vector<float> & hidden_rows,
        int32_t width,
        llama_seq_id seq_id,
        llama_pos pos_base,
        common_speculative_feature_view & view) {
    view = {};
    view.kind = COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE;
    view.width = width;

    if (width <= 0 || hidden_rows.empty() || hidden_rows.size() % (size_t) width != 0) {
        return false;
    }

    const size_t n_rows = hidden_rows.size() / (size_t) width;
    view.rows.reserve(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        view.rows.push_back({
            /* .seq_id = */ seq_id,
            /* .pos    = */ pos_base + (llama_pos) i,
            /* .data   = */ hidden_rows.data() + i * (size_t) width,
        });
    }

    return true;
}

static bool common_speculative_collect_target_batch_features(
        const common_speculative * spec,
        llama_context * ctx,
        const llama_batch & batch,
        common_speculative_feature_view & features) {
    features = {};
    if (common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_DFLASH)) {
        return llama_spec_get_dflash_feature_view(ctx, batch, features);
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    if (!llama_spec_get_hidden_feature_view(ctx, batch, features)) {
        return false;
    }

    return true;
}

static bool common_speculative_collect_target_seq_batch_features(
        const common_speculative * spec,
        llama_context * ctx,
        const llama_batch & batch,
        llama_seq_id seq_id,
        common_speculative_feature_view & features) {
    features = {};
    if (common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_DFLASH)) {
        return llama_spec_get_dflash_feature_view_for_seq(ctx, batch, seq_id, features);
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    if (!llama_spec_get_hidden_feature_view_for_seq(ctx, batch, seq_id, features)) {
        return false;
    }

    return true;
}

bool common_speculative_capture_output_hidden(
        common_speculative * spec,
        llama_context * ctx,
        int32_t output_index,
        llama_seq_id seq_id,
        llama_pos pos) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    common_speculative_feature_view features;
    if (!llama_spec_get_hidden_feature_view_from_output_index(ctx, output_index, seq_id, pos, features)) {
        return false;
    }

    return common_speculative_capture_target_features(spec, features);
}

bool common_speculative_ensure_sequence_hidden(
        common_speculative * spec,
        llama_context * ctx,
        llama_seq_id seq_id,
        llama_pos pos) {
    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP) || common_speculative_has_sequence_hidden(spec, seq_id)) {
        return true;
    }

    return common_speculative_capture_output_hidden(spec, ctx, -1, seq_id, pos);
}

int32_t common_speculative_on_target_seq_batch(
        common_speculative * spec,
        llama_context * ctx_tgt,
        const llama_batch & batch,
        llama_seq_id seq_id,
        bool is_prompt_warmup) {
    if (ctx_tgt == nullptr || batch.n_tokens <= 0) {
        return 0;
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_DFLASH)) {
        llama_context * ctx_mtp = common_speculative_get_companion_ctx(spec);
        ctx_mtp = ctx_mtp ? ctx_mtp : ctx_tgt;
        if (ctx_mtp == nullptr) {
            return 0;
        }

        const int n_embd_src = common_speculative_ctx_mtp_n_embd(ctx_tgt);
        const int n_embd_dst = common_speculative_ctx_mtp_n_embd(ctx_mtp);
        if (n_embd_src <= 0 || n_embd_dst <= 0) {
            return -1;
        }

        if (n_embd_src != n_embd_dst) {
            LOG_ERR("MTP warmup hidden state width mismatch: n_embd_src = %d, n_embd_dst = %d\n", n_embd_src, n_embd_dst);
            return -1;
        }
    }

    common_speculative_feature_view feature_view;
    const llama_batch * batch_for_spec = &batch;
    llama_batch seq_batch = {};
    const bool needs_seq_split = is_prompt_warmup && !common_speculative_batch_is_exact_single_seq(batch, seq_id);
    auto * dflash_state = common_speculative_get_dflash_state(spec);
    const bool measure_dflash_warmup_collect = dflash_state != nullptr && is_prompt_warmup;

    if (needs_seq_split) {
        const int n_seq_tokens = common_speculative_copy_seq_batch(batch, seq_id, seq_batch);
        if (n_seq_tokens <= 0) {
            return n_seq_tokens < 0 ? -1 : 0;
        }

        const int64_t t_collect_us = measure_dflash_warmup_collect ? ggml_time_us() : 0;
        if (!common_speculative_collect_target_seq_batch_features(spec, ctx_tgt, batch, seq_id, feature_view)) {
            llama_batch_free(seq_batch);
            return -1;
        }
        if (measure_dflash_warmup_collect) {
            dflash_state->t_warmup_collect_us += (uint64_t) (ggml_time_us() - t_collect_us);
            dflash_state->n_warmup_collect_calls++;
            dflash_state->n_warmup_collect_rows += (size_t) n_seq_tokens;
        }

        batch_for_spec = &seq_batch;
    } else {
        const int64_t t_collect_us = measure_dflash_warmup_collect ? ggml_time_us() : 0;
        if (!common_speculative_collect_target_batch_features(spec, ctx_tgt, batch, feature_view)) {
            return -1;
        }
        if (measure_dflash_warmup_collect) {
            dflash_state->t_warmup_collect_us += (uint64_t) (ggml_time_us() - t_collect_us);
            dflash_state->n_warmup_collect_calls++;
            dflash_state->n_warmup_collect_rows += (size_t) batch.n_tokens;
        }
    }

    const int32_t ret = common_speculative_on_target_batch(spec, *batch_for_spec, feature_view, is_prompt_warmup);
    if (needs_seq_split) {
        llama_batch_free(seq_batch);
    }

    return ret;
}

bool common_speculative_copy_output_hidden_rows(
        const common_speculative * spec,
        llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows) {
    hidden_rows.clear();
    if (common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_DFLASH)) {
        return llama_spec_copy_dflash_rows_from_output_indices(ctx, output_indices, hidden_rows);
    }

    if (!common_speculative_has_type(spec, COMMON_SPECULATIVE_TYPE_MTP)) {
        return true;
    }

    return llama_spec_copy_hidden_rows_from_output_indices(ctx, output_indices, hidden_rows);
}

static bool common_speculative_build_commit_tokens(
        common_speculative_type spec_type_used,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        std::vector<llama_token> & commit_tokens) {
    commit_tokens.clear();
    if (ids.empty()) {
        return true;
    }

    if (spec_type_used == COMMON_SPECULATIVE_TYPE_MTP) {
        commit_tokens = ids;
        return true;
    }

    commit_tokens.reserve(ids.size());
    commit_tokens.push_back(sampled_before);
    if (ids.size() > 1) {
        commit_tokens.insert(commit_tokens.end(), ids.begin(), ids.end() - 1);
    }

    return commit_tokens.size() == ids.size();
}

static bool common_speculative_apply_hidden_rows(
        common_speculative * spec,
        llama_seq_id seq_id,
        llama_pos pos_base,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    const int32_t feature_width = common_speculative_feature_width(spec);
    if (feature_width <= 0 || ids.empty()) {
        return true;
    }

    const size_t expected_floats = ids.size() * (size_t) feature_width;
    if (hidden_rows.size() != expected_floats) {
        return false;
    }

    llama_batch accepted_batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(accepted_batch, ids[i], pos_base + (llama_pos) i, { seq_id }, true);
    }

    common_speculative_feature_view feature_view;
    const bool have_feature_view = common_speculative_feature_view_from_hidden_rows(
        hidden_rows, feature_width, seq_id, pos_base, feature_view);
    const int32_t ret = have_feature_view
        ? common_speculative_on_target_batch(spec, accepted_batch, feature_view, false)
        : -1;

    llama_batch_free(accepted_batch);
    return ret == 0;
}

bool common_speculative_commit_accepted_hidden_rows(
        common_speculative * spec,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<float> & hidden_rows) {
    if (common_speculative_feature_width(spec) <= 0 || ids.empty()) {
        return true;
    }

    std::vector<llama_token> commit_tokens;
    if (!common_speculative_build_commit_tokens(spec_type_used, sampled_before, ids, commit_tokens)) {
        return false;
    }

    auto * dflash_state = common_speculative_get_dflash_state(spec);
    const int64_t t_commit_us = dflash_state != nullptr ? ggml_time_us() : 0;
    const bool ok = common_speculative_apply_hidden_rows(spec, seq_id, pos_base, commit_tokens, hidden_rows);
    if (dflash_state != nullptr) {
        dflash_state->t_accept_commit_us += (uint64_t) (ggml_time_us() - t_commit_us);
        dflash_state->n_accept_commit_calls++;
        dflash_state->n_accept_commit_rows += commit_tokens.size();
    }

    return ok;
}

bool common_speculative_commit_accepted_output(
        common_speculative * spec,
        llama_context * ctx,
        common_speculative_type spec_type_used,
        llama_seq_id seq_id,
        llama_pos pos_base,
        llama_token sampled_before,
        const std::vector<llama_token> & ids,
        const std::vector<int32_t> & output_indices) {
    if (common_speculative_feature_width(spec) <= 0 || ids.empty()) {
        return true;
    }

    std::vector<float> hidden_rows;
    auto * dflash_state = common_speculative_get_dflash_state(spec);
    const int64_t t_copy_us = dflash_state != nullptr ? ggml_time_us() : 0;
    if (!common_speculative_copy_output_hidden_rows(spec, ctx, output_indices, hidden_rows)) {
        return false;
    }
    if (dflash_state != nullptr) {
        dflash_state->t_accept_output_copy_us += (uint64_t) (ggml_time_us() - t_copy_us);
        dflash_state->n_accept_output_copy_calls++;
        dflash_state->n_accept_output_copy_rows += output_indices.size();
    }

    return common_speculative_commit_accepted_hidden_rows(
        spec,
        spec_type_used,
        seq_id,
        pos_base,
        sampled_before,
        ids,
        hidden_rows);
}

void common_speculative_print_stats(const common_speculative * spec, double slot_tps, int n_decoded, int n_past, common_params_speculative * active_params) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());

        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            const auto * dflash_state = dynamic_cast<const common_speculative_state_dflash *>(impl.get());
            if (dflash_state != nullptr) {
                llama_dflash_profile_stats capture_stats;
                llama_dflash_profile_stats graph_stats;
                const bool have_capture = llama_dflash_profile_get_stats(dflash_state->ctx_tgt, &capture_stats);
                const bool have_graph = llama_dflash_profile_get_stats(dflash_state->ctx_dft, &graph_stats);

                LOG_INF("statistics dflash detail: cross_ctx=%d, window_rows=%d, pos=[%d..%d], window_updates=%zu, rows_seen=%zu, rows_dropped=%zu, shifts=%zu, draft_fail(empty/set/decode)=%zu/%zu/%zu, next_draft_pos=%d\n",
                        dflash_state->cross_ctx,
                        dflash_state->target_window_rows,
                        dflash_state->target_window_pos.empty() ? -1 : (int) dflash_state->target_window_pos.front(),
                        dflash_state->target_window_pos.empty() ? -1 : (int) dflash_state->target_window_pos.back(),
                        dflash_state->n_window_updates,
                        dflash_state->n_rows_seen,
                        dflash_state->n_rows_dropped,
                        dflash_state->n_context_shifts,
                        dflash_state->n_draft_empty,
                        dflash_state->n_set_target_fail,
                        dflash_state->n_decode_fail,
                        (int) dflash_state->last_draft_pos_base);

                if (have_capture || have_graph) {
                    const double kv_cache_total_ms = (double) (
                        graph_stats.graph_kv_cache_build_us +
                        graph_stats.graph_kv_cache_reserve_us +
                        graph_stats.graph_kv_cache_reset_us +
                        graph_stats.graph_kv_cache_alloc_us +
                        graph_stats.graph_kv_cache_feature_upload_us +
                        graph_stats.graph_kv_cache_pos_upload_us +
                        graph_stats.graph_kv_cache_compute_us +
                        graph_stats.graph_kv_cache_sync_us +
                        graph_stats.graph_kv_cache_read_concat_pad_us) / 1000.0;
                    const double kv_upload_feature_ms = (double) graph_stats.graph_kv_cache_feature_upload_us / 1000.0;
                    const double kv_upload_pos_ms = (double) graph_stats.graph_kv_cache_pos_upload_us / 1000.0;
                    const double kv_upload_total_ms = kv_upload_feature_ms + kv_upload_pos_ms;
                    const double kv_compute_ms = (double) graph_stats.graph_kv_cache_compute_us / 1000.0;
                    const double kv_sync_ms = (double) graph_stats.graph_kv_cache_sync_us / 1000.0;
                    const double kv_workspace_total_ms = (double) (
                        graph_stats.graph_kv_workspace_build_us +
                        graph_stats.graph_kv_workspace_reserve_us +
                        graph_stats.graph_kv_workspace_reset_us +
                        graph_stats.graph_kv_workspace_alloc_us +
                        graph_stats.graph_kv_workspace_compute_us +
                        graph_stats.graph_kv_workspace_sync_us) / 1000.0;
                    const double draft_kv_traffic_ms = (double) (
                        graph_stats.graph_main_node_k_ctx_view_us +
                        graph_stats.graph_main_node_v_ctx_view_us +
                        graph_stats.graph_main_node_k_concat_us +
                        graph_stats.graph_main_node_v_concat_us +
                        graph_stats.graph_main_node_k_pad_us +
                        graph_stats.graph_main_node_v_pad_us +
                        graph_stats.graph_main_node_k_perm_cont_us +
                        graph_stats.graph_main_node_v_perm_cont_us) / 1000.0;
                    const double draft_main_profiled_ms = (double) (
                        graph_stats.graph_main_node_qcur_us +
                        graph_stats.graph_main_node_k_draft_us +
                        graph_stats.graph_main_node_v_draft_us +
                        graph_stats.graph_main_node_flash_attn_us +
                        graph_stats.graph_main_node_attn_out_us +
                        graph_stats.graph_main_node_ffn_us +
                        graph_stats.graph_main_node_result_rows_us +
                        graph_stats.graph_main_node_result_norm_us +
                        graph_stats.graph_main_node_result_us) / 1000.0;
                    const double replay_append_ms = (double) dflash_state->t_accept_append_us / 1000.0;
                    const double feature_path_ms = (double) (
                        capture_stats.capture_prepare_sync_us +
                        capture_stats.capture_materialize_us +
                        graph_stats.set_target_copy_us +
                        graph_stats.graph_feature_copy_us +
                        graph_stats.graph_pos_copy_us +
                        graph_stats.graph_mask_build_us) / 1000.0;
                    const double decode_internal_ms = (double) (
                        graph_stats.decode_prelude_us +
                        graph_stats.decode_sched_reset_us +
                        graph_stats.decode_build_graph_us +
                        graph_stats.decode_sched_alloc_graph_us +
                        graph_stats.decode_prepare_us +
                        graph_stats.decode_set_inputs_us +
                        graph_stats.decode_graph_compute_us +
                        graph_stats.decode_result_us +
                        graph_stats.decode_embedding_us +
                        graph_stats.decode_final_sched_reset_us) / 1000.0;

                        LOG_INF("statistics dflash profile: capture(sync/materialize)=%.3f/%.3f ms calls=%llu/%llu bytes=%llu phase(prompt/verify batches changes)=%llu/%llu %llu/%llu, set_target=%.3f ms rows=%llu bytes=%llu, decode(llama_output_reserve/prepare)=%.3f/%.3f ms calls=%llu/%llu realloc(bytes)=%llu/%llu, prep(total/features/pos/mask)=%.3f/%.3f/%.3f/%.3f ms kv_cache(total/build/reserve/reset/alloc/up_f/up_p/compute/sync/read)=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls(prepare/cache/read)=%llu/%llu/%llu bytes(feature/pos/mask/read)=%llu/%llu/%llu/%llu host_layers=%d, fallback_pos(copy/graph)=%llu/%llu, nonmono(copy/graph)=%llu/%llu, capture_fail=%llu/%llu decode_prepare_fail=%llu, visible_kv_max=%llu, last(rows=%d width=%d left_pad=%d n_tokens=%d n_kv=%d pos=[%d..%d])\n",
                            (double) capture_stats.capture_prepare_sync_us / 1000.0,
                            (double) capture_stats.capture_materialize_us / 1000.0,
                            (unsigned long long) capture_stats.capture_prepare_calls,
                            (unsigned long long) capture_stats.capture_materialize_calls,
                            (unsigned long long) capture_stats.capture_materialize_bytes,
                            (unsigned long long) capture_stats.capture_prompt_batches,
                            (unsigned long long) capture_stats.capture_prompt_shape_changes,
                            (unsigned long long) capture_stats.capture_verify_batches,
                            (unsigned long long) capture_stats.capture_verify_shape_changes,
                            (double) graph_stats.set_target_copy_us / 1000.0,
                            (unsigned long long) graph_stats.set_target_rows,
                            (unsigned long long) graph_stats.set_target_copy_bytes,
                            (double) graph_stats.decode_output_reserve_us / 1000.0,
                            (double) graph_stats.decode_prepare_us / 1000.0,
                            (unsigned long long) graph_stats.decode_output_reserve_calls,
                            (unsigned long long) graph_stats.decode_prepare_calls,
                            (unsigned long long) graph_stats.decode_output_reserve_reallocs,
                            (unsigned long long) graph_stats.decode_output_reserve_realloc_bytes,
                            (double) graph_stats.graph_prepare_total_us / 1000.0,
                            (double) graph_stats.graph_feature_copy_us / 1000.0,
                            (double) graph_stats.graph_pos_copy_us / 1000.0,
                            (double) graph_stats.graph_mask_build_us / 1000.0,
                            kv_cache_total_ms,
                            (double) graph_stats.graph_kv_cache_build_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_reserve_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_reset_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_alloc_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_feature_upload_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_pos_upload_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_compute_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_sync_us / 1000.0,
                            (double) graph_stats.graph_kv_cache_read_concat_pad_us / 1000.0,
                            (unsigned long long) graph_stats.graph_prepare_calls,
                            (unsigned long long) graph_stats.graph_kv_cache_calls,
                            (unsigned long long) graph_stats.graph_kv_cache_read_concat_pad_calls,
                            (unsigned long long) graph_stats.graph_feature_bytes,
                            (unsigned long long) graph_stats.graph_pos_bytes,
                            (unsigned long long) graph_stats.graph_mask_bytes,
                            (unsigned long long) graph_stats.graph_kv_cache_cached_bytes,
                            graph_stats.last_kv_cache_host_layers,
                            (unsigned long long) graph_stats.set_target_missing_positions,
                            (unsigned long long) graph_stats.graph_pos_fallbacks,
                            (unsigned long long) graph_stats.set_target_non_monotonic_positions,
                            (unsigned long long) graph_stats.graph_pos_non_monotonic,
                            (unsigned long long) capture_stats.capture_prepare_failures,
                            (unsigned long long) capture_stats.capture_materialize_failures,
                            (unsigned long long) graph_stats.decode_prepare_failures,
                            (unsigned long long) graph_stats.graph_visible_kv_max,
                            graph_stats.last_n_rows,
                            graph_stats.last_width,
                            graph_stats.last_left_pad,
                            graph_stats.last_n_tokens,
                            graph_stats.last_n_kv_total,
                            (int) graph_stats.last_pos_first,
                            (int) graph_stats.last_pos_last);

                            LOG_INF("statistics dflash features: total=%.3f ms capture(sync/materialize)=%.3f/%.3f ms set_target=%.3f ms prep(feature/pos/mask)=%.3f/%.3f/%.3f ms rows(materialize/set_target)=%llu/%llu bytes(materialize/set_target/feature/pos/mask)=%llu/%llu/%llu/%llu/%llu\n",
                                feature_path_ms,
                                (double) capture_stats.capture_prepare_sync_us / 1000.0,
                                (double) capture_stats.capture_materialize_us / 1000.0,
                                (double) graph_stats.set_target_copy_us / 1000.0,
                                (double) graph_stats.graph_feature_copy_us / 1000.0,
                                (double) graph_stats.graph_pos_copy_us / 1000.0,
                                (double) graph_stats.graph_mask_build_us / 1000.0,
                                (unsigned long long) capture_stats.capture_materialize_rows,
                                (unsigned long long) graph_stats.set_target_rows,
                                (unsigned long long) capture_stats.capture_materialize_bytes,
                                (unsigned long long) graph_stats.set_target_copy_bytes,
                                (unsigned long long) graph_stats.graph_feature_bytes,
                                (unsigned long long) graph_stats.graph_pos_bytes,
                                (unsigned long long) graph_stats.graph_mask_bytes);

                            LOG_INF("statistics dflash kv: total=%.3f ms build/reserve/reset/alloc/upload_f/upload_p/compute/sync/read=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%llu cached_bytes=%llu host_layers=%d\n",
                                kv_cache_total_ms,
                                (double) graph_stats.graph_kv_cache_build_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_reserve_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_reset_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_alloc_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_feature_upload_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_pos_upload_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_compute_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_sync_us / 1000.0,
                                (double) graph_stats.graph_kv_cache_read_concat_pad_us / 1000.0,
                                (unsigned long long) graph_stats.graph_kv_cache_calls,
                                (unsigned long long) graph_stats.graph_kv_cache_cached_bytes,
                                graph_stats.last_kv_cache_host_layers);

                            if (graph_stats.graph_kv_workspace_calls > 0) {
                                LOG_INF("statistics dflash kv workspace: total=%.3f ms build/reserve/reset/alloc/compute/sync=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%llu\n",
                                        kv_workspace_total_ms,
                                        (double) graph_stats.graph_kv_workspace_build_us / 1000.0,
                                        (double) graph_stats.graph_kv_workspace_reserve_us / 1000.0,
                                        (double) graph_stats.graph_kv_workspace_reset_us / 1000.0,
                                        (double) graph_stats.graph_kv_workspace_alloc_us / 1000.0,
                                        (double) graph_stats.graph_kv_workspace_compute_us / 1000.0,
                                        (double) graph_stats.graph_kv_workspace_sync_us / 1000.0,
                                        (unsigned long long) graph_stats.graph_kv_workspace_calls);
                            }

                            if (graph_stats.decode_internal_chunks > 0) {
                            LOG_INF("statistics dflash decode: llama_decode(total)=%.3f ms calls=%zu chunks=%llu rebuilds=%llu sync_points=%llu internal(total/prelude/sched_reset/build/alloc/prepare/set_inputs/compute/get_result/get_embedding/final_reset)=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms\n",
                                (double) dflash_state->t_draft_decode_us / 1000.0,
                                dflash_state->n_call_draft,
                                (unsigned long long) graph_stats.decode_internal_chunks,
                                (unsigned long long) graph_stats.decode_graph_rebuilds,
                                (unsigned long long) graph_stats.decode_sync_profile_points,
                                decode_internal_ms,
                                (double) graph_stats.decode_prelude_us / 1000.0,
                                (double) graph_stats.decode_sched_reset_us / 1000.0,
                                (double) graph_stats.decode_build_graph_us / 1000.0,
                                (double) graph_stats.decode_sched_alloc_graph_us / 1000.0,
                                (double) graph_stats.decode_prepare_us / 1000.0,
                                (double) graph_stats.decode_set_inputs_us / 1000.0,
                                (double) graph_stats.decode_graph_compute_us / 1000.0,
                                (double) graph_stats.decode_result_us / 1000.0,
                                (double) graph_stats.decode_embedding_us / 1000.0,
                                (double) graph_stats.decode_final_sched_reset_us / 1000.0);
                            }

                            if (graph_stats.graph_kv_node_fused_target_calls > 0 ||
                                    graph_stats.graph_kv_node_k_proj_calls > 0 ||
                                    graph_stats.graph_kv_node_k_norm_calls > 0 ||
                                    graph_stats.graph_kv_node_k_rope_calls > 0 ||
                                    graph_stats.graph_kv_node_v_proj_calls > 0 ||
                                    graph_stats.graph_kv_node_k_store_calls > 0 ||
                                    graph_stats.graph_kv_node_v_store_calls > 0) {
                                LOG_INF("statistics dflash kv nodes: fused_target/k_proj/k_norm/k_rope/v_proj/k_store/v_store=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%llu/%llu/%llu/%llu/%llu/%llu/%llu\n",
                                        (double) graph_stats.graph_kv_node_fused_target_us / 1000.0,
                                        (double) graph_stats.graph_kv_node_k_proj_us / 1000.0,
                                        (double) graph_stats.graph_kv_node_k_norm_us / 1000.0,
                                        (double) graph_stats.graph_kv_node_k_rope_us / 1000.0,
                                        (double) graph_stats.graph_kv_node_v_proj_us / 1000.0,
                                        (double) graph_stats.graph_kv_node_k_store_us / 1000.0,
                                        (double) graph_stats.graph_kv_node_v_store_us / 1000.0,
                                        (unsigned long long) graph_stats.graph_kv_node_fused_target_calls,
                                        (unsigned long long) graph_stats.graph_kv_node_k_proj_calls,
                                        (unsigned long long) graph_stats.graph_kv_node_k_norm_calls,
                                        (unsigned long long) graph_stats.graph_kv_node_k_rope_calls,
                                        (unsigned long long) graph_stats.graph_kv_node_v_proj_calls,
                                        (unsigned long long) graph_stats.graph_kv_node_k_store_calls,
                                        (unsigned long long) graph_stats.graph_kv_node_v_store_calls);
                            }

                                            if (graph_stats.graph_main_node_qcur_calls > 0 ||
                                                graph_stats.graph_main_node_k_draft_calls > 0 ||
                                                graph_stats.graph_main_node_v_draft_calls > 0 ||
                                                graph_stats.graph_main_node_flash_attn_calls > 0 ||
                                                graph_stats.graph_main_node_attn_out_calls > 0 ||
                                                graph_stats.graph_main_node_ffn_calls > 0 ||
                                                graph_stats.graph_main_node_result_rows_calls > 0 ||
                                                graph_stats.graph_main_node_result_norm_calls > 0 ||
                                                graph_stats.graph_main_node_result_calls > 0) {
                                            LOG_INF("statistics dflash draft nodes: profiled=%.3f ms graph_compute=%.3f ms qcur/k_draft/v_draft/flash_attn/attn_out/ffn/result_rows/result_norm/result=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu\n",
                                                draft_main_profiled_ms,
                                                (double) graph_stats.decode_graph_compute_us / 1000.0,
                                                (double) graph_stats.graph_main_node_qcur_us / 1000.0,
                                                (double) graph_stats.graph_main_node_k_draft_us / 1000.0,
                                                (double) graph_stats.graph_main_node_v_draft_us / 1000.0,
                                                (double) graph_stats.graph_main_node_flash_attn_us / 1000.0,
                                                (double) graph_stats.graph_main_node_attn_out_us / 1000.0,
                                                (double) graph_stats.graph_main_node_ffn_us / 1000.0,
                                                (double) graph_stats.graph_main_node_result_rows_us / 1000.0,
                                                (double) graph_stats.graph_main_node_result_norm_us / 1000.0,
                                                (double) graph_stats.graph_main_node_result_us / 1000.0,
                                                (unsigned long long) graph_stats.graph_main_node_qcur_calls,
                                                (unsigned long long) graph_stats.graph_main_node_k_draft_calls,
                                                (unsigned long long) graph_stats.graph_main_node_v_draft_calls,
                                                (unsigned long long) graph_stats.graph_main_node_flash_attn_calls,
                                                (unsigned long long) graph_stats.graph_main_node_attn_out_calls,
                                                (unsigned long long) graph_stats.graph_main_node_ffn_calls,
                                                (unsigned long long) graph_stats.graph_main_node_result_rows_calls,
                                                (unsigned long long) graph_stats.graph_main_node_result_norm_calls,
                                                (unsigned long long) graph_stats.graph_main_node_result_calls);
                                            }

                                            if (graph_stats.graph_main_node_k_ctx_view_calls > 0 ||
                                                graph_stats.graph_main_node_v_ctx_view_calls > 0 ||
                                                graph_stats.graph_main_node_k_concat_calls > 0 ||
                                                graph_stats.graph_main_node_v_concat_calls > 0 ||
                                                graph_stats.graph_main_node_k_pad_calls > 0 ||
                                                graph_stats.graph_main_node_v_pad_calls > 0 ||
                                                graph_stats.graph_main_node_k_perm_cont_calls > 0 ||
                                                graph_stats.graph_main_node_v_perm_cont_calls > 0) {
                                            LOG_INF("statistics dflash draft kv traffic: total=%.3f ms graph_compute=%.3f ms k_ctx_view/v_ctx_view/k_concat/v_concat/k_pad/v_pad/k_perm_cont/v_perm_cont=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu\n",
                                                draft_kv_traffic_ms,
                                                (double) graph_stats.decode_graph_compute_us / 1000.0,
                                                (double) graph_stats.graph_main_node_k_ctx_view_us / 1000.0,
                                                (double) graph_stats.graph_main_node_v_ctx_view_us / 1000.0,
                                                (double) graph_stats.graph_main_node_k_concat_us / 1000.0,
                                                (double) graph_stats.graph_main_node_v_concat_us / 1000.0,
                                                (double) graph_stats.graph_main_node_k_pad_us / 1000.0,
                                                (double) graph_stats.graph_main_node_v_pad_us / 1000.0,
                                                (double) graph_stats.graph_main_node_k_perm_cont_us / 1000.0,
                                                (double) graph_stats.graph_main_node_v_perm_cont_us / 1000.0,
                                                (unsigned long long) graph_stats.graph_main_node_k_ctx_view_calls,
                                                (unsigned long long) graph_stats.graph_main_node_v_ctx_view_calls,
                                                (unsigned long long) graph_stats.graph_main_node_k_concat_calls,
                                                (unsigned long long) graph_stats.graph_main_node_v_concat_calls,
                                                (unsigned long long) graph_stats.graph_main_node_k_pad_calls,
                                                (unsigned long long) graph_stats.graph_main_node_v_pad_calls,
                                                (unsigned long long) graph_stats.graph_main_node_k_perm_cont_calls,
                                                (unsigned long long) graph_stats.graph_main_node_v_perm_cont_calls);
                                            }

                            LOG_INF("statistics dflash hot: kv(upload_f/upload_p/upload/compute/sync)=%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%llu replay(accepted_prefix_append)=%.3f ms calls=%zu rows=%zu\n",
                                kv_upload_feature_ms,
                                kv_upload_pos_ms,
                                kv_upload_total_ms,
                                kv_compute_ms,
                                kv_sync_ms,
                                (unsigned long long) graph_stats.graph_kv_cache_calls,
                                replay_append_ms,
                                dflash_state->n_accept_append_calls,
                                dflash_state->n_accept_append_rows);

                    LOG_INF("statistics dflash stages: draft(decode/sample)=%.3f/%.3f ms warmup(collect/append)=%.3f/%.3f ms calls=%zu/%zu rows=%zu/%zu accept(total/output_copy/append)=%.3f/%.3f/%.3f ms calls=%zu/%zu/%zu rows=%zu/%zu/%zu\n",
                            (double) dflash_state->t_draft_decode_us / 1000.0,
                            (double) dflash_state->t_draft_sample_us / 1000.0,
                            (double) dflash_state->t_warmup_collect_us / 1000.0,
                            (double) dflash_state->t_warmup_append_us / 1000.0,
                            dflash_state->n_warmup_collect_calls,
                            dflash_state->n_warmup_append_calls,
                            dflash_state->n_warmup_collect_rows,
                            dflash_state->n_warmup_append_rows,
                            (double) dflash_state->t_accept_commit_us / 1000.0,
                            (double) dflash_state->t_accept_output_copy_us / 1000.0,
                            (double) dflash_state->t_accept_append_us / 1000.0,
                            dflash_state->n_accept_commit_calls,
                            dflash_state->n_accept_output_copy_calls,
                            dflash_state->n_accept_append_calls,
                            dflash_state->n_accept_commit_rows,
                            dflash_state->n_accept_output_copy_rows,
                            dflash_state->n_accept_append_rows);

                    if (dflash_state->n_accept_append_calls > 0) {
                        LOG_INF("statistics dflash replay: append(filter/window_alloc/replace/keep_old/new_rows/commit/log)=%.3f/%.3f/%.3f/%.3f/%.3f/%.3f/%.3f ms calls=%zu replace/slide=%zu/%zu\n",
                                (double) dflash_state->t_accept_append_filter_us / 1000.0,
                            (double) dflash_state->t_accept_append_window_alloc_us / 1000.0,
                                (double) dflash_state->t_accept_append_replace_us / 1000.0,
                                (double) dflash_state->t_accept_append_keep_old_us / 1000.0,
                                (double) dflash_state->t_accept_append_new_rows_us / 1000.0,
                                (double) dflash_state->t_accept_append_commit_detail_us / 1000.0,
                                (double) dflash_state->t_accept_append_log_us / 1000.0,
                                dflash_state->n_accept_append_calls,
                                dflash_state->n_accept_append_replace_calls,
                                dflash_state->n_accept_append_slide_calls);
                    }
                }
            }
        }
    }

    if (spec->tuner && spec->tuner->enabled && slot_tps > 0.0 && n_decoded > 0) {
        auto * mutable_spec = const_cast<common_speculative *>(spec);
        if (active_params) {
            mutable_spec->tuner->end_of_request(slot_tps, n_past, *active_params);
        } else {
            common_params_speculative tmp_params;
            mutable_spec->tuner->end_of_request(slot_tps, n_past, tmp_params);
        }
    }
}

// ----------------------------------------------------------------------------
// MTP
// ----------------------------------------------------------------------------

static common_speculative_state_mtp * common_speculative_get_mtp_state(common_speculative * spec) {
    if (!spec) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_MTP) {
            continue;
        }

        if (auto * mtp_state = dynamic_cast<common_speculative_state_mtp *>(impl.get())) {
            return mtp_state;
        }
    }

    return nullptr;
}

static const common_speculative_state_mtp * common_speculative_get_mtp_state(const common_speculative * spec) {
    return common_speculative_get_mtp_state(const_cast<common_speculative *>(spec));
}

static common_speculative_state_dflash * common_speculative_get_dflash_state(common_speculative * spec) {
    if (!spec) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            continue;
        }

        if (auto * dflash_state = dynamic_cast<common_speculative_state_dflash *>(impl.get())) {
            return dflash_state;
        }
    }

    return nullptr;
}

static const common_speculative_state_dflash * common_speculative_get_dflash_state(const common_speculative * spec) {
    return common_speculative_get_dflash_state(const_cast<common_speculative *>(spec));
}

static int32_t common_speculative_feature_width(const common_speculative * spec) {
    if (const auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        return dflash_state->n_target_features;
    }

    if (const auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
        return mtp_state->n_embd;
    }

    return 0;
}

static mtp_last_embd & mtp_get_last_embd(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto & last = state.draft_cache_by_seq[seq_id];
    if ((int) last.embd.size() != state.n_embd) {
        last.embd.resize(state.n_embd);
    }
    return last;
}

static void mtp_invalidate_cached_draft(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    auto it = state.draft_cache_by_seq.find(seq_id);
    if (it == state.draft_cache_by_seq.end()) {
        return;
    }

    it->second.last_id = -1;
    it->second.prob = 0.0f;
}

static void mtp_invalidate_cached_drafts(common_speculative_state_mtp & state) {
    for (auto & entry : state.draft_cache_by_seq) {
        entry.second.last_id = -1;
        entry.second.prob = 0.0f;
    }
}

static void mtp_store_target_hidden(
        common_speculative_state_mtp & state,
        llama_seq_id seq_id,
        const float * hidden,
        int32_t width) {
    if (hidden == nullptr || width <= 0) {
        return;
    }

    auto & stored = state.target_hidden_by_seq[seq_id];
    stored.assign(hidden, hidden + width);
}

static void mtp_clear_target_hidden(common_speculative_state_mtp & state, llama_seq_id seq_id) {
    state.target_hidden_by_seq.erase(seq_id);
    state.draft_cache_by_seq.erase(seq_id);
}

// DFlash target-window replay and maintenance helpers.
struct dflash_append_breakdown {
    uint64_t filter_us = 0;
    uint64_t window_alloc_us = 0;
    uint64_t replace_us = 0;
    uint64_t keep_old_us = 0;
    uint64_t new_rows_us = 0;
    uint64_t commit_us = 0;
    uint64_t log_us = 0;
    bool replace_call = false;
};

static void dflash_record_window_update(
        common_speculative_state_dflash & state,
        int32_t keep_rows,
        int32_t append_rows,
        bool replace) {
    state.target_window_keep_rows = std::max<int32_t>(0, keep_rows);
    state.target_window_append_rows = std::max<int32_t>(0, append_rows);
    state.target_window_replace = replace;
    state.target_window_version++;
}

static void dflash_ring_reset_rows(
        common_speculative_state_dflash & state,
        const float * rows,
        int32_t n_rows) {
    const size_t row_width = (size_t) state.n_target_features;
    if (n_rows <= 0 || rows == nullptr) {
        state.target_window_ring_write_pos = 0;
        state.target_window_ring_filled = 0;
        return;
    }

    if (state.target_window_ring.size() != (size_t) state.cross_ctx * row_width) {
        state.target_window_ring.resize((size_t) state.cross_ctx * row_width);
    }

    std::memcpy(state.target_window_ring.data(), rows, (size_t) n_rows * row_width * sizeof(float));
    state.target_window_ring_write_pos = n_rows % state.cross_ctx;
    state.target_window_ring_filled = n_rows;
    state.target_window_materialized = false;
}

static void dflash_ring_append_rows(
        common_speculative_state_dflash & state,
        const float * rows,
        int32_t n_rows) {
    const size_t row_width = (size_t) state.n_target_features;
    if (n_rows <= 0 || rows == nullptr) {
        return;
    }

    if (state.target_window_ring.size() != (size_t) state.cross_ctx * row_width) {
        state.target_window_ring.resize((size_t) state.cross_ctx * row_width);
    }

    int32_t write_pos = state.target_window_ring_write_pos;
    int32_t remaining = n_rows;
    const float * src = rows;
    while (remaining > 0) {
        const int32_t chunk_rows = std::min<int32_t>(remaining, state.cross_ctx - write_pos);
        std::memcpy(
                state.target_window_ring.data() + (size_t) write_pos * row_width,
                src,
                (size_t) chunk_rows * row_width * sizeof(float));
        src += (size_t) chunk_rows * row_width;
        remaining -= chunk_rows;
        write_pos = (write_pos + chunk_rows) % state.cross_ctx;
    }

    state.target_window_ring_write_pos = write_pos;
    state.target_window_ring_filled = std::min(state.cross_ctx, state.target_window_ring_filled + n_rows);
    state.target_window_materialized = false;
}

static void dflash_materialize_target_window_features(common_speculative_state_dflash & state) {
    if (state.target_window_materialized || state.target_window_rows <= 0) {
        return;
    }

    const size_t row_width = (size_t) state.n_target_features;
    state.target_window.resize((size_t) state.target_window_rows * row_width);

    const int32_t read_start = (state.target_window_ring_write_pos - state.target_window_rows + state.cross_ctx) % state.cross_ctx;
    const int32_t first_rows = std::min<int32_t>(state.target_window_rows, state.cross_ctx - read_start);
    std::memcpy(
            state.target_window.data(),
            state.target_window_ring.data() + (size_t) read_start * row_width,
            (size_t) first_rows * row_width * sizeof(float));

    const int32_t second_rows = state.target_window_rows - first_rows;
    if (second_rows > 0) {
        std::memcpy(
                state.target_window.data() + (size_t) first_rows * row_width,
                state.target_window_ring.data(),
                (size_t) second_rows * row_width * sizeof(float));
    }

    state.target_window_materialized = true;
}

static bool dflash_append_target_features(
        common_speculative_state_dflash & state,
        const common_speculative_feature_view & features,
        const llama_batch & batch,
        llama_seq_id seq_id,
        dflash_append_breakdown * breakdown = nullptr) {
    GGML_UNUSED(batch);

    if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE ||
            features.width != state.n_target_features ||
            features.rows.empty() ||
            state.cross_ctx <= 0) {
        return false;
    }

    const size_t row_width = (size_t) state.n_target_features;
    std::vector<float> new_rows;
    std::vector<llama_pos> new_positions;
    new_rows.reserve(features.rows.size() * row_width);
    new_positions.reserve(features.rows.size());

    const int64_t t_filter_us = ggml_time_us();
    for (const auto & row : features.rows) {
        if (row.seq_id != seq_id || row.data == nullptr) {
            continue;
        }

        new_positions.push_back(row.pos);
        new_rows.insert(new_rows.end(), row.data, row.data + row_width);
    }
    if (breakdown != nullptr) {
        breakdown->filter_us += (uint64_t) (ggml_time_us() - t_filter_us);
    }

    if (new_positions.empty()) {
        return false;
    }

    const int32_t n_rows = (int32_t) new_positions.size();
    state.n_window_updates++;
    state.n_rows_seen += (size_t) n_rows;
    if (n_rows >= state.cross_ctx) {
        state.n_rows_dropped += (size_t) state.target_window_rows + (size_t) (n_rows - state.cross_ctx);
        const int32_t keep_from = n_rows - state.cross_ctx;
        const int64_t t_replace_us = ggml_time_us();
        state.target_window_pos.assign(new_positions.begin() + keep_from, new_positions.end());
        state.target_window_append_features.assign(
                new_rows.begin() + (ptrdiff_t) keep_from * (ptrdiff_t) row_width,
                new_rows.end());
        dflash_ring_reset_rows(state, state.target_window_append_features.data(), state.cross_ctx);
        if (breakdown != nullptr) {
            breakdown->replace_us += (uint64_t) (ggml_time_us() - t_replace_us);
            breakdown->replace_call = true;
        }

        const int64_t t_commit_us = ggml_time_us();
        state.target_window_rows = state.cross_ctx;
        state.target_window_ring_filled = state.target_window_rows;
        state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
        dflash_record_window_update(state, 0, state.target_window_rows, true);
        if (breakdown != nullptr) {
            breakdown->commit_us += (uint64_t) (ggml_time_us() - t_commit_us);
        }

        const int64_t t_log_us = ggml_time_us();
        dflash_contract_log_append(state, seq_id, new_positions);
        if (breakdown != nullptr) {
            breakdown->log_us += (uint64_t) (ggml_time_us() - t_log_us);
        }
        return true;
    }

    const int32_t keep_old_rows = std::min<int32_t>(state.target_window_rows, state.cross_ctx - n_rows);
    state.n_rows_dropped += (size_t) std::max<int32_t>(0, state.target_window_rows - keep_old_rows);
    const int64_t t_window_alloc_us = ggml_time_us();
    std::vector<llama_pos> & next_window_pos = state.target_window_pos_stage;
    next_window_pos.resize((size_t) (keep_old_rows + n_rows));
    if (breakdown != nullptr) {
        breakdown->window_alloc_us += (uint64_t) (ggml_time_us() - t_window_alloc_us);
    }

    if (keep_old_rows > 0) {
        const int64_t t_keep_old_us = ggml_time_us();
        std::copy(state.target_window_pos.end() - keep_old_rows, state.target_window_pos.end(), next_window_pos.begin());
        if (breakdown != nullptr) {
            breakdown->keep_old_us += (uint64_t) (ggml_time_us() - t_keep_old_us);
        }
    }

    const int64_t t_new_rows_us = ggml_time_us();
    state.target_window_append_features.assign(new_rows.begin(), new_rows.end());
    dflash_ring_append_rows(state, state.target_window_append_features.data(), n_rows);
    std::copy(new_positions.begin(), new_positions.end(), next_window_pos.begin() + keep_old_rows);
    if (breakdown != nullptr) {
        breakdown->new_rows_us += (uint64_t) (ggml_time_us() - t_new_rows_us);
    }

    const int64_t t_commit_us = ggml_time_us();
    state.target_window_pos.swap(next_window_pos);
    next_window_pos.clear();
    state.target_window_rows = keep_old_rows + n_rows;
    state.target_window_ring_filled = state.target_window_rows;
    state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
    dflash_record_window_update(state, keep_old_rows, n_rows, false);
    if (breakdown != nullptr) {
        breakdown->commit_us += (uint64_t) (ggml_time_us() - t_commit_us);
    }

    const int64_t t_log_us = ggml_time_us();
    dflash_contract_log_append(state, seq_id, new_positions);
    if (breakdown != nullptr) {
        breakdown->log_us += (uint64_t) (ggml_time_us() - t_log_us);
    }
    return true;
}

static void dflash_clear_target_features(common_speculative_state_dflash & state) {
    state.target_window.clear();
    state.target_window_pos.clear();
    state.target_window_stage.clear();
    state.target_window_pos_stage.clear();
    state.target_window_append_features.clear();
    state.target_window_rows = 0;
    state.target_window_ring_write_pos = 0;
    state.target_window_ring_filled = 0;
    state.target_window_keep_rows = 0;
    state.target_window_append_rows = 0;
    state.target_window_replace = false;
    state.target_window_materialized = false;
    state.last_target_pos = -1;
    llama_reset_dflash_kv_cache_state(state.ctx_dft);
}

static void dflash_context_shift(
        common_speculative_state_dflash & state,
        llama_pos kv_keep,
        llama_pos kv_discard,
        llama_pos kv_past) {
    if (kv_discard <= 0 || state.target_window_rows <= 0 || state.target_window_pos.empty()) {
        return;
    }

    dflash_materialize_target_window_features(state);

    const size_t row_width = (size_t) state.n_target_features;
    const llama_pos discard_begin = kv_keep;
    const llama_pos discard_end = kv_keep + kv_discard;

    std::vector<float> shifted_rows;
    std::vector<llama_pos> shifted_positions;
    shifted_rows.reserve(state.target_window.size());
    shifted_positions.reserve(state.target_window_pos.size());

    for (int32_t row = 0; row < state.target_window_rows; ++row) {
        llama_pos pos = state.target_window_pos[(size_t) row];
        if (pos >= discard_begin && pos < discard_end) {
            continue;
        }

        if (pos >= discard_end && pos < kv_past) {
            pos -= kv_discard;
        }

        const float * row_src = state.target_window.data() + (size_t) row * row_width;
        shifted_rows.insert(shifted_rows.end(), row_src, row_src + row_width);
        shifted_positions.push_back(pos);
    }

    state.target_window = std::move(shifted_rows);
    state.target_window_pos = std::move(shifted_positions);
    state.target_window_rows = (int32_t) state.target_window_pos.size();
    dflash_ring_reset_rows(state, state.target_window.data(), state.target_window_rows);
    state.last_target_pos = state.target_window_pos.empty() ? -1 : state.target_window_pos.back();
    dflash_record_window_update(state, 0, state.target_window_rows, true);
    llama_reset_dflash_kv_cache_state(state.ctx_dft);
    state.n_context_shifts++;
}

static bool common_speculative_capture_target_features(common_speculative * spec, const common_speculative_feature_view & features) {
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr || features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width <= 0) {
        return false;
    }

    bool captured = false;
    for (const auto & row : features.rows) {
        if (row.data == nullptr) {
            continue;
        }

        mtp_store_target_hidden(*mtp_state, row.seq_id, row.data, features.width);
        mtp_invalidate_cached_draft(*mtp_state, row.seq_id);
        captured = true;
    }

    return captured;
}

bool common_speculative_has_sequence_hidden(const common_speculative * spec, llama_seq_id seq_id) {
    const auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return false;
    }

    auto it = mtp_state->target_hidden_by_seq.find(seq_id);
    return it != mtp_state->target_hidden_by_seq.end() && !it->second.empty();
}

void common_speculative_clear_sequence_hidden(common_speculative * spec, llama_seq_id seq_id) {
    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state != nullptr) {
        mtp_clear_target_hidden(*mtp_state, seq_id);
    }

    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        dflash_clear_target_features(*dflash_state);
    }
}

llama_context * common_speculative_get_companion_ctx(common_speculative * spec) {
    if (auto * mtp_state = common_speculative_get_mtp_state(spec); mtp_state != nullptr) {
        return mtp_state->ctx_mtp;
    }

    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        return dflash_state->ctx_dft;
    }

    return nullptr;
}

static int32_t mtp_accept_batch(
        common_speculative_state_mtp & state,
        const llama_batch & accepted_batch,
        llama_seq_id seq_id,
        const float * hidden_rows) {
    if (accepted_batch.n_tokens == 0 || hidden_rows == nullptr) {
        return 0;
    }

    const size_t hidden_rows_floats = (size_t) accepted_batch.n_tokens * state.n_embd;
    if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, hidden_rows, hidden_rows_floats)) {
        return -1;
    }
    if (mtp_update_kv_cache(state.ctx_mtp, accepted_batch, false) != 0) {
        return -1;
    }

    auto & last = mtp_get_last_embd(state, seq_id);
    const float * embd = llama_get_embeddings_ith(state.ctx_mtp, accepted_batch.n_tokens - 1);
    if (embd != nullptr) {
        std::memcpy(last.embd.data(), embd, last.embd.size() * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(state.ctx_mtp, last.embd.data(), last.embd.size())) {
            return -1;
        }
        last.last_id = common_sampler_sample_speculative(nullptr, state.ctx_mtp, accepted_batch.n_tokens - 1, &last.prob);
    }

    return 0;
}

int32_t common_speculative_on_target_batch(
        common_speculative * spec,
        const llama_batch & batch,
        const common_speculative_feature_view & features,
    bool is_prompt_warmup) {
    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || batch.n_tokens <= 0) {
            return 0;
        }

        if (features.width != dflash_state->n_target_features) {
            LOG_ERR("%s: DFlash feature width mismatch: got %d expected %d\n",
                    __func__, features.width, dflash_state->n_target_features);
            return -1;
        }

        if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[0] <= 0 || batch.seq_id[0] == nullptr) {
            return -1;
        }

        const llama_seq_id seq_id = batch.seq_id[0][0];
        for (int i = 0; i < batch.n_tokens; ++i) {
            if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
                return -1;
            }
        }

        dflash_append_breakdown append_breakdown;
        const int64_t t_append_us = ggml_time_us();
        if (!dflash_append_target_features(*dflash_state, features, batch, seq_id, &append_breakdown)) {
            return -1;
        }

        const uint64_t append_us = (uint64_t) (ggml_time_us() - t_append_us);
        if (is_prompt_warmup) {
            dflash_state->t_warmup_append_us += append_us;
            dflash_state->n_warmup_append_calls++;
            dflash_state->n_warmup_append_rows += (size_t) batch.n_tokens;
        } else {
            dflash_state->t_accept_append_us += append_us;
            dflash_state->t_accept_append_filter_us += append_breakdown.filter_us;
            dflash_state->t_accept_append_window_alloc_us += append_breakdown.window_alloc_us;
            dflash_state->t_accept_append_replace_us += append_breakdown.replace_us;
            dflash_state->t_accept_append_keep_old_us += append_breakdown.keep_old_us;
            dflash_state->t_accept_append_new_rows_us += append_breakdown.new_rows_us;
            dflash_state->t_accept_append_commit_detail_us += append_breakdown.commit_us;
            dflash_state->t_accept_append_log_us += append_breakdown.log_us;
            dflash_state->n_accept_append_calls++;
            dflash_state->n_accept_append_rows += (size_t) batch.n_tokens;
            if (append_breakdown.replace_call) {
                dflash_state->n_accept_append_replace_calls++;
            } else {
                dflash_state->n_accept_append_slide_calls++;
            }
        }

        return 0;
    }

    auto * mtp_state = common_speculative_get_mtp_state(spec);
    if (mtp_state == nullptr) {
        return 0;
    }

    if (features.kind != COMMON_SPECULATIVE_FEATURE_HIDDEN_STATE || features.width <= 0 || batch.n_tokens <= 0) {
        return 0;
    }

    if (batch.n_seq_id == nullptr || batch.seq_id == nullptr || batch.n_seq_id[0] <= 0 || batch.seq_id[0] == nullptr) {
        return -1;
    }

    const llama_seq_id seq_id = batch.seq_id[0][0];
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return -1;
        }
    }

    std::vector<float> hidden_rows_storage;
    if (!common_speculative_feature_view_copy_batch_rows(features, batch, seq_id, &hidden_rows_storage)) {
        return -1;
    }

    const float * first_hidden = hidden_rows_storage.data();
    const float * last_hidden = hidden_rows_storage.data() + (size_t) (batch.n_tokens - 1) * features.width;
    mtp_store_target_hidden(*mtp_state, seq_id, last_hidden, features.width);

    if (mtp_state->constant_draft_positions) {
        mtp_invalidate_cached_draft(*mtp_state, seq_id);
        return 0;
    }

    if (is_prompt_warmup) {
        if (!llama_set_draft_input_hidden_state_copy(mtp_state->ctx_mtp, hidden_rows_storage.data(), hidden_rows_storage.size())) {
            return -1;
        }
        const int32_t ret = mtp_update_kv_cache(mtp_state->ctx_mtp, batch, true);
        mtp_invalidate_cached_draft(*mtp_state, seq_id);
        return ret;
    }

    return mtp_accept_batch(*mtp_state, batch, seq_id, first_hidden);
}

common_speculative_type common_speculative_current_type(const common_speculative * spec) {
    if (spec == nullptr || spec->curr_impl == nullptr) {
        return COMMON_SPECULATIVE_TYPE_NONE;
    }

    return spec->curr_impl->type;
}

void common_speculative_context_shift(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_pos            kv_keep,
        llama_pos            kv_discard,
        llama_pos            kv_past) {
    if (auto * ctx_mtp = common_speculative_get_companion_ctx(spec); ctx_mtp != nullptr) {
        llama_kv_cache_seq_rm (ctx_mtp, seq_id, kv_keep, kv_keep + kv_discard);
        llama_kv_cache_seq_add(ctx_mtp, seq_id, kv_keep + kv_discard, kv_past, -kv_discard);
    }

    if (auto * dflash_state = common_speculative_get_dflash_state(spec); dflash_state != nullptr) {
        dflash_context_shift(*dflash_state, kv_keep, kv_discard, kv_past);
    }
}

std::vector<llama_token> mtp_speculative_gen_draft(
    common_speculative_state_mtp & state,
    struct common_sampler * smpl,
    struct llama_context * ctx,
    int n_draft,
    float p_min,
    llama_token id_last,
    llama_pos n_past,
    llama_seq_id seq_id,
    bool constant_draft_positions) {

    llama_tokens drafts;
    drafts.reserve(n_draft);

    if (!smpl) return drafts;

    if (n_draft <= 0) {
        mtp_invalidate_cached_draft(state, seq_id);
        return drafts;
    }

    common_sampler_reset(smpl);

    llama_batch mtp_batch = llama_batch_init(1, 0, 1);
    llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN);

    float prob;
    auto prob_ptr = p_min > 0 ? &prob : nullptr;

    llama_token current_input_id = id_last;
    llama_pos current_n_past = n_past;
    const int n_embd = llama_mtp_state_n_embd(ctx);

    auto & last = mtp_get_last_embd(state, seq_id);
    int i0 = 0;
    if (last.last_id >= 0) {
        if (last.prob < p_min) {
            n_draft = 1;
        }
        current_input_id = last.last_id;
        last.last_id = -1;
        drafts.push_back(current_input_id);
        current_n_past++;
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            llama_batch_free(mtp_batch);
            llama_set_mtp_op_type(ctx, MTP_OP_NONE);
            return drafts;
        }
        i0 = 1;
    }

    int n_decode = 0;
    for (int i = i0; i < n_draft; ++i) {
        mtp_batch.n_tokens = 0;
        const llama_pos draft_pos = constant_draft_positions ? n_past : current_n_past;
        common_batch_add(mtp_batch, current_input_id, draft_pos, {seq_id}, true);

        ++n_decode;
        if (llama_decode(ctx, mtp_batch) != 0) {
            break;
        }

        llama_token id_next = common_sampler_sample_speculative(smpl, ctx, 0, prob_ptr);

        if (i > 0 && prob_ptr && prob < p_min) {
            break;
        }

        drafts.push_back(id_next);

        const float * emb = llama_get_embeddings_ith(ctx, 0);
        if (!emb) {
            break;
        }

        // Keep a stable copy because later decode steps reuse ctx->embd storage.
        memcpy(last.embd.data(), emb, n_embd * sizeof(float));
        if (!llama_set_draft_input_hidden_state_copy(ctx, last.embd.data(), last.embd.size())) {
            break;
        }

        current_input_id = id_next;
        current_n_past++;

        if (prob_ptr && prob < p_min) {
            break;
        }
    }
    llama_batch_free(mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    // Purge the metadata for the draft tokens.
    // This prevents cache state corruption where two cells map to the same logical position.
    // If the state contained in `last` had a valid token id and probability, it means that we
    // have previously run an "accept" batch, where the token sampled from the main model was included.
    // In that case, we need to discard all tokens that we ran here to get the KV cache to the correct state.
    //   => for i0 = 1 we discard from n_past
    // But if we did not have a valid last token_id, it means the first token we run was sampled from the
    // main model. Hence we want to keep this token in the KV cache and discard all other tokens.
    //   => for i0 = 0 we discard from n_past + 1
    if (n_decode > 0) {
        llama_kv_cache_seq_rm(ctx, seq_id, n_past + 1 - i0, n_past + n_decode + 2);
    }

    return drafts;
}


int32_t mtp_update_kv_cache(struct llama_context * ctx, const llama_batch& batch, bool is_prompt_warmup) {
    if (batch.n_tokens == 0) {
        return 0;
    }

    llama_seq_id seq_id    = batch.seq_id[0][0];
    llama_pos    start_pos = batch.pos[0];

    if (llama_kv_cache_seq_pos_max(ctx, seq_id) >= start_pos) {
        llama_kv_cache_seq_rm(ctx, seq_id, start_pos, -1);
    }

    LOG_DBG("[MTP-UPDATE|%s] Updating %d tokens for seq_id %d from pos %d...\n",
            is_prompt_warmup ? "PROMPT_WARMUP" : "GEN_ACCEPTED", batch.n_tokens, seq_id, (int)start_pos);

    // We never need all logits. We only need the logits of the last token so we can sample
    // the next draft token. In the MTP_OP_WARMUP case we do not need logits at all, but just
    // in case we also get the logits of the last token.
    llama_batch mtp_batch = batch;
    for (int i = 0; i < mtp_batch.n_tokens; ++i) {
        mtp_batch.logits[i] = false;
    }
    mtp_batch.logits[mtp_batch.n_tokens-1] = true;
    if (is_prompt_warmup) {
        llama_set_mtp_op_type(ctx, MTP_OP_WARMUP);
    } else {
        llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    }

    const int32_t ret = llama_decode(ctx, mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);
    return ret;
}
