#include "speculative.h"

#include "common.h"
#include "sampling.h"
#include "llama-impl.h"

#include <cstring>
#include <algorithm>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct llama_speculative {
    struct llama_context * ctx;
    struct llama_sampling_context * smpl;

    llama_batch batch;
    std::vector<llama_token> prompt;
};

struct llama_speculative * llama_speculative_init(
        struct llama_context * ctx_dft) {
    auto * result = new llama_speculative {
        /* .ctx    = */ ctx_dft,
        /* .smpl   = */ nullptr,
        /* .batch  = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .prompt = */ {},
    };

    // TODO: optimize or pass from outside?
#if 0
    {
        llama_sampling_params params;
        params.no_perf = false;

        params.top_k = 40;
        params.top_p = 0.9;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = llama_sampler_init(llama_get_model(ctx_dft), params);
    }
#else
    {
        llama_sampling_params params;
        params.top_k = 10;
        params.samplers_sequence = {
            llama_sampler_type::TOP_K,
        };
        const auto *model_dft = llama_get_model(ctx_dft);
        result->smpl = llama_sampling_init(llama_get_model_vocab(model_dft), params);
    }
#endif

    return result;
}

void llama_speculative_free(struct llama_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    llama_sampling_free(spec->smpl);

    llama_batch_free(spec->batch);

    delete spec;
}

bool llama_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_get_model_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_get_model_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(model_tgt);
    LLAMA_LOG_INFO("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(model_dft);
    LLAMA_LOG_INFO("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LLAMA_LOG_ERROR("%s: draft model vocab type must match target model to use speculation but "
                     "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_add_bos_token(model_tgt) != llama_add_bos_token(model_dft) ||
        llama_add_eos_token(model_tgt) != llama_add_eos_token(model_dft) ||
        llama_token_bos(model_tgt) != llama_token_bos(model_dft) ||
        llama_token_eos(model_tgt) != llama_token_eos(model_dft)) {
        LLAMA_LOG_ERROR("%s: draft vocab special tokens must match target vocab to use speculation\n", __func__);
        LLAMA_LOG_ERROR("%s: tgt: bos = %d (%d), eos = %d (%d)\n", __func__, llama_token_bos(model_tgt), llama_add_bos_token(model_tgt), llama_token_eos(model_tgt), llama_add_eos_token(model_tgt));
        LLAMA_LOG_ERROR("%s: dft: bos = %d (%d), eos = %d (%d)\n", __func__, llama_token_bos(model_dft), llama_add_bos_token(model_dft), llama_token_eos(model_dft), llama_add_eos_token(model_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);

        const int model_diff = std::abs(n_vocab_tgt - n_vocab_dft);

        if (model_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LLAMA_LOG_ERROR("%s: draft model vocab must closely match target model to use speculation but "
                         "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    __func__, n_vocab_tgt, n_vocab_dft, model_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
            const char * token_text_dft = llama_token_get_text(model_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LLAMA_LOG_ERROR("%s: draft vocab vocab must match target vocab to use speculation but "
                             "token %d content differs - target '%s', draft '%s'\n", __func__, i,
                        llama_token_to_piece(ctx_tgt, i).c_str(),
                        llama_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

std::vector<llama_token> llama_speculative_gen_draft(
        struct llama_speculative * spec,
        struct llama_speculative_params params,
        const std::vector<llama_token> & prompt_tgt,
        llama_token id_last) {
    auto & batch  = spec->batch;
    auto & ctx    = spec->ctx;
    auto & smpl   = spec->smpl;
    auto & prompt = spec->prompt;

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(ctx) - params.n_draft;

    const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
    for (int i = 0; i < (int) prompt.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt_tgt.size() &&
               i       + cur < (int) prompt.size() &&
               prompt_tgt[i_start + cur] == prompt[i + cur]) {
            cur++;
        }

        if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LLAMA_LOG_INFO("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt.size());

    std::vector<llama_token> result;
    result.reserve(params.n_draft);

    if (reuse_n == 0) {
        llama_kv_cache_clear(ctx);

        prompt.clear();
    } else {
        // this happens when a previous draft has been discarded (for example, due to being too small), but the
        // target model agreed with it. in this case, we simply pass back the previous results to save compute
        if (reuse_i + reuse_n < (int) prompt.size() && prompt[reuse_i + reuse_n] == id_last) {
            for (int i = reuse_i + reuse_n + 1; i < (int) prompt.size(); ++i) {
                result.push_back(prompt[i]);

                if (params.n_draft <= (int) result.size()) {
                    break;
                }
            }

            return result;
        }

        if (reuse_i > 0) {
            llama_kv_cache_seq_rm (ctx, 0, 0, reuse_i);
            llama_kv_cache_seq_add(ctx, 0, reuse_i, -1, -reuse_i);

            prompt.erase(prompt.begin(), prompt.begin() + reuse_i);
        }

        if (reuse_n < (int) prompt.size()) {
            llama_kv_cache_seq_rm (ctx, 0, reuse_n, -1);

            prompt.erase(prompt.begin() + reuse_n, prompt.end());
        }
    }

    // prepare a batch to evaluate any new tokens in the prompt
    llama_batch_clear(batch);

    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        //LLAMA_LOG_INFO("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_tgt[i]);
        llama_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);

        prompt.push_back(prompt_tgt[i]);
    }

    // we should rarely end-up here during normal decoding
    if (batch.n_tokens > 0) {
        //LLAMA_LOG_INFO("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

        llama_decode(ctx, batch);
    }

    const llama_pos n_past = prompt.size();

    LLAMA_LOG_INFO("%s: n_past = %d\n", __func__, n_past);

    llama_batch_clear(batch);
    llama_batch_add  (batch, id_last, n_past, { 0 }, true);

    prompt.push_back(id_last);

    //LLAMA_LOG_INFO("%s: draft prompt: %s\n", __func__, string_from(ctx, prompt).c_str());

    llama_decode(ctx, batch);

    llama_sampling_reset(smpl);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < params.n_draft; ++i) {
        llama_batch_clear(batch);

        llama_sampling_sample(smpl, ctx, nullptr, 0);

        const auto * cur_p = llama_sampling_get_candidates(smpl);

        // for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
        //     LLAMA_LOG_INFO(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
        //             k, i, cur_p->data[k].id, cur_p->data[k].p, llama_token_to_piece(ctx, cur_p->data[k].id).c_str());
        // }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        llama_sampling_accept(smpl, ctx, id, true);

        result.push_back(id);

        if (params.n_draft <= (int) result.size()) {
            break;
        }

        // only collect very high-confidence draft tokens
        if (cur_p->data[0].p < params.p_min) {
            break;
        }

        llama_batch_add(batch, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(ctx, batch);

        prompt.push_back(id);
    }

    return result;
}
