#include "speculative.h"

#include "common.h"
#include "sampling.h"
#include "llama-impl.h"
#include "llama-vocab.h"
#include <cstring>
#include <algorithm>
#include <map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct llama_speculative {
    struct llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    struct llama_context * ctx_dft;
    struct llama_sampling_context * smpl;

    llama_batch batch;
    std::vector<llama_token> prompt_dft;
    bool vocab_dft_compatible = true; // whether retokenization is needed
    std::map<std::string, std::string> tgt_dft_replacements = {};
};

struct llama_speculative * llama_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    auto * result = new llama_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ ctx_dft,
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ false,
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

    result->vocab_dft_compatible = llama_speculative_are_compatible(ctx_tgt, ctx_dft);
    LLAMA_LOG_INFO("vocab_dft_compatible = %d\n", result->vocab_dft_compatible);

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
        LLAMA_LOG_INFO("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LLAMA_LOG_INFO("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_add_bos_token(model_tgt) != llama_add_bos_token(model_dft) ||
        llama_add_eos_token(model_tgt) != llama_add_eos_token(model_dft) ||
        llama_token_bos(model_tgt) != llama_token_bos(model_dft) ||
        llama_token_eos(model_tgt) != llama_token_eos(model_dft)
    ) {
        LLAMA_LOG_INFO("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);

        const int model_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (model_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LLAMA_LOG_INFO("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LLAMA_LOG_INFO("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, n_vocab_dft, model_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
            const char * token_text_dft = llama_token_get_text(model_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LLAMA_LOG_INFO("%s: draft model vocab must match target model to use speculation but ", __func__);
                LLAMA_LOG_INFO("token %d content differs - target '%s', draft '%s'\n", i,
                        llama_token_to_piece(ctx_tgt, i).c_str(),
                        llama_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

void llama_speculative_add_replacement_tgt_dft(
        struct llama_speculative * spec,
        const char *source, const char *dest) {
    spec->tgt_dft_replacements[source] = dest;
}

static std::string replace_to_dft(
        struct llama_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto & pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.first);
        while (pos != std::string::npos) {
            result.replace(pos, pair.first.length(), pair.second);
            pos = result.find(pair.first, pos + pair.second.length());
        }
    }
    return result;
}

static std::string replace_to_tgt(
        struct llama_speculative * spec,
        const std::string& input) {
    std::vector<std::pair<std::string, std::string>> sorted_pairs(spec->tgt_dft_replacements.begin(), spec->tgt_dft_replacements.end());
    std::sort(sorted_pairs.begin(), sorted_pairs.end(), [](const auto &a, const auto &b) {
        return a.second.length() > b.second.length(); // Sort by length in descending order
    });

    std::string result = input;
    for (const auto & pair : sorted_pairs) {
        size_t pos = 0;
        while ((pos = result.find(pair.second, pos)) != std::string::npos) {
            result.replace(pos, pair.second.length(), pair.first);
            pos += pair.first.length();
        }
    }
    return result;
}

std::vector<llama_token> llama_speculative_gen_draft(
        struct llama_speculative * spec,
        struct llama_speculative_params params,
        const std::vector<llama_token> & prompt_tgt_main_model, // specified in target model vocab
        llama_token id_last) {
    auto & batch  = spec->batch;
    auto & ctx_tgt = spec->ctx_tgt;
    auto & ctx_dft = spec->ctx_dft;
    auto & smpl   = spec->smpl;
    auto & prompt_dft = spec->prompt_dft;

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(ctx_dft) - params.n_draft;

    std::vector<llama_token> prompt_tgt_draft_model;
    if (!spec->vocab_dft_compatible) {
        std::string text;
        text = llama_detokenize(ctx_tgt, prompt_tgt_main_model, true);
        text = replace_to_dft(spec, text);
        LLAMA_LOG_INFO("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());
        prompt_tgt_draft_model = llama_tokenize(ctx_dft, text, false, true);

        // convert id_last to draft vocab
        std::vector<llama_token> id_last_vec(1, id_last);
        text = llama_detokenize(ctx_tgt, id_last_vec);
        LLAMA_LOG_INFO("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
        id_last = llama_tokenize(ctx_dft, text, false, true)[0];
    }
    // prompt_tgt's tokens will always be compatible with ctx_dft
    const std::vector<llama_token> &prompt_tgt =
        spec->vocab_dft_compatible ? prompt_tgt_main_model : prompt_tgt_draft_model;

    const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
    for (int i = 0; i < (int) prompt_dft.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt_tgt.size() &&
               i       + cur < (int) prompt_dft.size() &&
               prompt_tgt[i_start + cur] == prompt_dft[i + cur]) {
            cur++;
        }

        if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LLAMA_LOG_INFO("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

    std::vector<llama_token> result;
    result.reserve(params.n_draft);

    if (reuse_n == 0) {
        llama_kv_cache_clear(ctx_dft);

        prompt_dft.clear();
    } else {
        // this happens when a previous draft has been discarded (for example, due to being too small), but the
        // target model agreed with it. in this case, we simply pass back the previous results to save compute
        if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
            for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                result.push_back(prompt_dft[i]);

                if (params.n_draft <= (int) result.size()) {
                    break;
                }
            }

            return result;
        }

        if (reuse_i > 0) {
            llama_kv_cache_seq_rm (ctx_dft, 0, 0, reuse_i);
            llama_kv_cache_seq_add(ctx_dft, 0, reuse_i, -1, -reuse_i);

            prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
        }

        if (reuse_n < (int) prompt_dft.size()) {
            llama_kv_cache_seq_rm (ctx_dft, 0, reuse_n, -1);

            prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
        }
    }

    // prepare a batch to evaluate any new tokens in the prompt
    llama_batch_clear(batch);

    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        //LLAMA_LOG_INFO("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_tgt[i]);
        llama_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);

        prompt_dft.push_back(prompt_tgt[i]);
    }

    // we should rarely end-up here during normal decoding
    if (batch.n_tokens > 0) {
        //LLAMA_LOG_INFO("%s: draft prompt batch: %s\n", __func__, string_from(ctx_dft, batch).c_str());

        llama_decode(ctx_dft, batch);
    }

    const llama_pos n_past = prompt_dft.size();

    // LLAMA_LOG_INFO("%s: n_past = %d\n", __func__, n_past);

    llama_batch_clear(batch);
    llama_batch_add  (batch, id_last, n_past, { 0 }, true);

    prompt_dft.push_back(id_last);

    //LLAMA_LOG_INFO("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

    llama_decode(ctx_dft, batch);

    llama_sampling_reset(llama_get_vocab(ctx_dft), smpl);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < params.n_draft; ++i) {
        llama_batch_clear(batch);

        llama_sampling_sample(smpl, ctx_dft, nullptr, 0);

        const auto * cur_p = llama_sampling_get_candidates(smpl);

        // for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
        //     LLAMA_LOG_INFO(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
        //             k, i, cur_p->data[k].id, cur_p->data[k].p, llama_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
        // }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        llama_sampling_accept(smpl, ctx_dft, id, true);

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
        llama_decode(ctx_dft, batch);

        prompt_dft.push_back(id);
    }

    if (!spec->vocab_dft_compatible) {
        std::string detokenized = llama_detokenize(ctx_dft, result, true);
        detokenized = replace_to_tgt(spec, detokenized);
        LLAMA_LOG_INFO("draft->main detokenized string: '%s'\n", detokenized.c_str());
        result = llama_tokenize(ctx_tgt, detokenized, false, true);
        if (result.size() > (size_t)params.n_draft) {
            result.resize(params.n_draft);
        }
    }
    return result;
}

