#define LLAMA_API_INTERNAL
#include "sampling.h"
#include "llama-vocab.h"
#include "common.h"
#include <random>
#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    struct common_sampler * result = new common_sampler();

    result->params  = params;
    result->grammar = nullptr;


    struct llama_grammar* grmr;
    if (params.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    }
    else {
        std::vector<std::string> trigger_patterns;
        std::vector<std::string> patterns_anywhere;
        std::vector<llama_token> trigger_tokens;
        for (const auto& trigger : params.grammar_triggers) {
            switch (trigger.type) {
            case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            {
                const auto& word = trigger.value;
                patterns_anywhere.push_back(regex_escape(word));
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            {
                patterns_anywhere.push_back(trigger.value);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            {
                trigger_patterns.push_back(trigger.value);
                break;
            }
            case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            {
                const auto token = trigger.token;
                trigger_tokens.push_back(token);
                break;
            }
            default:
                GGML_ASSERT(false && "unknown trigger type");
            }
        }

        if (!patterns_anywhere.empty()) {
            trigger_patterns.push_back("^[\\s\\S]*?(" + string_join(patterns_anywhere, "|") + ")[\\s\\S]*");
        }

        std::vector<const char*> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto& regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }
        grmr = params.grammar_lazy
            ? llama_sampler_init_grammar_lazy_patterns(vocab, params.grammar.c_str(), "root",
                trigger_patterns_c.data(), trigger_patterns_c.size(),
                trigger_tokens.data(), trigger_tokens.size())
            : llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");

        result->prev.resize(params.n_prev);
        result->n_valid = 0;
	    result->grammar_str = params.grammar;
	    result->grammar_root = "root";
    }
    result->grammar = grmr;
    llama_sampling_set_rng_seed(result, params.seed);
    for (const auto& cnstr : params.samplers_sequence)
    {
        switch (cnstr)
        {
            case llama_sampler_type::DRY:
            {
                std::vector<const char*> c_breakers;
                c_breakers.reserve(params.dry_sequence_breakers.size());
                for (const auto& str : params.dry_sequence_breakers)
                {
                    c_breakers.push_back(str.c_str());
                }
                result->smpl=llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size());

                break;
            }
            case llama_sampler_type::ADAPTIVE_P:
            {
                GGML_ASSERT(vocab);
                auto n_vocab = llama_vocab_n_tokens(vocab);
                result->adapt_p_ctx = llama_init_adaptive_p(n_vocab, params.adaptive_target, params.adaptive_decay, params.adaptive_updt_w_cur, result->rng());
                break;
            }
            default:
                break;
        }
    }

    return result;
}

void common_sampler_free(struct common_sampler * ctx) {
    if (ctx->grammar != NULL) {
        llama_grammar_free(ctx->grammar);
    }
    if (ctx->smpl !=NULL)
        llama_sampler_dry_free(ctx->smpl);
    delete ctx;
}

static void llama_grammar_reset(common_sampler * ctx) {
    ctx->prev.clear();
    if (!ctx->grammar) {
        return;
    }
    std::vector<const char*>  trigger_patterns_c;
    trigger_patterns_c.reserve(ctx->grammar->trigger_patterns.size());
    for (auto& trigger_pattern : ctx->grammar->trigger_patterns) {
        trigger_patterns_c.push_back(trigger_pattern.pattern.c_str());
    }

    auto* grammar_new = llama_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str(),
        ctx->grammar->lazy, trigger_patterns_c.data(), trigger_patterns_c.size(),
        ctx->grammar->trigger_tokens.data(), ctx->grammar->trigger_tokens.size());

    llama_grammar_free_impl(ctx->grammar);
    ctx->grammar = grammar_new;
}

void common_sampler_reset(common_sampler * ctx) {
    llama_grammar_reset(ctx);
    llama_sampler_dry_reset(ctx->smpl);
}

void llama_sampling_set_rng_seed(struct common_sampler * ctx, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = std::random_device{}();
    }
    ctx->rng.seed(seed);
}

void common_sampler_clone(common_sampler * src, common_sampler * dst) {
    if (dst->grammar) {
        llama_grammar_free(dst->grammar);
        dst->grammar = nullptr;
    }

    if (src->grammar) {
        dst->grammar_root = src->grammar_root;
        dst->grammar_str = src->grammar_str;
        dst->grammar = llama_grammar_copy(src->grammar);
    }

    dst->prev = src->prev;
    dst->smpl = llama_sampler_dry_clone(src->smpl);
}

llama_token llama_sampling_last(common_sampler * ctx) {
    return ctx->prev.back();
}

std::string llama_sampling_prev_str(common_sampler * ctx_sampling, llama_context * ctx_main, int n) {
    const int size = ctx_sampling->prev.size();

    n = std::min(n, size);

    std::string result;

    for (int i = size - n; i < size; i++) {
        result += common_token_to_piece(ctx_main, ctx_sampling->prev[i]);
    }

    return result;
}

std::string llama_sampling_print(const common_params_sampling & params) {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\ttop_k = %d, tfs_z = %.3f, top_p = %.3f, min_p = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f\n"
            "\txtc_probability = %.3f, xtc_threshold = %.3f, top_n_sigma = %.3f\n"
            "\tadaptive_target = %.2f, adaptive_decay = %.2f",
            params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present,
            params.top_k, params.tfs_z, params.top_p, params.min_p, params.typical_p, params.temp,
            params.mirostat, params.mirostat_eta, params.mirostat_tau,
            params.xtc_probability, params.xtc_threshold, params.top_n_sigma,
            params.adaptive_target, params.adaptive_decay);

    return std::string(result);
}

std::string llama_sampling_order_print(const common_params_sampling & params) {
    std::string result = "CFG -> Penalties ";
    if (params.mirostat == 0) {
        for (auto sampler_type : params.samplers_sequence) {
            const auto sampler_type_name = llama_sampling_type_to_str(sampler_type);
            if (!sampler_type_name.empty()) {
                result += "-> " + sampler_type_name + " ";
            }
        }
    } else {
        result += "-> mirostat ";
    }

    return result;
}

std::string llama_sampling_type_to_str(llama_sampler_type sampler_type) {
    switch (sampler_type) {
        case llama_sampler_type::DRY:         return "dry";
        case llama_sampler_type::TOP_K:       return "top_k";
        case llama_sampler_type::TFS_Z:       return "tfs_z";
        case llama_sampler_type::TYPICAL_P:   return "typical_p";
        case llama_sampler_type::TOP_P:       return "top_p";
        case llama_sampler_type::MIN_P:       return "min_p";
        case llama_sampler_type::TEMPERATURE: return "temperature";
        case llama_sampler_type::XTC        : return "xtc";
        case llama_sampler_type::TOP_N_SIGMA: return "top_n_sigma";
        case llama_sampler_type::ADAPTIVE_P : return "adaptive_p";
        default : return "";
    }
}

std::vector<llama_sampler_type> llama_sampling_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, llama_sampler_type> sampler_canonical_name_map {
        {"dry",         llama_sampler_type::DRY},
        {"top_k",       llama_sampler_type::TOP_K},
        {"top_p",       llama_sampler_type::TOP_P},
        {"typical_p",   llama_sampler_type::TYPICAL_P},
        {"min_p",       llama_sampler_type::MIN_P},
        {"tfs_z",       llama_sampler_type::TFS_Z},
        {"xtc",         llama_sampler_type::XTC},
        {"top_n_sigma", llama_sampler_type::TOP_N_SIGMA},
        {"temperature", llama_sampler_type::TEMPERATURE},
        {"adaptive_p",  llama_sampler_type::ADAPTIVE_P},
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, llama_sampler_type> sampler_alt_name_map {
        {"dry",         llama_sampler_type::DRY},
        {"top-k",       llama_sampler_type::TOP_K},
        {"top-p",       llama_sampler_type::TOP_P},
        {"nucleus",     llama_sampler_type::TOP_P},
        {"typical-p",   llama_sampler_type::TYPICAL_P},
        {"typical",     llama_sampler_type::TYPICAL_P},
        {"min-p",       llama_sampler_type::MIN_P},
        {"tfs-z",       llama_sampler_type::TFS_Z},
        {"tfs",         llama_sampler_type::TFS_Z},
        {"xtc",         llama_sampler_type::XTC},
        {"top-n-sigma", llama_sampler_type::TOP_N_SIGMA},
        {"temp",        llama_sampler_type::TEMPERATURE},
        {"adaptive-p",  llama_sampler_type::ADAPTIVE_P},
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names.size());
    for (const auto & name : names)
    {
        auto sampler_item = sampler_canonical_name_map.find(name);
        if (sampler_item != sampler_canonical_name_map.end())
        {
            sampler_types.push_back(sampler_item->second);
        }
        else
        {
            if (allow_alt_names)
            {
                sampler_item = sampler_alt_name_map.find(name);
                if (sampler_item != sampler_alt_name_map.end())
                {
                    sampler_types.push_back(sampler_item->second);
                }
            }
        }
    }
    return sampler_types;
}

std::vector<llama_sampler_type> llama_sampling_types_from_chars(const std::string & names_string) {
    std::unordered_map<char, llama_sampler_type> sampler_name_map {
        {'d', llama_sampler_type::DRY},
        {'k', llama_sampler_type::TOP_K},
        {'p', llama_sampler_type::TOP_P},
        {'y', llama_sampler_type::TYPICAL_P},
        {'m', llama_sampler_type::MIN_P},
        {'f', llama_sampler_type::TFS_Z},
        {'x', llama_sampler_type::XTC},
        {'n', llama_sampler_type::TOP_N_SIGMA},
        {'t', llama_sampler_type::TEMPERATURE},
        {'w', llama_sampler_type::ADAPTIVE_P},
    };

    std::vector<llama_sampler_type> sampler_types;
    sampler_types.reserve(names_string.size());
    for (const auto & c : names_string) {
        const auto sampler_item = sampler_name_map.find(c);
        if (sampler_item != sampler_name_map.end()) {
            sampler_types.push_back(sampler_item->second);
        }
    }
    return sampler_types;
}

// no reasons to expose this function in header
static void sampler_queue(
    struct llama_context* ctx_main,
    const common_params_sampling& params,
    common_sampler * ctx_sampling,
    llama_token_data_array& cur_p,
    size_t   min_keep) {
    const float         temp = params.temp;
    const float         dynatemp_range = params.dynatemp_range;
    const float         dynatemp_exponent = params.dynatemp_exponent;
    const int32_t       top_k = params.top_k;
    const float         top_p = params.top_p;
    const float         min_p = params.min_p;
    const float         tfs_z = params.tfs_z;
    const float         typical_p = params.typical_p;
    const float         xtc_probability = params.xtc_probability;
    const float         xtc_threshold = params.xtc_threshold;
    const float         top_n_sigma = params.top_n_sigma;

    const std::vector<llama_sampler_type> & samplers_sequence = params.samplers_sequence;
    bool use_adaptive_p = false; // see below
    for (auto sampler_type : samplers_sequence) {
        switch (sampler_type) {
            case llama_sampler_type::DRY        : llama_sample_dry      (ctx_main, ctx_sampling->smpl, &cur_p); break;
            case llama_sampler_type::TOP_K      : llama_sample_top_k    (ctx_main, &cur_p, top_k,     min_keep); break;
            case llama_sampler_type::TFS_Z      : llama_sample_tail_free(ctx_main, &cur_p, tfs_z,     min_keep); break;
            case llama_sampler_type::TYPICAL_P  : llama_sample_typical  (ctx_main, &cur_p, typical_p, min_keep); break;
            case llama_sampler_type::TOP_P      : llama_sample_top_p    (ctx_main, &cur_p, top_p,     min_keep); break;
            case llama_sampler_type::MIN_P      : llama_sample_min_p    (ctx_main, &cur_p, min_p,     min_keep); break;
            case llama_sampler_type::XTC        : llama_sample_xtc      (ctx_main, &cur_p, xtc_probability, xtc_threshold, min_keep); break;
            case llama_sampler_type::TOP_N_SIGMA: llama_sample_top_n_sigma(ctx_main, &cur_p, top_n_sigma); break;
            case llama_sampler_type::DIST       : llama_sample_dist  (ctx_main, &cur_p); break;
            case llama_sampler_type::TEMPERATURE:
                if (dynatemp_range > 0) {
                    float dynatemp_min = std::max(0.0f, temp - dynatemp_range);
                    float dynatemp_max = std::max(0.0f, temp + dynatemp_range);
                    llama_sample_entropy(ctx_main, &cur_p, dynatemp_min, dynatemp_max, dynatemp_exponent);
                } else {
                    llama_sample_temp(ctx_main, &cur_p, temp);
                }
                break;
            case llama_sampler_type::ADAPTIVE_P:  use_adaptive_p = true; break;
            default : break;
        }

    }
    if (use_adaptive_p) {
        // adaptive p should be put to the last, so we ignore the order in the sampler
        llama_sample_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
    }
}

static llama_token llama_sampling_sample_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool is_resampling) {
    const common_params_sampling & params = ctx_sampling->params;

    const float   temp            = params.temp;
    const int     mirostat        = params.mirostat;
    const float   mirostat_tau    = params.mirostat_tau;
    const float   mirostat_eta    = params.mirostat_eta;
    const float   adaptive_target = params.adaptive_target;

    std::vector<float> original_logits;
    llama_sampling_prepare(ctx_sampling, ctx_main, ctx_cfg, idx, /* apply_grammar= */ is_resampling, &original_logits);
    llama_token_data_array & cur_p = ctx_sampling->cur_p;
    if (ctx_sampling->grammar != NULL && !is_resampling) {
        GGML_ASSERT(!original_logits.empty());
    }
    llama_token id = 0;
    // Sample grammar first for resampling
    if (ctx_sampling->grammar != NULL && is_resampling) {
        float* logits = llama_get_logits_ith(ctx_main, idx);
        // Apply grammar constraints to all candidates
        llama_grammar_sample(ctx_sampling->grammar, ctx_main, &cur_p);
    }

    if (temp < 0.0) {
        // greedy sampling, with probs
        llama_sample_softmax(ctx_main, &cur_p);
        id = cur_p.data[0].id;
    } else if (temp == 0.0) {
        // greedy sampling, no probs
        id = llama_sample_token_greedy(ctx_main, &cur_p);
    } else {
        if (mirostat == 1) {
            const int mirostat_m = 100;
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat(ctx_main, &cur_p, mirostat_tau, mirostat_eta, mirostat_m, &ctx_sampling->mirostat_mu);
        } else if (mirostat == 2) {
            llama_sample_temp(ctx_main, &cur_p, temp);
            id = llama_sample_token_mirostat_v2(ctx_main, &cur_p, mirostat_tau, mirostat_eta, &ctx_sampling->mirostat_mu);
        } else if (adaptive_target >= 0.0f && ctx_sampling->adapt_p_ctx!=nullptr) {
            // adaptive p sampling
            llama_prep_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
            sampler_queue(ctx_main, params, ctx_sampling, cur_p, std::max(1, params.min_keep));
            id = llama_sample_token_adaptive_p(ctx_main, &cur_p, ctx_sampling->adapt_p_ctx);
        } else {
            // temperature sampling
            size_t min_keep = std::max(1, params.min_keep);

            sampler_queue(ctx_main, params,ctx_sampling, cur_p, min_keep);           
            id = llama_sample_token_with_rng(ctx_main, &cur_p, ctx_sampling->rng);

        }
    }

    if (ctx_sampling->grammar != NULL && !is_resampling) {
        // Get a pointer to the logits
        float * logits = llama_get_logits_ith(ctx_main, idx);

        // Create an array with a single token data element for the sampled id
        llama_token_data single_token_data = {id, logits[id], 0.0f};
        llama_token_data_array single_token_data_array = { &single_token_data, 1, false };

        // Apply grammar constraints to the single token
        llama_grammar_sample(ctx_sampling->grammar, ctx_main, &single_token_data_array);

        // Check if the token is valid according to the grammar by seeing if its logit has been set to -INFINITY
        bool is_valid = single_token_data_array.data[0].logit != -INFINITY;

        // If the token is not valid according to the grammar, perform resampling
        if (!is_valid) {
            LOG("Resampling because token %d: '%s' does not meet grammar rules\n", id, common_token_to_piece(ctx_main, id).c_str());

            // Restore logits from the copy
            std::copy(original_logits.begin(), original_logits.end(), logits);

            return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ true);
        }
    }
    ctx_sampling->n_valid = temp == 0.0f ? 0 : cur_p.size;

    return id;
}

static llama_token_data_array llama_sampling_prepare_impl(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    const common_params_sampling & params = ctx_sampling->params;

    const int n_vocab = llama_n_vocab(llama_get_model(ctx_main));

    const int32_t penalty_last_n  = params.penalty_last_n < 0 ? params.n_prev : params.penalty_last_n;
    const float   penalty_repeat  = params.penalty_repeat;
    const float   penalty_freq    = params.penalty_freq;
    const float   penalty_present = params.penalty_present;

    const bool    penalize_nl     = params.penalize_nl;

    auto & prev = ctx_sampling->prev;
    auto & cur  = ctx_sampling->cur;

    // Get a pointer to the logits
    float * logits = llama_get_logits_ith(ctx_main, idx);

    if (ctx_sampling->grammar != NULL && !apply_grammar) {
        GGML_ASSERT(original_logits != NULL);
        // Only make a copy of the original logits if we are not applying grammar checks, not sure if I actually have to do this.
        *original_logits = {logits, logits + n_vocab};
    }

    // apply params.logit_bias map
    for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        logits[it->first] += it->second;
    }

    if (ctx_cfg) {
        float * logits_guidance = llama_get_logits_ith(ctx_cfg, idx);
        llama_sample_apply_guidance(ctx_main, logits, logits_guidance, params.cfg_scale);
    }

    cur.resize(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
    }

    ctx_sampling->cur_p = { cur.data(), cur.size(), false };

    llama_token_data_array & cur_p = ctx_sampling->cur_p;

    // apply penalties
    const auto& penalty_tokens = params.use_penalty_prompt_tokens ? params.penalty_prompt_tokens : prev;
    const int penalty_tokens_used_size = std::min((int)penalty_tokens.size(), penalty_last_n);
    if (penalty_tokens_used_size) {
        const float nl_logit = logits[llama_token_nl(llama_get_model(ctx_main))];

        llama_sample_repetition_penalties(ctx_main, &cur_p,
                penalty_tokens.data() + penalty_tokens.size() - penalty_tokens_used_size,
                penalty_tokens_used_size, penalty_repeat, penalty_freq, penalty_present);

        if (!penalize_nl) {
            for (size_t idx = 0; idx < cur_p.size; idx++) {
                if (cur_p.data[idx].id == llama_token_nl(llama_get_model(ctx_main))) {
                    cur_p.data[idx].logit = nl_logit;
                    break;
                }
            }
        }
    }

    // apply grammar checks before sampling logic
    if (apply_grammar && ctx_sampling->grammar != NULL) {
        llama_grammar_sample(ctx_sampling->grammar, ctx_main, &cur_p);
    }

    return cur_p;
}

llama_token common_sampler_sample_legacy(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx) {
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, ctx_cfg, idx, /* is_resampling= */ false);
}

llama_token common_sampler_sample(
    struct common_sampler * ctx_sampling,
    struct llama_context * ctx_main,
    const int idx,
    bool grammar_first) {
    // Call the implementation function with is_resampling set to false by default
    return llama_sampling_sample_impl(ctx_sampling, ctx_main, nullptr, idx, /* is_resampling= */ grammar_first);
}

llama_token_data_array llama_sampling_prepare(
                  struct common_sampler * ctx_sampling,
                  struct llama_context * ctx_main,
                  struct llama_context * ctx_cfg,
                  const int idx,
                  bool apply_grammar,
                  std::vector<float> * original_logits) {
    return llama_sampling_prepare_impl(ctx_sampling,ctx_main, ctx_cfg, idx, apply_grammar, original_logits);
}

void common_sampler_accept(
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar) {
    if (ctx_sampling->prev.size() > 0) {
    ctx_sampling->prev.erase(ctx_sampling->prev.begin());

    }
    ctx_sampling->prev.push_back(id);

    if (ctx_sampling->grammar != NULL && apply_grammar) {
        llama_grammar_accept_token(ctx_sampling->grammar, ctx_main, id);
    }
    if (ctx_sampling->smpl) {
        llama_sampler_dry_accept(ctx_sampling->smpl, id);
    }
}

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
            });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

std::vector<llama_token> llama_sampling_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<llama_token> & draft) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft);
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const std::vector<llama_token> & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, ctx, id, true);

        result.push_back(id);
    }

    return result;
}





template <>
json common_grammar_trigger::to_json() const {
    json out{
        {"type", (int)type},
        {"value", value},
    };
    if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out["token"] = (int)token;
    }
    return out;
}

template <>
common_grammar_trigger common_grammar_trigger::from_json(const json& in) {
    common_grammar_trigger out;
    out.type = (common_grammar_trigger_type)in.at("type").get<int>();
    out.value = in.at("value").get<std::string>();
    if (out.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out.token = (llama_token)in.at("token").get<int>();
    }
    return out;
}
