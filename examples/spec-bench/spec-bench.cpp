#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct spec_bench_options {
    std::string dataset_path;
    std::string output_path;
    std::vector<std::string> task_names;
    int repeat = 1;
    int retry = 0;
};

struct spec_bench_git_info {
    std::string branch = "unknown";
    std::string commit = LLAMA_COMMIT;
};

struct spec_bench_task {
    std::string id;
    std::string name;
    std::string category;
    std::string prompt;
    int max_tokens = -1;
    bool builtin = false;
};

struct spec_bench_stage_delta {
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;
    uint64_t num_drafts = 0;
    uint64_t accepted_drafts = 0;
    uint64_t draft_tokens = 0;
    uint64_t accepted_tokens = 0;
    int64_t t_begin_us = 0;
    int64_t t_draft_us = 0;
    int64_t t_accept_us = 0;
};

struct spec_bench_metrics_delta {
    std::vector<spec_bench_stage_delta> stages;
    uint64_t num_drafts = 0;
    uint64_t accepted_drafts = 0;
    uint64_t draft_tokens = 0;
    uint64_t accepted_tokens = 0;
    int64_t t_begin_us = 0;
    int64_t t_draft_us = 0;
    int64_t t_accept_us = 0;
};

struct spec_bench_attempt_result {
    bool ok = false;
    bool hit_eog = false;
    std::string error;
    std::string output_text;
    llama_tokens output_tokens;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int retries_used = 0;
    double prompt_s = 0.0;
    double decode_s = 0.0;
    double total_s = 0.0;
    spec_bench_metrics_delta spec_delta;
};

struct spec_bench_summary {
    int attempts = 0;
    int successes = 0;
    int failures = 0;
    int prompt_tokens = 0;
    int generated_tokens = 0;
    int retries_used = 0;
    double prompt_s = 0.0;
    double decode_s = 0.0;
    double total_s = 0.0;
    spec_bench_metrics_delta spec_delta;
};

static std::string spec_bench_trim_copy(std::string value) {
    return string_strip(value);
}

static std::string spec_bench_run_command_capture(const char * command) {
#if defined(_WIN32)
    FILE * pipe = _popen(command, "r");
#else
    FILE * pipe = popen(command, "r");
#endif
    if (pipe == nullptr) {
        return "";
    }

    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }

#if defined(_WIN32)
    const int rc = _pclose(pipe);
#else
    const int rc = pclose(pipe);
#endif
    if (rc != 0) {
        return "";
    }

    return spec_bench_trim_copy(output);
}

static spec_bench_git_info spec_bench_get_git_info() {
    spec_bench_git_info info;

    const std::string branch = spec_bench_run_command_capture("git rev-parse --abbrev-ref HEAD");
    if (!branch.empty()) {
        info.branch = branch;
    }

    const std::string commit = spec_bench_run_command_capture("git rev-parse HEAD");
    if (!commit.empty()) {
        info.commit = commit;
    }

    return info;
}

static std::vector<spec_bench_task> spec_bench_builtin_tasks() {
    return {
        {
            /* .id = */ "builtin-code",
            /* .name = */ "code",
            /* .category = */ "code",
            /* .prompt = */ "Write a compact C++ function that returns the Fibonacci sequence up to n as a vector. Add a short explanation after the code.",
            /* .max_tokens = */ 192,
            /* .builtin = */ true,
        },
        {
            /* .id = */ "builtin-extract",
            /* .name = */ "extract",
            /* .category = */ "extraction",
            /* .prompt = */ "Extract the fields name, company, city, and order_id from this text and answer as plain JSON only: Maria Silva from Orbit Labs in Recife confirmed order ZX-4912 after a phone call.",
            /* .max_tokens = */ 96,
            /* .builtin = */ true,
        },
        {
            /* .id = */ "builtin-story",
            /* .name = */ "story",
            /* .category = */ "creative",
            /* .prompt = */ "Write a vivid short story in three paragraphs about a maintenance robot repairing a weather station during a dust storm on Mars.",
            /* .max_tokens = */ 256,
            /* .builtin = */ true,
        },
    };
}

static void spec_bench_print_usage(const char * argv0) {
    LOG_TEE("usage: %s [benchmark options] [normal llama args]\n", argv0);
    LOG_TEE("\n");
    LOG_TEE("benchmark options:\n");
    LOG_TEE("  --dataset PATH        optional JSONL dataset override\n");
    LOG_TEE("  --task LIST           built-in tasks to run, e.g. code,extract,story\n");
    LOG_TEE("  --repeat N            repeat each task N times (default: 1)\n");
    LOG_TEE("  --retry N             retry each failed task up to N times (default: 0)\n");
    LOG_TEE("  --output PATH         write JSONL output to a file\n");
    LOG_TEE("\n");
}

static bool spec_bench_parse_args(
        int argc,
        char ** argv,
        spec_bench_options & opts,
        std::vector<std::string> & passthrough) {
    passthrough.clear();
    passthrough.reserve(argc);
    passthrough.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                LOG_TEE("%s: missing value after %s\n", __func__, name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--dataset") {
            const char * value = require_value("--dataset");
            if (!value) {
                return false;
            }
            opts.dataset_path = value;
            continue;
        }
        if (arg == "--task") {
            const char * value = require_value("--task");
            if (!value) {
                return false;
            }
            for (const auto & name : string_split(std::string(value), ',')) {
                const std::string trimmed = string_strip(name);
                if (!trimmed.empty()) {
                    opts.task_names.push_back(trimmed);
                }
            }
            continue;
        }
        if (arg == "--repeat") {
            const char * value = require_value("--repeat");
            if (!value) {
                return false;
            }
            opts.repeat = std::max(1, std::stoi(value));
            continue;
        }
        if (arg == "--retry") {
            const char * value = require_value("--retry");
            if (!value) {
                return false;
            }
            opts.retry = std::max(0, std::stoi(value));
            continue;
        }
        if (arg == "--output") {
            const char * value = require_value("--output");
            if (!value) {
                return false;
            }
            opts.output_path = value;
            continue;
        }

        passthrough.push_back(arg);
    }

    return true;
}

static std::vector<char *> spec_bench_make_argv(std::vector<std::string> & args) {
    std::vector<char *> out;
    out.reserve(args.size());
    for (std::string & arg : args) {
        out.push_back(arg.data());
    }
    return out;
}

static std::vector<spec_bench_task> spec_bench_load_dataset(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open dataset: " + path);
    }

    std::vector<spec_bench_task> tasks;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (string_strip(line).empty()) {
            continue;
        }

        const json row = json::parse(line);
        if (!row.contains("prompt") && !row.contains("input")) {
            throw std::runtime_error("dataset line " + std::to_string(line_no) + " must contain prompt or input");
        }

        spec_bench_task task;
        task.id = row.value("id", "dataset-" + std::to_string(line_no));
        task.name = row.value("name", row.value("task", task.id));
        task.category = row.value("category", "dataset");
        task.prompt = row.contains("prompt") ? row.at("prompt").get<std::string>() : row.at("input").get<std::string>();
        task.max_tokens = row.value("max_tokens", -1);
        task.builtin = false;
        tasks.push_back(std::move(task));
    }

    return tasks;
}

static std::vector<spec_bench_task> spec_bench_select_tasks(const spec_bench_options & opts) {
    if (!opts.dataset_path.empty()) {
        return spec_bench_load_dataset(opts.dataset_path);
    }

    std::vector<spec_bench_task> builtin = spec_bench_builtin_tasks();
    if (opts.task_names.empty()) {
        return builtin;
    }

    std::set<std::string> wanted;
    for (const auto & name : opts.task_names) {
        wanted.insert(string_lower(name));
    }

    std::vector<spec_bench_task> selected;
    std::set<std::string> matched;
    for (const auto & task : builtin) {
        const std::string normalized = string_lower(task.name);
        if (wanted.count(normalized) > 0) {
            selected.push_back(task);
            matched.insert(normalized);
        }
    }

    if (matched.size() != wanted.size()) {
        std::vector<std::string> unknown;
        for (const auto & name : wanted) {
            if (matched.count(name) == 0) {
                unknown.push_back(name);
            }
        }
        throw std::runtime_error("unknown built-in task name(s): " + string_join(unknown, ", "));
    }

    return selected;
}

static spec_bench_metrics_delta spec_bench_snapshot_delta(
        const common_speculative_metrics_snapshot & before,
        const common_speculative_metrics_snapshot & after) {
    spec_bench_metrics_delta delta;
    const size_t n = std::min(before.stages.size(), after.stages.size());
    delta.stages.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const auto & lhs = before.stages[i];
        const auto & rhs = after.stages[i];
        spec_bench_stage_delta stage;
        stage.type = rhs.type;
        stage.num_drafts = rhs.n_gen_drafts - lhs.n_gen_drafts;
        stage.accepted_drafts = rhs.n_acc_drafts - lhs.n_acc_drafts;
        stage.draft_tokens = rhs.n_gen_tokens - lhs.n_gen_tokens;
        stage.accepted_tokens = rhs.n_acc_tokens - lhs.n_acc_tokens;
        stage.t_begin_us = rhs.t_begin_us - lhs.t_begin_us;
        stage.t_draft_us = rhs.t_draft_us - lhs.t_draft_us;
        stage.t_accept_us = rhs.t_accept_us - lhs.t_accept_us;

        delta.num_drafts += stage.num_drafts;
        delta.accepted_drafts += stage.accepted_drafts;
        delta.draft_tokens += stage.draft_tokens;
        delta.accepted_tokens += stage.accepted_tokens;
        delta.t_begin_us += stage.t_begin_us;
        delta.t_draft_us += stage.t_draft_us;
        delta.t_accept_us += stage.t_accept_us;
        delta.stages.push_back(stage);
    }

    return delta;
}

static void spec_bench_accumulate(spec_bench_summary & summary, const spec_bench_attempt_result & result) {
    summary.attempts++;
    summary.successes += result.ok ? 1 : 0;
    summary.failures += result.ok ? 0 : 1;
    summary.prompt_tokens += result.prompt_tokens;
    summary.generated_tokens += result.generated_tokens;
    summary.retries_used += result.retries_used;
    summary.prompt_s += result.prompt_s;
    summary.decode_s += result.decode_s;
    summary.total_s += result.total_s;
    summary.spec_delta.num_drafts += result.spec_delta.num_drafts;
    summary.spec_delta.accepted_drafts += result.spec_delta.accepted_drafts;
    summary.spec_delta.draft_tokens += result.spec_delta.draft_tokens;
    summary.spec_delta.accepted_tokens += result.spec_delta.accepted_tokens;
    summary.spec_delta.t_begin_us += result.spec_delta.t_begin_us;
    summary.spec_delta.t_draft_us += result.spec_delta.t_draft_us;
    summary.spec_delta.t_accept_us += result.spec_delta.t_accept_us;

    if (summary.spec_delta.stages.size() < result.spec_delta.stages.size()) {
        summary.spec_delta.stages.resize(result.spec_delta.stages.size());
    }
    for (size_t i = 0; i < result.spec_delta.stages.size(); ++i) {
        auto & dst = summary.spec_delta.stages[i];
        const auto & src = result.spec_delta.stages[i];
        dst.type = src.type;
        dst.num_drafts += src.num_drafts;
        dst.accepted_drafts += src.accepted_drafts;
        dst.draft_tokens += src.draft_tokens;
        dst.accepted_tokens += src.accepted_tokens;
        dst.t_begin_us += src.t_begin_us;
        dst.t_draft_us += src.t_draft_us;
        dst.t_accept_us += src.t_accept_us;
    }
}

static json spec_bench_stage_json(const spec_bench_stage_delta & stage) {
    const double acceptance_rate = stage.draft_tokens > 0
        ? (double) stage.accepted_tokens / (double) stage.draft_tokens
        : 0.0;
    const double acceptance_length = stage.num_drafts > 0
        ? 1.0 + (double) stage.accepted_tokens / (double) stage.num_drafts
        : 0.0;

    return json{
        {"type", common_speculative_type_to_str(stage.type)},
        {"num_drafts", stage.num_drafts},
        {"accepted_drafts", stage.accepted_drafts},
        {"draft_tokens", stage.draft_tokens},
        {"accepted_tokens", stage.accepted_tokens},
        {"acceptance_rate", acceptance_rate},
        {"acceptance_length", acceptance_length},
        {"t_begin_s", stage.t_begin_us / 1e6},
        {"t_draft_s", stage.t_draft_us / 1e6},
        {"t_accept_s", stage.t_accept_us / 1e6},
    };
}

static json spec_bench_metrics_json(const spec_bench_metrics_delta & delta) {
    const double acceptance_rate = delta.draft_tokens > 0
        ? (double) delta.accepted_tokens / (double) delta.draft_tokens
        : 0.0;
    const double acceptance_length = delta.num_drafts > 0
        ? 1.0 + (double) delta.accepted_tokens / (double) delta.num_drafts
        : 0.0;

    json stages = json::array();
    for (const auto & stage : delta.stages) {
        stages.push_back(spec_bench_stage_json(stage));
    }

    return json{
        {"num_drafts", delta.num_drafts},
        {"accepted_drafts", delta.accepted_drafts},
        {"draft_tokens", delta.draft_tokens},
        {"accepted_tokens", delta.accepted_tokens},
        {"acceptance_rate", acceptance_rate},
        {"acceptance_length", acceptance_length},
        {"draft_tokens_per_step", delta.num_drafts > 0 ? (double) delta.draft_tokens / (double) delta.num_drafts : 0.0},
        {"t_begin_s", delta.t_begin_us / 1e6},
        {"t_draft_s", delta.t_draft_us / 1e6},
        {"t_accept_s", delta.t_accept_us / 1e6},
        {"stages", stages},
    };
}

static json spec_bench_stage_types_json(const common_params_speculative & params) {
    json stages = json::array();
    for (const auto & stage : params.get_resolved_stages()) {
        stages.push_back(common_speculative_type_to_str(stage.type));
    }
    return stages;
}

static json spec_bench_task_names_json(const std::vector<spec_bench_task> & tasks) {
    json names = json::array();
    for (const auto & task : tasks) {
        names.push_back(task.name);
    }
    return names;
}

static json spec_bench_sampler_json(const gpt_params & params) {
    return json{
        {"seed", params.seed},
        {"temp", params.sparams.temp},
        {"top_k", params.sparams.top_k},
        {"top_p", params.sparams.top_p},
        {"min_p", params.sparams.min_p},
        {"tfs_z", params.sparams.tfs_z},
        {"typical_p", params.sparams.typical_p},
        {"top_n_sigma", params.sparams.top_n_sigma},
        {"penalty_last_n", params.sparams.penalty_last_n},
        {"penalty_repeat", params.sparams.penalty_repeat},
        {"penalty_freq", params.sparams.penalty_freq},
        {"penalty_present", params.sparams.penalty_present},
        {"mirostat", params.sparams.mirostat},
        {"mirostat_tau", params.sparams.mirostat_tau},
        {"mirostat_eta", params.sparams.mirostat_eta},
        {"n_probs", params.sparams.n_probs},
        {"samplers_sequence", json(params.sparams.samplers_sequence)},
    };
}

static json spec_bench_runtime_json(const gpt_params & params) {
    return json{
        {"model", params.model},
        {"model_alias", params.model_alias},
        {"n_ctx", params.n_ctx},
        {"n_predict", params.n_predict},
        {"n_batch", params.n_batch},
        {"n_ubatch", params.n_ubatch},
        {"n_threads", params.n_threads},
        {"n_threads_batch", params.n_threads_batch},
        {"n_gpu_layers", params.n_gpu_layers},
        {"flash_attn", params.flash_attn},
        {"numa", (int) params.numa},
    };
}

static std::string spec_bench_decode_tokens(
        const llama_context * ctx,
        const llama_tokens & tokens,
        bool special) {
    std::string text;
    for (llama_token token : tokens) {
        text += common_token_to_piece(ctx, token, special);
    }
    return text;
}

static spec_bench_attempt_result spec_bench_run_attempt(
        const spec_bench_task & task,
        const gpt_params & params,
        llama_model * model,
        llama_context * ctx,
        common_speculative * spec,
        common_sampler * sampler) {
    spec_bench_attempt_result result;

    if (llama_model_has_encoder(model)) {
        result.error = "encoder-decoder models are not supported";
        return result;
    }

    const int task_max_tokens = task.max_tokens > 0
        ? task.max_tokens
        : (params.n_predict >= 0 ? params.n_predict : 256);
    if (task_max_tokens <= 0) {
        result.error = "max token budget resolved to zero";
        return result;
    }

    llama_tokens prompt_tokens = common_tokenize(ctx, task.prompt, true, true);
    result.prompt_tokens = (int) prompt_tokens.size();

    const int n_ctx = llama_n_ctx(ctx);
    if (result.prompt_tokens >= n_ctx - 2) {
        result.error = "prompt does not fit into context";
        return result;
    }

    common_sampler_reset(sampler);
    if (spec != nullptr) {
        common_speculative_clear_sequence_kv(spec, ctx, 0);
    } else {
        llama_kv_cache_clear(ctx);
    }
    llama_reset_timings(ctx);

    llama_tokens embd = prompt_tokens;
    llama_tokens speculative_tokens = prompt_tokens;
    int n_past = 0;
    int n_remain = task_max_tokens;
    bool embd_is_prompt = true;

    const auto spec_before = common_speculative_get_metrics_snapshot(spec);
    if (spec != nullptr) {
        common_speculative_begin(spec, prompt_tokens);
    }
    for (llama_token token : prompt_tokens) {
        common_sampler_accept(sampler, ctx, token, false);
    }

    const int64_t t_prompt_start_us = ggml_time_us();

    while (!embd.empty()) {
        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = std::min(params.n_batch, (int) embd.size() - i);
            llama_batch batch = llama_batch_get_one(embd.data() + i, n_eval, n_past, 0);
            if (llama_decode(ctx, batch) != 0) {
                result.error = "prompt decode failed";
                return result;
            }
            if (spec != nullptr && embd_is_prompt) {
                if (common_speculative_on_target_seq_batch(spec, ctx, batch, 0, true) != 0) {
                    result.error = "speculative prompt warmup failed";
                    return result;
                }
            }
            n_past += n_eval;
        }
        embd.clear();
    }

    const int64_t t_prompt_end_us = ggml_time_us();
    const int64_t t_decode_start_us = t_prompt_end_us;

    while (n_remain > 0) {
        llama_tokens next_embd;
        bool used_speculative = false;
        bool have_fallback_sampled = false;
        llama_token fallback_sampled = LLAMA_TOKEN_NULL;

        if (spec != nullptr && n_remain != 1) {
            common_params_speculative speculative_params = params.speculative;
            const llama_token sampled_before = common_sampler_sample_legacy(sampler, ctx, nullptr);
            have_fallback_sampled = true;
            fallback_sampled = sampled_before;
            common_sampler_accept(sampler, ctx, sampled_before, true);

            auto draft_result = common_speculative_draft_ex(
                spec,
                ctx,
                speculative_params,
                speculative_tokens,
                sampled_before,
                n_past,
                0);

            auto & draft = draft_result.tokens;
            int max_usable_draft = (int) draft.size();
            max_usable_draft = std::min(max_usable_draft, std::max(0, n_remain - 2));
            max_usable_draft = std::min(max_usable_draft, std::max(0, n_ctx - n_past - 2));
            max_usable_draft = std::min(max_usable_draft, std::max(0, (int) llama_n_batch(ctx) - 1));
            if ((int) draft.size() > max_usable_draft) {
                draft.resize(max_usable_draft);
            }

            const int min_usable_draft = params.speculative.get_min_usable_stage_n_min();
            if ((int) draft.size() >= min_usable_draft && !draft.empty()) {
                if (llama_model_has_recurrent(model)) {
                    if (!common_speculative_before_draft(
                            spec,
                            model,
                            ctx,
                            sampler,
                            params.sparams,
                            0,
                            n_past,
                            sampled_before,
                            (int) draft.size() + 1,
                            params.speculative.recurrent_ckpt_mode)) {
                        draft.clear();
                    }
                }

                if (!draft.empty()) {
                    llama_batch verify_batch = llama_batch_init((int) draft.size() + 1, 0, 1);
                    std::vector<int> verify_indices;
                    verify_indices.reserve(draft.size() + 1);

                    common_batch_add(verify_batch, sampled_before, n_past, {0}, true);
                    verify_indices.push_back(0);
                    for (size_t i = 0; i < draft.size(); ++i) {
                        common_batch_add(verify_batch, draft[i], n_past + 1 + (llama_pos) i, {0}, true);
                        verify_indices.push_back((int) i + 1);
                    }

                    if (llama_decode(ctx, verify_batch) != 0) {
                        llama_batch_free(verify_batch);
                        result.error = "speculative verify decode failed";
                        return result;
                    }

                    llama_tokens ids;
                    try {
                        ids = common_sampler_sample_and_accept_n(sampler, ctx, verify_indices, draft);
                    } catch (const std::exception & e) {
                        llama_batch_free(verify_batch);
                        result.error = e.what();
                        return result;
                    }

                    std::vector<int32_t> accepted_output_indices;
                    if (!ids.empty()) {
                        accepted_output_indices.assign(verify_indices.begin(), verify_indices.begin() + ids.size());
                    }

                    common_speculative_commit(
                        spec,
                        ctx,
                        sampler,
                        0,
                        sampled_before,
                        ids,
                        (int) draft.size(),
                        n_past + 1,
                        accepted_output_indices);

                    llama_batch_free(verify_batch);

                    if (!ids.empty()) {
                        result.output_tokens.push_back(sampled_before);
                        result.output_tokens.insert(result.output_tokens.end(), ids.begin(), ids.end());
                        next_embd.push_back(ids.back());
                        speculative_tokens.push_back(sampled_before);
                        if (ids.size() > 1) {
                            speculative_tokens.insert(speculative_tokens.end(), ids.begin(), ids.end() - 1);
                        }
                        n_past += (int) ids.size();
                        n_remain -= (int) (ids.size() + 1);
                        used_speculative = true;
                    }
                }
            }
        }

        if (!used_speculative) {
            const llama_token id = have_fallback_sampled
                ? fallback_sampled
                : common_sampler_sample_legacy(sampler, ctx, nullptr);
            if (!have_fallback_sampled) {
                common_sampler_accept(sampler, ctx, id, true);
            }

            result.output_tokens.push_back(id);
            next_embd.push_back(id);
            n_remain -= 1;
        }

        result.generated_tokens = (int) result.output_tokens.size();
        if (!result.output_tokens.empty() && llama_token_is_eog(model, result.output_tokens.back())) {
            result.hit_eog = true;
            break;
        }

        embd = std::move(next_embd);
        embd_is_prompt = false;
        if (embd.empty()) {
            break;
        }

        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = std::min(params.n_batch, (int) embd.size() - i);
            llama_batch batch = llama_batch_get_one(embd.data() + i, n_eval, n_past, 0);
            if (llama_decode(ctx, batch) != 0) {
                result.error = "decode failed";
                return result;
            }
            if (spec != nullptr) {
                speculative_tokens.insert(speculative_tokens.end(), embd.begin() + i, embd.begin() + i + n_eval);
            }
            n_past += n_eval;
        }

        embd.clear();

        if (n_past >= n_ctx - 2) {
            break;
        }
    }

    const int64_t t_decode_end_us = ggml_time_us();
    const auto spec_after = common_speculative_get_metrics_snapshot(spec);

    result.prompt_s = (t_prompt_end_us - t_prompt_start_us) / 1e6;
    result.decode_s = (t_decode_end_us - t_decode_start_us) / 1e6;
    result.total_s = (t_decode_end_us - t_prompt_start_us) / 1e6;
    result.generated_tokens = (int) result.output_tokens.size();
    result.output_text = spec_bench_decode_tokens(ctx, result.output_tokens, params.special);
    result.spec_delta = spec_bench_snapshot_delta(spec_before, spec_after);
    result.ok = true;

    return result;
}

static json spec_bench_attempt_json(
        const spec_bench_git_info & git_info,
        const gpt_params & params,
        const spec_bench_options & opts,
        const spec_bench_task & task,
        const spec_bench_attempt_result & result,
        int repeat_index) {
    const bool is_baseline = !params.speculative.has_stage_chain();
    const double decode_tps = result.decode_s > 0.0 ? result.generated_tokens / result.decode_s : 0.0;
    const double total_tps = result.total_s > 0.0 ? result.generated_tokens / result.total_s : 0.0;

    return json{
        {"row_type", "attempt"},
        {"task_id", task.id},
        {"task_name", task.name},
        {"task_category", task.category},
        {"repeat_index", repeat_index},
        {"builtin", task.builtin},
        {"git", {
            {"branch", git_info.branch},
            {"commit", git_info.commit},
            {"build_commit", LLAMA_COMMIT},
        }},
        {"dataset", opts.dataset_path.empty() ? "builtin-default" : opts.dataset_path},
        {"runtime", spec_bench_runtime_json(params)},
        {"variant", {
            {"is_baseline", is_baseline},
            {"spec_types", spec_bench_stage_types_json(params.speculative)},
            {"stage_chain", common_speculative_stage_chain_to_str(params.speculative)}
        }},
        {"sampler", spec_bench_sampler_json(params)},
        {"timing", {
            {"prompt_s", result.prompt_s},
            {"decode_s", result.decode_s},
            {"total_s", result.total_s},
            {"decode_tps", decode_tps},
            {"overall_tps", total_tps},
        }},
        {"tokens", {
            {"prompt", result.prompt_tokens},
            {"generated", result.generated_tokens},
        }},
        {"speculative", spec_bench_metrics_json(result.spec_delta)},
        {"quality", {
            {"ok", result.ok},
            {"error", result.error.empty() ? json(nullptr) : json(result.error)},
            {"retries_used", result.retries_used},
            {"hit_eog", result.hit_eog},
        }},
        {"prompt", task.prompt},
        {"output", result.output_text},
    };
}

static json spec_bench_summary_json(
        const spec_bench_git_info & git_info,
        const gpt_params & params,
        const spec_bench_options & opts,
        const std::vector<spec_bench_task> & tasks,
        const spec_bench_summary & summary) {
    const bool is_baseline = !params.speculative.has_stage_chain();
    const double decode_tps = summary.decode_s > 0.0 ? summary.generated_tokens / summary.decode_s : 0.0;
    const double total_tps = summary.total_s > 0.0 ? summary.generated_tokens / summary.total_s : 0.0;

    return json{
        {"row_type", "summary"},
        {"git", {
            {"branch", git_info.branch},
            {"commit", git_info.commit},
            {"build_commit", LLAMA_COMMIT},
        }},
        {"dataset", opts.dataset_path.empty() ? "builtin-default" : opts.dataset_path},
        {"requested_tasks", json(opts.task_names)},
        {"selected_tasks", spec_bench_task_names_json(tasks)},
        {"repeat", opts.repeat},
        {"retry", opts.retry},
        {"runtime", spec_bench_runtime_json(params)},
        {"variant", {
            {"is_baseline", is_baseline},
            {"spec_types", spec_bench_stage_types_json(params.speculative)},
            {"stage_chain", common_speculative_stage_chain_to_str(params.speculative)}
        }},
        {"sampler", spec_bench_sampler_json(params)},
        {"attempts", summary.attempts},
        {"successes", summary.successes},
        {"failures", summary.failures},
        {"retries_used", summary.retries_used},
        {"timing", {
            {"prompt_s", summary.prompt_s},
            {"decode_s", summary.decode_s},
            {"total_s", summary.total_s},
            {"decode_tps", decode_tps},
            {"overall_tps", total_tps},
        }},
        {"tokens", {
            {"prompt", summary.prompt_tokens},
            {"generated", summary.generated_tokens},
        }},
        {"speculative", spec_bench_metrics_json(summary.spec_delta)},
    };
}

static bool spec_bench_prepare_spec(
        gpt_params & params,
        llama_model * model,
        llama_context * ctx,
        common_speculative ** out_spec) {
    const bool requested_spec_user = params.speculative.has_stage_chain();
    if (!common_speculative_finalize_startup(params, model)) {
        return false;
    }

    const bool requested_spec = params.speculative.has_stage_chain();
    if (requested_spec_user && !requested_spec) {
        LOG_TEE("%s: speculative decoding was requested but is not runnable with this finalized configuration\n", __func__);
        return false;
    }

    if (!requested_spec) {
        *out_spec = nullptr;
        return true;
    }

    if (params.sparams.cfg_scale > 1.f || params.grp_attn_n != 1 || llama_model_has_encoder(model)) {
        LOG_TEE("%s: this benchmark only supports direct non-CFG decoder-only speculative runs\n", __func__);
        return false;
    }

    if (!common_speculative_is_compat(ctx)) {
        LOG_TEE("%s: speculative decoding is not supported by this context\n", __func__);
        return false;
    }

    switch (common_speculative_try_init(params.speculative, ctx, out_spec)) {
        case COMMON_SPECULATIVE_INIT_READY:
            return true;
        case COMMON_SPECULATIVE_INIT_SKIPPED:
            *out_spec = nullptr;
            return true;
        case COMMON_SPECULATIVE_INIT_ERR_RECURRENT:
            LOG_TEE("%s: recurrent speculative context initialization failure\n", __func__);
            return false;
        case COMMON_SPECULATIVE_INIT_ERR_MTP:
            LOG_TEE("%s: MTP speculative context initialization failure\n", __func__);
            return false;
        case COMMON_SPECULATIVE_INIT_ERR_GENERIC:
            LOG_TEE("%s: speculative context initialization failure\n", __func__);
            return false;
    }

    return false;
}

int main(int argc, char ** argv) {
    spec_bench_options bench_opts;
    std::vector<std::string> passthrough;
    if (!spec_bench_parse_args(argc, argv, bench_opts, passthrough)) {
        spec_bench_print_usage(argv[0]);
        return 1;
    }

    auto argv_storage = spec_bench_make_argv(passthrough);

    gpt_params params;
    if (!gpt_params_parse((int) argv_storage.size(), argv_storage.data(), params)) {
        spec_bench_print_usage(argv[0]);
        gpt_params_print_usage((int) argv_storage.size(), argv_storage.data(), params);
        return 1;
    }

    common_speculative_prepare_startup(params);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = 1234;
    }

    std::vector<spec_bench_task> tasks;
    try {
        tasks = spec_bench_select_tasks(bench_opts);
    } catch (const std::exception & e) {
        LOG_TEE("%s\n", e.what());
        return 1;
    }
    const spec_bench_git_info git_info = spec_bench_get_git_info();

    std::ofstream out_file;
    std::ostream * out = &std::cout;
    if (!bench_opts.output_path.empty()) {
        out_file.open(bench_opts.output_path, std::ios::out | std::ios::trunc);
        if (!out_file) {
            LOG_TEE("%s: failed to open output file %s\n", __func__, bench_opts.output_path.c_str());
            return 1;
        }
        out = &out_file;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model * model = init.model;
    llama_context * ctx = init.context;
    common_speculative * spec = nullptr;
    common_sampler * sampler = nullptr;

    if (model == nullptr || ctx == nullptr) {
        LOG_TEE("%s: failed to load model/context\n", __func__);
        params.speculative.clear_dft();
        llama_backend_free();
        return 1;
    }

    if (!spec_bench_prepare_spec(params, model, ctx, &spec)) {
        params.speculative.clear_dft();
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    sampler = common_sampler_init(model, params.sparams);
    if (sampler == nullptr) {
        LOG_TEE("%s: failed to initialize sampler\n", __func__);
        if (spec != nullptr) {
            common_speculative_free(spec);
        }
        params.speculative.clear_dft();
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    spec_bench_summary summary;
    int task_counter = 0;
    for (const auto & task : tasks) {
        for (int repeat_index = 0; repeat_index < bench_opts.repeat; ++repeat_index) {
            spec_bench_attempt_result best_result;
            bool success = false;

            for (int attempt = 0; attempt <= bench_opts.retry; ++attempt) {
                spec_bench_attempt_result run = spec_bench_run_attempt(task, params, model, ctx, spec, sampler);
                run.retries_used = attempt;
                best_result = run;
                if (run.ok) {
                    success = true;
                    break;
                }
            }

            best_result.retries_used = success ? best_result.retries_used : bench_opts.retry;
            spec_bench_accumulate(summary, best_result);
            *out << spec_bench_attempt_json(git_info, params, bench_opts, task, best_result, repeat_index).dump() << '\n';
            ++task_counter;
        }
    }

    *out << spec_bench_summary_json(git_info, params, bench_opts, tasks, summary).dump() << '\n';
    out->flush();

    common_sampler_free(sampler);
    if (spec != nullptr) {
        common_speculative_free(spec);
    }
    params.speculative.clear_dft();
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return summary.failures == 0 ? 0 : 2;
}
