#include "common.h"
#include "chat.h"
#include "speculative.h"
#include "llama.h"
#include "spec-bench-prompts.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct spec_bench_options {
    std::string prompts_path;
    std::vector<std::string> task_names;
    std::string output_format = "md";
    bool output_details = false;
    bool task_selection_seen = false;
    bool inline_prompt_seen = false;
    bool file_prompt_seen = false;
    int repeat = 1;
    int retry = 0;
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
    std::vector<uint64_t> drafted_by_position;
    std::vector<uint64_t> accepted_by_position;
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
    std::string effective_prompt;
    spec_bench_metrics_delta spec_delta;
};

struct spec_bench_record {
    spec_bench_task task;
    spec_bench_attempt_result result;
    int repeat_index = 0;
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

static std::vector<spec_bench_task> spec_bench_builtin_tasks() {
    const std::string extract_prompt = SPEC_BENCH_PROMPT_EXTRACT;

    return {
        {
            /* .id = */ "builtin-code",
            /* .name = */ "code",
            /* .category = */ "code",
            /* .prompt = */ SPEC_BENCH_PROMPT_CODE,
            /* .max_tokens = */ -1,
            /* .builtin = */ true,
        },
        {
            /* .id = */ "builtin-extract",
            /* .name = */ "extract",
            /* .category = */ "extraction",
            /* .prompt = */ extract_prompt,
            /* .max_tokens = */ -1,
            /* .builtin = */ true,
        },
        {
            /* .id = */ "builtin-story",
            /* .name = */ "story",
            /* .category = */ "long-form-summary",
            /* .prompt = */ SPEC_BENCH_PROMPT_STORY,
            /* .max_tokens = */ -1,
            /* .builtin = */ true,
        },
    };
}

static void spec_bench_print_usage(const char * argv0) {
    LOG_TEE("usage: %s [benchmark options] [normal llama args]\n", argv0);
    LOG_TEE("\n");
    LOG_TEE("benchmark options:\n");
    LOG_TEE("  -p PROMPT / -f FILE  one custom plain-text prompt (mutually exclusive)\n");
    LOG_TEE("  --prompts PATH        optional strict JSONL prompt-file override\n");
    LOG_TEE("  --task LIST           built-in tasks to run, e.g. code,extract,story\n");
    LOG_TEE("  --repeat N            repeat each task N times (default: 1)\n");
    LOG_TEE("  --retry N             retry each failed task up to N times (default: 0)\n");
    LOG_TEE("  --output-format md|jsonl emit Markdown by default, or JSONL\n");
    LOG_TEE("  --output-details     include prompt/output and detailed metrics\n");
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

        if (arg == "--prompts") {
            const char * value = require_value("--prompts");
            if (!value) {
                return false;
            }
            if (!opts.prompts_path.empty()) {
                LOG_TEE("--prompts may be specified only once\n");
                return false;
            }
            opts.prompts_path = value;
            continue;
        }
        if (arg == "-p" || arg == "--prompt") {
            if (opts.inline_prompt_seen) {
                LOG_TEE("inline prompt may be specified only once\n");
                return false;
            }
            const char * value = require_value(arg.c_str());
            if (!value) {
                return false;
            }
            opts.inline_prompt_seen = true;
            passthrough.push_back(arg);
            passthrough.push_back(value);
            continue;
        }
        if (arg == "-f" || arg == "--file") {
            if (opts.file_prompt_seen) {
                LOG_TEE("prompt file may be specified only once\n");
                return false;
            }
            const char * value = require_value(arg.c_str());
            if (!value) {
                return false;
            }
            opts.file_prompt_seen = true;
            passthrough.push_back(arg);
            passthrough.push_back(value);
            continue;
        }
        if (arg == "--dataset") {
            LOG_TEE("--dataset is no longer supported; use --prompts PATH\n");
            return false;
        }
        if (arg == "--task") {
            const char * value = require_value("--task");
            if (!value) {
                return false;
            }
            opts.task_selection_seen = true;
            for (const auto & selection : string_split(std::string(value), ",")) {
                if (string_strip(selection).empty()) {
                    LOG_TEE("--task must not contain empty selections\n");
                    return false;
                }
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
        if (arg == "--output-format") {
            const char * value = require_value("--output-format");
            if (!value) { return false; }
            opts.output_format = string_strip(value);
            if (opts.output_format != "md" && opts.output_format != "jsonl") {
                LOG_TEE("--output-format must be md or jsonl\n");
                return false;
            }
            continue;
        }
        if (arg == "--output-details") {
            opts.output_details = true;
            continue;
        }
        if (arg == "--output") {
            LOG_TEE("--output is not a benchmark destination; use --output-format jsonl and redirect stdout\n");
            return false;
        }

        passthrough.push_back(arg);
    }

    const int single_prompt_modes = (opts.inline_prompt_seen ? 1 : 0) + (opts.file_prompt_seen ? 1 : 0);
    if (single_prompt_modes > 1) {
        LOG_TEE("choose exactly one of -p/--prompt or -f/--file\n");
        return false;
    }
    if (!opts.prompts_path.empty() && (single_prompt_modes > 0 || opts.task_selection_seen)) {
        LOG_TEE("--prompts cannot be combined with --task, -p/--prompt, or -f/--file\n");
        return false;
    }
    if (single_prompt_modes > 0 && opts.task_selection_seen) {
        LOG_TEE("-p/--prompt and -f/--file cannot be combined with --task\n");
        return false;
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
    std::set<std::string> ids;
    const std::set<std::string> allowed_fields = {"id", "name", "category", "prompt", "max_tokens"};
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (string_strip(line).empty()) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " is empty");
        }

        json row;
        try {
            row = json::parse(line);
        } catch (const std::exception & e) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " is invalid JSON: " + e.what());
        }
        if (!row.is_object()) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " must be a JSON object");
        }

        for (const auto & item : row.items()) {
            if (allowed_fields.count(item.key()) == 0) {
                throw std::runtime_error("prompt file line " + std::to_string(line_no) + " has unknown field: " + item.key());
            }
        }
        if (!row.contains("prompt") || !row.at("prompt").is_string()) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " must contain a string prompt");
        }

        spec_bench_task task;
        task.id = row.contains("id") ? row.at("id").get<std::string>() : std::to_string(line_no);
        task.name = row.contains("name") ? row.at("name").get<std::string>() : task.id;
        task.category = row.contains("category") ? row.at("category").get<std::string>() : "dataset";
        task.prompt = row.at("prompt").get<std::string>();
        if (string_strip(task.id).empty() || string_strip(task.name).empty() || string_strip(task.category).empty()) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " has an empty id, name, or category");
        }
        if (string_strip(task.prompt).empty()) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " has an empty prompt");
        }
        if (!ids.insert(task.id).second) {
            throw std::runtime_error("prompt file line " + std::to_string(line_no) + " duplicates id: " + task.id);
        }

        task.max_tokens = -1;
        if (row.contains("max_tokens")) {
            const auto & max_tokens = row.at("max_tokens");
            if (!max_tokens.is_number_integer()) {
                throw std::runtime_error("prompt file line " + std::to_string(line_no) + " max_tokens must be a positive integer");
            }
            const int64_t value = max_tokens.get<int64_t>();
            if (value <= 0 || value > std::numeric_limits<int>::max()) {
                throw std::runtime_error("prompt file line " + std::to_string(line_no) + " max_tokens must be a positive integer");
            }
            task.max_tokens = (int) value;
        }
        task.builtin = false;
        tasks.push_back(std::move(task));
    }

    if (tasks.empty()) {
        throw std::runtime_error("prompt file contains no rows: " + path);
    }

    return tasks;
}

static std::string spec_bench_prompt_file_basename(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    const std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    return name.empty() ? "prompt" : name;
}

static std::vector<spec_bench_task> spec_bench_select_tasks(const spec_bench_options & opts, const gpt_params & params) {
    if (opts.inline_prompt_seen || opts.file_prompt_seen) {
        if (string_strip(params.prompt).empty()) {
            throw std::runtime_error("custom prompt must be non-empty");
        }
        const bool from_file = opts.file_prompt_seen;
        const std::string name = from_file ? spec_bench_prompt_file_basename(params.prompt_file) : "prompt";
        return {{
            {
                "custom-" + name,
                name,
                "custom",
                params.prompt,
                -1,
                false,
            },
        }};
    }
    if (!opts.prompts_path.empty()) {
        return spec_bench_load_dataset(opts.prompts_path);
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
        const size_t n_drafted_positions = std::max(lhs.drafted_by_position.size(), rhs.drafted_by_position.size());
        const size_t n_accepted_positions = std::max(lhs.accepted_by_position.size(), rhs.accepted_by_position.size());
        stage.drafted_by_position.resize(n_drafted_positions);
        stage.accepted_by_position.resize(n_accepted_positions);
        for (size_t position = 0; position < n_drafted_positions; ++position) {
            const uint64_t before_value = position < lhs.drafted_by_position.size() ? lhs.drafted_by_position[position] : 0;
            const uint64_t after_value = position < rhs.drafted_by_position.size() ? rhs.drafted_by_position[position] : 0;
            stage.drafted_by_position[position] = after_value >= before_value ? after_value - before_value : 0;
        }
        for (size_t position = 0; position < n_accepted_positions; ++position) {
            const uint64_t before_value = position < lhs.accepted_by_position.size() ? lhs.accepted_by_position[position] : 0;
            const uint64_t after_value = position < rhs.accepted_by_position.size() ? rhs.accepted_by_position[position] : 0;
            stage.accepted_by_position[position] = after_value >= before_value ? after_value - before_value : 0;
        }
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
        if (dst.drafted_by_position.size() < src.drafted_by_position.size()) {
            dst.drafted_by_position.resize(src.drafted_by_position.size());
        }
        if (dst.accepted_by_position.size() < src.accepted_by_position.size()) {
            dst.accepted_by_position.resize(src.accepted_by_position.size());
        }
        for (size_t position = 0; position < src.drafted_by_position.size(); ++position) {
            dst.drafted_by_position[position] += src.drafted_by_position[position];
        }
        for (size_t position = 0; position < src.accepted_by_position.size(); ++position) {
            dst.accepted_by_position[position] += src.accepted_by_position[position];
        }
        dst.t_begin_us += src.t_begin_us;
        dst.t_draft_us += src.t_draft_us;
        dst.t_accept_us += src.t_accept_us;
    }
}

static double spec_bench_acceptance_rate(uint64_t accepted, uint64_t drafted);
static double spec_bench_acceptance_length(uint64_t accepted, uint64_t rounds);

static json spec_bench_stage_json(const spec_bench_stage_delta & stage) {
    const double acceptance_rate = spec_bench_acceptance_rate(stage.accepted_tokens, stage.draft_tokens);
    const double acceptance_length = spec_bench_acceptance_length(stage.accepted_tokens, stage.num_drafts);

    json drafted_by_position = json::array();
    json accepted_by_position = json::array();
    json acceptance_rate_by_position = json::array();
    json conditional_acceptance_rate = json::array();
    for (size_t position = 0; position < stage.drafted_by_position.size(); ++position) {
        const uint64_t drafted = stage.drafted_by_position[position];
        const uint64_t accepted = position < stage.accepted_by_position.size()
            ? stage.accepted_by_position[position]
            : 0;
        drafted_by_position.push_back(drafted);
        accepted_by_position.push_back(accepted);
        acceptance_rate_by_position.push_back(drafted > 0 ? (double) accepted / (double) drafted : 0.0);
        if (position == 0) {
            conditional_acceptance_rate.push_back(nullptr);
        } else {
            const uint64_t previous_accepted = position - 1 < stage.accepted_by_position.size()
                ? stage.accepted_by_position[position - 1]
                : 0;
            conditional_acceptance_rate.push_back(previous_accepted > 0
                ? json((double) accepted / (double) previous_accepted)
                : json(nullptr));
        }
    }

    return json{
        {"type", common_speculative_type_to_str(stage.type)},
        {"num_drafts", stage.num_drafts},
        {"accepted_drafts", stage.accepted_drafts},
        {"draft_tokens", stage.draft_tokens},
        {"accepted_tokens", stage.accepted_tokens},
        {"acceptance_rate", acceptance_rate},
        {"acceptance_length", acceptance_length},
        {"drafted_by_position", drafted_by_position},
        {"accepted_by_position", accepted_by_position},
        {"acceptance_rate_by_position", acceptance_rate_by_position},
        {"conditional_acceptance_rate", conditional_acceptance_rate},
        {"t_begin_s", stage.t_begin_us / 1e6},
        {"t_draft_s", stage.t_draft_us / 1e6},
        {"t_accept_s", stage.t_accept_us / 1e6},
    };
}

static json spec_bench_metrics_json(const spec_bench_metrics_delta & delta) {
    const double acceptance_rate = spec_bench_acceptance_rate(delta.accepted_tokens, delta.draft_tokens);
    const double acceptance_length = spec_bench_acceptance_length(delta.accepted_tokens, delta.num_drafts);

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

static int spec_bench_resolve_max_tokens(const spec_bench_task & task, const gpt_params & params) {
    return task.max_tokens > 0 ? task.max_tokens : (params.n_predict > 0 ? params.n_predict : 256);
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

static std::string spec_bench_effective_prompt(
        llama_model * model,
        const gpt_params & params,
        const std::string & prompt) {
    if (!params.enable_chat_template) {
        return prompt;
    }
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
    if (!chat_templates) {
        throw std::runtime_error("failed to initialize chat templates");
    }
    return common_chat_format_single(chat_templates.get(), {}, common_chat_msg{"user", prompt}, true, params.use_jinja);
}

static llama_batch spec_bench_make_batch(
        const llama_tokens & tokens,
        int                  offset,
        int                  n_tokens,
        int                  n_past) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        // Keep positions and sequence ids available for MTP/DFlash feature capture.
        common_batch_add(batch, tokens[offset + i], n_past + i, { 0 }, true);
    }
    return batch;
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

    const int task_max_tokens = spec_bench_resolve_max_tokens(task, params);
    if (task_max_tokens <= 0) {
        result.error = "max token budget resolved to zero";
        return result;
    }

    result.effective_prompt = spec_bench_effective_prompt(model, params, task.prompt);
    llama_tokens prompt_tokens = common_tokenize(ctx, result.effective_prompt, true, true);
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
    if (params.has_mtp) {
        llama_set_embeddings(ctx, true);
    }

    llama_tokens embd = prompt_tokens;
    llama_tokens speculative_tokens = prompt_tokens;
    int n_past = 0;
    int n_remain = task_max_tokens;
    bool embd_is_prompt = true;
    int final_prompt_output_index = -1;
    llama_pos final_prompt_hidden_pos = -1;
    bool have_carry = false;
    llama_token carry_token = LLAMA_TOKEN_NULL;

    const auto spec_before = common_speculative_get_metrics_snapshot(spec);
    for (llama_token token : prompt_tokens) {
        common_sampler_accept(sampler, ctx, token, false);
    }

    const int64_t t_prompt_start_us = ggml_time_us();

    while (!embd.empty()) {
        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = std::min(params.n_batch, (int) embd.size() - i);
            llama_batch batch = spec_bench_make_batch(embd, i, n_eval, n_past);
            const int decode_result = llama_decode(ctx, batch);
            if (decode_result != 0) {
                llama_batch_free(batch);
                result.error = "prompt decode failed";
                return result;
            }
            if (spec != nullptr && embd_is_prompt) {
                if (common_speculative_on_target_seq_batch(spec, ctx, batch, 0, true) != 0) {
                    llama_batch_free(batch);
                    result.error = "speculative prompt warmup failed";
                    return result;
                }
            }
            if (embd_is_prompt && i + n_eval == (int) embd.size()) {
                final_prompt_output_index = n_eval - 1;
                final_prompt_hidden_pos = n_past + n_eval - 1;
            }
            llama_batch_free(batch);
            n_past += n_eval;
        }
        embd.clear();
    }

    if (spec != nullptr) {
        static const llama_tokens empty_speculative_prompt;
        const llama_tokens & speculative_prompt =
            params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) &&
            !params.speculative.has_composite_stage_chain()
                ? empty_speculative_prompt
                : speculative_tokens;
        common_speculative_begin(spec, speculative_prompt);
        if (params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) &&
            final_prompt_output_index >= 0 &&
            final_prompt_hidden_pos >= 0 &&
            !common_speculative_capture_output_hidden(spec, ctx, final_prompt_output_index, 0, final_prompt_hidden_pos)) {
            result.error = "failed to capture final prompt hidden state";
            return result;
        }
    }
    if (params.has_mtp) {
        llama_set_embeddings(ctx, false);
    }

    const int64_t t_prompt_end_us = ggml_time_us();
    const int64_t t_decode_start_us = t_prompt_end_us;

    while (n_remain > 0) {
        llama_tokens next_embd;
        bool used_speculative = false;
        bool have_fallback_sampled = false;
        llama_token fallback_sampled = LLAMA_TOKEN_NULL;

        if (spec != nullptr && n_remain >= 3) {
            static const llama_tokens empty_speculative_history;
            const llama_tokens & draft_history =
                params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) &&
                !params.speculative.has_composite_stage_chain()
                    ? empty_speculative_history
                    : speculative_tokens;
            auto round = common_speculative_run_round(
                spec, model, ctx, sampler, nullptr, params.speculative, params.sparams,
                0, n_past, n_remain, have_carry, draft_history, carry_token);
            if (round.failed) {
                result.error = round.error;
                return result;
            }
            if (round.sampled_before_ready && !round.used_speculative) {
                have_fallback_sampled = true;
                fallback_sampled = round.sampled_before;
            }
            if (round.used_speculative) {
                if (!round.sampled_before_from_carry) {
                    result.output_tokens.push_back(round.sampled_before);
                    n_remain -= 1;
                }
                result.output_tokens.insert(result.output_tokens.end(), round.ids.begin(), round.ids.end());
                n_remain -= (int) round.ids.size();
                n_past += (int) round.ids.size();
                carry_token = round.ids.back();
                have_carry = !llama_token_is_eog(model, carry_token);
                if (!have_carry) {
                    result.hit_eog = true;
                    n_remain = 0;
                }
                if (!params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) ||
                    params.speculative.has_composite_stage_chain()) {
                    speculative_tokens.push_back(round.sampled_before);
                    if (round.ids.size() > 1) {
                        speculative_tokens.insert(speculative_tokens.end(), round.ids.begin(), round.ids.end() - 1);
                    }
                }
                used_speculative = true;
            }
        }

        if (!used_speculative && have_carry) {
            next_embd.push_back(carry_token);
            have_carry = false;
            used_speculative = true;
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

        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
            int n_eval = std::min(params.n_batch, (int) embd.size() - i);
            llama_batch batch = spec_bench_make_batch(embd, i, n_eval, n_past);
            const int decode_result = llama_decode(ctx, batch);
            if (decode_result != 0) {
                llama_batch_free(batch);
                result.error = "decode failed";
                return result;
            }
            llama_batch_free(batch);
            if (spec != nullptr && (!params.speculative.has_stage_type(COMMON_SPECULATIVE_TYPE_MTP) || params.speculative.has_composite_stage_chain())) {
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

static std::string spec_bench_positions(const spec_bench_stage_delta & stage) {
    std::ostringstream out;
    const size_t count = std::max(stage.drafted_by_position.size(), stage.accepted_by_position.size());
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) { out << ", "; }
        const uint64_t drafted = i < stage.drafted_by_position.size() ? stage.drafted_by_position[i] : 0;
        const uint64_t accepted = i < stage.accepted_by_position.size() ? stage.accepted_by_position[i] : 0;
        out << accepted << "/" << drafted;
    }
    return out.str();
}

static json spec_bench_position_array(const std::vector<uint64_t> & values) {
    json result = json::array();
    for (const uint64_t value : values) {
        result.push_back(value);
    }
    return result;
}

static double spec_bench_acceptance_rate(uint64_t accepted, uint64_t drafted) {
    return drafted > 0 ? (double) accepted / (double) drafted : 0.0;
}

static double spec_bench_acceptance_length(uint64_t accepted, uint64_t rounds) {
    return rounds > 0 ? 1.0 + (double) accepted / (double) rounds : 0.0;
}

static json spec_bench_attempt_json(
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
        {"max_tokens", spec_bench_resolve_max_tokens(task, params)},
        {"repeat_index", repeat_index},
        {"builtin", task.builtin},
        {"prompts", opts.prompts_path.empty() ? "builtin-default" : opts.prompts_path},
        {"runtime", spec_bench_runtime_json(params)},
        {"variant", {{"is_baseline", is_baseline}, {"spec_types", spec_bench_stage_types_json(params.speculative)}, {"stage_chain", common_speculative_stage_chain_to_str(params.speculative)}}},
        {"sampler", spec_bench_sampler_json(params)},
        {"timing", {{"prompt_s", result.prompt_s}, {"decode_s", result.decode_s}, {"total_s", result.total_s}, {"decode_tps", decode_tps}, {"overall_tps", total_tps}}},
        {"tokens", {{"prompt", result.prompt_tokens}, {"generated", result.generated_tokens}}},
        {"speculative", spec_bench_metrics_json(result.spec_delta)},
        {"quality", {{"ok", result.ok}, {"error", result.error.empty() ? json(nullptr) : json(result.error)}, {"retries_used", result.retries_used}, {"hit_eog", result.hit_eog}}},
        {"prompt", result.effective_prompt},
        {"output", result.output_text},
    };
}

static json spec_bench_summary_json(
        const gpt_params & params,
        const spec_bench_options & opts,
        const std::vector<spec_bench_task> & tasks,
        const spec_bench_summary & summary) {
    const bool is_baseline = !params.speculative.has_stage_chain();
    const double decode_tps = summary.decode_s > 0.0 ? summary.generated_tokens / summary.decode_s : 0.0;
    const double total_tps = summary.total_s > 0.0 ? summary.generated_tokens / summary.total_s : 0.0;
    return json{
        {"row_type", "summary"},
        {"prompts", opts.prompts_path.empty() ? "builtin-default" : opts.prompts_path},
        {"requested_tasks", json(opts.task_names)},
        {"selected_tasks", spec_bench_task_names_json(tasks)},
        {"repeat", opts.repeat},
        {"retry", opts.retry},
        {"default_max_tokens", params.n_predict > 0 ? params.n_predict : 256},
        {"runtime", spec_bench_runtime_json(params)},
        {"variant", {{"is_baseline", is_baseline}, {"spec_types", spec_bench_stage_types_json(params.speculative)}, {"stage_chain", common_speculative_stage_chain_to_str(params.speculative)}}},
        {"sampler", spec_bench_sampler_json(params)},
        {"attempts", summary.attempts}, {"successes", summary.successes}, {"failures", summary.failures}, {"retries_used", summary.retries_used},
        {"timing", {{"prompt_s", summary.prompt_s}, {"decode_s", summary.decode_s}, {"total_s", summary.total_s}, {"decode_tps", decode_tps}, {"overall_tps", total_tps}}},
        {"tokens", {{"prompt", summary.prompt_tokens}, {"generated", summary.generated_tokens}}},
        {"speculative", spec_bench_metrics_json(summary.spec_delta)},
    };
}

static json spec_bench_compact_attempt_json(const spec_bench_task & task, const spec_bench_attempt_result & result, int repeat_index) {
    json stages = json::array();
    for (const auto & stage : result.spec_delta.stages) {
        stages.push_back({
            {"type", common_speculative_type_to_str(stage.type)},
            {"drafts", stage.num_drafts}, {"draft_tokens", stage.draft_tokens}, {"accepted", stage.accepted_tokens},
            {"accept_percent", 100.0 * spec_bench_acceptance_rate(stage.accepted_tokens, stage.draft_tokens)},
            {"accept_length", spec_bench_acceptance_length(stage.accepted_tokens, stage.num_drafts)},
            {"drafted_by_position", spec_bench_position_array(stage.drafted_by_position)},
            {"accepted_by_position", spec_bench_position_array(stage.accepted_by_position)},
        });
    }
    return json{{"row_type", "attempt"}, {"task", task.name}, {"run", repeat_index + 1}, {"ok", result.ok}, {"stop", !result.ok ? "fail" : result.hit_eog ? "eog" : "limit"}, {"generated", result.generated_tokens}, {"decode_s", result.decode_s}, {"decode_tps", result.decode_s > 0.0 ? result.generated_tokens / result.decode_s : 0.0}, {"stages", stages}, {"error", result.error.empty() ? json(nullptr) : json(result.error)}};
}

static json spec_bench_compact_summary_json(const spec_bench_summary & summary) {
    return json{{"row_type", "summary"}, {"attempts", summary.attempts}, {"successes", summary.successes}, {"failures", summary.failures}, {"generated", summary.generated_tokens}, {"decode_s", summary.decode_s}, {"decode_tps", summary.decode_s > 0.0 ? summary.generated_tokens / summary.decode_s : 0.0}, {"speculative", spec_bench_metrics_json(summary.spec_delta)}};
}

static void spec_bench_print_markdown(const spec_bench_options & opts, const std::vector<spec_bench_record> & records) {
    auto number = [](double value, int precision) {
        std::ostringstream out; out << std::fixed << std::setprecision(precision) << value; return out.str();
    };
    auto fit = [](const std::string & value, size_t width) { return value.size() <= width ? value : value.substr(0, width - 3) + "..."; };
    constexpr int stage_width = 14;
    auto error_text = [&](const std::string & value) {
        std::string result = value;
        for (char & ch : result) { if (ch == 10 || ch == 13 || ch == 9) { ch = 32; } }
        return fit(string_strip(result), 120);
    };
    size_t max_backticks = 3;
    auto inspect_fence = [&](const std::string & text) {
        size_t run = 0;
        for (const char ch : text) {
            if (ch == char(96)) {
                ++run;
                max_backticks = std::max(max_backticks, run);
            } else {
                run = 0;
            }
        }
    };
    for (const auto & record : records) {
        inspect_fence(record.result.effective_prompt);
        inspect_fence(record.result.output_text);
    }
    const std::string fence(max_backticks + 1, char(96));

    if (opts.output_details) {
        std::cout << "\n## Prompt and response details\n\n";
        for (const auto & record : records) {
            std::cout << "### " << record.task.name << " / run " << (record.repeat_index + 1) << "\n\n";
            std::cout << "Prompt:\n" << fence << "\n" << record.result.effective_prompt << "\n" << fence << "\n\n";
            std::cout << "Response:\n" << fence << "\n" << record.result.output_text << "\n" << fence << "\n\n";
        }
    }

    std::cout << "\n| " << std::left << std::setw(8) << "task" << " | " << std::right << std::setw(3) << "run"
              << " | " << std::left << std::setw(stage_width) << "stage" << " | " << std::right << std::setw(7) << "tokens"
              << " | " << std::left << std::setw(5) << "stop" << " | " << std::right << std::setw(7) << "time(s)"
              << " | " << std::setw(7) << "tok/s" << " | " << std::setw(6) << "rounds"
              << " | " << std::setw(11) << "accepted" << " | " << std::setw(7) << "rate"
              << " | " << std::setw(6) << "a.len" << " | " << std::left << std::setw(28) << "pos accept" << " |\n";
    std::cout << "|----------|" << std::string(stage_width + 2, '-') << "|---------|-------|---------|---------|--------|-------------|---------|--------|------------------------------|\n";

    auto position_percentages = [&](const spec_bench_stage_delta & stage) {
        std::ostringstream out;
        const size_t count = std::max(stage.drafted_by_position.size(), stage.accepted_by_position.size());
        for (size_t i = 0; i < count; ++i) {
            if (i > 0) { out << " "; }
            const uint64_t drafted = i < stage.drafted_by_position.size() ? stage.drafted_by_position[i] : 0;
            const uint64_t accepted = i < stage.accepted_by_position.size() ? stage.accepted_by_position[i] : 0;
            if (drafted == 0) { out << "-"; }
            else { out << std::fixed << std::setprecision(1) << (100.0 * accepted / drafted) << "%"; }
        }
        return out.str();
    };

    for (const auto & record : records) {
        const auto & result = record.result;
        const int run = record.repeat_index + 1;
        const std::string stop = !result.ok ? "fail" : result.hit_eog ? "eog" : "limit";
        const std::string tokens = result.ok ? std::to_string(result.generated_tokens) : "-";
        const double tps = result.decode_s > 0.0 ? result.generated_tokens / result.decode_s : 0.0;
        auto row = [&](const std::string & stage_name, uint64_t rounds, uint64_t drafted, uint64_t accepted, const std::string & positions) {
            const bool has_metrics = drafted > 0 || rounds > 0;
            std::cout << "| " << std::left << std::setw(8) << fit(record.task.name, 8) << " | " << std::right << std::setw(3) << run
                      << " | " << std::left << std::setw(stage_width) << fit(stage_name, stage_width) << " | " << std::right << std::setw(7) << fit(tokens, 7)
                      << " | " << std::left << std::setw(5) << stop << " | " << std::right << std::setw(7) << (result.ok ? number(result.decode_s, 3) : "-")
                      << " | " << std::setw(7) << (result.ok ? number(tps, 2) : "-") << " | " << std::setw(6) << (has_metrics ? std::to_string(rounds) : "-")
                      << " | " << std::setw(11) << (has_metrics ? std::to_string(accepted) + "/" + std::to_string(drafted) : "-")
                      << " | " << std::setw(7) << (has_metrics ? number(100.0 * spec_bench_acceptance_rate(accepted, drafted), 2) + "%" : "-")
                      << " | " << std::setw(6) << (has_metrics ? number(spec_bench_acceptance_length(accepted, rounds), 2) : "-")
                      << " | " << std::left << std::setw(28) << fit(positions.empty() ? "-" : positions, 28) << " |\n";
        };
        if (!result.ok && result.spec_delta.stages.empty()) { row("error", 0, 0, 0, ""); }
        else if (result.spec_delta.stages.empty()) { row("base", 0, 0, 0, ""); }
        else {
            for (const auto & stage : result.spec_delta.stages) {
                row(common_speculative_type_to_str(stage.type), stage.num_drafts, stage.draft_tokens, stage.accepted_tokens, position_percentages(stage));
            }
        }
    }
    std::cout << "\n";

    if (opts.output_details) {
        size_t raw_position_width = 30;
        for (const auto & record : records) {
            for (const auto & stage : record.result.spec_delta.stages) {
                raw_position_width = std::max(raw_position_width, spec_bench_positions(stage).size());
            }
        }
        std::cout << "## Detailed metrics\n\n";
        std::cout << "| " << std::left << std::setw(8) << "task" << " | " << std::right << std::setw(3) << "run"
                  << " | " << std::setw(10) << "prompt tok" << " | " << std::setw(9) << "prompt s"
                  << " | " << std::setw(9) << "total s" << " | " << std::left << std::setw(stage_width) << "stage"
                  << " | " << std::right << std::setw(9) << "draft s" << " | " << std::setw(9) << "accept s"
                  << " | " << std::left << std::setw(raw_position_width) << "accepted/drafted by position" << " |\n";
        std::cout << "|----------|-----|------------|-----------|-----------|" << std::string(stage_width + 2, '-') << "|-----------|-----------|"
                  << std::string(raw_position_width + 2, '-') << "|\n";
        for (const auto & record : records) {
            const auto & result = record.result;
            auto metric_row = [&](const std::string & stage_name, double draft_s, double accept_s, const std::string & positions) {
                std::cout << "| " << std::left << std::setw(8) << fit(record.task.name, 8) << " | " << std::right << std::setw(3) << (record.repeat_index + 1)
                          << " | " << std::setw(10) << result.prompt_tokens << " | " << std::setw(9) << number(result.prompt_s, 3)
                          << " | " << std::setw(9) << number(result.total_s, 3) << " | " << std::left << std::setw(stage_width) << fit(stage_name, stage_width)
                          << " | " << std::right << std::setw(9) << number(draft_s, 6) << " | " << std::setw(9) << number(accept_s, 6)
                          << " | " << std::left << std::setw(raw_position_width) << (positions.empty() ? "-" : positions) << " |\n";
            };
            if (result.spec_delta.stages.empty()) { metric_row("base", 0.0, 0.0, ""); }
            else {
                for (const auto & stage : result.spec_delta.stages) {
                    metric_row(common_speculative_type_to_str(stage.type), stage.t_draft_us / 1e6, stage.t_accept_us / 1e6, spec_bench_positions(stage));
                }
            }
        }
        std::cout << "\n";
    }

    if (opts.repeat > 1) {
        struct repeat_group { std::string task; std::string stage; std::vector<double> speed; std::vector<double> rate; std::vector<double> length; };
        std::vector<repeat_group> groups;
        auto get_group = [&](const std::string & task, const std::string & stage) -> repeat_group & {
            for (auto & group : groups) { if (group.task == task && group.stage == stage) { return group; } }
            groups.push_back({task, stage, {}, {}, {}}); return groups.back();
        };
        for (const auto & record : records) {
            if (!record.result.ok) { continue; }
            const double speed = record.result.decode_s > 0.0 ? record.result.generated_tokens / record.result.decode_s : 0.0;
            if (record.result.spec_delta.stages.empty()) { get_group(record.task.name, "base").speed.push_back(speed); continue; }
            for (const auto & stage : record.result.spec_delta.stages) {
                auto & group = get_group(record.task.name, common_speculative_type_to_str(stage.type));
                group.speed.push_back(speed);
                if (stage.num_drafts > 0 || stage.draft_tokens > 0) {
                    group.rate.push_back(spec_bench_acceptance_rate(stage.accepted_tokens, stage.draft_tokens));
                    group.length.push_back(spec_bench_acceptance_length(stage.accepted_tokens, stage.num_drafts));
                }
            }
        }
        auto mean_std = [](const std::vector<double> & values) {
            if (values.empty()) { return std::pair<double, double>{0.0, 0.0}; }
            double mean = 0.0; for (double value : values) { mean += value; } mean /= values.size();
            double variance = 0.0; for (double value : values) { const double delta = value - mean; variance += delta * delta; }
            return std::pair<double, double>{mean, std::sqrt(variance / values.size())};
        };
        std::cout << "Repeat summary (" << opts.repeat << " runs/task)\n\n";
        std::cout << "| " << std::left << std::setw(8) << "task" << " | " << std::setw(stage_width) << "stage" << " | " << std::right << std::setw(4) << "runs"
                  << " | " << std::setw(8) << "metric n"
                  << " | " << std::setw(15) << "tok/s mean/std" << " | " << std::setw(15) << "rate mean/std" << " | " << std::setw(15) << "a.len mean/std" << " |\n";
        std::cout << "|----------|" << std::string(stage_width + 2, '-') << "|------|----------|-----------------|-----------------|-----------------|\n";
        for (const auto & group : groups) {
            const auto speed = mean_std(group.speed); const auto rate = mean_std(group.rate); const auto length = mean_std(group.length);
            std::cout << "| " << std::left << std::setw(8) << fit(group.task, 8) << " | " << std::setw(stage_width) << fit(group.stage, stage_width)
                      << " | " << std::right << std::setw(4) << group.speed.size() << " | " << std::setw(8) << group.rate.size()
                      << " | " << std::setw(15) << number(speed.first, 2) + "/" + number(speed.second, 2)
                      << " | " << std::setw(15) << (group.rate.empty() ? "-" : number(100.0 * rate.first, 2) + "%/" + number(100.0 * rate.second, 2) + "%")
                      << " | " << std::setw(15) << (group.length.empty() ? "-" : number(length.first, 2) + "/" + number(length.second, 2)) << " |\n";
        }
        std::cout << "\n";
    }

    bool printed_errors = false;
    for (const auto & record : records) {
        if (!record.result.ok) {
            if (!printed_errors) { std::cout << "Errors:\n"; printed_errors = true; }
            std::cout << "- " << record.task.name << " run " << (record.repeat_index + 1) << ": " << error_text(record.result.error) << "\n";
        }
    }
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
        tasks = spec_bench_select_tasks(bench_opts, params);
    } catch (const std::exception & e) {
        LOG_TEE("%s\n", e.what());
        return 1;
    }

    std::ostream * out = &std::cout;

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
    std::vector<spec_bench_record> records;
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
            records.push_back({task, best_result, repeat_index});
            if (bench_opts.output_format == "jsonl") {
                *out << (bench_opts.output_details
                    ? spec_bench_attempt_json(params, bench_opts, task, best_result, repeat_index)
                    : spec_bench_compact_attempt_json(task, best_result, repeat_index)).dump() << '\n';
            }
        }
    }

    if (bench_opts.output_format == "md") {
        spec_bench_print_markdown(bench_opts, records);
    } else {
        *out << (bench_opts.output_details
            ? spec_bench_summary_json(params, bench_opts, tasks, summary)
            : spec_bench_compact_summary_json(summary)).dump() << '\n';
    }
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
