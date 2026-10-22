#include "common.h"
#include "ngram-cache.h"
#include "speculative.h"

#include <array>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

bool require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool same_cache(const common_ngram_cache & lhs, const common_ngram_cache & rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const auto & [ngram, followers] : lhs) {
        const auto it = rhs.find(ngram);
        if (it == rhs.end() || it->second.size() != followers.size()) {
            return false;
        }
        for (const auto & [token, count] : followers) {
            const auto follower = it->second.find(token);
            if (follower == it->second.end() || follower->second != count) {
                return false;
            }
        }
    }
    return true;
}

bool same_part(const common_ngram_cache_part & lhs, const std::map<llama_token, int32_t> & rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    auto expected = rhs.begin();
    for (const auto & entry : lhs) {
        if (expected == rhs.end() || entry.first != expected->first || entry.second != expected->second) {
            return false;
        }
        ++expected;
    }
    return expected == rhs.end();
}

bool test_sorted_follower_part() {
    const std::vector<std::vector<llama_token>> orders = {
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {3, 1, 5, 2, 4},
    };
    for (const auto & order : orders) {
        common_ngram_cache_part part;
        std::map<llama_token, int32_t> reference;
        for (llama_token token : order) {
            const auto [it, inserted] = part.emplace(token, token * 10);
            reference.emplace(token, token * 10);
            if (!require(inserted && it->first == token, "new follower insertion succeeds")) return false;
        }
        for (llama_token token : order) {
            const auto [it, inserted] = part.emplace(token, -1);
            if (!require(!inserted && it->second == token * 10,
                        "duplicate insertion preserves the original follower count")) return false;
        }
        if (!require(same_part(part, reference), "followers remain sorted and unique for every insertion order")) {
            return false;
        }
        for (llama_token token = 0; token <= 6; ++token) {
            const auto actual = part.find(token);
            const auto expected = reference.find(token);
            if (!require((actual == part.end()) == (expected == reference.end()) &&
                         (actual == part.end() || actual->second == expected->second),
                         "binary-search find agrees with ordered reference for hits and misses")) return false;
        }
    }

    common_ngram_cache_part empty;
    if (!require(empty.empty() && empty.find(0) == empty.end(), "empty follower part lookup")) return false;
    empty.emplace(7, 70);
    if (!require(empty.size() == 1 && empty.begin()->first == 7 && empty.find(7)->second == 70,
                 "singleton follower part lookup")) return false;

    // Exercise logarithmic-search boundary positions and mixed fanout sizes.
    for (int size : {2, 3, 8, 31, 127, 513}) {
        common_ngram_cache_part part;
        std::map<llama_token, int32_t> reference;
        for (int i = size - 1; i >= 0; --i) {
            const llama_token token = 1000 + i * 3;
            part.emplace(token, i + 1);
            reference.emplace(token, i + 1);
        }
        part.emplace(999, 9);                     // insert at front
        reference.emplace(999, 9);
        part.emplace(1000 + size * 3, 10);       // insert at end
        reference.emplace(1000 + size * 3, 10);
        if (!require(same_part(part, reference), "fanout insertion at either end matches reference")) return false;
        for (llama_token token = 998; token <= 1000 + size * 3 + 1; ++token) {
            const auto actual = part.find(token);
            const auto expected = reference.find(token);
            if (!require((actual == part.end()) == (expected == reference.end()),
                         "fanout hit/miss positions match trusted reference")) return false;
        }
    }
    return true;
}

common_ngram_cache make_repetitive_cache();

bool test_incremental_updates() {
    const llama_tokens stream = {
        11, 12, 13, 14, 15, 11, 12, 13, 21, 22, 23, 24,
        31, 32, 33, 34, 35, 36, 31, 32, 33, 34, 41, 42,
    };

    common_ngram_cache full;
    auto full_input = stream;
    common_ngram_cache_update(full, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
            full_input, static_cast<int>(full_input.size()), false);

    const std::array<std::vector<size_t>, 2> schedules = {{
        {1},
        {3, 2, 5},
    }};
    for (const auto & schedule : schedules) {
        common_ngram_cache incremental;
        llama_tokens accumulated;
        size_t offset = 0;
        size_t schedule_index = 0;
        while (offset < stream.size()) {
            const size_t requested = schedule[schedule_index++ % schedule.size()];
            const size_t count = std::min(requested, stream.size() - offset);
            const size_t old_size = accumulated.size();
            accumulated.insert(accumulated.end(), stream.begin() + offset,
                    stream.begin() + offset + count);
            common_ngram_cache_update(incremental, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    accumulated, static_cast<int>(accumulated.size() - old_size), false);
            offset += count;
        }
        if (!require(same_cache(full, incremental),
                    "full ingestion and incremental append schedules must count identical n-grams")) {
            return false;
        }
    }

    // Simulate newly accepted generated text as further appended tokens.
    auto with_accepted = stream;
    with_accepted.insert(with_accepted.end(), {51, 52, 53, 54});
    common_ngram_cache accepted_full;
    common_ngram_cache_update(accepted_full, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
            with_accepted, static_cast<int>(with_accepted.size()), false);
    size_t old_size = stream.size();
    common_ngram_cache_update(full, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
            with_accepted, static_cast<int>(with_accepted.size() - old_size), false);
    return require(same_cache(accepted_full, full),
            "accepted generated text must extend the same cache exactly once");
}

common_ngram_cache make_repetitive_cache() {
    llama_tokens corpus;
    for (int repeat = 0; repeat < 64; ++repeat) {
        for (llama_token token = 1; token <= 12; ++token) {
            corpus.push_back(token);
        }
    }
    common_ngram_cache result;
    common_ngram_cache_update(result, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
            corpus, static_cast<int>(corpus.size()), false);
    return result;
}

size_t draft_count(common_ngram_cache & cache, int n_max) {
    common_ngram_cache empty_context;
    common_ngram_cache empty_dynamic;
    llama_tokens input = {1, 2};
    llama_tokens draft = {2}; // carry token is not a proposal
    common_ngram_cache_draft(input, draft, n_max, LLAMA_NGRAM_MIN,
            LLAMA_NGRAM_MAX, empty_context, empty_dynamic, cache);
    return draft.size() - 1;
}

size_t draft_count_dynamic(common_ngram_cache & cache, int n_max) {
    common_ngram_cache empty_context;
    common_ngram_cache empty_static;
    llama_tokens input = {1, 2};
    llama_tokens draft = {2};
    common_ngram_cache_draft(input, draft, n_max, LLAMA_NGRAM_MIN,
            LLAMA_NGRAM_MAX, empty_context, cache, empty_static);
    return draft.size() - 1;
}

llama_tokens draft_from_history(
        llama_tokens history,
        int n_max,
        common_ngram_cache & context,
        common_ngram_cache & dynamic,
        common_ngram_cache & statik,
        int ngram_min = LLAMA_NGRAM_MIN,
        int ngram_max = LLAMA_NGRAM_MAX) {
    llama_tokens draft = history.empty() ? llama_tokens{99} : llama_tokens{history.back()};
    common_ngram_cache_draft(history, draft, n_max, ngram_min, ngram_max, context, dynamic, statik);
    draft.erase(draft.begin()); // the initial item is the carry token, not a proposal
    return draft;
}

common_ngram_cache make_repeated_sequence_cache(const llama_tokens & sequence, int repeats) {
    llama_tokens corpus;
    for (int i = 0; i < repeats; ++i) {
        corpus.insert(corpus.end(), sequence.begin(), sequence.end());
    }
    common_ngram_cache result;
    common_ngram_cache_update(result, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
            corpus, static_cast<int>(corpus.size()), false);
    return result;
}

common_ngram_cache make_follower_cache(
        const llama_tokens & key,
        const std::vector<std::pair<llama_token, int32_t>> & followers) {
    common_ngram_cache result;
    common_ngram_cache_part part;
    for (const auto & [token, count] : followers) {
        part.emplace(token, count);
    }
    result.emplace(common_ngram(key.data(), static_cast<int>(key.size())), std::move(part));
    return result;
}

bool test_short_histories_and_growing_draft() {
    common_ngram_cache empty_context;
    common_ngram_cache empty_dynamic;
    common_ngram_cache empty_static;

    for (const llama_tokens history : {llama_tokens{}, llama_tokens{10}}) {
        const auto drafted = draft_from_history(history, 4, empty_context, empty_dynamic, empty_static);
        if (!require(drafted.empty(), "empty and one-token histories retain the static-key minimum behavior")) {
            return false;
        }
    }

    auto static_two = make_repeated_sequence_cache({10, 11, 12}, 2);
    const auto two = draft_from_history({10, 11}, 1, empty_context, empty_dynamic, static_two);
    if (!require(two == llama_tokens{12}, "a two-token history can use the established static fallback")) {
        return false;
    }

    auto dynamic_three = make_repeated_sequence_cache({20, 21, 22, 23}, 3);
    const auto three = draft_from_history({20, 21, 22}, 1, empty_context, dynamic_three, empty_static);
    if (!require(three == llama_tokens{23}, "a three-token history selects its available 3-gram key")) {
        return false;
    }

    auto dynamic_four = make_repeated_sequence_cache({30, 31, 32, 33, 34}, 3);
    const auto four = draft_from_history({30, 31, 32, 33}, 1, empty_context, dynamic_four, empty_static);
    if (!require(four == llama_tokens{34}, "a four-token history selects its available 4-gram key")) {
        return false;
    }

    // The only available key is a 2-gram. A 2/4 split passes its 50% threshold
    // but not the 1-gram 66% threshold, exposing accidental index shifting.
    common_ngram_cache threshold_context;
    common_ngram_cache threshold_static;
    llama_tokens threshold_corpus = {40, 41, 42, 40, 41, 42, 40, 41, 43, 40, 41, 43};
    common_ngram_cache_update(threshold_context, 2, 2, threshold_corpus,
            static_cast<int>(threshold_corpus.size()), false);
    const common_ngram threshold_key(threshold_corpus.data(), 2);
    const auto threshold_part = threshold_context.find(threshold_key);
    if (!require(threshold_part != threshold_context.end() && threshold_part->second.size() == 2 &&
                 threshold_part->second.find(42)->second == 2 && threshold_part->second.find(43)->second == 2,
                 "threshold fixture contains the intended 2/4 follower distribution")) {
        return false;
    }
    const auto threshold = draft_from_history({40, 41}, 1, threshold_context,
            empty_dynamic, threshold_static, 2, 4);
    if (!require(threshold.size() == 1 && (threshold[0] == 42 || threshold[0] == 43),
                "available 2-gram keys use their own threshold slot")) {
        return false;
    }

    // The two-token history can draft from a 2-gram. That proposal exposes a
    // 3-gram to the next iteration, which must be formed only then.
    auto growing_dynamic = make_repeated_sequence_cache({50, 51, 52, 53}, 3);
    auto growing_static = make_repeated_sequence_cache({50, 51, 52}, 2);
    const auto growing = draft_from_history({50, 51}, 3, empty_context,
            growing_dynamic, growing_static);
    return require(growing.size() >= 2 && growing[0] == 52 && growing[1] == 53,
            "the available key length grows with proposals during one draft call");
}

bool test_reference_drafting() {
    const llama_tokens history = {70, 71};
    common_ngram_cache empty_context;
    common_ngram_cache empty_dynamic;
    common_ngram_cache empty_static;

    const auto empty_result = draft_from_history(history, 2, empty_context,
            empty_dynamic, empty_static, 2, 2);
    if (!require(empty_result.empty(), "empty caches and absent static entries fall back without a proposal")) {
        return false;
    }

    auto context = make_follower_cache(history, {{72, 4}});
    auto dynamic = make_follower_cache(history, {{73, 5}});
    auto statik = make_follower_cache(history, {{74, 6}});
    const auto context_before = context;
    const auto dynamic_before = dynamic;
    const auto static_before = statik;

    const auto context_winner = draft_from_history(history, 1, context, dynamic,
            statik, 2, 2);
    const auto dynamic_winner = draft_from_history(history, 1, empty_context,
            dynamic, empty_static, 2, 2);
    const auto static_winner = draft_from_history(history, 1, empty_context,
            empty_dynamic, statik, 2, 2);
    if (!require(context_winner == llama_tokens{72} && dynamic_winner == llama_tokens{73} &&
                 static_winner == llama_tokens{74},
                 "context, dynamic, and static paths retain independent unique winners")) {
        return false;
    }

    // Context does not meet its 50% confidence threshold. The existing static
    // fallback must still win after primary selection rejects it.
    auto low_confidence = make_follower_cache(history, {{80, 1}, {81, 1}, {82, 1}});
    const auto fallback = draft_from_history(history, 1, low_confidence,
            empty_dynamic, statik, 2, 2);
    if (!require(fallback == llama_tokens{74},
                 "low-confidence context continues to the established static fallback")) {
        return false;
    }

    bool repeated_unique = true;
    for (int i = 0; i < 8; ++i) {
        repeated_unique &= draft_from_history(history, 1, context, dynamic,
                statik, 2, 2) == llama_tokens{72};
    }
    if (!require(repeated_unique && same_cache(context, context_before) &&
                 same_cache(dynamic, dynamic_before) && same_cache(statik, static_before),
                 "repeated calls preserve unique output and do not mutate any cache")) {
        return false;
    }

    auto tied = make_follower_cache(history, {{90, 1}, {91, 1}});
    const auto tie_winner = draft_from_history(history, 1, tied,
            empty_dynamic, empty_static, 2, 2);
    if (!require(tie_winner.size() == 1 &&
                 (tie_winner[0] == 90 || tie_winner[0] == 91) &&
                 same_cache(tied, make_follower_cache(history, {{90, 1}, {91, 1}})),
                 "tied confidence accepts either valid winner without changing counts")) {
        return false;
    }
    std::cout << "tie_winner=" << tie_winner[0] << " valid_set=90,91\n";

    // Equal weighted scores retain the first key encountered. O2's sorted
    // iteration therefore visits 10 before 20; the second key is deliberately
    // arranged so the first winner reaches a valid next 3-gram proposal.
    const llama_tokens tied_key = {1, 2};
    auto tied_primary = make_follower_cache(tied_key, {{10, 2}, {20, 1}});
    auto tied_static = make_follower_cache(tied_key, {{10, 1}, {20, 2}});
    const llama_tokens next_key = {1, 2, 10};
    auto tied_context = make_follower_cache(next_key, {{11, 1}});
    auto tied_depth = draft_from_history({1, 2}, 2, tied_context,
            tied_primary, tied_static);
    if (!require(tied_depth == llama_tokens{10, 11},
                 "sorted tie order deterministically selects the lower token and its valid next key")) {
        return false;
    }
    std::cout << "tie_confidence_depth=2 sorted_winner=10 alternative=20\n";
    return true;
}

bool test_draft_budgets() {
    auto cache = make_repetitive_cache();
    for (const int n_max : {0, 1, 3, 8, 16}) {
        if (!require(draft_count(cache, n_max) == static_cast<size_t>(n_max),
                    "ngram cache must obey n_max and exclude the carry token")) {
            return false;
        }
    }

    common_params_speculative request;
    request.n_max = 16;
    common_speculative_stage_params override;
    override.type = COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
    override.n_max = 3;
    const auto effective = request.with_stage_overrides(override);
    return require(effective.n_max == 3 && draft_count(cache, effective.n_max) == 3,
            "per-request stage n_max override must determine the effective proposal budget");
}

common_speculative * make_cache_stage(
        llama_context * ctx,
        int n_max,
        const std::string & static_path = {},
        const std::string & dynamic_path = {}) {
    common_params_speculative params;
    params.type = COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
    params.n_max = n_max;
    params.lookup_cache_static = static_path;
    params.lookup_cache_dynamic = dynamic_path;
    return common_speculative_init(params, ctx);
}

llama_tokens stage_draft(common_speculative * spec, int n_max, const llama_tokens & history) {
    common_params_speculative params;
    params.type = COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
    params.n_max = n_max;
    if (history.empty()) {
        return {};
    }
    llama_tokens prompt(history.begin(), history.end() - 1);
    return common_speculative_draft(spec, params, prompt, history.back());
}

const llama_tokens k_stage_training = {
    20, 21, 10, 11, 12, 13, 14, 12, 13, 15, 12, 13, 15, 30, 31,
    10, 11, 12, 13,
};
const llama_tokens k_stage_short = {10, 11, 12, 13};
const llama_tokens k_stage_prefix = {20, 21, 10, 11};

bool test_actual_stage_lifecycle(llama_context * ctx, const std::filesystem::path & dir) {
    const auto empty_static = (dir / "empty-static.cache").string();
    const auto empty_dynamic = (dir / "empty-dynamic.cache").string();
    common_ngram_cache empty;
    common_ngram_cache_save(empty, empty_static);
    common_ngram_cache_save(empty, empty_dynamic);

    bool ok = true;
    {
        auto * reused = make_cache_stage(ctx, 1, empty_static, empty_dynamic);
        auto * fresh = make_cache_stage(ctx, 1, empty_static, empty_dynamic);
        ok &= require(reused != nullptr && fresh != nullptr, "initialize real ngram-cache stages through the public API");
        if (reused != nullptr && fresh != nullptr) {
            common_speculative_begin(reused, k_stage_training);
            const auto learned = stage_draft(reused, 1, k_stage_training);
            ok &= require(learned == llama_tokens{14}, "actual stage learns the unique 4-gram continuation");

            common_speculative_begin(reused, k_stage_short);
            const auto after_begin = stage_draft(reused, 1, k_stage_short);
            common_speculative_begin(fresh, k_stage_short);
            const auto fresh_result = stage_draft(fresh, 1, k_stage_short);
            ok &= require(after_begin == fresh_result && after_begin.empty(),
                    "begin resets context history for a shorter request whose ending suffix was learned earlier");
        }
        common_speculative_free(reused);
        common_speculative_free(fresh);
    }

    {
        auto * full = make_cache_stage(ctx, 1, empty_static, empty_dynamic);
        auto * incremental = make_cache_stage(ctx, 1, empty_static, empty_dynamic);
        ok &= require(full != nullptr && incremental != nullptr, "initialize full and incremental stage instances");
        if (full != nullptr && incremental != nullptr) {
            common_speculative_begin(full, k_stage_training);
            const auto full_result = stage_draft(full, 1, k_stage_training);

            common_speculative_begin(incremental, k_stage_prefix);
            const auto prefix_result = stage_draft(incremental, 1, k_stage_prefix);
            const auto incremental_result = stage_draft(incremental, 1, k_stage_training);
            ok &= require(prefix_result.empty(), "incremental seed has no unrelated continuation");
            ok &= require(full_result == llama_tokens{14} && incremental_result == full_result,
                    "actual stage full ingestion and cross-boundary incremental ingestion agree");

            // A non-prefix rewrite within one request must rebuild the current
            // context map rather than retaining the previous longer history.
            common_speculative_begin(full, k_stage_training);
            const auto before_rewrite = stage_draft(full, 1, k_stage_training);
            const auto rewritten = stage_draft(full, 1, k_stage_short);
            auto * fresh_short = make_cache_stage(ctx, 1, empty_static, empty_dynamic);
            common_speculative_begin(fresh_short, k_stage_short);
            const auto fresh_rewrite = stage_draft(fresh_short, 1, k_stage_short);
            ok &= require(before_rewrite == llama_tokens{14}, "rewrite fixture first learns its stale-suffix continuation");
            ok &= require(rewritten == fresh_rewrite && rewritten.empty(),
                    "shrink/rewrite rebuild agrees with a fresh stage state");
            common_speculative_free(fresh_short);
        }
        common_speculative_free(full);
        common_speculative_free(incremental);
    }

    // External caches are intentionally loaded data. Request begin must clear
    // only context history, retaining either loaded cache as a draft source.
    auto loaded_cache = make_repeated_sequence_cache({60, 61, 62}, 3);
    const auto loaded_path = (dir / "loaded.cache").string();
    common_ngram_cache_save(loaded_cache, loaded_path);
    for (const bool as_static : {true, false}) {
        auto * stage = make_cache_stage(ctx, 1,
                as_static ? loaded_path : empty_static,
                as_static ? empty_dynamic : loaded_path);
        ok &= require(stage != nullptr, "initialize stage with one intentionally loaded external cache");
        if (stage != nullptr) {
            common_speculative_begin(stage, {900});
            const auto result = stage_draft(stage, 1, {60, 61});
            ok &= require(result == llama_tokens{62}, as_static
                    ? "loaded static cache survives request begin"
                    : "loaded dynamic cache survives request begin");
        }
        common_speculative_free(stage);
    }
    return ok;
}

bool run_model_backed_stage_tests(const std::string & model_path) {
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!require(model != nullptr, "load the local model only to create a real target context")) {
        llama_backend_free();
        return false;
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 128;
    context_params.n_ubatch = 128;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    llama_context * ctx = llama_init_from_model(model, context_params);
    if (!require(ctx != nullptr, "create a small CPU context without running target inference")) {
        llama_free_model(model);
        llama_backend_free();
        return false;
    }

    const auto dir = std::filesystem::temp_directory_path() /
            ("ik-ngram-stage-test-" + std::to_string(ggml_time_us()));
    bool ok = std::filesystem::create_directory(dir);
    ok &= require(ok, "create an exclusive stage-test cache directory");
    if (ok) {
        ok &= test_actual_stage_lifecycle(ctx, dir);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        ok &= require(!ec, "remove only the stage-test-owned directory");
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    if (ok) {
        std::cout << "PASS: actual ngram-cache stage lifecycle via public API; no target inference\n";
    }
    return ok;
}

std::vector<char *> make_argv(std::vector<std::string> & args) {
    std::vector<char *> result;
    result.reserve(args.size());
    for (auto & arg : args) {
        result.push_back(arg.data());
    }
    return result;
}

bool test_lookup_path_plumbing() {
    const auto nonce = std::to_string(ggml_time_us());
    const auto dir = std::filesystem::temp_directory_path() / ("ik-ngram-cache-test-" + nonce);
    if (!std::filesystem::create_directory(dir)) {
        return require(false, "create exclusive temporary test directory");
    }
    const auto static_path = (dir / "static.cache").string();
    const auto dynamic_path = (dir / "dynamic.cache").string();
    bool ok = true;

    auto seed = make_repetitive_cache();
    common_ngram_cache_save(seed, static_path);
    common_ngram_cache_save(seed, dynamic_path);

    std::vector<std::string> args = {
        "test-ngram-cache", "--spec-type", "ngram-cache:n_max=16",
        "-lcs", static_path, "-lcd", dynamic_path,
    };
    auto argv = make_argv(args);
    gpt_params parsed;
    ok &= require(gpt_params_parse(static_cast<int>(argv.size()), argv.data(), parsed),
            "parse ngram-cache and -lcs/-lcd options");
    if (ok) {
        ok &= require(parsed.lookup_cache_static == static_path &&
                parsed.lookup_cache_dynamic == dynamic_path,
                "CLI paths must remain in their documented top-level fields");
        ok &= require(parsed.speculative.lookup_cache_static == static_path &&
                parsed.speculative.lookup_cache_dynamic == dynamic_path,
                "CLI paths must also reach the nested fields consumed by the stage");
    }
    if (ok) {
        auto loaded = common_ngram_cache_load(parsed.speculative.lookup_cache_static);
        ok &= require(draft_count(loaded, parsed.speculative.n_max) == 16,
                "the parsed static path must load contents that affect draft output");
        auto dynamic = common_ngram_cache_load(parsed.speculative.lookup_cache_dynamic);
        ok &= require(draft_count_dynamic(dynamic, parsed.speculative.n_max) > 0,
                "the parsed dynamic path must load contents that affect draft output");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    ok &= require(!ec, "remove only the test-owned temporary directory");
    return ok;
}

bool write_legacy_unsorted_cache(const std::filesystem::path & path) {
    const llama_tokens key_tokens = {44, 22};
    const common_ngram key(key_tokens.data(), static_cast<int>(key_tokens.size()));
    const int32_t count = 4;
    const std::array<std::pair<llama_token, int32_t>, 4> followers = {{{9, 2}, {3, 5}, {7, 1}, {3, 99}}};
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&key), sizeof(key));
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto & [token, value] : followers) {
        out.write(reinterpret_cast<const char *>(&token), sizeof(token));
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    return static_cast<bool>(out);
}

bool test_merge_and_legacy_serialization() {
    const auto dir = std::filesystem::temp_directory_path() /
            ("ik-ngram-vector-io-" + std::to_string(ggml_time_us()));
    if (!std::filesystem::create_directory(dir)) {
        return require(false, "create exclusive vector serialization test directory");
    }
    const auto old_path = dir / "old-unsorted.cache";
    const auto roundtrip_path = dir / "roundtrip.cache";
    bool ok = write_legacy_unsorted_cache(old_path);
    ok &= require(ok, "write legacy unsorted follower sequence with a duplicate token");

    if (ok) {
        auto loaded = common_ngram_cache_load(old_path.string());
        const llama_tokens key_tokens = {44, 22};
        const auto part = loaded.find(common_ngram(key_tokens.data(), 2));
        ok &= require(part != loaded.end() && part->second.size() == 3 &&
                      part->second.find(3)->second == 5 && part->second.find(7)->second == 1 &&
                      part->second.find(9)->second == 2,
                      "legacy unsorted records load sorted; duplicate emplace retains the first count");
        common_ngram_cache_save(loaded, roundtrip_path.string());
        auto roundtrip = common_ngram_cache_load(roundtrip_path.string());
        ok &= require(same_cache(loaded, roundtrip), "save/load roundtrip retains exact logical contents");
    }

    const llama_tokens shared_key_tokens = {1, 2};
    const llama_tokens disjoint_key_tokens = {8, 9};
    auto target = make_follower_cache(shared_key_tokens, {{4, 2}, {8, 1}});
    auto source = make_follower_cache(shared_key_tokens, {{4, 3}, {6, 5}});
    source.emplace(common_ngram(disjoint_key_tokens.data(), 2), common_ngram_cache_part{});
    source.find(common_ngram(disjoint_key_tokens.data(), 2))->second.emplace(11, 7);
    const auto logical_source_before = source;
    common_ngram_cache_merge(target, source);
    const auto merged = target.find(common_ngram(shared_key_tokens.data(), 2));
    ok &= require(merged != target.end() && merged->second.size() == 3 &&
                  merged->second.find(4)->second == 5 && merged->second.find(6)->second == 5 &&
                  merged->second.find(8)->second == 1,
                  "merge accumulates overlaps and inserts unseen followers");
    const auto disjoint = target.find(common_ngram(disjoint_key_tokens.data(), 2));
    ok &= require(disjoint != target.end() && disjoint->second.find(11)->second == 7,
                  "merge copies a disjoint n-gram part");
    ok &= require(same_cache(source, logical_source_before),
                  "merge leaves its source logical contents unchanged");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    ok &= require(!ec, "remove only the serialization test-owned directory");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--stage-model") {
        return run_model_backed_stage_tests(argv[2]) ? 0 : 1;
    }
    if (argc != 1) {
        std::cerr << "usage: test-ngram-cache [--stage-model MODEL.gguf]\n";
        return 2;
    }

    bool ok = true;
    ok &= test_sorted_follower_part();
    ok &= test_incremental_updates();
    ok &= test_short_histories_and_growing_draft();
    ok &= test_reference_drafting();
    ok &= test_draft_budgets();
    ok &= test_lookup_path_plumbing();
    ok &= test_merge_and_legacy_serialization();
    if (!ok) {
        return 1;
    }
    std::cout << "PASS: sorted follower parts, update/merge/legacy serialization, short-history, budget, drafting and CLI/path checks\n";
    return 0;
}
