#include "common.h"
#include "ngram-cache.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

bool check(bool ok, const char * what) {
    if (!ok) std::cerr << "FAIL: " << what << '\n';
    return ok;
}

common_ngram key(const llama_tokens & tokens) { return {tokens.data(), static_cast<int>(tokens.size())}; }

int count(const common_ngram_cache & cache, const llama_tokens & tokens, llama_token follower) {
    const auto part = cache.find(key(tokens));
    if (part == cache.end()) return 0;
    const auto item = part->second.find(follower);
    return item == part->second.end() ? 0 : item->second;
}

common_ngram_cache ingested(llama_tokens tokens, int low = 1, int high = 4) {
    common_ngram_cache result;
    common_ngram_cache_update(result, low, high, tokens, static_cast<int>(tokens.size()), false);
    return result;
}

bool test_updates_and_merge() {
    const llama_tokens stream = {1, 2, 1, 2, 1, 2};
    auto whole = ingested(stream, 1, 2);
    llama_tokens accumulated;
    common_ngram_cache incremental;
    for (const int n : {1, 2, 3}) {
        const int old = static_cast<int>(accumulated.size());
        accumulated.insert(accumulated.end(), stream.begin() + old,
                stream.begin() + old + n);
        common_ngram_cache_update(incremental, 1, 2, accumulated,
                static_cast<int>(accumulated.size()) - old, false);
    }
    if (!check(count(whole, {1}, 2) == 3 && count(whole, {2}, 1) == 2 &&
               count(whole, {1, 2}, 1) == 2 && count(whole, {2, 1}, 2) == 2 &&
               count(incremental, {1}, 2) == 3 && count(incremental, {2}, 1) == 2 &&
               count(incremental, {1, 2}, 1) == 2 && count(incremental, {2, 1}, 2) == 2,
               "whole and incremental ingestion count append-boundary n-grams once")) return false;

    auto target = ingested({1, 2, 1, 2}, 1, 2);
    auto source = ingested({1, 2, 1, 3, 8, 9}, 1, 2);
    common_ngram_cache_merge(target, source);
    return check(count(target, {1, 2}, 1) == 2 && count(target, {8}, 9) == 1,
                 "merge adds overlapping counts and unseen followers");
}

llama_tokens draft(llama_tokens history, int budget, common_ngram_cache & context,
        common_ngram_cache & dynamic, common_ngram_cache & statik, int low = 1, int high = 4) {
    llama_tokens result{history.empty() ? 99 : history.back()};
    common_ngram_cache_draft(history, result, budget, low, high, context, dynamic, statik);
    result.erase(result.begin()); // The initial token is the carry, not a proposal.
    return result;
}

bool test_history_budget_and_threshold() {
    common_ngram_cache empty;
    llama_tokens cycle;
    for (int i = 0; i < 64; ++i) cycle.insert(cycle.end(), {10, 11, 12, 13, 14});
    auto cache = ingested(cycle);
    if (!check(draft({}, 4, empty, cache, empty).empty() &&
               draft({10}, 4, empty, cache, empty).empty(), "empty and one-token histories are safe")) return false;
    const auto short_draft = draft({10, 11}, 4, empty, cache, empty);
    if (!check(short_draft == llama_tokens{12, 13, 14, 10} &&
               draft({10, 11, 12}, 1, empty, cache, empty) == llama_tokens{13} &&
               draft({10, 11, 12, 13}, 1, empty, cache, empty) == llama_tokens{14},
               "short histories use only available keys; proposals grow the history")) return false;
    for (const int limit : {0, 1, 16}) {
        if (!check(draft({10, 11}, limit, empty, cache, empty).size() == static_cast<size_t>(limit),
                   "draft budget excludes carry and is honored")) return false;
    }

    auto context = ingested({40, 41, 42, 40, 41, 43, 40, 41, 42, 40, 41, 43}, 2, 2);
    const auto threshold = draft({40, 41}, 1, context, empty, empty, 2, 4);
    return check(threshold.size() == 1 && (threshold[0] == 42 || threshold[0] == 43),
            "available 2-token key uses its 50-percent confidence threshold");
}

bool write_record(std::ofstream & out, const llama_tokens & tokens,
        const std::vector<std::pair<llama_token, int32_t>> & followers) {
    const auto ngram = key(tokens);
    const int32_t size = static_cast<int32_t>(followers.size());
    out.write(reinterpret_cast<const char *>(&ngram), sizeof(ngram));
    out.write(reinterpret_cast<const char *>(&size), sizeof(size));
    for (const auto & [token, value] : followers) {
        out.write(reinterpret_cast<const char *>(&token), sizeof(token));
        out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    }
    return static_cast<bool>(out);
}

bool legacy_counts(const common_ngram_cache & cache) {
    const auto part = cache.find(key({44, 22}));
    return part != cache.end() && part->second.size() == 3 &&
           part->second.find(3) != part->second.end() && part->second.find(3)->second == 5 &&
           part->second.find(7) != part->second.end() && part->second.find(7)->second == 1 &&
           part->second.find(9) != part->second.end() && part->second.find(9)->second == 2;
}

bool test_legacy_roundtrip() {
    const auto dir = std::filesystem::temp_directory_path() /
            ("ik-ngram-slim-" + std::to_string(ggml_time_us()));
    if (!std::filesystem::create_directory(dir)) return check(false, "create test-owned directory");
    const auto old = dir / "old.cache", saved = dir / "saved.cache";
    bool ok;
    {
        std::ofstream out(old, std::ios::binary);
        ok = write_record(out, {44, 22}, {{9, 2}, {3, 5}, {7, 1}, {3, 99}});
    }
    auto loaded = ok ? common_ngram_cache_load(old.string()) : common_ngram_cache{};
    ok &= check(legacy_counts(loaded), "unsorted legacy record keeps the first duplicate follower count");
    common_ngram_cache_save(loaded, saved.string());
    ok &= check(legacy_counts(common_ngram_cache_load(saved.string())), "logical legacy save/load roundtrip");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return check(ok && !ec, "remove test-owned files");
}

} // namespace

int main() {
    const bool ok = test_updates_and_merge() && test_history_budget_and_threshold() && test_legacy_roundtrip();
    if (ok) std::cout << "PASS: ngram update, draft boundary/budget, and legacy IO\n";
    return ok ? 0 : 1;
}
