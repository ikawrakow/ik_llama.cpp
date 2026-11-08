#include "llama-moe-cache-share.h"
#include "llama-moe-cache-policy.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool check(const char * name, const std::vector<float> & cumulative,
        const std::vector<bool> & eligible, const std::vector<double> & expected) {
    const auto actual = llama_moe_cache_detail::device_shares(cumulative, eligible);
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "%s: share count mismatch\n", name);
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-9) {
            std::fprintf(stderr, "%s: share %zu was %.12f, expected %.12f\n",
                    name, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool check_cache_policy() {
    using llama_moe_cache_policy_detail::kind;
    using llama_moe_cache_policy_detail::state;
    state default_lru, explicit_lru;
    default_lru.init(2, 8, 3, kind::lru);
    explicit_lru.init(2, 8, 3, kind::lru);
    std::vector<int32_t> maps_a(16, -1), maps_b(16, -1);
    default_lru.slot_map[0] = maps_a.data();
    explicit_lru.slot_map[0] = maps_b.data();
    const std::vector<std::vector<int32_t>> routes = {{1, 1, 2}, {2, 3}, {0, 3}, {4, 5}};
    for (const auto & route : routes) {
        std::vector<state::fill> a, b;
        size_t ah = 0, ae = 0, bh = 0, be = 0;
        if (!default_lru.plan(0, route.data(), route.size(), 1, a, ah, ae) ||
                !explicit_lru.plan(0, route.data(), route.size(), 1, b, bh, be) ||
                ah != bh || ae != be || a.size() != b.size() || maps_a != maps_b ||
                default_lru.key_of != explicit_lru.key_of) { std::fprintf(stderr, "LRU parity\n"); return false; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].expert != b[i].expert || a[i].slot != b[i].slot) { std::fprintf(stderr, "LRU fills\n"); return false; }
        }
    }

    state classlru;
    classlru.init(2, 8, 3, kind::classlru);
    std::vector<int32_t> layer0(8, -1), layer1(8, -1);
    classlru.slot_map[0] = layer0.data();
    classlru.slot_map[1] = layer1.data();
    auto plan = [&](int layer, std::vector<int32_t> ids, int tokens, size_t want_hits) {
        std::vector<state::fill> fills;
        size_t hits = 0, evictions = 0;
        return classlru.plan(layer, ids.data(), ids.size(), tokens, fills, hits, evictions) &&
                hits == want_hits && classlru.clock[layer] >= tokens;
    };
    if (!plan(0, {0, 1, 2}, 1, 0) || !plan(0, {0, 3}, 1, 1) || layer0[0] < 0 ||
            !plan(1, {0, 0}, 2, 0) || layer1[0] < 0 || classlru.key_of[layer1[0]] != 8 ||
            (layer0[0] >= 0 && classlru.key_of[layer0[0]] != 0)) { std::fprintf(stderr, "ClassLRU routes/maps\n"); return false; }
    std::vector<int32_t> before_key = classlru.key_of;
    std::vector<int32_t> before0 = layer0;
    std::vector<state::fill> fills;
    size_t hits = 0, evictions = 0;
    const int32_t too_many[] = {4, 5, 6, 7};
    if (classlru.plan(0, too_many, 4, 1, fills, hits, evictions) ||
            before_key != classlru.key_of || before0 != layer0) { std::fprintf(stderr, "failed-plan mutation\n"); return false; }
    if (classlru.crf[0] < 1.0f || classlru.clock[0] != 2 || classlru.clock[1] != 2) { std::fprintf(stderr, "ClassLRU clock/score\n"); return false; }
    state empty;
    empty.init(1, 8, 0, kind::classlru);
    empty.slot_map[0] = layer0.data();
    if (empty.plan(0, too_many, 4, 1, fills, hits, evictions)) { std::fprintf(stderr, "empty cache accepted fill\n"); return false; }
    return default_lru.crf.empty() && default_lru.last_use.empty() &&
            default_lru.clock.empty() && classlru.crf.size() == 16;
}

static bool check_token_geometry() {
    ggml_init_params params = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    bool ok = true;
    for (int tokens : {1, 2, 8, 32, 33}) {
        const int width = tokens == 1 ? 512 : 10*tokens;
        ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, tokens);
        ggml_tensor * selected = ggml_reshape_1d(ctx, source, width*tokens);
        int64_t recovered = -1;
        if (!llama_moe_cache_policy_detail::recover_selected_tokens(selected, 10, 512, recovered) || recovered != tokens) {
            std::fprintf(stderr, "token geometry %d recovered %lld\n", tokens, (long long) recovered); ok = false;
        }
        if (llama_moe_cache_policy_detail::slot_map_eligible(tokens, 10, 512, 1000, 32) != (tokens <= 32)) {
            std::fprintf(stderr, "token eligibility %d\n", tokens); ok = false;
        }
    }
    ggml_free(ctx);
    return ok;
}

int main() {
    bool ok = true;
    if (!check_cache_policy()) { std::fprintf(stderr, "cache policy regression failed\n"); ok = false; }
    if (!check_token_geometry()) { std::fprintf(stderr, "token geometry regression failed\n"); ok = false; }
    if (llama_context_default_params().moe_cache_policy != LLAMA_MOE_CACHE_POLICY_LRU) {
        std::fprintf(stderr, "default cache policy is not LRU\n"); ok = false;
    }
    if (llama_moe_cache_policy_detail::valid((enum llama_moe_cache_policy) 7) ||
            llama_init_from_model(nullptr, [] { auto p = llama_context_default_params();
                p.moe_cache_policy = (enum llama_moe_cache_policy) 7; return p; }()) != nullptr) {
        std::fprintf(stderr, "invalid API cache policy was accepted\n"); ok = false;
    }
    ok &= check("equal", {0.5f, 1.0f}, {true, true}, {0.5, 0.5});
    ok &= check("asymmetric", {0.25f, 1.0f}, {true, true}, {0.25, 0.75});
    ok &= check("single device", {1.0f}, {true}, {1.0});
    ok &= check("zero fallback", {0.0f, 0.0f}, {true, true}, {0.5, 0.5});
    ok &= check("ineligible device", {0.25f, 1.0f}, {false, true}, {0.0, 1.0});
    ok &= check("fallback with ineligible device", {0.0f, 0.0f}, {true, false}, {1.0, 0.0});
    return ok ? 0 : 1;
}
