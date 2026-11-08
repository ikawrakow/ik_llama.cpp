#include "llama-moe-cache-share.h"
#include "llama-moe-cache-policy.h"
#include "llama.h"

#include <algorithm>
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
        const size_t occupied = std::count_if(default_lru.key_of.begin(), default_lru.key_of.end(),
                [](int32_t key) { return key >= 0; });
        if (default_lru.filled_real_slots != occupied || explicit_lru.filled_real_slots != occupied) {
            std::fprintf(stderr, "LRU occupancy\n"); return false;
        }
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
    if (classlru.filled_real_slots != 3) { std::fprintf(stderr, "ClassLRU occupancy\n"); return false; }
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
    if (empty.filled_real_slots != 0) { std::fprintf(stderr, "empty occupancy\n"); return false; }
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
        const int width = tokens == 1 ? 512 : 10;
        ggml_tensor * source = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, tokens);
        for (ggml_tensor * selected : {source, ggml_reshape_1d(ctx, source, width*tokens)}) {
            int64_t recovered_width = -1, recovered_tokens = -1;
            if (!llama_moe_cache_policy_detail::recover_selected_geometry(selected, 512,
                    recovered_width, recovered_tokens) || recovered_tokens != tokens || recovered_width != width) {
                std::fprintf(stderr, "token geometry %d width %d recovered %lldx%lld\n", tokens, width,
                        (long long) recovered_width, (long long) recovered_tokens); ok = false;
            }
        }
        if (llama_moe_cache_policy_detail::slot_map_eligible(tokens, width, 512, 1000, 32) !=
                (tokens <= 32 && tokens*width <= 1000)) {
            std::fprintf(stderr, "token eligibility %d\n", tokens); ok = false;
        }
    }
    ggml_free(ctx);
    return ok;
}

static bool check_cpu_share_plan() {
    using namespace llama_moe_cache_policy_detail;
    state cache;
    cache.init(1, 8, 3, kind::lru);
    std::vector<int32_t> slots(8, -1);
    cache.slot_map[0] = slots.data();
    state::cpu_share_plan plan;
    const int32_t seed[] = {0, 1, 5};
    if (!cache.plan_with_cpu_share(0, seed, 3, 1, 3, 0.5, 3, plan) ||
            !plan.cpu.empty() || cache.filled_real_slots != 3) {
        std::fprintf(stderr, "underfull share plan\n"); return false;
    }

    // Expert 2 repeats within token 0, so it stays on a real GPU row; experts 3 and 4
    // can share zero row 0 because they occur in different tokens.
    const int32_t routes[] = {2, 2, 3, 4};
    if (!cache.plan_with_cpu_share(0, routes, 4, 2, 2, 1.0, 3, plan) ||
            plan.cpu.size() != 2 || plan.fills.size() != 1 || plan.fills[0].expert != 2 ||
            plan.gpu_map[3] != 3 || plan.gpu_map[4] != 4 ||
            plan.cpu_map[3] != 3 || plan.cpu_map[4] != 4 ||
            plan.gpu_bias_map[3] != 8 || plan.cpu_bias_map[3] != 3 ||
            plan.gpu_bias_map[4] != 8 || plan.cpu_bias_map[4] != 4 ||
            cache.filled_real_slots != 3) {
        std::fprintf(stderr, "mixed CPU-share route/zero-slot assignment\n"); return false;
    }

    const int32_t all_hit[] = {2, 2};
    if (!cache.plan_with_cpu_share(0, all_hit, 2, 1, 2, 1.0, 3, plan) ||
            plan.n_hit != 1 || !plan.cpu.empty() || !plan.fills.empty()) {
        std::fprintf(stderr, "all-hit CPU-share plan\n"); return false;
    }

    state zero, ordinary;
    zero.init(1, 8, 2, kind::classlru);
    ordinary.init(1, 8, 2, kind::classlru);
    std::vector<int32_t> zero_slots(8, -1), ordinary_slots(8, -1);
    zero.slot_map[0] = zero_slots.data(); ordinary.slot_map[0] = ordinary_slots.data();
    const int32_t route[] = {0, 1};
    std::vector<state::fill> fills;
    size_t hits = 0, evictions = 0;
    if (!ordinary.plan(0, route, 3, 1, fills, hits, evictions) ||
            !zero.plan_with_cpu_share(0, route, 3, 1, 3, 0.0, 0, plan) ||
            zero_slots != ordinary_slots || zero.key_of != ordinary.key_of ||
            zero.filled_real_slots != ordinary.filled_real_slots || !plan.cpu.empty()) {
        std::fprintf(stderr, "share-zero ClassLRU parity\n"); return false;
    }

    bank_geometry geometry;
    if (!cpu_share_bank_geometry(7, 1, 3, geometry) || geometry.real_slots != 7 ||
            geometry.zero_slots != 3 || geometry.guard_slots != 1 || geometry.total_slots != 11 ||
            !cpu_share_bank_geometry(7, 0, 3, geometry) || geometry.zero_slots != 0 ||
            cpu_share_bank_geometry(INT32_MAX, 65536, 1, geometry)) {
        std::fprintf(stderr, "CPU-share bank geometry\n"); return false;
    }
    return true;
}

int main() {
    bool ok = true;
    if (!check_cache_policy()) { std::fprintf(stderr, "cache policy regression failed\n"); ok = false; }
    if (!check_token_geometry()) { std::fprintf(stderr, "token geometry regression failed\n"); ok = false; }
    if (!check_cpu_share_plan()) { std::fprintf(stderr, "CPU-share plan regression failed\n"); ok = false; }
    if (llama_context_default_params().moe_cache_policy != LLAMA_MOE_CACHE_POLICY_LRU) {
        std::fprintf(stderr, "default cache policy is not LRU\n"); ok = false;
    }
    if (llama_context_default_params().moe_cache_cpu_share != 0.0f) {
        std::fprintf(stderr, "default CPU share is not zero\n"); ok = false;
    }
    if (llama_moe_cache_policy_detail::valid((enum llama_moe_cache_policy) 7) ||
            llama_init_from_model(nullptr, [] { auto p = llama_context_default_params();
                p.moe_cache_policy = (enum llama_moe_cache_policy) 7; return p; }()) != nullptr) {
        std::fprintf(stderr, "invalid API cache policy was accepted\n"); ok = false;
    }
    for (float share : {NAN, INFINITY, -0.01f, 1.01f}) {
        if (llama_init_from_model(nullptr, [share] { auto p = llama_context_default_params();
                p.moe_cache_cpu_share = share; return p; }()) != nullptr) {
            std::fprintf(stderr, "invalid API CPU share was accepted\n"); ok = false;
        }
    }
    ok &= check("equal", {0.5f, 1.0f}, {true, true}, {0.5, 0.5});
    ok &= check("asymmetric", {0.25f, 1.0f}, {true, true}, {0.25, 0.75});
    ok &= check("single device", {1.0f}, {true}, {1.0});
    ok &= check("zero fallback", {0.0f, 0.0f}, {true, true}, {0.5, 0.5});
    ok &= check("ineligible device", {0.25f, 1.0f}, {false, true}, {0.0, 1.0});
    ok &= check("fallback with ineligible device", {0.0f, 0.0f}, {true, false}, {1.0, 0.0});
    return ok ? 0 : 1;
}
