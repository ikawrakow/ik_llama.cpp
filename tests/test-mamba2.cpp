// Mamba-2 shared-op self-consistency: a chunked prefill must match the same tokens decoded one
// at a time, for the conv (rolling window) and the scan (state snapshots).

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d: FAILED: %s\n", __FILE__, __LINE__, #cond); \
        n_fail++; \
    } \
} while (0)

// NaN must not be swallowed by the maxima: `std::max(0.0, NaN)` returns 0.0, so a kernel that
// emits NaN would pass every tolerance below. Treat a NaN difference as infinitely bad.
static double worst_of(double acc, double d) {
    return (d != d) ? std::numeric_limits<double>::infinity() : (d > acc ? d : acc);
}

// deterministic pseudo-random values
static float rnd(int64_t i, float scale) {
    return scale * std::sin(0.37f * (float) i + 0.11f) + 0.3f * scale * std::cos(0.91f * (float) i);
}

// ---------------------------------------------------------------- mamba-2 conv (2-arg form)

struct conv_case {
    int64_t d_conv = 4;
    int64_t ch     = 8; // d_inner + 2*n_group*d_state in the real graph
    int64_t n_tok  = 5;
    int64_t n_s    = 1;

    std::vector<float> w;      // {d_conv, ch}
    std::vector<float> state0; // {d_conv-1, ch, n_s}
    std::vector<float> x;      // {n_tok, ch, n_s}

    void init() {
        w.resize((size_t) (d_conv * ch));
        state0.resize((size_t) ((d_conv - 1) * ch * n_s));
        x.resize((size_t) (n_tok * ch * n_s));
        for (size_t i = 0; i < w.size(); ++i)      w[i]      = rnd((int64_t) i, 0.5f);
        for (size_t i = 0; i < state0.size(); ++i) state0[i] = rnd((int64_t) i + 1000, 1.0f);
        for (size_t i = 0; i < x.size(); ++i)      x[i]      = rnd((int64_t) i + 2000, 1.0f);
    }

    // runs n tokens starting from `state`; returns the output and the updated state
    bool run(int64_t n_tok_run, const std::vector<float> & state_in,
             std::vector<float> & y_out, std::vector<float> & state_out) const {
        struct ggml_init_params ip = { 4 * 1024 * 1024, nullptr, false };
        struct ggml_context * ctx = ggml_init(ip);

        struct ggml_tensor * t_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, ch);
        struct ggml_tensor * t_s = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1, ch, n_s);
        memcpy(t_w->data, w.data(), w.size() * sizeof(float));
        memcpy(t_s->data, state_in.data(), state_in.size() * sizeof(float));

        // conv_x = concat(state {d_conv-1, ch}, transpose(xBC {ch, n_tok}), dim 0)
        // (xBC is channel-major in the graph: element [ch][tok] at tok*ch + ch)
        struct ggml_tensor * t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ch, n_tok_run);
        memcpy(t_x->data, x.data(), (size_t) (n_tok_run * ch * n_s) * sizeof(float));
        struct ggml_tensor * conv_x = ggml_concat(ctx, t_s, ggml_transpose(ctx, t_x), 0);

        struct ggml_tensor * y = ggml_ssm_conv_m2(ctx, conv_x, t_w);

        // the graph slides the window with a strided view + cpy (the last d_conv-1 positions of
        // every channel are not contiguous), so mirror that here
        struct ggml_tensor * last = ggml_view_3d(ctx, conv_x, d_conv - 1, ch, n_s,
                conv_x->nb[1], conv_x->nb[2], (size_t) n_tok_run * conv_x->nb[0]);
        struct ggml_tensor * next_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (d_conv - 1) * ch * n_s);
        struct ggml_tensor * store = ggml_cpy(ctx, last, next_state);

        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_build_forward_expand(g, store);
        ggml_graph_compute_with_ctx(ctx, g, 2);

        y_out.assign((size_t) (n_tok_run * ch * n_s), 0.0f);
        memcpy(y_out.data(), y->data, y_out.size() * sizeof(float));

        state_out.assign((size_t) ((d_conv - 1) * ch * n_s), 0.0f);
        memcpy(state_out.data(), next_state->data, state_out.size() * sizeof(float));

        ggml_free(ctx);
        return true;
    }
};

static void test_mamba2_conv(int64_t d_conv) {
    conv_case c;
    c.d_conv = d_conv;
    c.init();

    std::vector<float> y_all, s_all;
    c.run(c.n_tok, c.state0, y_all, s_all);

    std::vector<float> y_chain, state = c.state0;
    for (int64_t t = 0; t < c.n_tok; ++t) {
        conv_case one = c;                       // same weights
        one.n_tok = 1;
        one.x.assign(c.x.begin() + (size_t) (t * c.ch * c.n_s),
                     c.x.begin() + (size_t) ((t + 1) * c.ch * c.n_s));
        std::vector<float> y1, s1;
        one.run(1, state, y1, s1);
        y_chain.insert(y_chain.end(), y1.begin(), y1.end());
        state = s1;
    }

    double worst = 0.0;
    for (size_t i = 0; i < y_all.size(); ++i) {
        worst = worst_of(worst, std::fabs((double) y_all[i] - (double) y_chain[i]));
    }
    CHECK(worst < 1e-6);

    double worst_s = 0.0;
    for (size_t i = 0; i < s_all.size(); ++i) {
        worst_s = worst_of(worst_s, std::fabs((double) s_all[i] - (double) state[i]));
    }
    CHECK(worst_s < 1e-6);

    printf("mamba-2 conv d_conv=%lld: prefill vs token-by-token, worst |dy| %.2e, worst |dstate| %.2e\n",
            (long long) d_conv, worst, worst_s);
}

// ---------------------------------------------------------------- mamba-2 scan (K >= 1)

struct scan_case {
    int64_t d_state  = 8;
    int64_t head_dim = 4;
    int64_t n_head   = 2;
    int64_t n_group  = 2;
    int64_t n_tok    = 5;
    int64_t n_seq    = 1;

    std::vector<float> s0;  // {d_state, head_dim, n_head, slots}
    std::vector<float> x;   // {head_dim, n_head, n_tok, n_seq}
    std::vector<float> dt;  // {n_head, n_tok, n_seq}
    std::vector<float> A;   // {1, n_head}
    std::vector<float> B;   // {d_state, n_group, n_tok, n_seq}
    std::vector<float> C;   // same

    void init() {
        const int64_t slots = 1;
        s0.resize((size_t) (d_state * head_dim * n_head * slots));
        x.resize((size_t) (head_dim * n_head * n_tok * n_seq));
        dt.resize((size_t) (n_head * n_tok * n_seq));
        A.resize((size_t) n_head);
        B.resize((size_t) (d_state * n_group * n_tok * n_seq));
        C.resize((size_t) (d_state * n_group * n_tok * n_seq));

        for (size_t i = 0; i < s0.size(); ++i) s0[i] = rnd((int64_t) i + 3000, 0.5f);
        for (size_t i = 0; i < x.size(); ++i)  x[i]  = rnd((int64_t) i + 4000, 1.0f);
        // dt must stay in a range where exp(dt*A) is well conditioned
        for (size_t i = 0; i < dt.size(); ++i) dt[i] = 0.1f + 0.4f * std::fabs(rnd((int64_t) i + 5000, 1.0f));
        for (int64_t h = 0; h < n_head; ++h)   A[(size_t) h] = -0.2f - 0.1f * (float) h;
        for (size_t i = 0; i < B.size(); ++i) B[i] = rnd((int64_t) i + 6000, 1.0f);
        for (size_t i = 0; i < C.size(); ++i) C[i] = rnd((int64_t) i + 7000, 1.0f);
    }

    bool run(int64_t n_tok_run, const std::vector<float> & state_in,
             std::vector<float> & y_out, std::vector<float> & state_out) const {
        struct ggml_init_params ip = { 8 * 1024 * 1024, nullptr, false };
        struct ggml_context * ctx = ggml_init(ip);

        const int64_t slots = 1;
        struct ggml_tensor * t_s  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, slots);
        struct ggml_tensor * t_x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_tok_run, n_seq);
        struct ggml_tensor * t_dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_tok_run, n_seq);
        struct ggml_tensor * t_A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head);
        struct ggml_tensor * t_B  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_tok_run, n_seq);
        struct ggml_tensor * t_C  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_tok_run, n_seq);
        struct ggml_tensor * t_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq);

        memcpy(t_s->data,  state_in.data(), state_in.size() * sizeof(float));
        memcpy(t_A->data,  A.data(), A.size() * sizeof(float));
        memcpy(t_x->data,  x.data(),  (size_t) (head_dim * n_head * n_tok_run * n_seq) * sizeof(float));
        memcpy(t_dt->data, dt.data(), (size_t) (n_head * n_tok_run * n_seq) * sizeof(float));
        // B/C are {d_state, n_group, n_tok_run, n_seq}; the first n_tok_run tokens are the same
        // prefix as the full-size tensors because n_tok is the third dimension
        memcpy(t_B->data, B.data(), (size_t) (d_state * n_group * n_tok_run * n_seq) * sizeof(float));
        memcpy(t_C->data, C.data(), (size_t) (d_state * n_group * n_tok_run * n_seq) * sizeof(float));
        for (int64_t i = 0; i < n_seq; ++i) {
            ((int32_t *) t_id->data)[i] = 0;
        }

        struct ggml_tensor * y = ggml_ssm_scan_m2(ctx, t_s, t_x, t_dt, t_A, t_B, t_C, t_id, /*K=*/1);

        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_graph_compute_with_ctx(ctx, g, 2);

        const int64_t n_y = head_dim * n_head * n_tok_run * n_seq;
        y_out.assign((size_t) n_y, 0.0f);
        memcpy(y_out.data(), y->data, y_out.size() * sizeof(float));

        const int64_t n_state = d_state * head_dim * n_head * slots;
        state_out.assign((size_t) n_state, 0.0f);
        memcpy(state_out.data(), (const char *) y->data + (size_t) n_y * sizeof(float),
               state_out.size() * sizeof(float));

        ggml_free(ctx);
        return true;
    }
};

static void test_mamba2_scan(int64_t d_state, int64_t n_head, int64_t n_group) {
    scan_case c;
    c.d_state = d_state;
    c.n_head  = n_head;
    c.n_group = n_group;
    c.init();

    std::vector<float> y_all, s_all;
    c.run(c.n_tok, c.s0, y_all, s_all);

    std::vector<float> y_chain, state = c.s0;
    for (int64_t t = 0; t < c.n_tok; ++t) {
        scan_case one = c;
        const size_t nx  = (size_t) (c.head_dim * c.n_head);
        const size_t nd  = (size_t) c.n_head;
        const size_t nb  = (size_t) (c.d_state * c.n_group);
        one.x.assign(c.x.begin() + (size_t) t * nx, c.x.begin() + (size_t) (t + 1) * nx);
        one.dt.assign(c.dt.begin() + (size_t) t * nd, c.dt.begin() + (size_t) (t + 1) * nd);
        one.B.assign(c.B.begin() + (size_t) t * nb, c.B.begin() + (size_t) (t + 1) * nb);
        one.C.assign(c.C.begin() + (size_t) t * nb, c.C.begin() + (size_t) (t + 1) * nb);
        std::vector<float> y1, s1;
        one.run(1, state, y1, s1);
        y_chain.insert(y_chain.end(), y1.begin(), y1.end());
        state = s1;
    }

    double worst = 0.0;
    for (size_t i = 0; i < y_all.size(); ++i) {
        worst = worst_of(worst, std::fabs((double) y_all[i] - (double) y_chain[i]));
    }
    double worst_s = 0.0;
    for (size_t i = 0; i < s_all.size(); ++i) {
        worst_s = worst_of(worst_s, std::fabs((double) s_all[i] - (double) state[i]));
    }
    // the recurrence is evaluated in fp32 but the prefill path may reorder the accumulation
    const double scale = 1.0 + std::fabs((double) *std::max_element(y_all.begin(), y_all.end()));
    CHECK(worst   < 1e-4 * scale);
    CHECK(worst_s < 1e-4 * (1.0 + std::fabs((double) *std::max_element(s_all.begin(), s_all.end()))));

    printf("mamba-2 scan d_state=%lld n_head=%lld n_group=%lld: prefill vs token-by-token, "
           "worst |dy| %.2e, worst |dstate| %.2e\n",
            (long long) d_state, (long long) n_head, (long long) n_group, worst, worst_s);
}


// ---------------------------------------------------------------- mamba-2 scan snapshots (K > 1)
//
// The per-step checkpoint path in llama.cpp rewinds a partially accepted speculative block by
// restoring a state snapshot the scan wrote into its result. That only works if a snapshot really
// is the state after N tokens and if resuming from it reproduces the tail. This test measures the
// mapping instead of assuming it: it runs the full sequence with K = n_tok, then checks every
// snapshot against the state of a plain prefix run, and finally resumes from one.

struct scan_snap {
    int64_t d_state  = 8;
    int64_t head_dim = 4;
    int64_t n_head   = 2;
    int64_t n_group  = 2;
    int64_t n_tok    = 6;
    int64_t n_seq    = 1;

    std::vector<float> s0, x, dt, A, B, C;

    void init() {
        const int64_t slots = 1;
        s0.resize((size_t) (d_state * head_dim * n_head * slots));
        x.resize((size_t) (head_dim * n_head * n_tok * n_seq));
        dt.resize((size_t) (n_head * n_tok * n_seq));
        A.resize((size_t) n_head);
        B.resize((size_t) (d_state * n_group * n_tok * n_seq));
        C.resize((size_t) (d_state * n_group * n_tok * n_seq));
        for (size_t i = 0; i < s0.size(); ++i) s0[i] = rnd((int64_t) i + 3000, 0.5f);
        for (size_t i = 0; i < x.size(); ++i)  x[i]  = rnd((int64_t) i + 4000, 1.0f);
        for (size_t i = 0; i < dt.size(); ++i) dt[i] = 0.1f + 0.4f * std::fabs(rnd((int64_t) i + 5000, 1.0f));
        for (int64_t h = 0; h < n_head; ++h)   A[(size_t) h] = -0.2f - 0.1f * (float) h;
        for (size_t i = 0; i < B.size(); ++i) B[i] = rnd((int64_t) i + 6000, 1.0f);
        for (size_t i = 0; i < C.size(); ++i) C[i] = rnd((int64_t) i + 7000, 1.0f);
    }

    // runs tokens [k0, k0 + n_run) starting from state_in, keeping K snapshots in the result
    void run(int64_t k0, int64_t n_run, const std::vector<float> & state_in, int64_t K,
             std::vector<float> & y_out, std::vector<std::vector<float>> & snaps_out) const {
        struct ggml_init_params ip = { 16 * 1024 * 1024, nullptr, false };
        struct ggml_context * ctx = ggml_init(ip);

        const int64_t slots = 1;
        ggml_tensor * t_s  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, slots);
        ggml_tensor * t_x  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_run, n_seq);
        ggml_tensor * t_dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_run, n_seq);
        ggml_tensor * t_A  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_head);
        ggml_tensor * t_B  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_run, n_seq);
        ggml_tensor * t_C  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_run, n_seq);
        ggml_tensor * t_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq);

        const size_t nx = (size_t) (head_dim * n_head);
        const size_t nd = (size_t) n_head;
        const size_t nb = (size_t) (d_state * n_group);

        memcpy(t_s->data, state_in.data(), state_in.size() * sizeof(float));
        memcpy(t_A->data, A.data(), A.size() * sizeof(float));
        memcpy(t_x->data,  x.data()  + (size_t) k0 * nx, (size_t) n_run * nx * sizeof(float));
        memcpy(t_dt->data, dt.data() + (size_t) k0 * nd, (size_t) n_run * nd * sizeof(float));
        memcpy(t_B->data,  B.data()  + (size_t) k0 * nb, (size_t) n_run * nb * sizeof(float));
        memcpy(t_C->data,  C.data()  + (size_t) k0 * nb, (size_t) n_run * nb * sizeof(float));
        for (int64_t i = 0; i < n_seq; ++i) ((int32_t *) t_id->data)[i] = 0;

        ggml_tensor * y = ggml_ssm_scan_m2(ctx, t_s, t_x, t_dt, t_A, t_B, t_C, t_id, K);
        ggml_cgraph * g = ggml_new_graph(ctx);
        ggml_build_forward_expand(g, y);
        ggml_graph_compute_with_ctx(ctx, g, 2);

        const int64_t n_y     = head_dim * n_head * n_run * n_seq;
        const int64_t n_state = d_state * head_dim * n_head * n_seq;
        y_out.assign((size_t) n_y, 0.0f);
        memcpy(y_out.data(), y->data, y_out.size() * sizeof(float));
        snaps_out.assign((size_t) K, {});
        for (int64_t k = 0; k < K; ++k) {
            snaps_out[(size_t) k].assign((size_t) n_state, 0.0f);
            memcpy(snaps_out[(size_t) k].data(), (const char *) y->data + ((size_t) n_y + (size_t) k * n_state) * sizeof(float),
                   (size_t) n_state * sizeof(float));
        }
        ggml_free(ctx);
    }
};

static void test_mamba2_scan_snapshots(int64_t d_state, int64_t n_head, int64_t n_group) {
    scan_snap c;
    c.d_state = d_state;
    c.n_head  = n_head;
    c.n_group = n_group;
    c.init();

    std::vector<float> y_full;
    std::vector<std::vector<float>> snaps;
    c.run(0, c.n_tok, c.s0, c.n_tok, y_full, snaps);

    // every snapshot must equal the state of a plain prefix run over the same number of tokens
    int matched = 0;
    double worst = 0.0;
    for (int64_t s = 1; s < c.n_tok; ++s) {
        std::vector<float> y_pre, dummy;
        std::vector<std::vector<float>> one;
        c.run(0, s, c.s0, 1, y_pre, one);   // state after s tokens (K=1 puts it first)
        const std::vector<float> & state_s = one[0];

        // find which snapshot equals it, rather than assuming the writer's mapping
        int64_t found = -1;
        for (int64_t k = 0; k < c.n_tok; ++k) {
            double d = 0.0;
            for (size_t i = 0; i < state_s.size(); ++i) {
                d = std::max(d, std::fabs((double) state_s[i] - (double) snaps[(size_t) k][i]));
            }
            if (d < 1e-6 * (1.0 + std::fabs((double) state_s[0]))) { found = k; break; }
        }
        if (s == 1) printf("mamba-2 scan: state after 1 token is snapshot index %lld\n", (long long) found);
        // the writer stores the state after N tokens at index n_tok - N; pin it, otherwise any
        // permutation of the snapshots would pass
        CHECK(found == c.n_tok - s);
        if (found >= 0) ++matched;

        // and resuming from that snapshot must reproduce the tail exactly
        if (found >= 0) {
            std::vector<float> y_tail;
            std::vector<std::vector<float>> tmp;
            c.run(s, c.n_tok - s, snaps[(size_t) found], 1, y_tail, tmp);
            const int64_t n_tok_tail = c.n_tok - s;
            const size_t per_tok = (size_t) (c.head_dim * c.n_head);
            double worst_tail = 0.0;
            for (size_t i = 0; i < y_tail.size(); ++i) {
                worst_tail = worst_of(worst_tail, std::fabs((double) y_tail[i] - (double) y_full[(size_t) s * per_tok + i]));
            }
            (void) n_tok_tail;
            worst = std::max(worst, worst_tail);
            CHECK(worst_tail < 1e-4 * (1.0 + std::fabs((double) y_full[(size_t) s * per_tok])));
        }
    }
    printf("mamba-2 scan snapshots d_state=%lld n_head=%lld n_group=%lld: %d/%lld prefixes matched, "
           "worst resumed |dy| %.2e\n",
            (long long) d_state, (long long) n_head, (long long) n_group,
            matched, (long long) (c.n_tok - 1), worst);
}

int main() {
    // the CUDA conv kernel instantiates d_conv 3/4/5/9/15; the CPU reference must agree with
    // token-by-token decode for all of them
    for (int64_t d_conv : { 3, 4, 5 }) {
        test_mamba2_conv(d_conv);
    }

    // n_head == n_group is the identity mapping; n_head = 4, n_group = 2 exercises the
    // head -> group mapping. d_state 96 is a shape the CUDA scan kernel instantiates.
    test_mamba2_scan(8, 2, 2);
    test_mamba2_scan(8, 4, 2);
    test_mamba2_scan(96, 4, 2);

    test_mamba2_scan_snapshots(8, 2, 2);
    test_mamba2_scan_snapshots(96, 4, 2);

    if (n_fail == 0) {
        printf("mamba-2: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "mamba-2: %d check(s) failed\n", n_fail);
    return 1;
}
