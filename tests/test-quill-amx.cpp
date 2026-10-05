// quillama.cpp: correctness test for the AMX BF16 GEMM (ggml/src/quill/quill-amx.cpp)
#include "ggml.h"
#include "../ggml/src/quill/quill-amx.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float silu(float x) { return x/(1.0f + expf(-x)); }

static int64_t interleave_of(ggml_type type) {
    const char * name = ggml_type_name(type);
    const char * p = strrchr(name, '_');
    if (!p || p[1] != 'r' || !p[2]) return 1;
    return atoi(p + 2);
}

static bool run_case(ggml_type type, int64_t K, int64_t M, int64_t T, int nth, bool up_gate) {
    std::mt19937 rng(1234 + K + M*7 + T*13 + type);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf(M*K), wg(M*K), x(T*K);
    for (auto & v : wf) v = nd(rng);
    for (auto & v : wg) v = nd(rng);
    for (auto & v : x)  v = nd(rng);

    const size_t row_size = ggml_row_size(type, K);
    std::vector<char> wq(row_size*M + 64), wgq(row_size*M + 64);
    std::vector<float> wd(M*K), wgd(M*K);
    auto quant = [&](const std::vector<float> & src, std::vector<char> & q, std::vector<float> & dq) {
        if (type == GGML_TYPE_F32) { memcpy(q.data(), src.data(), M*K*sizeof(float)); dq = src; return; }
        ggml_quantize_chunk(type, src.data(), q.data(), 0, M, K, nullptr, nullptr);
        auto tr = ggml_internal_get_type_traits(type);
        const int64_t R = interleave_of(type);
        for (int64_t m = 0; m < M; m += R) tr.to_float(q.data() + m*row_size, dq.data() + m*K, R*K);
    };
    quant(wf, wq, wd);
    if (up_gate) quant(wg, wgq, wgd);
    {   // the dequantized weights must resemble the originals (catches a wrong row-group layout)
        double e = 0, n = 0;
        for (int64_t i = 0; i < M*K; ++i) { e += (wd[i] - wf[i])*(wd[i] - wf[i]); n += wf[i]*wf[i]; }
        if (e > 0.05*n) { printf("%s: dequantized weights do not match the originals (rel %.3f)\n", ggml_type_name(type), e/n); return false; }
    }

    std::vector<char> packed(quill_amx_packed_size(K, T));
    for (int ith = 0; ith < nth; ++ith) quill_amx_pack(x.data(), K*sizeof(float), nullptr, K, T, packed.data(), ith, nth);

    std::vector<float> out(T*M, NAN);
    for (int ith = 0; ith < nth; ++ith) {
        if (up_gate) {
            if (!quill_amx_gemm_up_gate(type, wq.data(), wgq.data(), row_size, K, M, T, packed.data(), out.data(), M*sizeof(float), nullptr,
                                        GGML_UNARY_OP_SILU, 0.0f, ith, nth)) return false;
        } else {
            quill_amx_gemm(type, wq.data(), row_size, K, M, T, packed.data(), out.data(), M*sizeof(float), nullptr, ith, nth);
        }
    }

    double max_err = 0, max_ref = 0;
    for (int64_t t = 0; t < T; ++t) for (int64_t m = 0; m < M; ++m) {
        double up = 0, g = 0;
        for (int64_t k = 0; k < K; ++k) { up += (double)wd[m*K+k]*x[t*K+k]; if (up_gate) g += (double)wgd[m*K+k]*x[t*K+k]; }
        double ref = up_gate ? up*silu((float)g) : up;
        double v = out[t*M + m];
        if (std::isnan(v)) { printf("  NaN/unwritten at t=%lld m=%lld\n", (long long)t, (long long)m); return false; }
        max_err = std::max(max_err, fabs(v - ref));
        max_ref = std::max(max_ref, fabs(ref));
    }
    const double rel = max_err/(max_ref + 1e-9);
    const bool ok = rel < (up_gate ? 2e-2 : 1e-2);
    printf("%-8s K=%5lld M=%5lld T=%4lld nth=%d %s: rel max err %.2e %s\n", ggml_type_name(type), (long long)K, (long long)M,
           (long long)T, nth, up_gate ? "up_gate" : "gemm   ", rel, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    if (!quill_amx_available()) { printf("AMX not available - skipping\n"); return 0; }
    struct ggml_init_params ip = { 16*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip); // initializes the fp16 tables
    bool ok = true;
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K,
                                GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_KS };
    for (auto t : types) {
        ok &= run_case(t, 256, 64, 32, 1, false);
        ok &= run_case(t, 512, 96, 45, 3, false);
        ok &= run_case(t, 512, 50, 17, 4, true);
    }
    const ggml_type repacked[] = { GGML_TYPE_Q4_K_R4, GGML_TYPE_Q5_K_R4, GGML_TYPE_Q6_K_R4, GGML_TYPE_IQ4_XS_R8,
                                   GGML_TYPE_Q8_0_R8, GGML_TYPE_Q4_0_R8, GGML_TYPE_IQ4_KS_R4, GGML_TYPE_Q8_K_R8 };
    for (auto t : repacked) {
        ok &= run_case(t, 512, 64, 40, 3, false);
        ok &= run_case(t, 512, 48, 33, 4, true);
    }
    ok &= run_case(GGML_TYPE_F32, 40, 33, 5, 2, false);      // K, M, T not multiples of the tile sizes
    ok &= run_case(GGML_TYPE_Q4_K, 2048, 300, 130, 4, false);
    ok &= run_case(GGML_TYPE_Q4_K, 2048, 300, 130, 4, true);
    ggml_free(ctx);
    printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
