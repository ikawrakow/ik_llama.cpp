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

static bool run_fa_case(int64_t D, int64_t N, int64_t n_kv, int64_t H, int64_t H_kv, ggml_type kvt, bool causal, float softcap, int nth) {
    struct ggml_init_params ip = { 512*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(42 + D + N + n_kv);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, N, H);
    ggml_tensor * kf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_kv, H_kv);
    ggml_tensor * vf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_kv, H_kv);
    for (int64_t i = 0; i < ggml_nelements(q); ++i)  ((float *)q->data)[i]  = nd(rng);
    for (int64_t i = 0; i < ggml_nelements(kf); ++i) ((float *)kf->data)[i] = nd(rng);
    for (int64_t i = 0; i < ggml_nelements(vf); ++i) ((float *)vf->data)[i] = nd(rng);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, kvt, D, n_kv, H_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, kvt, D, n_kv, H_kv);
    ggml_quantize_chunk(kvt, (const float *)kf->data, k->data, 0, n_kv*H_kv, D, nullptr, nullptr);
    ggml_quantize_chunk(kvt, (const float *)vf->data, v->data, 0, n_kv*H_kv, D, nullptr, nullptr);
    // reference uses the stored (rounded) K/V values
    std::vector<float> kd(D*n_kv*H_kv), vd(D*n_kv*H_kv);
    auto tr = ggml_internal_get_type_traits(kvt);
    tr.to_float(k->data, kd.data(), D*n_kv*H_kv);
    tr.to_float(v->data, vd.data(), D*n_kv*H_kv);
    const int64_t n_kv_pad = GGML_PAD(n_kv, 64), N_pad = GGML_PAD(N, 32);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_pad, N_pad);
    const int64_t past = n_kv - N;
    for (int64_t i = 0; i < N_pad; ++i) for (int64_t j = 0; j < n_kv_pad; ++j) {
        const bool ok = j < n_kv && (!causal || j <= past + i);
        ((ggml_fp16_t *)mask->data)[i*n_kv_pad + j] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
    }
    const float scale = 1.0f/sqrtf((float)D);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, softcap);
    if (!quill_amx_flash_attn_eligible(out)) { printf("FA case not eligible\n"); ggml_free(ctx); return false; }
    for (int ith = 0; ith < nth; ++ith) quill_amx_flash_attn(out, ith, nth);

    double max_err = 0, max_ref = 0;
    std::vector<double> sc(n_kv);
    for (int64_t h = 0; h < H; ++h) {
        const int64_t hk = h/(H/H_kv);
        for (int64_t i = 0; i < N; ++i) {
            const float * qi = (const float *)q->data + (h*N + i)*D;
            double mx = -INFINITY;
            for (int64_t j = 0; j < n_kv; ++j) {
                double s = 0;
                for (int64_t d = 0; d < D; ++d) s += (double)qi[d]*kd[(hk*n_kv + j)*D + d];
                s *= scale;
                if (softcap != 0.0f) s = softcap*tanh(s/softcap);
                const bool ok = !causal || j <= past + i;
                sc[j] = ok ? s : -INFINITY;
                mx = std::max(mx, sc[j]);
            }
            double l = 0;
            for (int64_t j = 0; j < n_kv; ++j) { sc[j] = std::isinf(sc[j]) ? 0.0 : exp(sc[j] - mx); l += sc[j]; }
            const float * o = (const float *)out->data + (i*H + h)*D;
            for (int64_t d = 0; d < D; ++d) {
                double r = 0;
                for (int64_t j = 0; j < n_kv; ++j) r += sc[j]*vd[(hk*n_kv + j)*D + d];
                r /= l;
                max_err = std::max(max_err, fabs(o[d] - r));
                max_ref = std::max(max_ref, fabs(r));
            }
        }
    }
    const double rel = max_err/(max_ref + 1e-9);
    const bool ok = rel < 2e-2;
    printf("FA D=%lld N=%lld n_kv=%lld H=%lld/%lld %-5s causal=%d softcap=%.0f nth=%d: rel max err %.2e %s\n",
           (long long)D, (long long)N, (long long)n_kv, (long long)H, (long long)H_kv, ggml_type_name(kvt), causal, softcap, nth, rel, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok;
}

int main() {
    if (!quill_amx_available()) { printf("AMX not available - skipped\n"); return 77; }  // ctest SKIP_RETURN_CODE
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
    ok &= run_fa_case(128, 64,  64,  4, 4, GGML_TYPE_F16, true,  0.0f, 1);
    ok &= run_fa_case(128, 45,  300, 8, 2, GGML_TYPE_F16, true,  0.0f, 3);
    ok &= run_fa_case(64,  100, 700, 4, 1, GGML_TYPE_Q8_0, true, 0.0f, 4);
    ok &= run_fa_case(256, 40,  90,  2, 2, GGML_TYPE_BF16, false, 0.0f, 2);
    ok &= run_fa_case(128, 33,  33,  2, 2, GGML_TYPE_F16, true, 30.0f, 2);
    ggml_free(ctx);
    printf(ok ? "ALL OK\n" : "FAILURES\n");
    return ok ? 0 : 1;
}
