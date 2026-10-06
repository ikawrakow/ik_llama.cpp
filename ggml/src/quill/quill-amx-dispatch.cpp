// quillama.cpp: AMX dispatch - CPU detection and eligibility checks.
//
// These run on every CPU (AMD, older Intel, ...) before any AMX kernel, so this file is compiled
// with the default flags. The kernels in quill-amx.cpp are compiled for AVX-512 + AMX and are only
// called once quill_amx_available() returned true.

#include "quill-amx.h"

#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && defined(__linux__)
#define QUILL_AMX_BUILD 1
#else
#define QUILL_AMX_BUILD 0
#endif

#if QUILL_AMX_BUILD

#include <cpuid.h>
#include <unistd.h>
#include <sys/syscall.h>

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA  18

namespace {

bool cpu_has_amx_bf16() {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    const bool amx_bf16 = edx & (1u << 22);
    const bool amx_tile = edx & (1u << 24);
    const bool avx512f  = ebx & (1u << 16);
    const bool avx512bw = ebx & (1u << 30);
    if (!(amx_bf16 && amx_tile && avx512f && avx512bw)) return false;
    if (!__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) return false;
    const bool avx512_bf16 = eax & (1u << 5);
    return avx512_bf16;
}

bool request_tile_permission() {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) == 0;
}

} // namespace

static int g_amx_state = -1;   // -1 unknown, 0 off, 1 on

extern "C" bool quill_amx_available(void) {
    if (g_amx_state < 0) {
        const char * env = getenv("QUILL_AMX");
        bool on = !(env && (strcmp(env, "0") == 0 || strcmp(env, "off") == 0));
        on = on && cpu_has_amx_bf16() && request_tile_permission();
        g_amx_state = on ? 1 : 0;
    }
    return g_amx_state == 1;
}

extern "C" int quill_amx_min_batch(void) {
    static int n = [] {
        const char * env = getenv("QUILL_AMX_MIN_BATCH");
        int v = env ? atoi(env) : 32;
        return v > 0 ? v : 32;
    }();
    return n;
}

// rows interleaved by a repacked type: the _r<N> suffix of its name (q4_k_r4, iq4_xs_r8, ...), else 1
extern "C" int64_t quill_amx_interleave(enum ggml_type type) {
    const char * name = ggml_type_name(type);
    if (!name) return 1;
    const char * p = strrchr(name, '_');
    if (!p || (p[1] != 'r' && p[1] != 'R') || !p[2]) return 1;
    for (const char * c = p + 2; *c; ++c) if (*c < '0' || *c > '9') return 1;
    const int64_t r = atoi(p + 2);
    return r == 4 || r == 8 || r == 16 ? r : 1;
}

extern "C" bool quill_amx_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) return true;
    if (!ggml_is_quantized(type)) return false;
    const auto traits = ggml_internal_get_type_traits(type);
    if (!traits.to_float) return false;
    // row-wise meta data (e.g. IQ4_KS) is fine: whole rows (or row groups) are always dequantized
    if (getenv("QUILL_AMX_NO_REPACKED") && quill_amx_interleave(type) > 1) return false;
    return true;
}

extern "C" size_t quill_amx_packed_size(int64_t K, int64_t T) {
    return (size_t)((T + 31)/32*32) * (size_t)((K + 31)/32*32) * sizeof(ggml_bf16_t) + 64;
}

extern "C" bool quill_amx_flash_attn_eligible(const struct ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * mask = dst->src[3];
    if (dst->src[4] || dst->src[5]) return false;                       // sinks / extra inputs
    if (dst->op_params[4] != 0) return false;                            // backend hints
    float max_bias; memcpy(&max_bias, (const float *)dst->op_params + 1, sizeof(float));
    if (max_bias != 0.0f) return false;                                  // ALiBi
    if (q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    auto kv_ok = [](ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_F32; };
    if (!kv_ok(k->type) || !kv_ok(v->type)) return false;
    const int64_t D = q->ne[0];
    if (k->ne[0] != D || v->ne[0] != D || D % 32 != 0 || D > 512) return false;
    if (q->ne[1] < quill_amx_min_batch()) return false;
    if (q->ne[2] % k->ne[2] != 0 || k->ne[2] != v->ne[2] || k->ne[1] != v->ne[1]) return false;
    if (q->ne[3] % k->ne[3] != 0 || q->ne[3] % v->ne[3] != 0) return false;
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1])) return false;
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) || v->nb[0] != ggml_type_size(v->type)) return false;
    if (dst->ne[0] != D || dst->ne[1] != q->ne[2] || dst->ne[2] != q->ne[1]) return false;
    return quill_amx_available();
}

#else // !QUILL_AMX_BUILD

extern "C" bool    quill_amx_available(void) { return false; }
extern "C" bool    quill_amx_type_supported(enum ggml_type) { return false; }
extern "C" int     quill_amx_min_batch(void) { return 1 << 30; }
extern "C" int64_t quill_amx_interleave(enum ggml_type) { return 1; }
extern "C" size_t  quill_amx_packed_size(int64_t, int64_t) { return 0; }
extern "C" bool    quill_amx_flash_attn_eligible(const struct ggml_tensor *) { return false; }

#endif
