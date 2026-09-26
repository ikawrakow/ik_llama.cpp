// Test for the packed KV-cache storage types.
//
//   test-kv-quants                self-test on embedded edge cases
//   test-kv-quants CORPUS OUT     quantise the vectors in CORPUS, write the results to OUT
//
// The types are GGML_TYPE_FP4_B16_E4M3, GGML_TYPE_FP4_B32_E8M0 and
// GGML_TYPE_FP8_B32_E8M0 (ggml-kv-quants.h). Everything here goes through the
// registered type_traits - so this also checks that the three types are
// registered with the right size, block size and name.
//
// The self-test covers what can be checked without an external oracle: the
// amax floors, saturation, the reference packing order, and the rounding ties
// (verified against the reference's own CUDA kernels). The emit mode writes a
// corpus/result pair that an external oracle can diff against reference vectors.
//
// Corpus line:  <type_name> <n> <v0> ... <v_{n-1}>
// Output line:  <type_name> <n> <scales_hex> <packed_hex> <y0_bits> ...

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-kv-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_checks   = 0;
static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        printf("  [FAIL] %s\n", what.c_str());
    }
}

static uint32_t f2b(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

struct kv_type {
    const char * name;
    ggml_type    type;
    int          block;
    int          bs;         // bytes per block
    int          scale_e4m3;
};

static const kv_type g_types[] = {
    { "fp4_B16_E4M3", GGML_TYPE_FP4_B16_E4M3, QK_FP4_B16, 1 + QK_FP4_B16/2, 1 },
    { "fp4_B32_E8M0", GGML_TYPE_FP4_B32_E8M0, QK_FP4_B32, 1 + QK_FP4_B32/2, 0 },
    { "fp8_B32_E8M0", GGML_TYPE_FP8_B32_E8M0, QK_FP8_B32, 1 + QK_FP8_B32,   0 },
};

static const kv_type * find_type(const std::string & name) {
    for (const kv_type & t : g_types) {
        if (name == t.name) {
            return &t;
        }
    }
    return nullptr;
}

// quantise through the registered traits, so the registration is part of the test
static void quantize_rows(ggml_type type, const float * x, int64_t n, uint8_t * y) {
    ggml_type_traits_t tr = ggml_internal_get_type_traits(type);
    tr.from_float(x, y, n);
}

static void dequantize_rows(ggml_type type, const uint8_t * x, float * y, int64_t n) {
    ggml_type_traits_t tr = ggml_internal_get_type_traits(type);
    tr.to_float(x, y, n);
}

static std::string to_hex(const uint8_t * p, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s += digits[p[i] >> 4];
        s += digits[p[i] & 0x0F];
    }
    return s;
}

// ---------------------------------------------------------------------------
// self-test
// ---------------------------------------------------------------------------

// An all-zero block gets a nonzero, representable scale, but its *values* stay
// zero - the floor is observable only in the scale byte. The byte's decode is
// checked by the nonzero cases (a zero code always dequantises to 0).
static void self_test_floors() {
    printf("floors (an all-zero block still gets a nonzero scale)\n");

    const struct { const char * name; uint8_t scale; } want[] = {
        { "fp8_B32_E8M0", 105 },  // 1e-4 / 448     -> 2^-22
        { "fp4_B32_E8M0",   1 },  // 6 * 2^-126 / 6 -> 2^-126
        { "fp4_B16_E4M3",   1 },  // 6 * 2^-9   / 6 -> 2^-9
    };

    for (const auto & w : want) {
        const kv_type * t = find_type(w.name);
        std::vector<float>   x(t->block, 0.0f);
        std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
        std::vector<float>   y(t->block);
        quantize_rows(t->type, x.data(), t->block, q.data());
        dequantize_rows(t->type, q.data(), y.data(), t->block);

        check(q[0] == w.scale, std::string(w.name) + ": zero block scale byte");
        for (int j = 0; j < t->block; ++j) {
            check(y[j] == 0.0f, std::string(w.name) + ": zero block stays zero");
        }
    }

    // ... and a block holding 1.0 only shows what that byte decodes to: amax 1
    // gives 2^-8 for fp8 and 2^-2 for fp4 (both exact powers of two).
    {
        const struct { const char * name; float y; } one[] = {
            { "fp8_B32_E8M0", 1.0f },
            { "fp4_B32_E8M0", 1.0f },
        };
        for (const auto & o : one) {
            const kv_type * t = find_type(o.name);
            std::vector<float>   x(t->block, 0.0f);
            std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
            std::vector<float>   y(t->block);
            x[0] = 1.0f;
            quantize_rows(t->type, x.data(), t->block, q.data());
            dequantize_rows(t->type, q.data(), y.data(), t->block);
            check(y[0] == o.y, std::string(o.name) + ": a single 1.0 round-trips");
        }
    }
}

static void self_test_saturation() {
    printf("saturation and the e4m3-scale clamp\n");

    // fp4_B32_E8M0: amax 6 -> s = 1, so the codes saturate at 6 (code 7)
    {
        const kv_type * t = find_type("fp4_B32_E8M0");
        std::vector<float> x(t->block, 6.0f);
        std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
        std::vector<float> y(t->block);
        quantize_rows(t->type, x.data(), t->block, q.data());
        dequantize_rows(t->type, q.data(), y.data(), t->block);
        check(q[0] == 127, "fp4_B32_E8M0: amax 6 gives scale 1");
        for (int j = 0; j < t->block; ++j) {
            check(((q[1 + j/2] >> (4*(j%2))) & 0x0F) == 7, "fp4_B32_E8M0: 6.0 saturates to code 7");
            check(y[j] == 6.0f, "fp4_B32_E8M0: 6.0 round-trips");
        }
    }
    // fp4_B16_E4M3: the e4m3 scale rounds to nearest, so it can be *below* amax/6
    // and the clamp to +-6 then binds: amax 100 -> s = 16 -> 100/16 = 6.25 -> 6
    {
        const kv_type * t = find_type("fp4_B16_E4M3");
        std::vector<float> x(t->block, 100.0f);
        std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
        std::vector<float> y(t->block);
        quantize_rows(t->type, x.data(), t->block, q.data());
        dequantize_rows(t->type, q.data(), y.data(), t->block);
        check(q[0] == 0x58, "fp4_B16_E4M3: amax 100 gives the e4m3 scale 16");
        for (int j = 0; j < t->block; ++j) {
            check(((q[1 + j/2] >> (4*(j%2))) & 0x0F) == 7, "fp4_B16_E4M3: the clamp binds, code 7");
            check(y[j] == 96.0f, "fp4_B16_E4M3: 100 -> 96");
        }
    }
    // the sign is bit 3 of an e2m1 code: -6.0 must come back as -6.0, not as
    // whatever a magnitude-only lookup happens to find at index 15
    {
        const kv_type * t = find_type("fp4_B32_E8M0");
        std::vector<float>   x(t->block, -6.0f);
        std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
        std::vector<float>   y(t->block);
        quantize_rows(t->type, x.data(), t->block, q.data());
        dequantize_rows(t->type, q.data(), y.data(), t->block);
        for (int j = 0; j < t->block; ++j) {
            check(((q[1 + j/2] >> (4*(j%2))) & 0x0F) == 0x0F, "fp4_B32_E8M0: -6.0 is code 0x0F");
            check(y[j] == -6.0f, "fp4_B32_E8M0: -6.0 round-trips with its sign");
        }
    }
    // fp8_B32_E8M0: amax 448 -> s = 1, the codes saturate at 448 (0x7E)
    {
        const kv_type * t = find_type("fp8_B32_E8M0");
        std::vector<float> x(t->block, 448.0f);
        std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
        std::vector<float> y(t->block);
        quantize_rows(t->type, x.data(), t->block, q.data());
        dequantize_rows(t->type, q.data(), y.data(), t->block);
        check(q[0] == 127, "fp8_B32_E8M0: amax 448 gives scale 1");
        for (int j = 0; j < t->block; ++j) {
            check(q[1 + j] == 0x7E, "fp8_B32_E8M0: 448 saturates to 0x7E");
            check(y[j] == 448.0f, "fp8_B32_E8M0: 448 round-trips");
        }
    }
}

static void self_test_packing() {
    printf("packing order (reference: element 2j low nibble, 2j+1 high)\n");

    const kv_type * t = find_type("fp4_B32_E8M0");
    std::vector<uint8_t> q(ggml_row_size(t->type, t->block));

    // even elements 6.0 (code 7), odd elements 0.0 (code 0)
    std::vector<float> x(t->block, 0.0f);
    for (int j = 0; j < t->block; j += 2) x[j] = 6.0f;
    quantize_rows(t->type, x.data(), t->block, q.data());
    for (int j = 0; j < t->block/2; ++j) check(q[1 + j] == 0x07, "even elements are the low nibbles");

    // the mirror image
    for (int j = 0; j < t->block; j += 2) x[j] = 0.0f;
    for (int j = 1; j < t->block; j += 2) x[j] = 6.0f;
    quantize_rows(t->type, x.data(), t->block, q.data());
    for (int j = 0; j < t->block/2; ++j) check(q[1 + j] == 0x70, "odd elements are the high nibbles");
}

// The tie cases below are the ones the reference's own CUDA kernels were probed
// with: each is a midpoint of two e2m1 / e4m3 grid values, where RNE and
// truncation disagree. The observed reference values are the RNE ones.
static void self_test_ties() {
    printf("rounding ties (verified against the reference kernels: RNE)\n");

    // fp4: amax 1.0 -> s = 2^ceil(log2(1/6)) = 0.25
    {
        const kv_type * t = find_type("fp4_B32_E8M0");
        const struct { float x; float y; } cases[] = {
            { 0.875f,  1.0f  },   // 3.5  -> 4.0
            { 0.4375f, 0.5f  },   // 1.75 -> 2.0
            { 0.1875f, 0.25f },   // 0.75 -> 1.0
            { 0.625f,  0.5f  },   // 2.5  -> 2.0
            { 1.25f,   1.0f  },   // 5.0  -> 4.0
        };
        for (const auto & c : cases) {
            std::vector<float> x(t->block, 0.0f);
            std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
            std::vector<float> y(t->block);
            x[0] = 1.0f;      // the block amax, so the scale is 0.25
            x[1] = c.x;
            quantize_rows(t->type, x.data(), t->block, q.data());
            dequantize_rows(t->type, q.data(), y.data(), t->block);
            char msg[128];
            snprintf(msg, sizeof(msg), "fp4 tie at %g -> %g (RNE)", c.x, c.y);
            check(y[1] == c.y, msg);
        }
    }
    // fp8: amax 448 -> s = 1
    {
        const kv_type * t = find_type("fp8_B32_E8M0");
        const struct { float x; float y; } cases[] = {
            { 1.1875f, 1.25f  },   // the midpoint of 1.125 / 1.25
            { 0.59375f, 0.625f },  // the midpoint of 0.5625 / 0.625
        };
        for (const auto & c : cases) {
            std::vector<float> x(t->block, 0.0f);
            std::vector<uint8_t> q(ggml_row_size(t->type, t->block));
            std::vector<float> y(t->block);
            x[0] = 448.0f;
            x[1] = c.x;
            quantize_rows(t->type, x.data(), t->block, q.data());
            dequantize_rows(t->type, q.data(), y.data(), t->block);
            char msg[128];
            snprintf(msg, sizeof(msg), "fp8 tie at %g -> %g (RNE)", c.x, c.y);
            check(y[1] == c.y, msg);
        }
    }
}

static int self_test() {
    printf("test-kv-quants - self-test\n\n");

    printf("registration (ggml type_traits)\n");
    for (const kv_type & t : g_types) {
        ggml_type_traits_t tr = ggml_internal_get_type_traits(t.type);
        check(tr.type_name != nullptr && strcmp(tr.type_name, t.name) == 0,
              std::string(t.name) + ": type_name");
        check(tr.blck_size == t.block, std::string(t.name) + ": blck_size");
        check((int)tr.type_size == t.bs, std::string(t.name) + ": type_size");
        check(tr.is_quantized, std::string(t.name) + ": is_quantized");
        check(tr.from_float != nullptr && tr.to_float != nullptr,
              std::string(t.name) + ": from_float and to_float");
        check(tr.vec_dot == nullptr, std::string(t.name) + ": no vec_dot (storage only)");
        check(ggml_row_size(t.type, t.block) == (size_t)t.bs, std::string(t.name) + ": row_size");
    }

    printf("\n");
    self_test_floors();
    printf("\n");
    self_test_saturation();
    printf("\n");
    self_test_packing();
    printf("\n");
    self_test_ties();

    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// emit mode: quantise a corpus, print the scales, the packed bytes and the
// dequantised values so an external oracle can diff them
// ---------------------------------------------------------------------------

static int emit(const char * corpus_path, const char * out_path) {
    std::ifstream in(corpus_path);
    if (!in) {
        fprintf(stderr, "cannot open %s\n", corpus_path);
        return 2;
    }
    std::ofstream out(out_path);
    if (!out) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }

    std::string line;
    int ncase = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name;
        int64_t n = 0;
        if (!(ss >> name >> n) || n <= 0) {
            fprintf(stderr, "bad corpus line: %s\n", line.c_str());
            return 2;
        }
        const kv_type * t = find_type(name);
        if (!t) {
            fprintf(stderr, "unknown type %s\n", name.c_str());
            return 2;
        }
        if (n % t->block != 0) {
            fprintf(stderr, "%s: n=%lld is not a multiple of %d\n", name.c_str(), (long long)n, t->block);
            return 2;
        }
        std::vector<float> x(n);
        for (int64_t i = 0; i < n; ++i) {
            if (!(ss >> x[i])) {
                fprintf(stderr, "bad value %lld in %s\n", (long long)i, line.c_str());
                return 2;
            }
        }

        const size_t rs = ggml_row_size(t->type, n);
        std::vector<uint8_t> q(rs);
        std::vector<float>    y(n);
        quantize_rows(t->type, x.data(), n, q.data());
        dequantize_rows(t->type, q.data(), y.data(), n);

        std::vector<uint8_t> scales;
        std::vector<uint8_t> packed;
        for (int64_t ib = 0; ib < n/t->block; ++ib) {
            const uint8_t * b = q.data() + ib*t->bs;
            scales.push_back(b[0]);
            packed.insert(packed.end(), b + 1, b + t->bs);
        }

        out << name << ' ' << n << ' ' << to_hex(scales.data(), scales.size()) << ' '
            << to_hex(packed.data(), packed.size());
        for (int64_t i = 0; i < n; ++i) {
            char buf[16];
            snprintf(buf, sizeof(buf), " %08x", f2b(y[i]));
            out << buf;
        }
        out << '\n';
        ++ncase;
    }
    printf("emitted %d cases to %s\n", ncase, out_path);
    return 0;
}

// ---------------------------------------------------------------------------
// backend mode: the same corpus, but through ggml_set_rows / ggml_get_rows on
// a real backend. The cache row is written with set_rows and read back with
// get_rows - the real cache path - so one run covers both the backend's
// write side (the destination bytes) and its read side (the F32 output).
//
//   test-kv-quants CORPUS OUT --backend CUDA0
//
// The output format is the one emit() writes, so the same oracle driver diffs
// the CPU type_traits path, the CPU backend and the CUDA backend against the
// same reference vectors.
// ---------------------------------------------------------------------------

static int emit_backend(const char * corpus_path, const char * out_path, const char * backend_name) {
    // the registry name is the family ("CUDA"), the backend name the device
    // ("CUDA0"), so accept either
    ggml_backend_t backend = NULL;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        ggml_backend_t b = ggml_backend_reg_init_backend(i, NULL);
        if (b == NULL) {
            continue;
        }
        if (strcmp(ggml_backend_reg_get_name(i), backend_name) == 0 ||
            strcmp(ggml_backend_name(b), backend_name) == 0) {
            backend = b;
            break;
        }
        ggml_backend_free(b);
    }
    if (!backend) {
        fprintf(stderr, "no backend named %s\n", backend_name);
        return 2;
    }
    printf("backend %s\n", ggml_backend_name(backend));

    std::ifstream in(corpus_path);
    if (!in) {
        fprintf(stderr, "cannot open %s\n", corpus_path);
        return 2;
    }
    std::ofstream out(out_path);
    if (!out) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 2;
    }

    std::string line;
    int ncase = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string name;
        int64_t n = 0;
        if (!(ss >> name >> n) || n <= 0) {
            fprintf(stderr, "bad corpus line: %s\n", line.c_str());
            return 2;
        }
        const kv_type * t = find_type(name);
        if (!t) {
            fprintf(stderr, "unknown type %s\n", name.c_str());
            return 2;
        }
        if (n % t->block != 0) {
            fprintf(stderr, "%s: n=%lld is not a multiple of %d\n", name.c_str(), (long long)n, t->block);
            return 2;
        }
        std::vector<float> x(n);
        for (int64_t i = 0; i < n; ++i) {
            if (!(ss >> x[i])) {
                fprintf(stderr, "bad value %lld in %s\n", (long long)i, line.c_str());
                return 2;
            }
        }

        // one row of n elements: dst = the cache row, src0 = the values, src1 = the row index
        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * dst_t = ggml_new_tensor_2d(ctx, t->type, n, 1);
        ggml_tensor * src0  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
        ggml_tensor * src1  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * set   = ggml_set_rows(ctx, dst_t, src0, src1);
        ggml_tensor * got   = ggml_get_rows(ctx, set, src1);
        ggml_cgraph * gf    = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, got);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            fprintf(stderr, "cannot allocate %s tensors\n", backend_name);
            return 2;
        }
        const int32_t idx = 0;
        ggml_backend_tensor_set(src0, x.data(), 0, n*sizeof(float));
        ggml_backend_tensor_set(src1, &idx, 0, sizeof(idx));
        ggml_backend_graph_compute(backend, gf);

        std::vector<uint8_t> q(ggml_row_size(t->type, n));
        std::vector<float>   y(n);
        ggml_backend_tensor_get(dst_t, q.data(), 0, q.size());
        ggml_backend_tensor_get(got,   y.data(), 0, n*sizeof(float));

        std::vector<uint8_t> scales;
        std::vector<uint8_t> packed;
        for (int64_t ib = 0; ib < n/t->block; ++ib) {
            const uint8_t * b = q.data() + ib*t->bs;
            scales.push_back(b[0]);
            packed.insert(packed.end(), b + 1, b + t->bs);
        }
        out << name << ' ' << n << ' ' << to_hex(scales.data(), scales.size()) << ' '
            << to_hex(packed.data(), packed.size());
        for (int64_t i = 0; i < n; ++i) {
            char buf[16];
            snprintf(buf, sizeof(buf), " %08x", f2b(y[i]));
            out << buf;
        }
        out << '\n';
        ++ncase;

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    printf("emitted %d cases to %s (backend %s)\n", ncase, out_path, ggml_backend_name(backend));
    ggml_backend_free(backend);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 1) {
        return self_test();
    }
    if (argc == 3) {
        return emit(argv[1], argv[2]);
    }
    if (argc == 5 && strcmp(argv[3], "--backend") == 0) {
        return emit_backend(argv[1], argv[2], argv[4]);
    }
    fprintf(stderr, "usage: %s [CORPUS OUT [--backend NAME]]\n", argv[0]);
    return 2;
}