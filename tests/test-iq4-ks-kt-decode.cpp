// Checks the IQ4_KS and IQ4_KT decode arithmetic against ggml's to_float.
//
// This is a CPU test and contains no Vulkan: it re-implements, in C++, the indexing that
// each of the four Vulkan shader families uses for these two types, and diffs the result
// against the CPU reference. It therefore validates the format transcription, not the
// compiled shaders -- it cannot catch a driver or codegen fault, and it has to be kept in
// step with ggml/src/vulkan-shaders/types.comp by hand. The shaders themselves are covered
// on device by test-backend-ops and by GGML_VULKAN_CHECK_RESULTS.
//
//   dequant_iq4_k*.comp        one thread per 32-weight subblock, global block index
//   get_rows_iq4_k*.comp       two weights per invocation, row index taken from the shape
//   mul_mm.comp A-load         LOAD_VEC_A weights per load index
//   mul_mat_vec_iq4_k*.comp    one thread per subblock, rows walked by the row stride
//
// Buffers are synthesised directly, so every bit pattern is reachable and no quantizer is
// needed, and they are laid out with ggml_row_size() so that a wrong stride in the mirrored
// a_row_words() shows up as a mismatch rather than shifting the reference too.

#include "ggml.h"

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const uint32_t QK_K_      = 256;
static const uint32_t KS_BLOCK_W = 34;   // 8 scale bytes + 128 quant bytes
static const uint32_t KT_BLOCK_W = 32;   // 8 shb words + 64 ql bytes + 32 qh bytes
static const uint32_t ROW_META_W = 1;    // the row scale

static int    g_checks   = 0;
static int    g_failures = 0;
static int    g_reported = 0;

static float bits_to_float(uint32_t u) {
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static uint32_t a_row_words(uint32_t ncols, uint32_t block_words) {
    return ROW_META_W + (ncols / QK_K_) * block_words;
}

static uint32_t rd_byte(const uint32_t * w, uint32_t base_word, uint32_t byte_idx) {
    return (w[base_word + byte_idx / 4] >> (8 * (byte_idx % 4))) & 0xFF;
}

// ---------------------------------------------------------------------------------------
// IQ4_KS, once per shader family

static float ks_scale(const uint32_t * w, uint32_t row_word, uint32_t blk, uint32_t ib32, uint32_t & vo) {
    const uint32_t sc = rd_byte(w, blk, ib32);
    vo = (sc & 1) << 4;
    return bits_to_float(w[row_word]) * (float)((int)(sc & 254) - 127);
}

// the 32 weights of one subblock; shared by the dequant and mul_mat_vec mirrors, which
// differ in addressing, not in how a subblock is decoded
static void ks_emit_subblock(const uint32_t * w, uint32_t blk, uint32_t ib32, float dl, uint32_t vo, float * out) {
    for (uint32_t l = 0; l < 4; ++l) {
        const uint32_t qs = w[blk + 2 + 4 * ib32 + l];
        for (uint32_t k = 0; k < 4; ++k) {
            const uint32_t q = (qs >> (8 * k)) & 0xFF;
            out[4 * l + k     ] = dl * (float)iq4k_values[vo + (q & 0xF)];
            out[4 * l + k + 16] = dl * (float)iq4k_values[vo + (q >>  4)];
        }
    }
}

static void ks_dequant_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    const uint32_t blocks_per_row = ncols / QK_K_;
    for (uint32_t ib = 0; ib < blocks_per_row * nrows; ++ib) {
        const uint32_t row_word = (ib / blocks_per_row) * a_row_words(ncols, KS_BLOCK_W);
        const uint32_t blk = row_word + ROW_META_W + (ib % blocks_per_row) * KS_BLOCK_W;
        for (uint32_t ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t vo;
            const float dl = ks_scale(w, row_word, blk, ib32, vo);
            ks_emit_subblock(w, blk, ib32, dl, vo, out + 256 * ib + 32 * ib32);
        }
    }
}

static float ks_get_rows_elem(const uint32_t * w, uint32_t row_word, uint32_t i00) {
    const uint32_t blk  = row_word + ROW_META_W + (i00 / QK_K_) * KS_BLOCK_W;
    const uint32_t pos  = i00 % QK_K_;
    const uint32_t ib32 = pos / 32;
    const uint32_t j    = pos % 32;

    uint32_t vo;
    const float dl = ks_scale(w, row_word, blk, ib32, vo);

    const uint32_t qb = 16 * ib32 + (j & 15);
    const uint32_t q  = rd_byte(w, blk + 2, qb);

    return dl * (float)iq4k_values[vo + (j < 16 ? (q & 0xF) : (q >> 4))];
}

static void ks_get_rows_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    for (uint32_t r = 0; r < nrows; ++r) {
        const uint32_t row_word = r * a_row_words(ncols, KS_BLOCK_W);
        for (uint32_t i00 = 0; i00 < ncols; i00 += 2) {
            out[r * ncols + i00    ] = ks_get_rows_elem(w, row_word, i00);
            out[r * ncols + i00 + 1] = ks_get_rows_elem(w, row_word, i00 + 1);
        }
    }
}

static void ks_mul_mm_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    const uint32_t load_vec_a = 2;
    for (uint32_t idx = 0; idx < nrows * ncols / load_vec_a; ++idx) {
        const uint32_t elem = idx * load_vec_a;
        const uint32_t row_word = (elem / ncols) * a_row_words(ncols, KS_BLOCK_W);
        const uint32_t acol = elem % ncols;

        const uint32_t blk  = row_word + ROW_META_W + (acol / QK_K_) * KS_BLOCK_W;
        const uint32_t ib32 = (acol % QK_K_) / 32;
        const uint32_t j    = acol % 32;

        uint32_t vo;
        const float dl = ks_scale(w, row_word, blk, ib32, vo);

        const uint32_t qb = 16 * ib32 + (j & 15);
        const uint32_t qw = w[blk + 2 + qb / 4] >> (8 * (qb % 4));
        const uint32_t qshift = j < 16 ? 0 : 4;

        out[elem    ] = dl * (float)iq4k_values[vo + ((qw >> qshift) & 0xF)];
        out[elem + 1] = dl * (float)iq4k_values[vo + ((qw >> (8 + qshift)) & 0xF)];
    }
}

static void ks_mul_mat_vec_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    const uint32_t words_per_row = a_row_words(ncols, KS_BLOCK_W);
    for (uint32_t i = 0; i < ncols / QK_K_; ++i) {
        for (uint32_t ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t row_word = 0;
            for (uint32_t n = 0; n < nrows; ++n) {
                const uint32_t blk = row_word + ROW_META_W + i * KS_BLOCK_W;
                uint32_t vo;
                const float dl = ks_scale(w, row_word, blk, ib32, vo);
                ks_emit_subblock(w, blk, ib32, dl, vo, out + n * ncols + i * QK_K_ + 32 * ib32);
                row_word += words_per_row;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// IQ4_KT, once per shader family

// the single trellis step, as written in types.comp
static float kt_next(uint32_t & x) {
    x *= 0xCBAC1FEDu;
    const uint32_t s = x & 0x3F3F3F3Fu;
    return (float)((int32_t)((s & 0xFF) + ((s >> 8) & 0xFF) + ((s >> 16) & 0xFF) + (s >> 24)) - 126);
}

// index assembly for one group of four weights, shared by the four mirrors below
static uint32_t kt_group_x(const uint32_t * w, uint32_t blk, uint32_t ib32, uint32_t ig, float d, float & sl) {
    const uint32_t shb = w[blk + ib32];
    sl = d * (float)((int)((shb & 0xFF) >> 1) - 64);
    const uint32_t offset = (shb & 1) != 0 ? 4096 + 32768 : 4096;

    const uint32_t jj = 8 * ib32 + ig;
    const uint32_t ql = rd_byte(w, blk + 8, jj);
    const uint32_t jh = jj % 32;
    const uint32_t qh = rd_byte(w, blk + 24, jh);

    return offset + (ql | ((qh << (8 - 4 * (jj / 32))) & 0xF00) | (((shb >> (8 + 3 * ig)) & 7) << 12));
}

// the 32 weights of one subblock; shared by the dequant and mul_mat_vec mirrors, which
// differ in addressing, not in how a subblock is decoded
static void kt_emit_subblock(const uint32_t * w, uint32_t blk, uint32_t ib32, float d, float * out) {
    for (uint32_t ig = 0; ig < 8; ++ig) {
        float sl;
        uint32_t x = kt_group_x(w, blk, ib32, ig, d, sl);
        for (uint32_t k = 0; k < 4; ++k) {
            out[4 * ig + k] = sl * kt_next(x);
        }
    }
}

static void kt_dequant_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    const uint32_t blocks_per_row = ncols / QK_K_;
    for (uint32_t ib = 0; ib < blocks_per_row * nrows; ++ib) {
        const uint32_t row_word = (ib / blocks_per_row) * a_row_words(ncols, KT_BLOCK_W);
        const uint32_t blk = row_word + ROW_META_W + (ib % blocks_per_row) * KT_BLOCK_W;
        const float d = bits_to_float(w[row_word]);
        for (uint32_t ib32 = 0; ib32 < 8; ++ib32) {
            kt_emit_subblock(w, blk, ib32, d, out + 256 * ib + 32 * ib32);
        }
    }
}

static void kt_get_rows_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    for (uint32_t r = 0; r < nrows; ++r) {
        const uint32_t row_word = r * a_row_words(ncols, KT_BLOCK_W);
        const float d = bits_to_float(w[row_word]);
        for (uint32_t i00 = 0; i00 < ncols; i00 += 2) {
            const uint32_t blk = row_word + ROW_META_W + (i00 / QK_K_) * KT_BLOCK_W;
            const uint32_t pos = i00 % QK_K_;
            float sl;
            uint32_t x = kt_group_x(w, blk, pos / 32, (pos % 32) / 4, d, sl);

            const uint32_t k0 = pos % 4;
            float v0 = 0.0f;
            float v1 = 0.0f;
            for (uint32_t l = 0; l < 4; ++l) {
                const float v = sl * kt_next(x);
                if (l == k0    ) v0 = v;
                if (l == k0 + 1) v1 = v;
            }
            out[r * ncols + i00    ] = v0;
            out[r * ncols + i00 + 1] = v1;
        }
    }
}

static void kt_mul_mm_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    const uint32_t load_vec_a = 4;
    for (uint32_t idx = 0; idx < nrows * ncols / load_vec_a; ++idx) {
        const uint32_t elem = idx * load_vec_a;
        const uint32_t row_word = (elem / ncols) * a_row_words(ncols, KT_BLOCK_W);
        const uint32_t acol = elem % ncols;

        const uint32_t blk = row_word + ROW_META_W + (acol / QK_K_) * KT_BLOCK_W;
        float sl;
        uint32_t x = kt_group_x(w, blk, (acol % QK_K_) / 32, (acol % 32) / 4, bits_to_float(w[row_word]), sl);

        for (uint32_t k = 0; k < 4; ++k) {
            out[elem + k] = sl * kt_next(x);
        }
    }
}

static void kt_mul_mat_vec_shader(const uint32_t * w, uint32_t ncols, uint32_t nrows, float * out) {
    const uint32_t words_per_row = a_row_words(ncols, KT_BLOCK_W);
    for (uint32_t i = 0; i < ncols / QK_K_; ++i) {
        for (uint32_t ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t row_word = 0;
            for (uint32_t n = 0; n < nrows; ++n) {
                const uint32_t blk = row_word + ROW_META_W + i * KT_BLOCK_W;
                kt_emit_subblock(w, blk, ib32, bits_to_float(w[row_word]),
                                 out + n * ncols + i * QK_K_ + 32 * ib32);
                row_word += words_per_row;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------

// Copied from QuantizerIQKT<32, 4, 15, false, true>::set_values in ggml/src/iqk/iqk_quantize.cpp.
// It reads the accumulator through int8_t, so it carries the same little-endian assumption
// the original has. Used only to cross-check kt_next() over the whole index range.
static void ref_set_values(uint32_t i, float * result, float scale, int offset) {
    uint32_t x = i + offset;
    const uint32_t ka = 0xCBAC1FED;
    uint32_t s;
    const int8_t * i8 = (const int8_t *)&s;
    for (int k = 0; k < 4; ++k) {
        x = ka*x;
        s = x & 0x3f3f3f3f;
        result[k] = scale*(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
    }
}

static void check(const char * what, const std::vector<float> & got, const std::vector<float> & ref) {
    g_checks++;
    size_t bad = 0;
    size_t first = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (got[i] != ref[i]) {
            if (bad == 0) {
                first = i;
            }
            bad++;
        }
    }
    if (bad == 0) {
        return;
    }
    g_failures++;
    if (g_reported < 12) {
        g_reported++;
        printf("  FAIL %-42s %zu/%zu differ, first at %zu: got %.9g want %.9g\n",
               what, bad, ref.size(), first, got[first], ref[first]);
    }
}

// Fills a quantized buffer. payload < 0 gives a reproducible pseudo-random pattern,
// otherwise every payload byte is set to that value. Every byte pattern is a valid encoding
// for both types, so this reaches states a quantizer would never emit. The first eight
// bytes of every block are then overwritten with the given scale bytes.
static void fill_pattern(std::vector<uint32_t> & w, uint32_t ncols, uint32_t nrows, uint32_t block_words,
                         uint32_t words_per_row, uint32_t seed, int payload, const uint8_t * scales) {
    const float row_scales[4] = { 0.0125f, -0.5f, 1.0f, 3.75e-3f };

    uint32_t rng = seed * 2654435761u + 1u;
    for (uint32_t r = 0; r < nrows; ++r) {
        uint32_t bits;
        const float d = row_scales[r % 4];
        memcpy(&bits, &d, sizeof(bits));
        w[r * words_per_row] = bits;

        for (uint32_t k = ROW_META_W; k < words_per_row; ++k) {
            if (payload >= 0) {
                w[r * words_per_row + k] = 0x01010101u * (uint32_t)payload;
            } else {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                w[r * words_per_row + k] = rng;
            }
        }

        for (uint32_t b = 0; b < ncols / QK_K_; ++b) {
            const uint32_t blk = r * words_per_row + ROW_META_W + b * block_words;
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t word = blk + k / 4;
                const uint32_t shift = 8 * (k % 4);
                w[word] = (w[word] & ~(0xFFu << shift)) | ((uint32_t)scales[k] << shift);
            }
        }
    }
}

static void run_shape(ggml_type type, uint32_t ncols, uint32_t nrows, uint32_t seed,
                      int payload, const uint8_t * scales, const char * tag) {
    const uint32_t block_words = type == GGML_TYPE_IQ4_KS ? KS_BLOCK_W : KT_BLOCK_W;
    // laid out with ggml's own row size, so that a wrong stride in a_row_words() shows up as
    // a mismatch rather than shifting the reference along with the mirrors
    const uint32_t words_per_row = (uint32_t)ggml_row_size(type, ncols) / sizeof(uint32_t);

    std::vector<uint32_t> w(words_per_row * nrows);
    fill_pattern(w, ncols, nrows, block_words, words_per_row, seed, payload, scales);

    // ggml's own dequantizer, one row at a time
    std::vector<float> ref((size_t)ncols * nrows);
    ggml_type_traits_t tt = ggml_internal_get_type_traits(type);
    for (uint32_t r = 0; r < nrows; ++r) {
        tt.to_float(&w[r * words_per_row], &ref[(size_t)r * ncols], ncols);
    }

    struct mirror {
        const char * family;
        void (*fn)(const uint32_t *, uint32_t, uint32_t, float *);
    };

    const mirror ks[4] = {
        { "dequant",     ks_dequant_shader     },
        { "get_rows",    ks_get_rows_shader    },
        { "mul_mm",      ks_mul_mm_shader      },
        { "mul_mat_vec", ks_mul_mat_vec_shader },
    };
    const mirror kt[4] = {
        { "dequant",     kt_dequant_shader     },
        { "get_rows",    kt_get_rows_shader    },
        { "mul_mm",      kt_mul_mm_shader      },
        { "mul_mat_vec", kt_mul_mat_vec_shader },
    };
    const mirror * mirrors = type == GGML_TYPE_IQ4_KS ? ks : kt;

    std::vector<float> got((size_t)ncols * nrows);
    for (int m = 0; m < 4; ++m) {
        std::fill(got.begin(), got.end(), 0.0f);
        mirrors[m].fn(w.data(), ncols, nrows, got.data());

        char name[128];
        snprintf(name, sizeof(name), "%s %-11s %ux%u %s",
                 ggml_type_name(type), mirrors[m].family, nrows, ncols, tag);
        check(name, got, ref);
    }
}

static void run_trellis_sweep(void) {
    const int   offsets[2] = { 4096, 4096 + 32768 };
    const float scales[3]  = { 1.0f, -0.25f, 7.5f };

    size_t bad = 0;
    for (uint32_t idx = 0; idx < 32768; ++idx) {
        for (int o = 0; o < 2; ++o) {
            for (int s = 0; s < 3; ++s) {
                float ref[4];
                ref_set_values(idx, ref, scales[s], offsets[o]);

                uint32_t x = (uint32_t)offsets[o] + idx;
                for (int k = 0; k < 4; ++k) {
                    if (scales[s] * kt_next(x) != ref[k]) {
                        bad++;
                    }
                }
            }
        }
    }

    g_checks++;
    if (bad != 0) {
        g_failures++;
        printf("  FAIL iq4_kt trellis sweep: %zu of %d values differ\n", bad, 32768 * 2 * 3 * 4);
    }
}

int main(void) {
    // scale bytes chosen to hit both codebook halves and both ends of the scale field:
    // IQ4_KS uses (b & 254) - 127, which reaches -127 and +127; IQ4_KT uses (b >> 1) - 64,
    // which reaches -64 and +63. Bit 0 selects the codebook half / the index offset.
    const uint8_t extremes[8] = { 0x00, 0x01, 0xFE, 0xFF, 0x80, 0x81, 0x7E, 0x7F };
    const uint8_t mid[8]      = { 0x40, 0x41, 0x7A, 0x0B, 0xC0, 0xC1, 0x2A, 0x95 };

    // several row and block counts, so the row stride is exercised and not just the
    // addressing inside a block
    const uint32_t shapes[5][2] = { {256, 1}, {256, 3}, {1024, 5}, {1024, 4}, {4096, 2} };

    printf("iq4_kt trellis: all 32768 indices x 2 offsets x 3 scales\n");
    run_trellis_sweep();

    printf("multi-row buffers against ggml to_float\n");
    for (int s = 0; s < 5; ++s) {
        run_shape(GGML_TYPE_IQ4_KS, shapes[s][0], shapes[s][1], 1u + s, -1, extremes, "rand/extreme-scales");
        run_shape(GGML_TYPE_IQ4_KT, shapes[s][0], shapes[s][1], 1u + s, -1, extremes, "rand/extreme-scales");
        run_shape(GGML_TYPE_IQ4_KS, shapes[s][0], shapes[s][1], 9u + s, -1, mid,      "rand/mid-scales");
        run_shape(GGML_TYPE_IQ4_KT, shapes[s][0], shapes[s][1], 9u + s, -1, mid,      "rand/mid-scales");
    }

    printf("saturated payloads\n");
    for (int p = 0; p <= 0xFF; p += 0xFF) {
        run_shape(GGML_TYPE_IQ4_KS, 512, 2, 0, p, extremes, p == 0 ? "all-zero" : "all-ones");
        run_shape(GGML_TYPE_IQ4_KT, 512, 2, 0, p, extremes, p == 0 ? "all-zero" : "all-ones");
        run_shape(GGML_TYPE_IQ4_KS, 512, 2, 0, p, mid,      p == 0 ? "all-zero" : "all-ones");
        run_shape(GGML_TYPE_IQ4_KT, 512, 2, 0, p, mid,      p == 0 ? "all-zero" : "all-ones");
    }

    printf("%d checks, %d failed\n", g_checks, g_failures);

    return g_failures == 0 ? 0 : 1;
}
