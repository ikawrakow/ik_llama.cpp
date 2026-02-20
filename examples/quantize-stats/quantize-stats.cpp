//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#define LLAMA_API_INTERNAL
#include "common.h"
#include "ggml.h"
#include "llama.h"

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "../ggml/src/ggml-common.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <array>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#include <intrin.h>
#include <ammintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
inline int popcount(uint8_t x) { return __popcnt(x); }
inline int popcount(uint16_t x) { return __popcnt(x); }
inline int popcount(uint32_t x) { return __popcnt(x); }
inline int popcount(uint64_t x) { return _mm_popcnt_u64(x); }
#else
constexpr int popcount(uint8_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint16_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint32_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint64_t x) { return __builtin_popcountll(x); }
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

struct quantize_stats_params {
    std::string model = DEFAULT_MODEL_PATH;
    bool verbose = false;
    bool per_layer_stats = false;
    bool print_histogram = false;
    bool reference = false;
    std::vector<std::string> include_layers;
    std::vector<std::string> exclude_layers;
    std::vector<enum ggml_type> include_types;
};

constexpr size_t HISTOGRAM_BUCKETS = 150;
constexpr double HISTOGRAM_RANGE = 0.03;

struct error_stats {
    size_t num_samples;
    double total_error;
    double max_error;
    double sum_x2;
    uint64_t error_histogram[HISTOGRAM_BUCKETS];
};

static void quantize_stats_print_usage(int /*argc*/, char ** argv) {
    quantize_stats_params params;
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -r, --reference\n");
    fprintf(stderr, "                        use reference implementation (default: false)\n");
    fprintf(stderr, "  -v, --verbose\n");
    fprintf(stderr, "                        verbose output (default: false)\n");
    fprintf(stderr, "  -p, --per-layer-stats\n");
    fprintf(stderr, "                        print stats per layer (default: false)\n");
    fprintf(stderr, "  --histogram\n");
    fprintf(stderr, "                        print error histogram (default: false)\n");
    fprintf(stderr, "  -l LAYER, --include-layer LAYER\n");
    fprintf(stderr, "                        only test layers matching pattern\n");
    fprintf(stderr, "  -L LAYER, --exclude-layer LAYER\n");
    fprintf(stderr, "                        exclude layers matching pattern\n");
    fprintf(stderr, "  -t TYPE, --type TYPE\n");
    fprintf(stderr, "                        only test given type (q4_0, q4_1)\n");
    fprintf(stderr, "\n");
}

// Check if a layer is included/excluded by command line
static bool layer_included(const quantize_stats_params & params, const std::string & layer) {
    for (const auto& excluded : params.exclude_layers) {
        if (std::regex_search(layer, std::regex(excluded))) {
            return false;
        }
    }
    for (const auto& included : params.include_layers) {
        if (std::regex_search(layer, std::regex(included))) {
            return true;
        }
    }
    return params.include_layers.empty();
}

// Update error statistics given vectors with the before/after result of quantization
static void update_error_stats(int64_t nelements, const float * input, const float * output, error_stats & stats) {
    for (int64_t i = 0; i < nelements; i++) {
        double diff = input[i] - output[i];
        stats.total_error += diff * diff;
        stats.max_error = fmax(fabs(diff), stats.max_error);
        stats.sum_x2 += input[i]*input[i];
        stats.error_histogram[std::max(std::min((size_t) floor(fabs(diff) / HISTOGRAM_RANGE * HISTOGRAM_BUCKETS), HISTOGRAM_BUCKETS-1), (size_t) 0)]++;
    }
    stats.num_samples += nelements;
}

static void combine_error_stats(error_stats & into, const error_stats & from) {
    into.num_samples += from.num_samples;
    into.total_error += from.total_error;
    into.sum_x2      += from.sum_x2;
    if (from.max_error > into.max_error) into.max_error = from.max_error;
    for (size_t i=0; i<HISTOGRAM_BUCKETS; ++i) into.error_histogram[i] += from.error_histogram[i];
}

static double find_quantile(const error_stats & stats, double quantile) {
    double sum = std::accumulate(std::begin(stats.error_histogram), std::end(stats.error_histogram), 0.0);

    double accum = 0;
    for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
        accum += stats.error_histogram[i];
        if (accum >= sum*quantile) {
            return (i+1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
        }
    }
    return INFINITY;
}

static void print_error_stats(const std::string & name, const error_stats & stats, bool print_histogram) {
    double rmse = sqrt(stats.total_error / (double) stats.num_samples);
    double av_x = sqrt(stats.sum_x2 / (double) stats.num_samples);
    double median = find_quantile(stats, .5);
    double pct95 = find_quantile(stats, .95);
    printf("%-40s: rmse %.8f, %.6f  maxerr %.8f, %.6f  95pct<%.4f,  median<%.4f\n", name.c_str(), rmse, rmse/av_x,
            stats.max_error, stats.max_error/av_x, pct95, median);
    if (print_histogram) {
        printf("Error distribution:\n");
        for (size_t i = 0; i < HISTOGRAM_BUCKETS; i++) {
            double lower = i * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
            double upper = (i+1) * HISTOGRAM_RANGE / HISTOGRAM_BUCKETS;
            if (i == HISTOGRAM_BUCKETS -1) upper = INFINITY;
            printf("[%3.4f, %3.4f): %11" PRIu64 "\n", lower, upper, stats.error_histogram[i]);
        }
    }
}

// copied from ggml.h - verify that we can access this as a flat array
static bool tensor_is_contiguous(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return
        tensor->nb[0] == ggml_type_size(tensor->type) &&
        tensor->nb[1] == (tensor->nb[0]*tensor->ne[0])/ggml_blck_size(tensor->type) &&
        tensor->nb[2] == tensor->nb[1]*tensor->ne[1] &&
        tensor->nb[3] == tensor->nb[2]*tensor->ne[2];
}

static void test_roundtrip_on_chunk(
    const ggml_tensor * layer, int64_t offset, int64_t chunk_size, const ggml_type_traits_t & qfns, bool use_reference,
    float * input_scratch, char * quantized_scratch, float * output_scratch, error_stats & stats
) {
    if (layer->type == GGML_TYPE_F16) {
        for (int i = 0; i < chunk_size; i++) {
            input_scratch[i] = ggml_get_f32_1d(layer, i + offset);
        }
    } else {
        input_scratch = ggml_get_data_f32(layer) + offset;
    }

    if (use_reference) {
        qfns.from_float_ref(input_scratch, quantized_scratch, chunk_size);
    } else {
        qfns.from_float(input_scratch, quantized_scratch, chunk_size);
    }
    qfns.to_float(quantized_scratch, output_scratch, chunk_size);

    update_error_stats(chunk_size, input_scratch, output_scratch, stats);
}


// Run quantization function for a single layer and update error stats
static void test_roundtrip_on_layer(
    std::string & name, bool print_layer_stats, const ggml_type_traits_t & qfns, bool use_reference,
    const ggml_tensor * layer, std::vector<float> & input_scratch, std::vector<char> & quantized_scratch,
    std::vector<float> & output_scratch, error_stats & total_error, int max_thread = 0
) {
    assert(tensor_is_contiguous(layer));
    error_stats layer_error {};
    uint64_t nelements = ggml_nelements(layer);

    float* input_scratch_ptr = nullptr;
    if (layer->type == GGML_TYPE_F16) {
        if (input_scratch.size() < nelements) input_scratch.resize(nelements);
        input_scratch_ptr = input_scratch.data();
    }
    if (quantized_scratch.size() < 4*nelements) quantized_scratch.resize(4*nelements);
    if (output_scratch.size() < nelements) output_scratch.resize(nelements);

    if (max_thread < 1) max_thread = std::thread::hardware_concurrency();
    int chunk_size = 32*512;
    int num_chunks = (nelements + chunk_size - 1)/chunk_size;

    if (num_chunks < 2 || max_thread < 2) {
        test_roundtrip_on_chunk(layer, 0, nelements, qfns, use_reference, input_scratch_ptr, quantized_scratch.data(),
                output_scratch.data(), print_layer_stats ? layer_error : total_error);
    } else {
        auto & stats = print_layer_stats ? layer_error : total_error;
        std::mutex mutex;
        uint64_t counter = 0;
        auto compute = [&mutex, &counter, &stats, &qfns, nelements, layer, use_reference, input_scratch_ptr,
             &quantized_scratch, &output_scratch, chunk_size] () {
            error_stats local_stats {};
            while (true) {
                std::unique_lock<std::mutex> lock(mutex);
                uint64_t offset = counter; counter += chunk_size;
                if (offset >= nelements) {
                    combine_error_stats(stats, local_stats);
                    break;
                }
                lock.unlock();
                uint64_t chunk = offset + chunk_size < nelements ? chunk_size : nelements - offset;
                test_roundtrip_on_chunk(layer, offset, chunk, qfns, use_reference, input_scratch_ptr + offset,
                        quantized_scratch.data() + 4*offset, output_scratch.data() + offset, local_stats);
            }
        };
        int nthread = std::min(num_chunks, max_thread);
        std::vector<std::thread> workers(nthread-1);
        for (auto& w : workers) w = std::thread(compute);
        compute();
        for (auto& w : workers) w.join();
    }

    if (print_layer_stats) {
        print_error_stats(name, layer_error, false);
        combine_error_stats(total_error, layer_error);
    }
}

static inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static const int8_t scale_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static std::vector<float> make_values(int nval, int n_per_val, float scale = 16.f) {
    std::vector<float> result(nval*n_per_val);
    uint16_t m16 = ggml_fp32_to_fp16(0.922f);
    uint32_t m32 = (uint32_t(m16) << 16) | m16;
    const uint32_t a = 89226354, b = 64248484;
    float * data = result.data();
    for (int i = 0; i < nval; ++i) {
        uint32_t x = i + 4096;
        for (int k = 0; k < n_per_val; ++k) {
            x = a*x + b;
            uint32_t s = (x & 0b10001111111111111000111111111111) ^ m32;
            float val = ggml_fp16_to_fp32(s & 65535) + ggml_fp16_to_fp32(s >> 16);
            int ival = nearest_int(scale*val);
            data[k] = ival;
        }
        data += n_per_val;
    }
    return result;
}

#ifdef __AVX2__
static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
static __m256 hsum_float_8x8(__m256 * accm) {
     for (int i = 0; i < 4; ++i) {
         accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
                                   _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
     }
     for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i+2]), _mm256_unpackhi_ps(accm[i], accm[i+2]));
     return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}
#endif

const int8_t scale_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_scale(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = scale_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}
inline int best_index_iq4nl(const int8_t * values, float x) { return best_index_scale(values, x); }

static float find_best_scale(int block_size, const float * xb, const float * weight, const int8_t * values, int ntry) {
    float amax = 0, max = 0;
    for (int j = 0; j < block_size; ++j) {
        float ax = fabsf(xb[j]);
        if (ax > amax) {
            amax = ax; max = xb[j];
        }
    }
    return amax/96.f; //120.f; //127.f;
    if (!amax) return 0.f;
    float d = ntry > 0 ? -max/values[0] : max/values[0];
    float id = 1/d;
    float sumqx_p = 0, sumq2_p = 0;
    float sumqx_m = 0, sumq2_m = 0;
    for (int j = 0; j < block_size; ++j) {
        float w = weight[j];
        float al = id*xb[j];
        int l = best_index_iq4nl(values, al);
        float q = values[l];
        sumqx_p += w*q*xb[j];
        sumq2_p += w*q*q;
        l = best_index_iq4nl(values, -al);
        q = values[l];
        sumqx_m += w*q*xb[j];
        sumq2_m += w*q*q;
    }
    d = sumqx_p/sumq2_p;
    float best = d*sumqx_p;
    if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
        d = sumqx_m/sumq2_m; best = d*sumqx_m;
    }
    for (int itry = -ntry; itry <= ntry; ++itry) {
        id = (itry + values[0])/max;
        sumqx_p = sumq2_p = 0;
        sumqx_m = sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j];
            sumq2_p += w*q*q;
            l = best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j];
            sumq2_m += w*q*q;
        }
        if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
            d = sumqx_p/sumq2_p; best = d * sumqx_p;
        }
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d * sumqx_m;
        }
    }
    return d;
}

static std::vector<float> cluster_points(const std::vector<float>& points, int ndim, int ncluster, int niter) {
    if (points.size() % ndim != 0) {
        printf("%s: bad input\n", __func__); return {};
    }
    int npoint = points.size() / ndim;
    if (npoint < 2*ncluster) {
        printf("%s: bad input\n", __func__); return {};
    }
    std::vector<std::pair<float, float>> range(ndim, std::make_pair(INFINITY, -INFINITY));
    double Fo = 0;
    for (int i = 0; i < npoint; ++i) {
        auto v = points.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) {
            Fo += v[k]*v[k];
            range[k].first  = std::min(range[k].first, v[k]);
            range[k].second = std::max(range[k].second, v[k]);
        }
    }
    printf("%s (ndim = %d, npoint = %d): Fo = %g\n", __func__, ndim, npoint, Fo/points.size());
    std::mt19937 rndm(1234);
    float scale = 1.f/4294967296.f;
    std::vector<float> result(ncluster*ndim);
    for (int i = 0; i < ncluster; ++i) {
        auto v = result.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) v[k] = range[k].first + (range[k].second - range[k].first)*scale*rndm();
    }
    std::vector<float> sump(ncluster*ndim);
    std::vector<int> counts(ncluster);
    std::vector<int> which_cluster(npoint, -1);
    double Flast = Fo;
    for (int iter = 0; iter < niter; ++iter) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        int nchanged = 0;
        double F = 0;
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            float best = INFINITY; int ibest = -1;
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = result.data() + ndim*ic;
                float dist2 = 0;
                for (int k = 0; k < ndim; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                if (dist2 < best) {
                    best = dist2; ibest = ic;
                }
            }
            if (ibest < 0) { printf("Oops.\n"); exit(1); }
            F += best;
            if (which_cluster[ip] != ibest) ++nchanged;
            which_cluster[ip] = ibest;
            ++counts[ibest];
            auto vc = sump.data() + ndim*ibest;
            for (int k = 0; k < ndim; ++k) vc[k] += vp[k];
        }
        if (nchanged == 0) break;
        for (int ic = 0; ic < ncluster; ++ic) {
            float norm = counts[ic] > 0 ? 1.f/counts[ic] : 0.f;
            auto vc = sump.data() + ndim*ic;
            auto r  = result.data() + ndim*ic;
            for (int k = 0; k < ndim; ++k) r[k] = vc[k]*norm;
        }
        printf("%s(iteration %d): F = %g, nchanged = %d\n", __func__, iter+1, F/points.size(), nchanged);
        if (iter > 1 && Flast/F - 1 < 1e-6) break;
        Flast = F;
    }
    return result;
}

static void analyze_x_v2(const char * name, int nrows, int n_per_row, const float * values, float& tot_mse, float& tot_mse_q, float& tot_elements) {
    constexpr int kNumVal = 1 << 15;
    constexpr int kBlockSize = 32;
    constexpr int kGroupSize = 8;
    constexpr int kNg = kBlockSize/kGroupSize;
    constexpr int kSuperBlockSize = 256;
    static_assert(kNumVal%8 == 0);
    static std::vector<float> codes, clusters;
    static std::vector<std::vector<int>> p_in_cluster;
    if (codes.empty()) {
        codes = make_values(kNumVal, kGroupSize, 31.75f);
        clusters = cluster_points(codes, kGroupSize, kNumVal/512, 200);
        if (clusters.empty()) { printf("Oops\n"); exit(1); }
        int ncluster = clusters.size()/kGroupSize;
        p_in_cluster.resize(ncluster);
        std::vector<int> which_cluster(4*kNumVal);
        GGML_ASSERT(ncluster%8 == 0);
        for (int ip = 0; ip < kNumVal; ++ip) {
            auto vp = codes.data() + ip*kGroupSize;
            float best[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
            int ibest[4] = {-1, -1, -1, -1};
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = clusters.data() + ic*kGroupSize;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                if (dist2 < best[0]) {
                    best[3] = best[2]; ibest[3] = ibest[2];
                    best[2] = best[1]; ibest[2] = ibest[1];
                    best[1] = best[0]; ibest[1] = ibest[0];
                    best[0] = dist2;   ibest[0] = ic;
                }
                else if (dist2 < best[1]) {
                    best[3] = best[2]; ibest[3] = ibest[2];
                    best[2] = best[1]; ibest[2] = ibest[1];
                    best[1] = dist2;   ibest[1] = ic;
                }
                else if (dist2 < best[2]) {
                    best[3] = best[2]; ibest[3] = ibest[2];
                    best[2] = dist2;   ibest[2] = ic;
                }
                else if (dist2 < best[3]) {
                    best[3] = dist2;   ibest[3] = ic;
                }
            }
            GGML_ASSERT(ibest[0] >= 0 && ibest[1] >= 0 && ibest[2] >= 0 && ibest[3] >= 0);
            p_in_cluster[ibest[0]].push_back(ip);
            p_in_cluster[ibest[1]].push_back(ip);
            p_in_cluster[ibest[2]].push_back(ip);
            p_in_cluster[ibest[3]].push_back(ip);
            std::memcpy(which_cluster.data() + 4*ip, ibest, 4*sizeof(int));
        }
        std::vector<std::pair<float, int>> extra;
        extra.reserve(kNumVal);
        for (int ic = 0; ic < ncluster; ++ic) {
            auto& points = p_in_cluster[ic];
            if (!points.empty() && points.size()%8 == 0) continue;
            extra.clear();
            auto vc = clusters.data() + ic*kGroupSize;
            for (int ip = 0; ip < kNumVal; ++ip) {
                if (which_cluster[4*ip] == ic || which_cluster[4*ip+1] == ic || which_cluster[4*ip+2] == ic || which_cluster[4*ip+3] == ic) continue;
                auto vp = codes.data() + ip*kGroupSize;
                float dist2 = 0;
                for (int k = 0; k < kGroupSize; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                extra.push_back(std::make_pair(dist2, ip));
            }
            std::sort(extra.begin(), extra.end());
            int nadd = 8*((points.size()+7)/8) - points.size();
            for (int i = 0; i < nadd; ++i) points.push_back(extra[i].second);
            GGML_ASSERT(points.size()%8 == 0);
        }
        auto min = p_in_cluster.front().size(), max = p_in_cluster.front().size();
        int nzero = 0;
        for (auto& points : p_in_cluster) {
            min = std::min(min, points.size());
            max = std::max(max, points.size());
            if (points.empty()) ++nzero;
        }
        printf("%s: prepared %d clusters\n", __func__, ncluster);
        printf("    min number of points in a cluster: %d\n", int(min));
        printf("    max number of points in a cluster: %d\n", int(max));
        if (nzero > 0) {
            printf("    there are %d empty clusters\n", nzero);
            for (auto& points : p_in_cluster) {
                if (!points.empty()) continue;
                points.reserve(kNumVal);
                for (int j = 0; j < kNumVal; ++j) points.push_back(j); // i.e., if we end iup picking an empty cluster, we just check all points
            }
        }
    }
    int nthread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int chunk = (nrows + 8*nthread - 1)/(8*nthread);
    std::mutex mutex;
    int counter = 0;
    float mse = 0, mse_q = 0;
    auto compute = [&mutex, &counter, &mse, &mse_q, values, nrows, n_per_row, chunk] () {
        constexpr int kNumVal = 1 << 15;
        constexpr int kBlockSize = 32;
        constexpr int kGroupSize = 8;
        constexpr int kNg = kBlockSize/kGroupSize;
        double lmse = 0, lmse_q = 0;
        std::vector<float> scales(n_per_row/kBlockSize);
        std::vector<int> best_idx(n_per_row/kGroupSize);
        std::vector<float> weight(kBlockSize, 1.f);
        int ncluster = clusters.size() / kGroupSize;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int first = counter; counter += chunk;
            if (first >= nrows) {
                mse += lmse; mse_q += lmse_q;
                return;
            }
            lock.unlock();
            int last = std::min(first + chunk, nrows);
#ifdef __AVX2__
            __m256 sqx[8];
            __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            float sx[8];
            int   index[8];
#endif
            for (int row = first; row < last; ++row) {
                auto xr = values + row*n_per_row;
                float sigma2 = 0;
                for (int j = 0; j < n_per_row; ++j) sigma2 += xr[j]*xr[j];
                sigma2 /= n_per_row;
                for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                    auto xb = xr + kBlockSize*ib;
                    //for (int i = 0; i < kBlockSize; ++i) weight[i] = 0.25f*sigma2 + xb[i]*xb[i];
                    float d = find_best_scale(kBlockSize, xb, weight.data(), iq4k_values, 5);
                    float id = d ? 1/d : 0.f;
#ifdef __AVX2__
                    auto vid = _mm256_set1_ps(id);
                    for (int l = 0; l < kNg; ++l) {
                        auto xl = xb + 8*l;
                        auto wl = weight.data() + 8*l;
                        auto vx = _mm256_mul_ps(vid, _mm256_loadu_ps(xl));
                        auto vw = _mm256_loadu_ps(wl);
                        auto vbest = _mm256_set1_ps(INFINITY);
                        auto best_index = _mm256_set1_epi32(-1);
                        float best = INFINITY; int jbest = -1;
                        for (int j = 0; j < ncluster; j += 8) {
                            auto idx = _mm256_add_epi32(_mm256_set1_epi32(j), add_idx);
                            for (int i = 0; i < 8; ++i) {
                                auto vq = _mm256_loadu_ps(clusters.data() + kGroupSize*(j+i));
                                auto vdiff = _mm256_sub_ps(vq, vx);
                                sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                            }
                            auto score = hsum_float_8x8(sqx);
                            auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                            best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                                      _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                            vbest = _mm256_min_ps(vbest, score);
                        }
                        _mm256_store_ps(sx, vbest);
                        _mm256_store_si256((__m256i *)index, best_index);
                        for (int i = 0; i < 8; ++i) {
                            if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                        }
                        auto& points = p_in_cluster[jbest];
                        if (points.empty()) {
                            printf("Oops: empty cluster %d\n", jbest);
                            auto vc = clusters.data() + kGroupSize*jbest;
                            printf("Cluster:\n");
                            for (int j = 0; j < kGroupSize; ++j) printf("%d  %g  %g\n", j, vc[j], xl[j]);
                            GGML_ASSERT(false);
                        }
                        int jbest_cluster = jbest;
                        vbest = _mm256_set1_ps(INFINITY);
                        best_index = _mm256_set1_epi32(-1);
                        best = INFINITY; jbest = -1;
                        for (int j = 0; j < int(points.size()); j += 8) {
                            auto idx = _mm256_loadu_si256((const __m256i*)(points.data() + j));
                            for (int i = 0; i < 8; ++i) {
                                auto vq = _mm256_loadu_ps(codes.data() + kGroupSize*points[j+i]);
                                auto vdiff = _mm256_sub_ps(vq, vx);
                                sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                            }
                            auto score = hsum_float_8x8(sqx);
                            auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                            best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                                      _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                            vbest = _mm256_min_ps(vbest, score);
                        }
                        _mm256_store_ps(sx, vbest);
                        _mm256_store_si256((__m256i *)index, best_index);
                        for (int i = 0; i < 8; ++i) {
                            if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                        }
                        if (jbest < 0) {
                            printf("Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                            GGML_ASSERT(false);
                        }
                        GGML_ASSERT(jbest >= 0);
                        best_idx[ib*kNg + l] = jbest;
                    }
                    auto vqx = _mm256_setzero_ps();
                    auto vq2 = _mm256_setzero_ps();
                    for (int l = 0; l < kNg; ++l) {
                        auto vx = _mm256_loadu_ps(xb+8*l);
                        auto vw = _mm256_loadu_ps(weight.data() + 8*l);
                        auto vq = _mm256_loadu_ps(codes.data() + kGroupSize*best_idx[ib*kNg + l]);
                        auto vqw = _mm256_mul_ps(vq, vw);
                        vqx = _mm256_fmadd_ps(vqw, vx, vqx);
                        vq2 = _mm256_fmadd_ps(vqw, vq, vq2);
                    }
                    auto sumqx = hsum_float_8(vqx);
                    auto sumq2 = hsum_float_8(vq2);
                    scales[ib] = sumq2 > 0 ? sumqx/sumq2 : 0.f;
#else
#endif
                }
                float amax_scale = std::abs(scales[0]);
                float max_scale  = scales[0];
                for (int ib = 1; ib < n_per_row/kBlockSize; ++ib) {
                    float ax = std::abs(scales[ib]);
                    if (ax > amax_scale) {
                        amax_scale = ax;
                        max_scale = scales[ib];
                    }
                }
                float d = max_scale/scale_values[0];
                float id = d ? 1/d : 0.f;
                for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                    int ls = best_index_scale(scale_values, id*scales[ib]);
                    float dl = d * scale_values[ls];
                    auto xb = xr + kBlockSize*ib;
                    for (int l = 0; l < kNg; ++l) {
                        auto q = codes.data() + kGroupSize*best_idx[ib*kNg+l];
                        for (int k = 0; k < kGroupSize; ++k) {
                            float diff1 = xb[kGroupSize*l + k] - scales[ib]*q[k];
                            float diff2 = xb[kGroupSize*l + k] - dl*q[k];
                            lmse += diff1*diff1;
                            lmse_q += diff2*diff2;
                        }
                    }
                }
            }
        }
    };
    std::vector<std::thread> workers(nthread);
    for (auto& w : workers) w = std::thread(compute);
    for (auto& w : workers) w.join();
    tot_mse += mse;
    tot_mse_q += mse_q;
    tot_elements += n_per_row*nrows;
    printf("%s:   %g    %g      %g   %g\n", name, sqrt(mse/(n_per_row*nrows)), sqrt(tot_mse/tot_elements),
            sqrt(mse_q/(n_per_row*nrows)), sqrt(tot_mse_q/tot_elements));
}

static void analyze_x(const char * name, int nrows, int n_per_row, const float * values, float& tot_mse, float& tot_mse_q, float& tot_elements) {
    constexpr int kNumVal = 1 << 12;
    constexpr int kBlockSize = 8;
    constexpr int kSuperBlockSize = 256;
    static_assert(kNumVal%8 == 0);
    auto codes = make_values(kNumVal, kBlockSize);
    std::vector<float> sumq2i(kNumVal);
    for (int j = 0; j < kNumVal; ++j) {
        auto data = codes.data() + kBlockSize*j;
        float sum = 0; for (int k = 0; k < kBlockSize; ++k) sum += data[k]*data[k];
        sumq2i[j] = sum > 0 ? 1/sum : 0.f;;
    }
    int nthread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int chunk = (nrows + 8*nthread - 1)/(8*nthread);
    std::mutex mutex;
    int counter = 0;
    float mse = 0, mse_q = 0;
    auto compute = [&mutex, &counter, &mse, &mse_q, &codes, &sumq2i, values, nrows, n_per_row, chunk] () {
        constexpr int kBlockSize = 8;
        constexpr int kNumVal = 1 << 12;
        float lmse = 0, lmse_q = 0;
        std::vector<float> scales(n_per_row/kBlockSize);
        std::vector<int> best_idx(n_per_row/kBlockSize);
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int first = counter; counter += chunk;
            if (first >= nrows) {
                mse += lmse; mse_q += lmse_q;
                return;
            }
            lock.unlock();
            int last = std::min(first + chunk, nrows);
#ifdef __AVX2__
            __m256 vx[kBlockSize/8];
            __m256 sqx[8];
            __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            float sx[8];
            int   index[8];
#endif
            for (int row = first; row < last; ++row) {
                auto xr = values + row*n_per_row;
                for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                    float best = 0, d = 0; int jbest = -1;
                    auto xb = xr + kBlockSize*ib;
#ifdef __AVX2__
                    for (int l = 0; l < kBlockSize/8; ++l) {
                        vx[l] = _mm256_loadu_ps(xb+8*l);
                    }
                    auto vbest = _mm256_set1_ps(0.f);
                    auto best_index = _mm256_set1_epi32(-1);
                    for (int j = 0; j < kNumVal; j += 8) {
                        auto idx = _mm256_add_epi32(_mm256_set1_epi32(j), add_idx);
                        for (int i = 0; i < 8; ++i) {
                            sqx[i] = _mm256_setzero_ps();
                            for (int l = 0; l < kBlockSize/8; ++l) {
                                auto qv = _mm256_loadu_ps(codes.data() + kBlockSize*(j+i) + 8*l);
                                sqx[i] = _mm256_fmadd_ps(vx[l], qv, sqx[i]);
                            }
                        }
                        auto sumqx = hsum_float_8x8(sqx);
                        auto score = _mm256_mul_ps(_mm256_mul_ps(sumqx, sumqx), _mm256_loadu_ps(sumq2i.data() + j));
                        auto mask  = _mm256_cmp_ps(score, vbest, _CMP_GT_OQ);
                        best_index = _mm256_or_si256(_mm256_and_si256(idx, _mm256_castps_si256(mask)), _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                        vbest = _mm256_max_ps(vbest, score);
                    }
                    _mm256_store_ps(sx, vbest);
                    _mm256_store_si256((__m256i *)index, best_index);
                    best = sx[0]; jbest = index[0];
                    for (int j = 1; j < 8; ++j) {
                        if (sx[j] > best) { best = sx[j]; jbest = index[j]; }
                    }
                    auto qv = codes.data() + kBlockSize*jbest;
                    float sumqx = 0;
                    for (int k = 0; k < kBlockSize; ++k) sumqx += xb[k]*qv[k];
                    d = sumqx*sumq2i[jbest];
#else
                    for (int j = 0; j < kNumVal; ++j) {
                        if (!sumq2i[j]) continue;
                        auto qv = codes.data() + kBlockSize*j;
                        float sumqx = 0;
                        for (int k = 0; k < kBlockSize; ++k) sumqx += qv[k]*xb[k];
                        if (sumqx*sumqx*sumq2i[j] > best) {
                            d = sumqx*sumq2i[j]; best = d*sumqx; jbest = j;
                        }
                    }
                    auto qv = codes.data() + kBlockSize*jbest;
#endif
                    scales[ib] = d;
                    best_idx[ib] = jbest;
                    for (int k = 0; k < kBlockSize; ++k) {
                        float diff = xb[k] - d*qv[k];
                        lmse += diff*diff;
                    }
                }
                float amax_scale = std::abs(scales[0]);
                float max_scale  = scales[0];
                for (int ib = 1; ib < n_per_row/kBlockSize; ++ib) {
                    float ax = std::abs(scales[ib]);
                    if (ax > amax_scale) {
                        amax_scale = ax;
                        max_scale = scales[ib];
                    }
                }
                float d = max_scale/scale_values[0];
                float id = d ? 1/d : 0.f;
                for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                    int ls = best_index_scale(scale_values, id*scales[ib]);
                    float dl = d * scale_values[ls];
                    auto xb = xr + kBlockSize*ib;
                    auto qv = codes.data() + kBlockSize*best_idx[ib];
                    for (int k = 0; k < kBlockSize; ++k) {
                        float diff = xb[k] - dl*qv[k];
                        lmse_q += diff*diff;
                    }
                }
            }
        }
    };
    std::vector<std::thread> workers(nthread);
    for (auto& w : workers) w = std::thread(compute);
    for (auto& w : workers) w.join();
    tot_mse += mse;
    tot_mse_q += mse_q;
    tot_elements += n_per_row*nrows;
    printf("%s:   %g    %g      %g   %g\n", name, sqrt(mse/(n_per_row*nrows)), sqrt(tot_mse/tot_elements),
            sqrt(mse_q/(n_per_row*nrows)), sqrt(tot_mse_q/tot_elements));
}

static const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
static inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}

static void analyze_iq2kl([[maybe_unused]] const char * name,
                          [[maybe_unused]] int nrows,
                          [[maybe_unused]] int n_per_row,
                          [[maybe_unused]] const float * x_values,
                          [[maybe_unused]] const float * imatrix,
                          [[maybe_unused]] float& tot_mse,
                          [[maybe_unused]] float& tot_elements) {
#if 0
    constexpr int kBlockSize = 32;
    constexpr int ntry = 5;
    static const int k_index[64] = {-1, 0, -2, 1, -3, -4, 2, -5, -6, -7, -8, 3, -9, 4, -10, -11, 5, 6, 7, -12, 8, 9, 10, 11, -13, -14, -15, -16, 12, 13,
        -17, -18, -19, -20, 14, 15, 16, 17, 18, -21, 19, 20, 21, 22, 23, 24, -22, 25, -23, -24, 26, -25, 27, -26, 28, -27, -28, 29, -29, 30, -30, -31, 31, -32,};
    static const std::vector<std::vector<int>> k_neighbours = {
        { 0, 5, 6, 1, 7, 3, 8, 14,  },
        { 1, 0, 3, 7, 4, 6, 8, 2,  },
        { 1, 3, 4, 2, 8, 0, 9, 7,  },
        { 2, 1, 4, 3, 9, 8, 10, 11,  },
        { 2, 11, 4, 10, 9, 1, 8, 3,  },
        { 5, 6, 0, 7, 3, 19, 14, 1,  },
        { 6, 0, 7, 5, 3, 1, 8, 14,  },
        { 3, 7, 6, 1, 0, 8, 4, 12,  },
        { 3, 4, 8, 9, 1, 7, 12, 10,  },
        { 4, 10, 9, 2, 11, 8, 13, 3,  },
        { 11, 10, 2, 4, 9, 18, 13, 8,  },
        { 8, 7, 3, 12, 9, 15, 16, 13,  },
        { 5, 19, 6, 20, 14, 7, 21, 15,  },
        { 6, 14, 7, 20, 5, 21, 15, 19,  },
        { 14, 7, 15, 6, 21, 12, 16, 22,  },
        { 12, 15, 16, 8, 14, 7, 13, 22,  },
        { 18, 10, 13, 17, 9, 11, 12, 24,  },
        { 11, 18, 25, 10, 13, 17, 9, 24,  },
        { 19, 5, 20, 6, 14, 21, 7, 26,  },
        { 20, 14, 21, 6, 19, 7, 15, 26,  },
        { 25, 18, 11, 10, 28, 17, 13, 24,  },
        { 18, 24, 28, 25, 17, 23, 13, 16,  },
        { 19, 20, 29, 26, 21, 14, 5, 22,  },
        { 20, 26, 29, 21, 19, 14, 22, 30,  },
        { 27, 26, 22, 23, 30, 21, 15, 24,  },
        { 27, 24, 28, 23, 31, 17, 22, 16,  },
        { 25, 28, 31, 18, 24, 17, 27, 23,  },
        { 29, 19, 20, 26, 21, 30, 14, 22,  },
        { 30, 29, 26, 27, 21, 22, 20, 23,  },
        { 30, 27, 31, 26, 28, 23, 22, 24,  },
        { 31, 27, 30, 28, 24, 23, 26, 22,  },
        { 31, 28, 25, 24, 18, 27, 30, 17,  },
    };
    //static const int k_index[64] = {-1, -2, -3, 0, -4, -5, -6, -7, -8, 1, -9, -10, -11, 2, 3, -12, -13, -14, 4, 5, 6, 7, 8, -15, 9, -16, 10, 11, 12, 13, 14,
    //    -17, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, 25, -22, -23, 26, 27, 28, 29, 30, 31, -24, -25, -26, -27, -28, -29, -30, -31, -32,};
    //static const std::vector<std::vector<int>> k_neighbours = {
    //    { 1, 4,  },
    //    { 1, 0, 4, 5,  },
    //    { 0, 1, 4, 5, 6,  },
    //    { 0, 2, 6, 3, 5, 7, 4, 8,  },
    //    { 2, 3, 0, 7, 6, 8, 5,  },
    //    { 3, 2, 8, 7, 6,  },
    //    { 3, 2, 8, 7,  },
    //    { 1, 9, 4, 10,  },
    //    { 1, 4, 0, 5, 10, 6, 11, 9,  },
    //    { 0, 5, 4, 6, 1, 2, 11, 7,  },
    //    { 2, 6, 0, 5, 7, 3, 12, 4,  },
    //    { 3, 8, 2, 7, 14, 13,  },
    //    { 9, 1, 4, 10, 15,  },
    //    { 1, 4, 9, 10, 5, 11, 15, 0,  },
    //    { 8, 3, 14, 7, 2, 13, 19, 18,  },
    //    { 9, 10, 4, 15, 1, 11, 20, 5,  },
    //    { 14, 8, 19, 13, 3, 7, 18, 25,  },
    //    { 9, 20, 15, 10, 21, 26, 4, 27,  },
    //    { 15, 20, 9, 10, 21, 16, 26, 4,  },
    //    { 19, 14, 25, 18, 8, 13, 24, 31,  },
    //    { 20, 26, 9, 21, 15, 27, 10,  },
    //    { 25, 19, 31, 24, 14, 18, 30, 13,  },
    //    { 26, 20, 27, 21, 15,  },
    //    { 31, 25, 30, 19, 24, 18,  },
    //    { 26, 20, 27, 21,  },
    //    { 26, 27, 20, 21, 28, 22,  },
    //    { 27, 26, 28, 21, 20, 22, 29, 23,  },
    //    { 28, 27, 29, 22, 21, 23, 26, 30,  },
    //    { 29, 28, 30, 23, 22, 24, 27, 31,  },
    //    { 30, 29, 31, 24, 23, 25, 28, 22,  },
    //    { 31, 30, 25, 24, 29, 23,  },
    //    { 31, 25, 30, 24,  },
    //};
    auto values = iq3nl_values;
    std::vector<std::pair<int8_t, int8_t>> grid(32);
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }
    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        // d*q - x1 = d*(q - x1/d)
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        //printf("sx1 = %g, sx2 = %g, l1 = %d, l2 = %d, %d neighbours\n", sx1, sx2, l1, l2, int(neigh.size()));
        for (auto& n : neigh) {
            //printf("  neigh %d,%d: %d %d\n", grid[n].first, grid[n].second, values[grid[n].first ], values[grid[n].second]);
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };
    auto compute_1row = [&] (const float * xr) {
        float weight[kBlockSize];
        int nblock = n_per_row/kBlockSize;
        int last_ibl = -1;
        float sigma2 = 0;
        float mse = 0, sum_x2 = 0;
        for (int ib = 0; ib < nblock; ++ib) {
            auto xb = xr + ib*kBlockSize;
            int ibl = ib/8;
            if (ibl != last_ibl) {
                int n = std::min(256, n_per_row - ib*kBlockSize);
                float sumx2 = 0;
                for (int j = 0; j < n; ++j) sumx2 += xb[j]*xb[j];
                sigma2 = 2*sumx2/n;
                last_ibl = ibl;
            }
            if (imatrix) {
                auto qw = imatrix + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            id = 1/d;
            float block_mse = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                int idx = index(id, xb[j+0], xb[j+1], weight[j], weight[j+1]);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                float diff1 = d*q1 - xb[j+0];
                float diff2 = d*q2 - xb[j+1];
                block_mse += diff1*diff1 + diff2*diff2;
                sum_x2 += xb[j+0]*xb[j+0] + xb[j+1]*xb[j+1];
            }
            mse += block_mse;
        }
        return std::make_pair(mse, sum_x2);
    };
    std::mutex mutex;
    int counter = 0;
    float mse = 0, sum_x2 = 0;
    auto compute = [&mutex, &counter, &compute_1row, &mse, &sum_x2, x_values, nrows, n_per_row] () {
        float local_mse = 0, local_x2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int row = counter++;
            if (row >= nrows) {
                mse += local_mse; sum_x2 += local_x2;
                return;
            }
            lock.unlock();
            auto [row_mse, row_x2] = compute_1row(x_values + row*n_per_row);
            local_mse += row_mse;
            local_x2  += row_x2;
        }
    };
    int nthread = std::thread::hardware_concurrency()/2;
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();
    //float weight[kBlockSize];
    //int nblock = n_per_row/kBlockSize;
    //int last_ibl = -1;
    //float sigma2 = 0;
    //auto shifted_values = values + 8;
    //float mse = 0, sum_x2 = 0;
    //for (int row = 0; row < nrows; ++row) {
    //    auto xr = x_values + row*n_per_row;
    //    for (int ib = 0; ib < nblock; ++ib) {
    //        auto xb = xr + ib*kBlockSize;
    //        int ibl = ib/8;
    //        if (ibl != last_ibl) {
    //            int n = std::min(256, n_per_row - ib*kBlockSize);
    //            float sumx2 = 0;
    //            for (int j = 0; j < n; ++j) sumx2 += xb[j]*xb[j];
    //            sigma2 = 2*sumx2/n;
    //            last_ibl = ibl;
    //        }
    //        if (imatrix) {
    //            auto qw = imatrix + ib*kBlockSize;
    //            for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
    //        } else {
    //            for (int j = 0; j < kBlockSize; ++j) weight[j] = xb[j]*xb[j];
    //        }
    //        float amax = 0, max = 0;
    //        for (int j = 0; j < kBlockSize; ++j) {
    //            float ax = std::abs(xb[j]);
    //            if (ax > amax) {
    //                amax = ax; max = xb[j];
    //            }
    //        }
    //        if (!amax) {
    //            continue;
    //        }
    //        float d = ntry > 0 ? -max/values[0] : max/values[0];
    //        float id = 1/d;
    //        float sumqx_p = 0, sumq2_p = 0;
    //        float sumqx_m = 0, sumq2_m = 0;
    //        for (int j = 0; j < kBlockSize; j += 2) {
    //            float w1 = weight[j+0];
    //            float w2 = weight[j+1];
    //            int idx = index(id, xb[j+0], xb[j+1], w1, w2);
    //            float q1 = grid[idx].first ;
    //            float q2 = grid[idx].second;
    //            sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
    //            sumq2_p += w1*q1*q1 + w2*q2*q2;
    //            idx = index(-id, xb[j+0], xb[j+1], w1, w2);
    //            q1 = grid[idx].first ;
    //            q2 = grid[idx].second;
    //            sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
    //            sumq2_m += w1*q1*q1 + w2*q2*q2;
    //        }
    //        d = sumqx_p/sumq2_p;
    //        float best = d*sumqx_p;
    //        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
    //            d = sumqx_m/sumq2_m; best = d*sumqx_m;
    //        }
    //        for (int itry = -ntry; itry <= ntry; ++itry) {
    //            id = (itry + values[0])/max;
    //            sumqx_p = sumq2_p = 0;
    //            sumqx_m = sumq2_m = 0;
    //            for (int j = 0; j < kBlockSize; j += 2) {
    //                float w1 = weight[j+0];
    //                float w2 = weight[j+1];
    //                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
    //                float q1 = grid[idx].first ;
    //                float q2 = grid[idx].second;
    //                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
    //                sumq2_p += w1*q1*q1 + w2*q2*q2;
    //                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
    //                q1 = grid[idx].first ;
    //                q2 = grid[idx].second;
    //                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
    //                sumq2_m += w1*q1*q1 + w2*q2*q2;
    //            }
    //            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
    //                d = sumqx_p/sumq2_p; best = d * sumqx_p;
    //            }
    //            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
    //                d = sumqx_m/sumq2_m; best = d * sumqx_m;
    //            }
    //        }
    //        id = 1/d;
    //        float block_mse = 0;
    //        for (int j = 0; j < kBlockSize; j += 2) {
    //            int idx = index(id, xb[j+0], xb[j+1], weight[j], weight[j+1]);
    //            float q1 = grid[idx].first ;
    //            float q2 = grid[idx].second;
    //            float diff1 = d*q1 - xb[j+0];
    //            float diff2 = d*q2 - xb[j+1];
    //            block_mse += diff1*diff1 + diff2*diff2;
    //            sum_x2 += xb[j+0]*xb[j+0] + xb[j+1]*xb[j+1];
    //        }
    //        mse += block_mse;
    //    }
    //}
    tot_mse += mse;
    tot_elements += sum_x2;
    printf("%s:  %g, %g  %g\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(mse/sum_x2), sqrt(tot_mse/tot_elements));
#endif
}

static void analyze_iq3ks(const char * name, int nrows, int n_per_row, const float * x_values, const float * imatrix, float& tot_mse, float& tot_elements,
        std::vector<int64_t>& Htot) {
    constexpr int kBlockSize = 32;
    constexpr int ntry = 5;
    float weight[kBlockSize];
    int nblock = n_per_row/kBlockSize;
    int last_ibl = -1;
    float sigma2 = 0;
    auto values = iq3nl_values;
    auto shifted_values = values + 8;
    std::vector<int64_t> H(64, 0);
    float mse = 0;
    for (int row = 0; row < nrows; ++row) {
        auto xr = x_values + row*n_per_row;
        for (int ib = 0; ib < nblock; ++ib) {
            auto xb = xr + ib*kBlockSize;
            int ibl = ib/8;
            if (ibl != last_ibl) {
                int n = std::min(256, n_per_row - ib*kBlockSize);
                float sumx2 = 0;
                for (int j = 0; j < n; ++j) sumx2 += xb[j]*xb[j];
                sigma2 = 2*sumx2/n;
                last_ibl = ibl;
            }
            if (imatrix) {
                auto qw = imatrix + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (!amax) {
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                //id = (itry + shifted_values[0])/max;
                //sumqx_p = sumq2_p = 0;
                //sumqx_m = sumq2_m = 0;
                //for (int j = 0; j < kBlockSize; ++j) {
                //    float w = weight[j];
                //    float al = id*xb[j];
                //    int l = best_index_iq3nl(shifted_values, al);
                //    float q = shifted_values[l];
                //    sumqx_p += w*q*xb[j];
                //    sumq2_p += w*q*q;
                //    l = best_index_iq3nl(shifted_values, -al);
                //    q = shifted_values[l];
                //    sumqx_m += w*q*xb[j];
                //    sumq2_m += w*q*q;
                //}
                //if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                //    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                //}
                //if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                //    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                //}
            }
            auto block_values = is_shifted ? shifted_values : values;
            id = 1/d;
            float block_mse = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                int l1 = best_index_iq3nl(block_values, id*xb[j+0]);
                int l2 = best_index_iq3nl(block_values, id*xb[j+1]);
                float diff1 = d*block_values[l1] - xb[j+0];
                float diff2 = d*block_values[l2] - xb[j+1];
                block_mse += diff1*diff1 + diff2*diff2;
                ++H[8*l1+l2];
            }
            mse += block_mse;
        }
    }
    tot_mse += mse;
    tot_elements += nrows*n_per_row;
    printf("%s:  %g  %f\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(tot_mse/tot_elements));

    if (Htot.empty()) Htot = std::move(H);
    else {
        if (Htot.size() != H.size()) printf("Oops: inconsistent H sizes %zu vs %zu\n", H.size(), Htot.size());
        else for (int j = 0; j < (int)H.size(); ++j) Htot[j] += H[j];
    }
}

static void analyze_iq4ks(const char * name, int nrows, int n_per_row, const float * values, float& tot_mse, float& tot_elements) {
    int row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    int nthread = std::max(1, int(std::thread::hardware_concurrency()/2));
    int chunk = (nrows + 8*nthread - 1)/(8*nthread);
    std::mutex mutex;
    int counter = 0;
    float mse0 = 0, mse = 0;
    auto compute = [&mutex, &counter, &mse0, &mse, values, row_size, nblock, nrows, n_per_row, chunk] () {
        std::vector<char> Q(row_size);
        float diff[4];
        float xv[4];
        float lmse0 = 0, lmse = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int first = counter; counter += chunk;
            if (first >= nrows) {
                mse += lmse; mse0 += lmse0;
                return;
            }
            lock.unlock();
            int last = std::min(first + chunk, nrows);
            for (int row = first; row < last; ++row) {
                auto xr = values + row*n_per_row;
                ggml_quantize_chunk(GGML_TYPE_IQ4_KS, xr, (void *)Q.data(), 0, 1, n_per_row, nullptr);
                const float * dptr = (const float *)Q.data();
                const float d = *dptr;
                const block_iq4_ks * iq4 = (const block_iq4_ks *)(dptr + 1);
                for (int ibl = 0; ibl < nblock; ++ibl) {
                    const float * xbl = xr + ibl*QK_K;
                    auto qs = iq4[ibl].qs;
                    for (int ib = 0; ib < QK_K/32; ++ib) {
                        const float * xb = xbl + 32*ib;
                        const float dl = d * ((iq4[ibl].scales[ib] & 254) - 127);
                        const int8_t * values = iq4k_values + ((iq4[ibl].scales[ib] & 1) << 4);
                        for (int j = 0; j < 16; j += 2) {
                            uint16_t v0 = *(const uint16_t *)(qs + j);
                            int non = popcount(v0);
                            xv[0] = xb[j+ 0]; xv[1] = xb[j+16]; xv[2] = xb[j+ 1]; xv[3] = xb[j+17];
                            diff[0] = xv[0] - dl*values[qs[j+0] & 0xf];
                            diff[1] = xv[1] - dl*values[qs[j+0] >>  4];
                            diff[2] = xv[2] - dl*values[qs[j+1] & 0xf];
                            diff[3] = xv[3] - dl*values[qs[j+1] >>  4];
                            float diff4 = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3];
                            lmse0 += diff4;
                            if (non%2 == 0) {
                                lmse += diff4;
                            } else {
                                float best = std::numeric_limits<float>::max();
                                for (int k = 0; k < 4; ++k) {
                                    uint16_t v = (v0 >> 4*k) & 0xf;
                                    auto pc = popcount(v);
                                    if (v > 0 && popcount(v-1u) != pc) {
                                        float this_diff = xv[k] - dl*values[v-1u];
                                        float score = diff4 - diff[k]*diff[k] + this_diff*this_diff;
                                        if (score < best) best = score;
                                    }
                                    if (v < 15 && popcount(v + 1u) != pc) {
                                        float this_diff = xv[k] - dl*values[v+1u];
                                        float score = diff4 - diff[k]*diff[k] + this_diff*this_diff;
                                        if (score < best) best = score;
                                    }
                                }
                                lmse += best;
                            }
                        }
                        qs += 16;
                    }
                }
            }
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();
    tot_mse += mse;
    tot_elements += n_per_row*nrows;
    printf("%s:  %g  %g    %g\n", name, sqrt(mse0/(n_per_row*nrows)), sqrt(mse/(n_per_row*nrows)), sqrt(tot_mse/tot_elements));
}

static void analyze_iq4ks(const ggml_tensor * t, float& tot_mse, float& tot_mse_q, float& tot_elements) {
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
        return;
    }
    if (t->type == GGML_TYPE_F32) {
        analyze_x_v2(t->name, t->ne[1], t->ne[0], (const float *)t->data, tot_mse, tot_mse_q, tot_elements);
    } else {
        std::vector<float> aux(t->ne[0]*t->ne[1]);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, aux.data(), aux.size());
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)t->data, aux.data(), aux.size());
        }
        analyze_x_v2(t->name, t->ne[1], t->ne[0], aux.data(), tot_mse, tot_mse_q, tot_elements);
    }
}

static void analyze_iq2kl(const ggml_tensor * t, float& tot_mse, float& tot_elements) {
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
        return;
    }
    if (t->type == GGML_TYPE_F32) {
        analyze_iq2kl(t->name, t->ne[1], t->ne[0], (const float *)t->data, nullptr, tot_mse, tot_elements);
    } else {
        std::vector<float> aux(t->ne[0]*t->ne[1]);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, aux.data(), aux.size());
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)t->data, aux.data(), aux.size());
        }
        analyze_iq2kl(t->name, t->ne[1], t->ne[0], aux.data(), nullptr, tot_mse, tot_elements);
    }
}

static void analyze_iq3ks(const ggml_tensor * t, float& tot_mse, float& tot_elements, std::vector<int64_t>& Htot) {
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
        return;
    }
    if (t->type == GGML_TYPE_F32) {
        analyze_iq3ks(t->name, t->ne[1], t->ne[0], (const float *)t->data, nullptr, tot_mse, tot_elements, Htot);
    } else {
        std::vector<float> aux(t->ne[0]*t->ne[1]);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, aux.data(), aux.size());
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)t->data, aux.data(), aux.size());
        }
        analyze_iq3ks(t->name, t->ne[1], t->ne[0], aux.data(), nullptr, tot_mse, tot_elements, Htot);
    }
}

static void print_fp_stats(const char * msg, const uint64_t * counts) {
    printf("===== %s\n", msg);
    uint64_t tot = 0; for (int i = 0; i < 32; ++i) tot += counts[i];
    double norm = 1./tot;
    for (int i = 0; i < 32; ++i) {
        if (!counts[i]) continue;
        uint16_t val = i << 10;
        float f = ggml_fp16_to_fp32(val);
        printf("%2d    %f   %g\n", i, norm*counts[i], f);
    }
}

static void analyze_tensor_fp(const ggml_tensor * t, uint64_t * H) {
    if (t->type != GGML_TYPE_F16) return;
    if (!ggml_is_contiguous(t)) return;
    int n = ggml_nelements(t);
    const uint16_t * x = (const uint16_t *)t->data;
    std::array<uint64_t, 32> counts = {};
    for (int j = 0; j < n; ++j) {
        ++counts[(x[j] >> 10) & 31];
    }
    for (int i = 0; i < 32; ++i) H[i] += counts[i];
    print_fp_stats(t->name, counts.data());
}

int main(int argc, char ** argv) {
    ggml_time_init();

    quantize_stats_params params;

    // read command line

    int max_thread = 0;
    bool invalid_param = false;
    bool analyze_fp = false;
    bool analyze = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            quantize_stats_print_usage(argc, argv);
            exit(0);
        } else if (arg == "-r" || arg == "--reference") {
            params.reference = true;
        } else if (arg == "-v") {
            params.verbose = true;
        } else if (arg == "-p" || arg == "--per-layer-stats") {
            params.per_layer_stats = true;
        } else if (arg == "-afp" || arg == "--analyze-fp") {
            analyze_fp = true;
        } else if (arg == "-a" || arg == "--analyze") {
            analyze = true;
        } else if (arg == "--histogram") {
            params.print_histogram = true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-l" || arg == "--include-layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_layers.emplace_back(argv[i]);
        } else if (arg == "-L" || arg == "--exclude-layer") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.exclude_layers.emplace_back(argv[i]);
        } else if (arg == "-t" || arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int j;
            for (j = 0; j < GGML_TYPE_COUNT; ++j) {
               const auto * name = ggml_type_name((ggml_type) j);
               if (name && strcmp(argv[i], name) == 0) break;
            }
            if (j < GGML_TYPE_COUNT) {
                params.include_types.push_back((ggml_type) j);
            } else {
                fprintf(stderr, "error: %s not in list of types\n", argv[i]);
                invalid_param = true;
            }
        } else if (arg == "-n" || arg == "--num-threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            max_thread = atoi(argv[i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            quantize_stats_print_usage(argc, argv);
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        quantize_stats_print_usage(argc, argv);
        return 1;
    }

    print_build_info();

    // load the model
    fprintf(stderr, "Loading model\n");

    const int64_t t_main_start_us = ggml_time_us();
    llama_model * model;
    llama_context * ctx;

    {
        auto mparams = llama_model_default_params();
        mparams.use_mlock  = false;

        model = llama_model_load_from_file(params.model.c_str(), mparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        auto cparams = llama_context_default_params();
        cparams.n_ctx      = 256;
        cparams.seed       = 1;

        ctx = llama_init_from_model(model, cparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
            llama_free_model(model);
            return 1;
        }
    }

    const auto &tensors = llama_internal_get_tensor_map(ctx);

    // check layer tensors
    int included_layers = 0;
    int64_t max_nelements = 0;
    bool is_f16 = false;
    for (const auto& kv_tensor : tensors) {
        if (!layer_included(params, kv_tensor.first)) {
            continue;
        }
        if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
            // we never quantize those
            continue;
        }
        if (params.verbose) {
            printf("%s: type %s, size %" PRId64 "\n", kv_tensor.first.c_str(), ggml_type_name(kv_tensor.second->type), ggml_nelements(kv_tensor.second));
        }
        if (kv_tensor.second->type == GGML_TYPE_F16) {
            is_f16 = true;
        } else if (kv_tensor.second->type != GGML_TYPE_F32) {
            fprintf(stderr, "%s: error: Quantization should be tested with a float model, "
                "this model contains already quantized layers (%s is type %d)\n", __func__, kv_tensor.first.c_str(), kv_tensor.second->type);
            llama_free(ctx);
            llama_free_model(model);
            return 1;
        }
        included_layers++;
        max_nelements = std::max(max_nelements, ggml_nelements(kv_tensor.second));
    }

    if (is_f16) {
        printf("note: source model is f16\n");
    }
    printf("testing %d layers with max size %" PRId64 "\n", included_layers, max_nelements);
    // allocate scratch space
    std::vector<float> input_scratch;
    std::vector<char> quantized_scratch;
    std::vector<float> output_scratch;

    if (analyze) {
        float tot_mse = 0, tot_elements = 0;
        //std::vector<int64_t> Htot;
        for (const auto& kv_tensor : tensors) {
            if (!layer_included(params, kv_tensor.first)) {
                continue;
            }
            if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
                // we never quantize those
                continue;
            }
            //analyze_iq3ks(kv_tensor.second, tot_mse, tot_elements, Htot);
            analyze_iq2kl(kv_tensor.second, tot_mse, tot_elements);
        }
        //if (!Htot.empty()) {
        //    printf("=============================== pair histogram\n");
        //    for (int i = 0; i < (int)Htot.size(); ++i) {
        //        int i1 = i/8, i2 = i%8;
        //        printf("%d  %d  %d    %g\n", i, i1, i2, 1.*Htot[i]);
        //    }
        //}
        return 0;
    }

    if (analyze) {
        float tot_mse = 0, tot_mse_q = 0, tot_elements = 0;
        for (const auto& kv_tensor : tensors) {
            if (!layer_included(params, kv_tensor.first)) {
                continue;
            }
            if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
                // we never quantize those
                continue;
            }
            analyze_iq4ks(kv_tensor.second, tot_mse, tot_mse_q, tot_elements);
        }
        return 0;
    }

    if (analyze_fp) {
        for (const auto& kv_tensor : tensors) {
            if (!layer_included(params, kv_tensor.first)) {
                continue;
            }
            if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
                // we never quantize those
                continue;
            }
            std::array<uint64_t, 32> H = {};
            analyze_tensor_fp(kv_tensor.second, H.data());
            print_fp_stats("Total", H.data());
        }
        return 0;
    }

    // loop throught quantization types
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const ggml_type type = (ggml_type) i;
        if (!params.include_types.empty() && std::find(params.include_types.begin(), params.include_types.end(), i) == params.include_types.end()) {
            continue;
        }
        ggml_type_traits_t qfns = ggml_internal_get_type_traits(type);
        if (qfns.from_float && qfns.to_float) {
            if (params.verbose) {
                printf("testing %s ...\n",  ggml_type_name(type));
            }

            ggml_quantize_init(type);

            error_stats global_stats {};

            for (const auto& kv_tensor : tensors) {
                if (!layer_included(params, kv_tensor.first)) {
                    continue;
                }
                if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
                    // we never quantize those
                    continue;
                }
                if (params.verbose) {
                    printf("  %s ...\n",  kv_tensor.first.c_str());
                }
                std::string layer_name { ggml_type_name(type) };
                layer_name += "::" + kv_tensor.first;
                test_roundtrip_on_layer(
                        layer_name,
                        params.per_layer_stats,
                        qfns,
                        params.reference,
                        kv_tensor.second,
                        input_scratch,
                        quantized_scratch,
                        output_scratch,
                        global_stats,
                        max_thread
                );
            }

            print_error_stats(ggml_type_name(type), global_stats, params.print_histogram);
        }
    }


    llama_free(ctx);
    llama_free_model(model);
    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    return 0;
}
