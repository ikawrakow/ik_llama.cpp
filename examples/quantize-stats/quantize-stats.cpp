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

//static void fast_ht(int n, float * values) {
//    constexpr float ksqrt2 = 0.707106781f;
//    float scale = 1;
//    int h = 1;
//    while (h < n) {
//        for (int i = 0; i < n; i += 2*h) {
//            for (int j = i; j < i + h; ++j) {
//                float x = values[j], y = values[j + h];
//                values[j+0] = x + y;
//                values[j+h] = x - y;
//            }
//        }
//        h *= 2;
//        scale *= ksqrt2;
//    }
//    for (int i = 0; i < n; ++i) values[i] *= scale;
//}

static const int8_t scale_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

//static std::vector<float> make_values(int nval, int n_per_val) {
//    GGML_ASSERT(n_per_val%4 == 0);
//    std::vector<float> result(nval*n_per_val);
//    const uint32_t a = 89226354, b = 64248484;
//    float * data = result.data();
//    uint32_t aux32;
//    const uint8_t * q = (const uint8_t *)&aux32;
//    for (int i = 0; i < nval; ++i) {
//        uint32_t x = i + 32767;
//        for (int k = 0; k < n_per_val/4; ++k) {
//            x = a*x + b;
//            aux32 = x & 0x0f0f0f0f;
//            for (int l = 0; l < 4; ++l) data[4*k+l] = scale_values[q[l]];
//        }
//        data += n_per_val;
//    }
//    return result;
//}

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

//static std::vector<float> make_values(int nval, int n_per_val) {
//    std::vector<float> result(nval*n_per_val);
//    const uint32_t a = 34038481, b = 76625530;
//    float * data = result.data();
//    for (int i = 0; i < nval; ++i) {
//        uint32_t x = i + 4096;
//        for (int k = 0; k < n_per_val; ++k) {
//            x = a*x + b;
//            uint32_t s = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255);
//            data[k] = (s - 510.f)/147.8f;
//        }
//        data += n_per_val;
//    }
//    return result;
//}

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

template <typename T>
static inline int best_index(int n, const T * values, float x) {
    if (x <= values[0]) return 0;
    if (x >= values[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu - ml > 1) {
        int mav = (mu + ml)/2;
        if (x < values[mav]) mu = mav;
        else ml = mav;
    }
    return x - values[mu-1] < values[mu] - x ? mu - 1 : mu;
}

static void prepare_values(int n_per_row, const float * xr, const float * weights, int8_t * quants, float * scales,
        float& mse, float& mse2, float& mse3, float& tot_sigma2,
        std::vector<float>& all_steps) {

    constexpr int kBlockSize = 64;
    constexpr float kMinGamma = 1.625f;

    float max_amax = 0;
    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        auto xb = xr + ib*kBlockSize;
        float amax = 0;
        for (int j = 0; j < kBlockSize; ++j) amax = std::max(amax, std::abs(xb[j]));
        scales[ib] = amax;
        max_amax = std::max(amax, max_amax);
    }
    if (!max_amax) return;
    float idm = 16/max_amax;
    //float idm = 31/max_amax;
    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        int l = nearest_int(idm*scales[ib]);
        l = std::max(1, std::min(16, l));
        scales[ib] = 0.0625f*max_amax*l;
        //int l = nearest_int(0.5f*(idm*scales[ib]-1));
        //l = std::max(0, std::min(15, l));
        //l = 2*l + 1;
        //scales[ib] =  l/idm;
    }
    float sigma2 = 0, amax = 0, max = 0;
    int8_t int_values[16], next_values[16];
    float grad[16];

    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
        auto xb = xr + ib*kBlockSize;
        float norm = 1/scales[ib];
        for (int j = 0; j < kBlockSize; ++j) {
            sigma2 += xb[j]*xb[j];
            float xs = norm*xb[j];
            float axs = std::abs(xs);
            if (axs > amax) {
                amax = axs; max = xs;
            }
        }
    }
    {
        auto values1 = iq4k_values;
        auto values2 = iq4k_values + 16;
        tot_sigma2 += sigma2;
        float best = 0, d = max/values1[0];
        bool is_shifted = false;
        for (int itry = -9; itry <= 9; ++itry) {
            float id = (values1[0] + itry)/max;
            float sumqx = 0, sumq2 = 0;
            for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                auto xb = xr + ib*kBlockSize;
                auto wb = weights + ib*kBlockSize;
                float norm = 1/scales[ib];
                for (int j = 0; j < kBlockSize; ++j) {
                    int idx = best_index_iq4nl(values1, id*norm*xb[j]);
                    float q = values1[idx]*scales[ib];
                    sumqx += wb[j]*q*xb[j];
                    sumq2 += wb[j]*q*q;
                }
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
            }
            id = (values2[0] + itry)/max;
            sumqx = sumq2 = 0;
            for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                auto xb = xr + ib*kBlockSize;
                auto wb = weights + ib*kBlockSize;
                float norm = 1/scales[ib];
                for (int j = 0; j < kBlockSize; ++j) {
                    int idx = best_index_iq4nl(values2, id*norm*xb[j]);
                    float q = values2[idx]*scales[ib];
                    sumqx += wb[j]*q*xb[j];
                    sumq2 += wb[j]*q*q;
                }
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
            }
        }
        auto values = is_shifted ? values2 : values1;
        float row_mse = 0;
        float id = 1/d;
        for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
            auto xb = xr + ib*kBlockSize;
            float norm = 1/scales[ib];
            for (int j = 0; j < kBlockSize; ++j) {
                int idx = best_index_iq4nl(values, id*norm*xb[j]);
                quants[ib*kBlockSize+j] = idx;
                float q = values[idx]*scales[ib];
                float diff = xb[j] - d*q;
                row_mse += diff*diff;
            }
        }
        mse2 += row_mse;

        for (int itry = 0; itry < 3; ++itry) {
            id = 1/d;
            int nchanged = 0;
            for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                auto xb = xr + ib*kBlockSize;
                auto wb = weights + ib*kBlockSize;
                float best_mse = 0;
                for (int j = 0; j < kBlockSize; ++j) {
                    float q = scales[ib]*values[quants[ib*kBlockSize+j]];
                    float diff = xb[j] - d*q;
                    best_mse += wb[j]*diff*diff;
                }
                int l = nearest_int(16*scales[ib]/max_amax);
                if (l > 1) {
                    float try_scale = 0.0625*max_amax*(l-1);
                    float norm = 1/try_scale;
                    float this_mse = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        int idx = best_index_iq4nl(values, id*norm*xb[j]);
                        float q = values[idx]*try_scale;
                        float diff = xb[j] - d*q;
                        this_mse += wb[j]*diff*diff;
                    }
                    if (this_mse < best_mse) {
                        best_mse = this_mse; scales[ib] = try_scale;
                        ++nchanged;
                    }
                }
                if (l < 16) {
                    float try_scale = 0.0625*max_amax*(l+1);
                    float norm = 1/try_scale;
                    float this_mse = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        int idx = best_index_iq4nl(values, id*norm*xb[j]);
                        float q = values[idx]*try_scale;
                        float diff = xb[j] - d*q;
                        this_mse += wb[j]*diff*diff;
                    }
                    if (this_mse < best_mse) {
                        best_mse = this_mse; scales[ib] = try_scale;
                        ++nchanged;
                    }
                }
            }
            if (nchanged == 0) break;

            row_mse = 0;
            float sumqx = 0, sumq2 = 0;
            for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                auto xb = xr + ib*kBlockSize;
                auto wb = weights + ib*kBlockSize;
                float norm = 1/scales[ib];
                for (int j = 0; j < kBlockSize; ++j) {
                    int idx = best_index_iq4nl(values, id*norm*xb[j]);
                    quants[ib*kBlockSize+j] = idx;
                    float q = values[idx]*scales[ib];
                    sumqx += wb[j]*q*xb[j];
                    sumq2 += wb[j]*q*q;
                }
            }
            d = sumqx/sumq2;
        }

        id = 1/d;
        for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
            auto xb = xr + ib*kBlockSize;
            float norm = 1/scales[ib];
            for (int j = 0; j < kBlockSize; ++j) {
                int idx = best_index_iq4nl(values, id*norm*xb[j]);
                quants[ib*kBlockSize+j] = idx;
                float q = values[idx]*scales[ib];
                float diff = xb[j] - d*q;
                row_mse += diff*diff;
            }
        }
        mse3 += row_mse;

        return;

        for (int i = 0; i < 16; ++i) int_values[i] = values[i];

        for (int iter = 0; iter < 9; ++iter) {
            id = 1/d;
            std::memset(grad, 0, 16*sizeof(float));
            float sumqx, sumq2 = 0;
            for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                auto xb = xr + ib*kBlockSize;
                auto wb = weights + ib*kBlockSize;
                float norm = 1/scales[ib];
                float db = d*scales[ib];
                for (int j = 0; j < kBlockSize; ++j) {
                    int idx = best_index(16, int_values, id*norm*xb[j]);
                    float q = scales[ib]*int_values[idx];
                    grad[idx] += wb[j]*db*(xb[j] - d*q);
                    quants[ib*kBlockSize+j] = idx;
                    sumqx += wb[j]*q*xr[j];
                    sumq2 += wb[j]*q*q;
                }
            }
            all_steps.clear();
            for (int i = 0; i < 16; ++i) {
                int l = int_values[i];
                if (grad[i] > 0) {
                    int lmax = std::min(127, l + 5);
                    if (i < 16) lmax = std::min(lmax, int_values[i+1] - 1);
                    for (int k = l + 1; k <= lmax; ++k) {
                        float step = (k - 0.4999f - l)/grad[i];
                        all_steps.push_back(step);
                    }
                }
                else if (grad[i] < 0) {
                    int lmin = std::max(-128, l - 5);
                    if (i > 0) lmin = std::max(lmin, int_values[i-1]+1);
                    for (int k = l-1; k >= lmin; --k) {
                        float step = (k + 0.499f - l)/grad[i];
                        all_steps.push_back(step);
                    }
                }
            }
            float best = sumqx*sumqx/sumq2;
            int best_is = -1;
            int nstep = std::min(5, int(all_steps.size()));
            std::partial_sort(all_steps.begin(), all_steps.begin() + nstep, all_steps.end());
            float last_sumqx = sumqx, last_sumq2 = sumq2;
            for (int is = 0; is < nstep; ++is) {
                for (int i = 0; i < 16; ++i) {
                    int l = nearest_int(int_values[i] + all_steps[is]*grad[i]);
                    next_values[i] = std::max(-128, std::min(127, l));
                }
                sumqx = last_sumqx, sumq2 = last_sumq2;
                for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
                    auto xb = xr + ib*kBlockSize;
                    auto wb = weights + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) {
                        int l = quants[ib*kBlockSize+j];
                        int lnew = l;
                        float dist = std::abs(id*xb[j] - scales[ib]*next_values[l]);
                        if (l > 0) {
                            float dist1 = std::abs(id*xb[j] - scales[ib]*next_values[l-1]);
                            if (dist1 < dist) { dist = dist1; lnew = l - 1; }
                        }
                        if (l < 15) {
                            float dist1 = std::abs(id*xb[j] - scales[ib]*next_values[l+1]);
                            if (dist1 < dist) { dist = dist1; lnew = l + 1; }
                        }
                        if (next_values[lnew] == int_values[l]) continue;
                        float q = scales[ib]*int_values[l];
                        sumqx -= wb[j]*q*xb[j];
                        sumq2 -= wb[j]*q*q;
                        q = scales[ib]*next_values[lnew];
                        sumqx += wb[j]*q*xb[j];
                        sumq2 += wb[j]*q*q;
                    }
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    d = sumqx/sumq2; best = d*sumqx; best_is = is;
                }
            }
            if (best_is < 0) break;
            for (int i = 0; i < 16; ++i) {
                int l = nearest_int(int_values[i] + all_steps[best_is]*grad[i]);
                int_values[i] = l;
            }
        }
        row_mse = 0;
        for (int j = 0; j < n_per_row; ++j) {
            float diff = xr[j] - d*scales[j/kBlockSize]*int_values[quants[j]];
            row_mse += diff*diff;
        }
        mse3 += row_mse;
        return;
    }


    for (int j = 0; j < n_per_row; ++j) {
        sigma2 += xr[j]*xr[j];
        float ax = std::abs(xr[j]);
        if (ax > amax) {
            amax = ax; max = xr[j];
        }
    }
    if (!sigma2) return;
    tot_sigma2 += sigma2;
    float best = 0;
    float d = max/iq4k_values[0];
    //bool is_shifted = false;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (iq4k_values[0] + itry)/max;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < n_per_row; ++j) {
            int idx = best_index_iq4nl(iq4k_values, id*xr[j]);
            float q = iq4k_values[idx];
            sumqx += weights[j]*q*xr[j];
            sumq2 += weights[j]*q*q;
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx; //is_shifted = false;
        }
        //id = max/(iq4k_values[16] + itry);
        //sumqx = sumq2 = 0;
        //for (int j = 0; j < n_per_row; ++j) {
        //    int idx = best_index_iq4nl(iq4k_values + 16, id*xr[j]);
        //    float q = iq4k_values[idx + 16];
        //    sumqx += weights[j]*q*xr[j];
        //    sumq2 += weights[j]*q*q;
        //}
        //if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
        //    d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
        //}
    }
    float id = 1/d;
    //for (int i = 0; i < 16; ++i) int_values[i] = iq4k_values[i + (is_shifted ? 16 : 0)];
    for (int i = 0; i < 16; ++i) int_values[i] = iq4k_values[i];
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < n_per_row; ++j) {
        quants[j] = best_index_iq4nl(int_values, id*xr[j]);
        float q = int_values[quants[j]];
        sumqx += weights[j]*q*xr[j];
        sumq2 += weights[j]*q*q;
    }
    d = sumqx/sumq2;
    //float sigma = sqrt(sigma2/n_per_row);
    //float gamma = amax/sigma;
    //float alpha = gamma > kMinGamma ? (gamma/kMinGamma - 1)/gamma : 0.f;
    //float d = -max/(8*sigma*(1 + alpha*gamma));
    //float id = 1/d;
    //float row_mse = 0;
    //for (int j = 0; j < n_per_row; ++j) {
    //    float xs = xr[j]/sigma;
    //    float z = xs/(1 + alpha*std::abs(xs));
    //    int l = nearest_int(id*z);
    //    l = std::max(-8, std::min(7, l));
    //    quants[j] = l;
    //    float q = sigma*l/(1 - alpha*std::abs(d*l));
    //    float diff = xr[j] - d*q;
    //    row_mse += diff*diff;
    //}
    //mse += row_mse;
    //alpha = std::abs(alpha*d);
    //for (int iter = 0; iter < 9; ++iter) {
    //    float sumqx = 0, sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float q = sigma*quants[j]/(1 - alpha*std::abs(quants[j]));
    //        sumqx += weights[j]*q*xr[j];
    //        sumq2 += weights[j]*q*q;
    //    }
    //    if (sumq2 > 0) d = sumqx/sumq2;
    //    int nchanged = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float xs = xr[j]/(d*sigma);
    //        float z = xs/(1 + alpha*std::abs(xs));
    //        int l = nearest_int(z);
    //        l = std::max(-8, std::min(7, l));
    //        if (l != quants[j]) ++nchanged;
    //        quants[j] = l;
    //    }
    //    if (nchanged == 0) break;
    //}
    //row_mse = 0;
    //for (int j = 0; j < n_per_row; ++j) {
    //    float diff = xr[j] - d*sigma*quants[j]/(1 - alpha*std::abs(quants[j]));
    //    row_mse += diff*diff;
    //}
    //mse2 += row_mse;
    //float c = 15.f*(1 - 8*alpha);
    //for (int i = 0; i < 16; ++i) {
    //    int_values[i] = nearest_int(c*(i-8)/(1-alpha*std::abs(i-8)));
    //}
    //float sumqx = 0, sumq2 = 0;
    //for (int j = 0; j < n_per_row; ++j) {
    //    quants[j] += 8;
    //    float q = int_values[quants[j]];
    //    sumqx += weights[j]*q*xr[j];
    //    sumq2 += weights[j]*q*q;
    //}
    //d = sumqx/sumq2;

    for (int iter = 0; iter < 9; ++iter) {
        id = 1/d;
        std::memset(grad, 0, 16*sizeof(float));
        sumqx = sumq2 = 0;
        for (int j = 0; j < n_per_row; ++j) {
            int idx = best_index(16, int_values, id*xr[j]);
            float q = int_values[idx];
            grad[idx] += weights[j]*d*(xr[j] - d*q);
            quants[j] = idx;
            sumqx += weights[j]*q*xr[j];
            sumq2 += weights[j]*q*q;
        }
        all_steps.clear();
        for (int i = 0; i < 16; ++i) {
            int l = int_values[i];
            if (grad[i] > 0) {
                int lmax = std::min(127, l + 5);
                if (i < 16) lmax = std::min(lmax, int_values[i+1] - 1);
                for (int k = l + 1; k <= lmax; ++k) {
                    float step = (k - 0.4999f - l)/grad[i];
                    all_steps.push_back(step);
                }
            }
            else if (grad[i] < 0) {
                int lmin = std::max(-128, l - 5);
                if (i > 0) lmin = std::max(lmin, int_values[i-1]+1);
                for (int k = l-1; k >= lmin; --k) {
                    float step = (k + 0.499f - l)/grad[i];
                    all_steps.push_back(step);
                }
            }
        }
        float best = sumqx*sumqx/sumq2;
        int best_is = -1;
        int nstep = std::min(5, int(all_steps.size()));
        std::partial_sort(all_steps.begin(), all_steps.begin() + nstep, all_steps.end());
        float last_sumqx = sumqx, last_sumq2 = sumq2;
        for (int is = 0; is < nstep; ++is) {
            for (int i = 0; i < 16; ++i) {
                int l = nearest_int(int_values[i] + all_steps[is]*grad[i]);
                next_values[i] = std::max(-128, std::min(127, l));
            }
            sumqx = last_sumqx, sumq2 = last_sumq2;
            for (int j = 0; j < n_per_row; ++j) {
                int l = quants[j];
                int lnew = l;
                float dist = std::abs(id*xr[j] - next_values[l]);
                if (l > 0) {
                    float dist1 = std::abs(id*xr[j] - next_values[l-1]);
                    if (dist1 < dist) { dist = dist1; lnew = l - 1; }
                }
                if (l < 15) {
                    float dist1 = std::abs(id*xr[j] - next_values[l+1]);
                    if (dist1 < dist) { dist = dist1; lnew = l + 1; }
                }
                if (next_values[lnew] == int_values[l]) continue;
                //if (next_values[l] == int_values[l] &&
                //    (l == 0 || next_values[l-1] == int_values[l-1]) &&
                //    (l < 15 || next_values[l+1] == int_values[l+1])) continue;
                //int idx = best_index(16, next_values, id*xr[j]);
                //if (idx == l) continue;
                float q = int_values[l];
                sumqx -= weights[j]*q*xr[j];
                sumq2 -= weights[j]*q*q;
                q = next_values[lnew];
                sumqx += weights[j]*q*xr[j];
                sumq2 += weights[j]*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d*sumqx; best_is = is;
            }
        }
        if (best_is < 0) break;
        for (int i = 0; i < 16; ++i) {
            int l = nearest_int(int_values[i] + all_steps[best_is]*grad[i]);
            int_values[i] = l;
        }
    }
    float row_mse = 0;
    for (int j = 0; j < n_per_row; ++j) {
        float diff = xr[j] - d*int_values[quants[j]];
        row_mse += diff*diff;
    }
    mse3 += row_mse;
}

static void analyze_atan(const char * name, int nrows, int n_per_row, const float * values,
        float& tot_mse, float& tot_elements, std::vector<int64_t>& H) {


    int max_thread = std::thread::hardware_concurrency()/2;
    int nthread = std::min(nrows, max_thread);
    std::vector<std::thread> workers(nthread-1);
    std::mutex mutex;
    int counter = 0;
    float tot_sigma2 = 0, mse = 0, mse2 = 0, mse3 = 0;
    auto compute = [&mutex, &counter, &tot_sigma2, &mse, &mse2, &mse3, nrows, n_per_row, values] () {
        float l_tot_sigma2 = 0, l_mse = 0, l_mse2 = 0, l_mse3 = 0;
        std::vector<int8_t> quants(n_per_row);
        std::vector<float> all_steps;
        std::vector<float> weights(n_per_row);
        std::vector<float> scales(n_per_row/16);
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int row = counter++;
            if (row >= nrows) {
                tot_sigma2 += l_tot_sigma2;
                mse += l_mse;
                mse2 += l_mse2;
                mse3 += l_mse3;
                return;
            }
            lock.unlock();
            auto xr = values + row*n_per_row;
            float sigma2 = 0;
            for (int j = 0; j < n_per_row; ++j) sigma2 += xr[j]*xr[j];
            sigma2 *= 2.f/n_per_row;
            for (int j = 0; j < n_per_row; ++j) weights[j] = 1; //0.25f*sigma2 + xr[j]*xr[j];
            prepare_values(n_per_row, xr, weights.data(), quants.data(), scales.data(), l_mse, l_mse2, l_mse3, l_tot_sigma2, all_steps);
        }
    };
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();
    constexpr float kMinGamma = 1.625f;
    //int8_t int_values[16], next_values[16];
    //std::vector<int8_t> quants(n_per_row);
    //std::vector<float> all_steps;
    //float table[16];
    //float grad[16];
    //for (int row = 0; row < nrows; ++row) {
    //    auto xr = values + row*n_per_row;
    //    float sigma2 = 0, amax = 0, max = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        sigma2 += xr[j]*xr[j];
    //        float ax = std::abs(xr[j]);
    //        if (ax > amax) {
    //            amax = ax; max = xr[j];
    //        }
    //    }
    //    if (!sigma2) continue;
    //    tot_sigma2 += sigma2;
    //    float sigma = sqrt(sigma2/n_per_row);
    //    float gamma = amax/sigma;
    //    float alpha = gamma > kMinGamma ? (gamma/kMinGamma - 1)/gamma : 0.f;
    //    float d = -max/(8*sigma*(1 + alpha*gamma));
    //    float id = 1/d;
    //    float row_mse = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float xs = xr[j]/sigma;
    //        float z = xs/(1 + alpha*std::abs(xs));
    //        int l = nearest_int(id*z);
    //        l = std::max(-8, std::min(7, l));
    //        quants[j] = l;
    //        float q = sigma*l/(1 - alpha*std::abs(d*l));
    //        float diff = xr[j] - d*q;
    //        row_mse += diff*diff;
    //    }
    //    mse += row_mse;
    //    //float rmse = sqrt(row_mse/n_per_row);
    //    //printf("Row %d: rmse = %g, %g, gamma = %g, alpha = %g\n", row, rmse, rmse/sigma, gamma, alpha);
    //    alpha = std::abs(alpha*d);
    //    for (int iter = 0; iter < 3; ++iter) {
    //        float sumqx = 0, sumq2 = 0;
    //        for (int j = 0; j < n_per_row; ++j) {
    //            float q = sigma*quants[j]/(1 - alpha*std::abs(quants[j]));
    //            sumqx += q*xr[j];
    //            sumq2 += q*q;
    //        }
    //        if (sumq2 > 0) d = sumqx/sumq2;
    //        //sumqx = sumq2 = 0;
    //        //for (int j = 0; j < n_per_row; ++j) {
    //        //    int l = quants[j];
    //        //    if (!l) continue;
    //        //    float z = xr[j]/(d*sigma);
    //        //    sumqx += std::abs(z)*(z/l-1);
    //        //    sumq2 += z*z;
    //        //}
    //        //if (sumqx > 0 && sumq2 > 0) alpha = sumqx/sumq2;
    //        //row_mse = 0;
    //        //for (int j = 0; j < n_per_row; ++j) {
    //        //    float diff = xr[j] - d*sigma*quants[j]/(1 - alpha*std::abs(quants[j]));
    //        //    row_mse += diff*diff;
    //        //}
    //        //rmse = sqrt(row_mse/n_per_row);
    //        //id = 1/d;
    //        int nchanged = 0;
    //        for (int j = 0; j < n_per_row; ++j) {
    //            float xs = xr[j]/(d*sigma);
    //            float z = xs/(1 + alpha*std::abs(xs));
    //            int l = nearest_int(z);
    //            l = std::max(-8, std::min(7, l));
    //            if (l != quants[j]) ++nchanged;
    //            quants[j] = l;
    //        }
    //        if (nchanged == 0) break;
    //        //printf("    iteration: d = %g, alpha = %g, %g, rmse = %g, %g, nchanged = %d\n", d, alpha, std::abs(alpha/d), rmse, rmse/sigma, nchanged);
    //    }
    //    row_mse = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float diff = xr[j] - d*sigma*quants[j]/(1 - alpha*std::abs(quants[j]));
    //        row_mse += diff*diff;
    //    }
    //    mse2 += row_mse;
    //    //float amax_table = 0, max_table = 0;
    //    //for (int i = 0; i < 16; ++i) {
    //    //    table[i] = d*sigma*(i-8)/(1 - alpha*std::abs(i-8));
    //    //    float at = std::abs(table[i]);
    //    //    if (at > amax_table) {
    //    //        amax_table = at; max_table = table[i];
    //    //    }
    //    //}
    //    //float c = -max_table/124;
    //    //float ic = 1/c;
    //    //for (int i = 0; i < 16; ++i) {
    //    //    int_values[i] = nearest_int(ic*table[i]);
    //    //}
    //    float c = 15.f*(1 - 8*alpha);
    //    //printf("int_values:");
    //    for (int i = 0; i < 16; ++i) {
    //        int_values[i] = nearest_int(c*(i-8)/(1-alpha*std::abs(i-8)));
    //        //printf(" %d", int_values[i]);
    //    }
    //    ////printf("\n");
    //    float sumqx = 0, sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        quants[j] += 8;
    //        float q = int_values[quants[j]];
    //        sumqx += q*xr[j];
    //        sumq2 += q*q;
    //    }
    //    //printf("Previous d: %g, %g, new d: %g\n", d, d*sigma, sumqx/sumq2);
    //    d = sumqx/sumq2;
    //    //row_mse = 0;
    //    //for (int j = 0; j < n_per_row; ++j) {
    //    //    float diff = xr[j] - d*int_values[quants[j]];
    //    //    row_mse += diff*diff;
    //    //}
    //    //mse3 += row_mse;
    //    //continue;

    //    //printf("d = %g, score = %g\n", d, d*sumqx);
    //    for (int iter = 0; iter < 3; ++iter) {
    //        id = 1/d;
    //        std::memset(grad, 0, 16*sizeof(float));
    //        sumqx = sumq2 = 0;
    //        for (int j = 0; j < n_per_row; ++j) {
    //            int idx = best_index(16, int_values, id*xr[j]);
    //            float q = int_values[idx];
    //            grad[idx] += d*(xr[j] - d*q);
    //            quants[j] = idx;
    //            sumqx += q*xr[j];
    //            sumq2 += q*q;
    //        }
    //        all_steps.clear();
    //        for (int i = 0; i < 16; ++i) {
    //            int l = int_values[i];
    //            if (grad[i] > 0) {
    //                int lmax = std::min(127, l + 5);
    //                if (i < 16) lmax = std::min(lmax, int_values[i+1] - 1);
    //                for (int k = l + 1; k <= lmax; ++k) {
    //                    float step = (k - 0.4999f - l)/grad[i];
    //                    all_steps.push_back(step);
    //                }
    //            }
    //            else if (grad[i] < 0) {
    //                int lmin = std::max(-128, l - 5);
    //                if (i > 0) lmin = std::max(lmin, int_values[i-1]+1);
    //                for (int k = l-1; k >= lmin; --k) {
    //                    float step = (k + 0.499f - l)/grad[i];
    //                    all_steps.push_back(step);
    //                }
    //            }
    //        }
    //        float best = sumqx*sumqx/sumq2;
    //        //printf("Iteration %d: best = %g\n", iter, best);
    //        int best_is = -1;
    //        int nstep = std::min(10, int(all_steps.size()));
    //        std::partial_sort(all_steps.begin(), all_steps.begin() + nstep, all_steps.end());
    //        for (int is = 0; is < nstep; ++is) {
    //            //printf("step = %g\n", all_steps[is]);
    //            for (int i = 0; i < 16; ++i) {
    //                int l = nearest_int(int_values[i] + all_steps[is]*grad[i]);
    //                next_values[i] = std::max(-128, std::min(127, l));
    //                //if (next_values[i] != int_values[i]) printf("%d: %d -> %d\n", i, int_values[i], next_values[i]);
    //            }
    //            sumqx = sumq2 = 0;
    //            for (int j = 0; j < n_per_row; ++j) {
    //                int idx = best_index(16, next_values, id*xr[j]);
    //                float q = next_values[idx];
    //                sumqx += q*xr[j];
    //                sumq2 += q*q;
    //            }
    //            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
    //                d = sumqx/sumq2; best = d*sumqx; best_is = is;
    //                //printf("New best: %g\n", best);
    //            }
    //        }
    //        if (best_is < 0) break;
    //        for (int i = 0; i < 16; ++i) {
    //            int l = nearest_int(int_values[i] + all_steps[best_is]*grad[i]);
    //            int_values[i] = l;
    //        }
    //    }
    //    row_mse = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float diff = xr[j] - d*int_values[quants[j]];
    //        row_mse += diff*diff;
    //    }
    //    mse3 += row_mse;
    //}
    tot_mse += mse3;
    tot_elements += tot_sigma2;
    printf("%s:    %g  %g    %g  %g    %g  %g    %g\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(mse/tot_sigma2),
            sqrt(mse2/(nrows*n_per_row)), sqrt(mse2/tot_sigma2),
            sqrt(mse3/(nrows*n_per_row)), sqrt(mse3/tot_sigma2),
            sqrt(tot_mse/tot_elements));
    return;

    //constexpr int kBlockSize = 32;

    ////std::vector<float> all_alphas(n_per_row/kBlockSize);

    //float tot_sigma2 = 0, mse = 0, mse2 = 0, mse3 = 0;
    //for (int row = 0; row < nrows; ++row) {
    //    auto xr = values + row*n_per_row;
    //    float sigma2 = 0, amax = 0, max = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        sigma2 += xr[j]*xr[j];
    //        float ax = std::abs(xr[j]);
    //        if (ax > amax) {
    //            amax = ax; max = xr[j];
    //        }
    //    }
    //    if (!sigma2) continue;
    //    tot_sigma2 += sigma2;
    //    float sigma = sqrt(sigma2/n_per_row);
    //    float row_mse = 0;
    //    float row_mse2 = 0;
    //    for (int ib = 0; ib < n_per_row/kBlockSize; ++ib) {
    //        auto xb = xr + ib*kBlockSize;
    //        float max = 0, amax = 0;
    //        for (int j = 0; j < kBlockSize; ++j) {
    //            float ax = std::abs(xb[j]);
    //            if (ax > amax) {
    //                amax = ax; max = xb[j];
    //            }
    //        }
    //        float gamma = amax/sigma;
    //        float alpha = gamma > kMinGamma ? (gamma/kMinGamma - 1)/gamma : 0.f;
    //        float d = -max/(8*sigma*(1 + alpha*gamma));
    //        float id = 1/d;
    //        float sumqx = 0, sumq2 = 0;
    //        for (int j = 0; j < kBlockSize; ++j) {
    //            float xs = xb[j]/sigma;
    //            float z = xs/(1 + alpha*std::abs(xs));
    //            int l = nearest_int(id*z);
    //            l = std::max(-8, std::min(7, l));
    //            float q = sigma*l/(1 - alpha*std::abs(d*l));
    //            float diff = xb[j] - d*q;
    //            row_mse += diff*diff;
    //            sumqx += q*xb[j];
    //            sumq2 += q*q;
    //        }
    //        float dnew = sumq2 > 0 ? sumqx/sumq2 : d;
    //        for (int j = 0; j < kBlockSize; ++j) {
    //            float xs = xb[j]/sigma;
    //            float z = xs/(1 + alpha*std::abs(xs));
    //            int l = nearest_int(id*z);
    //            l = std::max(-8, std::min(7, l));
    //            float q = sigma*l/(1 - alpha*std::abs(d*l));
    //            float diff = xb[j] - dnew*q;
    //            row_mse2 += diff*diff;
    //        }
    //        //float d = amax/(15*sigma*(1 + alpha*gamma));
    //        //float id = 1/d;
    //        //for (int j = 0; j < kBlockSize; ++j) {
    //        //    float xs = xb[j]/sigma;
    //        //    float z = xs/(1 + alpha*std::abs(xs));
    //        //    int l = nearest_int(0.5f*(id*z+15));
    //        //    l = std::max(0, std::min(15, l));
    //        //    l = 2*l - 15;
    //        //    float diff = xb[j] - sigma*d*l/(1 - alpha*std::abs(d*l));
    //        //    row_mse += diff*diff;
    //        //}
    //    }
    //    mse += row_mse;
    //    mse2 += row_mse2;
    //}
    //tot_mse += mse2;
    //tot_elements += tot_sigma2;
    //printf("%s:    %g  %g    %g  %g    %g\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(mse/tot_sigma2),
    //        sqrt(mse2/(nrows*n_per_row)), sqrt(mse2/tot_sigma2), sqrt(tot_mse/tot_elements));
    //return;

    //float xc[16], sumx[16], sumw[16];
    //std::vector<uint8_t> quants(n_per_row);
    //for (int row = 0; row < nrows; ++row) {
    //    auto xr = values + row*n_per_row;
    //    float sigma2 = 0, amax = 0, max = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        sigma2 += xr[j]*xr[j];
    //        float ax = std::abs(xr[j]);
    //        if (ax > amax) {
    //            amax = ax; max = xr[j];
    //        }
    //    }
    //    if (!sigma2) continue;
    //    tot_sigma2 += sigma2;
    //    float sigma = sqrt(sigma2/n_per_row);
    //    float gamma = amax/sigma;
    //    float alpha = gamma > kMinGamma ? (gamma/kMinGamma - 1)/gamma : 0.f;
    //    float d = -max/(8*sigma*(1 + alpha*gamma));
    //    float id = 1/d;
    //    float row_mse = 0;
    //    std::memset(sumx, 0, 16*sizeof(float));
    //    std::memset(sumw, 0, 16*sizeof(float));
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float xs = xr[j]/sigma;
    //        float z = xs/(1 + alpha*std::abs(xs));
    //        int l = nearest_int(id*z);
    //        l = std::max(-8, std::min(7, l));
    //        float diff = xr[j] - sigma*d*l/(1 - alpha*std::abs(d*l));
    //        row_mse += diff*diff;
    //        l += 8;
    //        quants[j] = l;
    //        sumx[l] += xr[j];
    //        sumw[l] += 1;
    //    }
    //    mse += row_mse;
    //    for (int i = 0; i < 16; ++i) {
    //        xc[i] = sumw[i] > 0 ? sumx[i]/sumw[i] : 0; //d*(i-8);
    //    }
    //    float sumqx = 0, sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float q = xc[quants[j]];
    //        sumqx += xr[j]*q;
    //        sumq2 += q*q;
    //    }
    //    if (sumq2 > 0) {
    //        float d = sumqx/sumq2;
    //        for (int i = 0; i < 16; ++i) xc[i] *= d;
    //    }
    //    row_mse = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        //float al = xc[quants[j]];
    //        float diff = xr[j] - xc[quants[j]]; //sigma*d*al/(1 - alpha*std::abs(d*al));
    //        row_mse += diff*diff;
    //    }
    //    mse2 += row_mse;
    //    //for (int iter = 0; iter < 5; ++iter) {
    //    //    std::memset(sumx, 0, 16*sizeof(float));
    //    //    std::memset(sumw, 0, 16*sizeof(float));
    //    //    for (int j = 0; j < n_per_row; ++j) {
    //    //        int idx = best_index(16, xc, xr[j]);
    //    //        quants[j] = idx;
    //    //        sumx[idx] += xr[j];
    //    //        sumw[idx] += 1;
    //    //    }
    //    //    printf("Iteration %d:\n", iter);
    //    //    for (int i = 0; i < 16; ++i) {
    //    //        float xnew = sumw[i] > 0 ? sumx[i]/sumw[i] : xc[i];
    //    //        printf("%d  %g  %g\n", i, xnew, xc[i]);
    //    //        xc[i] = xnew;
    //    //        //if (sumw[i] > 0) xc[i] = sumx[i]/sumw[i];
    //    //    }
    //    //}
    //    //row_mse = 0;
    //    //for (int j = 0; j < n_per_row; ++j) {
    //    //    float diff = xr[j] - xc[quants[j]];
    //    //    row_mse += diff*diff;
    //    //}
    //    //mse3 += row_mse;
    //}
    //tot_mse += mse2;
    //tot_elements += tot_sigma2;
    //printf("%s:    %g   %g    %g  %g      %g\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(mse/tot_sigma2), sqrt(mse2/(nrows*n_per_row)), sqrt(mse2/tot_sigma2),
    //        sqrt(tot_mse/tot_elements));
    ////printf("%s:    %g   %g    %g  %g    %g  %g\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(mse/tot_sigma2),
    ////        sqrt(mse2/(nrows*n_per_row)), sqrt(mse2/tot_sigma2), sqrt(mse3/(nrows*n_per_row)), sqrt(mse3/tot_sigma2));
    //return;


    int nbin = H.size();
    if (!nbin) {
        nbin = 256;
        H = std::vector<int64_t>(nbin, 0);
    }
    //float delta = 2*kMinGamma/nbin;
    float delta = kMinGamma/nbin;
    float idelta = 1/delta;
    for (int row = 0; row < nrows; ++row) {
        auto xr = values + row*n_per_row;
        float sigma2 = 0, amax = 0;
        for (int j = 0; j < n_per_row; ++j) {
            sigma2 += xr[j]*xr[j];
            amax = std::max(amax, std::abs(xr[j]));
        }
        if (!sigma2) return;
        float sigma = sqrt(sigma2/n_per_row);
        float gamma = amax/sigma;
        float alpha = gamma > kMinGamma ? (gamma/kMinGamma - 1)/gamma : 0;
        for (int j = 0; j < n_per_row; ++j) {
            float xs = xr[j]/sigma;
            //float z = xs/(1 + alpha*std::abs(xs));
            //int bin = int((z + kMinGamma)*idelta);
            float z = std::abs(xs)/(1 + alpha*std::abs(xs));
            int bin = int(z*idelta);
            bin = std::max(0, std::min(nbin-1, bin));
            ++H[bin];
        }
    }

    //std::vector<float> xaux(n_per_row);
    //std::vector<int8_t> quants(n_per_row);
    //float mse = 0, max_err = 0, tot_sigma2 = 0;
    //float xc[16];
    //float sumx[16], sumw[16];
    //for (int row = 0; row < nrows; ++row) {
    //    auto xr = values + row*n_per_row;
    //    float sigma2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) sigma2 += xr[j]*xr[j];
    //    if (!sigma2) continue;
    //    tot_sigma2 += sigma2;
    //    float sigma = sqrt(sigma2/n_per_row);
    //    float isigma = 1/sigma;
    //    for (int j = 0; j < n_per_row; ++j) xaux[j] = atan(isigma*xr[j]);
    //    float max = xaux[0], min = xaux[0];
    //    for (int j = 0; j < n_per_row; ++j) {
    //        max = std::max(max, xaux[j]);
    //        min = std::min(min, xaux[j]);
    //    }
    //    float delta = (max - min)/15;
    //    for (int i = 0; i < 16; ++i) xc[i] = min + delta*i;
    //    for (int iter = 0; iter < 5; ++iter) {
    //        std::memset(sumx, 0, 16*sizeof(float));
    //        std::memset(sumw, 0, 16*sizeof(float));
    //        for (int j = 0; j < n_per_row; ++j) {
    //            int idx = best_index(16, xc, xaux[j]);
    //            sumx[idx] += xaux[j];
    //            sumw[idx] += 1;
    //        }
    //        for (int i = 0; i < 16; ++i) {
    //            if (sumw[i] > 0) xc[i] = sumx[i]/sumw[i];
    //        }
    //    }
    //    float sumqx = 0, sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        int idx = best_index(16, xc, xaux[j]);
    //        quants[j] = idx;
    //        float q = xc[idx];
    //        sumqx += xaux[j]*q;
    //        sumq2 += q*q;
    //    }
    //    float d = sumqx/sumq2;
    //    for (int i = 0; i < 16; ++i) xc[i] = tan(d*xc[i]);
    //    sumqx = sumq2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float q = xc[quants[j]];
    //        sumqx += q*xr[j];
    //        sumq2 += q*q;
    //    }
    //    sigma = sumqx/sumq2;
    //    float row_mse = 0;
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float diff = xr[j] - sigma*xc[quants[j]];
    //        row_mse += diff*diff;
    //        max_err = std::max(max_err, std::abs(diff));
    //    }
    //    mse += row_mse;
    //}
    //float sigma = sqrt(tot_sigma2/(nrows*n_per_row));
    //printf("%s:   %g  %g  %g  %g\n", name, sqrt(mse/(nrows*n_per_row)), sqrt(mse/tot_sigma2), max_err, max_err/sigma);
    //return;

    //int nbin = H.size();
    //if (!nbin) {
    //    nbin = 256;
    //    H = std::vector<int64_t>(nbin, 0);
    //}
    //float delta = M_PI/nbin;
    //float idelta = 1/delta;
    //for (int row = 0; row < nrows; ++row) {
    //    auto xr = values + row*n_per_row;
    //    float sigma2 = 0;
    //    for (int j = 0; j < n_per_row; ++j) sigma2 += xr[j]*xr[j];
    //    if (!sigma2) return;
    //    float isigma = 1/sqrt(sigma2/n_per_row);
    //    for (int j = 0; j < n_per_row; ++j) {
    //        float z = atan(isigma*xr[j]);
    //        int bin = int((z + M_PI/2)*idelta);
    //        bin = std::max(0, std::min(nbin-1, bin));
    //        ++H[bin];
    //    }
    //}
    printf("Finished %s\n", name);
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
                        //int jbest_cluster = jbest;
                        //best = INFINITY; jbest = -1;
                        //for (auto ip : points) {
                        //    auto vc = codes.data() + ip*kGroupSize;
                        //    float diff2 = 0;
                        //    for (int k = 0; k < kGroupSize; ++k) {
                        //        float delta = d*vc[k] - xl[k];
                        //        diff2 += wl[k]*delta*delta;
                        //    }
                        //    if (diff2 < best) {
                        //        best = diff2; jbest = ip;
                        //    }
                        //}
                        if (jbest < 0) {
                            printf("Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                            GGML_ASSERT(false);
                        }
                        GGML_ASSERT(jbest >= 0);
                        //for (int j = 0; j < kNumVal; j += 8) {
                        //    auto idx = _mm256_add_epi32(_mm256_set1_epi32(j), add_idx);
                        //    for (int i = 0; i < 8; ++i) {
                        //        auto vq = _mm256_loadu_ps(codes.data() + kGroupSize*(j+i));
                        //        auto vdiff = _mm256_sub_ps(vq, vx);
                        //        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                        //    }
                        //    auto score = hsum_float_8x8(sqx);
                        //    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                        //    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                        //                              _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                        //    vbest = _mm256_min_ps(vbest, score);
                        //}
                        //_mm256_store_ps(sx, vbest);
                        //_mm256_store_si256((__m256i *)index, best_index);
                        //for (int i = 0; i < 8; ++i) {
                        //    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                        //}
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
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
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
        float lmse = 0, lmse_q = 0;
        std::vector<float> scales(n_per_row/kBlockSize);
        std::vector<int> best_idx(n_per_row/kBlockSize);
        //float xtmp[kBlockSize];
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
                    //std::memcpy(xtmp, xb, kBlockSize*sizeof(float));
                    //fast_ht(kBlockSize, xtmp);
#ifdef __AVX2__
                    for (int l = 0; l < kBlockSize/8; ++l) {
                        //vx[l] = _mm256_loadu_ps(xtmp+8*l);
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
                    //for (int k = 0; k < kBlockSize; ++k) sumqx += xtmp[k]*qv[k];
                    d = sumqx*sumq2i[jbest];
#else
                    for (int j = 0; j < kNumVal; ++j) {
                        if (!sumq2i[j]) continue;
                        auto qv = codes.data() + kBlockSize*j;
                        float sumqx = 0;
                        for (int k = 0; k < kBlockSize; ++k) sumqx += qv[k]*xb[k];
                        if (sumqx*sumqx*sumq2i[j] > best]) {
                            d = sumqx*sumq2i[j]; best = d*sumqx; jbest = j;
                        }
                    }
                    auto qv = codes.data() + kBlockSize*jbest;
#endif
                    scales[ib] = d;
                    best_idx[ib] = jbest;
                    for (int k = 0; k < kBlockSize; ++k) {
                        float diff = xb[k] - d*qv[k];
                        //float diff = xtmp[k] - d*qv[k];
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
                    //std::memcpy(xtmp, xb, kBlockSize*sizeof(float));
                    //fast_ht(kBlockSize, xtmp);
                    auto qv = codes.data() + kBlockSize*best_idx[ib];
                    for (int k = 0; k < kBlockSize; ++k) {
                        float diff = xb[k] - dl*qv[k];
                        //float diff = xtmp[k] - dl*qv[k];
                        lmse_q += diff*diff;
                    }
                }
                //for (int ibl = 0; ibl < n_per_row/kSuperBlockSize; ++ibl) {
                //    auto sb = scales.data() + ibl*(kSuperBlockSize/kBlockSize);
                //    auto idx = best_idx.data() + ibl*(kSuperBlockSize/kBlockSize);
                //    auto xbl = xr + ibl*kSuperBlockSize;
                //    float amax_scale = 0, max_scale = 0;
                //    for (int ib = 0; ib < kSuperBlockSize/kBlockSize; ++ib) {
                //        float ax = std::abs(sb[ib]);
                //        if (ax > amax_scale) {
                //            amax_scale = ax; max_scale = sb[ib];
                //        }
                //        //amax_scale = std::max(amax_scale, std::abs(sb[ib]));
                //    }
                //    float d = max_scale/scale_values[0];
                //    float id = d ? 1/d : 0.f;
                //    //float id = amax_scale > 0 ? 15/amax_scale : 0;
                //    //float d = amax_scale/15;
                //    for (int ib = 0; ib < kSuperBlockSize/kBlockSize; ++ib) {
                //        int ls = best_index_scale(scale_values, id*sb[ib]);
                //        float dl = d * scale_values[ls];
                //        //int ls = nearest_int(0.5f*(id*sb[ib]+15));
                //        //ls = std::max(0, std::min(ls, 15));
                //        //float dl = d*(2*ls - 15);
                //        auto xb = xbl + kBlockSize*ib;
                //        auto qv = codes.data() + kBlockSize*idx[ib];
                //        for (int k = 0; k < kBlockSize; ++k) {
                //            float diff = xb[k] - dl*qv[k];
                //            lmse_q += diff*diff;
                //        }
                //    }
                //}
            }
        }
    };
    std::vector<std::thread> workers(nthread-1);
    for (auto& w : workers) w = std::thread(compute);
    compute();
    for (auto& w : workers) w.join();
    tot_mse += mse;
    tot_mse_q += mse_q;
    tot_elements += n_per_row*nrows;
    printf("%s:   %g    %g      %g   %g\n", name, sqrt(mse/(n_per_row*nrows)), sqrt(tot_mse/tot_elements),
            sqrt(mse_q/(n_per_row*nrows)), sqrt(tot_mse_q/tot_elements));
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
                                //for (int k = 0; k < 16; k += 4) {
                                //    uint16_t v = v0 ^ (1 << k);
                                //    uint8_t v1 = v;
                                //    uint8_t v2 = v >> 8;
                                //    diff1 = xb[j+ 0] - dl*values[v1 & 0xf];
                                //    diff2 = xb[j+16] - dl*values[v1 >>  4];
                                //    diff3 = xb[j+ 1] - dl*values[v2 & 0xf];
                                //    diff4 = xb[j+17] - dl*values[v2 >>  4];
                                //    float score = diff1*diff1 + diff2*diff2 + diff3*diff3 + diff4*diff4;
                                //    if (score < best) best = score;
                                //}
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

static void analyze_iq4ks(const ggml_tensor * t, float& tot_mse, float& tot_mse_q, float& tot_elements, std::vector<int64_t>& H) {
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
        return;
    }
    if (t->type == GGML_TYPE_F32) {
        //analyze_iq4ks(t->name, t->ne[1], t->ne[0], (const float *)t->data, tot_mse, tot_elements);
        //analyze_x_v2(t->name, t->ne[1], t->ne[0], (const float *)t->data, tot_mse, tot_mse_q, tot_elements);
        analyze_atan(t->name, t->ne[1], t->ne[0], (const float *)t->data, tot_mse, tot_elements, H);
    } else {
        std::vector<float> aux(t->ne[0]*t->ne[1]);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, aux.data(), aux.size());
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)t->data, aux.data(), aux.size());
        }
        //analyze_iq4ks(t->name, t->ne[1], t->ne[0], aux.data(), tot_mse, tot_elements);
        //analyze_x_v2(t->name, t->ne[1], t->ne[0], aux.data(), tot_mse, tot_mse_q, tot_elements);
        analyze_atan(t->name, t->ne[1], t->ne[0], aux.data(), tot_mse, tot_elements, H);
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

        model = llama_load_model_from_file(params.model.c_str(), mparams);

        if (model == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        auto cparams = llama_context_default_params();
        cparams.n_ctx      = 256;
        cparams.seed       = 1;

        ctx = llama_new_context_with_model(model, cparams);

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
        float tot_mse = 0, tot_mse_q = 0, tot_elements = 0;
        std::vector<int64_t> H;
        for (const auto& kv_tensor : tensors) {
            if (!layer_included(params, kv_tensor.first)) {
                continue;
            }
            if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
                // we never quantize those
                continue;
            }
            analyze_iq4ks(kv_tensor.second, tot_mse, tot_mse_q, tot_elements, H);
        }
        for (int j = 0; j < int(H.size()); ++j) printf("%d  %g\n", j, 1.*H[j]);
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
