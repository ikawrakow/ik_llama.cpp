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

static void analyze_iq4ks(const ggml_tensor * t, float& tot_mse, float& tot_elements) {
    if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16)) {
        return;
    }
    if (t->type == GGML_TYPE_F32) {
        analyze_iq4ks(t->name, t->ne[1], t->ne[0], (const float *)t->data, tot_mse, tot_elements);
    } else {
        std::vector<float> aux(t->ne[0]*t->ne[1]);
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, aux.data(), aux.size());
        } else {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *)t->data, aux.data(), aux.size());
        }
        analyze_iq4ks(t->name, t->ne[1], t->ne[0], aux.data(), tot_mse, tot_elements);
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
        float tot_mse = 0, tot_elements = 0;
        for (const auto& kv_tensor : tensors) {
            if (!layer_included(params, kv_tensor.first)) {
                continue;
            }
            if (kv_tensor.second->ne[0] == 1 || kv_tensor.second->ne[1] == 1) {
                // we never quantize those
                continue;
            }
            analyze_iq4ks(kv_tensor.second, tot_mse, tot_elements);
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
