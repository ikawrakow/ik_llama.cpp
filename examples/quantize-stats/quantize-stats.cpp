#define LLAMA_API_INTERNAL
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "iqk/iqk_quantize.h"

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
#include <chrono>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
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
    float * input_scratch, char * quantized_scratch, float * output_scratch, error_stats & stats, bool fill_input) {
    if (fill_input) {
        if (layer->type == GGML_TYPE_F16) {
            for (int i = 0; i < chunk_size; i++) {
                input_scratch[i] = ggml_get_f32_1d(layer, i + offset);
            }
        } else {
            input_scratch = ggml_get_data_f32(layer) + offset;
        }
    }

    if (use_reference) {
        qfns.from_float_ref(input_scratch, quantized_scratch, chunk_size);
    } else {
        qfns.from_float(input_scratch, quantized_scratch, chunk_size);
    }
    qfns.to_float(quantized_scratch, output_scratch, chunk_size);

    update_error_stats(chunk_size, input_scratch, output_scratch, stats);
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
#endif

static void f_helper(int nrows, int stride, float norm, const float * g, const float * q, float * f, float& sum_f) {
#ifdef __AVX2__
    auto vnorm = _mm256_set1_ps(norm);
    __m256 sums[8] = {};
    for (int row = 0; row < nrows; ++row) {
        __m256 vg = _mm256_set1_ps(g[row]);
        auto qr = q + row*stride;
        for (int k = 0; k < 8; ++k) {
            auto vq = _mm256_loadu_ps(qr + 8*k);
            sums[k] = _mm256_fmadd_ps(vg, vq, sums[k]);
        }
    }
    __m256 tot = _mm256_setzero_ps();
    for (int k = 0; k < 8; ++k) {
        sums[k] = _mm256_mul_ps(vnorm, sums[k]);
        _mm256_storeu_ps(f + 8*k, sums[k]);
        tot = _mm256_fmadd_ps(sums[k], sums[k], tot);
        sums[k] = _mm256_setzero_ps();
    }
    sum_f += hsum_float_8(tot);
#else
    std::memset(f, 0, 64*sizeof(float));
    for (int row = 0; row < nrows; ++row) {
        auto qr = q + row*stride;
        for (int k = 0; k < 64; ++k) {
            f[k] += qr[k]*g[row];
        }
    }
    float s = 0;
    for (int k = 0; k < 64; ++k) {
        f[k] *= norm;
        s += f[k]*f[k];
    }
    sum_f += s;
#endif
}

static void g_helper(int n_per_row, const float * qr, const float * f, float norm, float& g, float& mse) {
    float sum_g = 0;
    float sum = 0;
#ifdef __AVX2__
    __m256 vsum = _mm256_setzero_ps();
    for (int j = 0; j < n_per_row; j += 8) {
        auto vq = _mm256_loadu_ps(qr + j);
        auto vf = _mm256_loadu_ps(f + j);
        vsum = _mm256_fmadd_ps(vq, vf, vsum);
    }
    sum = hsum_float_8(vsum);
#else
    for (int j = 0; j < n_per_row; ++j) sum += qr[j]*f[j];
#endif
    g = sum * norm;
#ifdef __AVX2__
    auto vg = _mm256_set1_ps(g);
    auto vmse = _mm256_setzero_ps();
    for (int j = 0; j < n_per_row; j += 8) {
        auto vq = _mm256_loadu_ps(qr + j);
        auto vf = _mm256_loadu_ps(f + j);
        auto vdiff = _mm256_sub_ps(vq, _mm256_mul_ps(vg, vf));
        vmse = _mm256_fmadd_ps(vdiff, vdiff, vmse);
    }
    mse += hsum_float_8(vmse);
#else
    for (int j = 0; j < n_per_row; ++j) {
        float diff = qr[j] - g*f[j];
        mse += diff*diff;
    }
#endif
}

static void do_svd_iteration(int n_per_row, int nrows, const float * q, float * f, float * g, float& f_norm, float& mse,
        std::vector<std::thread>& workers, std::vector<float>& work) {
    GGML_ASSERT(n_per_row % 64 == 0);
    GGML_ASSERT(nrows % 16 == 0);
    GGML_ASSERT(!workers.empty());

    if (work.size() < 2*workers.size()) work.resize(2*workers.size());
    int nblock = n_per_row/64;

    auto compute_f = [&] (int ith) {
        float sum_f = 0;
        for (int i = ith; i < nblock; i += workers.size()) {
            f_helper(nrows, n_per_row, f_norm, g, q + 64*i, f + 64*i, sum_f);
        }
        work[ith] = sum_f;
    };
    for (int i = 0; i < int(workers.size())-1; ++i) workers[i] = std::thread(compute_f, i);
    compute_f(workers.size()-1);
    for (int i = 0; i < int(workers.size())-1; ++i) workers[i].join();

    float sum_f = 0; for (int i = 0; i < int(workers.size()); ++i) sum_f += work[i];
    float g_norm = 1/sum_f;

    nblock = nrows/16;
    auto compute_g = [&] (int ith) {
        float sum_g = 0, mse = 0;
        for (int i = ith; i < nblock; i += workers.size()) {
            for (int j = 0; j < 16; ++j) {
                g_helper(n_per_row, q + (16*i + j)*n_per_row, f, g_norm, g[16*i + j], mse);
                sum_g += g[16*i + j]*g[16*i + j];
            }
        }
        work[2*ith+0] = sum_g;
        work[2*ith+1] = mse;
    };
    for (int i = 0; i < int(workers.size())-1; ++i) workers[i] = std::thread(compute_g, i);
    compute_g(workers.size()-1);
    for (int i = 0; i < int(workers.size())-1; ++i) workers[i].join();

    float sum_g = 0; mse = 0;
    for (int i = 0; i < int(workers.size()); ++i) {
        sum_g += work[2*i+0];
        mse   += work[2*i+1];
    }

    f_norm = 1/sum_g;

}

static void try_lora(int n_per_row, int nrows, const float * x, float * q, int nsvd_iter, int verbosity = 1) {
    constexpr int kNiter = 10;
    if (nsvd_iter < 1) nsvd_iter = kNiter;
    std::vector<float> f(n_per_row, 1), aux(n_per_row), g(nrows, 1);
    for (int iter = 0; iter < nsvd_iter; ++iter) {
        float mse0 = 0;
        for (int row = 0; row < nrows; ++row) {
            const float * xr = x + row*n_per_row;
            const float * qr = q + row*n_per_row;
            float sumqx = 0, sumq2 = 0;
            for (int j = 0; j < n_per_row; ++j) {
                float diff = xr[j] - g[row]*f[j]*qr[j];
                mse0 += diff*diff;
                float w = f[j]*qr[j];
                sumqx += xr[j]*w;
                sumq2 += w*w;
            }
            g[row] = sumq2 > 0 ? sumqx/sumq2 : 1;
        }
        std::memset(f.data(), 0, f.size()*sizeof(float));
        std::memset(aux.data(), 0, aux.size()*sizeof(float));
        for (int row = 0; row < nrows; ++row) {
            const float * xr = x + row*n_per_row;
            const float * qr = q + row*n_per_row;
            for (int j = 0; j < n_per_row; ++j) {
                float w = g[row]*qr[j];
                f[j] += w*xr[j];
                aux[j] += w*w;
            }
        }
        for (int j = 0; j < n_per_row; ++j) if (aux[j] > 0) f[j] /= aux[j];
        float mse = 0;
        for (int row = 0; row < nrows; ++row) {
            const float * xr = x + row*n_per_row;
            const float * qr = q + row*n_per_row;
            for (int j = 0; j < n_per_row; ++j) {
                float diff = xr[j] - g[row]*f[j]*qr[j];
                mse += diff*diff;
            }
        }
        printf("%s(%d): rmse0 = %g, rmse = %g\n", __func__, iter, sqrt(mse0/(n_per_row*nrows)), sqrt(mse/(n_per_row*nrows)));
    }
}

static void try_svd(int n_per_row, int nrows, const float * b, float * q, int nsvd, int nsvd_iter, char * scratch, int verbosity = 1) {
    constexpr int kNiter = 10;
    if (nsvd_iter < 1) nsvd_iter = kNiter;
    if (nsvd > nrows) nsvd = nrows;
    auto tim1 = std::chrono::steady_clock::now();
    int nelem = n_per_row*nrows;
    double mse = 0;
    bool use_avx2 = false;
#ifdef __AVX2__
    GGML_ASSERT(n_per_row%64 == 0);
    use_avx2 = true;
#endif
    float max_error = 0;
    for (int j = 0; j < nelem; ++j) {
        q[j] = b[j] - q[j];
        mse += q[j]*q[j];
        max_error = std::max(max_error, std::abs(q[j]));
    }
    int nthread = std::max(1, int(std::thread::hardware_concurrency()/2));
    std::vector<std::thread> workers(nthread);
    std::vector<float> work;
    if (verbosity > 0) printf("===================== %s(%d x %d, %d, %d): rmse = %g, max_err = %g\n", __func__,
            n_per_row, nrows, nsvd, use_avx2, sqrt(mse/nelem), max_error);
    float mse_old = mse;
    std::vector<float> f(n_per_row), g(nrows, 1);
    for (int isvd = 0; isvd < nsvd; ++isvd) {
        if (verbosity > 1) printf("--- isvd = %d\n", isvd);
        float norm = 1.f/nrows;
        for (int iter = 0; iter < nsvd_iter; ++iter) {
            float this_mse = 0;
            do_svd_iteration(n_per_row, nrows, q, f.data(), g.data(), norm, this_mse, workers, work);
            if (verbosity > 1) printf("    after %d iterations: %g\n", iter+1, sqrt(this_mse/nelem));
            if (mse_old/this_mse - 1 < 1e-6f) break;
            mse_old = this_mse;
        }
        if (false) {
            quantize_iq2_k(f.data(), (block_iq2_k *)scratch, 1, n_per_row, nullptr);
            dequantize_row_iq2_k((block_iq2_k *)scratch, f.data(), n_per_row);
            quantize_iq2_k(g.data(), (block_iq2_k *)scratch, 1, nrows, nullptr);
            dequantize_row_iq2_k((block_iq2_k *)scratch, g.data(), nrows);
            //quantize_iq4_k(f.data(), (block_iq4_k *)scratch, 1, n_per_row, nullptr);
            //dequantize_row_iq4_k((block_iq4_k *)scratch, f.data(), n_per_row);
            //quantize_iq4_k(g.data(), (block_iq4_k *)scratch, 1, nrows, nullptr);
            //dequantize_row_iq4_k((block_iq4_k *)scratch, g.data(), nrows);
        }
#ifdef __AVX2__
        for (int row = 0; row < nrows; ++row) {
            auto qr = q + row*n_per_row;
            auto vg = _mm256_set1_ps(g[row]);
            for (int j = 0; j < n_per_row; j += 8) {
                auto vf = _mm256_loadu_ps(f.data() + j);
                auto vq = _mm256_loadu_ps(qr + j);
                vq = _mm256_sub_ps(vq, _mm256_mul_ps(vf, vg));
                _mm256_storeu_ps(qr + j, vq);
            }
            g[row] = 1;
        }
#else
        for (int row = 0; row < nrows; ++row) {
            auto qr = q + row*n_per_row;
            for (int j = 0; j < n_per_row; ++j) {
                qr[j] -= g[row]*f[j];
            }
            g[row] = 1;
        }
#endif
    }
    auto tim2 = std::chrono::steady_clock::now();
    if (verbosity > 0) {
        max_error = 0;
#ifdef __AVX2__
        auto vmax = _mm256_setzero_ps();
        auto sign = _mm256_set1_ps(-0.0f);
        for (int row = 0; row < nrows; ++row) {
            auto qr = q + row*n_per_row;
            for (int j = 0; j < n_per_row; j += 8) {
                auto vq = _mm256_loadu_ps(qr + j);
                vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, vq));
            }
        }
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(vmax, 1), _mm256_castps256_ps128(vmax));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        max_error = _mm_cvtss_f32(max4);
#else
        for (int row = 0; row < nrows; ++row) {
            auto qr = q + row*n_per_row;
            for (int j = 0; j < n_per_row; ++j) {
                max_error = std::max(max_error, std::abs(qr[j]));
            }
        }
#endif
        printf("%s: finished in %g s. Final rmse = %g max_error = %g\n", __func__,
                1e-3*std::chrono::duration_cast<std::chrono::milliseconds>(tim2-tim1).count(), sqrt(mse_old/nelem), max_error);
    }
}

// Run quantization function for a single layer and update error stats
static void test_roundtrip_on_layer(
    std::string & name, bool print_layer_stats, const ggml_type_traits_t & qfns, bool use_reference,
    const ggml_tensor * layer, std::vector<float> & input_scratch, std::vector<char> & quantized_scratch,
    std::vector<float> & output_scratch, error_stats & total_error, int nsvd_before, int nsvd_after,
    bool do_lora, int nsvd_iter, int verbosity, int max_thread = 0) {
    assert(tensor_is_contiguous(layer));
    error_stats layer_error {};
    uint64_t nelements = ggml_nelements(layer);

    float* input_scratch_ptr = nullptr;
    if (layer->type == GGML_TYPE_F16) {
        if (input_scratch.size() < nelements) input_scratch.resize(nelements);
        input_scratch_ptr = input_scratch.data();
    }
    if (output_scratch.size() < nelements) output_scratch.resize(nelements);
    if (quantized_scratch.size() < 4*nelements) quantized_scratch.resize(4*nelements);

    bool fill_input = true;
    if (nsvd_before > 0 && layer->ne[0] > 1 && layer->ne[1] > 1 && layer->ne[2] == 1 && layer->ne[3] == 1) {
        if (layer->type == GGML_TYPE_F16) {
            for (int i = 0; i < nelements; i++) {
                input_scratch[i] = ggml_get_f32_1d(layer, i);
            }
        } else {
            printf("%s: f32 is not supported\n", __func__);
            return;
            //input_scratch = ggml_get_data_f32(layer) + 0;
        }
        std::memset(output_scratch.data(), 0, nelements*sizeof(float));
        try_svd(layer->ne[0], layer->ne[1], input_scratch_ptr, output_scratch.data(), nsvd_before, nsvd_iter, quantized_scratch.data(), verbosity);
        std::memcpy(input_scratch_ptr, output_scratch.data(), nelements*sizeof(float));
        fill_input = false;
    }

    if (max_thread < 1) max_thread = std::thread::hardware_concurrency();
    int chunk_size = 32*512;
    int num_chunks = (nelements + chunk_size - 1)/chunk_size;

    if (num_chunks < 2 || max_thread < 2) {
        test_roundtrip_on_chunk(layer, 0, nelements, qfns, use_reference, input_scratch_ptr, quantized_scratch.data(),
                output_scratch.data(), print_layer_stats ? layer_error : total_error, fill_input);
    } else {
        auto & stats = print_layer_stats ? layer_error : total_error;
        std::mutex mutex;
        uint64_t counter = 0;
        auto compute = [&mutex, &counter, &stats, &qfns, nelements, layer, use_reference, input_scratch_ptr,
             &quantized_scratch, &output_scratch, chunk_size, fill_input] () {
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
                        quantized_scratch.data() + 4*offset, output_scratch.data() + offset, local_stats, fill_input);
            }
        };
        int nthread = std::min(num_chunks, max_thread);
        std::vector<std::thread> workers(nthread-1);
        for (auto& w : workers) w = std::thread(compute);
        compute();
        for (auto& w : workers) w.join();
    }

    if (do_lora) {
        try_lora(layer->ne[0], layer->ne[1], input_scratch_ptr, output_scratch.data(), nsvd_iter, verbosity);
    }

    if (print_layer_stats) {
        print_error_stats(name, layer_error, false);
        combine_error_stats(total_error, layer_error);
    }

    if (nsvd_after > 0 && layer->ne[0] > 1 && layer->ne[1] > 1 && layer->ne[2] == 1 && layer->ne[3] == 1) {
        try_svd(layer->ne[0], layer->ne[1], input_scratch_ptr, output_scratch.data(), nsvd_after, nsvd_iter, quantized_scratch.data(), verbosity);
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();

    quantize_stats_params params;

    // read command line

    int max_thread = 0;
    int nsvd_before = 0;
    int nsvd_after  = 0;
    int nsvd_iter = 0;
    int verbosity = 1;
    bool do_lora = false;
    bool invalid_param = false;
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
        } else if (arg == "--histogram") {
            params.print_histogram = true;
        } else if (arg == "--lora") {
            do_lora = true;
        } else if (arg == "--svd-before") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            nsvd_before = atoi(argv[i]);
        } else if (arg == "--svd-after") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            nsvd_after = atoi(argv[i]);
        } else if (arg == "-ni" || arg == "--svd-iterations") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            nsvd_iter = atoi(argv[i]);
        } else if (arg == "-sv" || arg == "--svd-verbosity") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            verbosity = atoi(argv[i]);
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
    if (do_lora && (nsvd_before > 0 || nsvd_after > 0)) {
        fprintf(stderr, "error: lora cannot be combined with SVD\n");
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
                        nsvd_before, nsvd_after,
                        do_lora,
                        nsvd_iter,
                        verbosity,
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
