#define LLAMA_API_INTERNAL
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "iqk/iqk_quantize.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"

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
#include <fstream>
#include <memory>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct quantize_stats_params {
    std::string model = DEFAULT_MODEL_PATH;
    bool verbose = false;
    bool per_layer_stats = false;
    bool print_histogram = false;
    bool reference = false;
    bool transpose = false;
    std::vector<std::string> include_layers;
    std::vector<std::string> exclude_layers;
    std::vector<enum ggml_type> include_types;
};

using Imatrix = std::unordered_map<std::string, std::vector<float>>;

std::pair<std::vector<float>, std::vector<float>> split_tensor(const ggml_tensor * layer,
        std::vector<float> & input_scratch, const Imatrix& imatrix) {

    if (layer->ne[0] == 1 || layer->ne[1] == 1) return {};

    if (!ggml_is_contiguous(layer)) {
        printf("%s: %s is not contiguous\n", __func__, layer->name);
        return {};
    }

    int nelements = ggml_nelements(layer);

    float* input_scratch_ptr = nullptr;
    if (layer->type == GGML_TYPE_F16) {
        if ((int)input_scratch.size() < nelements) input_scratch.resize(nelements);
        for (int i = 0; i < nelements; ++i) {
            input_scratch[i] = ggml_get_f32_1d(layer, i);
        }
        input_scratch_ptr = input_scratch.data();
    } else {
        input_scratch_ptr = ggml_get_data_f32(layer);
    }

    int n_per_row = layer->ne[0];
    int nrows  = nelements/n_per_row;

    const float * imatrix_data = nullptr;
    if (auto it = imatrix.find(layer->name); it != imatrix.end() && int(it->second.size()) == n_per_row) {
        imatrix_data = it->second.data();
    }

    std::vector<uint16_t> order(n_per_row);
    if (!iqk_reorder(layer, imatrix_data, order.data())) {
        return {};
    }

    int nblock = n_per_row/256;
    int nblock_high = int(nblock*0.1f + 0.5f);
    if (nblock_high == 0) return {};

    std::sort(order.data(), order.data() + 256*nblock_high);
    std::sort(order.data() + 256*nblock_high, order.data() + 256*nblock);

    std::vector<float> part1(256*nblock_high*nrows);
    std::vector<float> part2(256*(nblock-nblock_high)*nrows);

    for (int row = 0; row < nrows; ++row) {
        auto x = input_scratch_ptr + row*n_per_row;
        auto yh = part1.data() + 256*nblock_high*row;
        auto yl = part2.data() + 256*(nblock-nblock_high)*row;
        for (int j = 0; j < 256*nblock_high; ++j) yh[j] = x[order[j]];
        for (int j = 256*nblock_high; j < 256*nblock; ++j) yl[j-256*nblock_high] = x[order[j]];
    }

    return std::make_pair(std::move(part1), std::move(part2));
}

ggml_type get_better_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_K: return GGML_TYPE_IQ3_K;
        case GGML_TYPE_IQ3_K: return GGML_TYPE_IQ4_K;
        case GGML_TYPE_IQ4_K: return GGML_TYPE_IQ5_K;
        case GGML_TYPE_IQ5_K: return GGML_TYPE_IQ6_K;
        case GGML_TYPE_IQ6_K: return GGML_TYPE_Q8_0;
        case GGML_TYPE_Q2_K: return GGML_TYPE_Q3_K;
        case GGML_TYPE_Q3_K: return GGML_TYPE_Q4_K;
        case GGML_TYPE_Q4_K: return GGML_TYPE_Q5_K;
        case GGML_TYPE_Q5_K: return GGML_TYPE_Q6_K;
        case GGML_TYPE_Q6_K: return GGML_TYPE_Q8_0;
        case GGML_TYPE_IQ2_XXS: return GGML_TYPE_IQ3_XXS;
        case GGML_TYPE_IQ2_XS:  return GGML_TYPE_IQ3_XXS;
        case GGML_TYPE_IQ2_S:   return GGML_TYPE_IQ3_S;
        case GGML_TYPE_IQ3_XXS: return GGML_TYPE_IQ4_XS;
        case GGML_TYPE_IQ3_S:   return GGML_TYPE_IQ4_XS;
        case GGML_TYPE_IQ4_XS:  return GGML_TYPE_IQ5_K;
        default: throw std::runtime_error("No better type");
    }
}


static void analyze_layer(const std::string & name, const ggml_tensor * layer, std::vector<float> & input_scratch,
        const Imatrix& imatrix) {

    if (layer->ne[0] == 1 || layer->ne[1] == 1) return;

    if (!ggml_is_contiguous(layer)) {
        printf("%s: %s is not contiguous\n", __func__, layer->name);
        return;
    }

    int nelements = ggml_nelements(layer);

    float* input_scratch_ptr = nullptr;
    if (layer->type == GGML_TYPE_F16) {
        if ((int)input_scratch.size() < nelements) input_scratch.resize(nelements);
        for (int i = 0; i < nelements; ++i) {
            input_scratch[i] = ggml_get_f32_1d(layer, i);
        }
        input_scratch_ptr = input_scratch.data();
    } else {
        input_scratch_ptr = ggml_get_data_f32(layer);
    }

    int n_per_row = layer->ne[0];
    int nrows  = nelements/n_per_row;

    std::vector<std::pair<float,int>> sumv(n_per_row);
    for (int j = 0; j < n_per_row; ++j) sumv[j] = {0.f, j};

    for (int row = 0; row < nrows; ++row) {
        auto x = input_scratch_ptr + row*n_per_row;
        for (int j = 0; j < n_per_row; ++j) sumv[j].first += x[j]*x[j];
    }

    auto it = imatrix.find(layer->name);
    bool have_imatrix = false;
    if (it != imatrix.end() && int(it->second.size()) == n_per_row) {
        have_imatrix = true;
        for (int j = 0; j < n_per_row; ++j) sumv[j].first *= it->second[j];
    }
    std::sort(sumv.begin(), sumv.end(), std::greater<std::pair<float,int>>{});

    printf("%s:  %g  %g  %g\n", name.c_str(), sumv.front().first, sumv[n_per_row/2].first, sumv.back().first);

    int nblock = n_per_row/256;
    int nblock_high = int(nblock*0.1f + 0.5f);
    if (nblock_high == 0) return;

    std::vector<float> part1(256*nblock_high*nrows);
    std::vector<float> part2(256*(nblock-nblock_high)*nrows);

    for (int row = 0; row < nrows; ++row) {
        auto x = input_scratch_ptr + row*n_per_row;
        auto yh = part1.data() + 256*nblock_high*row;
        auto yl = part2.data() + 256*(nblock-nblock_high)*row;
        for (int j = 0; j < 256*nblock_high; ++j) yh[j] = x[sumv[j].second];
        for (int j = 256*nblock_high; j < 256*nblock; ++j) yl[j-256*nblock_high] = x[sumv[j].second];
    }

    std::vector<float> reordered_imatrix;
    if (have_imatrix) {
        reordered_imatrix.resize(256*nblock);
        for (int j = 0; j < 256*nblock; ++j) reordered_imatrix[j] = it->second[sumv[j].second];
    }

    auto row_size_h = ggml_row_size(GGML_TYPE_IQ3_K, 256*nblock_high);
    auto row_size_l = ggml_row_size(GGML_TYPE_IQ2_K, 256*(nblock-nblock_high));

    std::vector<char> qdata_h(row_size_h*nrows);
    std::vector<char> qdata_l(row_size_l*nrows);

    ggml_quantize_chunk(GGML_TYPE_IQ3_K, part1.data(), (void *)qdata_h.data(), 0, nrows, 256*nblock_high, reordered_imatrix.data());
    ggml_quantize_chunk(GGML_TYPE_IQ2_K, part2.data(), (void *)qdata_l.data(), 0, nrows, 256*(nblock-nblock_high),
            have_imatrix ? reordered_imatrix.data() + 256*nblock_high : nullptr);

    std::vector<float> deq_part1(256*nblock_high*nrows);
    std::vector<float> deq_part2(256*(nblock-nblock_high)*nrows);

    dequantize_row_iq3_k((const block_iq3_k *)qdata_h.data(), deq_part1.data(), deq_part1.size());
    dequantize_row_iq2_k((const block_iq2_k *)qdata_l.data(), deq_part2.data(), deq_part2.size());

    double mse = 0, sumw = 0;
    for (int row = 0; row < nrows; ++row) {
        auto xh = part1.data() + 256*nblock_high*row;
        auto xl = part2.data() + 256*(nblock-nblock_high)*row;
        auto yh = deq_part1.data() + 256*nblock_high*row;
        auto yl = deq_part2.data() + 256*(nblock-nblock_high)*row;
        for (int j = 0; j < 256*nblock_high; ++j) {
            float w = have_imatrix ? reordered_imatrix[j] : 1;
            float diff = xh[j] - yh[j];
            mse += w*diff*diff;
            sumw += w;
        }
        for (int j = 0; j < 256*(nblock-nblock_high); ++j) {
            float w = have_imatrix ? reordered_imatrix[j+256*nblock_high] : 1;
            float diff = xl[j] - yl[j];
            mse += w*diff*diff;
            sumw += w;
        }
    }
    printf("    rmse = %g\n", sqrt(mse/sumw));

}

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

static void test_roundtrip_on_chunk(bool fill_data,
    const ggml_tensor * layer, int64_t offset, int64_t chunk_size, const ggml_type_traits_t & qfns, bool use_reference,
    float * input_scratch, char * quantized_scratch, float * output_scratch, error_stats & stats) {
    if (fill_data) {
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


// Run quantization function for a single layer and update error stats
static void test_roundtrip_on_layer(bool transpose,
    const std::string & name, bool print_layer_stats, const ggml_type_traits_t & qfns, bool use_reference,
    const ggml_tensor * layer, std::vector<float> & input_scratch, std::vector<char> & quantized_scratch,
    std::vector<float> & output_scratch, error_stats & total_error, int max_thread = 0) {
    assert(tensor_is_contiguous(layer));
    error_stats layer_error {};
    uint64_t nelements = ggml_nelements(layer);

    float* input_scratch_ptr = nullptr;
    if (transpose) {
        if (layer->ne[2] > 1 || layer->ne[3] > 1 || layer->ne[1] < 256 || !ggml_is_contiguous(layer)) {
            printf("%s: transpose option requires contiguous 2D tensor with >= 256 rows\n", __func__);
            return;
        }
        if (input_scratch.size() < nelements) input_scratch.resize(nelements);
        if (layer->type == GGML_TYPE_F16) {
            const ggml_fp16_t * data = (const ggml_fp16_t *)layer->data;
            for (int i = 0; i < layer->ne[0]; ++i) for (int j = 0; j < layer->ne[1]; ++j) {
                input_scratch[i*layer->ne[1] + j] = ggml_fp16_to_fp32(data[j*layer->ne[0] + i]);
            }
        }
        else if (layer->type == GGML_TYPE_F32) {
            const float * data = (const float *)layer->data;
            for (int i = 0; i < layer->ne[0]; ++i) for (int j = 0; j < layer->ne[1]; ++j) {
                input_scratch[i*layer->ne[1] + j] = data[j*layer->ne[0] + i];
            }
        }
        else {
            printf("%s: unsupported type %s\n", __func__, ggml_type_name(layer->type));
            return;
        }
        input_scratch_ptr = input_scratch.data();
    }
    else {
        if (layer->type == GGML_TYPE_F16) {
            if (input_scratch.size() < nelements) input_scratch.resize(nelements);
            input_scratch_ptr = input_scratch.data();
        }
    }
    if (quantized_scratch.size() < 4*nelements) quantized_scratch.resize(4*nelements);
    if (output_scratch.size() < nelements) output_scratch.resize(nelements);

    if (max_thread < 1) max_thread = std::thread::hardware_concurrency();
    int chunk_size = 32*512;
    int num_chunks = (nelements + chunk_size - 1)/chunk_size;

    if (num_chunks < 2 || max_thread < 2) {
        test_roundtrip_on_chunk(!transpose, layer, 0, nelements, qfns, use_reference, input_scratch_ptr, quantized_scratch.data(),
                output_scratch.data(), print_layer_stats ? layer_error : total_error);
    } else {
        auto & stats = print_layer_stats ? layer_error : total_error;
        std::mutex mutex;
        uint64_t counter = 0;
        auto compute = [&mutex, &counter, &stats, &qfns, nelements, layer, use_reference, input_scratch_ptr,
             &quantized_scratch, &output_scratch, chunk_size, transpose] () {
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
                test_roundtrip_on_chunk(!transpose, layer, offset, chunk, qfns, use_reference, input_scratch_ptr + offset,
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

static int load_imatrix(const std::string & imatrix_file, std::string & imatrix_dataset, Imatrix& imatrix_data) {
    std::ifstream in(imatrix_file.c_str(), std::ios::binary);
    if (!in) {
        printf("%s: failed to open %s\n",__func__, imatrix_file.c_str());
        exit(1);
    }
    int n_entries;
    in.read((char *)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        printf("%s: no data in file %s\n", __func__, imatrix_file.c_str());
        exit(1);
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            printf("%s: failed reading name for entry %d from %s\n", __func__, i+1, imatrix_file.c_str());
            exit(1);
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto & e = imatrix_data[name];
        int ncall;
        in.read((char *)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            printf("%s: failed reading number of values for entry %d\n", __func__, i);
            imatrix_data = {};
            exit(1);
        }
        e.resize(nval);
        in.read((char *)e.data(), nval*sizeof(float));
        if (in.fail()) {
            printf("%s: failed reading data for entry %d\n", __func__, i);
            imatrix_data = {};
            exit(1);
        }
        if (ncall > 0) {
            for (auto& v : e) v /= ncall;
        }

        if (getenv("LLAMA_TRACE")) {
            printf("%s: loaded data (size = %6d, ncall = %6d) for '%s'\n", __func__, int(e.size()), ncall, name.c_str());
        }
    }

    // latest imatrix version contains the dataset filename at the end of the file
    int m_last_call = 0;
    if (in.peek() != EOF) {
        in.read((char *)&m_last_call, sizeof(m_last_call));
        int dataset_len;
        in.read((char *)&dataset_len, sizeof(dataset_len));
        std::vector<char> dataset_as_vec(dataset_len);
        in.read(dataset_as_vec.data(), dataset_len);
        imatrix_dataset.assign(dataset_as_vec.begin(), dataset_as_vec.end());
        printf("%s: imatrix dataset='%s'\n", __func__, imatrix_dataset.c_str());
    }
    printf("%s: loaded %d importance matrix entries from %s computed on %d chunks\n", __func__, int(imatrix_data.size()), imatrix_file.c_str(), m_last_call);
    return m_last_call;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    quantize_stats_params params;

    // read command line

    int max_thread = 0;
    bool invalid_param = false;
    bool analyze = false;
    std::string arg;
    std::string imatrix_file;
    float split_fraction = -1.f;
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
        } else if (arg == "-a" || arg == "--analyze") {
            analyze = true;
        } else if (arg == "--transpose") {
            params.transpose = true;
        } else if (arg == "--histogram") {
            params.print_histogram = true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "--split") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            split_fraction = atof(argv[i]);
        } else if (arg == "-im" || arg == "--imatrix") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            imatrix_file = argv[i];
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

    Imatrix imatrix;
    if (!imatrix_file.empty()) {
        std::string dum;
        load_imatrix(imatrix_file, dum, imatrix);
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
                if (analyze) {
                    analyze_layer(layer_name, kv_tensor.second, input_scratch, imatrix);
                } else if (split_fraction > 0 && split_fraction < 1) {
                    auto [part_h, part_l] = split_tensor(kv_tensor.second, input_scratch, imatrix);
                    if (part_h.empty() || part_l.empty()) continue;
                    auto h_type = get_better_type(type);
                    auto h_qfns = ggml_internal_get_type_traits(h_type);
                    if (!h_qfns.from_float || !h_qfns.to_float) continue;
                    std::string name1{kv_tensor.second->name}, name2(name1);
                    name1 += "_part1";
                    name2 += "_part2";
                    ggml_tensor part1, part2;
                    std::memcpy(part1.name, name1.data(), name1.size() < 64 ? name1.size() + 1 : 64);
                    std::memcpy(part2.name, name2.data(), name2.size() < 64 ? name2.size() + 1 : 64);
                    //snprintf(part1.name, 64, "%s_part1", kv_tensor.second->name);
                    //snprintf(part2.name, 64, "%s_part2", kv_tensor.second->name);
                    auto nrows = ggml_nrows(kv_tensor.second);
                    part1.ne[0] = part_h.size()/nrows;
                    part1.ne[1] = part_h.size()/part1.ne[0];
                    part1.ne[2] = part1.ne[3] = 1;
                    part2.ne[0] = part_l.size()/nrows;
                    part2.ne[1] = part_l.size()/part2.ne[0];
                    part2.ne[2] = part2.ne[3] = 1;
                    part1.type = part2.type = GGML_TYPE_F32;
                    part1.nb[0] = part2.nb[0] = sizeof(float);
                    for (int k = 1; k < 4; ++k) part1.nb[k] = part1.nb[k-1]*part1.ne[k-1];
                    for (int k = 1; k < 4; ++k) part2.nb[k] = part2.nb[k-1]*part2.ne[k-1];
                    part1.data = (void *)part_h.data();
                    part2.data = (void *)part_l.data();
                    test_roundtrip_on_layer(false,
                            std::string(part1.name),
                            params.per_layer_stats,
                            h_qfns,
                            params.reference,
                            &part1,
                            input_scratch,
                            quantized_scratch,
                            output_scratch,
                            global_stats,
                            max_thread);
                    test_roundtrip_on_layer(false,
                            std::string(part2.name),
                            params.per_layer_stats,
                            qfns,
                            params.reference,
                            &part2,
                            input_scratch,
                            quantized_scratch,
                            output_scratch,
                            global_stats,
                            max_thread);
                } else {
                    test_roundtrip_on_layer(params.transpose,
                            layer_name,
                            params.per_layer_stats,
                            qfns,
                            params.reference,
                            kv_tensor.second,
                            input_scratch,
                            quantized_scratch,
                            output_scratch,
                            global_stats,
                            max_thread);
                }
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
