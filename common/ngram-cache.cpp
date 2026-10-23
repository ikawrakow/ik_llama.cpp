#include "ngram-cache.h"
#include "common.h"
#include "log.h"

#include <cstdint>
#include <fstream>

void common_ngram_cache_update(common_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = ggml_time_ms();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;

    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            common_ngram ngram(&inp[ngram_start], ngram_size);
            const llama_token token = inp[i];

            common_ngram_cache::iterator part_it = ngram_cache.find(ngram);
            if (part_it == ngram_cache.end()) {
                common_ngram_cache_part part;
                part.emplace(token, 1);
                ngram_cache.emplace(ngram, part);
            } else {
                common_ngram_cache_part::iterator token_count_it = part_it->second.find(token);
                if (token_count_it == part_it->second.end()) {
                    part_it->second.emplace(token, 1);
                } else {
                    token_count_it->second++;
                }
            }
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = ggml_time_ms();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 2,  2,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
static llama_token try_draft(const common_ngram_cache & nc_static, const common_ngram ngram_static) {
    common_ngram_cache::const_iterator part_static_it = nc_static.find(ngram_static);
    if (part_static_it == nc_static.end()) {
        return LLAMA_TOKEN_NULL;
    }
    const common_ngram_cache_part & part_static = part_static_it->second;

    int max_count_static  = 0;
    int sum_count_static  = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    for (const auto & token_count_static : part_static) {
        const llama_token token = token_count_static.first;
        const int32_t count_static  = token_count_static.second;

        if (count_static > max_count_static) {
            max_token        = token;
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    if (sum_count_static < draft_min_sample_size_lax[LLAMA_NGRAM_STATIC-1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100*max_count_static < draft_min_percent_lax[LLAMA_NGRAM_STATIC-1]*sum_count_static) {
        return LLAMA_TOKEN_NULL;
    }
    return max_token;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static llama_token try_draft(
    const common_ngram_cache & nc_primary, const std::vector<std::pair<int, common_ngram>> & ngrams_primary, const common_ngram_cache_part & part_static,
    const int * min_sample_size, const int * min_percent) {

    llama_token drafted_token = LLAMA_TOKEN_NULL;

    for (size_t i = ngrams_primary.size(); i > 0 && drafted_token == LLAMA_TOKEN_NULL; --i) {
        const int ngram_size = ngrams_primary[i - 1].first;
        const common_ngram ngram_primary = ngrams_primary[i - 1].second;

        common_ngram_cache::const_iterator part_primary_it = nc_primary.find(ngram_primary);
        if (part_primary_it == nc_primary.end()) {
            continue;
        }
        const common_ngram_cache_part & part_primary = part_primary_it->second;

        int max_count_primary = 0;
        int max_count_static  = 0;
        int sum_count_primary = 0;
        llama_token max_token = LLAMA_TOKEN_NULL;

        for (const auto & token_count_primary : part_primary) {
            const llama_token token = token_count_primary.first;

            common_ngram_cache_part::const_iterator token_count_static_it = part_static.find(token);

            const int32_t count_primary = token_count_primary.second;
            const int32_t count_static  = token_count_static_it != part_static.end() ? 100*token_count_static_it->second : 1;

            if (count_primary*count_static > max_count_primary*max_count_static) {
                max_token         = token;
                max_count_primary = count_primary;
                max_count_static  = count_static;
            }
            sum_count_primary += count_primary;
        }

        const int threshold_index = ngram_size - 1;
        GGML_ASSERT(threshold_index >= 0 && threshold_index < LLAMA_NGRAM_MAX);
        if (sum_count_primary < min_sample_size[threshold_index]) {
            continue;
        }
        if (100*max_count_primary < min_percent[threshold_index]*sum_count_primary) {
            continue;;
        }
        drafted_token = max_token;
    }

    return drafted_token;
}

void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    common_ngram_cache & nc_context, common_ngram_cache & nc_dynamic, common_ngram_cache & nc_static
) {
    GGML_ASSERT(draft.size() == 1);
    if (inp.size() < LLAMA_NGRAM_STATIC) {
        return;
    }

    while ((int) draft.size()-1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;
        const size_t n_available = inp.size() + draft.size() - 1;

        // Only input and prior proposals are history, the carry slot is not.
        const int ngram_max_available = std::min(ngram_max,
                static_cast<int>(std::min<size_t>(n_available, LLAMA_NGRAM_MAX)));

        common_ngram ngram_static;
        if (n_available >= LLAMA_NGRAM_STATIC) {
            const size_t ngram_start_static = n_available - LLAMA_NGRAM_STATIC;
            for (size_t j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
                ngram_static.tokens[j-ngram_start_static] = get_token(inp, draft, j);
            }
        }
        static const common_ngram_cache_part part_static_empty;
        common_ngram_cache::const_iterator part_static_it = n_available >= LLAMA_NGRAM_STATIC
                ? nc_static.find(ngram_static)
                : nc_static.end();
        const common_ngram_cache_part & part_static = part_static_it != nc_static.end()
                ? part_static_it->second
                : part_static_empty;

        // cd = context + dynamic
        std::vector<std::pair<int, common_ngram>> ngrams_cd;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max_available; ++ngram_size_cd) {
            const size_t ngram_start_cd = n_available - ngram_size_cd;
            common_ngram ngram_cd;
            for (size_t j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                ngram_cd.tokens[j-ngram_start_cd] = get_token(inp, draft, j);
            }
            ngrams_cd.emplace_back(ngram_size_cd, ngram_cd);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams_cd, part_static, draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams_cd, part_static, draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == LLAMA_TOKEN_NULL && n_available >= LLAMA_NGRAM_STATIC) {
            drafted_token = try_draft(nc_static, ngram_static);
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }

        LOG_DBG(" - draft candidate: token=%d\n", drafted_token);
        draft.push_back(drafted_token);
    }
}

void common_ngram_cache_save(common_ngram_cache & ngram_cache, const std::string & filename) {
    std::ofstream file_out(filename, std::ios::binary);
    for (std::pair<common_ngram, common_ngram_cache_part> item : ngram_cache) {
        const common_ngram      ngram        = item.first;
        common_ngram_cache_part token_counts = item.second;
        GGML_ASSERT(!token_counts.empty());
        const int32_t ntokens = token_counts.size();
        GGML_ASSERT(ntokens > 0);

        file_out.write(reinterpret_cast<const char *>(&ngram),   sizeof(common_ngram));
        file_out.write(reinterpret_cast<const char *>(&ntokens), sizeof(int32_t));
        for (std::pair<llama_token, int32_t> item2 : token_counts) {
            const llama_token token = item2.first;
            const int32_t     count = item2.second;
            GGML_ASSERT(count > 0);

            file_out.write(reinterpret_cast<const char *>(&token), sizeof(llama_token));
            file_out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }
}

common_ngram_cache common_ngram_cache_load(const std::string & filename) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }
    common_ngram_cache ngram_cache;

    common_ngram ngram;
    int32_t     ntokens;
    llama_token token;
    int32_t     count;

    char * ngramc   = reinterpret_cast<char*>(&ngram);
    char * ntokensc = reinterpret_cast<char*>(&ntokens);
    char * tokenc   = reinterpret_cast<char*>(&token);
    char * countc   = reinterpret_cast<char*>(&count);
    while(hashmap_file.read(ngramc, sizeof(common_ngram))) {
        GGML_ASSERT(!hashmap_file.eof());
        GGML_ASSERT(hashmap_file.read(ntokensc, sizeof(int32_t)));
        GGML_ASSERT(ntokens > 0);
        common_ngram_cache_part::container_type token_counts;

        for (int i = 0; i < ntokens; ++i) {
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(tokenc, sizeof(llama_token)));
            GGML_ASSERT(!hashmap_file.eof());
            GGML_ASSERT(hashmap_file.read(countc, sizeof(int32_t)));
            GGML_ASSERT(count > 0);
            token_counts.emplace_back(token, count);
        }

        ngram_cache.emplace(ngram, common_ngram_cache_part::from_unsorted(std::move(token_counts)));
    }
    GGML_ASSERT(hashmap_file.eof());

    return ngram_cache;
}

void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, common_ngram_cache & ngram_cache_add) {
    for (std::pair<common_ngram, common_ngram_cache_part> ngram_part : ngram_cache_add) {
        const common_ngram      ngram = ngram_part.first;
        common_ngram_cache_part  part = ngram_part.second;

        common_ngram_cache::iterator part_merged_it = ngram_cache_target.find(ngram);
        if (part_merged_it == ngram_cache_target.end()) {
            ngram_cache_target.emplace(ngram, part);
            continue;
        }

        for (std::pair<llama_token, int32_t> token_count : part) {
            const llama_token token = token_count.first;
            const int32_t     count = token_count.second;
            GGML_ASSERT(count > 0);

            common_ngram_cache_part::iterator token_count_merged_it = part_merged_it->second.find(token);
            if (token_count_merged_it == part_merged_it->second.end()) {
                part_merged_it->second.emplace(token, count);
                continue;
            }

            token_count_merged_it->second += count;
        }
    }
}
