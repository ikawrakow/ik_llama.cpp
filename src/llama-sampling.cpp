#include "llama-sampling.h"
#include "llama-vocab.h"
#include "llama-grammar.h"

#include "iqk/iqk_cpu_ops.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <cfloat>
#include <numeric>
#include <unordered_map>

static void llama_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}

void llama_set_rng_seed_impl(struct llama_sampling * smpl, uint32_t seed) {
    if (seed == LLAMA_DEFAULT_SEED) {
        seed = time(NULL);
    }

    smpl->rng.seed(seed);
}

static void llama_sort(llama_token_data_array * candidates, int32_t k) {
    if (candidates->sorted || candidates->size < 2) {
        return;
    }
    if (k < 0) {
        k = candidates->size;
    }
    auto comp = [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit > b.logit;
    };
    if (k <= 1024) { //128) {
        if (k == int(candidates->size)) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
    } else {
        constexpr int   nbuckets     = 128;
        constexpr float bucket_low   = -10.0f;
        constexpr float bucket_high  =  10.0f;
        constexpr float bucket_scale = nbuckets/(bucket_high - bucket_low);
        constexpr float bucker_inter = -bucket_low * bucket_scale;

        std::vector<int> bucket_idx(candidates->size);
        std::vector<int> histo(nbuckets, 0);

        for (int i = 0; i < (int)candidates->size; ++i) {
            const float val = candidates->data[i].logit;
            int ib = int(bucket_scale * val + bucker_inter); //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
            ib = std::max(0, std::min(nbuckets-1, ib));
            bucket_idx[i] = ib;
            ++histo[ib];
        }
        int nhave = 0;
        int ib = nbuckets - 1;
        for ( ; ib >= 0; --ib) {
            nhave += histo[ib];
            if (nhave >= k) break;
        }
        std::vector<llama_token_data> tmp_tokens(nhave);
        auto ptr = tmp_tokens.data();
        std::vector<llama_token_data*> bucket_ptrs;
        bucket_ptrs.reserve(nbuckets - ib);
        for (int j = nbuckets - 1; j >= ib; --j) {
            bucket_ptrs.push_back(ptr);
            ptr += histo[j];
        }
        for (int i = 0; i < (int)candidates->size; ++i) {
            int j = bucket_idx[i];
            if (j >= ib) {
                *bucket_ptrs[nbuckets-1-j]++ = candidates->data[i];
            }
        }

        ptr = tmp_tokens.data();
        int ndone = 0;
        for (int j = nbuckets-1; j > ib; --j) {
            std::sort(ptr, ptr + histo[j], comp);
            ptr += histo[j];
            ndone += histo[j];
        }
        std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);

        std::memcpy(candidates->data, tmp_tokens.data(), k*sizeof(llama_token_data));

    }
    candidates->sorted = true;
}

void llama_sample_softmax_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    GGML_ASSERT(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order if necessary
    llama_sort(candidates, -1);

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_k_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, int32_t k, size_t min_keep) {

    const int64_t t_start_sample_us = ggml_time_us();

    if (k <= 0) {
        k = candidates->size;
    }

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    llama_sort(candidates, k);

    candidates->size = k;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_top_p_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    llama_sample_softmax_impl(smpl, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is at least p or if we have kept at least min_keep tokens
        // we set the last index to i+1 to indicate that the current iterate should be included in the set
        if (cum_sum >= p && i + 1 >= min_keep) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_min_p_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    if (p <= 0.0f || !candidates->size) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    bool min_p_applied = false;

    // if the candidates aren't sorted, try the unsorted implementation first
    if (!candidates->sorted) {
        std::vector<llama_token_data> filtered_tokens;

        float max_logit = -FLT_MAX;
        for (size_t i = 0; i < candidates->size; ++i) {
            max_logit = std::max(max_logit, candidates->data[i].logit);
        }
        const float min_logit = max_logit + logf(p); // min logit for p_i >= p * p_max

        for (size_t i = 0; i < candidates->size; ++i) {
            if (candidates->data[i].logit >= min_logit) {
                filtered_tokens.push_back(candidates->data[i]);
            }
        }

        // if we have enough values the operation was a success
        if (filtered_tokens.size() >= min_keep) {
            memcpy(candidates->data, filtered_tokens.data(), filtered_tokens.size()*sizeof(llama_token_data));
            candidates->size = filtered_tokens.size();
            min_p_applied = true;
        }
    }

    // if the candidates are sorted or the unsorted implementation failed, use this implementation
    if (!min_p_applied) {
        // Sort the logits in descending order if needed
        llama_sort(candidates, -1);

        const float min_logit = candidates->data[0].logit + logf(p); // min logit for p_i >= p * p_max
        size_t i = 1; // first token always matches

        for (; i < candidates->size; ++i) {
            if (candidates->data[i].logit < min_logit && i >= min_keep) {
                break; // prob too small
            }
        }

        // Resize the output vector to keep only the matching tokens
        candidates->size = i;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_tail_free_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);
    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = std::abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    {
        const float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);

        if (second_derivatives_sum > 1e-6f) {
            for (float & value : second_derivatives) {
                value /= second_derivatives_sum;
            }
        } else {
            for (float & value : second_derivatives) {
                value = 1.0f / second_derivatives.size();
            }
        }
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_typical_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    // Compute the softmax of logits and calculate entropy
    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    const int64_t t_start_sample_us = ggml_time_us();

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        shifted_scores[i] = fabsf(-logf(candidates->data[i].p) - entropy);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<llama_token_data> new_candidates(last_idx);
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates[i] = candidates->data[idx];
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();
    candidates->sorted = false;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_entropy_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float min_temp, float max_temp, float exponent_val) {
    const int64_t t_start_sample_us = ggml_time_us();

    // no need to do anything if there is only one (or zero) candidates
    if(candidates->size <= 1) {
        return;
    }

    // Calculate maximum possible entropy
    float max_entropy = -logf(1.0f / candidates->size);

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    // Calculate entropy of the softmax probabilities
    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float prob = candidates->data[i].p;
        if (prob > 0.0f) { // Ensure no log(0)
            entropy -= prob * logf(prob);
        }
    }

    // Normalize the entropy (max_entropy cannot be 0 here because we checked candidates->size != 1 above)
    float normalized_entropy = entropy / max_entropy;

    // Map the normalized entropy to the desired temperature range using the power function
    float dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent_val);

#ifdef DEBUG
    LLAMA_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
    LLAMA_LOG_INFO("Entropy: %f\n", entropy);
    LLAMA_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
    LLAMA_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
    LLAMA_LOG_INFO("Exponent: %f\n", exponent_val);
    LLAMA_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
#endif

    // Apply the dynamically calculated temperature scaling
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= dyn_temp;
    }

    // Re-compute softmax probabilities after scaling logits with dynamic temperature
    double max_l_double = candidates->data[0].logit;
    double cum_sum_double = 0.0;
    for (size_t i = 0; i < candidates->size; ++i) {
        double p = exp(candidates->data[i].logit - max_l_double);
        candidates->data[i].p = p; // Store the scaled probability
        cum_sum_double += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum_double; // Re-normalize the probabilities
    }

#ifdef DEBUG
    // Print the updated top 25 probabilities after temperature scaling
    LLAMA_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t i = 0; i < 25 && i < candidates->size; ++i) {
        LLAMA_LOG_INFO("Token %zu: %f%%\n", i + 1, candidates->data[i].p * 100.0f);
    }
#endif

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_temp_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].logit /= temp;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_xtc_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float probability, float threshold, size_t min_keep) {
    if (probability <= 0 || threshold > 0.5f || candidates->size < 2) {
        return;
    }
    GGML_ASSERT(smpl);
    const int64_t t_start_sample_us = ggml_time_us();
    if (probability < 1) {
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        float chance = distribution(smpl->rng);
        if (chance > probability) return;
    }

    llama_sample_softmax_impl(nullptr, candidates);

    int pos_last = 0;

    for (size_t i = 0; i < candidates->size; ++i) {
        if (candidates->data[i].p >= threshold) {
            pos_last = i;
        } else break;
    }

    if (candidates->size - pos_last >= min_keep && pos_last > 0) {
        candidates->data += pos_last;
        candidates->size -= pos_last;
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

}

void llama_sample_top_n_sigma_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float top_n_sigma) {

    if (top_n_sigma <= 0.0f || candidates->size < 4) {
        // top_n_sigma <= 0: disabled
        // candidates->size < 4: no point in applying the transformation for fewer than 4 logits.
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    float max = candidates->data[0].logit;
    float mean = 0;
    size_t count = 0;
    for (int i = 0; i < (int)candidates->size; ++i) {
        // Only count non-negative infinity values
        if (candidates->data[i].logit != -INFINITY) {
            max = std::max(max, candidates->data[i].logit);
            mean += candidates->data[i].logit;
            ++count;
        }
    }
    if (count < 4) {
        return; // again, tandard deviation is not well defined for so few logits (4 is actually pushing it)
    }
    mean /= count;

    float sigma2 = 0;
    for (int i = 0; i < (int)candidates->size; ++i) {
        if (candidates->data[i].logit != -INFINITY) {
            float delta = candidates->data[i].logit - mean;
            sigma2 += delta*delta;
        }
    }
    float sigma = sqrtf(sigma2/count);
    float thresh = max - top_n_sigma*sigma;

    int n_masked = 0;
    for (int i = 0; i < (int)candidates->size; ++i) {
        if (candidates->data[i].logit != -INFINITY && candidates->data[i].logit < thresh) {
            candidates->data[i].logit = -INFINITY;
            ++n_masked;
        }
    }

    // do we really want to compute softmax unconditionally?
    // The following coresponds to mainline implementation with the minor optimization
    // that we only call the relativly expensive softmax if we masked away some tokens.
    if (n_masked > 0 || !candidates->sorted) {
        llama_sample_softmax_impl(nullptr, candidates);
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
    }
}


void llama_sample_repetition_penalties_impl(
        struct llama_sampling * smpl,
       llama_token_data_array * candidates,
            const llama_token * last_tokens,
                       size_t   penalty_last_n,
                       float   penalty_repeat,
                       float   penalty_freq,
                       float   penalty_present) {
    if (penalty_last_n == 0 || (penalty_repeat == 1.0f && penalty_freq == 0.0f && penalty_present == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<llama_token, int> token_count;
    for (size_t i = 0; i < penalty_last_n; ++i) {
        token_count[last_tokens[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        const auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        const int count = token_iter->second;

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty_repeat;
        } else {
            candidates->data[i].logit /= penalty_repeat;
        }

        candidates->data[i].logit -= float(count) * penalty_freq + float(count > 0) * penalty_present;
    }

    candidates->sorted = false;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void llama_sample_apply_guidance_impl(
        struct llama_sampling * smpl,
                        float * logits,
                        float * logits_guidance,
                        float   scale) {
    GGML_ASSERT(smpl);

    const auto t_start_sample_us = ggml_time_us();
    const auto n_vocab = smpl->n_vocab;

    llama_log_softmax(logits, n_vocab);
    llama_log_softmax(logits_guidance, n_vocab);

    for (int i = 0; i < n_vocab; ++i) {
              auto & l = logits[i];
        const auto & g = logits_guidance[i];

        l = scale * (l - g) + g;
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
}

llama_token llama_sample_token_mirostat_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu) {
    GGML_ASSERT(smpl);

    const int32_t n_vocab = float(smpl->n_vocab);

    int64_t t_start_sample_us = ggml_time_us();

    llama_sample_softmax_impl((struct llama_sampling *) nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(n_vocab, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    llama_sample_top_k_impl((struct llama_sampling *) nullptr, candidates, int(k), 1);
    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    llama_token X = llama_sample_token_impl(smpl, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    return X;
}

llama_token llama_sample_token_mirostat_v2_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    llama_sample_softmax_impl(smpl, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    if (candidates->size == 0) {
        candidates->size = 1;
    }

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }

    // Normalize the probabilities of the remaining words
    llama_sample_softmax_impl(smpl, candidates);

    // Sample the next word X from the remaining words
    llama_token X = llama_sample_token_impl(smpl, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const llama_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

llama_token llama_sample_token_greedy_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto * max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const llama_token_data & a, const llama_token_data & b) {
        return a.logit < b.logit;
    });

    llama_token result = max_iter->id;
    if (smpl) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
    }
    return result;
}

llama_token llama_sample_token_with_rng_impl(struct llama_sampling * smpl, llama_token_data_array * candidates, std::mt19937 & rng) {
    GGML_ASSERT(smpl);

    const int64_t t_start_sample_us = ggml_time_us();

    if (candidates->size < 2) {
        smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
        smpl->n_sample++;
        return candidates->data[0].id;
    }

    std::vector<float> probs(candidates->size);
    probs[0] = candidates->data[0].logit;
    float max = probs[0];
    for (int j = 1; j < candidates->size; ++j) {
        probs[j] = candidates->data[j].logit;
        max = std::max(max, probs[j]);
    }

    float sump = 0;
    for (int j = 0; j < candidates->size; ++j) {
        float p = expf(probs[j] - max);
        sump += p;
        probs[j] = sump;
    }
    probs.back() += sump;

    auto p = sump * rng() / rng.max();
    auto iter = std::upper_bound(probs.begin(), probs.end(), p);
    GGML_ASSERT(iter != probs.end());
    auto idx = std::distance(probs.begin(), iter);
    auto id  = candidates->data[idx].id;

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

    return id;
}

llama_token llama_sample_token_impl(struct llama_sampling * smpl, llama_token_data_array * candidates) {
    return llama_sample_token_with_rng_impl(smpl, candidates, smpl->rng);
}


// DRY

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void get_overlapping_token_sequences(const llama_vocab& vocab, const std::string& str, std::unordered_multimap<llama_token, std::vector<llama_token>>& token_sequences, int max_tail_len = -1) {
    for (llama_token token_id = 0; token_id < (llama_token)vocab.n_tokens(); token_id++) {
        auto word = vocab.detokenize( { token_id }, true);
        if (word.find(str) != std::string::npos) {
            token_sequences.emplace(token_id, std::vector<llama_token>());
        }
        else {
            size_t word_len = word.size(), str_len = str.size();
            size_t pos = -1;
            while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
                bool match = true;
                size_t i;
                for (i = 1; i < str_len && i + pos < word_len; ++i) {
                    if (word[pos + i] != str[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    auto tokenization = vocab.tokenize(str.substr(i), false, false);
                    //std::vector<llama_token> tokenization = llama_tokenize_internal(vocab, str.substr(i), false, false);
                    if (max_tail_len >= 0 && tokenization.size() > (size_t)max_tail_len) {
                        tokenization.resize(max_tail_len);
                    }

                    // Ensure we don't already have a duplicate matching tokenization
                    auto its = token_sequences.equal_range(token_id);
                    bool found = false;
                    for (auto it = its.first; it != its.second; ++it) {
                        if (tokenization == it->second) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        token_sequences.emplace(token_id, tokenization);
                    }
                }
            }
        }
    }
}

static const char* llama_sampler_dry_name(const struct llama_sampler* /*smpl*/) {
    return "dry";
}



// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
void llama_sampler_dry_apply(struct llama_sampler_dry* smpl, llama_token_data_array* cur_p) {
    if (smpl->dry_multiplier == 0.0f || smpl->dry_base < 1.0f || smpl->dry_penalty_last_n == 0) {
        return;
    }

    int32_t effective_dry_penalty_last_n = (smpl->dry_penalty_last_n == -1) ? smpl->total_context_size : std::max(smpl->dry_penalty_last_n, 0);
    int last_n_repeat = std::min(std::min((int)smpl->last_tokens.size(), effective_dry_penalty_last_n), smpl->total_context_size);

    if (last_n_repeat <= smpl->dry_allowed_length) {
        return;
    }

    smpl->dry_repeat_count.assign(last_n_repeat, 0);
    smpl->dry_max_token_repeat.clear();

    // Step 1: Look for restart sequences to limit the maximum repetition length.
    // Work backwards through the context looking for any token that begins a restart sequence.
    //
    // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
    // sequences that together comprise a restart sequence. This allows us to quickly check
    // whether each token is the head of a complete sequence. Most restart sequences are actually
    // a single token, and for these the "tail" is an empty vector.
    //
    // If the token is a "head", test all restart sequences that begin with this token
    // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
    // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
    // longest matching sequence (if any) is used to limit the maximum repetition length.
    //
    // Note that in the case case of a short sequence contained in a longer one, this might fail to
    // find the smallest value for `rep_limit`. For example, if 'amniotic' and 'ni' are both used as
    // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
    // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
    //
    // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
    // have already clamped the maximum tail sequence length when generating `restart_sequences`.
    // With clamping, this scan is O(N) in the context length.

    int rep_limit = last_n_repeat;
    for (int i = 0; i < last_n_repeat; ++i) {
        llama_token token = smpl->last_tokens.rat(i);
        auto its = smpl->dry_processed_breakers.equal_range(token);
        if (its.first == smpl->dry_processed_breakers.end()) {
            continue;
        }
        int longest_match = -1;
        for (auto it = its.first; it != its.second; ++it) {
            // Note that (*it) does not contain the head character, so seq_len will be
            // the restart sequence length minus 1.
            // In the common case of a single-token restart sequence, (*it) will be empty
            // and we will trivially match.
            int seq_len = (int)it->second.size();
            if (seq_len > longest_match && seq_len <= (int)i) {
                bool match = true;
                for (int offset = 0; offset < seq_len; ++offset) {
                    // The -1 when indexing `last_tokens` is because we already matched the head.
                    if (it->second[offset] != smpl->last_tokens.rat(i - offset - 1)) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    longest_match = seq_len;
                }
            }
        }
        if (longest_match >= 0) {
            // We found a restart sequence starting `i` tokens from the end and continuing for
            // `longest_match` tokens.
            rep_limit = i - longest_match;
            break;
        }
    }
    if (rep_limit < smpl->dry_allowed_length) {
        return;
    }

    // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
    // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
    // elsewhere in the context. We limit the suffix length to `rep_limit` to respect restart sequences.
    //
    // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
    // https://ivanyu.me/blog/2014/10/15/z-algorithm/
    //
    // The code below is adapted from the public domain implementation by the same author here:
    // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
    //
    // Example:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //                    ^
    //   This `3` means that the last three tokens of the context (a b c) also appear here.
    //
    // This step is worst case O(N) since the Z-algorithm is linear, despite the appearance of nested
    // for/while loops. This can be seen by observing that the `lt` and `rt` bounds are set after each
    // repeated suffix is detected (i.e. after each while loop when n > 0). These bound variables
    // ensure that the inner while loops only examine each token in the context once as the outer
    // for loop iterates over the context.

    {
        const int last = last_n_repeat - 1;
        int rt = 0, lt = 0;

        for (int k = 1; k < last_n_repeat; ++k) {
            if (k > rt) {
                // If k is outside the current Z-box, do naive computation.
                int n = 0;
                while (n + k < last_n_repeat && smpl->last_tokens.rat(n) == smpl->last_tokens.rat(n + k)) {
                    ++n;
                }
                smpl->dry_repeat_count[last - k] = std::min(n, rep_limit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            }
            else {
                // If k is inside the current Z-box, consider two cases.

                int p = k - lt; // Pair index.
                int right_part_len = rt - k + 1;

                if (smpl->dry_repeat_count[last - p] < right_part_len) {
                    int n = std::min(smpl->dry_repeat_count[last - p], rep_limit);
                    smpl->dry_repeat_count[last - k] = n;
                }
                else {
                    int i = rt + 1;
                    while (i < last_n_repeat && smpl->last_tokens.rat(i) == smpl->last_tokens.rat(i - k)) {
                        i += 1;
                    }

                    int n = std::min(i - k, rep_limit);
                    smpl->dry_repeat_count[last - k] = n;
                    lt = k;
                    rt = i - 1;
                }
            }
        }
    }

    // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
    // that would be generated by emitting each new token that would extend a sequence.
    //
    // Following the same example as above:
    // Last N tokens: a b c c b c y a b c
    // Repeat counts: 0 0 3 1 0 2 0 0 0 0
    //
    // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
    // c: 3 -> 4 (from `a b c` to `a b c c`)
    // b: 1 -> 2 (from `c` to `c b`)
    // y: 2 -> 3 (from `b c` to `b c y`)

    for (int i = 0; i < last_n_repeat - 1; ++i) {
        int repeat_len = smpl->dry_repeat_count[i];
        if (repeat_len >= smpl->dry_allowed_length) {
            // This token ends a repeat, so the next token would continue one.
            // By convention, the value of `repeat_len` only includes the tokens currently
            // in the context, not the new token that would be added.
            llama_token token = smpl->last_tokens.rat(last_n_repeat - 2 - i);
            // Track the maximum sequence ending in this token.
            const auto& it = smpl->dry_max_token_repeat.find(token);
            if (it == smpl->dry_max_token_repeat.end() || it->second < repeat_len) {
                smpl->dry_max_token_repeat[token] = repeat_len;
            }
        }
    }

    // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

    // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `max_exponent`.
    // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
    const float FLOAT_MAX_LOG = 88.7228391f;
    int max_exponent = 0;
    if (smpl->dry_base > 1.000001f) {
        max_exponent = FLOAT_MAX_LOG / std::log(smpl->dry_base);
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
        const auto& af_kvp = smpl->dry_max_token_repeat.find(cur_p->data[i].id);
        if (af_kvp != smpl->dry_max_token_repeat.end()) {
            // Check all sequence breakers starting with this token
            auto range = smpl->dry_processed_breakers.equal_range(cur_p->data[i].id);
            bool is_single_token_breaker = false;

            for (auto it = range.first; it != range.second; ++it) {
                if (it->second.empty()) {
                    is_single_token_breaker = true;
                    break;
                }
            }

            // Apply penalty only if it's not a single-token sequence breaker
            if (!is_single_token_breaker) {
                int repeat_exp = af_kvp->second - smpl->dry_allowed_length;
                if (max_exponent > 0 && repeat_exp > max_exponent) {
                    repeat_exp = max_exponent;
                }
                float penalty = smpl->dry_multiplier * std::pow(smpl->dry_base, repeat_exp);
                cur_p->data[i].logit -= penalty;
            }
        }
    }

    cur_p->sorted = false;
}



struct llama_sampler_dry* llama_sampler_init_dry_impl(const struct llama_vocab& vocab, int32_t context_size, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers) {
    int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1) ? context_size : std::max(dry_penalty_last_n, 0);
    std::unordered_multimap<llama_token, std::vector<llama_token>> processed_breakers;
    const int MAX_CHAR_LEN = 40;
    const int MAX_SEQ_LEN = 20;

    const bool dry_enabled = (dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0);

    if (dry_enabled && seq_breakers != nullptr && num_breakers > 0) {
        // Process sequence breakers
        for (size_t i = 0; i < num_breakers; ++i) {
            if (seq_breakers[i] == nullptr || std::strlen(seq_breakers[i]) == 0) {
                LLAMA_LOG_WARN("skipping null or empty DRY sequence breaker at index %zu\n", i);
                continue;
            }

            std::string sequence_break(seq_breakers[i]);
            if (sequence_break.empty()) {
                LLAMA_LOG_WARN("skipping empty DRY sequence breaker\n");
                continue;
            }

            if (sequence_break.size() > MAX_CHAR_LEN) {
                LLAMA_LOG_WARN("truncating DRY sequence breaker to %d characters\n", MAX_CHAR_LEN);
                sequence_break.resize(MAX_CHAR_LEN);
            }

            get_overlapping_token_sequences(vocab, sequence_break, processed_breakers, MAX_SEQ_LEN);
        }
    }

    return  new llama_sampler_dry {
            /* .total_context_size     = */ context_size,
            /* .dry_multiplier         = */ dry_multiplier,
            /* .dry_base               = */ dry_base,
            /* .dry_allowed_length     = */ dry_allowed_length,
            /* .dry_penalty_last_n     = */ dry_penalty_last_n,
            /* .dry_processed_breakers = */ std::move(processed_breakers),
            /* .dry_repeat_count       = */ dry_enabled ? std::vector<int>(effective_dry_penalty_last_n, 0) : std::vector<int>{},
            /* .dry_max_token_repeat   = */ {},
            /* .last_tokens            = */ dry_enabled ? ring_buffer<llama_token>(effective_dry_penalty_last_n) : ring_buffer<llama_token>(0),
    };
}


// adaptive p

llama_token llama_sample_token_adaptive_p_impl(
              struct llama_sampling * smpl,
             llama_token_data_array * candidates,
    struct llama_sampler_adaptive_p * adapt_p_ctx) {
    GGML_ASSERT(candidates->size > 0);
    const int64_t t_start_sample_us = ggml_time_us();

    struct llama_sampler_adaptive_p * ctx = adapt_p_ctx;
    ctx->cum_probs.resize(candidates->size);

    // compute cumulative probability distribution
    const float max_logit = ctx->max_xform_logit;
    float cum_prob = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        cum_prob += expf(candidates->data[i].logit - max_logit);
        ctx->cum_probs[i] = cum_prob;
    }
    ctx->cum_probs.back() += 1.0f;  // safety margin in case rng() ~= rng.max()

    // select first token whose cum_prob > target_cum_prob
    const float target_cum_prob = cum_prob * (float)ctx->rng() / (float)ctx->rng.max();
    auto iter = std::upper_bound(ctx->cum_probs.begin(), ctx->cum_probs.end(), target_cum_prob);
    GGML_ASSERT(iter != ctx->cum_probs.end());
    const size_t idx = std::distance(ctx->cum_probs.begin(), iter);
    llama_token id = candidates->data[idx].id;

    GGML_ASSERT(id < int(ctx->orig_prob.size()));
    if (auto update_prob = ctx->orig_prob[id]; update_prob > 0) {
        ctx->weighted_sum = ctx->decay * ctx->weighted_sum + update_prob;
        ctx->total_weight = ctx->decay * ctx->total_weight + 1.0f;
    }

    smpl->t_sample_us += ggml_time_us() - t_start_sample_us;
    smpl->n_sample++;

    return id;
}

void llama_sample_adaptive_p_impl(struct llama_sampling * ctx, llama_token_data_array * candidates,
        struct llama_sampler_adaptive_p * adapt_p_ctx) {
    if (adapt_p_ctx->target < 0.0f) {
        // sampler is disabled
        llama_sample_softmax_impl(nullptr, candidates);
        return;
    }

    auto t_start = ggml_time_us();

    // incomplete softmax because final division can be fused
    float max_l = candidates->data[0].logit;
    if (!candidates->sorted) {
        for (size_t i = 1; i < candidates->size; ++i) {
            max_l = std::max(max_l, candidates->data[i].logit);
        }
    }
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        const float prob = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = prob;
        cum_sum += prob;
    }

    // compute adapted target probability
    const float target = std::clamp(adapt_p_ctx->target, 0.0f, 1.0f);
    const float adapted_target = std::clamp(adapt_p_ctx->total_weight == 0.0f
        ? target
        : 2.0f * target - (adapt_p_ctx->weighted_sum / adapt_p_ctx->total_weight),
        0.0f, 1.0f);

    // transformation constants
    static constexpr float peak_logit_value = 5.0f;
    static constexpr float inv_width = 1.0f / 0.3f;
    static constexpr float sharpness = 10.0f;

    const float fused_target = adapted_target * inv_width;
    const float fused_width = inv_width / cum_sum;

    // quadratic near target for finite differentiation, transitioning to linear decay in tails
    // unbounded negative logits suppress far-from-target tokens after softmax
    float max_logit = -INFINITY;
    for (size_t i = 0; i < candidates->size; ++i) {
        const float dist = std::abs(candidates->data[i].p * fused_width - fused_target);
        const float logit = peak_logit_value - sharpness * dist * dist / (1.0f + dist);
        candidates->data[i].logit = logit;
        max_logit = std::max(max_logit, logit);
    }
    candidates->sorted = false;
    adapt_p_ctx->max_xform_logit = max_logit;

    ctx->t_sample_us += ggml_time_us() - t_start;
}

void llama_prep_adaptive_p_impl(
              struct llama_sampling * smpl,
             llama_token_data_array * candidates,
    struct llama_sampler_adaptive_p * adapt_p_ctx) {
    constexpr float kDelta = 30.0f; //16.6f;
    auto t_start = ggml_time_us();
    auto & orig_prob = adapt_p_ctx->orig_prob;
    if (candidates->size != orig_prob.size() || candidates->sorted) {
        LLAMA_LOG_ERROR("%s: this function must be called before any other sampler has been applied\n", __func__);
        LLAMA_LOG_ERROR("%s: the sampler has been initialized with a vocabulary of %zu, but is being called with %zu candidates\n",
                __func__, orig_prob.size(), candidates->size);
        GGML_ABORT("Bad candidates in adaptive_p sampler");
    }

    float max_logit = -INFINITY;
    for (int j = 0; j < int(candidates->size); ++j) {
        orig_prob[j] = candidates->data[j].logit;
        max_logit = std::max(max_logit, orig_prob[j]);
    }
    adapt_p_ctx->cum_orig_prob = iqk_exp_with_thresh(orig_prob.size(), orig_prob.data(), max_logit, max_logit - kDelta);

    if (smpl) smpl->t_sample_us += ggml_time_us() - t_start;
}

struct llama_sampler_adaptive_p * llama_init_adaptive_p_impl(int n_vocab,
       const float target,
       const float decay,
    const uint32_t seed) {
    GGML_ASSERT(n_vocab > 0);
    const float clamped_decay = std::clamp(decay, 0.0f, 0.99f);
    auto result = new llama_sampler_adaptive_p {
        /* .target          = */ target,
        /* .decay           = */ clamped_decay,
        /* .rng             = */ std::mt19937(seed),
        /* .weighted_sum    = */ target / (1.0f - clamped_decay),
        /* .total_weight    = */ 1.0f / (1.0f - clamped_decay),
        /* .orig_prob       = */ {},
        /* .cum_orig_prob   = */ 0.0f,
        /* .max_xform_logit = */ -INFINITY,
        /* .cum_probs       = */ {},
    };
    result->orig_prob.resize(n_vocab);
    return result;
}

// grammar

struct llama_sampler_grammar {
    const struct llama_vocab* vocab;

    std::string grammar_str;
    std::string grammar_root;

    struct llama_grammar* grammar;
};

static const char* llama_sampler_grammar_name(const struct llama_sampler* /*smpl*/) {
    return "grammar";
}

static void llama_sampler_grammar_accept_impl(struct llama_sampler* smpl, llama_token token) {
    auto* ctx = (llama_sampler_grammar*)smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_accept_token_impl(ctx->grammar,ctx->vocab ,nullptr, token);
    }
}

static void llama_sampler_grammar_apply(struct llama_sampler* smpl, llama_token_data_array* cur_p) {
    auto* ctx = (llama_sampler_grammar*)smpl->ctx;
    if (ctx->grammar) {
        llama_grammar_sample_impl(ctx->grammar, ctx->vocab, nullptr, cur_p);
    }
}

void llama_sampler_reset(struct llama_sampler* smpl) {
    if (smpl->iface->reset) {
        smpl->iface->reset(smpl);
    }
}

// Fwd declare to break reset --> init_impl --> llama_sampler_grammar_i --> reset cycle.
static struct llama_grammar* llama_sampler_init_grammar_impl(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root,
    bool lazy,
    const char** trigger_words,
    size_t num_trigger_words,
    const llama_token* trigger_tokens,
    size_t num_trigger_tokens,
    const char** trigger_patterns,
    size_t num_trigger_patterns);

static void llama_sampler_grammar_reset(struct llama_sampler* smpl) {
    auto* ctx = (llama_sampler_grammar*)smpl->ctx;
    if (!ctx->grammar) {
        return;
    }

    std::vector<const char*>  trigger_patterns_c;
    trigger_patterns_c.reserve(ctx->grammar->trigger_patterns.size());
    for (auto& trigger_pattern : ctx->grammar->trigger_patterns) {
        trigger_patterns_c.push_back(trigger_pattern.pattern.c_str());
    }
    auto* grammar_new = llama_grammar_init_impl(ctx->grammar->vocab, ctx->grammar_str.c_str(), ctx->grammar_root.c_str(),
        ctx->grammar->lazy, trigger_patterns_c.data(), trigger_patterns_c.size(),
        ctx->grammar->trigger_tokens.data(), ctx->grammar->trigger_tokens.size());

    llama_grammar_free_impl(ctx->grammar);
    ctx->grammar = grammar_new;
}

//static struct llama_sampler* llama_sampler_grammar_clone(const struct llama_sampler* smpl) {
//    const auto* ctx = (const llama_sampler_grammar*)smpl->ctx;
//
//    auto* result = llama_sampler_init_grammar_impl(ctx->vocab, nullptr, nullptr, false, nullptr, 0, nullptr, 0);
//
//    // copy the state
//    {
//        auto* result_ctx = (llama_sampler_grammar*)result->ctx;
//
//        if (ctx->grammar) {
//            result_ctx->grammar_str = ctx->grammar_str;
//            result_ctx->grammar_root = ctx->grammar_root;
//
//            result_ctx->grammar = llama_grammar_copy_impl(ctx->grammar);
//        }
//    }
//
//    return result;
//}

static void llama_sampler_grammar_free(struct llama_sampler* smpl) {
    const auto* ctx = (llama_sampler_grammar*)smpl->ctx;

    if (ctx->grammar) {
        llama_grammar_free_impl(ctx->grammar);
    }

    delete ctx;
}

// ?
//static struct llama_sampler_i llama_sampler_grammar_i = {
//    /* .name   = */ llama_sampler_grammar_name,
//    /* .accept = */ llama_sampler_grammar_accept_impl,
//    /* .apply  = */ llama_sampler_grammar_apply,
//    /* .reset  = */ llama_sampler_grammar_reset,
//    /* .clone  = */ NULL,
//    /* .free   = */ llama_sampler_grammar_free,
//};

struct llama_grammar* llama_sampler_init_grammar_impl(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root,
    bool lazy,
    const char** trigger_words,
    size_t num_trigger_words,
    const llama_token* trigger_tokens,
    size_t num_trigger_tokens,
    const char** trigger_patterns,
    size_t num_trigger_patterns) {
    // Huh? this is not used and leaks. auto* ctx = new llama_sampler_grammar;
    struct llama_grammar* grammar;
    if (grammar_str != nullptr && grammar_str[0] != '\0') {
        // TODO: remove trigger_words support.
        if (trigger_words != nullptr && num_trigger_words > 0) {
            GGML_ASSERT(trigger_patterns == nullptr && num_trigger_patterns == 0);
            std::string trigger_pattern("[\\s\\S]*?(");
            for (size_t i = 0; i < num_trigger_words; ++i) {
                static const std::regex special_chars("[.^$|()*+?\\[\\]{}\\\\]");
                if (i > 0) {
                    trigger_pattern += "|";
                }
                trigger_pattern += std::regex_replace(trigger_words[i], special_chars, "\\$0");
            }
            trigger_pattern += ")[\\s\\S]*";
            auto trigger_pattern_c = trigger_pattern.c_str();
            trigger_patterns = &trigger_pattern_c;
            num_trigger_patterns = 1;
        }
        grammar = llama_grammar_init_impl(vocab, grammar_str, grammar_root, lazy, trigger_patterns, num_trigger_patterns, trigger_tokens, num_trigger_tokens);
        if (!grammar) {
            return nullptr;
        }
    } else {
        grammar = nullptr;
    }
    return grammar;
}

struct llama_grammar* llama_sampler_init_grammar(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root, /* lazy= */ false, nullptr, 0, nullptr, 0, nullptr, 0);
}

struct llama_grammar* llama_sampler_init_grammar_lazy(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root,
    const char** trigger_words,
    size_t num_trigger_words,
    const llama_token* trigger_tokens,
    size_t num_trigger_tokens) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root, /* lazy= */ true, trigger_words, num_trigger_words, trigger_tokens, num_trigger_tokens, nullptr, 0);
}

struct llama_grammar* llama_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root,
    const char** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token* trigger_tokens,
    size_t num_trigger_tokens) {
    return llama_sampler_init_grammar_impl(vocab, grammar_str, grammar_root, /* lazy= */ true, nullptr, 0, trigger_tokens, num_trigger_tokens, trigger_patterns, num_trigger_patterns);
}
