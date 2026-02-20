//
// Copyright (C) 2024 Iwan Kawrakow
// Copyright (C) 2023-2024 The ggml authors
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <optional>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s \\\n"
            "       -m model.gguf -f some-text.txt [-o imatrix.dat] [--process-output] [--verbosity 1] \\\n"
            "       [--no-ppl] [--chunk 123] [--output-frequency 10] [--save-frequency 0] \\\n"
            "       [--in-file imatrix-prev-0.dat --in-file imatrix-prev-1.dat ...]\n" , argv[0]);
    LOG_TEE("\n");
}

struct Stats {
    std::vector<float> values;
    std::vector<int> counts;
    int ncall = 0;
    int n_as = 1;
};

class IMatrixCollector {
public:
    IMatrixCollector() = default;
    void set_params(gpt_params params) { m_params = std::move(params); }
    bool collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data);
    void save_imatrix(int ncall = -1) const;
    bool load_imatrix(const char * file_name);
    void set_collect_lsim(bool yes_or_no) { m_collect_lsim = yes_or_no; }
    void print_layer_importance();
private:
    std::unordered_map<std::string, Stats> m_stats;
    gpt_params                             m_params;
    std::mutex                             m_mutex;
    int                                    m_last_call = 0;
    int                                    m_last_layer = 9999;
    int                                    m_last_ffn = -1;
    std::vector<char>                      m_src1_data;
    std::vector<char>                      m_ids; // the expert ids from ggml_mul_mat_id
    std::vector<float>                     m_last_input;
    std::vector<float>                     m_ffn_input;
    std::vector<std::pair<double,int>>     m_layer_sim;
    std::vector<std::pair<double,int>>     m_attn_sim;
    std::vector<std::pair<double,int>>     m_ffn_sim;
    bool                                   m_collect_lsim = false;

    std::optional<int> layer_index(const std::string& name) const {
        if (name == m_params.output_tensor_name && m_last_layer < 199) {
            return m_last_layer + 1;
        }
        if (auto pos = name.find("blk."); pos == 0) {
            pos += 4;
            if (auto pos1 = name.find('.', pos); pos1 != std::string::npos) {
                auto index_str = name.substr(pos, pos1 - pos);
                std::istringstream str(index_str);
                int index; str >> index;
                if (!str.fail()) return index;
            }
        }
        return std::nullopt;
    }

    static inline double cosine_similarity(int n, const float * x, const float * y) {
        double sumxy = 0, sumx2 = 0, sumy2 = 0;
        for (int j = 0; j < n; ++j) {
            sumxy += x[j]*y[j]; sumx2 += x[j]*x[j]; sumy2 += y[j]*y[j];
        }
        double cos_sim = sumx2 > 0 && sumy2 > 0 ? sumxy/sqrt(sumx2*sumy2) : 0;
        return cos_sim;
    }

    static inline void collect_cos_similarity(int nrow, int n, const float * x, const float * y, std::pair<double, int>& p) {
        for (int row = 0; row < nrow; ++row) {
            p.first  += cosine_similarity(n, x, y);
            p.second += 1;
            x += n;
            y += n;
        }
    }

    static void print_layer_importance(const char * msg, const std::vector<std::pair<double, int>>& sim);
};

// remove any prefix and suffixes from the name
// CUDA0#blk.0.attn_k.weight#0 => blk.0.attn_k.weight
static std::string filter_tensor_name(const char * name) {
    std::string wname;
    const char * p = strchr(name, '#');
    if (p != NULL) {
        p = p + 1;
        const char * q = strchr(p, '#');
        if (q != NULL) {
            wname = std::string(p, q - p);
        } else {
            wname = p;
        }
    } else {
        wname = name;
    }
    return wname;
}

void IMatrixCollector::print_layer_importance(const char * msg, const std::vector<std::pair<double, int>>& sim) {
    if (sim.empty()) return;
    std::vector<std::pair<float, int>> layers;
    layers.reserve(sim.size());
    for (int i = 0; i < int(sim.size()); ++i) {
        if (sim[i].second > 0) layers.emplace_back(float(std::abs(sim[i].first/sim[i].second)), i);
    }
    if (layers.empty()) return;
    std::sort(layers.begin(), layers.end());
    printf("%s\n", msg);
    //printf("======================== sorted layer importances\n");
    int j = 0;
    for (auto& p : layers) {
        int i = p.second;
        printf("%3d: Layer %3d, <cos_sim> = %g\n", j++, i, sim[i].first/sim[i].second);
    }
}

void IMatrixCollector::print_layer_importance() {
    print_layer_importance("\n======================== sorted layer importances", m_layer_sim);
    print_layer_importance("\n======================== sorted attention importances", m_attn_sim);
    print_layer_importance("\n======================== sorted ffn importances", m_ffn_sim);
    //printf("%s: have %d layers\n", __func__, int(m_layer_sim.size()));
    //if (m_layer_sim.empty()) return;
    //std::vector<std::pair<float, int>> layers;
    //layers.reserve(m_layer_sim.size());
    //for (int i = 0; i < int(m_layer_sim.size()); ++i) {
    //    if (m_layer_sim[i].second > 0) layers.emplace_back(float(std::abs(m_layer_sim[i].first/m_layer_sim[i].second)), i);
    //}
    //if (layers.empty()) return;
    //std::sort(layers.begin(), layers.end());
    //printf("======================== sorted layer importances\n");
    //int j = 0;
    //for (auto& p : layers) {
    //    int i = p.second;
    //    printf("%3d: Layer %3d, <cos_sim> = %g\n", j++, i, m_layer_sim[i].first/m_layer_sim[i].second);
    //}
}

bool IMatrixCollector::collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];
    std::string wname = filter_tensor_name(src0->name);

    // when ask is true, the scheduler wants to know if we are interested in data from this tensor
    // if we return true, a follow-up call will be made with ask=false in which we can do the actual collection
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true; // collect all indirect matrix multiplications
        if (t->op != GGML_OP_MUL_MAT) return false;
        // why are small batches ignored (<16 tokens)?
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        //printf("wname = %s\n", wname.c_str());
        if (!(wname.substr(0, 4) == "blk." || ((m_params.process_output || m_collect_lsim) && wname == m_params.output_tensor_name))) return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);

    if (!is_host) {
        auto nbytes = ggml_nbytes(src1);
        m_src1_data.resize(nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, nbytes);
    }

    const float * data = is_host ? (const float *) src1->data : (const float *)m_src1_data.data();

    if (m_collect_lsim) {
        if (wname.find(".ffn_") != std::string::npos) {
            if (auto index = layer_index(wname); index.has_value() && *index == m_last_layer && *index != m_last_ffn) {
                int n = src1->ne[0];
                int nrow = t->op == GGML_OP_MUL_MAT_ID ? src1->ne[2] : src1->ne[1];
                if (t->op == GGML_OP_MUL_MAT_ID) {
                    GGML_ASSERT(src1->ne[1] == 1);
                }
                if (m_ffn_input.empty()) {
                    m_ffn_input.resize(nrow*n);
                } else {
                    if ((int)m_ffn_input.size() != nrow*n) {
                        printf("Oops, inconsistent ffn size\n"); exit(1);
                    }
                }
                std::memcpy(m_ffn_input.data(), data, nrow*n*sizeof(float));
                if (m_ffn_input.size() != m_last_input.size()) {
                    printf("Oops, inconsistent ffn vs last_input size\n"); exit(1);
                }
                if (m_attn_sim.size() < *index + 1) m_attn_sim.resize(*index + 1);
                auto& p = m_attn_sim[*index];
                collect_cos_similarity(nrow, n, m_ffn_input.data(), m_last_input.data(), p);
                m_last_ffn = *index;
            }
        }
    }

    // this has been adapted to the new format of storing merged experts in a single 3d tensor
    // ref: https://github.com/ggerganov/llama.cpp/pull/6387
    if (t->op == GGML_OP_MUL_MAT_ID) {
        //   ids  -> [n_experts_used, n_tokens]
        //   src1 -> [cols, n_expert_used, n_tokens]
        const ggml_tensor * ids = t->src[2];
        const int n_as = src0->ne[2];
        const int n_ids = ids->ne[0];

        // the top-k selected expert ids are stored in the ids tensor
        // for simplicity, always copy ids to host, because it is small
        // take into account that ids is not contiguous!

        GGML_ASSERT(ids->ne[1] == src1->ne[2]);

        m_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, m_ids.data(), 0, ggml_nbytes(ids));

        auto & e = m_stats[wname];

        ++e.ncall;

        if (e.values.empty()) {
            e.values.resize(src1->ne[0]*n_as, 0);
            e.counts.resize(src1->ne[0]*n_as, 0);
            e.n_as = n_as;
        }
        else if (e.values.size() != (size_t)src1->ne[0]*n_as) {
            fprintf(stderr, "Oops: inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0]*n_as);
            exit(1); //GGML_ABORT("fatal error");
        }
        else if (e.n_as != n_as) {
            fprintf(stderr, "Oops: inconsistent n_as for %s (%d vs %d)\n", wname.c_str(), e.n_as, n_as);
        }
        if (m_params.verbosity > 1) {
            printf("%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_call, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[2], (int)src1->type);
        }
        // loop over all possible experts, regardless if they are used or not in the batch
        for (int ex = 0; ex < n_as; ++ex) {
            size_t e_start = ex*src1->ne[0];

            for (int idx = 0; idx < n_ids; ++idx) {
                for (int row = 0; row < (int)src1->ne[2]; ++row) {
                    const int excur = *(const int32_t *) (m_ids.data() + row*ids->nb[1] + idx*ids->nb[0]);

                    GGML_ASSERT(excur >= 0 && excur < n_as); // sanity check

                    if (excur != ex) continue;

                    const int64_t i11 = idx % src1->ne[1];
                    const int64_t i12 = row;
                    const float * x = (const float *)((const char *)data + i11*src1->nb[1] + i12*src1->nb[2]);

                    for (int j = 0; j < (int)src1->ne[0]; ++j) {
                        e.values[e_start + j] += x[j]*x[j];
                        e.counts[e_start + j]++;
                        if (!std::isfinite(e.values[e_start + j])) {
                            fprintf(stderr, "%f detected in %s\n", e.values[e_start + j], wname.c_str());
                            exit(1);
                        }
                    }
                }
            }
            if (e.ncall > m_last_call) {
                m_last_call = e.ncall;
                if (m_last_call % m_params.n_out_freq == 0) {
                    save_imatrix();
                }
                if (m_params.n_save_freq > 0 && m_last_call%m_params.n_save_freq == 0) {
                    save_imatrix(m_last_call);
                }
            }
        }
    } else {
        if (m_collect_lsim) {
            // We only need to do it here and not in the MoE branch above because the first tensor in a layer
            // never is a MoE tensor
            if (auto index = layer_index(wname); index.has_value()) {
                if (*index != m_last_layer) {
                    if (*index > 0) {
                        if (m_last_input.size() != src1->ne[0]*src1->ne[1]) {
                            printf("Oops: different size (%d vs %d). Tensor name was %s, m_last_layer = %d\n",
                                    (int)(src1->ne[0]*src1->ne[1]), (int)m_last_input.size(), src0->name, m_last_layer);
                            exit(1);
                        }
                        if (*index > m_layer_sim.size()) m_layer_sim.resize(*index);
                        auto& p = m_layer_sim[*index - 1];
                        collect_cos_similarity(src1->ne[1], src1->ne[0], m_last_input.data(), (const float *)data, p);
                        if (*index == m_last_ffn + 1) {
                            if (*index > m_ffn_sim.size()) m_ffn_sim.resize(*index);
                            auto& p1 = m_ffn_sim[*index-1];
                            collect_cos_similarity(src1->ne[1], src1->ne[0], m_ffn_input.data(), (const float *)data, p1);
                        }
                    }
                    m_last_layer = *index;
                    if (m_last_input.empty()) {
                        m_last_input.resize(src1->ne[0]*src1->ne[1]);
                    } else {
                        if (m_last_input.size() != src1->ne[0]*src1->ne[1]) {
                            printf("Oops\n"); exit(1);
                        }
                    }
                    //printf("Copying src1 to m_last_input\n");
                    std::memcpy(m_last_input.data(), data, src1->ne[0]*src1->ne[1]*sizeof(float));
                }
            }
        }
        auto & e = m_stats[wname];
        if (e.values.empty()) {
            if (src0->ne[3] > 1) {
                fprintf(stderr, "Unsupported 4D tensor %s\n", wname.c_str());
                exit(1);
            }
            // If we have a 3D tensor as it is the case for the attn_k_b and attn_v_b for DeepSeek MLA models,
            // than we need to compute the imatrix for each head, and not just one imatrx for all heads.
            // Hence, the storage we need is src0->ne[0]*src0->ne[2].
            e.values.resize(src0->ne[0]*src0->ne[2], 0);
            e.counts.resize(src0->ne[0]*src0->ne[2], 0);
        }
        else if (e.values.size() != (size_t)(src0->ne[0]*src0->ne[2])) {
            fprintf(stderr, "Oops: inconsistent size for %s (%d vs %d)\n", wname.c_str(), (int)e.values.size(), (int)src1->ne[0]);
            exit(1); //GGML_ABORT("fatal error");
        }
        ++e.ncall;
        if (m_params.verbosity > 1) {
            printf("%s[%d]: %32s, %s, %5d x %5d, %d\n", __func__, m_last_call, wname.c_str(), ggml_op_name(t->op), (int)src1->ne[0], (int)src1->ne[1], (int)src1->type);
        }
        int rk2 = src1->ne[2]/src0->ne[2];
        for (int i12 = 0; i12 < (int)src1->ne[2]; ++i12) {  // i.e., loop over attention heads for MLA models
            int i02 = i12/rk2;
            auto values = e.values.data() + i02*src0->ne[0];
            auto counts = e.counts.data() + i02*src0->ne[0];
            for (int i11 = 0; i11 < (int)src1->ne[1]; ++i11) {
                const float * x = (const float *)((const char *)data + i11*src1->nb[1] + i12*src1->nb[2]);
                for (int j = 0; j < (int)src1->ne[0]; ++j) {
                    values[j] += x[j]*x[j];
                    counts[j]++;
                    if (!std::isfinite(values[j])) {
                        fprintf(stderr, "%f detected in %s\n", e.values[j], wname.c_str());
                        exit(1);
                    }
                }
            }
        }
        if (e.ncall > m_last_call) {
            m_last_call = e.ncall;
            if (m_last_call % m_params.n_out_freq == 0) {
                save_imatrix();
            }
            if (m_params.n_save_freq > 0 && m_last_call%m_params.n_save_freq == 0) {
                save_imatrix(m_last_call);
            }
        }
    }

    return true;
}

void IMatrixCollector::save_imatrix(int ncall) const {
    auto fname = m_params.out_file;
    if (fname.empty()) {
        fname = "imatrix.dat";
    }

    if (ncall > 0) {
        fname += ".at_";
        fname += std::to_string(ncall);
    }

    // avoid writing imatrix entries that do not have full data
    // this can happen with MoE models where some of the experts end up not being exercised by the provided training data

    int n_entries = 0;
    std::vector<std::string> to_store;

    bool is_first = true; // for printing
    for (const auto & kv : m_stats) {
        const int n_all = kv.second.counts.size();

        if (n_all == 0) {
            continue;
        }

        int n_zeros = 0;
        for (const int c : kv.second.counts) {
            if (c == 0) {
                n_zeros++;
            }
        }

        if (n_zeros != 0 && is_first) {
            fprintf(stderr, "\n");
            is_first = false;
        }

        if (n_zeros == n_all) {
            fprintf(stderr, "%s: entry '%40s' has no data - skipping\n", __func__, kv.first.c_str());
            continue;
        }

        if (n_zeros > 0) {
            fprintf(stderr, "%s: entry '%40s' has partial data (%.2f%%)", __func__, kv.first.c_str(), 100.0f * (n_all - n_zeros) / n_all);
            bool store_it = false;
            if (kv.second.n_as > 1) {
                int n_per_expert = n_all / kv.second.n_as;
                std::vector<int> bad_experts;
                bad_experts.reserve(kv.second.n_as);
                for (int i = 0; i < kv.second.n_as; ++i) {
                    auto counts = kv.second.counts.data() + i*n_per_expert;
                    int nz_i = 0;
                    for (int j = 0; j < n_per_expert; ++j) {
                        if (counts[j] == 0) ++nz_i;
                    }
                    if (nz_i > 0) bad_experts.push_back(i);
                }
                fprintf(stderr, " %d out of %d experts are missing data", int(bad_experts.size()), kv.second.n_as);
                if (bad_experts.size() < round(kv.second.n_as * 0.05)) {
                    fprintf(stderr, " Storing **but be aware**\n");
                    store_it = true;
                    for (auto i : bad_experts) {
                        auto counts = (int *)kv.second.counts.data() + i*n_per_expert;
                        auto values = (float *)kv.second.values.data() + i*n_per_expert;
                        for (int j = 0; j < n_per_expert; ++j) {
                            counts[j] = 1;
                            values[j] = 1;
                        }
                    }
                }
            }
            if (!store_it) {
                fprintf(stderr, " - skipping\n");
                continue;
            }
        }

        n_entries++;
        to_store.push_back(kv.first);
    }

    if (to_store.size() < m_stats.size()) {
        fprintf(stderr, "%s: warning: storing only %zu out of %zu entries\n", __func__, to_store.size(), m_stats.size());
    }

    std::ofstream out(fname, std::ios::binary);
    out.write((const char *) &n_entries, sizeof(n_entries));
    for (const auto & name : to_store) {
        const auto & stat = m_stats.at(name);
        int len = name.size();
        out.write((const char *) &len, sizeof(len));
        out.write(name.c_str(), len);
        out.write((const char *) &stat.ncall, sizeof(stat.ncall));
        int nval = stat.values.size();
        out.write((const char *) &nval, sizeof(nval));
        if (nval > 0) {
            std::vector<float> tmp(nval);
            for (int i = 0; i < nval; i++) {
                tmp[i] = (stat.values[i] / static_cast<float>(stat.counts[i])) * static_cast<float>(stat.ncall);
            }
            out.write((const char*)tmp.data(), nval*sizeof(float));
        }
    }

    // Write the number of call the matrix was computed with
    out.write((const char *) &m_last_call, sizeof(m_last_call));

    // Write the input filename at the end of the file to later on specify it in quantize
    {
        int len = m_params.prompt_file.size();
        out.write((const char *) &len, sizeof(len));
        out.write(m_params.prompt_file.c_str(), len);
    }

    if (m_params.verbosity > 0) {
        fprintf(stderr, "\n%s: stored collected data after %d chunks in %s\n", __func__, m_last_call, fname.c_str());
    }
}

bool IMatrixCollector::load_imatrix(const char * fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in) {
        printf("%s: failed to open %s\n",__func__, fname);
        return false;
    }
    int n_entries;
    in.read((char*)&n_entries, sizeof(n_entries));
    if (in.fail() || n_entries < 1) {
        printf("%s: no data in file %s\n", __func__, fname);
        return false;
    }
    for (int i = 0; i < n_entries; ++i) {
        int len; in.read((char *)&len, sizeof(len));
        std::vector<char> name_as_vec(len+1);
        in.read((char *)name_as_vec.data(), len);
        if (in.fail()) {
            printf("%s: failed reading name for entry %d from %s\n",__func__,i+1, fname);
            return false;
        }
        name_as_vec[len] = 0;
        std::string name{name_as_vec.data()};
        auto & e = m_stats[std::move(name)];
        int ncall;
        in.read((char*)&ncall, sizeof(ncall));
        int nval;
        in.read((char *)&nval, sizeof(nval));
        if (in.fail() || nval < 1) {
            printf("%s: failed reading number of values for entry %d\n",__func__,i);
            m_stats = {};
            return false;
        }

        if (e.values.empty()) {
            e.values.resize(nval, 0);
            e.counts.resize(nval, 0);
        }

        std::vector<float> tmp(nval);
        in.read((char*)tmp.data(), nval*sizeof(float));
        if (in.fail()) {
            printf("%s: failed reading data for entry %d\n",__func__,i);
            m_stats = {};
            return false;
        }

        // Recreate the state as expected by save_imatrix(), and corerct for weighted sum.
        for (int i = 0; i < nval; i++) {
            e.values[i] += tmp[i];
            e.counts[i] += ncall;
        }
        e.ncall += ncall;

    }
    return true;
}

static IMatrixCollector g_collector;

static bool ik_collect_imatrix(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_imatrix(t, ask, user_data);
}


struct results_log_softmax {
    double log_softmax;
    float  logit;
    float  prob;
};

static std::vector<float> softmax(const std::vector<float> & logits) {
    std::vector<float> probs(logits.size());
    float max_logit = logits[0];
    for (float v : logits) {
        max_logit = std::max(max_logit, v);
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); i++) {
        // Subtract the maximum logit value from the current logit value for numerical stability
        const float logit = logits[i] - max_logit;
        const float exp_logit = expf(logit);
        sum_exp += exp_logit;
        probs[i] = exp_logit;
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum_exp;
    }
    return probs;
}

static results_log_softmax log_softmax(int n_vocab, const float * logits, int tok) {
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    return {logits[tok] - max_logit - log(sum_exp), logits[tok], expf(logits[tok] - max_logit) / (float) sum_exp};
}

static void process_logits(
    int n_vocab, const float * logits, const int * tokens, int n_token, std::vector<std::thread> & workers,
    double & nll, double & nll2, float * logit_history, float * prob_history) {
    std::mutex mutex;
    int counter = 0;
    auto compute = [&mutex, &counter, &nll, &nll2, logit_history, prob_history, n_vocab, logits, tokens, n_token] () {
        double local_nll  = 0;
        double local_nll2 = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int i = counter++;
            if (i >= n_token) {
                nll += local_nll; nll2 += local_nll2;
                break;
            }
            lock.unlock();
            const results_log_softmax results = log_softmax(n_vocab, logits + i*n_vocab, tokens[i+1]);
            const double v = -results.log_softmax;
            local_nll += v;
            local_nll2 += v*v;

            logit_history[i] = results.logit;
            prob_history[i]  = results.prob;
        }
    };
    for (auto & w : workers) {
        w = std::thread(compute);
    }
    compute();
    for (auto & w : workers) {
        w.join();
    }
}

static bool compute_imatrix(llama_context * ctx, const gpt_params & params) {
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx));
    GGML_ASSERT(llama_add_eos_token(llama_get_model(ctx)) != 1);
    const int n_ctx = llama_n_ctx(ctx);

    auto tim1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = ::common_tokenize(ctx, params.prompt, true);

    auto tim2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "%s: tokenization took %g ms\n",__func__,1e-3*std::chrono::duration_cast<std::chrono::microseconds>(tim2-tim1).count());

    if (params.i_chunk > 0) {
        if (size_t((params.i_chunk + 2)*n_ctx) >= tokens.size()) {
            fprintf(stderr, "%s: there will be not enough tokens left after removing %d chunks\n", __func__, params.i_chunk);
            return false;
        }
        fprintf(stderr, "%s: removing initial %d chunks (%d tokens)\n", __func__, params.i_chunk, params.i_chunk*n_ctx);
        tokens.erase(tokens.begin(), tokens.begin() + params.i_chunk*n_ctx);
    }

    if (int(tokens.size()) < 2*n_ctx) {
        fprintf(stderr, "%s: you need at least %d tokens for a context of %d tokens\n",__func__,2*n_ctx,
                n_ctx);
        fprintf(stderr, "%s: the data file you provided tokenizes to only %zu tokens\n",__func__,tokens.size());
        return false;
    }

    std::vector<float> logit_history;
    std::vector<float> prob_history;

    if (params.compute_ppl) {
        logit_history.resize(tokens.size());
        prob_history.resize(tokens.size());
    }

    const int n_chunk_max = tokens.size() / n_ctx;

    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    const int n_batch = params.n_batch;

    int count = 0;
    double nll = 0.0;
    double nll2 = 0.0;

    fprintf(stderr, "%s: computing over %d chunks with batch_size %d\n", __func__, n_chunk, n_batch);

    std::vector<std::thread> workers(std::thread::hardware_concurrency() - 1);

    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    std::vector<float> logits;
    if (params.compute_ppl && num_batches > 1) {
        logits.reserve((size_t)n_ctx * n_vocab);
    }

    for (int i = 0; i < n_chunk; ++i) {
        const int start =     i * n_ctx;
        const int end   = start + n_ctx;

        std::vector<float> logits;

        const auto t_start = std::chrono::high_resolution_clock::now();

        // clear the KV cache
        llama_kv_cache_clear(ctx);

        for (int j = 0; j < num_batches; ++j) {
            const int batch_start = start + j * n_batch;
            const int batch_size  = std::min(end - batch_start, n_batch);

            // save original token and restore it after eval
            const auto token_org = tokens[batch_start];

            // add BOS token for the first batch of each chunk
            if (add_bos && j == 0) {
                tokens[batch_start] = llama_token_bos(llama_get_model(ctx));
            }

            // TODO: use batch.logits to save computations instead of relying on logits_all == true
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + batch_start, batch_size, j * n_batch, 0))) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return false;
            }

            // restore the original token in case it was set to BOS
            tokens[batch_start] = token_org;

            if (params.compute_ppl && num_batches > 1) {
                const auto * batch_logits = llama_get_logits(ctx);
                logits.insert(logits.end(), batch_logits, batch_logits + batch_size * n_vocab);
            }
        }

        const auto t_end = std::chrono::high_resolution_clock::now();

        if (i == 0) {
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            fprintf(stderr, "%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60*60) {
                fprintf(stderr, "%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            fprintf(stderr, "%.2f minutes\n", total_seconds / 60.0);
        }

        if (params.compute_ppl) {
            const int first = n_ctx/2;
            const auto all_logits = num_batches > 1 ? logits.data() : llama_get_logits(ctx);
            process_logits(n_vocab, all_logits + first*n_vocab, tokens.data() + start + first, n_ctx - 1 - first,
                    workers, nll, nll2, logit_history.data() + start + first, prob_history.data() + start + first);
            count += n_ctx - first - 1;

            printf("[%d]%.4lf,", i + 1, std::exp(nll / count));
            fflush(stdout);

            logits.clear();
        }
    }
    printf("\n");

    if (params.compute_ppl) {
        nll2 /= count;
        nll /= count;
        const double ppl = exp(nll);
        nll2 -= nll * nll;
        if (nll2 > 0) {
            nll2 = sqrt(nll2/(count-1));
            printf("Final estimate: PPL = %.4lf +/- %.5lf\n", ppl, nll2*ppl);
        } else {
            printf("Unexpected negative standard deviation of log(prob)\n");
        }
    }

    return true;
}

int main(int argc, char ** argv) {
    gpt_params params;

    params.n_ctx = 512;
    params.logits_all = true;
    params.verbosity = 1;

    bool lsim = false;
    //
    // Do not pollute common with totally imatrix specific arguments as it was done in mainline.
    // Instead, parse imatrix specific args here, push unknown args into a new array of args,
    // and pass that to gpt_params_parse().
    //
    std::vector<char*> args;
    args.reserve(argc);
    args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg{argv[i]};
        if (arg == "-lsim" || arg == "--layer-similarity") {
            lsim = true;
        } else {
            args.push_back(argv[i]);
        }
    }

    if (!gpt_params_parse(args.size(), args.data(), params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    params.n_batch = std::min(params.n_batch, params.n_ctx);

    g_collector.set_params(params);
    g_collector.set_collect_lsim(lsim);

    for (const auto & in_file : params.in_files) {
        printf("%s : loading imatrix from '%s'\n", __func__, in_file.c_str());
        if (!g_collector.load_imatrix(in_file.c_str())) {
            fprintf(stderr, "%s : failed to load %s\n", __func__, in_file.c_str());
            return 1;
        }
    }

    if (params.in_files.size() > 1) {
        printf("%s : saving combined imatrix to '%s'\n", __func__, params.out_file.c_str());
        g_collector.save_imatrix();
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = ik_collect_imatrix;
    params.cb_eval_user_data = NULL;
    params.warmup = false;

    // init
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    if (params.n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, params.n_ctx);
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    if (!compute_imatrix(ctx, params)) {
        return 1;
    }

    g_collector.save_imatrix();
    g_collector.print_layer_importance();

    llama_print_timings(ctx);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
