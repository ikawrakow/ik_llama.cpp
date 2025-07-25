//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <cmath>

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<struct quant_option> QUANT_OPTIONS = {
    { "Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0,     " 3.56G, +0.2166 ppl @ LLaMA-v1-7B", },
    { "Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1,     " 3.90G, +0.1585 ppl @ LLaMA-v1-7B", },
    { "Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0,     " 4.33G, +0.0683 ppl @ LLaMA-v1-7B", },
    { "Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1,     " 4.70G, +0.0349 ppl @ LLaMA-v1-7B", },
    { "Q6_0",     LLAMA_FTYPE_MOSTLY_Q6_0,     " 6.5 bpw quantization",             },
    { "IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS,  " 2.06 bpw quantization",            },
    { "IQ2_XXS_R4",LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4,"IQ2_XXS repacked",            },
    { "IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS,   " 2.31 bpw quantization",            },
    { "IQ2_XS_R4",LLAMA_FTYPE_MOSTLY_IQ2_XS_R4,"IQ2_XS repacked",            },
    { "IQ2_S",    LLAMA_FTYPE_MOSTLY_IQ2_S,    " 2.5  bpw quantization",            },
    { "IQ2_M",    LLAMA_FTYPE_MOSTLY_IQ2_M,    " 2.7  bpw quantization",            },
    { "IQ2_M_R4", LLAMA_FTYPE_MOSTLY_IQ2_M_R4, " 2.7  bpw quantization",            },
    { "IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S,    " 1.56 bpw quantization",            },
    { "IQ1_S_R4", LLAMA_FTYPE_MOSTLY_IQ1_S_R4, " 1.5 bpw quantization",             },
    { "IQ1_M_R4", LLAMA_FTYPE_MOSTLY_IQ1_M_R4, " 1.75 bpw quantization",            },
    { "IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M,    " 1.75 bpw quantization",            },
    { "IQ1_BN",   LLAMA_FTYPE_MOSTLY_IQ1_BN,   " 1.62 bpw quantization (Bitnet)",   },
    { "IQ2_BN",   LLAMA_FTYPE_MOSTLY_IQ2_BN,   " 2.00 bpw quantization (Bitnet)",   },
    { "IQ2_BN_R4",LLAMA_FTYPE_MOSTLY_IQ2_BN_R4," 2.00 bpw quantization (Bitnet)",   },
    { "Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K,     " 2.63G, +0.6717 ppl @ LLaMA-v1-7B", },
    { "Q2_K_R4",  LLAMA_FTYPE_MOSTLY_Q2_K_R4,  "Q2_K_S repacked", },
    { "Q2_K_S",   LLAMA_FTYPE_MOSTLY_Q2_K_S,   " 2.16G, +9.0634 ppl @ LLaMA-v1-7B", },
    { "IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS,  " 3.06 bpw quantization",            },
    { "IQ3_KT",   LLAMA_FTYPE_MOSTLY_IQ3_KT,   " 3.125 bpw trellis quantization",   },
    { "IQ4_KT",   LLAMA_FTYPE_MOSTLY_IQ4_KT,   " 4.0 bpw trellis quantization",     },
    { "IQ3_XXS_R4",LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4,"IQ3_XXS repacked",            },
    { "IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S,    " 3.44 bpw quantization",            },
    { "IQ3_S_R4", LLAMA_FTYPE_MOSTLY_IQ3_S_R4, "IQ3_S repacked",            },
    { "IQ3_M",    LLAMA_FTYPE_MOSTLY_IQ3_M,    " 3.66 bpw quantization mix",        },
    { "Q3_K",     LLAMA_FTYPE_MOSTLY_Q3_K_M,   "alias for Q3_K_M" },
    { "Q3_K_R4",  LLAMA_FTYPE_MOSTLY_Q3_K_R4,  "Q3_K_S repacked" },
    { "IQ3_XS",   LLAMA_FTYPE_MOSTLY_IQ3_XS,   " 3.3 bpw quantization"   ,          },
    { "Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S,   " 2.75G, +0.5551 ppl @ LLaMA-v1-7B", },
    { "Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M,   " 3.07G, +0.2496 ppl @ LLaMA-v1-7B", },
    { "Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L,   " 3.35G, +0.1764 ppl @ LLaMA-v1-7B", },
    { "IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL,   " 4.50 bpw non-linear quantization", },
    { "IQ4_NL_R4",LLAMA_FTYPE_MOSTLY_IQ4_NL_R4," 4.50 bpw non-linear quantization", },
    { "IQ4_XS_R8",LLAMA_FTYPE_MOSTLY_IQ4_XS_R8," 4.25 bpw non-linear quantization", },
    { "Q4_0_R8",  LLAMA_FTYPE_MOSTLY_Q4_0_R8,  " 4.50 bpw quantization",            },
    { "Q5_0_R4",  LLAMA_FTYPE_MOSTLY_Q5_0_R4,  " 5.50 bpw quantization",            },
    { "Q6_0_R4",  LLAMA_FTYPE_MOSTLY_Q6_0_R4,  " 6.50 bpw quantization",            },
    { "Q8_0_R8",  LLAMA_FTYPE_MOSTLY_Q8_0_R8,  " 8.50 bpw quantization",            },
    { "Q8_KV",    LLAMA_FTYPE_MOSTLY_Q8_KV,    " 8.00 bpw quantization",            },
    { "IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS,   " 4.25 bpw non-linear quantization", },
    { "IQ4_KS",   LLAMA_FTYPE_MOSTLY_IQ4_KS,   " 4.25 bpw non-linear quantization", },
    { "IQ4_KS_R4",LLAMA_FTYPE_MOSTLY_IQ4_KS_R4,"IQ4_KS repacked", },
    { "IQ5_KS_R4",LLAMA_FTYPE_MOSTLY_IQ5_KS_R4,"IQ5_KS repacked", },
    { "IQ4_KSS",  LLAMA_FTYPE_MOSTLY_IQ4_KSS,  " 4.0 bpw non-linear quantization",  },
    { "IQ5_KS",   LLAMA_FTYPE_MOSTLY_IQ5_KS,   " 5.25 bpw non-linear quantization", },
    { "IQ2_K",    LLAMA_FTYPE_MOSTLY_IQ2_K,    " 2.375 bpw non-linear quantization",},
    { "IQ2_K_R4", LLAMA_FTYPE_MOSTLY_IQ2_K_R4, "IQ2_K repacked",},
    { "IQ2_KS",   LLAMA_FTYPE_MOSTLY_IQ2_KS,   " 2.1875 bpw non-linear quantization",},
    { "IQ1_KT",   LLAMA_FTYPE_MOSTLY_IQ1_KT,   " 1.75 bpw trellis quantization",   },
    { "IQ2_KT",   LLAMA_FTYPE_MOSTLY_IQ2_KT,   " 2.125 bpw trellis quantization",   },
    { "IQ2_KL",   LLAMA_FTYPE_MOSTLY_IQ2_KL,   " 2.69 bpw non-linear quantization", },
    { "IQ3_KS",   LLAMA_FTYPE_MOSTLY_IQ3_KS,   " 3.19 bpw non-linear quantization", },
    { "IQ3_K",    LLAMA_FTYPE_MOSTLY_IQ3_K,    " 3.44 bpw non-linear quantization", },
    { "IQ3_K_R4", LLAMA_FTYPE_MOSTLY_IQ3_K_R4, "IQ3_K repacked", },
    { "IQ3_KL",   LLAMA_FTYPE_MOSTLY_IQ3_KL,   " 4 bpw non-linear quantization mix",},
    { "IQ4_K",    LLAMA_FTYPE_MOSTLY_IQ4_K,    " 4.5 bpw non-linear quantization",  },
    { "IQ4_K_R4", LLAMA_FTYPE_MOSTLY_IQ4_K_R4, "IQ4_K repacked",  },
    { "IQ5_K",    LLAMA_FTYPE_MOSTLY_IQ5_K,    " 5.5 bpw non-linear quantization",  },
    { "IQ5_K_R4", LLAMA_FTYPE_MOSTLY_IQ5_K_R4, "IQ5_K repacked",  },
    { "IQ6_K",    LLAMA_FTYPE_MOSTLY_IQ6_K,    " 6.6 bpw non-linear quantization",  },
    { "Q4_K",     LLAMA_FTYPE_MOSTLY_Q4_K_M,   "alias for Q4_K_M", },
    { "Q4_K_R4",  LLAMA_FTYPE_MOSTLY_Q4_K_R4,  "Q4_K_S repacked", },
    { "Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S,   " 3.59G, +0.0992 ppl @ LLaMA-v1-7B", },
    { "Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M,   " 3.80G, +0.0532 ppl @ LLaMA-v1-7B", },
    { "Q5_K",     LLAMA_FTYPE_MOSTLY_Q5_K_M,   "alias for Q5_K_M", },
    { "Q5_K_R4",  LLAMA_FTYPE_MOSTLY_Q5_K_R4,  "Q5_K_S repacked", },
    { "Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S,   " 4.33G, +0.0400 ppl @ LLaMA-v1-7B", },
    { "Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M,   " 4.45G, +0.0122 ppl @ LLaMA-v1-7B", },
    { "Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K,     " 5.15G, +0.0008 ppl @ LLaMA-v1-7B", },
    { "Q6_K_R4",  LLAMA_FTYPE_MOSTLY_Q6_K_R4,  "Q6_K repacked", },
    { "Q8_K_R8",  LLAMA_FTYPE_MOSTLY_Q8_K_R8,  "Q8_K repacked", },
    { "Q8_KV_R8", LLAMA_FTYPE_MOSTLY_Q8_KV_R8, "Q8_KV repacked", },
    { "Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0,     " 6.70G, +0.0004 ppl @ LLaMA-v1-7B", },
    { "Q4_0_4_4", LLAMA_FTYPE_MOSTLY_Q4_0_4_4, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_0_4_8", LLAMA_FTYPE_MOSTLY_Q4_0_4_8, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "Q4_0_8_8", LLAMA_FTYPE_MOSTLY_Q4_0_8_8, " 4.34G, +0.4685 ppl @ Llama-3-8B",  },
    { "F16",      LLAMA_FTYPE_MOSTLY_F16,      "14.00G, -0.0020 ppl @ Mistral-7B", },
    { "BF16",     LLAMA_FTYPE_MOSTLY_BF16,     "14.00G, -0.0050 ppl @ Mistral-7B", },
    { "BF16_R16", LLAMA_FTYPE_MOSTLY_BF16_R16, "14.00G, -0.0050 ppl @ Mistral-7B", },
    { "F32",      LLAMA_FTYPE_ALL_F32,         "26.00G              @ 7B", },
    // Note: Ensure COPY comes after F32 to avoid ftype 0 from matching.
    { "COPY",     LLAMA_FTYPE_ALL_F32,         "only copy tensors, no quantizing",  },
};

static const char * const LLM_KV_QUANTIZE_IMATRIX_FILE       = "quantize.imatrix.file";
static const char * const LLM_KV_QUANTIZE_IMATRIX_DATASET    = "quantize.imatrix.dataset";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES  = "quantize.imatrix.entries_count";
static const char * const LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS   = "quantize.imatrix.chunks_count";

static bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (auto & it : QUANT_OPTIONS) {
        if (it.name == ftype_str) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

// usage:
//  ./llama-quantize [--allow-requantize] [--leave-output-tensor] [--pure] models/llama/ggml-model.gguf [models/llama/ggml-model-quant.gguf] type [nthreads]
//
[[noreturn]]
static void usage(const char * executable) {
    printf("usage: %s [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix] [--hide-imatrix] [--include-weights] [--exclude-weights] [--output-tensor-type] [--token-embedding-type] [--attn-q-type] [--attn-k-type] [--attn-v-type] [--attn-qkv-type] [--attn-output-type] [--ffn-gate-type] [--ffn-down-type] [--ffn-up-type] [--keep-split] [--override-kv] model-f32.gguf [model-quant.gguf] type [nthreads]\n\n", executable);
    printf("  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit\n");
    printf("  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing\n");
    printf("  --pure: Disable k-quant mixtures and quantize all tensors to the same type\n");
    printf("  --imatrix file_name: use data in file_name as importance matrix for quant optimizations\n");
    printf("  --hide-imatrix: do not store imatrix details in the quantized model\n");
    printf("  --include-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --exclude-weights tensor_name: use importance matrix for this/these tensor(s)\n");
    printf("  --output-tensor-type ggml_type: use this ggml_type for the output.weight tensor.\n");
    printf("  --token-embedding-type ggml_type: use this ggml_type for the token_embd.weight tensor.\n\n");
    printf("  --custom-q regex1=type1,regex2=type2...: use this to specify custom quantization type rules.\n\n");
    printf("  --repack Repack all tensors to the corresponding _r4/8 variant if available.\n\n");
    printf("  --repack-pattern Comma separated list of regexs to use for matching tensor names to be repacked.\n\n");
    printf("Additional specific tensor quantization types used in the custom quant scheme 'CQS (default is Q2_K):\n");
    printf("      --attn-q-type ggml_type: use this ggml_type for the attn_q.weight tensor.\n");
    printf("      --attn-k-type ggml_type: use this ggml_type for the attn_k.weight tensor.\n");
    printf("      --attn-v-type ggml_type: use this ggml_type for the attn_v.weight tensor.\n");
    printf("      --attn-qkv-type ggml_type: use this ggml_type for the attn_qkv.weight tensor.\n");
    printf("      --attn-output-type ggml_type: use this ggml_type for the attn_output.weight tensor.\n");
    printf("      --ffn-gate-type ggml_type: use this ggml_type for the ffn_gate tensor.\n");
    printf("      --ffn-down-type ggml_type: use this ggml_type for the ffn_down tensor.\n");
    printf("      --ffn-up-type ggml_type: use this ggml_type for the ffn_up tensor.\n\n");
    printf("  --keep-split: will generate quantized model in the same shards as input\n");
    printf("  --override-kv KEY=TYPE:VALUE\n");
    printf("      Advanced option to override model metadata by key in the quantized model. May be specified multiple times.\n\n");
    printf("Note: --include-weights and --exclude-weights cannot be used together\n");
    printf("Note: The token embeddings tensor is loaded in system RAM, even in case of full GPU/VRAM offload.\n");
    printf("Note: The recommanded type for the output tensor is q6_K for the ffn types > iq3_xxs and < q8_0.\n\n");
    printf("Note for the Custom Quant Scheme FTYPE:\n");
    printf("    Write the specific tensor legacy quants as qN_N, the K-Quants as qN_K, the IQ-Quants as iqN_xx.\n");
    printf("    Usually, attn-q-type can be one type below the chosen ffn type, and attn-v-type should be one type above.\n");
    printf("    attn-qkv-type replaces the types attn-q, attn-k and attn-v on some models.\n");
    //TODO: - eventually - harmonize the CAPS writing of the FTYPEs, and non CAPS writing of the GGML_TYPEs.
    printf("\nAllowed quantization types:\n");
    for (auto & it : QUANT_OPTIONS) {
        if (it.name != "COPY") {
            printf("  %2d  or  ", it.ftype);
        } else {
            printf("          ");
        }
        printf("%-7s : %s\n", it.name.c_str(), it.desc.c_str());
    }
    exit(1);
}

static int load_imatrix(const std::string & imatrix_file, std::string & imatrix_dataset, std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
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

static int prepare_imatrix(const std::string & imatrix_file,
        std::string & imatrix_dataset,
        const std::vector<std::string> & included_weights,
        const std::vector<std::string> & excluded_weights,
        std::unordered_map<std::string, std::vector<float>> & imatrix_data) {
    int m_last_call = -1;
    if (!imatrix_file.empty()) {
        m_last_call = load_imatrix(imatrix_file, imatrix_dataset, imatrix_data);
    }
    if (imatrix_data.empty()) {
        return m_last_call;
    }
    if (!excluded_weights.empty()) {
        for (auto& name : excluded_weights) {
            for (auto it = imatrix_data.begin(); it != imatrix_data.end(); ) {
                auto pos = it->first.find(name);
                if (pos != std::string::npos) it = imatrix_data.erase(it);
                else ++it;
            }
        }
    }
    if (!included_weights.empty()) {
        std::unordered_map<std::string, std::vector<float>> tmp;
        for (auto& name : included_weights) {
            for (auto& e : imatrix_data) {
                auto pos = e.first.find(name);
                if (pos != std::string::npos) {
                    tmp.emplace(std::move(e));
                }
            }
        }
        imatrix_data = std::move(tmp);
    }
    if (!imatrix_data.empty()) {
        printf("%s: have %d importance matrix entries\n", __func__, int(imatrix_data.size()));
    }
    return m_last_call;
}

static ggml_type parse_ggml_type(const char * arg) {
    ggml_type result = GGML_TYPE_COUNT;
    for (int j = 0; j < GGML_TYPE_COUNT; ++j) {
        auto type = ggml_type(j);
        const auto * name = ggml_type_name(type);
        if (name && strcmp(arg, name) == 0) {
            result = type; break;
        }
    }
    return result;
}

using CustomQ = std::pair<std::string, ggml_type>;

static bool parse_custom_quants(const std::string& arg, std::vector<CustomQ>& custom_quants) {
    for (const auto & item : string_split<std::string>(arg, ',')) {
        auto pos = item.find('=');
        if (pos == std::string::npos) {
            fprintf(stderr, "Invalid custom quantization input %s\n", arg.c_str());
            return false;
        }
        auto pattern = item.substr(0, pos);
        auto type_as_string = item.substr(pos + 1);
        auto type = parse_ggml_type(type_as_string.c_str());
        if (type == GGML_TYPE_COUNT) {
            fprintf(stderr, "Invalid quantization type '%s' in custom quantization input %s\n", type_as_string.c_str(), item.c_str());
            return false;
        }
        printf("Adding custom rule %s -> %s\n", pattern.c_str(), ggml_type_name(type));
        custom_quants.emplace_back(std::move(pattern), type);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
    }

    llama_model_quantize_params params = llama_model_quantize_default_params();

    int arg_idx = 1;
    std::string imatrix_file;
    std::vector<std::string> included_weights, excluded_weights;
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<CustomQ> custom_quants;

    std::vector<std::string> repack_patterns;

    bool hide_imatrix = false;

    for (; arg_idx < argc && strncmp(argv[arg_idx], "--", 2) == 0; arg_idx++) {
        if (strcmp(argv[arg_idx], "--leave-output-tensor") == 0) {
            params.quantize_output_tensor = false;
        } else if (strcmp(argv[arg_idx], "--ignore-imatrix-rules") == 0) {
            params.ignore_imatrix_rules = true;
        } else if (strcmp(argv[arg_idx], "--repack") == 0) {
            params.only_repack = true;
        } else if (strcmp(argv[arg_idx], "--repack-pattern") == 0) {
            if (arg_idx < argc-1) {
                auto p = string_split(argv[++arg_idx], ',');
                repack_patterns.insert(repack_patterns.end(), p.begin(), p.end());
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--output-tensor-type") == 0) {
            if (arg_idx < argc-1) {
                params.output_tensor_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--token-embedding-type") == 0) {
            if (arg_idx < argc-1) {
                params.token_embedding_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-q-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_q_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-k-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_k_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-v-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_v_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-qkv-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_qkv_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--attn-output-type") == 0) {
            if (arg_idx < argc-1) {
                params.attn_output_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-gate-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_gate_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-down-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_down_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--ffn-up-type") == 0) {
            if (arg_idx < argc-1) {
                params.ffn_up_type = parse_ggml_type(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--override-kv") == 0) {
            if (arg_idx == argc-1 || !string_parse_kv_override(argv[++arg_idx], kv_overrides)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--custom-q") == 0) {
            if (arg_idx == argc-1 || !parse_custom_quants(argv[++arg_idx], custom_quants)) {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--allow-requantize") == 0) {
            params.allow_requantize = true;
        } else if (strcmp(argv[arg_idx], "--pure") == 0) {
            params.pure = true;
        } else if (strcmp(argv[arg_idx], "--imatrix") == 0) {
            if (arg_idx < argc-1) {
                imatrix_file = argv[++arg_idx];
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--hide-imatrix") == 0) {
            hide_imatrix = true;
        } else if (strcmp(argv[arg_idx], "--include-weights") == 0) {
            if (arg_idx < argc-1) {
                included_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--exclude-weights") == 0) {
            if (arg_idx < argc-1) {
                excluded_weights.emplace_back(argv[++arg_idx]);
            } else {
                usage(argv[0]);
            }
        } else if (strcmp(argv[arg_idx], "--keep-split") == 0) {
            params.keep_split = true;
        } else {
            usage(argv[0]);
        }
    }

    if (!repack_patterns.empty()) {
        params.repack_pattern = &repack_patterns;
    }

    if (argc - arg_idx < 2) {
        printf("%s: bad arguments\n", argv[0]);
        usage(argv[0]);
    }
    if (!included_weights.empty() && !excluded_weights.empty()) {
        usage(argv[0]);
    }

    std::string imatrix_dataset;
    std::unordered_map<std::string, std::vector<float>> imatrix_data;
    int m_last_call = prepare_imatrix(imatrix_file, imatrix_dataset, included_weights, excluded_weights, imatrix_data);
    if (!imatrix_data.empty()) {
        params.imatrix = &imatrix_data;
        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_FILE);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            if (hide_imatrix) {
                strncpy(kvo.val_str, "top_secret", 127);
            } else {
                strncpy(kvo.val_str, imatrix_file.c_str(), 127);
            }
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }
        if (!imatrix_dataset.empty()) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_DATASET);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
            if (hide_imatrix) {
                strncpy(kvo.val_str, "top_secret", 127);
            } else {
                strncpy(kvo.val_str, imatrix_dataset.c_str(), 127);
            }
            kvo.val_str[127] = '\0';
            kv_overrides.emplace_back(std::move(kvo));
        }

        {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_ENTRIES);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            if (hide_imatrix) {
                kvo.val_i64 = 0;
            } else {
                kvo.val_i64 = imatrix_data.size();
            }
            kv_overrides.emplace_back(std::move(kvo));
        }

        if (m_last_call > 0) {
            llama_model_kv_override kvo;
            std::strcpy(kvo.key, LLM_KV_QUANTIZE_IMATRIX_N_CHUNKS);
            kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
            if (hide_imatrix) {
                kvo.val_i64 = 0;
            } else {
                kvo.val_i64 = m_last_call;
            }
            kv_overrides.emplace_back(std::move(kvo));
        }
    }
    if (!kv_overrides.empty()) {
        kv_overrides.emplace_back();
        kv_overrides.back().key[0] = 0;
        params.kv_overrides = &kv_overrides;
    }
    if (!custom_quants.empty()) {
        params.custom_quants = &custom_quants;
    }

    llama_backend_init();

    // parse command line arguments
    const std::string fname_inp = argv[arg_idx];
    arg_idx++;
    std::string fname_out;

    std::string ftype_str;
    std::string suffix = ".gguf";
    if (try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
        std::string fpath;
        const size_t pos = fname_inp.find_last_of("/\\");
        if (pos != std::string::npos) {
            fpath = fname_inp.substr(0, pos + 1);
        }

        // export as [inp path]/ggml-model-[ftype]. Only add extension if there is no splitting
        fname_out = fpath + "ggml-model-" + ftype_str;
        if (!params.keep_split) {
            fname_out += suffix;
        }
        arg_idx++;
        if (ftype_str == "COPY") {
            params.only_copy = true;
        }
    } else {
        fname_out = argv[arg_idx];
        if (params.keep_split && fname_out.find(suffix) != std::string::npos) {
            fname_out = fname_out.substr(0, fname_out.length() - suffix.length());
        }
        arg_idx++;

        if (argc <= arg_idx) {
            fprintf(stderr, "%s: missing ftype\n", __func__);
            return 1;
        }
        if (!try_parse_ftype(argv[arg_idx], params.ftype, ftype_str)) {
            fprintf(stderr, "%s: invalid ftype '%s'\n", __func__, argv[3]);
            return 1;
        }
        if (ftype_str == "COPY") {
           params.only_copy = true;
        }
        arg_idx++;
    }

    // parse nthreads
    if (argc > arg_idx) {
        try {
            params.nthread = std::stoi(argv[arg_idx]);
        }
        catch (const std::exception & e) {
            fprintf(stderr, "%s: invalid nthread '%s' (%s)\n", __func__, argv[arg_idx], e.what());
            return 1;
        }
    }

    if (!params.ignore_imatrix_rules && imatrix_data.empty() &&
        (params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4 ||
         params.ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S || params.ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS_R4 ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_S  ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_S_R4 ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_M_R4 ||
         params.ftype == LLAMA_FTYPE_MOSTLY_IQ1_M)) {
        fprintf(stderr, "\n==========================================================================================================\n");
        fprintf(stderr, "Please do not use IQ1_S, IQ1_M, IQ2_S, IQ2_XXS, IQ2_XS or Q2_K_S quantization without an importance matrix\n");
        fprintf(stderr, "==========================================================================================================\n\n\n");
        return 1;
    }

    print_build_info();

    fprintf(stderr, "%s: quantizing '%s' to '%s' as %s", __func__, fname_inp.c_str(), fname_out.c_str(), ftype_str.c_str());
    if (params.nthread > 0) {
        fprintf(stderr, " using %d threads", params.nthread);
    }
    fprintf(stderr, "\n");

    const int64_t t_main_start_us = llama_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = llama_time_us();

        if (llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = llama_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = llama_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us/1000.0);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0);
    }

    llama_backend_free();

    return 0;
}
