//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-grammar.h"
#include "llama-sampling.h"
#include "llama-arch.h"
#include "llama-mmap.h"
#include "llama-model-loader.h"
#include "llama-model.h"
#include "llama-build-context.h"
#include "llama-cparams.h"
#include "llama-hparams.h"
#include "llama-context.h"

#include "unicode.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// TODO: fix this include
#include "iqk/iqk_quantize.h"

#define IK_PRINT_TIMING 0

#ifdef GGML_USE_RPC
#  include "ggml-rpc.h"
#endif

#ifdef GGML_USE_CUDA
#  include "ggml-cuda.h"
#elif defined(GGML_USE_VULKAN)
#  include "ggml-vulkan.h"
#elif defined(GGML_USE_SYCL)
#  include "ggml-sycl.h"
#elif defined(GGML_USE_KOMPUTE)
#   include "ggml-kompute.h"
#elif defined(GGML_USE_CANN)
#   include "ggml-cann.h"
#endif

#ifdef GGML_USE_BLAS
#  include "ggml-blas.h"
#endif

#ifdef GGML_USE_METAL
#  include "ggml-metal.h"
#endif

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

//#if __cplusplus >= 202000L
//    #define LU8(x) (const char*)(u8##x)
//#else
//    #define LU8(x) u8##x
//#endif
#define LU8(x) (const char*)(u8##x)
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <future>
#include <initializer_list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <unordered_set>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <regex>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// bump if necessary
#define LLAMA_MAX_LAYERS  512

//
// helpers
//


static bool is_utf8_whitespace(uint8_t c) {
    // Basic ASCII whitespace
    if (c <= 0x7F) return isspace(c);
    // Else: Not whitespace (or you'd need a full Unicode table)
    return false;
}

static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && is_utf8_whitespace(str[start])) start++;
    while (end > start && is_utf8_whitespace(str[end - 1])) end--;
    return str.substr(start, end - start);
}


static std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

// extract ip and port from RPC[ip:port] for rpc and keep other device names
static std::vector<rpc_device>  extract_device_from_rpc_device(std::vector<std::string> devices) {
    std::vector<rpc_device> rpc_servers;
    for (auto & device : devices) {
        rpc_device rpc;
        auto value = string_split(device, "|");
        if (value.size() == 2) {
            rpc.device = std::stoi(value[1]);
            rpc.endpoint = value[0];
        }
        rpc_servers.push_back(rpc);
    }
    return rpc_servers;
}


enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
    LLM_CHAT_TEMPLATE_MISTRAL_V1,
    LLM_CHAT_TEMPLATE_MISTRAL_V3,
    LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
    LLM_CHAT_TEMPLATE_MISTRAL_V7,
    LLM_CHAT_TEMPLATE_PHI_3,
    LLM_CHAT_TEMPLATE_FALCON_3,
    LLM_CHAT_TEMPLATE_FALCON_E,
    LLM_CHAT_TEMPLATE_ZEPHYR,
    LLM_CHAT_TEMPLATE_MONARCH,
    LLM_CHAT_TEMPLATE_GEMMA,
    LLM_CHAT_TEMPLATE_ORION,
    LLM_CHAT_TEMPLATE_OPENCHAT,
    LLM_CHAT_TEMPLATE_VICUNA,
    LLM_CHAT_TEMPLATE_VICUNA_ORCA,
    LLM_CHAT_TEMPLATE_DEEPSEEK,
    LLM_CHAT_TEMPLATE_DEEPSEEK_2,
    LLM_CHAT_TEMPLATE_DEEPSEEK_3,
    LLM_CHAT_TEMPLATE_COMMAND_R,
    LLM_CHAT_TEMPLATE_LLAMA_3,
    LLM_CHAT_TEMPLATE_CHATGLM_3,
    LLM_CHAT_TEMPLATE_CHATGLM_4,
    LLM_CHAT_TEMPLATE_MINICPM,
    LLM_CHAT_TEMPLATE_EXAONE_3,
    LLM_CHAT_TEMPLATE_RWKV_WORLD,
    LLM_CHAT_TEMPLATE_GRANITE,
    LLM_CHAT_TEMPLATE_GIGACHAT,
    LLM_CHAT_TEMPLATE_MEGREZ,
    LLM_CHAT_TEMPLATE_LLAMA4,
    LLM_CHAT_TEMPLATE_BITNET,
    LLM_CHAT_TEMPLATE_DOTS1,
    LLM_CHAT_TEMPLATE_HUNYUAN_MOE,
    LLM_CHAT_TEMPLATE_KIMI_K2,
    LLM_CHAT_TEMPLATE_OPENAI_MOE,
    LLM_CHAT_TEMPLATE_GROK_2,
    LLM_CHAT_TEMPLATE_BAILING,
    LLM_CHAT_TEMPLATE_BAILING_THINK,
    LLM_CHAT_TEMPLATE_BAILING2,
    LLM_CHAT_TEMPLATE_UNKNOWN,
};

static const std::map<std::string, llm_chat_template> LLM_CHAT_TEMPLATES = {
    { "chatml",            LLM_CHAT_TEMPLATE_CHATML            },
    { "llama2",            LLM_CHAT_TEMPLATE_LLAMA_2           },
    { "llama2-sys",        LLM_CHAT_TEMPLATE_LLAMA_2_SYS       },
    { "llama2-sys-bos",    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS   },
    { "llama2-sys-strip",  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP },
    { "mistral-v1",        LLM_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        LLM_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        LLM_CHAT_TEMPLATE_MISTRAL_V7        },
    { "phi3",              LLM_CHAT_TEMPLATE_PHI_3             },
    { "falcon3",           LLM_CHAT_TEMPLATE_FALCON_3          },
    { "falcon_e",          LLM_CHAT_TEMPLATE_FALCON_E          },
    { "zephyr",            LLM_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           LLM_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             LLM_CHAT_TEMPLATE_GEMMA             },
    { "orion",             LLM_CHAT_TEMPLATE_ORION             },
    { "openchat",          LLM_CHAT_TEMPLATE_OPENCHAT          },
    { "vicuna",            LLM_CHAT_TEMPLATE_VICUNA            },
    { "vicuna-orca",       LLM_CHAT_TEMPLATE_VICUNA_ORCA       },
    { "deepseek",          LLM_CHAT_TEMPLATE_DEEPSEEK          },
    { "deepseek2",         LLM_CHAT_TEMPLATE_DEEPSEEK_2        },
    { "deepseek3",         LLM_CHAT_TEMPLATE_DEEPSEEK_3        },
    { "command-r",         LLM_CHAT_TEMPLATE_COMMAND_R         },
    { "llama3",            LLM_CHAT_TEMPLATE_LLAMA_3           },
    { "chatglm3",          LLM_CHAT_TEMPLATE_CHATGLM_3         },
    { "chatglm4",          LLM_CHAT_TEMPLATE_CHATGLM_4         },
    { "minicpm",           LLM_CHAT_TEMPLATE_MINICPM           },
    { "exaone3",           LLM_CHAT_TEMPLATE_EXAONE_3          },
    { "rwkv-world",        LLM_CHAT_TEMPLATE_RWKV_WORLD        },
    { "granite",           LLM_CHAT_TEMPLATE_GRANITE           },
    { "gigachat",          LLM_CHAT_TEMPLATE_GIGACHAT          },
    { "megrez",            LLM_CHAT_TEMPLATE_MEGREZ            },
    { "llama4",            LLM_CHAT_TEMPLATE_LLAMA4            },
    { "hunyuan-moe",       LLM_CHAT_TEMPLATE_HUNYUAN_MOE       },
    { "kimi-k2",           LLM_CHAT_TEMPLATE_KIMI_K2           },
    { "gpt-oss",           LLM_CHAT_TEMPLATE_OPENAI_MOE        },
    { "bitnet",            LLM_CHAT_TEMPLATE_BITNET            },
    { "grok-2",            LLM_CHAT_TEMPLATE_GROK_2            },
    { "bailing",           LLM_CHAT_TEMPLATE_BAILING           },
    { "bailing-think",     LLM_CHAT_TEMPLATE_BAILING_THINK     },
    { "bailing2",          LLM_CHAT_TEMPLATE_BAILING2          },

};

//
// gguf helpers
//

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const bool *)data)[i] ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

std::string gguf_kv_to_str(const gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

//
// llama helpers
//

ggml_backend_buffer_type_t llama_default_buffer_type_cpu(bool host_buffer) {
    ggml_backend_buffer_type_t buft = nullptr;

#if defined(GGML_USE_CUDA)
    // host buffers should only be used when data is expected to be copied to/from the GPU
    if (host_buffer) {
        buft = ggml_backend_cuda_host_buffer_type();
    }
#elif defined(GGML_USE_SYCL)
    if (host_buffer) {
        buft = ggml_backend_sycl_host_buffer_type();
    }
#elif defined(GGML_USE_CPU_HBM)
    buft = ggml_backend_cpu_hbm_buffer_type();
#elif defined(GGML_USE_VULKAN)
    if (host_buffer) {
        buft = ggml_backend_vk_host_buffer_type();
    }
#endif

    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
    }
    return buft;

    GGML_UNUSED(host_buffer);
}

//
// globals
//

struct llama_state {
    llama_state() {
#ifdef GGML_USE_METAL
        ggml_backend_metal_log_set_callback(log_callback, log_callback_user_data);
#elif defined(GGML_USE_CUDA)
        ggml_backend_cuda_log_set_callback(log_callback, log_callback_user_data);
#elif defined(GGML_USE_CANN)
        ggml_backend_cann_log_set_callback(log_callback, log_callback_user_data);
#endif
    }

    // We save the log callback globally
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_state g_state;

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

static const char * llama_expert_gating_func_name(llm_expert_gating_func_type type) {
    switch (type) {
        case LLM_EXPERT_GATING_FUNC_SOFTMAX: return "softmax";
        case LLM_EXPERT_GATING_FUNC_SIGMOID: return "sigmoid";
        case LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT: return "softmax_weight";
        default:                             return "unknown";
    }
}

llama_model::~llama_model() {
    for (struct ggml_context * ctx : ctxs) {
        ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buf : bufs) {
#ifdef GGML_USE_CUDA
        if (ggml_backend_buffer_get_type(buf) == ggml_backend_cpu_buffer_type()) {
            ggml_backend_cuda_unregister_host_buffer(ggml_backend_buffer_get_base(buf));
        }
#endif
        ggml_backend_buffer_free(buf);
    }
    while (!lora_adapters.empty()) {
        llama_lora_adapter_free(*lora_adapters.begin());
    }
}

static size_t llama_get_device_count(const llama_model & model) {
    size_t count = 1;
#if defined(GGML_USE_CUDA)
    count = ggml_backend_cuda_get_device_count();
#elif defined(GGML_USE_SYCL)
    count = ggml_backend_sycl_get_device_count();
#elif defined(GGML_USE_VULKAN)
    count = ggml_backend_vk_get_device_count();
#elif defined(GGML_USE_CANN)
    return ggml_backend_cann_get_device_count();
#endif
#if defined(GGML_USE_RPC)
    count += model.rpc_servers.size();
#endif
    return count;
    GGML_UNUSED(model);
}

static ggml_backend_buffer_type_t llama_default_buffer_type_offload(const llama_model & model, int gpu) {
    ggml_backend_buffer_type_t buft = nullptr;

#if defined(GGML_USE_RPC)
    int dev_count = (int)llama_get_device_count(model);
    int rpc_count = (int)model.rpc_servers.size();
    if (gpu >= dev_count - rpc_count) {
        int rpc_idx = gpu - dev_count + rpc_count;
        rpc_device rpc = model.rpc_servers[rpc_idx];
        const char * endpoint = rpc.endpoint.c_str();
        return ggml_backend_rpc_buffer_type(endpoint, rpc.device);
    }
#endif
#if defined(GGML_USE_METAL)
    buft = ggml_backend_metal_buffer_type();
#elif defined(GGML_USE_CUDA)
    buft = ggml_backend_cuda_buffer_type(gpu);
#elif defined(GGML_USE_VULKAN)
    buft = ggml_backend_vk_buffer_type(gpu);
#elif defined(GGML_USE_SYCL)
    buft = ggml_backend_sycl_buffer_type(gpu);
#elif defined(GGML_USE_KOMPUTE)
    buft = ggml_backend_kompute_buffer_type(gpu);
    if (buft == nullptr) {
        LLAMA_LOG_WARN("%s: cannot use GPU %d, check `vulkaninfo --summary`\n", __func__, gpu);
    }
#elif defined(GGML_USE_CANN)
    buft = ggml_backend_cann_buffer_type(gpu);
#endif

    if (buft == nullptr) {
        buft = llama_default_buffer_type_cpu(true);
    }
    return buft;
    GGML_UNUSED(model);
    GGML_UNUSED(gpu);
}

static ggml_backend_buffer_type_t llama_default_buffer_type_split(const llama_model & model, int fallback_gpu) {
    ggml_backend_buffer_type_t buft = nullptr;

#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 1) {
        buft = ggml_backend_cuda_split_buffer_type(model.splits.data());
    }
#endif

#ifdef GGML_USE_SYCL
    if (ggml_backend_sycl_get_device_count() > 1) {
        buft = ggml_backend_sycl_split_buffer_type(model.splits.data());
    }
#endif

    if (buft == nullptr) {
        buft = llama_default_buffer_type_offload(model, fallback_gpu);
    }
    return buft;

}

int llama_model::device_count() const {
    return llama_get_device_count(*this);
}

ggml_backend_buffer_type_t llama_model::default_buffer_type_offload(int device) const {
    return llama_default_buffer_type_offload(*this, device);
}

static size_t llama_get_device_memory(const llama_model & model, int device) {
#if defined(GGML_USE_RPC)
    int dev_count = (int)llama_get_device_count(model);
    int rpc_count = (int)model.rpc_servers.size();
    if (device >= dev_count - rpc_count) {
        size_t total;
        size_t free;
        rpc_device rpc = model.rpc_servers[device - dev_count + rpc_count];
        const char * endpoint = rpc.endpoint.c_str();
        ggml_backend_rpc_get_device_memory(endpoint, rpc.device, &free, &total);
        return free;
    }
#endif
#if defined(GGML_USE_CUDA)
    size_t total;
    size_t free;
    ggml_backend_cuda_get_device_memory(device, &free, &total);
    return free;
#elif defined(GGML_USE_SYCL)
    size_t total;
    size_t free;
    ggml_backend_sycl_get_device_memory(device, &free, &total);
    return free;
#elif defined(GGML_USE_VULKAN)
    size_t total;
    size_t free;
    ggml_backend_vk_get_device_memory(device, &free, &total);
    return free;
#elif defined(GGML_USE_CANN)
    size_t total;
    size_t free;
    ggml_backend_cann_get_device_memory(device, &free, &total);
    return free;
#else
    return 1;
#endif
    GGML_UNUSED(model);
    GGML_UNUSED(device);
}

struct llama_context::Prev {
    int all_seq_id;
    int n_outputs;
    int n_kv;
    ggml_cgraph * graph;
};

void llama_context::reset_scheduler() {
    ggml_backend_sched_reset(sched);
    prev.reset();
}

bool llama_context::can_reuse_graph(const llama_batch & u_batch) {
    if (!prev || !prev->graph) return false;
    if (u_batch.n_tokens > 1) return false;
    if (u_batch.embd) return false;
    if (!cparams.graph_reuse) return false;
    return u_batch.all_seq_id == prev->all_seq_id &&
           kv_self.head > 0 &&
           kv_self.n == prev->n_kv &&
           n_outputs == prev->n_outputs &&
           update_cache_copies();
}

bool llama_context::update_cache_copies() {
    int n_layer = model.hparams.n_layer - model.hparams.nextn_predict_layers; //cache_copies.size()/2;
    if ((int)kv_self.k_l.size() != n_layer) return false;
    if (!(kv_self.v_l.empty() || (int)kv_self.v_l.size() == n_layer)) return false;
    if ((model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) && model.splits.size() > 1) {
        for (int il = 0; il < n_layer; ++il) {
            auto kl = (ggml_split_tensor_t *)kv_self.k_l[il]->extra;
            auto vl = !kv_self.v_l.empty() && kv_self.v_l[il] ? (ggml_split_tensor_t *)kv_self.v_l[il]->extra : nullptr;
            GGML_ASSERT(kl && (!kv_self.v_l[il] || vl));
            if (vl) {
                GGML_ASSERT(kl->n_device == vl->n_device);
            }
            for (int id = 0; id < kl->n_device; ++id) {
                auto& c = cache_copies[2*model.splits.size()*il + 2*id + 0];
                if (!c.cpy || c.cpy->op != GGML_OP_CPY || c.cpy->view_src != kl->splits[id]) return false;
                c.cpy->view_offs = kv_self.head*c.step;
                c.cpy->src[1]->data = (char *)kl->splits[id]->data + c.cpy->view_offs;
                c.cpy->data = c.cpy->src[1]->data;
            }
            if (!vl) continue;
            for (int id = 0; id < vl->n_device; ++id) {
                auto& c = cache_copies[2*model.splits.size()*il + 2*id + 1];
                if (!c.cpy || c.cpy->op != GGML_OP_CPY || c.cpy->view_src != vl->splits[id]) return false;
                c.cpy->view_offs = kv_self.head*c.step;
                c.cpy->src[1]->data = (char *)vl->splits[id]->data + c.cpy->view_offs;
                c.cpy->data = c.cpy->src[1]->data;
            }
        }
    } else {
        for (int il = 0; il < n_layer; ++il) {
            auto& c = cache_copies[2*il+0];
            if (!c.cpy || c.cpy->op != GGML_OP_CPY || c.cpy->view_src != kv_self.k_l[il]) return false;
            c.cpy->view_offs = kv_self.head*c.step;
            c.cpy->src[1]->data = (char *)kv_self.k_l[il]->data + c.cpy->view_offs;
            c.cpy->data = c.cpy->src[1]->data;
        }
        if (kv_self.v_l.empty()) return true;
        for (int il = 0; il < n_layer; ++il) {
            auto& c = cache_copies[2*il+1];
            if (!c.cpy || c.cpy->op != GGML_OP_CPY || c.cpy->view_src != kv_self.v_l[il]) return false;
            c.cpy->view_offs = kv_self.head*c.step;
            c.cpy->src[1]->data = (char *)kv_self.v_l[il]->data + c.cpy->view_offs;
            c.cpy->data = c.cpy->src[1]->data;
        }
    }
    return true;
}

llama_context::llama_context(const llama_model & model)
    : model(model) , sampling(llama_n_vocab(&model)) , t_start_us(model.t_start_us) , t_load_us(model.t_load_us) {
    const auto & hparams = model.hparams;
    if ((model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) && model.splits.size() > 1) {
        cache_copies.resize(2*model.splits.size()*hparams.n_layer);
    } else {
        cache_copies.resize(2*hparams.n_layer);
    }
}

llama_context::~llama_context() {
    ggml_backend_sched_free(sched);

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    ggml_backend_buffer_free(buf_output);
}

//
// kv cache helpers
//

static bool llama_kv_cache_init(
             struct llama_kv_cache & cache,
               const llama_context * ctx,
                         ggml_type   type_k,
                         ggml_type   type_v,
                          uint32_t   kv_size,
                              bool   offload) {
    const llama_model & model = ctx->model;
    const llama_cparams & cparams = ctx->cparams;

    const struct llama_hparams & hparams = model.hparams;

    const int64_t  n_layer = hparams.n_layer - hparams.nextn_predict_layers;

    cache.has_shift = false;

    // TODO: find a nicer way to add other recurrent model architectures
    cache.recurrent = model.arch == LLM_ARCH_MAMBA;
    cache.v_trans   = !cache.recurrent && !cparams.flash_attn;

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k  = type_k;
    cache.type_v  = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);

    if (cache.recurrent) {
        // init state copy sources
        for (uint32_t i = 0; i < cache.size; ++i) {
            cache.cells[i].src = i;
        }
    }

    bool split_cache   = false;
    if ((model.split_mode == LLAMA_SPLIT_MODE_GRAPH || model.split_mode == LLAMA_SPLIT_MODE_ATTN) && model.arch != LLM_ARCH_DEEPSEEK2 && offload) {
        cache.split_k_l.reserve(n_layer);
        cache.split_v_l.reserve(n_layer);
        split_cache = true;
    }

    // count used buffer types
    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    if (offload) {
        for (int64_t i = 0; i < n_layer; ++i) {
            if (split_cache) {
                buft_layer_count[model.buft_layer[i].buft_matrix]++;
            } else {
                buft_layer_count[model.buft_layer[i].buft]++;
            }
        }
    } else {
        buft_layer_count[llama_default_buffer_type_cpu(true)] = n_layer;
    }

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (auto & it : buft_layer_count) {
        int n_layers = it.second;
        size_t ctx_mem_size = 5u*n_layers*ggml_tensor_overhead();
        if (split_cache) ctx_mem_size += 2*model.splits.size()*n_layers*ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_mem_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for kv cache\n", __func__);
            return false;
        }
        ctx_map[it.first] = ctx;
        cache.ctxs.push_back(ctx);
    }

    if (model.arch == LLM_ARCH_DEEPSEEK2) {
        bool have_wkv_b = true;
        for (auto& l : model.layers) {
            if (!l.wkv_b) {
                have_wkv_b = false;
                break;
            }
        }
        if (!have_wkv_b) {
            if (cparams.mla_attn != 1) {
                LLAMA_LOG_WARN("=========================================================\n");
                LLAMA_LOG_WARN("%s: missing wkv_b tensor(s)\n", __func__);
                LLAMA_LOG_WARN("%s: changing MLA from %d to 1\n", __func__, cparams.mla_attn);
                if (cparams.mla_attn > 1) {
                    LLAMA_LOG_WARN("%s: ** Prompt processing performance will be crippled **\n", __func__);
                }
                LLAMA_LOG_WARN("=========================================================\n");
                // Sorry for the hack.
                auto& non_cparams = const_cast<llama_cparams&>(cparams);
                non_cparams.mla_attn = 1;
            }
        }
    }

    bool needs_v_cache = true;
    cache.k_l.reserve(n_layer);
    if (model.arch == LLM_ARCH_DEEPSEEK2 && cparams.mla_attn) {
        needs_v_cache = cparams.mla_attn == 1 && !cparams.flash_attn;
    }
    if (needs_v_cache) cache.v_l.reserve(n_layer);

    std::vector<size_t> mem_split(model.splits.size(), 0);

    int n_mla = 0;
    for (int i = 0; i < (int) n_layer; i++) {
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();
        const uint32_t n_head_kv    = hparams.n_head_kv(i);
        const uint32_t n_embd_head_k= hparams.n_embd_head_k;

        struct ggml_context * ctx = split_cache ? ctx_map.at(model.buft_layer[i].buft_matrix) : offload ? ctx_map.at(model.buft_layer[i].buft) : cache.ctxs.front();
        ggml_tensor * k;
        ggml_tensor * v;
        if (model.arch == LLM_ARCH_DEEPSEEK2 && cparams.mla_attn) {
            // DeepSeek MLA
            const uint32_t n_embd_head_qk_rope = hparams.n_rot;
            const uint32_t kv_lora_rank = hparams.n_lora_kv;
            //LLAMA_LOG_INFO("%s: layer %d: n_embd_head_qk_rope = %d, kv_lora_rank = %d\n", __func__, i, n_embd_head_qk_rope, kv_lora_rank);
            if (cparams.flash_attn) {
                ggml_tensor * kv = ggml_new_tensor_2d(ctx, cache.type_k, kv_lora_rank + n_embd_head_qk_rope, kv_size);
                ggml_format_name(kv, "cache_k_l%d", i);
                cache.k_l.push_back(kv);
            } else {
                auto kv_type = cparams.mla_attn == 1 ? cache.type_k : cache.type_v;
                ggml_tensor * kv = ggml_new_tensor_2d(ctx, kv_type, kv_lora_rank + n_embd_head_qk_rope, kv_size);
                ggml_format_name(kv, "cache_k_l%d", i);
                cache.k_l.push_back(kv);
                if (cparams.mla_attn == 1) {
                    ggml_tensor * kvt = ggml_new_tensor_1d(ctx, cache.type_v, kv_lora_rank*kv_size);
                    ggml_format_name(kvt, "cache_v_l%d", i);
                    cache.v_l.push_back(kvt);
                }
            }
            n_mla++;
        }
        else {
            k = ggml_new_tensor_2d(ctx, type_k, n_embd_head_k, n_head_kv*kv_size);
            v = ggml_new_tensor_1d(ctx, type_v, n_embd_v_gqa*kv_size);
            auto k_name = std::string{"cache_k_l"} + std::to_string(i);
            auto v_name = std::string{"cache_v_l"} + std::to_string(i);
            ggml_set_name(k, k_name.c_str());
            ggml_set_name(v, v_name.c_str());
            //ggml_format_name(k, "cache_k_l%d", i);
            //ggml_format_name(v, "cache_v_l%d", i);
            cache.k_l.push_back(k);
            cache.v_l.push_back(v);
            if (split_cache) {
                auto K = model.layers[i].wk;
                auto V = model.layers[i].wv;
                if (K && V && K->extra && V->extra) {
                    auto extra_K = (const ggml_split_tensor_t *)K->extra;
                    auto extra_V = (const ggml_split_tensor_t *)V->extra;
                    auto & split_k_l = cache.split_k_l.emplace_back();
                    auto & split_v_l = cache.split_v_l.emplace_back();
                    split_k_l.tensor_splits.resize(extra_K->n_device, nullptr);
                    split_v_l.tensor_splits.resize(extra_V->n_device, nullptr);
                    for (int is = 0; is < extra_K->n_device; ++is) {
                        auto split = extra_K->splits[is];
                        if (!split) continue;
                        split_k_l.tensor_splits[is] = ggml_new_tensor_2d(ctx, type_k, n_embd_head_k, split->ne[1]/n_embd_head_k * kv_size);
                        auto split_name = k_name + '.' + std::to_string(is);
                        ggml_set_name(split_k_l.tensor_splits[is], split_name.c_str());
                        mem_split[is] += ggml_nbytes(split_k_l.tensor_splits[is]);
                    }
                    split_k_l.ggml.n_device  = extra_K->n_device;
                    split_k_l.ggml.split_dim = 0;
                    split_k_l.ggml.splits    = split_k_l.tensor_splits.data();
                    for (int is = 0; is < extra_V->n_device; ++is) {
                        auto split = extra_V->splits[is];
                        if (!split) continue;
                        split_v_l.tensor_splits[is] = ggml_new_tensor_1d(ctx, type_v, split->ne[1] * kv_size);
                        auto split_name = v_name + '.' + std::to_string(is);
                        ggml_set_name(split_v_l.tensor_splits[is], split_name.c_str());
                        mem_split[is] += ggml_nbytes(split_v_l.tensor_splits[is]);
                    }
                    split_v_l.ggml.n_device  = extra_V->n_device;
                    split_v_l.ggml.split_dim = 0;
                    split_v_l.ggml.splits    = split_v_l.tensor_splits.data();
                    k->extra = (void *)&split_k_l.ggml;
                    v->extra = (void *)&split_v_l.ggml;
                }
                //} else {
                //    printf("Oops: don't have yet K and V for layer %d\n", i);
                //}
            }
        }
    }
    if (model.arch == LLM_ARCH_DEEPSEEK2 && cparams.mla_attn && n_mla < n_layer && n_mla > 0) {
        LLAMA_LOG_ERROR("%s: unexpected situation with %d out of %d layers having MLA enabled\n", __func__, n_mla, int(n_layer));
        LLAMA_LOG_ERROR("%s: bailing out\n", __func__);
        GGML_ABORT("fatal error");
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        int ntensor = 0;
        for (auto t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            ++ntensor;
        }
        if (ntensor > 0) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
            cache.bufs.push_back(buf);
        }
    }
    if (split_cache) {
        LLAMA_LOG_INFO("%s: KV cache size per device:\n", __func__);
        for (int i = 0; i < int(mem_split.size()); ++i) printf("    Device %d:  %g MiB\n", i, mem_split[i]/1024./1024.);
    }

#if 0
    for (int il = 0; il < n_layer; ++il) {
        if (cache.k_l[il]->extra) {
            printf("Layer %2d, K-buffer: %p:", il, (void *)cache.k_l[il]->buffer);
            auto split_kl = (ggml_split_tensor_t *)cache.k_l[il]->extra;
            for (int id = 0; id < split_kl->n_device; ++id) {
                if (split_kl->splits[id]) printf(" %p,%p", (void *)split_kl->splits[id]->data, (void *)split_kl->splits[id]->buffer);
            }
            printf("\n");
        }
        if (cache.v_l[il]->extra) {
            printf("Layer %2d, V-buffer: %p:", il, (void *)cache.v_l[il]->buffer);
            auto split_vl = (ggml_split_tensor_t *)cache.v_l[il]->extra;
            for (int id = 0; id < split_vl->n_device; ++id) {
                if (split_vl->splits[id]) printf(" %p,%p", (void *)split_vl->splits[id]->data, (void *)split_vl->splits[id]->buffer);
            }
            printf("\n");
        }
    }
#endif

    return true;
}

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
// Note: On success, it's important that cache.head points
// to the first cell of the slot.
static bool llama_kv_cache_find_slot(
           struct llama_kv_cache & cache,
        const struct llama_batch & batch) {
    const uint32_t n_tokens = batch.n_tokens;

    if (cache.recurrent) {
        // For recurrent state architectures (like Mamba),
        // each KV cache cell can store the state for a whole sequence.

        llama_seq_id min = cache.size - 1;
        llama_seq_id max = 0;

        for (uint32_t i = 0; i < n_tokens; ++i) {
            for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
                llama_seq_id seq_id = batch.seq_id[i][j];
                // make sure it's a valid seq_id
                if ((uint32_t) seq_id < cache.size) {
                    if (seq_id > max) {
                        max = seq_id;
                    }
                    if (seq_id < min) {
                        min = seq_id;
                    }
                    // Assuming the tokens are in-order
                    if (batch.pos[i] != cache.cells[seq_id].pos + 1) {
                        // What should happen when the pos backtracks or skips a value?
                        // Clearing the state mid-batch would require special-casing which isn't done.
                        LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d\n",
                            __func__, batch.pos[i], cache.cells[seq_id].pos, seq_id);
                    }
                    if (cache.cells[seq_id].pos < 0 && 0 <= batch.pos[i]) {
                        cache.used += 1;
                    }
                    cache.cells[seq_id].pos = batch.pos[i];
                    // NOTE: seq_ids are not inserted here; they are handled when the input tensors are set
                } else {
                    // too big seq_id
                    // TODO: would it be possible to resize the KV cache size instead?
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= kv_size=%d Try using a bigger --parallel value\n", __func__, seq_id, cache.size);
                    return false;
                }
            }
        }

        // allow getting the range of used cells, from head to head + n
        cache.head = min;
        cache.n    = max - min + 1;

        // sanity check
        return max >= min;
    }
    // otherwise, one cell per token.

    if (n_tokens > cache.size) {
        LLAMA_LOG_ERROR("%s: n_tokens=%d > cache.size=%d\n", __func__, n_tokens, cache.size);
        return false;
    }

    uint32_t n_tested = 0;

    while (true) {
        if (cache.head + n_tokens > cache.size) {
            n_tested += cache.size - cache.head;
            cache.head = 0;
            continue;
        }

        bool found = true;
        for (uint32_t i = 0; i < n_tokens; i++) {
            if (cache.cells[cache.head + i].pos >= 0) {
                found = false;
                cache.head += i + 1;
                n_tested   += i + 1;
                break;
            }
        }

        if (found) {
            break;
        }

        if (n_tested >= cache.size) {
            //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
            return false;
        }
    }

    for (uint32_t i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];

        for (int32_t j = 0; j < batch.n_seq_id[i]; j++) {
            cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i][j]);
        }
    }

    cache.used += n_tokens;

    return true;
}

// find how many cells are currently in use
static uint32_t llama_kv_cache_cell_max(const struct llama_kv_cache & cache) {
    for (uint32_t i = cache.size; i > 0; --i) {
        const llama_kv_cell & cell = cache.cells[i - 1];

        if (cell.pos >= 0 && !cell.is_empty()) {
            return i;
        }
    }

    return 0;
}

static void llama_kv_cache_clear(struct llama_kv_cache & cache) {
    for (int32_t i = 0; i < (int32_t) cache.size; ++i) {
        cache.cells[i].pos = -1;
        cache.cells[i].seq_id.clear();
    }
    cache.head = 0;
    cache.used = 0;

    for (auto & buf : cache.bufs) {
        ggml_backend_buffer_clear(buf, 0);
    }
}

static bool llama_kv_cache_seq_rm(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    // models like Mamba can't have a state partially erased
    if (cache.recurrent) {
        if (seq_id >= (int64_t) cache.size) {
            // could be fatal
            return false;
        }
        if (0 <= seq_id) {
            // partial intersection is invalid
            if ((0 < p0 && p0 <= cache.cells[seq_id].pos) || (0 < p1 && p1 <= cache.cells[seq_id].pos)) {
                return false;
            }
        } else {
            // seq_id is negative, then the range should include everything or nothing
            if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
                return false;
            }
        }
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            if (seq_id < 0) {
                cache.cells[i].seq_id.clear();
            } else if (cache.cells[i].has_seq_id(seq_id)) {
                cache.cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cache.cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cache.cells[i].pos >= 0) cache.used--;

                cache.cells[i].pos = -1;
                if (new_head == cache.size) new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;

    return true;
}

static void llama_kv_cache_seq_cp(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id_src,
                 llama_seq_id   seq_id_dst,
                    llama_pos   p0,
                    llama_pos   p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    if (cache.recurrent) {
        if ((uint32_t) seq_id_dst < cache.size && (uint32_t) seq_id_src < cache.size) {
            seq_id_src = cache.cells[seq_id_src].src;
            GGML_ASSERT((uint32_t) seq_id_src < cache.size);
            // intent to "copy from"
            // supports copy chains thanks to taking the source of the source
            cache.cells[seq_id_dst].src = seq_id_src;

            // preserve the "keep or clear" status of the copied sequence
            if (cache.cells[seq_id_src].has_seq_id(seq_id_src)) {
                cache.cells[seq_id_dst].seq_id.insert(seq_id_dst);
            } else {
                cache.cells[seq_id_dst].seq_id.erase(seq_id_dst);
            }

            cache.do_copy = true;

            cache.cells[seq_id_dst].pos = cache.cells[seq_id_src].pos;
        }
        return;
    }
    // otherwise, this is the KV cache of a Transformer-like model

    cache.head = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id_src) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.cells[i].seq_id.insert(seq_id_dst);
        }
    }
}

static void llama_kv_cache_seq_keep(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    uint32_t new_head = cache.size;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (!cache.cells[i].has_seq_id(seq_id)) {
            if (cache.cells[i].pos >= 0) cache.used--;
            cache.cells[i].pos = -1;
            cache.cells[i].seq_id.clear();
            if (new_head == cache.size) new_head = i;
        } else {
            cache.cells[i].seq_id.clear();
            cache.cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cache.size && new_head < cache.head) cache.head = new_head;
}

static void llama_kv_cache_seq_add(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                    llama_pos   delta) {
    uint32_t new_head = cache.size;

    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) return;

    if (cache.recurrent) {
        // for Mamba-like models, only the pos needs to be shifted
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            llama_kv_cell & cell = cache.cells[seq_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += delta;
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;
            cache.cells[i].pos   += delta;
            cache.cells[i].delta += delta;

            if (cache.cells[i].pos < 0) {
                if (!cache.cells[i].is_empty()) {
                    cache.used--;
                }
                cache.cells[i].pos = -1;
                cache.cells[i].seq_id.clear();
                if (new_head == cache.size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    cache.head = new_head != cache.size ? new_head : 0;
}

static void llama_kv_cache_seq_div(
        struct llama_kv_cache & cache,
                 llama_seq_id   seq_id,
                    llama_pos   p0,
                    llama_pos   p1,
                          int   d) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) return;

    if (cache.recurrent) {
        // for Mamba-like models, only the pos needs to be changed
        if (0 <= seq_id && seq_id < (int64_t) cache.size) {
            llama_kv_cell & cell = cache.cells[seq_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
        return;
    }

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id) && cache.cells[i].pos >= p0 && cache.cells[i].pos < p1) {
            cache.has_shift = true;

            {
                llama_pos p_old = cache.cells[i].pos;
                cache.cells[i].pos   /= d;
                cache.cells[i].delta += cache.cells[i].pos - p_old;
            }
        }
    }
}

static llama_pos llama_kv_cache_seq_pos_max(struct llama_kv_cache & cache, llama_seq_id seq_id) {
    llama_pos result = 0;

    for (uint32_t i = 0; i < cache.size; ++i) {
        if (cache.cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cache.cells[i].pos);
        }
    }

    return result;
}

static void llama_kv_cache_defrag(struct llama_kv_cache & cache) {
    cache.do_defrag = true;
}

static uint32_t llama_kv_cache_get_padding(const struct llama_cparams & cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

//
// model loading and saving
//

//
// load LLaMA models
//

void llm_load_arch(llama_model_loader & ml, llama_model & model) {
    model.arch = ml.get_arch();
    if (model.arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

static void llm_load_print_meta(llama_model_loader & ml, llama_model & model) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const char * rope_scaling_type = hparams.rope_scaling_type_name(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)> & f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        } else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: format           = %s\n",     __func__, llama_file_version_name(ml.fver));
    LLAMA_LOG_INFO("%s: arch             = %s\n",     __func__, llama_model_arch_name(model.arch));

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n",     __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n",     __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer          = %u\n",     __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head           = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv        = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot            = %u\n",     __func__, hparams.n_rot);
        LLAMA_LOG_INFO("%s: n_swa            = %u\n",     __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: n_swa_pattern    = %u\n",     __func__, hparams.n_swa_pattern);
        LLAMA_LOG_INFO("%s: n_embd_head_k    = %u\n",     __func__, hparams.n_embd_head_k);
        LLAMA_LOG_INFO("%s: n_embd_head_v    = %u\n",     __func__, hparams.n_embd_head_v);
        LLAMA_LOG_INFO("%s: n_gqa            = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa     = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa     = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n",   __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n",   __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n",   __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n",   __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale    = %.1e\n",   __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: n_ff             = %s\n",     __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert         = %u\n",     __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used    = %u\n",     __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n",     __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n",     __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n",     __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n",     __func__, rope_scaling_type);
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n",   __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n",     __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n",     __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n",     __func__, hparams.rope_finetuned ? "yes" : "unknown");
        // MRoPE (Multi-axis Rotary Position Embedding) sections
        if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
            LLAMA_LOG_INFO("%s: mrope sections   = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
        }
        LLAMA_LOG_INFO("%s: ssm_d_conv       = %u\n",     __func__, hparams.ssm_d_conv);
        LLAMA_LOG_INFO("%s: ssm_d_inner      = %u\n",     __func__, hparams.ssm_d_inner);
        LLAMA_LOG_INFO("%s: ssm_d_state      = %u\n",     __func__, hparams.ssm_d_state);
        LLAMA_LOG_INFO("%s: ssm_dt_rank      = %u\n",     __func__, hparams.ssm_dt_rank);
    }

    LLAMA_LOG_INFO("%s: model type       = %s\n",     __func__, llama_model_type_name(model.type));
    LLAMA_LOG_INFO("%s: model ftype      = %s\n",     __func__, llama_model_ftype_name(model.ftype).c_str());
    if (ml.n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.3f T\n", __func__, ml.n_elements*1e-12);
    } else if (ml.n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.3f B\n", __func__, ml.n_elements*1e-9);
    } else if (ml.n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.3f M\n", __func__, ml.n_elements*1e-6);
    } else {
        LLAMA_LOG_INFO("%s: model params     = %.3f K\n", __func__, ml.n_elements*1e-3);
    }
    if (ml.n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: model size       = %.3f MiB (%.3f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0,        ml.n_bytes*8.0/ml.n_elements);
    } else {
        LLAMA_LOG_INFO("%s: model size       = %.3f GiB (%.3f BPW) \n", __func__, ml.n_bytes/1024.0/1024.0/1024.0, ml.n_bytes*8.0/ml.n_elements);
    }
    {
        auto n_bytes = ml.n_bytes;
        auto n_elements = ml.n_elements;
        auto meta_tke = ml.get_tensor_meta("token_embd.weight");
        auto meta_out = ml.get_tensor_meta("output.weight");
        if (meta_tke && meta_out) {
            n_bytes -= ggml_nbytes(meta_tke);
            n_elements -= ggml_nelements(meta_tke);
            n_bytes -= ggml_nbytes(meta_out);
            n_elements -= ggml_nelements(meta_out);
            if (n_bytes < GiB) {
                LLAMA_LOG_INFO("%s: repeating layers = %.3f MiB (%.3f BPW", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
            } else {
                LLAMA_LOG_INFO("%s: repeating layers = %.3f GiB (%.3f BPW", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
            }
            if (ml.n_elements >= 1e9) {
                LLAMA_LOG_INFO(", %.3f B parameters)\n", n_elements*1e-9);
            } else {
                LLAMA_LOG_INFO(", %.3f M parameters)\n", n_elements*1e-6);
            }
        }
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n",    __func__, model.name.c_str());

    if (model.arch == LLM_ARCH_DEEPSEEK2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_lora_q             = %d\n",     __func__, hparams.n_lora_q);
        LLAMA_LOG_INFO("%s: n_lora_kv            = %d\n",     __func__, hparams.n_lora_kv);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((enum llm_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul    = %.4f\n",   __func__, hparams.rope_yarn_log_mul);
    }

    if (model.arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n",     __func__, hparams.n_ff_shexp);
    }

    if (model.arch == LLM_ARCH_QWEN3MOE || model.arch == LLM_ARCH_OPENAI_MOE || model.arch == LLM_ARCH_QWEN3VLMOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n",     __func__, hparams.n_ff_exp);
    }

    if (model.arch == LLM_ARCH_GRANITE || model.arch == LLM_ARCH_GRANITE_MOE) {
        LLAMA_LOG_INFO("%s: f_embedding_scale = %f\n", __func__, hparams.f_embedding_scale);
        LLAMA_LOG_INFO("%s: f_residual_scale  = %f\n", __func__, hparams.f_residual_scale);
        LLAMA_LOG_INFO("%s: f_attention_scale = %f\n", __func__, hparams.f_attention_scale);
    }

    if (model.arch == LLM_ARCH_BAILINGMOE2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n",     __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n",     __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp           = %d\n",     __func__, hparams.n_ff_shexp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n",     __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: n_expert_groups      = %d\n",     __func__, hparams.n_expert_groups);
        LLAMA_LOG_INFO("%s: n_group_used         = %d\n",     __func__, hparams.n_group_used);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n",   __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: expert_weights_norm  = %d\n",     __func__, hparams.expert_weights_norm);
        LLAMA_LOG_INFO("%s: expert_gating_func   = %s\n",     __func__, llama_expert_gating_func_name((llm_expert_gating_func_type) hparams.expert_gating_func));
        LLAMA_LOG_INFO("%s: nextn_predict_layers = %d\n",     __func__, hparams.nextn_predict_layers);
    }

    vocab.print_info();

}

static void llm_prepare_mla(llama_model & model, int mla) {
    if (model.arch != LLM_ARCH_DEEPSEEK2) return;
    const auto& hparams = model.hparams;
    const int n_layer = model.layers.size();
    int n_to_compute = 0;
    for (auto& l : model.layers) {
        if (!l.wk_b) ++n_to_compute;
    }
    if (mla > 0 && n_to_compute > 0) {
        // Prepare wk_b tensors to enable MLA usage also for model files that do not include
        // the wk_b tensors (because, e.g., they were converted using mainline llama.cpp)
        // We do it here because otherwise wkv_b may get run-time-repacked, which will make
        // preparation of wk_b impossible. It also has the benefit that wk_b will get automatically
        // run-time repacked if the rtr option is set. The downside is that we will prepare wk_b
        // even if it is not needed (because MLA is not being used). If we wanted to avoid
        // computing wk_b from wkv_b if not needed, we would need to propagate the context parameters
        // to the model loading function. On the other hand, in some hypothetical bright future,
        // where we are able to use the optimum settings for the computation, which for DeepSeekV3/R1/Lite
        // is no MLA + FA for prompt processing, and MLA + FA for token generation, it would be useful
        // to change the MLA setting on the fly, depending on context. In that case, having prepared
        // the MLA tensors here is the right ting to do^TM.
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;
        const int32_t n_embd_head_v = hparams.n_embd_head_v;
        const int32_t n_head        = hparams.n_head(0);
        std::vector<uint8_t> work_data;
        LLAMA_LOG_INFO("============ %s: need to compute %d wk_b/wv_b tensors\n", __func__, n_to_compute);
        for (int il = 1; il < n_layer; ++il) {
            // Somehow the number of heads is being defined as being per layer. Not sure why this is the
            // case, but for now we do not support strange models that have different numbers of heads
            // in different model layers.
            if ((int)hparams.n_head(il) != n_head) throw std::runtime_error("Unsupported configuration");
        }
        size_t max_wkv_size = 0;
        size_t max_wk_size = 0;
        for (auto& l : model.layers) {
            if (!l.wk_b) {
                auto new_type = ggml_is_quantized(l.wkv_b->type) ? GGML_TYPE_Q8_0 : l.wkv_b->type;
                auto size = ggml_row_size(new_type, n_embd_head_qk_nope)*kv_lora_rank*n_head;
                max_wk_size = std::max(max_wk_size, size);
                if (!ggml_backend_buffer_is_host(l.wkv_b->buffer)) {
                    max_wkv_size = std::max(max_wkv_size, ggml_nbytes(l.wkv_b));
                }
            }
        }
        auto context_size = max_wk_size + 2*n_embd_head_qk_nope*kv_lora_rank*n_head*sizeof(float);
        context_size *= 2; // just in case;
        std::vector<uint8_t> wkv_buffer;
        if (max_wkv_size > 0) wkv_buffer.resize(max_wkv_size);
        // So, transposing tensors and then making them contiguous as needed for wk_b may or may not
        // be supported on all backends. Hence, to be sure that the preparation of wk_b will
        // work correctly, we do it on the CPU backend. We then copy the resulting tensor data to
        // the bacikend where wkv_b is stored.
        ggml_init_params params{context_size, nullptr, true};
        auto ctx = ggml_init(params);
        auto graph = ggml_new_graph_custom(ctx, 8, false);
        std::vector<uint8_t> tensor_data(2*n_embd_head_qk_nope*kv_lora_rank*n_head*sizeof(float) + max_wk_size);
        for (int il = 0; il < n_layer; ++il) {
            auto& l = model.layers[il];
            if (l.wk_b) continue;
            auto wkv_b = *l.wkv_b;
            if (!ggml_backend_buffer_is_host(l.wkv_b->buffer)) {
                ggml_backend_tensor_get(l.wkv_b, wkv_buffer.data(), 0, ggml_nbytes(l.wkv_b));
                wkv_b.data = wkv_buffer.data();
            }
            auto wk_b_view = ggml_view_3d(ctx, &wkv_b, kv_lora_rank, n_embd_head_qk_nope, n_head,
                    l.wkv_b->nb[1], l.wkv_b->nb[1]*(n_embd_head_qk_nope + n_embd_head_v), 0);
            auto wk_b_f32 = ggml_cast(ctx, wk_b_view, GGML_TYPE_F32);
            wk_b_f32->data = tensor_data.data();
            auto wk_b_f32_tview = ggml_transpose(ctx, wk_b_f32);
            auto wk_b_f32_t = ggml_cont(ctx, wk_b_f32_tview);
            wk_b_f32_t->data = (char *)wk_b_f32->data + ggml_nbytes(wk_b_f32);

            auto new_type = ggml_is_quantized(wkv_b.type) ?
                wkv_b.type >= GGML_TYPE_Q4_0_R8 && wkv_b.type <= GGML_TYPE_Q8_K_R8 ? GGML_TYPE_Q8_0_R8 : GGML_TYPE_Q8_0 : wkv_b.type;
            auto wk_b = ggml_cast(ctx, wk_b_f32_t, new_type);
            wk_b->data = (char *)wk_b_f32_t->data + ggml_nbytes(wk_b_f32_t);

            ggml_build_forward_expand(graph, wk_b);

            auto plan = ggml_graph_plan(graph, std::thread::hardware_concurrency()/2);
            if (plan.work_size > work_data.size()) work_data.resize(plan.work_size);
            plan.work_data = work_data.data();

            auto status = ggml_graph_compute(graph, &plan);
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Failed to compute wk_b");

            auto name = std::string{"blk."} + std::to_string(il) + ".attn_k_b.weight";

            l.computed_wk_b = std::make_unique<ggml_tensor>(*wk_b);
            l.computed_wk_b->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_buffer_get_type(l.wkv_b->buffer), ggml_nbytes(wk_b));
            l.computed_wk_b->data   = ggml_backend_buffer_get_base(l.computed_wk_b->buffer);
            l.computed_wk_b->op = GGML_OP_NONE; // we absolutely need to do this, else the backend will attempt to find the parents
                                                // of wk_b, which no longer exist, and will therefore crash.
            for (int j = 0; j < GGML_MAX_SRC; ++j) l.computed_wk_b->src[j] = nullptr;
            ggml_set_name(l.computed_wk_b.get(), name.c_str());
            ggml_backend_buffer_set_usage(l.computed_wk_b->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(l.computed_wk_b.get(), wk_b->data, 0, ggml_nbytes(wk_b));
            if (ggml_backend_buffer_is_host(l.computed_wk_b->buffer)) {
                iqk_modify_tensor(l.computed_wk_b.get());
            }

            l.wk_b = l.computed_wk_b.get();
            model.tensors_by_name.push_back(std::make_pair(name, l.wk_b));

            printf("Computed %s as %ld x %ld x %ld and stored in buffer %s\n", name.c_str(), wk_b->ne[0], wk_b->ne[1], wk_b->ne[2],
                    ggml_backend_buffer_name(l.computed_wk_b->buffer));

            ggml_graph_clear(graph);
            auto wv_b = ggml_cont(ctx, ggml_view_3d(ctx, &wkv_b, kv_lora_rank, n_embd_head_v, n_head,
                        l.wkv_b->nb[1], l.wkv_b->nb[1]*(n_embd_head_qk_nope + n_embd_head_v), l.wkv_b->nb[1]*n_embd_head_qk_nope));
            wv_b->data = tensor_data.data();
            ggml_build_forward_expand(graph, wv_b);
            plan = ggml_graph_plan(graph, std::thread::hardware_concurrency()/2);
            if (plan.work_size > work_data.size()) work_data.resize(plan.work_size);
            plan.work_data = work_data.data();
            status = ggml_graph_compute(graph, &plan);
            if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Failed to compute wv_b");

            name = std::string{"blk."} + std::to_string(il) + ".attn_v_b.weight";

            l.computed_wv_b = std::make_unique<ggml_tensor>(*wv_b);
            l.computed_wv_b->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_buffer_get_type(l.wkv_b->buffer), ggml_nbytes(wv_b));
            l.computed_wv_b->data   = ggml_backend_buffer_get_base(l.computed_wv_b->buffer);
            l.computed_wv_b->op = GGML_OP_NONE; // we absolutely need to do this, else the backend will attempt to find the parents
                                                // of wk_b, which no longer exist, and will therefore crash.
            for (int j = 0; j < GGML_MAX_SRC; ++j) l.computed_wv_b->src[j] = nullptr;
            ggml_set_name(l.computed_wv_b.get(), name.c_str());
            ggml_backend_buffer_set_usage(l.computed_wv_b->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(l.computed_wv_b.get(), wv_b->data, 0, ggml_nbytes(wv_b));
            if (ggml_backend_buffer_is_host(l.computed_wv_b->buffer)) {
                iqk_modify_tensor(l.computed_wv_b.get());
            }

            l.wv_b = l.computed_wv_b.get();
            model.tensors_by_name.push_back(std::make_pair(name, l.wv_b));

            printf("Computed %s as %ld x %ld x %ld and stored in buffer %s\n", name.c_str(), wv_b->ne[0], wv_b->ne[1], wv_b->ne[2],
                    ggml_backend_buffer_name(l.computed_wv_b->buffer));

            ggml_graph_clear(graph);
        }
        ggml_free(ctx);
    }
    if (mla == 1) return;

    n_to_compute = 0;
    for (auto& l : model.layers) {
        if (l.wk_b && l.wv_b && !l.wkv_b) ++n_to_compute;
    }
    if (n_to_compute == 0) return;

    //
    // Prepare wkv_b tensors to enable MLA=2,3 usage also for model files that have been
    // crippled to the mainline llama.cpp MLA implementation (MLA=1 here).
    // We do it here because otherwise wk_b and wv_b may get run-time-repacked, which will make
    // preparation of wkv_b impossible. It also has the benefit that wkv_b will get automatically
    // run-time repacked if the rtr option is set.
    //
    const int32_t n_head        = hparams.n_head(0);
    std::vector<uint8_t> work_data;
    LLAMA_LOG_INFO("============ %s: need to compute %d wkv_b tensors\n", __func__, n_to_compute);
    for (int il = 1; il < n_layer; ++il) {
        // Somehow the number of heads is being defined as being per layer. Not sure why this is the
        // case, but for now we do not support strange models that have different numbers of heads
        // in different model layers.
        if ((int)hparams.n_head(il) != n_head) throw std::runtime_error("Unsupported configuration");
    }

    size_t context_size = ggml_tensor_overhead()*16*n_layer;

    ggml_init_params params{context_size, nullptr, true};
    auto ctx = ggml_init(params);
    auto graph = ggml_new_graph_custom(ctx, 8, false);

    std::vector<char> wk_buffer, wv_buffer;
    std::vector<char> tmp_buffer;
    for (int il = 0; il < n_layer; ++il) {
        auto& l = model.layers[il];
        if (l.wkv_b || !l.wk_b || !l.wv_b) continue;
        auto wk_b = *l.wk_b;
        auto wv_b = *l.wv_b;
        if (!ggml_backend_buffer_is_host(l.wk_b->buffer)) {
            auto nbytes = ggml_nbytes(l.wk_b);
            if (wk_buffer.size() < nbytes) wk_buffer.resize(nbytes);
            ggml_backend_tensor_get(l.wk_b, wk_buffer.data(), 0, nbytes);
            wk_b.data = wk_buffer.data();
        }
        if (!ggml_backend_buffer_is_host(l.wv_b->buffer)) {
            auto nbytes = ggml_nbytes(l.wv_b);
            if (wv_buffer.size() < nbytes) wv_buffer.resize(nbytes);
            ggml_backend_tensor_get(l.wv_b, wv_buffer.data(), 0, nbytes);
            wv_b.data = wv_buffer.data();
        }

        auto n_wk = ggml_nelements(&wk_b);
        auto n_wv = ggml_nelements(&wv_b);

        size_t tot_size = 0;
        if (wk_b.type != GGML_TYPE_F32) {
            tot_size += n_wk*sizeof(float);
        }
        tot_size += n_wk*sizeof(float); // ggml_cont(ctx, ggml_transpose(ctx, wk_b_used));
        if (wv_b.type != GGML_TYPE_F32) {
            tot_size += n_wv*sizeof(float);
        }
        tot_size += (n_wk + n_wv)*sizeof(float); // ggml_concat(ctx, wk_b_transposed, wv_b_used, 0);
        tot_size += (n_wk + n_wv)*sizeof(float); // ggml_cast(ctx, wkv_b_f32, new_type);

        if (tmp_buffer.size() < tot_size) tmp_buffer.resize(tot_size);

        auto ptr = tmp_buffer.data();

        auto wk_b_used = &wk_b;
        if (wk_b.type != GGML_TYPE_F32) {
            wk_b_used = ggml_cast(ctx, &wk_b, GGML_TYPE_F32);
            wk_b_used->data = ptr;
            ptr += ggml_nbytes(wk_b_used);
        }
        auto wk_b_transposed = ggml_cont(ctx, ggml_transpose(ctx, wk_b_used));
        wk_b_transposed->data = ptr;
        ptr += ggml_nbytes(wk_b_transposed);

        auto wv_b_used = &wv_b;
        if (wv_b.type != GGML_TYPE_F32) {
            wv_b_used = ggml_cast(ctx, &wv_b, GGML_TYPE_F32);
            wv_b_used->data = ptr;
            ptr += ggml_nbytes(wv_b_used);
        }

        auto wkv_b_f32_3d = ggml_concat(ctx, wk_b_transposed, wv_b_used, 1);
        wkv_b_f32_3d->data = ptr;
        ptr += ggml_nbytes(wkv_b_f32_3d);

        auto wkv_b_f32 = ggml_view_2d(ctx, wkv_b_f32_3d, wkv_b_f32_3d->ne[0], wkv_b_f32_3d->ne[1]*wkv_b_f32_3d->ne[2],
                wkv_b_f32_3d->nb[1], 0);

        auto new_type = wk_b.type == GGML_TYPE_BF16 && wv_b.type == GGML_TYPE_BF16 ? GGML_TYPE_BF16
                      : wk_b.type == GGML_TYPE_F16  && wv_b.type == GGML_TYPE_F16  ? GGML_TYPE_F16
                      : GGML_TYPE_Q8_0;

        auto wkv_b = ggml_cast(ctx, wkv_b_f32, new_type);
        wkv_b->data = ptr;
        ptr += ggml_nbytes(wkv_b);

        ggml_build_forward_expand(graph, wkv_b);

        auto plan = ggml_graph_plan(graph, std::thread::hardware_concurrency()/2);
        if (plan.work_size > work_data.size()) work_data.resize(plan.work_size);
        plan.work_data = work_data.data();

        auto status = ggml_graph_compute(graph, &plan);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Failed to compute wkv_b");

        auto name = std::string{"blk."} + std::to_string(il) + ".attn_kv_b.weight";

        l.computed_wkv_b = std::make_unique<ggml_tensor>(*wkv_b);
        l.computed_wkv_b->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_buffer_get_type(l.wk_b->buffer), ggml_nbytes(wkv_b));
        l.computed_wkv_b->data   = ggml_backend_buffer_get_base(l.computed_wkv_b->buffer);
        l.computed_wkv_b->op = GGML_OP_NONE; // we absolutely need to do this, else the backend will attempt to find the parents
                                             // of wkv_b, which no longer exist, and will therefore crash.
        for (int j = 0; j < GGML_MAX_SRC; ++j) l.computed_wkv_b->src[j] = nullptr;
        ggml_set_name(l.computed_wkv_b.get(), name.c_str());
        ggml_backend_buffer_set_usage(l.computed_wkv_b->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_set(l.computed_wkv_b.get(), wkv_b->data, 0, ggml_nbytes(wkv_b));
        if (ggml_backend_buffer_is_host(l.computed_wkv_b->buffer)) {
            iqk_modify_tensor(l.computed_wkv_b.get());
        }

        l.wkv_b = l.computed_wkv_b.get();
        model.tensors_by_name.push_back(std::make_pair(name, l.wkv_b));

        printf("Computed %s as %ld x %ld and stored in buffer %s\n", name.c_str(), wkv_b->ne[0], wkv_b->ne[1],
                    ggml_backend_buffer_name(l.computed_wkv_b->buffer));

        ggml_graph_clear(graph);
    }
    ggml_free(ctx);
}

// Backend (reg) enumeration
static bool striequals(const char* a, const char* b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

ggml_backend_t llama_context::ggml_backend_by_name(const char* name) {
    for (auto backend : backends) {
        const char* backend_name = ggml_backend_name(backend);
        if (striequals(backend_name, name)) {
            return backend;
        }
    }
    return nullptr;
}

static bool item_in_list(const std::vector<std::string>& devices, const char* name) {
    for (auto& device : devices) {
        if (striequals(device.c_str(), name)) {
            return true;
        }
    }
    return false;
}

static void ggml_backend_add_from_device(llama_context* ctx, ggml_backend_t backend) {
    const char* name = ggml_backend_name(backend);
    if (ctx->cparams.devices.size()) {
        if (item_in_list(ctx->cparams.devices, name)) {
            ctx->backends.push_back(backend);
        }
    } else {
        ctx->backends.push_back(backend);
    }
}

static bool is_model_split_supported(const llama_model & model) {
    static std::unordered_set<llm_arch> k_supported = {
        LLM_ARCH_LLAMA,
        LLM_ARCH_QWEN3MOE,
        LLM_ARCH_GLM4_MOE,
        LLM_ARCH_MISTRAL3,
        LLM_ARCH_COHERE2,
        LLM_ARCH_MIMO2,
        LLM_ARCH_QWEN3,
        LLM_ARCH_QWEN3VL,
        LLM_ARCH_HUNYUAN_MOE,
        LLM_ARCH_OPENAI_MOE,
        LLM_ARCH_ERNIE4_5_MOE,
    };
    auto it =  k_supported.find(model.arch);
    return it != k_supported.end();
}

// Returns false if cancelled by progress_callback
static bool llm_load_tensors(
        llama_model_loader & ml,
        llama_model & model,
        int n_gpu_layers,
        int mla_attn,
        enum llama_split_mode split_mode,
        int main_gpu,
        int max_gpu,
        const float * tensor_split,
        bool use_mlock,
        bool validate_quants,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    model.t_start_us = ggml_time_us();

    auto & hparams = model.hparams;

    if (split_mode == LLAMA_SPLIT_MODE_GRAPH || split_mode == LLAMA_SPLIT_MODE_ATTN) {
        if (!is_model_split_supported(model)) {
            LLAMA_LOG_WARN("\n=======================================================\n");
            LLAMA_LOG_WARN("Split mode 'graph' is not supported for this model\n");
            LLAMA_LOG_WARN("  => changing split mode to 'layer'\n");
            LLAMA_LOG_WARN("=======================================================\n\n");
            split_mode = LLAMA_SPLIT_MODE_LAYER;
        } else {
            if (model.arch == LLM_ARCH_MIMO2 && model.devices.size() > 4 && (max_gpu == 0 || max_gpu > 4)) {
                LLAMA_LOG_WARN("\n================================================================\n");
                LLAMA_LOG_WARN("Split mode 'graph' for Mimo2 does not work with more than 4 GPUs\n");
                LLAMA_LOG_WARN("  => setting max_gpu to 4\n");
                LLAMA_LOG_WARN("================================================================\n\n");
                max_gpu = 4;
            }
        }
    }

    model.split_mode   = split_mode;
    model.main_gpu     = main_gpu;
    model.max_gpu      = max_gpu;
    model.n_gpu_layers = n_gpu_layers;

    const int n_layer     = hparams.n_layer;
    const int i_gpu_start = std::max((int) hparams.n_layer - n_gpu_layers, (int) 0);
    bool use_mmap_buffer = true;

    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    model.buft_input = llama_default_buffer_type_cpu(true);

    model.buft_layer.resize(n_layer);

    // assign cpu layers
    for (int i = 0; i < i_gpu_start; ++i) {
        model.buft_layer[i] = llama_default_buffer_type_cpu(true);
    }

    if (int device_count = model.devices.size(); device_count > 1) {
        bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + device_count, [](float x) { return x == 0.0f; });
        std::vector<float> splits(device_count);
        if (all_zero) {
            // default split, by free memory
            for (int i = 0; i < device_count; ++i) {
                splits[i] = llama_get_device_memory(model, model.devices[i]);
            }
        } else {
            std::copy(tensor_split, tensor_split + device_count, splits.begin());
        }

        // sum and normalize the splits to get the split points
        float split_sum = 0.0f;
        for (int i = 0; i < device_count; ++i) {
            split_sum += splits[i];
            splits[i] = split_sum;
        }
        for (int i = 0; i < device_count; ++i) {
            splits[i] /= split_sum;
        }
        model.splits = std::move(splits);
    } else {
        model.splits = { 1.0f };
    }

    int device_count = model.splits.size();
    // assign the repeating layers to the devices according to the splits
    int act_gpu_layers = std::min(n_gpu_layers, (int)n_layer + 1);
    if (split_mode == LLAMA_SPLIT_MODE_LAYER) {

        for (int i = i_gpu_start; i < n_layer; ++i) {
            int layer_gpu = std::upper_bound(model.splits.begin(), model.splits.begin() + device_count, float(i - i_gpu_start)/act_gpu_layers) - model.splits.begin();
            model.buft_layer[i] = llama_default_buffer_type_offload(model, model.devices[layer_gpu]);
        }
        // assign the output layer
        if (n_gpu_layers > n_layer) {
            int layer_gpu = std::upper_bound(model.splits.begin(), model.splits.begin() + device_count, float(act_gpu_layers - 1)/act_gpu_layers) - model.splits.begin();
            model.buft_output = llama_default_buffer_type_offload(model, model.devices[layer_gpu]);
        } else {
            model.buft_output = llama_default_buffer_type_cpu(true);
        }
    } else {
        ggml_backend_buffer_type_t split_buft;
        if ((split_mode == LLAMA_SPLIT_MODE_GRAPH || split_mode == LLAMA_SPLIT_MODE_ATTN) && model.splits.size() > 1) {
            split_buft = llama_default_buffer_type_split(model, model.devices[main_gpu]);
            model.split_buft = split_buft;
        } else {
            // LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_LAYER in backends where it is not supported
            split_buft = llama_default_buffer_type_offload(model, model.devices[main_gpu]);
        }
        auto buft_layer = llama_default_buffer_type_offload(model, model.devices[main_gpu]);
        // assign the repeating layers
        for (int i = i_gpu_start; i < n_layer; ++i) {
            if (split_mode == LLAMA_SPLIT_MODE_ATTN) {
                int layer_gpu = std::upper_bound(model.splits.begin(), model.splits.begin() + device_count,
                        float(i - i_gpu_start)/act_gpu_layers) - model.splits.begin();
                model.buft_layer[i] = { split_buft, llama_default_buffer_type_offload(model, model.devices[layer_gpu]) };
                printf("Layer %d: assigning buft_layer to GPU %d\n", i, layer_gpu);
            } else {
                model.buft_layer[i] = { split_buft, buft_layer };
            }
        }
        // assign the output layer
        if (n_gpu_layers > n_layer) {
            model.buft_output = {
                split_buft,
                llama_default_buffer_type_offload(model, model.devices[main_gpu])
            };
        } else {
            model.buft_output = llama_default_buffer_type_cpu(true);
        }
    }

    auto cth = create_tensors_helper_interface::instance(ml, model);

    auto ctx_size = cth->get_ctx_size();
    auto & ctx_map = cth->get_ctx_map();

    LLAMA_LOG_INFO("%s: ggml ctx size = %7.2f MiB\n", __func__, model.ctxs.size()*ctx_size/1024.0/1024.0);

    if (hparams.n_expert > 0 && hparams.n_expert_used == 0) {
        throw std::runtime_error("model has expert layers but no expert layers are used");
    }

    use_mmap_buffer = cth->create_tensors();
    if (!use_mmap_buffer) {
        ml.use_mmap = false;
    }

    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &model.mlock_mmaps : nullptr, ml.use_thp);
    model.mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context *, llama_buf_map>> ctx_bufs;
    ctx_bufs.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    model.bufs.reserve(n_max_backend_buffer);

    for (auto & it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx              = it.second;

        llama_buf_map bufs;
        bufs.reserve(n_max_backend_buffer);

        // only the mmap region containing the tensors in the model is mapped to the backend buffer
        // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
        // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
        if (ml.use_mmap && use_mmap_buffer && (buft == llama_default_buffer_type_cpu(true) || buft == ggml_backend_cpu_buffer_type())) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                void * addr = nullptr;
                size_t first, last;
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr((char *) addr + first, last - first);
                if (buf == nullptr) {
                    throw std::runtime_error("unable to allocate backend CPU buffer");
                }
                model.bufs.push_back(buf);
                bufs.emplace(idx, buf);
#ifdef GGML_USE_CUDA
                if (n_layer >= n_gpu_layers) {
                    ggml_backend_cuda_register_host_buffer(
                        ggml_backend_buffer_get_base(buf),
                        ggml_backend_buffer_get_size(buf));
                }
#endif
            }
        }
#ifdef GGML_USE_METAL
        else if (ml.use_mmap && use_mmap_buffer && buft == ggml_backend_metal_buffer_type()) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                void * addr = nullptr;
                size_t first, last;
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                ggml_backend_buffer_t buf = ggml_backend_metal_buffer_from_ptr((char *) addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw std::runtime_error("unable to allocate backend metal buffer");
                }
                model.bufs.push_back(buf);
                bufs.emplace(idx, buf);
            }
        }
#endif
        else {
            int ntensor = 0;
            for (auto t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                ++ntensor;
            }
            if (ntensor > 0) {
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
                if (buf == nullptr) {
                    LLAMA_LOG_ERROR("Failed to allocate buffer type %s\n", ggml_backend_buft_name(buft));
                    throw std::runtime_error("unable to allocate backend buffer");
                }
                model.bufs.push_back(buf);
                if (use_mlock && ggml_backend_buffer_is_host(buf)) {
                    model.mlock_bufs.emplace_back(new llama_mlock);
                    auto & mlock_buf = model.mlock_bufs.back();
                    mlock_buf->init   (ggml_backend_buffer_get_base(buf));
                    mlock_buf->grow_to(ggml_backend_buffer_get_size(buf));
                }
                for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                    bufs.emplace(idx, buf);
                }
            }
        }

        if (bufs.empty()) {
            LLAMA_LOG_WARN("No tensors in buffer type %s\n", ggml_backend_buft_name(buft));
            continue;
            //throw std::runtime_error("failed to allocate buffer (1)");
        }

        for (auto & buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling -> ops that use a weight are preferably scheduled to the backend that contains the weight
            ggml_backend_buffer_set_usage(buf.second, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_bufs.emplace_back(ctx, bufs);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int) hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers       = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements
    for (ggml_backend_buffer_t buf : model.bufs) {
        LLAMA_LOG_INFO("%s: %10s buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (ggml_context * ctx : model.ctxs) {
        for (auto * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
            model.tensors_by_name.emplace_back(ggml_get_name(cur), cur);
        }
    }

    // load tensor data
    for (auto & it : ctx_bufs) {
        ggml_context * ctx = it.first;
        auto & bufs = it.second;
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &model.mlock_mmaps : NULL, progress_callback, progress_callback_user_data)) {
            return false;
        }
    }

    if (model.arch == LLM_ARCH_DEEPSEEK2) {
        llm_prepare_mla(model, mla_attn);
    }

    if (use_mmap_buffer) {
        for (auto & mapping : ml.mappings) {
            model.mappings.emplace_back(std::move(mapping));
        }
    }

    if (!ml.use_mmap) {
        int n_modified = 0;
        for (auto& it : model.tensors_by_name) {
            if (ggml_backend_buffer_is_host(it.second->buffer)) {
                if (iqk_modify_tensor(it.second)) ++n_modified;
            }
        }
        if (n_modified > 0) printf("============ Modified %d tensors\n", n_modified);
    }

    if (validate_quants) {
        int nbad = 0;
        for (auto& it : model.tensors_by_name) {
            if (ggml_backend_buffer_is_host(it.second->buffer)) {
                if (!iqk_validate_tensor(it.second)) ++nbad;
            }
        }
        if (nbad > 0) {
            LLAMA_LOG_ERROR("Found %d bad tensors in model\n", nbad);
            throw std::runtime_error("Bad tensors in model");
        }
    }

    if (!ml.use_mmap && ml.repack_tensors) {
        int n_repacked = 0;
        for (auto& it : model.tensors_by_name) {
            if (ggml_backend_buffer_is_host(it.second->buffer)) {
                auto orig_type = it.second->type;
                if (it.second->view_src) continue;
                iqk_repack_tensor(it.second);
                if (it.second->type != orig_type) ++n_repacked;
            }
        }
        if (n_repacked > 0) printf("============ Repacked %d tensors\n", n_repacked);
    }

    if (model.arch == LLM_ARCH_BITNET) {
        auto set_scale = [] (ggml_tensor * w, ggml_tensor * s) {
            if (!s) {
                float one = 1;
                std::memcpy(w->op_params, &one, sizeof(one));
                return;
            }
            float scale = 1;
            if (ggml_backend_buffer_is_host(s->buffer)) {
                scale = *(const float *)s->data;
            } else {
                ggml_backend_tensor_get(s, &scale, 0, sizeof(float));
            }
            std::memcpy(w->op_params, &scale, sizeof(scale));
        };
        for (auto& l : model.layers) {
            set_scale(l.ffn_up, l.ffn_up_scale);
            set_scale(l.ffn_gate, l.ffn_gate_scale);
            set_scale(l.ffn_down, l.ffn_down_scale);
            set_scale(l.wq, l.wq_scale);
            set_scale(l.wk, l.wk_scale);
            set_scale(l.wv, l.wv_scale);
            set_scale(l.wo, l.wo_scale);
        }
    }

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_time_us() - model.t_start_us;
    return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, llama_model & model, llama_model_params & params) {
    try {
        llama_model_loader ml(fname, params.use_mmap, params.check_tensors,
                params.repack_tensors, params.use_thp, params.merge_qkv, params.merge_up_gate_exps,
                params.kv_overrides, params.tensor_buft_overrides);

        model.hparams.vocab_only = params.vocab_only;

        try {
            llm_load_arch(ml, model);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            llm_load_hparams(ml, model);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            LLM_KV kv(model.arch);
            model.vocab.load(ml, kv);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        llm_load_print_meta(ml, model);

        if (model.vocab.get_type() != LLAMA_VOCAB_TYPE_NONE &&
            model.hparams.n_vocab != model.vocab.n_tokens()) {
            throw std::runtime_error("vocab size mismatch");
        }

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

#ifdef GGML_USE_KOMPUTE
        if (params.n_gpu_layers > 0 && (
            !(model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_FALCON)
            || !(
                model.ftype == LLAMA_FTYPE_ALL_F32 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_F16 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_BF16 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ||
                model.ftype == LLAMA_FTYPE_MOSTLY_Q4_1
            )
        )) {
            // TODO(cebtenzzre): propagate this error outside of llama_load_model_from_file
            LLAMA_LOG_WARN("%s: disabling Kompute due to unsupported model arch or quantization\n", __func__);
            params.n_gpu_layers = 0;
        }
#endif

        if (!llm_load_tensors(
            ml, model, params.n_gpu_layers, params.mla, params.split_mode,  params.main_gpu, params.max_gpu, params.tensor_split,
            params.use_mlock, params.validate_quants,
            params.progress_callback, params.progress_callback_user_data
        )) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

//
// llm_build
//

static void llama_set_k_shift(llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_K_shift->buffer));

    int32_t * data = (int32_t *) lctx.inp_K_shift->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].delta;
    }
}

static void llama_set_s_copy(llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_s_copy->buffer));

    int32_t * data = (int32_t *) lctx.inp_s_copy->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].src;
    }
}

static int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }
    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
    return relative_bucket;
}

static void llama_set_inputs(llama_context & lctx, const llama_batch & batch) {
    //
    // set input data
    //

    const auto & hparams = lctx.model.hparams;
    const auto & cparams = lctx.cparams;
    const auto & kv_self = lctx.kv_self;

    if (batch.token) {
#if IK_PRINT_TIMING == 2
        auto tim1 = ggml_time_us();
#endif
        const int64_t n_tokens = batch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_tokens, batch.token, 0, n_tokens*ggml_element_size(lctx.inp_tokens));
#if IK_PRINT_TIMING == 2
        auto tim2 = ggml_time_us();
        printf("set_inputs(token): %d us\n", int(tim2-tim1));
#endif
    }

    if (batch.embd) {
#if IK_PRINT_TIMING == 2
        auto tim1 = ggml_time_us();
#endif
        const int64_t n_embd   = hparams.n_embd;
        const int64_t n_tokens = batch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_embd, batch.embd, 0, n_tokens*n_embd*ggml_element_size(lctx.inp_embd));
#if IK_PRINT_TIMING == 2
        auto tim2 = ggml_time_us();
        printf("set_inputs(embd): %d us\n", int(tim2-tim1));
#endif
    }

    if (batch.pos && lctx.inp_pos) {
#if IK_PRINT_TIMING == 2
        auto tim1 = ggml_time_us();
#endif
        const int64_t n_tokens = batch.n_tokens;
        const int n_pos_per_embd =  hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
        if (batch.token && n_pos_per_embd == 4) {
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = batch.pos[i];
                pos_data[    n_tokens + i] = batch.pos[i];
                pos_data[2 * n_tokens + i] = batch.pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(lctx.inp_pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(lctx.inp_pos));
        } else {
            ggml_backend_tensor_set(lctx.inp_pos, batch.pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(lctx.inp_pos));
        }
#if IK_PRINT_TIMING == 2
        auto tim2 = ggml_time_us();
        printf("set_inputs(pos): %d us\n", int(tim2-tim1));
#endif
    }

    if (lctx.inp_pos && lctx.inp_scale) {
#if IK_PRINT_TIMING == 2
        auto tim1 = ggml_time_us();
#endif
        int n_tokens = batch.n_tokens;
        GGML_ASSERT(ggml_nelements(lctx.inp_scale) >= n_tokens);
        if (int(lctx.scale_data.size()) < n_tokens) lctx.scale_data.resize(n_tokens);
        int n_pos_per_token = 1;
        for (int i = 0; i < n_tokens; ++i) {
            lctx.scale_data[i] = std::log(std::floor((batch.pos[i] + 1.0f) / hparams.n_attn_temp_floor_scale) + 1.0f) * hparams.f_attn_temp_scale + 1.0f;
        }
        ggml_backend_tensor_set(lctx.inp_scale, lctx.scale_data.data(), 0, n_tokens*n_pos_per_token*ggml_element_size(lctx.inp_scale));
#if IK_PRINT_TIMING == 2
        auto tim2 = ggml_time_us();
        printf("set_inputs(scale): %d us\n", int(tim2-tim1));
#endif
    }

    if (hparams.causal_attn || cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
#if IK_PRINT_TIMING == 2
        auto tim1 = ggml_time_us();
#endif
        GGML_ASSERT(lctx.inp_out_ids && "every model that can must skip unused outputs");
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_out_ids->buffer));
        int32_t * data = (int32_t *) lctx.inp_out_ids->data;

        if (lctx.n_outputs == n_tokens) {
            for (int i = 0; i < n_tokens; ++i) {
                data[i] = i;
            }
        } else if (batch.logits) {
            int32_t n_outputs = 0;
            for (int i = 0; i < n_tokens; ++i) {
                if (batch.logits[i]) {
                    data[n_outputs++] = i;
                }
            }
            // the graph needs to have been passed the correct number of outputs
            GGML_ASSERT(lctx.n_outputs == n_outputs);
        } else if (lctx.n_outputs == 1) {
            // only keep last output
            data[0] = n_tokens - 1;
        } else {
            GGML_ASSERT(lctx.n_outputs == 0);
        }
#if IK_PRINT_TIMING == 2
        auto tim2 = ggml_time_us();
        printf("set_inputs(outputs): %d us\n", int(tim2-tim1));
#endif
    }

    GGML_ASSERT(
        // (!a || b) is a logical implication (a -> b)
        // !hparams.causal_attn -> !cparams.causal_attn
        (hparams.causal_attn || !cparams.causal_attn) &&
        "causal attention is not supported by this model"
    );

    if (lctx.inp_KQ_mask || lctx.inp_KQ_mask_swa) {
#if IK_PRINT_TIMING == 2
        auto tim1 = ggml_time_us();
#endif
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn && !lctx.is_encoding) {
            const int64_t n_kv     = kv_self.n;
            const int64_t n_tokens = batch.n_tokens;


            float * data     = nullptr;
            float * data_swa = nullptr;
            ggml_half * data_f16     = nullptr;
            ggml_half * data_swa_f16 = nullptr;

            if (lctx.inp_KQ_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));
                if (cparams.flash_attn) {
                    data_f16 = (ggml_half *)lctx.inp_KQ_mask->data;
                } else {
                    data = (float *) lctx.inp_KQ_mask->data;
                }
            }

            if (lctx.inp_KQ_mask_swa) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask_swa->buffer));
                if (cparams.flash_attn) {
                    data_swa_f16 = (ggml_half *) lctx.inp_KQ_mask_swa->data;
                } else {
                    data_swa = (float *) lctx.inp_KQ_mask_swa->data;
                }
            }

            auto noalibi_f16 = [&lctx, &hparams, n_kv, data_f16, data_swa_f16] (int j, llama_pos pos, llama_seq_id seq_id, int first, int last) {
                ggml_half h_inf  = ggml_fp32_to_fp16(-INFINITY);
                ggml_half h_zero = ggml_fp32_to_fp16(0.f);
                for (int i = first; i < last; ++i) {
                    ggml_half h = !lctx.kv_self.cells[i].has_seq_id(seq_id) || lctx.kv_self.cells[i].pos > pos ? h_inf : h_zero;
                    if (data_f16) data_f16[j*n_kv + i] = h;
                    if (data_swa_f16) {
                        if (h != h_inf) {
                            if (hparams.n_attn_chunk) {
                                llama_pos pos_chunk_start = (pos / hparams.n_attn_chunk) * hparams.n_attn_chunk;
                                if (lctx.kv_self.cells[i].pos < pos_chunk_start || pos < pos_chunk_start) {
                                    h = h_inf;
                                }
                            } else {
                                if (pos - lctx.kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    h = h_inf;
                                }
                            }
                        }
                        data_swa_f16[j*n_kv + i] = h;
                    }
                }
            };

            if (n_kv >= 1024 && n_tokens >= 32) {
                int n_thread = std::max(1, int(std::thread::hardware_concurrency()/2));
                int npt = (n_kv + n_thread - 1)/n_thread;
                auto compute = [&batch, &lctx, &hparams, &cparams, &noalibi_f16, n_tokens, n_kv, npt, data, data_swa, data_f16, data_swa_f16] (int ith) {
                    int first = ith * npt;
                    int last  = std::min(int(n_kv), first + npt);
                    if (last <= first) return;
                    for (int j = 0; j < n_tokens; ++j) {
                        const llama_pos    pos    = batch.pos[j];
                        const llama_seq_id seq_id = batch.seq_id[j][0];

                        if (!hparams.use_alibi && cparams.flash_attn) {
                            noalibi_f16(j, pos, seq_id, first, last);
                            continue;
                        }

                        for (int i = first; i < last; ++i) {
                            float f;
                            if (!lctx.kv_self.cells[i].has_seq_id(seq_id) || lctx.kv_self.cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(lctx.kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[j*n_kv + i] = f;
                            }
                            if (data_f16) {
                                data_f16[j*n_kv + i] = ggml_fp32_to_fp16(f);
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa || data_swa_f16) {
                                if (f > -INFINITY) {
                                    if (hparams.n_attn_chunk) {
                                        llama_pos pos_chunk_start = (pos / hparams.n_attn_chunk) * hparams.n_attn_chunk;
                                        if (lctx.kv_self.cells[i].pos < pos_chunk_start || pos < pos_chunk_start) {
                                            f = -INFINITY;
                                        }
                                    } else {
                                        if (pos - lctx.kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                            f = -INFINITY;
                                        }
                                    }
                                }
                                if (data_swa) {
                                    data_swa[j*n_kv + i] = f;
                                }
                                if (data_swa_f16) {
                                    data_swa_f16[j*n_kv + i] = ggml_fp32_to_fp16(f);
                                }
                            }
                        }
                    }
                };
                std::vector<std::thread> workers(n_thread-1);
                int it = 0;
                for (auto& w : workers) w = std::thread(compute, it++);
                compute(it);
                for (auto& w : workers) w.join();
                int64_t n_tokens_padded = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
                if (n_tokens_padded > n_tokens) {
                    if (data) {
                        std::fill(data + int64_t(n_tokens)*n_kv, data + n_tokens_padded*n_kv, -INFINITY);
                    }
                    if (data_f16) {
                        ggml_half h_inf = ggml_fp32_to_fp16(-INFINITY);
                        std::fill(data_f16 + int64_t(n_tokens)*n_kv, data_f16 + n_tokens_padded*n_kv, h_inf);
                    }
                    if (data_swa) {
                        std::fill(data_swa + int64_t(n_tokens)*n_kv, data_swa + n_tokens_padded*n_kv, -INFINITY);
                    }
                    if (data_swa_f16) {
                        ggml_half h_inf = ggml_fp32_to_fp16(-INFINITY);
                        std::fill(data_swa_f16 + int64_t(n_tokens)*n_kv, data_swa_f16 + n_tokens_padded*n_kv, h_inf);
                    }
                }
            }
            else {

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the batch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    const llama_pos    pos    = batch.pos[j];
                    const llama_seq_id seq_id = batch.seq_id[j][0];

                    if (!hparams.use_alibi && cparams.flash_attn) {
                        noalibi_f16(j, pos, seq_id, 0, n_kv);
                        continue;
                    }

                    for (int i = 0; i < n_kv; ++i) {
                        float f;
                        if (!lctx.kv_self.cells[i].has_seq_id(seq_id) || lctx.kv_self.cells[i].pos > pos) {
                            f = -INFINITY;
                        } else {
                            if (hparams.use_alibi) {
                                f = -std::abs(lctx.kv_self.cells[i].pos - pos);
                            } else {
                                f = 0.0f;
                            }
                        }

                        if (data) {
                            data[h*(n_kv*n_tokens) + j*n_kv + i] = f;
                        }
                        if (data_f16) {
                            data_f16[h*(n_kv*n_tokens) + j*n_kv + i] = ggml_fp32_to_fp16(f);
                        }

                        // may need to cut off old tokens for sliding window
                        if (data_swa || data_swa_f16) {
                            if (hparams.n_attn_chunk) {
                                llama_pos pos_chunk_start = (pos / hparams.n_attn_chunk) * hparams.n_attn_chunk;
                                if (lctx.kv_self.cells[i].pos < pos_chunk_start || pos < pos_chunk_start) {
                                    f = -INFINITY;
                                }
                            } else {
                                if (pos - kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                            }
                            if (data_swa) {
                                data_swa[h*(n_kv*n_tokens) + j*n_kv + i] = f;
                            }
                            if (data_swa_f16) {
                                data_swa_f16[h*(n_kv*n_tokens) + j*n_kv + i] = ggml_fp32_to_fp16(f);
                            }
                        }
                    }
                }

                int64_t n_tokens_padded = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
                if (n_tokens_padded > n_tokens) {
                    if (data) {
                        std::fill(data + int64_t(n_tokens)*n_kv, data + n_tokens_padded*n_kv, -INFINITY);
                    }
                    if (data_f16) {
                        ggml_half h_inf = ggml_fp32_to_fp16(-INFINITY);
                        std::fill(data_f16 + int64_t(n_tokens)*n_kv, data_f16 + n_tokens_padded*n_kv, h_inf);
                    }
                    if (data_swa) {
                        std::fill(data_swa + int64_t(n_tokens)*n_kv, data_swa + n_tokens_padded*n_kv, -INFINITY);
                    }
                    if (data_swa_f16) {
                        ggml_half h_inf = ggml_fp32_to_fp16(-INFINITY);
                        std::fill(data_swa_f16 + int64_t(n_tokens)*n_kv, data_swa_f16 + n_tokens_padded*n_kv, h_inf);
                    }
                }
            }
            }
#if IK_PRINT_TIMING == 2
            auto tim2 = ggml_time_us();
            printf("set_inputs(mask1): %d us\n", int(tim2-tim1));
#endif
        } else {
            // when using kv cache, the mask needs to match the kv cache size
            const int64_t n_tokens = batch.n_tokens;
            const int64_t n_stride = hparams.causal_attn && !lctx.is_encoding ? kv_self.n : n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));

            float * data = (float *) lctx.inp_KQ_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    const llama_seq_id seq_id = batch.seq_id[j][0];

                    for (int i = 0; i < n_tokens; ++i) {
                        float f = -INFINITY;
                        for (int s = 0; s < batch.n_seq_id[i]; ++s) {
                            if (batch.seq_id[i][s] == seq_id) {
                                if (hparams.use_alibi) {
                                    f = -std::abs(batch.pos[i] - batch.pos[j]);
                                } else {
                                    f = 0.0f;
                                }
                                break;
                            }
                        }

                        data[h*(n_tokens*n_tokens) + j*n_stride + i] = f;
                    }

                    for (int i = n_tokens; i < n_stride; ++i) {
                        data[h*(n_tokens*n_tokens) + j*n_stride + i] = -INFINITY;
                    }
                }
            }
#if IK_PRINT_TIMING == 2
            auto tim2 = ggml_time_us();
            printf("set_inputs(mask2): %d us\n", int(tim2-tim1));
#endif
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(lctx.inp_mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_mean->buffer));

        float * data = (float *) lctx.inp_mean->data;
        memset(lctx.inp_mean->data, 0, n_tokens * n_tokens * ggml_element_size(lctx.inp_mean));

        std::vector<uint64_t> sum(n_tokens, 0);
        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];

            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += 1;
        }

        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];
            data[seq_id*n_tokens + i] = div[seq_id];
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_CLS) {
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(lctx.inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_cls->buffer));

        uint32_t * data = (uint32_t *) lctx.inp_cls->data;
        memset(lctx.inp_cls->data, 0, n_tokens * ggml_element_size(lctx.inp_cls));

        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];
            const llama_pos    pos    = batch.pos[i];

            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == CLS");

            if (pos == 0) {
                data[seq_id] = i;
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST) {
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(lctx.inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_cls->buffer));

        uint32_t * data = (uint32_t *) lctx.inp_cls->data;
        memset(lctx.inp_cls->data, 0, n_tokens * ggml_element_size(lctx.inp_cls));

        std::vector<int> last_pos(n_tokens, -1);
        std::vector<int> last_row(n_tokens, -1);

        for (int i = 0; i < n_tokens; ++i) {
            const llama_seq_id seq_id = batch.seq_id[i][0];
            const llama_pos    pos    = batch.pos[i];

            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == LAST");

            if (pos >= last_pos[seq_id]) {
                last_pos[seq_id] = pos;
                last_row[seq_id] = i;
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            if (last_row[i] >= 0) {
                data[i] = last_row[i];
            }
        }
    }

    if (kv_self.recurrent) {
        const int64_t n_kv = kv_self.n;

        if (lctx.inp_s_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_s_mask->buffer));
            float * data = (float *) lctx.inp_s_mask->data;

            // states which are not affected by the current batch are left untouched
            for (int i = 0; i < n_kv; ++i) {
                llama_seq_id    seq_id       = i + lctx.kv_self.head;
                llama_kv_cell & kv_cell      = lctx.kv_self.cells[seq_id];
                bool            has_self_seq = kv_cell.has_seq_id(seq_id);

                data[i] = (float) has_self_seq;

                // ensure current sequences will be kept
                if (!has_self_seq && kv_cell.pos >= 0) {
                    kv_cell.seq_id.insert(seq_id);
                }
            }
        }
        // For Mamba (and other recurrent architectures),
        // update the correct state(s)/sequence(s) for each token of the batch.
        // Like with the KQ_mask, if a token in the batch has multiple sequences,
        // they are assumed to be equivalent (not here, but in ggml_ssm_scan and ggml_ssm_conv).
        if (lctx.inp_s_seq) {
            const int64_t n_tokens = batch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_s_seq->buffer));
            int32_t * data = (int32_t *) lctx.inp_s_seq->data;

            for (int j = 0; j < n_tokens; ++j) {
                const int32_t n_seq = batch.n_seq_id[j];
                GGML_ASSERT(0 < n_seq); // a token should be part of at least 1 sequence

                for (int i = 0; i < n_kv; ++i) {
                    if (i < n_seq) {
                        // for this type of model, the head is the minimum seq_id of the batch
                        data[j*n_kv + i] = batch.seq_id[j][i] - kv_self.head;
                    } else {
                        data[j*n_kv + i] = -1;
                    }
                }
            }
        }
    }

    if (lctx.inp_pos_bucket) {
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_pos_bucket->buffer));

        int32_t * data = (int32_t *) lctx.inp_pos_bucket->data;

        if (!lctx.is_encoding) {
            const int64_t n_kv = kv_self.n;
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_kv; ++i) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(lctx.kv_self.cells[i].pos, batch.pos[j], hparams.n_rel_attn_bkts, lctx.is_encoding);
                    }
                }
            }
        } else {
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_tokens; ++i) {
                        data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(batch.pos[i], batch.pos[j], hparams.n_rel_attn_bkts, lctx.is_encoding);
                    }
                }
            }
        }
    }

    if (!lctx.is_encoding && lctx.inp_embd_enc) {
        assert(lctx.inp_embd_enc->type == GGML_TYPE_F32);
        assert((size_t) ggml_nelements(lctx.inp_embd_enc) == lctx.embd_enc.size());

        ggml_backend_tensor_set(lctx.inp_embd_enc, lctx.embd_enc.data(), 0, ggml_nbytes(lctx.inp_embd_enc));
    }

    if (!lctx.is_encoding && lctx.inp_KQ_mask_cross) {
        const int64_t n_output_enc = lctx.embd_enc.size() / hparams.n_embd;
        const int64_t n_tokens = batch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask_cross->buffer));

        float * data = (float *) lctx.inp_KQ_mask_cross->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_output_enc; ++i) {
                    float f = -INFINITY;
                    for (int s = 0; s < batch.n_seq_id[j]; ++s) {
                        const llama_seq_id seq_id = batch.seq_id[j][s];
                        if (lctx.seq_ids_enc[i].find(seq_id) != lctx.seq_ids_enc[i].end()) {
                            f = 0.0f;
                        }
                    }
                    data[h*(n_output_enc*n_tokens) + j*n_output_enc + i] = f;
                }
            }

            for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (int j = 0; j < n_output_enc; ++j) {
                    data[h*(n_output_enc*n_tokens) + i*n_output_enc + j] = -INFINITY;
                }
            }
        }
    }
}

// Make sure enough space is available for outputs.
// Returns max number of outputs for which space was reserved.
static size_t llama_output_reserve(llama_context & lctx, size_t n_outputs) {
    const auto & cparams = lctx.cparams;
    const auto & hparams = lctx.model.hparams;

    const size_t n_outputs_max = std::max(n_outputs, (size_t) cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = hparams.n_vocab;
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool has_logits = !cparams.embeddings;
    const bool has_embd   =  lctx.is_encoding || (cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE));

    const size_t logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    const size_t embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (lctx.output_ids.empty()) {
        // init, never resized afterwards
        lctx.output_ids.resize(n_batch);
    }

    const size_t prev_size = lctx.buf_output ? ggml_backend_buffer_get_size(lctx.buf_output) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!lctx.buf_output || prev_size < new_size) {
        if (lctx.buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            ggml_backend_buffer_free(lctx.buf_output);
            lctx.buf_output = nullptr;
            lctx.logits = nullptr;
            lctx.embd = nullptr;
        }

        lctx.buf_output = ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), new_size);
        if (lctx.buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(lctx.buf_output);

    lctx.logits = has_logits ? output_base               : nullptr;
    lctx.embd   = has_embd   ? output_base + logits_size : nullptr;

    lctx.output_size = n_outputs_max;
    lctx.logits_size = logits_size;
    lctx.embd_size   = embd_size;

    // set all ids as invalid (negative)
    std::fill(lctx.output_ids.begin(), lctx.output_ids.end(), -1);

    ggml_backend_buffer_clear(lctx.buf_output, 0);

    lctx.n_outputs = 0;

    return n_outputs_max;
}


static void llama_graph_compute(
        llama_context & lctx,
          ggml_cgraph * gf,
                  int   n_threads) {
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(lctx.backend_metal)) {
        ggml_backend_metal_set_n_cb(lctx.backend_metal, n_threads);
    }
#endif

    if (lctx.backend_cpu != nullptr) {
        ggml_backend_cpu_set_n_threads(lctx.backend_cpu, n_threads);
        ggml_backend_cpu_set_abort_callback(lctx.backend_cpu, lctx.abort_callback, lctx.abort_callback_data);
    }
#ifdef GGML_USE_BLAS
    if (lctx.backend_blas != nullptr) {
        ggml_backend_blas_set_n_threads(lctx.backend_blas, n_threads);
    }
#endif

    ggml_backend_sched_graph_compute_async(lctx.sched, gf);

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(lctx.sched));
}

// decode a batch of tokens by evaluating the transformer
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
static int llama_decode_internal(
         llama_context & lctx,
           llama_batch   batch_all) { // TODO: rename back to batch

    lctx.is_encoding = false;
    const uint32_t n_tokens_all = batch_all.n_tokens;

    if (n_tokens_all == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0", __func__);
        return -1;
    }
#if IK_PRINT_TIMING > 2
    printf("===== %s: %ld\n", __func__, ggml_time_us());
#endif

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    GGML_ASSERT((!batch_all.token && batch_all.embd) || (batch_all.token && !batch_all.embd)); // NOLINT

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (lctx.t_compute_start_us == 0) {
        lctx.t_compute_start_us = ggml_time_us();
    }
    lctx.n_queued_tokens += n_tokens_all;

    auto & kv_self = lctx.kv_self;

    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_vocab = hparams.n_vocab;

    uint32_t n_outputs = 0;
    uint32_t n_outputs_prev = 0;

    const auto n_ubatch = cparams.n_ubatch;

    // TODO: simplify or deprecate
    std::vector<llama_pos> pos;
    std::vector<int32_t>                   n_seq_id;
    std::vector<llama_seq_id *>            seq_id_arr;
    std::vector<std::vector<llama_seq_id>> seq_id;

    // this indicates we are doing pooled embedding, so we ignore batch.logits and output all tokens
    const bool embd_pooled = cparams.embeddings && cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;

    // count outputs
    if (batch_all.logits && !embd_pooled) {
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            n_outputs += batch_all.logits[i] != 0;
        }
    } else if (lctx.logits_all || embd_pooled) {
        n_outputs = n_tokens_all;
    } else {
        // keep last output only
        n_outputs = 1;
    }

    // reserve output buffer
    if (llama_output_reserve(lctx, n_outputs) < n_outputs) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_outputs);
        return -2;
    };

    // set output mappings
    if (batch_all.logits) {
        int32_t i_logits = 0;
        for (uint32_t i = 0; i < n_tokens_all; ++i) {
            if (batch_all.logits[i]) {
                lctx.output_ids[i] = i_logits++;
            }
        }
    } else {
        for (uint32_t i = 0; i < n_outputs; ++i) {
            lctx.output_ids[i] = i;
        }
    }

    for (uint32_t cur_token = 0; cur_token < n_tokens_all; cur_token += n_ubatch) {
#if IK_PRINT_TIMING
        auto tim1 = ggml_time_us();
#endif
        const uint32_t n_tokens = std::min(n_ubatch, n_tokens_all - cur_token);
        llama_batch u_batch = {
            /* .n_tokens   = */ (int32_t) n_tokens,
            /* .token      = */ batch_all.token     ? batch_all.token    + cur_token        : nullptr,
            /* .embd       = */ batch_all.embd      ? batch_all.embd     + cur_token*n_embd : nullptr,
            /* .pos        = */ batch_all.pos       ? batch_all.pos      + cur_token        : nullptr,
            /* .n_seq_id   = */ batch_all.n_seq_id  ? batch_all.n_seq_id + cur_token        : nullptr,
            /* .seq_id     = */ batch_all.seq_id    ? batch_all.seq_id   + cur_token        : nullptr,
            /* .logits     = */ batch_all.logits    ? batch_all.logits   + cur_token        : nullptr,
            /* .all_pos_0  = */ batch_all.all_pos_0 + (llama_pos) cur_token*batch_all.all_pos_1,
            /* .all_pos_1  = */ batch_all.all_pos_1,
            /* .all_seq_id = */ batch_all.all_seq_id,
        };

        // count the outputs in this u_batch
        {
            int32_t n_outputs_new = 0;

            if (u_batch.logits && !embd_pooled) {
                for (uint32_t i = 0; i < n_tokens; i++) {
                    n_outputs_new += u_batch.logits[i] != 0;
                }
            } else if (n_outputs == n_tokens_all) {
                n_outputs_new = n_tokens;
            } else {
                // keep last output only
                if (cur_token + n_tokens >= n_tokens_all) {
                    n_outputs_new = 1;
                }
            }

            // needs to happen before the graph is built
            lctx.n_outputs = n_outputs_new;
        }

        int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
        GGML_ASSERT(n_threads > 0);

        // helpers for smoother batch API transition
        // after deprecating the llama_eval calls, these will be removed
        if (u_batch.pos == nullptr) {
            pos.resize(n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                pos[i] = u_batch.all_pos_0 + i*u_batch.all_pos_1;
            }

            u_batch.pos = pos.data();
        }

        if (u_batch.seq_id == nullptr) {
            n_seq_id.resize(n_tokens);
            seq_id.resize(n_tokens);
            seq_id_arr.resize(n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                n_seq_id[i] = 1;
                seq_id[i].resize(1);
                seq_id[i][0] = u_batch.all_seq_id;
                seq_id_arr[i] = seq_id[i].data();
            }

            u_batch.n_seq_id = n_seq_id.data();
            u_batch.seq_id = seq_id_arr.data();
        }

        // non-causal masks do not use the KV cache
        if (hparams.causal_attn) {
            int32_t ret = llama_kv_cache_update(&lctx);
            if (ret != 0) {
                return ret;
            }

            // if we have enough unused cells before the current head ->
            //   better to start searching from the beginning of the cache, hoping to fill it
            if (kv_self.head > kv_self.used + 2*n_tokens) {
                kv_self.head = 0;
            }

            if (!llama_kv_cache_find_slot(kv_self, u_batch)) {
                return 1;
            }

            if (!kv_self.recurrent) {
                // a heuristic, to avoid attending the full cache if it is not yet utilized
                // after enough generations, the benefit from this heuristic disappears
                // if we start defragmenting the cache, the benefit from this will be more important
                const uint32_t pad = llama_kv_cache_get_padding(cparams);
                kv_self.n = std::min(kv_self.size, std::max(pad, GGML_PAD(llama_kv_cache_cell_max(kv_self), pad)));
                //kv_self.n = llama_kv_cache_cell_max(kv_self);
            }
        }
#if IK_PRINT_TIMING
        auto tim2 = ggml_time_us();
        printf("prelude(...): %d us\n", int(tim2-tim1));
#endif


        //if (n_tokens_all == 1) {
        //    printf("================= %s\n", __func__);
        //    printf("    all_pos_0 = %d, all_pos_1 = %d, all_seq_id = %d\n", batch_all.all_pos_0, batch_all.all_pos_1, batch_all.all_seq_id);
        //    printf("    embd = %p, logits = %p, token = %p\n", (const void *)batch_all.embd, (const void *)batch_all.logits, (const void *)batch_all.token);
        //    printf("    n_outputs = %d, kv_self.n = %d\n", n_outputs, kv_self.n);
        //}
        //printf("kv_self.n = %5d, kv_self.used = %5d, kv_self.head = %5d\n", kv_self.n, kv_self.used, kv_self.head);

#if IK_PRINT_TIMING
        tim1 = ggml_time_us();
#endif
        ggml_cgraph * gf = nullptr;
        if (!lctx.can_reuse_graph(u_batch)) {
            lctx.reset_scheduler();
            ggml_backend_sched_set_eval_callback(lctx.sched, lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);
#if IK_PRINT_TIMING
            tim2 = ggml_time_us();
            printf("sched_reset(...): %d us\n", int(tim2-tim1));
#endif

#if IK_PRINT_TIMING
            tim1 = ggml_time_us();
#endif
            gf = llm_build_context::llama_build_graph(lctx, u_batch, false);
#if IK_PRINT_TIMING
            tim2 = ggml_time_us();
            printf("build_graph(...): %d us\n", int(tim2-tim1));
#endif

#if IK_PRINT_TIMING
            tim1 = ggml_time_us();
#endif
            ggml_backend_sched_alloc_graph(lctx.sched, gf);
#if IK_PRINT_TIMING
            tim2 = ggml_time_us();
            printf("sched_alloc_graph(...): %d us\n", int(tim2-tim1));
#endif
            if (u_batch.n_tokens == 1 && u_batch.embd == nullptr && lctx.cparams.graph_reuse) {
                lctx.prev = std::make_unique<llama_context::Prev>(llama_context::Prev{
                        (int)u_batch.all_seq_id, (int)lctx.n_outputs, (int)lctx.kv_self.n, gf});
            }
        } else {
            //printf("Reusing graph\n");
            gf = lctx.prev->graph;
        }

        // the output is always the last tensor in the graph
        struct ggml_tensor * res  = gf->nodes[gf->n_nodes - 1];
        struct ggml_tensor * embd = gf->nodes[gf->n_nodes - 2];

        if (lctx.n_outputs == 0) {
            // no output
            res  = nullptr;
            embd = nullptr;
        } else if (cparams.embeddings) {
            res  = nullptr; // do not extract logits for embedding case
            embd = nullptr;
            for (int i = gf->n_nodes - 1; i >= 0; --i) {
                if (strcmp(gf->nodes[i]->name, "result_embd_pooled") == 0) {
                    embd = gf->nodes[i];
                    break;
                }
            }
            GGML_ASSERT(embd != nullptr && "missing embeddings tensor");
        } else {
            embd = nullptr; // do not extract embeddings when not needed
            GGML_ASSERT(strcmp(res->name, "result_output") == 0 && "missing result_output tensor");
        }
        // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0, gf->n_nodes, gf->n_leafs);

#if IK_PRINT_TIMING == 1
        tim1 = ggml_time_us();
#endif
        llama_set_inputs(lctx, u_batch);
#if IK_PRINT_TIMING == 1
        tim2 = ggml_time_us();
        printf("set_inputs(...): %d us\n", int(tim2-tim1));
#endif

#if IK_PRINT_TIMING
        tim1 = ggml_time_us();
#endif
        llama_graph_compute(lctx, gf, n_threads);
#if IK_PRINT_TIMING
        llama_synchronize(&lctx);
        tim2 = ggml_time_us();
        printf("graph_compute(...): %d us\n", int(tim2-tim1));
#endif

        // update the kv ring buffer
        {
            kv_self.head += n_tokens;

            // Ensure kv cache head points to a valid index.
            if (kv_self.head >= kv_self.size) {
                kv_self.head = 0;
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        // extract logits
        if (res) {
#if IK_PRINT_TIMING
            tim1 = ggml_time_us();
#endif
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(lctx.sched, res);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(lctx.logits != nullptr);

            float * logits_out = lctx.logits + n_outputs_prev*n_vocab;
            const int32_t n_outputs_new = lctx.n_outputs;

            if (n_outputs_new) {
                GGML_ASSERT( n_outputs_prev + n_outputs_new <= n_outputs);
                GGML_ASSERT((n_outputs_prev + n_outputs_new)*n_vocab <= (int64_t) lctx.logits_size);
                ggml_backend_tensor_get_async(backend_res, res, logits_out, 0, n_outputs_new*n_vocab*sizeof(float));
            }
#if IK_PRINT_TIMING
            tim2 = ggml_time_us();
            printf("get_result(...): %d us\n", int(tim2-tim1));
#endif
        }

        // extract embeddings
        if (embd) {
#if IK_PRINT_TIMING
            tim1 = ggml_time_us();
#endif
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(lctx.sched, embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(lctx.embd != nullptr);
                        float * embd_out = lctx.embd + n_outputs_prev*n_embd;
                        const int32_t n_outputs_new = lctx.n_outputs;

                        if (n_outputs_new) {
                            GGML_ASSERT( n_outputs_prev + n_outputs_new <= n_outputs);
                            GGML_ASSERT((n_outputs_prev + n_outputs_new)*n_embd <= (int64_t) lctx.embd_size);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_outputs_new*n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings
                        auto & embd_seq_out = lctx.embd_seq;
                        embd_seq_out.clear();

                        for (uint32_t i = 0; i < n_tokens; i++) {
                            const llama_seq_id seq_id = u_batch.seq_id[i][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
#if IK_PRINT_TIMING
            tim2 = ggml_time_us();
            printf("get_embedding(...): %d us\n", int(tim2-tim1));
#endif
        }
        n_outputs_prev += lctx.n_outputs;
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    lctx.n_outputs = n_outputs;

    // wait for the computation to finish (automatically done when obtaining the model output)
    //llama_synchronize(&lctx);

    // decide if we need to defrag the kv cache
    if (cparams.causal_attn && cparams.defrag_thold >= 0.0f) {
        const float fragmentation = kv_self.n >= 128 ? 1.0f - float(kv_self.used)/float(kv_self.n) : 0.0f;

        // queue defragmentation for next llama_kv_cache_update
        if (fragmentation > cparams.defrag_thold) {
            LLAMA_LOG_INFO("fragmentation: %.2f\n", fragmentation);

            llama_kv_cache_defrag(kv_self);
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
#if IK_PRINT_TIMING
    auto tim1 = ggml_time_us();
#endif
    if (!lctx.prev) {
        lctx.reset_scheduler();
    }
#if IK_PRINT_TIMING
        auto tim2 = ggml_time_us();
        printf("sched_reset(...): %d us\n", int(tim2-tim1));
#endif

    return 0;
}

// encode a batch of tokens by evaluating the encoder part of the transformer
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
static int llama_encode_internal(
         llama_context & lctx,
           llama_batch   batch) {

    lctx.is_encoding = true;

    const uint32_t n_tokens = batch.n_tokens;

    if (n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0", __func__);
        return -1;
    }

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;
    const auto & cparams = lctx.cparams;

    GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd)); // NOLINT

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (lctx.t_compute_start_us == 0) {
        lctx.t_compute_start_us = ggml_time_us();
    }

    lctx.n_queued_tokens += n_tokens;

    const int64_t n_embd = hparams.n_embd;

    // TODO: simplify or deprecate
    std::vector<llama_pos> pos;
    std::vector<int32_t>                   n_seq_id;
    std::vector<llama_seq_id *>            seq_id_arr;
    std::vector<std::vector<llama_seq_id>> seq_id;

    // reserve output buffer
    if (llama_output_reserve(lctx, n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        lctx.output_ids[i] = i;
    }

    lctx.inp_embd_enc = NULL;
    lctx.n_outputs = n_tokens;

    const int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
    GGML_ASSERT(n_threads > 0);

    // helpers for smoother batch API transition
    // after deprecating the llama_eval calls, these will be removed
    if (batch.pos == nullptr) {
        pos.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            pos[i] = batch.all_pos_0 + i*batch.all_pos_1;
        }

        batch.pos = pos.data();
    }

    if (batch.seq_id == nullptr) {
        n_seq_id.resize(n_tokens);
        seq_id.resize(n_tokens);
        seq_id_arr.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            n_seq_id[i] = 1;
            seq_id[i].resize(1);
            seq_id[i][0] = batch.all_seq_id;
            seq_id_arr[i] = seq_id[i].data();
        }

        batch.n_seq_id = n_seq_id.data();
        batch.seq_id = seq_id_arr.data();
    }

    lctx.reset_scheduler();
    ggml_backend_sched_set_eval_callback(lctx.sched, lctx.cparams.cb_eval, lctx.cparams.cb_eval_user_data);

    ggml_cgraph * gf = llm_build_context::llama_build_graph(lctx, batch, false);

    // the output embeddings after the final encoder normalization
    struct ggml_tensor * embd = nullptr;

    // there are two cases here
    if (llama_model_has_decoder(&lctx.model)) {
        // first case is an encoder-decoder T5 model where embeddings are passed to decoder
        embd = gf->nodes[gf->n_nodes - 1];
        GGML_ASSERT(strcmp(embd->name, "result_norm") == 0 && "missing result_output tensor");
    } else {
        // second case is an encoder-only T5 model
        if (cparams.embeddings) {
            // only output embeddings if required
            embd = gf->nodes[gf->n_nodes - 1];
            if (strcmp(embd->name, "result_embd_pooled") != 0) {
                embd = gf->nodes[gf->n_nodes - 2];
            }
            GGML_ASSERT(strcmp(embd->name, "result_embd_pooled") == 0 && "missing embeddings tensor");
        }
    }

    ggml_backend_sched_alloc_graph(lctx.sched, gf);

    llama_set_inputs(lctx, batch);

    llama_graph_compute(lctx, gf, n_threads);

    // extract embeddings
    if (embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(lctx.sched, embd);
        GGML_ASSERT(backend_embd != nullptr);

        if (llama_model_has_decoder(&lctx.model)) {
            lctx.embd_enc.resize(n_tokens*n_embd);
            float * embd_out = lctx.embd_enc.data();

            ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_tokens*n_embd*sizeof(float));

            // remember the sequence ids used during the encoding - needed for cross attention later
            lctx.seq_ids_enc.resize(n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                for (int s = 0; s < batch.n_seq_id[i]; s++) {
                    llama_seq_id seq_id = batch.seq_id[i][s];
                    lctx.seq_ids_enc[i].insert(seq_id);
                }
            }
        } else {
            GGML_ASSERT(lctx.embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(lctx.embd != nullptr);
                        float * embd_out = lctx.embd;

                        GGML_ASSERT(n_tokens*n_embd <= (int64_t) lctx.embd_size);
                        ggml_backend_tensor_get_async(backend_embd, embd, embd_out, 0, n_tokens*n_embd*sizeof(float));
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings
                        auto & embd_seq_out = lctx.embd_seq;
                        embd_seq_out.clear();

                        for (uint32_t i = 0; i < n_tokens; i++) {
                            const llama_seq_id seq_id = batch.seq_id[i][0];
                            if (embd_seq_out.find(seq_id) != embd_seq_out.end()) {
                                continue;
                            }
                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, embd, embd_seq_out[seq_id].data(), (n_embd*seq_id)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }
    }

    // Reset state for the next token before backend sync, to allow the CPU activities in the reset to
    // overlap with device computation.
    lctx.reset_scheduler();

    return 0;
}

// find holes from the beginning of the KV cache and fill them by moving data from the end of the cache
static void llama_kv_cache_defrag_internal(struct llama_context & lctx) {
    auto & kv_self = lctx.kv_self;

    const auto & hparams = lctx.model.hparams;

    const uint32_t n_layer = hparams.n_layer;

    const uint32_t n_kv   = llama_kv_cache_cell_max(kv_self);
    const uint32_t n_used = kv_self.used;

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // each move requires 6*n_layer tensors (see build_defrag)
    //   - source view, destination view, copy operation
    //   - x2 for keys and values
    //const uint32_t max_moves = model.max_nodes()/(6*n_layer);
    // TODO: tmp fix https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (lctx.model.max_nodes() - 2*n_layer)/(6*n_layer);

    // determine which KV cells to move where
    //
    //  cell i moves to ids[i]
    //
    //  if ids[i] == i || ids[i] == n_kv, then cell i is not moved
    //
    std::vector<uint32_t> ids(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = kv_self.cells[i0];

        if (!cell0.is_empty()) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && kv_self.cells[i0 + nh].is_empty()) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            const auto & cell1 = kv_self.cells[is];

            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // should we stop searching for the next move?
        bool stop = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = kv_self.cells[i1];

            if (cell1.is_empty() || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            // move the cell meta data
            kv_self.cells[i0 + nf] = cell1;

            // clear the old cell and move the head there
            cell1 = llama_kv_cell();
            kv_self.head = n_used;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return;
    }

    //LLAMA_LOG_INFO("(tmp log) KV defrag cell moves: %u\n", n_moves);

    //LLAMA_LOG_INFO("expected gf nodes: %u\n", 6*n_moves*n_layer);

#if 0
    // CPU defrag
    //
    // TODO: optimizations are possible:
    //       - multiple threads
    //       - avoid copying to the host memory when already there
    //
    // likely not worth the effort, as we have ggml_graph based defrag
    //

    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa();

    const uint32_t kv_size = kv_self.size;

    std::vector<uint8_t> buf_k;
    std::vector<uint8_t> buf_v;

    for (uint32_t il = 0; il < n_layer; ++il) {
        const size_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
        const size_t k_size     = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa*kv_size);

        const size_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
        const size_t v_size    = ggml_row_size (kv_self.v_l[il]->type, n_embd_v_gqa*kv_size);

        buf_k.resize(k_size);
        buf_v.resize(v_size);

        ggml_backend_tensor_get(kv_self.k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_get(kv_self.v_l[il], buf_v.data(), 0, buf_v.size());

        // batch move [i, i+nm) to [id, id+nm)
        // note: cells can move only to a lower index
        for (uint32_t i = 0; i < n_kv; ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == n_kv) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < n_kv && ids[i + nm] == id + nm) {
                nm++;
            }

            // move keys
            {
                const int64_t os =  i*k_size_row;
                const int64_t od = id*k_size_row;

                memcpy(buf_k.data() + od, buf_k.data() + os, nm*k_size_row);
            }

            // move values (note: they are transposed)
            {
                const int64_t os =  i;
                const int64_t od = id;

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    memcpy(buf_v.data() + (od + j*kv_size)*v_size_el, buf_v.data() + (os + j*kv_size)*v_size_el, nm*v_size_el);
                }
            }

            i += nm - 1;
        }

        ggml_backend_tensor_set(kv_self.k_l[il], buf_k.data(), 0, buf_k.size());
        ggml_backend_tensor_set(kv_self.v_l[il], buf_v.data(), 0, buf_v.size());
    }
#else
    // ggml_graph defrag

    lctx.reset_scheduler();

    ggml_cgraph * gf = llm_build_context::llama_build_graph_defrag(lctx, ids);

    llama_graph_compute(lctx, gf, lctx.cparams.n_threads);
#endif

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_INFO("(tmp log) KV defrag time: %.3f ms\n", (t_end - t_start)/1000.0);
}

static int32_t llama_kv_cache_update_internal(struct llama_context & lctx) {
    bool need_reserve = false;

    // apply K-shift if needed
    if (lctx.model.hparams.rope_type != LLAMA_ROPE_TYPE_NONE && lctx.kv_self.has_shift) {
        if (lctx.model.arch == LLM_ARCH_DEEPSEEK2) { // not supported due to MLA
            return 1;
        }

        {
            lctx.reset_scheduler();

            ggml_cgraph * gf = llm_build_context::llama_build_graph_k_shift(lctx);

            ggml_backend_sched_alloc_graph(lctx.sched, gf);

            llama_set_k_shift(lctx);

            llama_graph_compute(lctx, gf, lctx.cparams.n_threads);

            need_reserve = true;
        }

        {
            auto & kv_self = lctx.kv_self;

            kv_self.has_shift = false;

            for (uint32_t i = 0; i < kv_self.size; ++i) {
                kv_self.cells[i].delta = 0;
            }
        }
    }

    if (lctx.kv_self.recurrent && lctx.kv_self.do_copy) {
        {
            lctx.reset_scheduler();

            ggml_cgraph * gf = llm_build_context::llama_build_graph_s_copy(lctx);

            ggml_backend_sched_alloc_graph(lctx.sched, gf);

            llama_set_s_copy(lctx);

            llama_graph_compute(lctx, gf, lctx.cparams.n_threads);

            need_reserve = true;
        }

        {
            auto & kv_self = lctx.kv_self;

            kv_self.do_copy = false;

            for (uint32_t i = 0; i < kv_self.size; ++i) {
                kv_self.cells[i].src = i;
            }
        }
    }

    // defragment the KV cache if needed
    if (lctx.kv_self.do_defrag) {
        llama_kv_cache_defrag_internal(lctx);

        need_reserve = true;

        lctx.kv_self.do_defrag = false;
    }

    // reserve a worst case graph again
    if (need_reserve) {
        // TODO: extract to a function
        // build worst-case graph
        int n_tokens = (int)std::min(lctx.cparams.n_ctx, lctx.cparams.n_ubatch);
        int n_past = lctx.cparams.n_ctx - n_tokens;
        llama_token token = llama_token_bos(&lctx.model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
        ggml_cgraph * gf = llm_build_context::llama_build_graph(lctx, llama_batch_get_one(&token, n_tokens, n_past, 0), true);

        // initialize scheduler with the worst-case graph
        lctx.reset_scheduler();
        if (!ggml_backend_sched_reserve(lctx.sched, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        }
    }
    return 0;
}

static void llama_lora_adapter_init_internal(struct llama_model * model, const char * path_lora, struct llama_lora_adapter & adapter) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(path_lora, meta_gguf_params);
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter file from " + std::string(path_lora));
    }

    // check metadata
    {
        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(ctx_gguf, key.c_str());
            return id < 0 ? "" : std::string(gguf_get_val_str(ctx_gguf, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(ctx_gguf, key.c_str());
            return id < 0 ? 0.0f : gguf_get_val_f32(ctx_gguf, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            gguf_free(ctx_gguf);
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model->arch) {
            gguf_free(ctx_gguf);
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            gguf_free(ctx_gguf);
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf);

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto get_ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            struct ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            ctx_map[buft] = buft_ctx;
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else {
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_lora_weight & w = it.second;

        if (!w.a || !w.b) {
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        auto * model_tensor = llama_get_model_tensor(model, name.c_str());
        if (!model_tensor) {
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model");
        }
        struct ggml_context * dev_ctx = get_ctx_for_buft(ggml_backend_buffer_get_type(model_tensor->buffer));
        // validate tensor shape
        if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            throw std::runtime_error("tensor '" + name + "' has incorrect shape");
        }
        if (w.a->ne[1] != w.b->ne[0]) {
            gguf_free(ctx_gguf);
            ggml_free(ctx);
            throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
        }
        // save tensor to adapter
        struct ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        struct ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);
        adapter.ab_map[name] = llama_lora_weight(tensor_a, tensor_b);
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft);
            if (!buf) {
                gguf_free(ctx_gguf);
                ggml_free(ctx);
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
            adapter.ctxs.push_back(ctx_dev);
            adapter.bufs.push_back(buf);
        }
    }

    // set tensor data
    {
        llama_file gguf_file(path_lora, "rb");
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](struct ggml_tensor * orig, struct ggml_tensor * dev) {
            size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, gguf_find_tensor(ctx_gguf, orig->name));
            size_t size = ggml_nbytes(orig);
            read_buf.resize(size);
            gguf_file.seek(offs, SEEK_SET);
            gguf_file.read_raw(read_buf.data(), size);
            ggml_backend_tensor_set(dev, read_buf.data(), 0, size);
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    LLAMA_LOG_INFO("%s: loaded %ld tensors from lora file\n", __func__, adapter.ab_map.size()*2);

    // free ctx for reading gguf
    gguf_free(ctx_gguf);
    ggml_free(ctx);
}

int32_t llama_lora_adapter_set(
            struct llama_context * ctx,
            struct llama_lora_adapter * adapter,
            float scale) {
    if (ctx->cparams.flash_attn) {
        LLAMA_LOG_ERROR("%s: flash_attn is not compatible with LoRA\n", __func__);
        return -1;
    }
    ctx->lora_adapters[adapter] = scale;
    return 0;
}

int32_t llama_lora_adapter_remove(
            struct llama_context * ctx,
            struct llama_lora_adapter * adapter) {
    auto pos = ctx->lora_adapters.find(adapter);
    if (pos != ctx->lora_adapters.end()) {
        ctx->lora_adapters.erase(pos);
        return 0;
    }
    return -1;
}

void llama_lora_adapter_clear(struct llama_context * ctx) {
    ctx->lora_adapters.clear();
}

void llama_lora_adapter_free(struct llama_lora_adapter * adapter) {
    delete adapter;
}

//
// interface implementation
//
struct llama_model_params llama_model_default_params() {
    struct llama_model_params result = {
        /*.devices                 =*/ nullptr,
        /*.n_gpu_layers                =*/ 0,
        /*.mla                         =*/ 0,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.max_gpu                     =*/ 0,
        /*.tensor_split                =*/ nullptr,
        /*.rpc_servers                 =*/ nullptr,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_buft_overrides       =*/ nullptr,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
        /*.repack_tensors              =*/ false,
        /*.use_thp                     =*/ false,
        /*.validate_quants             =*/ false,
        /*.merge_qkv                   =*/ false,
        /*.merge_up_gate_exps          =*/ false,
    };

#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    result.n_gpu_layers = 999;
#endif

    return result;
}

struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        /*.seed                        =*/ LLAMA_DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.max_extra_alloc             =*/ 256,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.logits_all                  =*/ false,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.flash_attn                  =*/ true,
        /*.mla_attn                    =*/ 3,
        /*.attn_max_batch              =*/ 0,
        /*.fused_moe_up_gate           =*/ true,
        /*.grouped_expert_routing      =*/ false,
        /*.fused_up_gate               =*/ true,
        /*.fused_mmad                  =*/ true,
        /*.rope_cache                  =*/ false,
        /*.graph_reuse                 =*/ true,
        /*.min_experts                 =*/ -1,
        /*.thtesh_experts              =*/ 0.0f,
        /*.only_active_experts         =*/ false,
        /*.k_cache_hadamard            =*/ false,
        /*.split_mode_graph_scheduling =*/ false,
        /*.split_mode_f16              =*/ true,
        /*.scheduler_async             =*/ false,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.offload_policy              =*/ nullptr,
        /*.cuda_params                 =*/ nullptr,
    };

    return result;
}

struct llama_model_quantize_params llama_model_quantize_default_params() {
    struct llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.attn_q_type                 =*/ GGML_TYPE_COUNT,
        /*.attn_k_type                 =*/ GGML_TYPE_COUNT,
        /*.attn_v_type                 =*/ GGML_TYPE_COUNT,
        /*.attn_qkv_type               =*/ GGML_TYPE_COUNT,
        /*.attn_output_type            =*/ GGML_TYPE_COUNT,
        /*.ffn_gate_type               =*/ GGML_TYPE_COUNT,
        /*.ffn_down_type               =*/ GGML_TYPE_COUNT,
        /*.ffn_up_type                 =*/ GGML_TYPE_COUNT,
        /*.ffn_gat_inp_type            =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.ignore_imatrix_rules        =*/ false,
        /*.only_repack                 =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.custom_quants               =*/ nullptr,
        /*.repack_pattern              =*/ nullptr,
    };

    return result;
}

size_t llama_max_devices(void) {
#if defined(GGML_USE_RPC)
    return GGML_RPC_MAX_SERVERS;
#elif defined(GGML_USE_METAL)
    return 1;
#elif defined(GGML_USE_CUDA)
    return GGML_CUDA_MAX_DEVICES;
#elif defined(GGML_USE_SYCL)
    return GGML_SYCL_MAX_DEVICES;
#elif defined(GGML_USE_VULKAN)
    return GGML_VK_MAX_DEVICES;
#elif defined(GGML_USE_CANN)
    return GGML_CANN_MAX_DEVICES;
#else
    return 1;
#endif
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
#if defined(GGML_USE_CUDA) || defined(GGML_USE_METAL)   || defined(GGML_USE_VULKAN) || \
    defined(GGML_USE_SYCL) || defined(GGML_USE_KOMPUTE) || defined(GGML_USE_RPC)
    // Defined when llama.cpp is compiled with support for offloading model layers to GPU.
    return true;
#else
    return false;
#endif
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        ggml_numa_init(numa);
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

static std::string create_rpc_name(std::string endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    return dev_name;
}

struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params   params) {
    ggml_time_init();

    llama_model * model = new llama_model;

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_INFO(".");
                if (percentage >= 100) {
                    LLAMA_LOG_INFO("\n");
                }
            }
            return true;
        };
    }
    model->set_tensor_overrides(params);
    // model->devices hold device indices that are used to offload
    // use model->devices to determine offload device
    // if no device is specified, all device are included
    // if device is specified, only those in the devices are included in the model->devices

    std::vector<std::string> params_devices;
    if (params.devices && !striequals(params.devices, "")) {
        params_devices = string_split(params.devices, ",");
    }

    std::map<std::string, int32_t> buffer_names;
    std::vector<std::string> gpu_names;
    bool has_rpc = params.rpc_servers != nullptr && params.rpc_servers[0] != '\0';
    int32_t idx = 0;
    int dev_count = (int)llama_get_device_count(*model);
    // list all buffer type names
    for (idx = 0; idx < dev_count; idx++) {
        ggml_backend_buffer_type_t buft = llama_default_buffer_type_offload(*model, idx);
        const char* name = ggml_backend_buft_name(buft);
        buffer_names.insert({ std::string(name), idx });
        gpu_names.push_back(std::string(name));
    }
    if (has_rpc) {
        model->rpc_servers = extract_device_from_rpc_device(string_split(params.rpc_servers, ","));
        for (auto rpc : model->rpc_servers) {
            buffer_names.insert({ create_rpc_name(rpc.endpoint, rpc.device), idx});
            idx++;
        }
    }
    std::vector<std::string> device_names;
    if (params_devices.size()) {
        device_names = params_devices;
    } else {
        // add RPC servers at the front of the list to minimize the network transfers
        if (has_rpc) {
            for (auto& it : model->rpc_servers) {
                device_names.push_back(create_rpc_name(it.endpoint, it.device));
            }     
        }
        device_names.insert(device_names.end(), gpu_names.begin(), gpu_names.end());
    }

    for (auto & device : device_names) {
        if (buffer_names.count(device)) {
            model->devices.push_back(buffer_names[device]);
        } else {
            LLAMA_LOG_ERROR("%s backend not available.\n", device.c_str());
        }
    }

 
    // no gpu used, so set layers offload to be 0
    if (!model->devices.size()) {
        params.n_gpu_layers = 0;
        LLAMA_LOG_INFO("CPU: using device CPU\n");
    } else {
        for (auto i : model->devices) {
            ggml_backend_buffer_type_t buft = llama_default_buffer_type_offload(*model, i);
            const char* name = ggml_backend_buft_name(buft);
            const char* description = name;
            size_t description_size = llama_get_device_memory(*model, i);
            LLAMA_LOG_INFO("%s: using device %s - %zu MiB free\n", 
                name, description,
                description_size / 1024 / 1024);
        }
    }
    int status = llama_model_load(path_model, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }
        delete model;
        return nullptr;
    }

    return model;
}

void llama_free_model(struct llama_model * model) {
    delete model;
}

static void llama_repack_up_gate_exps(llama_context & lctx) {
    auto & model = lctx.model;
    bool needs_repack = false;
    for (auto & l : model.layers) {
        if (l.ffn_up_gate_exps && l.ffn_up_exps && l.ffn_gate_exps) {
            needs_repack = true; break;
        }
    }
    if (!needs_repack) return;

    std::vector<char> aux_buffer_up, aux_buffer_gate, aux_buffer_up_gate;
    for (int il = 0; il < int(model.layers.size()); ++il) {
        auto & l = model.layers[il];
        if (l.ffn_up_gate_exps && l.ffn_up_exps && l.ffn_gate_exps) {
            GGML_ASSERT(l.ffn_up_gate_exps->type  == l.ffn_up_exps->type  && l.ffn_up_gate_exps->type  == l.ffn_gate_exps->type);
            GGML_ASSERT(l.ffn_up_gate_exps->ne[0] == l.ffn_up_exps->ne[0] && l.ffn_up_gate_exps->ne[0] == l.ffn_gate_exps->ne[0]);
            GGML_ASSERT(l.ffn_up_gate_exps->ne[2] == l.ffn_up_exps->ne[2] && l.ffn_up_gate_exps->ne[2] == l.ffn_gate_exps->ne[2]);
            GGML_ASSERT(l.ffn_up_gate_exps->ne[1] == l.ffn_up_exps->ne[1] + l.ffn_gate_exps->ne[1]);
            auto nbytes = ggml_nbytes(l.ffn_up_exps);
            GGML_ASSERT(nbytes == ggml_nbytes(l.ffn_gate_exps));
            if (nbytes > aux_buffer_up.size()) {
                aux_buffer_up.resize(nbytes);
            }
            if (nbytes > aux_buffer_gate.size()) {
                aux_buffer_gate.resize(nbytes);
            }
            printf("%s: repacking up/gate experts weight in layer %d\n", __func__, il);
            ggml_backend_tensor_get(l.ffn_up_exps, aux_buffer_up.data(), 0, nbytes);
            ggml_backend_tensor_get(l.ffn_gate_exps, aux_buffer_gate.data(), 0, nbytes);
            if (aux_buffer_up_gate.size() < 2*nbytes) {
                aux_buffer_up_gate.resize(2*nbytes);
            }
            size_t offset_up_gate = 0;
            size_t offset_up = 0;
            auto expert_size = l.ffn_up_exps->ne[1]*l.ffn_up_exps->nb[1];
            for (int i2 = 0; i2 < (int)l.ffn_up_gate_exps->ne[2]; ++i2) {
                std::memcpy(aux_buffer_up_gate.data() + offset_up_gate, aux_buffer_up.data() + offset_up, expert_size);
                offset_up_gate += expert_size;
                std::memcpy(aux_buffer_up_gate.data() + offset_up_gate, aux_buffer_gate.data() + offset_up, expert_size);
                offset_up_gate += expert_size;
                offset_up      += expert_size;
            }
            ggml_backend_tensor_set(l.ffn_up_gate_exps, aux_buffer_up_gate.data(), 0, 2*expert_size*l.ffn_up_gate_exps->ne[2]);
            if (l.ffn_up_gate_exps_b && l.ffn_up_exps_b && l.ffn_gate_exps_b) {
                nbytes = ggml_nbytes(l.ffn_up_exps_b);
                GGML_ASSERT(nbytes == ggml_nbytes(l.ffn_gate_exps_b));
                if (nbytes > aux_buffer_up.size()) {
                    aux_buffer_up.resize(nbytes);
                }
                if (nbytes > aux_buffer_gate.size()) {
                    aux_buffer_gate.resize(nbytes);
                }
                printf("%s: repacking up/gate experts bias in layer %d\n", __func__, il);
                ggml_backend_tensor_get(l.ffn_up_exps_b, aux_buffer_up.data(), 0, nbytes);
                ggml_backend_tensor_get(l.ffn_gate_exps_b, aux_buffer_gate.data(), 0, nbytes);
                if (aux_buffer_up_gate.size() < 2*nbytes) {
                    aux_buffer_up_gate.resize(2*nbytes);
                }
                offset_up_gate = 0;
                offset_up = 0;
                expert_size = l.ffn_up_exps_b->nb[1];
                for (int i1 = 0; i1 < (int)l.ffn_up_gate_exps_b->ne[1]; ++i1) {
                    std::memcpy(aux_buffer_up_gate.data() + offset_up_gate, aux_buffer_up.data() + offset_up, expert_size);
                    offset_up_gate += expert_size;
                    std::memcpy(aux_buffer_up_gate.data() + offset_up_gate, aux_buffer_gate.data() + offset_up, expert_size);
                    offset_up_gate += expert_size;
                    offset_up      += expert_size;
                }
                ggml_backend_tensor_set(l.ffn_up_gate_exps_b, aux_buffer_up_gate.data(), 0, 2*expert_size*l.ffn_up_gate_exps_b->ne[1]);
            }
        }
    }
}

struct llama_context * llama_new_context_with_model(
                 struct llama_model * model,
        struct llama_context_params   params) {

    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn = false;
    }

    //if (params.flash_attn && model->hparams.n_embd_head_k != model->hparams.n_embd_head_v) {
    //    LLAMA_LOG_WARN("%s: flash_attn requires n_embd_head_k == n_embd_head_v - forcing off\n", __func__);
    //    params.flash_attn = false;
    //}

    if (params.type_v != GGML_TYPE_F16 && params.type_v != GGML_TYPE_BF16 && !params.flash_attn) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.k_cache_hadamard && !ggml_is_quantized(params.type_k)) {
        LLAMA_LOG_WARN("%s: there is no point in Hadamard transforms with not quantized K-cache. Turning Hadamard off\n", __func__);
        params.k_cache_hadamard = false;
    }

    llama_context * ctx = new llama_context(*model);

    // add devices to ctx->cparams from model
    for (int i : model->devices) {
        ggml_backend_buffer_type_t buft = llama_default_buffer_type_offload(*model, i);
        const char* name = ggml_backend_buft_name(buft);
        std::string device(name);
        ctx->cparams.devices.push_back(device);
    }

    const auto & hparams = model->hparams;
    auto       & cparams = ctx->cparams;


    cparams.n_seq_max        = std::max(1u, params.n_seq_max);
    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor >= 0.0f ? params.yarn_ext_factor : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast >= 0.0f ? params.yarn_beta_fast : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow >= 0.0f ? params.yarn_beta_slow : hparams.yarn_beta_slow;
    cparams.defrag_thold     = params.defrag_thold;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.flash_attn       = params.flash_attn;
    cparams.mla_attn         = params.mla_attn;
    cparams.attn_max_batch   = params.attn_max_batch;
    cparams.fused_moe_up_gate= params.fused_moe_up_gate;
    cparams.grouped_expert_routing = params.grouped_expert_routing;
    cparams.fused_up_gate    = params.fused_up_gate;
    cparams.fused_mmad       = params.fused_mmad;
    cparams.rope_cache       = params.rope_cache;
    cparams.graph_reuse      = params.graph_reuse;
    cparams.k_cache_hadamard = params.k_cache_hadamard;
    cparams.split_mode_graph_scheduling = params.split_mode_graph_scheduling;
    cparams.split_mode_f16   = params.split_mode_f16;
    cparams.scheduler_async  = params.scheduler_async;
    cparams.min_experts      = params.min_experts;
    cparams.thresh_experts   = params.thresh_experts;
    cparams.cuda_params      = params.cuda_params;

    cparams.pooling_type     = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    // this is necessary due to kv_self.n being padded later during inference
    cparams.n_ctx            = GGML_PAD(cparams.n_ctx, llama_kv_cache_get_padding(cparams));

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch          = hparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // the batch has to be at least GGML_KQ_MASK_PAD because we will be padding the KQ_mask
    // this is required by GPU kernels in order to avoid out-of-bounds accesses (e.g. ggml_flash_attn_ext)
    // ref: https://github.com/ggerganov/llama.cpp/pull/5021
    if (cparams.n_batch < GGML_KQ_MASK_PAD) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }

    cparams.n_ubatch         = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    if (model->arch != LLM_ARCH_DEEPSEEK2 && cparams.mla_attn != 0) {
        //LLAMA_LOG_WARN("=====================================================================\n");
        //LLAMA_LOG_WARN(" MLA is only available for LLM_ARCH_DEEPSEEK2 -> turning off MLA\n");
        //LLAMA_LOG_WARN("=====================================================================\n");
        cparams.mla_attn = 0;
    }
    if (model->arch == LLM_ARCH_OPENAI_MOE && model->split_mode == LLAMA_SPLIT_MODE_GRAPH) {
        if (cparams.split_mode_f16) {
            LLAMA_LOG_WARN("=====================================================================\n");
            LLAMA_LOG_WARN("GPT-OSS with split mode graph requires f32 precision\n");
            LLAMA_LOG_WARN("    => changing cparams.split_mode_f16 to 'false'\n");
            LLAMA_LOG_WARN("=====================================================================\n");
            cparams.split_mode_f16 = false;
        }
    }

    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",     __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",     __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",     __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: flash_attn    = %d\n",     __func__, cparams.flash_attn);
    if (model->arch == LLM_ARCH_DEEPSEEK2) {
    LLAMA_LOG_INFO("%s: mla_attn      = %d\n",     __func__, cparams.mla_attn);
    }
    LLAMA_LOG_INFO("%s: attn_max_b    = %d\n",     __func__, cparams.attn_max_batch);
    LLAMA_LOG_INFO("%s: fused_moe     = %d\n",     __func__, cparams.fused_moe_up_gate);
    LLAMA_LOG_INFO("%s: grouped er    = %d\n",     __func__, cparams.grouped_expert_routing);
    LLAMA_LOG_INFO("%s: fused_up_gate = %d\n",     __func__, cparams.fused_up_gate);
    LLAMA_LOG_INFO("%s: fused_mmad    = %d\n",     __func__, cparams.fused_mmad);
    LLAMA_LOG_INFO("%s: rope_cache    = %d\n",     __func__, cparams.rope_cache);
    LLAMA_LOG_INFO("%s: graph_reuse   = %d\n",     __func__, cparams.graph_reuse);
    LLAMA_LOG_INFO("%s: k_cache_hadam = %d\n",     __func__, cparams.k_cache_hadamard);
    LLAMA_LOG_INFO("%s: split_mode_graph_scheduling = %d\n",   __func__, cparams.split_mode_graph_scheduling);
    LLAMA_LOG_INFO("%s: split_mode_f16= %d\n",     __func__, cparams.split_mode_f16);
    LLAMA_LOG_INFO("%s: sched_async   = %d\n",     __func__, cparams.scheduler_async);
    LLAMA_LOG_INFO("%s: ser           = %d, %g\n", __func__, cparams.min_experts, cparams.thresh_experts);
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n",   __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",     __func__, cparams.rope_freq_scale);
    if (cparams.cuda_params) {
        LLAMA_LOG_INFO("%s: cuda_params   = %s\n", __func__, (const char *)cparams.cuda_params);
    }

    ctx->abort_callback      = params.abort_callback;
    ctx->abort_callback_data = params.abort_callback_data;

    ctx->sampling.rng = std::mt19937(params.seed);
    ctx->logits_all   = params.logits_all;
    // build worst-case graph for encoder if a model contains encoder
    ctx->is_encoding  = llama_model_has_encoder(model);

    uint32_t kv_size = cparams.n_ctx;
    ggml_type type_k = params.type_k;
    ggml_type type_v = params.type_v;

    // Mamba only needs a constant number of KV cache cells per sequence
    if (model->arch == LLM_ARCH_MAMBA) {
        // Mamba needs at least as many KV cells as there are sequences kept at any time
        kv_size = std::max((uint32_t) 1, params.n_seq_max);
        // it's probably best to keep as much precision as possible for the states
        type_k = GGML_TYPE_F32; // required by ggml_ssm_conv for Mamba's conv_states
        type_v = GGML_TYPE_F32; // required by ggml_ssm_scan for Mamba's ssm_states
    }

    GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);
    if (!hparams.vocab_only) {
        // initialize backends
#if defined(GGML_USE_METAL)
        if (model->n_gpu_layers > 0) {
            ctx->backend_metal = ggml_backend_metal_init();
            if (ctx->backend_metal == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize Metal backend\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
            ggml_backend_add_from_device(ctx,  ctx->backend_metal);
        }
#elif defined(GGML_USE_CUDA)
        if (model->split_mode == LLAMA_SPLIT_MODE_NONE) {
            // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_GRAPH, only the main GPU backend is used
            ggml_backend_t backend = ggml_backend_cuda_init(model->main_gpu, cparams.cuda_params);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize CUDA%d backend\n", __func__, model->main_gpu);
                llama_free(ctx);
                return nullptr;
            }
            ggml_backend_add_from_device(ctx, backend);

        } else {
            // LLAMA_SPLIT_MODE_LAYER and LLAMA_SPLIT_MODE_GRAPH require a backend for each GPU
            auto params = cparams.cuda_params;
            std::string new_params;
            if (model->split_mode == LLAMA_SPLIT_MODE_GRAPH) {
                static const std::string extra_string{"graphs=0"};
                if (params) new_params = std::string{(const char *)params} + ',';
                new_params += extra_string;
                params = new_params.data();
            }
            for (int device = 0; device < ggml_backend_cuda_get_device_count(); ++device) {
                ggml_backend_t backend = ggml_backend_cuda_init(device, params);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize CUDA%d backend\n", __func__, device);
                    llama_free(ctx);
                    return nullptr;
                }
                ggml_backend_add_from_device(ctx, backend);
            }
        }
#elif defined(GGML_USE_VULKAN)
        if (model->split_mode == LLAMA_SPLIT_MODE_GRAPH || model->split_mode == LLAMA_SPLIT_MODE_ATTN) {
            LLAMA_LOG_ERROR("%s: split mode 'graph' or 'attn' not supported. Failed to initialize Vulkan backend\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
        if (model->split_mode == LLAMA_SPLIT_MODE_NONE) {
            ggml_backend_t backend = ggml_backend_vk_init(model->main_gpu);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize Vulkan backend\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
            ggml_backend_add_from_device(ctx, backend);
        } else {
            for (int device = 0; device < ggml_backend_vk_get_device_count(); ++device) {
                ggml_backend_t backend = ggml_backend_vk_init(device);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize Vulkan%d backend\n", __func__, device);
                    llama_free(ctx);
                    return nullptr;
                }
                ggml_backend_add_from_device(ctx, backend);
            }
        }
#elif defined(GGML_USE_SYCL)
        // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_GRAPH, only the main GPU backend is used
        if (model->split_mode == LLAMA_SPLIT_MODE_NONE || model->split_mode == LLAMA_SPLIT_MODE_GRAPH) {
            ggml_backend_t backend = ggml_backend_sycl_init(model->main_gpu);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize SYCL%d backend\n", __func__, model->main_gpu);
                llama_free(ctx);
                return nullptr;
            }
            ctx->backends.push_back(backend);
        } else {
            // LLAMA_SPLIT_LAYER requires a backend for each GPU
            for (int i = 0; i < ggml_backend_sycl_get_device_count(); ++i) {
                ggml_backend_t backend = ggml_backend_sycl_init(i);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize SYCL%d for No.%d backend\n", __func__, i, i);
                    llama_free(ctx);
                    return nullptr;
                }
                ggml_backend_add_from_device(ctx, backend);
            }
        }
#elif defined(GGML_USE_KOMPUTE)
        if (model->n_gpu_layers > 0) {
            auto * backend = ggml_backend_kompute_init(model->main_gpu);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize Kompute backend\n", __func__);
                llama_free(ctx);
                return nullptr;
            }
            ggml_backend_add_from_device(ctx, backend);
        }
#elif defined(GGML_USE_CANN)
    // with split_mode LLAMA_SPLIT_MODE_NONE or LLAMA_SPLIT_MODE_GRAPH, only the main GPU backend is used
    // TODO: ggml_backend_cann is not support split tensor now, just leave code here.
    if (model->split_mode == LLAMA_SPLIT_MODE_NONE || model->split_mode == LLAMA_SPLIT_MODE_GRAPH) {
        ggml_backend_t backend = ggml_backend_cann_init(model->main_gpu);
        if (backend == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CANN%d backend\n", __func__, model->main_gpu);
            llama_free(ctx);
            return nullptr;
        }
        ggml_backend_add_from_device(ctx, backend);
    } else {
        // LLAMA_SPLIT_MODE_LAYER requires a backend for each GPU
        // TODO: currently, CANN can't use multi-gpus, just leave code here for further cann version.
        for (int32_t device = 0; device < ggml_backend_cann_get_device_count(); ++device) {
            ggml_backend_t backend = ggml_backend_cann_init(device);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize CANN%d backend\n", __func__, device);
                llama_free(ctx);
                return nullptr;
            }
            ggml_backend_add_from_device(ctx, backend);
        }
    }
#endif

#ifdef GGML_USE_BLAS
        ctx->backend_blas = ggml_backend_blas_init();
        if (ctx->backend_blas == nullptr) {
            LLAMA_LOG_WARN("%s: failed to initialize BLAS backend\n", __func__);
        } else {
            ggml_backend_add_from_device(ctx, ctx->backend_blas);
        }
#endif

#if defined(GGML_USE_RPC)
        if (model->n_gpu_layers > 0) {
            for (const auto & device : model->rpc_servers) {
                ggml_backend_t backend = ggml_backend_rpc_init(device.endpoint.c_str(), device.device);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize RPC%d to '%s'\n", __func__, device.device,
                        device.endpoint.c_str());
                    llama_free(ctx);
                    return nullptr;
                }
                ggml_backend_add_from_device(ctx, backend);
            }
        }
#endif
        if (ctx->cparams.devices.size()) {
            // reorder the backend from devices params
            std::vector<ggml_backend_t> backends = {};
            std::vector<const char*> device_list = {};
            for (auto device : ctx->cparams.devices) {
                ggml_backend_t backend = ctx->ggml_backend_by_name(device.c_str());
                if (backend) {
                    backends.push_back(backend);
                }
            }
            ctx->backends = std::move(backends);
        }

        ctx->backend_cpu = ggml_backend_cpu_init();
        if (ctx->backend_cpu == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
            llama_free(ctx);
            return nullptr;
        }
        ctx->backends.push_back(ctx->backend_cpu);

        if (!llama_kv_cache_init(ctx->kv_self, ctx, type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            llama_free(ctx);
            return nullptr;
        }

        {
            size_t memory_size_k = 0;
            size_t memory_size_v = 0;

            for (auto & k : ctx->kv_self.k_l) {
                memory_size_k += ggml_nbytes(k);
            }

            for (auto & v : ctx->kv_self.v_l) {
                memory_size_v += ggml_nbytes(v);
            }

            if (memory_size_k + memory_size_v > 0) {
	        if (cparams.mla_attn != 0 && !cparams.flash_attn) {
                    LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, c^KV (%s): %7.2f MiB, kv^T (%s): %7.2f MiB\n", __func__,
                            (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                            ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                            ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
                } else if (cparams.mla_attn != 0 && cparams.flash_attn) {
                    LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, c^KV (%s): %7.2f MiB, kv^T: not used\n", __func__,
                            (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                            ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f));
                } else {
                    LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                            (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                            ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                            ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
                }
            }
        }

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if (llama_output_reserve(*ctx, params.n_seq_max) < params.n_seq_max) {
                LLAMA_LOG_ERROR("%s: failed to reserve initial output buffer\n", __func__);
                llama_free(ctx);
                return nullptr;
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name(ctx->buf_output),
                    ggml_backend_buffer_get_size(ctx->buf_output) / 1024.0 / 1024.0);
        }

        // scheduler and compute buffers
        {
            // buffer types used for the compute buffer of each backend
            std::vector<ggml_backend_buffer_type_t> backend_buft;
            for (auto * backend : ctx->backends) {
                if (ggml_backend_is_cpu(backend)) {
                    // use host buffers for the CPU backend compute buffer
                    backend_buft.push_back(llama_default_buffer_type_cpu(true));
                } else {
                    backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
                }
            }

            const size_t max_nodes = model->max_nodes();

            // buffer used to store the computation graph and the tensor meta data
            ctx->buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

            // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
            bool pipeline_parallel =
                llama_get_device_count(*model) > 1 &&
                model->n_gpu_layers > (int)model->hparams.n_layer &&
                model->split_mode == LLAMA_SPLIT_MODE_LAYER &&
                params.offload_kqv && !model->has_tensor_overrides();
#ifndef GGML_USE_CUDA
            // pipeline parallelism requires support for async compute and events
            // currently this is only implemented in the CUDA backend
            pipeline_parallel = false;
#endif
            ctx->sched = ggml_backend_sched_new(ctx->backends.data(), backend_buft.data(), ctx->backends.size(), max_nodes, pipeline_parallel);

            if (pipeline_parallel) {
                LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(ctx->sched));
            }

            llama_repack_up_gate_exps(*ctx);

            // build worst-case graph
            int n_tokens = (int)std::min(cparams.n_ctx, cparams.n_ubatch);
            int n_past = cparams.n_ctx - n_tokens;
            llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
            ggml_cgraph * gf = llm_build_context::llama_build_graph(*ctx, llama_batch_get_one(&token, n_tokens, n_past, 0), true);

            // initialize scheduler with the worst-case graph
            bool gf_success = ggml_backend_sched_reserve(ctx->sched, gf);
            if (!gf_success)
            {
                if (pipeline_parallel) {
                    LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                    ctx->sched = ggml_backend_sched_new(ctx->backends.data(), backend_buft.data(), ctx->backends.size(), max_nodes, false);
                    gf_success = ggml_backend_sched_reserve(ctx->sched, gf);
                }
                if (!gf_success) {
                    LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
                    llama_free(ctx);
                    return nullptr;
                }
            }

            for (size_t i = 0; i < ctx->backends.size(); i++) {
                ggml_backend_t backend = ctx->backends[i];
                ggml_backend_buffer_type_t buft = backend_buft[i];
                size_t size = ggml_backend_sched_get_buffer_size(ctx->sched, backend);
                if (size > 1) {
                    LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                            ggml_backend_buft_name(buft),
                            size / 1024.0 / 1024.0);
                }
            }

            // note: the number of splits during measure is higher than during inference due to the kv shift
            int n_splits = ggml_backend_sched_get_n_splits(ctx->sched);
            LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, gf->n_nodes);
            LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits);
        }
    }

    if (params.offload_policy) {
        const std::vector<std::pair<int, int>>& policy = *(const std::vector<std::pair<int, int>>*)params.offload_policy;
        for (auto [op, on_off] : policy) {
            if (op < 0 || op >= int(GGML_OP_COUNT)) {
                LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXX Setting offload policy for all ops to %s\n", on_off ? "ON" : "OFF");
            } else {
                LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXX Setting offload policy for op %s to %s\n",
                        ggml_op_name(ggml_op(op)), on_off ? "ON" : "OFF");
            }
            ggml_backend_sched_set_op_offload(ctx->sched, ggml_op(op), on_off);
        }
    }

    if (params.only_active_experts) {
        LLAMA_LOG_INFO("XXXXXXXXXXXXXXXXXXXXX Setting only active experts offload\n");
        ggml_backend_sched_set_only_active_experts(ctx->sched, true);
    }
    if (model->split_mode == LLAMA_SPLIT_MODE_GRAPH && (!model->has_tensor_overrides() || cparams.split_mode_graph_scheduling)) {
        ggml_backend_sched_set_split_mode_graph(ctx->sched, true, cparams.scheduler_async);
        ggml_backend_sched_set_max_extra_alloc(ctx->sched, params.max_extra_alloc);
        if (model->has_tensor_overrides() && cparams.split_mode_graph_scheduling) {
            LLAMA_LOG_INFO("XXXXXXXX Split Mode Graph Scheduling is FORCED despite tensor overrides due to user choice.\n");
            LLAMA_LOG_INFO("XXXXXXXX It may or might NOT infer properly due to unsupported combinations between SMGS and every possible tensor overrides.\n");
        }
    }

    return ctx;
}

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

const struct llama_vocab* llama_model_get_vocab(const struct llama_model* model) {
    return &model->vocab;
}

const struct llama_model * llama_get_model(const struct llama_context * ctx) {
    return &ctx->model;
}

const struct llama_vocab * llama_get_vocab(const struct llama_context * ctx) {
    return &ctx->model.vocab;
}

uint32_t llama_n_ctx(const struct llama_context * ctx) {
    return ctx->cparams.n_ctx;
}

uint32_t llama_n_batch(const struct llama_context * ctx) {
    return ctx->cparams.n_batch;
}

uint32_t llama_n_ubatch(const struct llama_context * ctx) {
    return ctx->cparams.n_ubatch;
}

uint32_t llama_n_seq_max(const struct llama_context * ctx) {
    return ctx->kv_self.size;
}

enum llama_vocab_type llama_vocab_type(const struct llama_model * model) {
    return model->vocab.get_type();
}

const struct llama_vocab* llama_get_model_vocab(const struct llama_model* model) {
    return &model->vocab;
}

enum llama_rope_type llama_rope_type(const struct llama_model * model) {
    switch (model->arch) {
        // these models do not use RoPE
        case LLM_ARCH_GPT2:
        case LLM_ARCH_GPTJ:
        case LLM_ARCH_MPT:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_BLOOM:
        case LLM_ARCH_MAMBA:
        case LLM_ARCH_JINA_BERT_V2:
        case LLM_ARCH_T5:
        case LLM_ARCH_T5ENCODER:
        case LLM_ARCH_JAIS:
            return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_DECI:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_BAICHUAN:
        case LLM_ARCH_STARCODER:
        case LLM_ARCH_PLAMO:
        case LLM_ARCH_ORION:
        case LLM_ARCH_INTERNLM2:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_XVERSE:
        case LLM_ARCH_COMMAND_R:
        case LLM_ARCH_OLMO:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_CHATGLM:
        case LLM_ARCH_GLM4:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_COHERE2:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_SMOLLM3:
        case LLM_ARCH_MISTRAL3:
            return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
        case LLM_ARCH_FALCON:
        case LLM_ARCH_GROK:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        case LLM_ARCH_STABLELM:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_BITNET:
        case LLM_ARCH_BITNET_25:
        case LLM_ARCH_BITNET_B158:
        case LLM_ARCH_QWEN:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_PHI2:
        case LLM_ARCH_PHI3:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
        case LLM_ARCH_GEMMA3:
        case LLM_ARCH_STARCODER2:
        case LLM_ARCH_OPENELM:
        case LLM_ARCH_GPTNEOX:
        case LLM_ARCH_CODESHELL:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MIMO2:
            return LLAMA_ROPE_TYPE_NEOX;

        case LLM_ARCH_QWEN2VL:
            return LLAMA_ROPE_TYPE_MROPE;

        case LLM_ARCH_QWEN3VL:
        case LLM_ARCH_QWEN3VLMOE:
            return LLAMA_ROPE_TYPE_IMROPE;

        // all model arches should be listed explicitly here
        case LLM_ARCH_UNKNOWN:
            GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx) {
    return ctx->cparams.pooling_type;
}

int32_t llama_n_vocab(const struct llama_model * model) {
    return model->hparams.n_vocab;
}

int32_t llama_n_ctx_train(const struct llama_model * model) {
    return model->hparams.n_ctx_train;
}

int32_t llama_n_embd(const struct llama_model * model) {
    return model->hparams.n_embd;
}

int32_t llama_model_n_embd_inp(const llama_model* model) {
    return model->hparams.n_embd_inp();
}

int32_t llama_n_layer(const struct llama_model * model) {
    return model->hparams.n_layer;
}

float llama_rope_freq_scale_train(const struct llama_model * model) {
    return model->hparams.rope_freq_scale_train;
}

int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size) {
    const auto & it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_meta_count(const struct llama_model * model) {
    return (int)model->gguf_kv.size();
}

int32_t llama_model_meta_key_by_index(const struct llama_model * model, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)model->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = model->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size) {
    return snprintf(buf, buf_size, "%s %s %s",
            llama_model_arch_name(model->arch),
            llama_model_type_name(model->type),
            llama_model_ftype_name(model->ftype).c_str());
}

uint64_t llama_model_size(const struct llama_model * model) {
    uint64_t size = 0;
    for (const auto & it : model->tensors_by_name) {
        size += ggml_nbytes(it.second);
    }
    return size;
}

const char* llama_model_chat_template(const struct llama_model* model, const char* name) {
    const auto key = name ? LLM_KV(model->arch, name)(LLM_KV_TOKENIZER_CHAT_TEMPLATE)
        : LLM_KV(model->arch)(LLM_KV_TOKENIZER_CHAT_TEMPLATE);
    const auto& it = model->gguf_kv.find(key);
    if (it == model->gguf_kv.end()) {
        // one-off fix for very popular models (so we are not flooded with issues)
        // do not extend this list unless absolutely necessary
        // Mistral-Small-2503 does not have built-in chat template
        llama_vocab_pre_type pre_type = model->vocab.get_pre_type();
        if (!name && pre_type == LLAMA_VOCAB_PRE_TYPE_TEKKEN && model->layers.size() == 40) {
            return "mistral-v7-tekken";
        }

        return nullptr;
    }
    return it->second.c_str();
}

uint64_t llama_model_n_params(const struct llama_model * model) {
    uint64_t nparams = 0;
    for (const auto & it : model->tensors_by_name) {
        nparams += ggml_nelements(it.second);
    }
    return nparams;
}

struct ggml_tensor * llama_get_model_tensor(struct llama_model * model, const char * name) {
    auto it = std::find_if(model->tensors_by_name.begin(), model->tensors_by_name.end(),
            [name](const std::pair<std::string, struct ggml_tensor *> & it) {
                return it.first == name;
            });
    if (it == model->tensors_by_name.end()) {
        return nullptr;
    }
    return it->second;
}

bool llama_model_has_encoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5:        return true;
        case LLM_ARCH_T5ENCODER: return true;
        default:                 return false;
    }
}

bool llama_model_has_decoder(const struct llama_model * model) {
    switch (model->arch) {
        case LLM_ARCH_T5ENCODER: return false;
        default:                 return true;
    }
}

llama_token llama_model_decoder_start_token(const struct llama_model * model) {
    return model->hparams.dec_start_token_id;
}

struct llama_lora_adapter * llama_lora_adapter_init(struct llama_model * model, const char * path_lora) {
    try {
        struct llama_lora_adapter * adapter = new llama_lora_adapter(model);
        llama_lora_adapter_init_internal(model, path_lora, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return nullptr;
    }
}

static bool llama_control_vector_init(struct llama_control_vector & cvec, const llama_model & model) {
    GGML_ASSERT(cvec.tensors.empty());
    GGML_ASSERT(cvec.ctxs.empty());
    GGML_ASSERT(cvec.bufs.empty());

    // count layer buffer types
    std::map<ggml_backend_buffer_type_t, int> buft_layer_count;
    for (int64_t i = 0; i < model.hparams.n_layer; i++) {
        buft_layer_count[model.buft_layer[i].buft]++;
    }

    // allocate contexts
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    for (auto & it : buft_layer_count) {
        int n_layers = it.second;
        struct ggml_init_params params = {
            /*.mem_size   =*/ n_layers * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return 1;
        }
        ctx_map[it.first] = ctx;
    }

    // make tensors
    cvec.tensors.reserve(model.hparams.n_layer);
    cvec.tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        struct ggml_context * ctx = ctx_map.at(model.buft_layer[il].buft);
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_embd);
        cvec.tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    cvec.ctxs.reserve(ctx_map.size());
    cvec.bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        cvec.ctxs.push_back(ctx);
        cvec.bufs.push_back(buf);
    }

    return true;
}

int32_t llama_control_vector_apply(struct llama_context * lctx, const float * data, size_t len, int32_t n_embd, int32_t il_start, int32_t il_end) {
    const llama_model & model = lctx->model;
    llama_control_vector & cvec = lctx->cvec;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        cvec.layer_start = -1;
        cvec.layer_end   = -1;
        return 0;
    }

    if (n_embd != (int) model.hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return 1;
    }

    if (cvec.tensors.empty()) {
        if (!llama_control_vector_init(cvec, model)) {
            return 1;
        }
    }

    cvec.layer_start = il_start;
    cvec.layer_end   = il_end;

    for (size_t il = 1; il < model.hparams.n_layer; il++) {
        assert(cvec.tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(cvec.tensors[il], data + off, 0, n_embd * ggml_element_size(cvec.tensors[il]));
        }
    }

    return 0;
}

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max) {
    struct llama_kv_cache_view result = {
        /*.n_cells            = */ 0,
        /*.n_seq_max          = */ n_seq_max,
        /*.token_count        = */ 0,
        /*.used_cells         = */ llama_get_kv_cache_used_cells(ctx),
        /*.max_contiguous     = */ 0,
        /*.max_contiguous_idx = */ -1,
        /*.cells              = */ nullptr,
        /*.cells_sequences    = */ nullptr,
    };
    return result;
}

void llama_kv_cache_view_free(struct llama_kv_cache_view * view) {
    if (view->cells != nullptr) {
        free(view->cells);
        view->cells = nullptr;
    }
    if (view->cells_sequences != nullptr) {
        free(view->cells_sequences);
        view->cells_sequences = nullptr;
    }
}

void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view) {
    if (uint32_t(view->n_cells) < ctx->kv_self.size || view->cells == nullptr) {
        view->n_cells = int32_t(ctx->kv_self.size);
        void * p = realloc(view->cells, sizeof(struct llama_kv_cache_view_cell) * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells");
        view->cells = (struct llama_kv_cache_view_cell *)p;
        p = realloc(view->cells_sequences, sizeof(llama_seq_id) * view->n_seq_max * view->n_cells);
        GGML_ASSERT(p != nullptr && "Failed to alloc kv_cache_view cells sequences");
        view->cells_sequences = (llama_seq_id *)p;
    }

    const std::vector<llama_kv_cell> & kv_cells = ctx->kv_self.cells;
    llama_kv_cache_view_cell * c_curr = view->cells;
    llama_seq_id * cs_curr = view->cells_sequences;
    int32_t used_cells = 0;
    int32_t token_count = 0;
    int32_t curr_contig_idx = -1;
    uint32_t max_contig = 0;
    int32_t max_contig_idx = -1;

    for (int32_t i = 0; i < int32_t(ctx->kv_self.size); i++, c_curr++, cs_curr += view->n_seq_max) {
        const size_t curr_size = kv_cells[i].seq_id.size();
        token_count += curr_size;
        c_curr->pos = kv_cells[i].pos + kv_cells[i].delta;

        if (curr_size > 0) {
            if (curr_contig_idx >= 0 && uint32_t(i - curr_contig_idx) > max_contig) {
                max_contig = i - curr_contig_idx;
                max_contig_idx = curr_contig_idx;
            }
            curr_contig_idx = -1;
        } else if (curr_contig_idx < 0) {
            curr_contig_idx = i;
        }

        int seq_idx = 0;
        for (const llama_seq_id it : kv_cells[i].seq_id) {
            if (seq_idx >= view->n_seq_max) {
                break;
            }
            cs_curr[seq_idx] = it;
            seq_idx++;
        }
        if (seq_idx != 0) {
            used_cells++;
        }
        for (; seq_idx < view->n_seq_max; seq_idx++) {
            cs_curr[seq_idx] = -1;
        }
    }
    if (curr_contig_idx >= 0 && kv_cells.size() - curr_contig_idx > max_contig) {
        max_contig_idx = curr_contig_idx;
        max_contig = kv_cells.size() - curr_contig_idx;
    }
    view->max_contiguous = max_contig;
    view->max_contiguous_idx = max_contig_idx;
    view->token_count = token_count;
    view->used_cells = used_cells;
    if (uint32_t(used_cells) != ctx->kv_self.used) {
        LLAMA_LOG_ERROR("%s: used cells mismatch. kv_cache says %d but we calculated %d\n",
            __func__, ctx->kv_self.used, used_cells);
    }
}

int32_t llama_get_kv_cache_token_count(const struct llama_context * ctx) {
    int result = 0;

    for (uint32_t i = 0; i < ctx->kv_self.size; i++) {
        result += ctx->kv_self.cells[i].seq_id.size();
    }

    return result;
}

int32_t llama_get_kv_cache_used_cells(const struct llama_context * ctx) {
    return ctx->kv_self.used;
}

void llama_kv_cache_clear(struct llama_context * ctx) {
    llama_kv_cache_clear(ctx->kv_self);
}

bool llama_kv_cache_seq_rm(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return llama_kv_cache_seq_rm(ctx->kv_self, seq_id, p0, p1);
}

void llama_kv_cache_seq_cp(struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }
    llama_kv_cache_seq_cp(ctx->kv_self, seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_seq_keep(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_kv_cache_seq_keep(ctx->kv_self, seq_id);
}

void llama_kv_cache_seq_add(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (delta == 0) {
        return;
    }

    llama_kv_cache_seq_add(ctx->kv_self, seq_id, p0, p1, delta);
}

void llama_kv_cache_seq_div(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    llama_kv_cache_seq_div(ctx->kv_self, seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_seq_pos_max(struct llama_context * ctx, llama_seq_id seq_id) {
    return llama_kv_cache_seq_pos_max(ctx->kv_self, seq_id);
}

void llama_kv_cache_defrag(struct llama_context * ctx) {
    llama_kv_cache_defrag(ctx->kv_self);
}

int32_t llama_kv_cache_update(struct llama_context * ctx) {
    return llama_kv_cache_update_internal(*ctx);
}

// deprecated
size_t llama_get_state_size(struct llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(struct llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// TODO: replace all non-fatal assertions with returned errors or exceptions
struct llama_data_write {
    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size, int il) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_write() = default;

    void write_string(const std::string & str) {
        uint32_t str_size = str.size();

        write(&str_size,  sizeof(str_size));
        write(str.data(), str_size);
    }

    void write_model_info(const struct llama_context * ctx) {
        std::string arch_str = llama_model_arch_name(ctx->model.arch);
        write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    void write_rng(const std::mt19937 & rng) {
        std::ostringstream rng_ss;
        rng_ss << rng;

        const std::string & rng_str = rng_ss.str();

        write_string(rng_str);
    }

    void write_output_ids(const struct llama_context * ctx) {
        const uint32_t n_outputs = ctx->n_outputs;

        std::vector<int32_t> output_pos;

        const size_t    n_batch = ctx->cparams.n_batch;
        const auto & output_ids = ctx->output_ids;

        GGML_ASSERT(n_outputs <= ctx->output_size);

        output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch; ++i) {
            // map an output id to a position in the batch
            int32_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT((uint32_t) pos < n_outputs);
                output_pos[pos] = i;
            }
        }

        write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            write(output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    void write_logits(const struct llama_context * ctx) {
        const uint64_t logits_size = std::min((uint64_t) ctx->logits_size, (uint64_t) ctx->n_outputs * ctx->model.hparams.n_vocab);

        write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            write(ctx->logits, logits_size * sizeof(float));
        }
    }

    void write_embeddings(const struct llama_context * ctx) {
        const uint64_t embeddings_size = std::min((uint64_t) ctx->embd_size, (uint64_t) ctx->n_outputs * ctx->model.hparams.n_embd);

        write(&embeddings_size, sizeof(embeddings_size));

        if (embeddings_size) {
            write(ctx->embd, embeddings_size * sizeof(float));
        }
    }

    void write_kv_cache_meta(const llama_kv_cache & kv_self, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) {

        for (const auto & range : cell_ranges) {
            for (uint32_t i = range.first; i < range.second; ++i) {
                const auto & cell = kv_self.cells[i];
                const llama_pos pos      = cell.pos;
                const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

                write(&pos,      sizeof(pos));
                write(&n_seq_id, sizeof(n_seq_id));

                if (n_seq_id) {
                    for (auto seq_id : cell.seq_id) {
                        write(&seq_id, sizeof(seq_id));
                    }
                }
            }
        }
    }

    void write_kv_cache_data(const struct llama_context * ctx, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) {
        const struct llama_kv_cache & kv_self = ctx->kv_self;
        const struct llama_hparams & hparams = ctx->model.hparams;

        // v_state: 0 -> not transposed V cache
        //          1 -> transposed V cache
        //          2 -> no V cache (as it may be the case with MLA)
        const uint32_t v_state = kv_self.v_l.empty() ? 2 : kv_self.v_trans ? 1 : 0;
        const uint32_t n_layer = kv_self.k_l.size();

        write(&v_state, sizeof(v_state));
        write(&n_layer, sizeof(n_layer));

        std::vector<uint8_t> tmp_buf;

        // Iterate and write all the keys first, each row is a cell
        // Get whole range at a time
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();
            const uint32_t n_embd_head_qk_rope = hparams.n_rot;
            const uint32_t kv_lora_rank = hparams.n_lora_kv;

            // Write key type
            const int32_t k_type_i = (int32_t)kv_self.k_l[il]->type;
            write(&k_type_i, sizeof(k_type_i));

            // Write row size of key
            const uint64_t k_size_row = (ctx->cparams.mla_attn == 0) ? ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa) : ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope);
            write(&k_size_row, sizeof(k_size_row));

            // Read each range of cells of k_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * k_size_row;
                write_tensor_data(kv_self.k_l[il], range.first * k_size_row, buf_size, il);
            }
        }

        if (v_state == 0) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Write value type
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                write(&v_type_i, sizeof(v_type_i));

                // Write row size of value
                const uint64_t v_size_row = ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa);
                write(&v_size_row, sizeof(v_size_row));

                // Read each range of cells of v_size length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t buf_size = range_size * v_size_row;
                    write_tensor_data(kv_self.v_l[il], range.first * v_size_row, buf_size, il);
                }
            }
        }
        else if (v_state == 1) {
            // When v is transposed, we also need the element size and get the element ranges from each row
            const uint32_t kv_size = kv_self.size;
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Write value type
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                write(&v_type_i, sizeof(v_type_i));

                // Write element size
                const uint32_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
                write(&v_size_el, sizeof(v_size_el));

                // Write GQA embedding size
                write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

                // For each row, we get the element values of each cell
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    // Read each range of cells of v_size_el length each into tmp_buf and write out
                    for (const auto & range : cell_ranges) {
                        const size_t range_size = range.second - range.first;
                        const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                        const size_t buf_size = range_size * v_size_el;
                        write_tensor_data(kv_self.v_l[il], src_offset, buf_size, il);
                    }
                }
            }
        }
    }

    void write_kv_cache(const struct llama_context * ctx, llama_seq_id seq_id = -1) {
        const struct llama_kv_cache & kv_self = ctx->kv_self;
        std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
        uint32_t cell_count = 0;

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = kv_self.size;
        for (uint32_t i = 0; i < kv_self.size; ++i) {
            const auto & cell = kv_self.cells[i];
            if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
                ++cell_count;
                if (cell_range_begin == kv_self.size) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != kv_self.size) {
                    cell_ranges.emplace_back(cell_range_begin, i);
                    cell_range_begin = kv_self.size;
                }
            }
        }
        if (cell_range_begin != kv_self.size) {
            cell_ranges.emplace_back(cell_range_begin, kv_self.size);
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cell_ranges) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        write(&cell_count, sizeof(cell_count));

        write_kv_cache_meta(kv_self, cell_ranges, seq_id);
        write_kv_cache_data(ctx, cell_ranges);
    }
};

struct llama_data_read {
    virtual const uint8_t * read(size_t size) = 0;
    virtual void read_to(void * dst, size_t size) = 0;
    virtual size_t get_size_read() = 0;
    virtual ~llama_data_read() = default;

    void read_string(std::string & str) {
        uint32_t str_size;
        read_to(&str_size, sizeof(str_size));

        str.assign((const char *) read(str_size), str_size);
    }

    // validate model information
    void read_model_info(const struct llama_context * ctx) {
        std::string cur_arch_str = llama_model_arch_name(ctx->model.arch);
        std::string arch_str;
        read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    void read_rng(std::mt19937 & rng) {
        std::string rng_str;
        read_string(rng_str);

        std::istringstream rng_ss(rng_str);
        rng_ss >> rng;

        if (rng_ss.fail()) {
            throw std::runtime_error("failed to load RNG state");
        }
    }

    void read_output_ids(struct llama_context * ctx) {
        std::vector<int32_t> output_pos;

        uint32_t n_outputs;
        read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > llama_output_reserve(*ctx, n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        if (n_outputs) {
            output_pos.resize(n_outputs);
            read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= ctx->cparams.n_batch) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, ctx->cparams.n_batch));
                }
                ctx->output_ids[id] = i;
            }

            ctx->n_outputs = n_outputs;
        }
    }

    void read_logits(struct llama_context * ctx) {
        uint64_t logits_size;
        read_to(&logits_size, sizeof(logits_size));

        if (ctx->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            read_to(ctx->logits, logits_size * sizeof(float));
        }
    }

    void read_embeddings(struct llama_context * ctx) {
        uint64_t embeddings_size;
        read_to(&embeddings_size, sizeof(embeddings_size));

        if (ctx->embd_size < embeddings_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embeddings_size) {
            read_to(ctx->embd, embeddings_size * sizeof(float));
        }
    }

    bool read_kv_cache_meta(struct llama_context * ctx, uint32_t cell_count, llama_seq_id dest_seq_id = -1) {
        struct llama_kv_cache & kv_self = ctx->kv_self;

        if (dest_seq_id != -1) {
            // single sequence

            llama_kv_cache_seq_rm(kv_self, dest_seq_id, -1, -1);

            llama_batch batch = llama_batch_init(cell_count, 0, 1);
            batch.n_tokens = cell_count;
            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_pos pos;
                uint32_t n_seq_id;

                read_to(&pos, sizeof(pos));
                read_to(&n_seq_id, sizeof(n_seq_id));

                if (n_seq_id != 0) {
		    llama_batch_free(batch);
                    LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                    return false;
                }

                batch.pos[i] = pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = dest_seq_id;
            }
            if (!llama_kv_cache_find_slot(kv_self, batch)) {
                llama_batch_free(batch);
                LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
                return false;
            }

            // DEBUG CHECK: kv_self.head should be our first cell, kv_self.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
            // Assume that this is one contiguous block of cells
            GGML_ASSERT(kv_self.head + cell_count <= kv_self.size);
            GGML_ASSERT(kv_self.cells[kv_self.head].pos == batch.pos[0]);
            GGML_ASSERT(kv_self.cells[kv_self.head + cell_count - 1].pos == batch.pos[cell_count - 1]);
            GGML_ASSERT(kv_self.cells[kv_self.head].has_seq_id(dest_seq_id));
            GGML_ASSERT(kv_self.cells[kv_self.head + cell_count - 1].has_seq_id(dest_seq_id));

            // Cleanup
            llama_batch_free(batch);
        } else {
            // whole KV cache restore

            if (cell_count > kv_self.size) {
                LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
                return false;
            }

            llama_kv_cache_clear(kv_self);

            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_kv_cell & cell = kv_self.cells[i];

                llama_pos pos;
                uint32_t  n_seq_id;

                read_to(&pos,      sizeof(pos));
                read_to(&n_seq_id, sizeof(n_seq_id));

                cell.pos = pos;

                for (uint32_t j = 0; j < n_seq_id; ++j) {
                    llama_seq_id seq_id;
                    read_to(&seq_id, sizeof(seq_id));

                    if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx)) {
                        LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, llama_n_seq_max(ctx));
                        return false;
                    }

                    cell.seq_id.insert(seq_id);
                }
            }

            kv_self.head = 0;
            kv_self.used = cell_count;
        }

        return true;
    }

    void read_kv_cache_data_split(llama_context * ctx, ggml_tensor * tensor, const uint8_t * data, size_t head, size_t row_size, int nrows, int il) {
        GGML_ASSERT(il >= 0 && il < int(ctx->model.layers.size()));
        GGML_ASSERT(ggml_internal_get_type_traits(tensor->type).row_meta_size == 0);
        auto kv = tensor->ne[1] > 1 ? ctx->model.layers[il].wk : ctx->model.layers[il].wv;
        auto extra = (ggml_split_tensor_t *)tensor->extra;
        auto kv_extra = (ggml_split_tensor_t *)kv->extra;
        GGML_ASSERT(extra && kv_extra);
        auto ne = kv->ne[1];
        size_t sum_ne = 0;
        size_t sum_split_row_size = 0;
        GGML_ASSERT(row_size == ggml_row_size(tensor->type, ne));
        std::vector<uint8_t> aux;
        for (int id = 0; id < extra->n_device; ++id) {
            auto split = extra->splits[id];
            auto kv_split = kv_extra->splits[id];
            GGML_ASSERT((split && kv_split) || (!split && !kv_split));
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            auto split_row_size = ggml_row_size(tensor->type, kv_split->ne[1]);
            aux.resize(split_row_size*nrows);
            auto src = data + sum_split_row_size;
            auto dst = aux.data();
            for (int row = 0; row < nrows; ++row) {
                std::memcpy(dst, src, split_row_size);
                dst += split_row_size;
                src += row_size;
            }
            ggml_backend_tensor_set(split, aux.data(), head*split_row_size, nrows*split_row_size);
            sum_ne += kv_split->ne[1];
            sum_split_row_size += split_row_size;
        }
        GGML_ASSERT(sum_ne == ne);
        GGML_ASSERT(sum_split_row_size == row_size);
    }

    bool read_kv_cache_data(struct llama_context * ctx, uint32_t cell_count) {
        const struct llama_hparams & hparams = ctx->model.hparams;
        struct llama_kv_cache & kv_self = ctx->kv_self;

        // v_state: 0 -> not transposed V cache
        //          1 -> transposed V cache
        //          2 -> no V cache (as it may be the case with MLA)
        uint32_t v_state;
        uint32_t n_layer;
        read_to(&v_state, sizeof(v_state));
        read_to(&n_layer, sizeof(n_layer));

        if (n_layer != kv_self.k_l.size()) {
            LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
            return false;
        }
        if (cell_count > kv_self.size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, kv_self.size);
            return false;
        }

	// Currently the only way there is no V cache (and thus v_state is 2) requires flash_attn, and flash_attn sets kv_self.v_trans to false
        if (kv_self.v_trans != (v_state == 1)) {
            LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
            return false;
        }

        // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();
            const uint32_t n_embd_head_qk_rope = hparams.n_rot;
            const uint32_t kv_lora_rank = hparams.n_lora_kv;


            // Read type of key
            int32_t k_type_i_ref;
            read_to(&k_type_i_ref, sizeof(k_type_i_ref));
            const int32_t k_type_i = (int32_t)kv_self.k_l[il]->type;
            if (k_type_i != k_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
                return false;
            }

            // Read row size of key
            uint64_t k_size_row_ref;
            read_to(&k_size_row_ref, sizeof(k_size_row_ref));
            const uint64_t k_size_row = (ctx->cparams.mla_attn == 0) ? ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa) : ggml_row_size(kv_self.k_l[il]->type, kv_lora_rank + n_embd_head_qk_rope);
            if (k_size_row != k_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the keys for the whole cell range
                if (kv_self.k_l[il]->extra) {
                    read_kv_cache_data_split(ctx, kv_self.k_l[il], read(cell_count * k_size_row), kv_self.head, k_size_row, cell_count, il);
                } else {
                    ggml_backend_tensor_set(kv_self.k_l[il], read(cell_count * k_size_row), kv_self.head * k_size_row, cell_count * k_size_row);
                }
            }
        }

        if (v_state == 0) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Read type of value
                int32_t v_type_i_ref;
                read_to(&v_type_i_ref, sizeof(v_type_i_ref));
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                // Read row size of value
                uint64_t v_size_row_ref;
                read_to(&v_size_row_ref, sizeof(v_size_row_ref));
                const size_t v_size_row = ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa);
                if (v_size_row != v_size_row_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                    return false;
                }

                if (cell_count) {
                    // Read and set the values for the whole cell range
                    if (kv_self.v_l[il]->extra) {
                        read_kv_cache_data_split(ctx, kv_self.v_l[il], read(cell_count * v_size_row), kv_self.head, v_size_row, cell_count, il);
                    } else {
                        ggml_backend_tensor_set(kv_self.v_l[il], read(cell_count * v_size_row), kv_self.head * v_size_row, cell_count * v_size_row);
                    }
                }
            }
        }
        else if (v_state == 1) {
            // For each layer, read the values for each cell (transposed)
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Read type of value
                int32_t v_type_i_ref;
                read_to(&v_type_i_ref, sizeof(v_type_i_ref));
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                // Read element size of value
                uint32_t v_size_el_ref;
                read_to(&v_size_el_ref, sizeof(v_size_el_ref));
                const size_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
                if (v_size_el != v_size_el_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                    return false;
                }

                // Read GQA embedding size
                uint32_t n_embd_v_gqa_ref;
                read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
                if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                    return false;
                }

                if (cell_count) {
                    if (kv_self.v_l[il]->extra) {
                        throw std::runtime_error("Transposed V cache is not sypported with split mode 'graph'");
                    }
                    // For each row in the transposed matrix, read the values for the whole cell range
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (kv_self.head + j * kv_self.size) * v_size_el;
                        ggml_backend_tensor_set(kv_self.v_l[il], read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                    }
                }
            }
        }
        return true;
    }

    void read_kv_cache(struct llama_context * ctx, llama_seq_id seq_id = -1) {
        uint32_t cell_count;
        read_to(&cell_count, sizeof(cell_count));

        bool res = read_kv_cache_meta(ctx, cell_count, seq_id) && read_kv_cache_data(ctx, cell_count);

        if (!res) {
            if (seq_id == -1) {
                llama_kv_cache_clear(ctx);
            } else {
                llama_kv_cache_seq_rm(ctx, seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
};

struct llama_data_write_dummy : llama_data_write {
    size_t size_written = 0;

    llama_data_write_dummy() {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor_data(const struct ggml_tensor * /* tensor */, size_t /* offset */, size_t size, int /* il */) override {
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_write_buffer : llama_data_write {
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    const llama_model & model;

    std::vector<uint8_t> aux_buffer;

    llama_data_write_buffer(uint8_t * p, size_t len, const llama_model & _model) : ptr(p), buf_size(len), model(_model) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size, int il) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        if (tensor->extra) {
            get_tensor_data_split(tensor, offset, size, il);
        } else {
            ggml_backend_tensor_get(tensor, ptr, offset, size);
        }
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void get_tensor_data_split(const ggml_tensor * tensor, size_t offset, size_t size, int il) {
        auto tt = ggml_internal_get_type_traits(tensor->type);
        if (tt.row_meta_size > 0) {
            throw std::runtime_error(std::string{"Split cache for type "} + ggml_type_name(tensor->type) + " is not supported");
        }
        GGML_ASSERT(il >= 0 && il < int(model.layers.size()));
        auto kv = tensor->ne[1] > 1 ? model.layers[il].wk : model.layers[il].wv;
        get_tensor_data_split(ptr, tensor, kv, aux_buffer, offset, size);
    }

    static void get_tensor_data_split(uint8_t * ptr, const ggml_tensor * tensor, const ggml_tensor * kv,
            std::vector<uint8_t> & aux_buffer, size_t offset, size_t size) {
        auto ne = kv->ne[1];
        auto full_row_size = ggml_row_size(tensor->type, ne);
        GGML_ASSERT(offset % full_row_size == 0);
        GGML_ASSERT(size   % full_row_size == 0);
        auto first_row = offset / full_row_size;
        auto num_rows  = size / full_row_size;
        auto extra = (const ggml_split_tensor_t *)tensor->extra;
        auto kv_extra = (const ggml_split_tensor_t *)kv->extra;
        GGML_ASSERT(extra && kv_extra);
        size_t split_offset = 0;
        size_t total_size   = 0;
        for (int id = 0; id < extra->n_device; ++id) {
            auto split = extra->splits[id];
            auto kv_split = kv_extra->splits[id];
            GGML_ASSERT((split && kv_split) || (!split && !kv_split));
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            auto split_row_size = ggml_row_size(tensor->type, kv_split->ne[1]);
            auto split_size = split_row_size * num_rows;
            if (split_size > aux_buffer.size()) aux_buffer.resize(split_size);
            ggml_backend_tensor_get(split, aux_buffer.data(), first_row*split_row_size, split_size);
            auto dst = ptr + split_offset;
            auto src = aux_buffer.data();
            for (int row = 0; row < (int)num_rows; ++row) {
                std::memcpy(dst, src, split_row_size);
                dst += full_row_size;
                src += split_row_size;
            }
            split_offset += split_row_size;
            total_size   += split_row_size * num_rows;
        }
        GGML_ASSERT(total_size == size);
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_read_buffer : llama_data_read {
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    llama_data_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t get_size_read() override {
        return size_read;
    }
};

struct llama_data_write_file : llama_data_write {
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
    std::vector<uint8_t> aux_buffer;

    const llama_model & model;

    llama_data_write_file(llama_file * f, const llama_model & _model) : file(f), model(_model) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size, int il) override {
        temp_buffer.resize(size);
        if (tensor->extra) {
            get_tensor_data_split(tensor, offset, size, il);
        } else {
            ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        }
        write(temp_buffer.data(), temp_buffer.size());
    }

    void get_tensor_data_split(const struct ggml_tensor * tensor, size_t offset, size_t size, int il) {
        GGML_ASSERT(il >= 0 && il < int(model.layers.size()));
        auto kv = tensor->ne[1] > 1 ? model.layers[il].wk : model.layers[il].wv;
        temp_buffer.resize(size);
        llama_data_write_buffer::get_tensor_data_split(temp_buffer.data(), tensor, kv, aux_buffer, offset, size);
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_read_file : llama_data_read {
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;

    llama_data_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t get_size_read() override {
        return size_read;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_write_file data_ctx(&file);
 * llama_state_get_data_internal(ctx, data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_write_buffer data_ctx(buf.data(), max_size);
 * llama_state_get_data_internal(ctx, data_ctx);
 *
*/
static size_t llama_state_get_data_internal(struct llama_context * ctx, llama_data_write & data_ctx) {
    llama_synchronize(ctx);

    data_ctx.write_model_info(ctx);

    data_ctx.write_rng(ctx->sampling.rng);

    // copy outputs
    data_ctx.write_output_ids(ctx);
    data_ctx.write_logits(ctx);
    data_ctx.write_embeddings(ctx);

    data_ctx.write_kv_cache(ctx);

    return data_ctx.get_size_written();
}

size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size) {
    llama_data_write_buffer data_ctx(dst, size, ctx->model);
    try {
        return llama_state_get_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(struct llama_context * ctx) {
    llama_data_write_dummy data_ctx;
    try {
        return llama_state_get_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_set_data_internal(struct llama_context * ctx, llama_data_read & data_ctx) {
    llama_synchronize(ctx);

    data_ctx.read_model_info(ctx);

    // set rng
    data_ctx.read_rng(ctx->sampling.rng);

    // set outputs
    data_ctx.read_output_ids(ctx);
    data_ctx.read_logits(ctx);
    data_ctx.read_embeddings(ctx);

    data_ctx.read_kv_cache(ctx);

    return data_ctx.get_size_read();
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size) {
    llama_data_read_buffer data_ctx(src, size);
    try {
        return llama_state_set_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static bool llama_state_load_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_data_read_file data_ctx(&file);
        const size_t n_read = llama_state_set_data_internal(ctx, data_ctx);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }
    return true;
}

bool llama_state_load_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_state_load_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

static bool llama_state_save_file_internal(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_write_file data_ctx(&file, ctx->model);
    llama_state_get_data_internal(ctx, data_ctx);

    return true;
}

bool llama_state_save_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    try {
        return llama_state_save_file_internal(ctx, path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

static size_t llama_state_seq_get_data_internal(struct llama_context * ctx, llama_data_write & data_ctx, llama_seq_id seq_id) {
    llama_synchronize(ctx);

    data_ctx.write_kv_cache(ctx, seq_id);

    return data_ctx.get_size_written();
}

size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_data_write_dummy data_ctx;
    return llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
}

size_t llama_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    llama_data_write_buffer data_ctx(dst, size, ctx->model);
    try {
        return llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_seq_set_data_internal(struct llama_context * ctx, llama_data_read & data_ctx, llama_seq_id dest_seq_id) {
    llama_synchronize(ctx);

    data_ctx.read_kv_cache(ctx, dest_seq_id);

    return data_ctx.get_size_read();
}

size_t llama_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id) {
    llama_data_read_buffer data_ctx(src, size);
    try {
        return llama_state_seq_set_data_internal(ctx, data_ctx, dest_seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_seq_save_file_internal(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_write_file data_ctx(&file, ctx->model);
    llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + data_ctx.get_size_written());
    return res;
}

static size_t llama_state_seq_load_file_internal(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_data_read_file data_ctx(&file);
        const size_t nread = llama_state_seq_set_data_internal(ctx, data_ctx, dest_seq_id);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_state_seq_save_file(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    try {
        return llama_state_seq_save_file_internal(ctx, filepath, seq_id, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_state_seq_load_file_internal(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

void llama_set_n_threads(struct llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch) {
    ctx->cparams.n_threads       = n_threads;
    ctx->cparams.n_threads_batch = n_threads_batch;
}

uint32_t llama_n_threads(struct llama_context * ctx) {
    return ctx->cparams.n_threads;
}

uint32_t llama_n_threads_batch(struct llama_context * ctx) {
    return ctx->cparams.n_threads_batch;
}

void llama_set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->abort_callback      = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

void llama_set_embeddings(struct llama_context * ctx, bool embeddings) {
    ctx->cparams.embeddings = embeddings;
}

void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn) {
    ctx->cparams.causal_attn = causal_attn;
}

struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens,
               llama_pos   pos_0,
            llama_seq_id   seq_id) {
    return {
        /*n_tokens       =*/ n_tokens,
        /*tokens         =*/ tokens,
        /*embd           =*/ nullptr,
        /*pos            =*/ nullptr,
        /*n_seq_id       =*/ nullptr,
        /*seq_id         =*/ nullptr,
        /*logits         =*/ nullptr,
        /*all_pos_0      =*/ pos_0,
        /*all_pos_1      =*/ 1,
        /*all_seq_id     =*/ seq_id,
    };
}

struct llama_batch llama_batch_init(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    llama_batch batch = { 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, };

    if (embd) {
        batch.embd = (float *) malloc(sizeof(float) * n_tokens_alloc * embd);
    } else {
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens_alloc);
    }

    batch.pos      = (llama_pos *)     malloc(sizeof(llama_pos)      * n_tokens_alloc);
    batch.n_seq_id = (int32_t *)       malloc(sizeof(int32_t)        * n_tokens_alloc);
    batch.seq_id   = (llama_seq_id **) malloc(sizeof(llama_seq_id *) * (n_tokens_alloc + 1));
    for (int i = 0; i < n_tokens_alloc; ++i) {
        batch.seq_id[i] = (llama_seq_id *) malloc(sizeof(llama_seq_id) * n_seq_max);
    }
    batch.seq_id[n_tokens_alloc] = nullptr;

    batch.logits   = (int8_t *)        malloc(sizeof(int8_t)         * n_tokens_alloc);

    return batch;
}

void llama_batch_free(struct llama_batch batch) {
    if (batch.token)    free(batch.token);
    if (batch.embd)     free(batch.embd);
    if (batch.pos)      free(batch.pos);
    if (batch.n_seq_id) free(batch.n_seq_id);
    if (batch.seq_id) {
        for (int i = 0; batch.seq_id[i] != nullptr; ++i) {
            free(batch.seq_id[i]);
        }
        free(batch.seq_id);
    }
    if (batch.logits)   free(batch.logits);
}

int32_t llama_encode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = llama_encode_internal(*ctx, batch);
    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        struct llama_context * ctx,
          struct llama_batch   batch) {
    const int ret = llama_decode_internal(*ctx, batch);
    if (ret < 0) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

void llama_synchronize(struct llama_context * ctx) {
    ggml_backend_sched_synchronize(ctx->sched);

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (ctx->n_queued_tokens == 1) {
        ctx->t_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        ctx->n_eval++;
    } else if (ctx->n_queued_tokens > 1) {
        ctx->t_p_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        ctx->n_p_eval += ctx->n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (ctx->n_queued_tokens > 0 && !ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    ctx->n_queued_tokens = 0;
    ctx->t_compute_start_us = 0;
}

float * llama_get_logits(struct llama_context * ctx) {
    llama_synchronize(ctx);

    return ctx->logits;
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    int32_t j = -1;
    llama_synchronize(ctx);

    try {
        if (ctx->logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {
            j = ctx->n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", ctx->n_outputs));
            }
        } else if ((size_t) i >= ctx->output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %lu)", ctx->output_ids.size()));
        } else {
            j = ctx->output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= ctx->n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
        }

        return ctx->logits + j*ctx->model.hparams.n_vocab;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#endif
        return nullptr;
    }
}

float * llama_get_embeddings(struct llama_context * ctx) {
    llama_synchronize(ctx);

    return ctx->embd;
}

float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i) {
    int32_t j = -1;

    llama_synchronize(ctx);

    try {
        if (ctx->embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        if (i < 0) {
            j = ctx->n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", ctx->n_outputs));
            }
        } else if ((size_t) i >= ctx->output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %lu)", ctx->output_ids.size()));
        } else {
            j = ctx->output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= ctx->n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
        }

        return ctx->embd + j*ctx->model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#endif
        return nullptr;
    }
}

float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_synchronize(ctx);

    auto it = ctx->embd_seq.find(seq_id);
    if (it == ctx->embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

//
// vocab
//

const char * llama_token_get_text(const struct llama_model * model, llama_token token) {
    return model->vocab.token_get_text(token);
}

float llama_token_get_score(const struct llama_model * model, llama_token token) {
    return model->vocab.token_get_score(token);
}

enum llama_token_attr llama_token_get_attr(const struct llama_model * model, llama_token token) {
    return model->vocab.token_get_attr(token);
}

bool llama_token_is_eog(const struct llama_model * model, llama_token token) {
    return model->vocab.is_eog(token);
}

bool llama_token_is_control(const struct llama_model * model, llama_token token) {
    return model->vocab.is_control(token);
}

llama_token llama_token_bos(const struct llama_model * model) {
    return model->vocab.token_bos();
}

llama_token llama_token_eos(const struct llama_model * model) {
    return model->vocab.token_eos();
}

llama_token llama_token_sep(const struct llama_model * model) {
    return model->vocab.token_sep();
}

llama_token llama_token_nl (const struct llama_model * model) {
    return model->vocab.token_nl();
}

llama_token llama_token_pad(const struct llama_model * model) {
    return model->vocab.token_pad();
}

int32_t llama_add_bos_token(const struct llama_model * model) {
    return model->vocab.get_add_bos();
}

int32_t llama_add_eos_token(const struct llama_model * model) {
    return model->vocab.get_add_eos();
}

llama_token llama_token_prefix(const struct llama_model * model) {
    return model->vocab.token_prefix();
}

llama_token llama_token_middle(const struct llama_model * model) {
    return model->vocab.token_middle();
}

llama_token llama_token_suffix(const struct llama_model * model) {
    return model->vocab.token_suffix();
}

llama_token llama_token_eot(const struct llama_model * model) {
    return model->vocab.token_eot();
}

//
// tokenization
//

int32_t llama_tokenize(
    const struct llama_model * model,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    return model->vocab.tokenize(text, text_len, tokens, n_tokens_max, add_special, parse_special);
}

int32_t llama_token_to_piece(
    const struct llama_model * model,
                 llama_token   token,
                        char * buf,
                     int32_t   length,
                     int32_t   lstrip,
                        bool   special) {
    return model->vocab.token_to_piece(token, buf, length, lstrip, special);
}

int32_t llama_detokenize(
    const struct llama_model * model,
           const llama_token * tokens,
                     int32_t   n_tokens,
                        char * text,
                     int32_t   text_len_max,
                        bool   remove_special,
                        bool   unparse_special) {
    return model->vocab.detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}

//
// chat templates
//

static llm_chat_template llama_chat_detect_template(const std::string & tmpl) {
    if (auto it = LLM_CHAT_TEMPLATES.find(tmpl); it != LLM_CHAT_TEMPLATES.end()) {
        return it->second;
    }
    auto tmpl_contains = [&tmpl](const char * haystack) -> bool {
        return tmpl.find(haystack) != std::string::npos;
    };
    if (tmpl_contains("<|im_start|>")) {
        return LLM_CHAT_TEMPLATE_CHATML;
    } else if (tmpl.find("mistral") == 0 || tmpl_contains("[INST]")) {
        if (tmpl_contains("[SYSTEM_PROMPT]")) {
            return LLM_CHAT_TEMPLATE_MISTRAL_V7;
        } else if (
            // catches official 'v1' template
            tmpl_contains("' [INST] ' + system_message")
            // catches official 'v3' and 'v3-tekken' templates
            || tmpl_contains("[AVAILABLE_TOOLS]")
        ) {
            // Official mistral 'v1', 'v3' and 'v3-tekken' templates
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
            if (tmpl_contains(" [INST]")) {
                return LLM_CHAT_TEMPLATE_MISTRAL_V1;
            } else if (tmpl_contains("\"[INST]\"")) {
                return LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN;
            }
            return LLM_CHAT_TEMPLATE_MISTRAL_V3;
        } else {
            // llama2 template and its variants
            // [variant] support system message
            // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            bool support_system_message = tmpl_contains("<<SYS>>");
            bool add_bos_inside_history = tmpl_contains("bos_token + '[INST]");
            bool strip_message = tmpl_contains("content.strip()");
            if (strip_message) {
                return LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
            } else if (add_bos_inside_history) {
                return LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
            } else if (support_system_message) {
                return LLM_CHAT_TEMPLATE_LLAMA_2_SYS;
            } else {
                return LLM_CHAT_TEMPLATE_LLAMA_2;
            }
        }
    } else if (tmpl_contains("[gMASK]sop")) {
        // chatglm3-6b
        return LLM_CHAT_TEMPLATE_CHATGLM_3;
    } else if (tmpl_contains("[gMASK]<sop>")) {
        return LLM_CHAT_TEMPLATE_CHATGLM_4;
    } else if (tmpl_contains("<|assistant|>") && tmpl_contains("<|end|>")) {
        return LLM_CHAT_TEMPLATE_PHI_3;
    } else if (tmpl_contains("<|assistant|>") && tmpl_contains("<|user|>")) {
        return LLM_CHAT_TEMPLATE_FALCON_3;
    } else if (tmpl == "falcon_e" && (tmpl_contains("assistant") && tmpl_contains("user"))) {
        return LLM_CHAT_TEMPLATE_FALCON_E;
    } else if (tmpl_contains("<|user|>") && tmpl_contains("<|endoftext|>")) {
        return LLM_CHAT_TEMPLATE_ZEPHYR;
    } else if (tmpl_contains("bos_token + message['role']")) {
        return LLM_CHAT_TEMPLATE_MONARCH;
    } else if (tmpl_contains("<start_of_turn>")) {
        return LLM_CHAT_TEMPLATE_GEMMA;
    } else if (tmpl_contains("'\\n\\nAssistant: ' + eos_token")) {
        // OrionStarAI/Orion-14B-Chat
        return LLM_CHAT_TEMPLATE_ORION;
    } else if (tmpl_contains("GPT4 Correct ")) {
        // openchat/openchat-3.5-0106
        return LLM_CHAT_TEMPLATE_OPENCHAT;
    } else if (tmpl_contains("USER: ") && tmpl_contains("ASSISTANT: ")) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        if (tmpl_contains("SYSTEM: ")) {
            return LLM_CHAT_TEMPLATE_VICUNA_ORCA;
        }
        return LLM_CHAT_TEMPLATE_VICUNA;
    } else if (tmpl_contains("### Instruction:") && tmpl_contains("<|EOT|>")) {
        // deepseek-ai/deepseek-coder-33b-instruct
        return LLM_CHAT_TEMPLATE_DEEPSEEK;
    } else if (tmpl_contains("<|START_OF_TURN_TOKEN|>") && tmpl_contains("<|USER_TOKEN|>")) {
        // CohereForAI/c4ai-command-r-plus
        return LLM_CHAT_TEMPLATE_COMMAND_R;
    } else if (tmpl_contains("<|start_header_id|>") && tmpl_contains("<|end_header_id|>")) {
        return LLM_CHAT_TEMPLATE_LLAMA_3;
    } else if (tmpl_contains(LU8("<用户>"))) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        return LLM_CHAT_TEMPLATE_MINICPM;
    } else if (tmpl_contains("'Assistant: ' + message['content'] + eos_token")) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_2;
    } else if (tmpl_contains(LU8("<｜Assistant｜>")) && tmpl_contains(LU8("<｜User｜>")) && tmpl_contains(LU8("<｜end▁of▁sentence｜>"))) {
         // original: if (tmpl_contains(LU8("'<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>'"))) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_3;
    } else if (tmpl_contains("[|system|]") && tmpl_contains("[|assistant|]") && tmpl_contains("[|endofturn|]")) {
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        return LLM_CHAT_TEMPLATE_EXAONE_3;
    } else if (tmpl_contains("rwkv-world")) {
        return LLM_CHAT_TEMPLATE_RWKV_WORLD;
    } else if (tmpl_contains("<|start_of_role|>")) {
        return LLM_CHAT_TEMPLATE_GRANITE;
    } else if (tmpl_contains("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]")) {
        return LLM_CHAT_TEMPLATE_GIGACHAT;
    } else if (tmpl_contains("<|role_start|>")) {
        return LLM_CHAT_TEMPLATE_MEGREZ;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("'HUMAN'")) {
        return LLM_CHAT_TEMPLATE_BAILING;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("\"HUMAN\"") && tmpl_contains("<think>")) {
        return LLM_CHAT_TEMPLATE_BAILING_THINK;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("<role>HUMAN</role>") && tmpl_contains("<|role_end|>")) {
        return LLM_CHAT_TEMPLATE_BAILING2;
    } else if (tmpl_contains("<|header_start|>") && tmpl_contains("<|header_end|>")) {
        return LLM_CHAT_TEMPLATE_LLAMA4;
    } else if (tmpl_contains("<|endofuserprompt|>")) {
        return LLM_CHAT_TEMPLATE_DOTS1;
    } else if (tmpl_contains("<|startoftext|>") && tmpl_contains("<|extra_4|>")) {
        return LLM_CHAT_TEMPLATE_HUNYUAN_MOE;
    } else if (tmpl_contains("<|im_middle|>") && tmpl_contains("<|im_end|>")) {
        return LLM_CHAT_TEMPLATE_KIMI_K2;
    } else if (tmpl_contains("'Assistant: '  + message['content'] + '<|separator|>")) {
        return LLM_CHAT_TEMPLATE_GROK_2;
    } else if (tmpl_contains("<|start|>") && tmpl_contains("<|channel|>")) {
        return LLM_CHAT_TEMPLATE_OPENAI_MOE;
    }
    return LLM_CHAT_TEMPLATE_UNKNOWN;
}

static int32_t llama_chat_apply_template_internal(
    const llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggerganov/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == LLM_CHAT_TEMPLATE_CHATML) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7) {
        // Official mistral 'v7' template
        // See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411#basic-instruct-template-v7
        for (auto message : chat) {
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                ss << "[SYSTEM_PROMPT] " << content << "[/SYSTEM_PROMPT]";
            } else if (role == "user") {
                ss << "[INST] " << content << "[/INST]";
            }
            else {
                ss << " " << content << "</s>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1
            || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3
            || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN) {
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
        std::string leading_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1 ? " " : "";
        std::string trailing_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN ? "" : " ";
        bool trim_assistant_message = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3;
        bool is_inside_turn = false;
        for (auto message : chat) {
            if (!is_inside_turn) {
                ss << leading_space << "[INST]" << trailing_space;
                is_inside_turn = true;
            }
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                ss << content << "\n\n";
            } else if (role == "user") {
                ss << content << leading_space << "[/INST]";
            } else {
                ss << trailing_space << (trim_assistant_message ? trim(content) : content) << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (
            tmpl == LLM_CHAT_TEMPLATE_LLAMA_2
            || tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS
            || tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS
            || tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP) {
        // llama2 template and its variants
        // [variant] support system message
        // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        bool support_system_message = tmpl != LLM_CHAT_TEMPLATE_LLAMA_2;
        // [variant] add BOS inside history
        bool add_bos_inside_history = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
        // [variant] trim spaces from the input message
        bool strip_message = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
        // construct the prompt
        bool is_inside_turn = true; // skip BOS at the beginning
        ss << "[INST] ";
        for (auto message : chat) {
            std::string content = strip_message ? trim(message->content) : message->content;
            std::string role(message->role);
            if (!is_inside_turn) {
                is_inside_turn = true;
                ss << (add_bos_inside_history ? "<s>[INST] " : "[INST] ");
            }
            if (role == "system") {
                if (support_system_message) {
                    ss << "<<SYS>>\n" << content << "\n<</SYS>>\n\n";
                } else {
                    // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                    ss << content << "\n";
                }
            } else if (role == "user") {
                ss << content << " [/INST]";
            } else {
                ss << content << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_PHI_3) {
        // Phi 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>\n" << message->content << "<|end|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_FALCON_3) {
        // Falcon 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>\n" << message->content << "\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_FALCON_E) {
        // Falcon Edge
        for (auto message : chat) {
            std::string role(message->role);
            ss << role << message->content << "\n";
        }
        if (add_ass) {
            ss << "assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_ZEPHYR) {
        // zephyr template
        for (auto message : chat) {
            ss << "<|" << message->role << "|>" << "\n" << message->content << "<|endoftext|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MONARCH) {
        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        for (auto message : chat) {
            std::string bos = (message == chat.front()) ? "" : "<s>"; // skip BOS for first message
            ss << bos << message->role << "\n" << message->content << "</s>\n";
        }
        if (add_ass) {
            ss << "<s>assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GEMMA) {
        // google/gemma-7b-it
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                system_prompt = trim(message->content);
                continue;
            }
            // in gemma, "assistant" is "model"
            role = role == "assistant" ? "model" : message->role;
            ss << "<start_of_turn>" << role << "\n";
            if (!system_prompt.empty() && role != "model") {
                ss << system_prompt << "\n\n";
                system_prompt = "";
            }
            ss << trim(message->content) << "<end_of_turn>\n";
        }
        if (add_ass) {
            ss << "<start_of_turn>model\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_ORION) {
        // OrionStarAI/Orion-14B-Chat
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message support, we will merge it with user prompt
                system_prompt = message->content;
                continue;
            } else if (role == "user") {
                ss << "Human: ";
                if (!system_prompt.empty()) {
                    ss << system_prompt << "\n\n";
                    system_prompt = "";
                }
                ss << message->content << "\n\nAssistant: </s>";
            } else {
                ss << message->content << "</s>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_OPENCHAT) {
        // openchat/openchat-3.5-0106,
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "<|end_of_turn|>";
            } else {
                role[0] = toupper(role[0]);
                ss << "GPT4 Correct " << role << ": " << message->content << "<|end_of_turn|>";
            }
        }
        if (add_ass) {
            ss << "GPT4 Correct Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_VICUNA || tmpl == LLM_CHAT_TEMPLATE_VICUNA_ORCA) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // Orca-Vicuna variant uses a system prefix
                if (tmpl == LLM_CHAT_TEMPLATE_VICUNA_ORCA) {
                    ss << "SYSTEM: " << message->content << "\n";
                } else {
                    ss << message->content << "\n\n";
                }
            } else if (role == "user") {
                ss << "USER: " << message->content << "\n";
            } else if (role == "assistant") {
                ss << "ASSISTANT: " << message->content << "</s>\n";
            }
        }
        if (add_ass) {
            ss << "ASSISTANT:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK) {
        // deepseek-ai/deepseek-coder-33b-instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content;
            } else if (role == "user") {
                ss << "### Instruction:\n" << message->content << "\n";
            } else if (role == "assistant") {
                ss << "### Response:\n" << message->content << "\n<|EOT|>\n";
            }
        }
        if (add_ass) {
            ss << "### Response:\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_COMMAND_R) {
        // CohereForAI/c4ai-command-r-plus
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "user") {
                ss << "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "assistant") {
                ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            }
        }
        if (add_ass) {
            ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_LLAMA_3) {
        // Llama 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << trim(message->content) << "<|eot_id|>";
        }
        if (add_ass) {
            ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_CHATGLM_3) {
        // chatglm3-6b
        ss << "[gMASK]" << "sop";
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n " << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_CHATGLM_4) {
        ss << "[gMASK]" << "<sop>";
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n" << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MINICPM) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "user") {
                ss << LU8("<用户>");
                ss << trim(message->content);
                ss << "<AI>";
            } else {
                ss << trim(message->content);
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK_2) {
        // DeepSeek-V2
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "\n\n";
            } else if (role == "assistant") {
                ss << "Assistant: " << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK_3) {
        // DeepSeek-V3
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << LU8("<｜User｜>") << message->content;
            } else if (role == "assistant") {
                ss << LU8("<｜Assistant｜>") << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << LU8("<｜Assistant｜>");
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_EXAONE_3) {
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "[|system|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "user") {
                ss << "[|user|]" << trim(message->content) << "\n";
            } else if (role == "assistant") {
                ss << "[|assistant|]" << trim(message->content) << "[|endofturn|]\n";
            }
        }
        if (add_ass) {
            ss << "[|assistant|]";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_RWKV_WORLD) {
        // this template requires the model to have "\n\n" as EOT token
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "user") {
                ss << "User: " << message->content << "\n\nAssistant:";
            } else {
                ss << message->content << "\n\n";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GRANITE) {
        // IBM Granite template
        for (const auto & message : chat) {
            std::string role(message->role);
            ss << "<|start_of_role|>" << role << "<|end_of_role|>";
            if (role == "assistant_tool_call") {
                ss << "<|tool_call|>";
            }
            ss << message->content << "<|end_of_text|>\n";
        }
        if (add_ass) {
            ss << "<|start_of_role|>assistant<|end_of_role|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GIGACHAT) {
        // GigaChat template
        bool has_system = !chat.empty() && std::string(chat[0]->role) == "system";

        // Handle system message if present
        if (has_system) {
            ss << "<s>" << chat[0]->content << "<|message_sep|>";
        } else {
            ss << "<s>";
        }

        // Process remaining messages
        for (size_t i = has_system ? 1 : 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << "user<|role_sep|>" << chat[i]->content << "<|message_sep|>"
                << "available functions<|role_sep|>[]<|message_sep|>";
            } else if (role == "assistant") {
                ss << "assistant<|role_sep|>" << chat[i]->content << "<|message_sep|>";
            }
        }

        // Add generation prompt if needed
        if (add_ass) {
            ss << "assistant<|role_sep|>";
        }
    }  else if (tmpl == LLM_CHAT_TEMPLATE_MEGREZ) {
        // Megrez template
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|role_start|>" << role << "<|role_end|>" << message->content << "<|turn_end|>";
        }

        if (add_ass) {
            ss << "<|role_start|>assistant<|role_end|>";
        }
    }  else if (tmpl == LLM_CHAT_TEMPLATE_BAILING || tmpl == LLM_CHAT_TEMPLATE_BAILING_THINK) {
        // Bailing (Ling/Ring) template
        for (auto message : chat) {
            std::string role(message->role);

            if (role == "user") {
                role = "HUMAN";
            } else {
                std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            }

            ss << "<role>" << role << "</role>" << message->content;
        }

        if (add_ass) {
            ss << "<role>ASSISTANT</role>";

            if (tmpl == LLM_CHAT_TEMPLATE_BAILING_THINK) {
                ss << "<think>";
            }
        }
    }  else if (tmpl == LLM_CHAT_TEMPLATE_BAILING2) {
        // Bailing2 (Ling 2.0) template
        bool has_system = !chat.empty() && std::string(chat[0]->role) == "system";

        if (!has_system) {
            ss << "<role>SYSTEM</role>detailed thinking off<|role_end|>";
        }

        for (auto message : chat) {
            std::string role(message->role);

            if (role == "user") {
                role = "HUMAN";
            } else {
                std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            }

            ss << "<role>" << role << "</role>" << message->content << "<|role_end|>";
        }
        if (add_ass) {
            ss << "<role>ASSISTANT</role>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_LLAMA4) {
        // Llama 4
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|header_start|>" << role << "<|header_end|>\n\n" << trim(message->content) << "<|eot|>";
        }
        if (add_ass) {
            ss << "<|header_start|>assistant<|header_end|>\n\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_BITNET) {
        // bitnet-25
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "System: ";
                ss << message->content;
            } else if (role == "user") {
                ss << "User: ";
                if (!system_prompt.empty()) {
                    ss << system_prompt;
                    system_prompt = "";
                }
                ss << message->content << "<|eot_id|>Assistant: ";
            } else {
                ss << message->content;
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DOTS1) {
        // dots.llm1.inst (DOTS1)
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|system|>" << message->content << "<|endofsystem|>";
            } else if (role == "user") {
                ss << "<|userprompt|>" << message->content << "<|endofuserprompt|>";
            } else {
                ss << "<|response|>" << message->content << "<|endofresponse|>";
            }
        }
        if (add_ass) {
            ss << "<|response|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_HUNYUAN_MOE) {
        // tencent/Hunyuan-A13B-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|startoftext|>" << message->content << "<|extra_4|>";
            } else if (role == "assistant") {
                ss << "<|startoftext|>" << message->content << "<|eos|>";
            } else {
                ss << "<|startoftext|>" << message->content << "<|extra_0|>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_KIMI_K2) {
        // moonshotai/Kimi-K2-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|im_system|>system<|im_middle|>" << message->content << "<|im_end|>";
            } else if (role == "assistant") {
                ss << "<|im_user|>user<|im_middle|>" << message->content << "<|im_end|>";
            } else {
                ss << "<|im_assistant|>assistant<|im_middle|>" << message->content << "<|im_end|>";
            }
        }
        if (add_ass) {
            ss << "<|im_assistant|>assistant<|im_middle|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_OPENAI_MOE) {
        // OpenAI MoE (based on Harmony chat template)
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|start|>" << role << "<|message|>" << message->content;
            ss << (role == "assistant" ? "<|return|>" : "<|end|>");
        }
        if (add_ass) {
            ss << "<|start|>assistant";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GROK_2) {
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "System: " << trim(message->content) << "<|separator|>\n\n";
            }
            else if (role == "user") {
                ss << "Human: " << trim(message->content) << "<|separator|>\n\n";
            }
            else if (role == "assistant") {
                ss << "Assistant: " << message->content << "<|separator|>\n\n";
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else {
        // template not supported
        return -1;
    }
    dest = ss.str();
    return dest.size();
}

int32_t llama_chat_apply_template(
                const struct llama_model * model,
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    std::string curr_tmpl(tmpl == nullptr ? "" : tmpl);
    if (tmpl == nullptr) {
        GGML_ASSERT(model != nullptr);

        // load template from model, if available
        const auto & it = model->gguf_kv.find("tokenizer.chat_template");
        if (it != model->gguf_kv.end() && it->second.size() > 0) {
            curr_tmpl = it->second;
        }
        else {
            // worst case: there is no information about template, we will use chatml by default
            curr_tmpl = "chatml";  // see llama_chat_apply_template_internal
        }
    }

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llama_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llama_chat_apply_template_internal(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

int32_t llama_chat_builtin_templates(const char ** output, size_t len) {
    auto it = LLM_CHAT_TEMPLATES.begin();
    for (size_t i = 0; i < std::min(len, LLM_CHAT_TEMPLATES.size()); i++) {
        output[i] = it->first.c_str();
        std::advance(it, 1);
    }
    return (int32_t) LLM_CHAT_TEMPLATES.size();
}
//
// grammar
//

struct llama_grammar * llama_grammar_init(
        const llama_grammar_element ** rules,
        size_t    n_rules,
        size_t    start_rule_index) {
    return llama_grammar_init_impl(rules, n_rules, start_rule_index);
}

void llama_grammar_free(struct llama_grammar * grammar) {
    llama_grammar_free_impl(grammar);
}
//
//void llama_grammar_init_lazy(struct llama_sampler* smpl) {
//
//    if (!grammar) {
//        return;
//    }
//    std::vector<const char*>  trigger_patterns_c;
//    trigger_patterns_c.reserve(grammar.grammar->trigger_patterns.size());
//    for (auto& trigger_pattern : grammar.grammar->trigger_patterns) {
//        trigger_patterns_c.push_back(trigger_pattern.pattern.c_str());
//    }
//    //auto* grammar_new = llama_grammar_init_impl(grammar->vocab, "", "root",
//    //    grammar->lazy, trigger_patterns_c.data(), trigger_patterns_c.size(),
//    //    grammar->trigger_tokens.data(), grammar->trigger_tokens.size());
//
//}


struct llama_grammar * llama_grammar_copy(const struct llama_grammar * grammar) {
    return llama_grammar_copy_impl(grammar);
}

void llama_grammar_sample(
      const struct llama_grammar * grammar,
      const struct llama_context * ctx,
          llama_token_data_array * candidates) {
    llama_grammar_sample_impl(grammar, &ctx->model.vocab, &ctx->sampling, candidates);
}

void llama_sample_grammar(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
      const struct llama_grammar * grammar) {
    llama_grammar_sample(grammar, ctx, candidates);
}

void llama_grammar_accept_token(
            struct llama_grammar * grammar,
            struct llama_context * ctx,
                     llama_token   token) {
    llama_grammar_accept_token_impl(grammar, &ctx->model.vocab, &ctx->sampling, token);
}

//
// sampling
//

void llama_set_rng_seed(struct llama_context * ctx, uint32_t seed) {
    llama_set_rng_seed_impl(&ctx->sampling, seed);
}

void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates) {
    llama_sample_softmax_impl(ctx ? &ctx->sampling : nullptr, candidates);
}

void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int32_t k, size_t min_keep) {
    llama_sample_top_k_impl(ctx ? &ctx->sampling : nullptr, candidates, k, min_keep);
}

void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    llama_sample_top_p_impl(ctx ? &ctx->sampling : nullptr, candidates, p, min_keep);
}

void llama_sample_min_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    llama_sample_min_p_impl(ctx ? &ctx->sampling : nullptr, candidates, p, min_keep);
}

void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep) {
    llama_sample_tail_free_impl(ctx ? &ctx->sampling : nullptr, candidates, z, min_keep);
}

void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep) {
    llama_sample_typical_impl(ctx ? &ctx->sampling : nullptr, candidates, p, min_keep);
}

void llama_sample_entropy(struct llama_context * ctx, llama_token_data_array * candidates_p, float min_temp, float max_temp, float exponent_val) {
    llama_sample_entropy_impl(ctx ? &ctx->sampling : nullptr, candidates_p, min_temp, max_temp, exponent_val);
}

void llama_sample_temp(struct llama_context * ctx, llama_token_data_array * candidates_p, float temp) {
    llama_sample_temp_impl(ctx ? &ctx->sampling : nullptr, candidates_p, temp);
}

void llama_sample_xtc(struct llama_context * ctx, llama_token_data_array * candidates_p,
                           float   probability, float threshold, size_t min_keep) {
    llama_sample_xtc_impl(ctx ? &ctx->sampling : nullptr, candidates_p, probability, threshold, min_keep);
}

void llama_sample_top_n_sigma(struct llama_context * ctx, llama_token_data_array * candidates_p, float top_n_sigma) {
    llama_sample_top_n_sigma_impl(ctx ? &ctx->sampling : nullptr, candidates_p, top_n_sigma);
}


void llama_sample_dry([[maybe_unused]] struct llama_context* ctx, struct llama_sampler_dry* smpl,  llama_token_data_array* candidates_p) {
    llama_sampler_dry_apply(smpl, candidates_p);
}

void llama_sample_adaptive_p(
    [[maybe_unused]] struct llama_context * ctx,
          struct llama_sampler_adaptive_p * adapt_p_ctx,
                   llama_token_data_array * candidates) {
    llama_sampler_adaptive_p_apply(adapt_p_ctx, candidates);
}

void llama_sample_repetition_penalties(
            struct llama_context * ctx,
          llama_token_data_array * candidates,
               const llama_token * last_tokens,
                          size_t   penalty_last_n,
                           float   penalty_repeat,
                           float   penalty_freq,
                           float   penalty_present) {
    llama_sample_repetition_penalties_impl(ctx ? &ctx->sampling : nullptr, candidates, last_tokens, penalty_last_n, penalty_repeat, penalty_freq, penalty_present);
}

void llama_sample_apply_guidance(
          struct llama_context * ctx,
                         float * logits,
                         float * logits_guidance,
                         float   scale) {
    llama_sample_apply_guidance_impl(&ctx->sampling, logits, logits_guidance, scale);
}

llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int32_t m, float * mu) {
    return llama_sample_token_mirostat_impl(&ctx->sampling, candidates, tau, eta, m, mu);
}

llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu) {
    return llama_sample_token_mirostat_v2_impl(ctx ? &ctx->sampling : nullptr, candidates, tau, eta, mu);
}

llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates) {
    return llama_sample_token_greedy_impl(ctx ? &ctx->sampling : nullptr, candidates);
}

llama_token llama_sample_token_with_rng(struct llama_context * ctx, llama_token_data_array * candidates, std::mt19937 & rng) {
    return llama_sample_token_with_rng_impl(&ctx->sampling, candidates, rng);
}

llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates) {
    return llama_sample_token_with_rng_impl(&ctx->sampling, candidates, ctx->sampling.rng);
}

llama_token llama_sample_token_adaptive_p(
               struct llama_context * ctx,
             llama_token_data_array * candidates,
    struct llama_sampler_adaptive_p * adapt_p_ctx,
                              float * orig_probs)
{
    return llama_sample_token_adaptive_p_impl(&ctx->sampling, candidates, adapt_p_ctx, orig_probs);
}

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}


struct llama_sampler_dry * llama_sampler_init_dry(const struct llama_vocab* vocab, float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const char** seq_breakers, size_t num_breakers) {
    return llama_sampler_init_dry_impl(*vocab, vocab->n_tokens(), dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers);
}

void llama_sampler_dry_reset(struct llama_sampler_dry* smpl) {
    if (!smpl) {
        return;
    }
    smpl->last_tokens.clear();
    smpl->dry_repeat_count.clear();
    smpl->dry_max_token_repeat.clear();
}

void llama_sampler_dry_free(struct llama_sampler_dry* smpl) {
    delete smpl;
}

struct llama_sampler_dry* llama_sampler_dry_clone(struct llama_sampler_dry* smpl) {
    // nullptr is passed as vocab because it is only needed for raw sequence breaker processing, which we have already done and will be copying
    auto* result = llama_sampler_init_dry(nullptr, smpl->dry_multiplier, smpl->dry_base, smpl->dry_allowed_length, smpl->dry_penalty_last_n, NULL, 0);
    // Copy the state, including the processed breakers
    {
        auto* result_ctx = smpl;
        result_ctx->dry_processed_breakers = smpl->dry_processed_breakers;
        result_ctx->dry_repeat_count = smpl->dry_repeat_count;
        result_ctx->dry_max_token_repeat = smpl->dry_max_token_repeat;
        result_ctx->last_tokens = smpl->last_tokens;
    }

    return result;
}

void llama_sampler_dry_accept(struct llama_sampler_dry* smpl, llama_token token) {
    if (!smpl) {
        return;
    }
    if (smpl->dry_multiplier == 0.0f || smpl->dry_base < 1.0f || smpl->dry_penalty_last_n == 0) {
        return;
    }
    smpl->last_tokens.push_back(token);
}


struct llama_sampler_adaptive_p * llama_sampler_init_adaptive_p(const float target, const float decay, const uint32_t seed)
{
    return llama_sampler_init_adaptive_p_impl(target, decay, seed);
}


int llama_split_prefix(char * dest, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if dest ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(dest, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

struct llama_timings llama_get_timings(struct llama_context * ctx) {
    struct llama_timings result = {
        /*.t_start_ms  =*/ 1e-3 * ctx->t_start_us,
        /*.t_end_ms    =*/ 1.00 * ggml_time_ms(),
        /*.t_load_ms   =*/ 1e-3 * ctx->t_load_us,
        /*.t_sample_ms =*/ 1e-3 * ctx->sampling.t_sample_us,
        /*.t_p_eval_ms =*/ 1e-3 * ctx->t_p_eval_us,
        /*.t_eval_ms   =*/ 1e-3 * ctx->t_eval_us,

        /*.n_sample =*/ std::max(1, ctx->sampling.n_sample),
        /*.n_p_eval =*/ std::max(0, ctx->n_p_eval),
        /*.n_eval   =*/ std::max(1, ctx->n_eval),
    };

    return result;
}

void llama_print_timings(struct llama_context * ctx) {
    const llama_timings timings = llama_get_timings(ctx);

    LLAMA_LOG_INFO("\n");
    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, timings.t_load_ms);
    LLAMA_LOG_INFO("%s:      sample time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_sample_ms, timings.n_sample, timings.t_sample_ms / timings.n_sample, 1e3 / timings.t_sample_ms * timings.n_sample);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_p_eval_ms, timings.n_p_eval, timings.t_p_eval_ms / timings.n_p_eval, 1e3 / timings.t_p_eval_ms * timings.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, timings.t_eval_ms, timings.n_eval, timings.t_eval_ms / timings.n_eval, 1e3 / timings.t_eval_ms * timings.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (timings.t_end_ms - timings.t_start_ms), (timings.n_p_eval + timings.n_eval));
}

void llama_reset_timings(struct llama_context * ctx) {
    ctx->t_start_us  = ggml_time_us();
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;

    ctx->sampling.reset_timings();
}

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX_VNNI = "    + std::to_string(ggml_cpu_has_avx_vnni())    + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "AVX512_BF16 = " + std::to_string(ggml_cpu_has_avx512_bf16()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "SVE = "         + std::to_string(ggml_cpu_has_sve())         + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "SSSE3 = "       + std::to_string(ggml_cpu_has_ssse3())       + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";
    s += "MATMUL_INT8 = " + std::to_string(ggml_cpu_has_matmul_int8()) + " | ";
    s += "LLAMAFILE = "   + std::to_string(ggml_cpu_has_llamafile())   + " | ";

    return s.c_str();
}

void llama_dump_timing_info_yaml(FILE * stream, const llama_context * ctx) {
    fprintf(stream, "\n");
    fprintf(stream, "###########\n");
    fprintf(stream, "# Timings #\n");
    fprintf(stream, "###########\n");
    fprintf(stream, "\n");

    fprintf(stream, "mst_eval: %.2f  # ms / token during generation\n",
            1.0e-3 * ctx->t_eval_us / ctx->n_eval);
    fprintf(stream, "mst_p_eval: %.2f  # ms / token during prompt processing\n",
            1.0e-3 * ctx->t_p_eval_us / ctx->n_p_eval);
    fprintf(stream, "mst_sample: %.2f  # ms / token during sampling\n",
            1.0e-3 * ctx->sampling.t_sample_us / ctx->sampling.n_sample);
    fprintf(stream, "n_eval: %d  # number of tokens generated (excluding the first one)\n", ctx->n_eval);
    fprintf(stream, "n_p_eval: %d  # number of tokens processed in batches at the beginning\n", ctx->n_p_eval);
    fprintf(stream, "n_sample: %d  # number of sampled tokens\n", ctx->sampling.n_sample);
    fprintf(stream, "t_eval_us: %" PRId64 "  # total microseconds spent generating tokens\n", ctx->t_eval_us);
    fprintf(stream, "t_load_us: %" PRId64 "  # total microseconds spent loading the model\n", ctx->t_load_us);
    fprintf(stream, "t_p_eval_us: %" PRId64 "  # total microseconds spent prompt processing\n", ctx->t_p_eval_us);
    fprintf(stream, "t_sample_us: %" PRId64 "  # total microseconds spent sampling\n", ctx->sampling.t_sample_us);
    fprintf(stream, "ts_eval: %.2f  # tokens / second during generation\n",
            1.0e6 * ctx->n_eval / ctx->t_eval_us);
    fprintf(stream, "ts_p_eval: %.2f  # tokens / second during prompt processing\n",
            1.0e6 * ctx->n_p_eval / ctx->t_p_eval_us);
    fprintf(stream, "ts_sample: %.2f  # tokens / second during sampling\n",
            1.0e6 * ctx->sampling.n_sample / ctx->sampling.t_sample_us);
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
) {
    return ctx->model.tensors_by_name;
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    g_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_state.log_callback_user_data = user_data;
#ifdef GGML_USE_METAL
    ggml_backend_metal_log_set_callback(g_state.log_callback, g_state.log_callback_user_data);
#elif defined(GGML_USE_CUDA)
    ggml_backend_cuda_log_set_callback(g_state.log_callback, g_state.log_callback_user_data);
#elif defined(GGML_USE_CANN)
    ggml_backend_cann_log_set_callback(g_state.log_callback, g_state.log_callback_user_data);
#endif
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_state.log_callback(level, buffer, g_state.log_callback_user_data);
    } else {
        char* buffer2 = new char[len+1];
        vsnprintf(buffer2, len+1, format, args_copy);
        buffer2[len] = 0;
        g_state.log_callback(level, buffer2, g_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void llama_set_offload_policy(struct llama_context * lctx, int op, bool on_or_off) {
    if (!lctx || !lctx->sched) return;
    const char * op_name = op < 0 || op >= int(GGML_OP_COUNT) ? "all ops" : ggml_op_name(ggml_op(op));
    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXX offload(%s) = %d\n", op_name, on_or_off);
    ggml_backend_sched_set_op_offload(lctx->sched, ggml_op(op), on_or_off);
}
