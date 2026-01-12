#include "llama-model-loader.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "ggml.h"
//#include "ggml-backend.h"

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

#include <set>
#include <map>
#include <array>
#include <future>
#include <regex>

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

#define LLAMA_API_INTERNAL

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            return ArrayInfo {
                gguf_get_arr_type(ctx, k),
                size_t(gguf_get_arr_n(ctx, k)),
                gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

llama_model_loader::llama_model_loader(const std::string & fname, bool use_mmap, bool check_tensors,
        bool repack_tensors, bool use_thp, bool merge_qkv, bool merge_up_gate_exps,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

#ifdef _WIN32
    // Only bump maxstdio if the user really wants large contexts:
#if defined(GGML_MAX_CONTEXTS) && (GGML_MAX_CONTEXTS > 512)
    // Cap at MSVC's hard limit of 8192 - https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmaxstdio?view=msvc-160
#if (GGML_MAX_CONTEXTS > 8192)
#define _GGML_STDIO_TARGET 8192
#else
#define _GGML_STDIO_TARGET GGML_MAX_CONTEXTS
#endif
    int _setmaxstdio_ret = _setmaxstdio(_GGML_STDIO_TARGET);
    if (_setmaxstdio_ret == -1) {
        LLAMA_LOG_INFO("%s: failed to set max stdio to %d. (setmaxstdio returned -1)\n", __func__, _GGML_STDIO_TARGET);
    } else {
        LLAMA_LOG_INFO("%s: max stdio successfully set to %d\n", __func__, _setmaxstdio_ret);
    }
#endif // GGML_MAX_CONTEXTS > 512
#endif // _WIN32

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    struct ggml_context * ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    meta = gguf_init_from_file(fname.c_str(), params);
    if (!meta) {
        throw std::runtime_error(format("%s: failed to load model from %s\n", __func__, fname.c_str()));
    }

    get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
    llm_kv = LLM_KV(llm_arch_from_string(arch_name));

    files.emplace_back(new llama_file(fname.c_str(), "rb"));
    contexts.emplace_back(ctx);

    // Save tensors data offset of the main file.
    // For subsidiary files, `meta` tensor data offset must not be used,
    // so we build a unified tensors index for weights.
    for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
        weights.emplace_back(files.back().get(), 0, cur->name, meta, cur);
    }
    uint16_t n_split = 0;
    get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

    // Load additional GGML contexts
    if (n_split > 1) {
        uint16_t idx = 0;
        get_key(llm_kv(LLM_KV_SPLIT_NO), idx);
        if (idx != 0) {
            throw std::runtime_error(format("illegal split file: %d, model must be loaded with the first split", idx));
        }

        char split_prefix[PATH_MAX] = {0};
        if (!llama_split_prefix(split_prefix, sizeof(split_prefix), fname.c_str(), idx, n_split)) {
            throw std::runtime_error(format("invalid split file: %s", fname.c_str()));
        }

        if (trace > 0) {
            LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
        }

        char split_path[PATH_MAX] = {0};
        for (idx = 1; idx < n_split; idx++) {
            llama_split_path(split_path, sizeof(split_path), split_prefix, idx, n_split);

            struct gguf_init_params split_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &ctx,
            };
            struct gguf_context * ctx_gguf = gguf_init_from_file(split_path, split_params);
            if (!ctx_gguf) {
                throw std::runtime_error(format("%s: failed to load GGUF split from %s\n", __func__, split_path));
            }

            files.emplace_back(new llama_file(split_path, "rb"));
            contexts.emplace_back(ctx);

            // Save tensors data offset info of the shard.
            for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                weights.emplace_back(files.back().get(), idx, cur->name, ctx_gguf, cur);
            }

            gguf_free(ctx_gguf);
        }

        get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

        // sanity check
        {
            const int n_tensors_loaded = (int) weights.size();
            if (n_tensors != n_tensors_loaded) {
                throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
            }
        }

        LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
    }

    n_kv      = gguf_get_n_kv(meta);
    n_tensors = weights.size();

    fver = (enum llama_fver) gguf_get_version(meta);

    std::set<std::string> tensor_names;
    for (auto & w : weights) {
        n_elements += ggml_nelements(w.tensor);
        n_bytes    += ggml_nbytes(w.tensor);
        // make sure there is no duplicated tensor names
        const std::string name(w.tensor->name);
        auto found = tensor_names.find(name);
        if (found != tensor_names.end()) {
            throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", w.tensor->name));
        }
        tensor_names.insert(name);
    }

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (int i = 0; i < n_tensors; i++) {
            const ggml_tensor * tensor = weights.at(i).tensor;
            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = weights.at(i).idx;
                LLAMA_LOG_INFO("%s: - tensor %4d, split %2d: %32s %-8s [ %s ]\n", __func__, i, sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str());
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_BF16_R16:ftype = LLAMA_FTYPE_MOSTLY_BF16_R16;break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q6_0:    ftype = LLAMA_FTYPE_MOSTLY_Q6_0;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q8_KV:   ftype = LLAMA_FTYPE_MOSTLY_Q8_KV;   break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q3_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q3_K_R4; break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q4_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q4_K_R4; break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q5_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_K_R4; break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_Q6_K_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_K_R4; break;
            case GGML_TYPE_Q8_K_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_K_R8; break;
            case GGML_TYPE_Q8_KV_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_KV_R8; break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XXS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS_R4; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_XS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS_R4; break;
            case GGML_TYPE_IQ2_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_M;   break;
            case GGML_TYPE_IQ2_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_M_R4;break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ3_XXS_R4: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS_R4; break;
            case GGML_TYPE_IQ1_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_KT;  break;
            case GGML_TYPE_IQ2_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KT;  break;
            case GGML_TYPE_IQ3_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KT;  break;
            case GGML_TYPE_IQ4_KT:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KT;  break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_S_R4;break;
            case GGML_TYPE_IQ1_M_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ1_M_R4;break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ1_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ1_BN;  break;
            case GGML_TYPE_IQ2_BN:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN;  break;
            case GGML_TYPE_IQ2_BN_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_BN_R4;break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_NL_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL_R4;break;
            case GGML_TYPE_IQ4_XS_R8:ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS_R8;break;
            case GGML_TYPE_Q4_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_R8; break;
            case GGML_TYPE_Q5_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q5_0_R4; break;
            case GGML_TYPE_Q6_0_R4: ftype = LLAMA_FTYPE_MOSTLY_Q6_0_R4; break;
            case GGML_TYPE_Q8_0_R8: ftype = LLAMA_FTYPE_MOSTLY_Q8_0_R8; break;
            case GGML_TYPE_MXFP4:   ftype = LLAMA_FTYPE_MOSTLY_MXFP4;   break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ4_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS;  break;
            case GGML_TYPE_IQ4_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_KS_R4;  break;
            case GGML_TYPE_IQ5_KS_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS_R4;  break;
            case GGML_TYPE_IQ4_KSS: ftype = LLAMA_FTYPE_MOSTLY_IQ4_KSS; break;
            case GGML_TYPE_IQ5_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ5_KS;  break;
            case GGML_TYPE_IQ2_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_K;   break;
            case GGML_TYPE_IQ2_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ2_K_R4;break;
            case GGML_TYPE_IQ3_KS:  ftype = LLAMA_FTYPE_MOSTLY_IQ3_KS;  break;
            case GGML_TYPE_IQ2_KL:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_KL;  break;
            case GGML_TYPE_IQ3_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_K;   break;
            case GGML_TYPE_IQ3_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_K_R4;break;
            case GGML_TYPE_IQ4_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ4_K;   break;
            case GGML_TYPE_IQ4_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ4_K_R4;break;
            case GGML_TYPE_IQ5_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ5_K;   break;
            case GGML_TYPE_IQ5_K_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ5_K_R4;break;
            case GGML_TYPE_IQ6_K:   ftype = LLAMA_FTYPE_MOSTLY_IQ6_K;   break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_IQ3_S_R4:ftype = LLAMA_FTYPE_MOSTLY_IQ3_S_R4;break;
            case GGML_TYPE_Q4_0_4_4: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_4; break;
            case GGML_TYPE_Q4_0_4_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_4_8; break;
            case GGML_TYPE_Q4_0_8_8: ftype = LLAMA_FTYPE_MOSTLY_Q4_0_8_8; break;
            default:
                {
                     LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                     ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            const int kid = gguf_find_key(meta, "general.file_type"); // TODO: use LLM_KV
            if (kid >= 0) {
                ftype = (llama_ftype) gguf_get_val_u32(meta, kid);
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(meta, i);
            const enum gguf_type type   = gguf_get_kv_type(meta, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%d]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(meta, i)), gguf_get_arr_n(meta, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(meta, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }
    if (repack_tensors) {
        use_mmap = false;
    }

    this->use_mmap = use_mmap;
    this->check_tensors = check_tensors;
    this->repack_tensors = repack_tensors;
    this->use_thp = use_thp;
    this->merge_qkv = merge_qkv;
    this->merge_up_gate_exps = merge_up_gate_exps;
}

llama_model_loader::~llama_model_loader() {
    if (meta) {
        gguf_free(meta);
    }
    for (auto * ctx : contexts) {
        ggml_free(ctx);
    }
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const std::string & key, T & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);


    result = arr_info.length;
    return true;
}

template<typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
llama_model_loader::get_arr_n(const enum llm_kv kid, T & result, const bool required) {
    return get_arr_n(llm_kv(kid), result, required);
}

template<typename T>
bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T, float>::value)); break;
        case GGUF_TYPE_INT32:   GGML_ASSERT(
                                        (std::is_same<T,  int32_t>::value) ||
                                        (std::is_same<T, uint32_t>::value));  break;
        default:
                                throw std::runtime_error(format("%s is not a float32, int32 array", key.c_str()));
    }

    result.resize(arr_info.length);
    result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);

    return true;
}

template<typename T, size_t N_MAX>
bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_ARRAY) {
        if (required) {
            throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
        }
        return false;
    }

    struct GGUFMeta::ArrayInfo arr_info =
        GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

    switch (arr_info.gt) {
        case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T, float>::value)); break;
        case GGUF_TYPE_INT32:   GGML_ASSERT(
                                        (std::is_same<T,  int32_t>::value) ||
                                        (std::is_same<T, uint32_t>::value));  break;
        default:
                                throw std::runtime_error(format("%s is not a float32, int32 array", key.c_str()));
    }

    if (arr_info.length > N_MAX) {
        throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
    }

    std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());

    return true;
}

template<typename T>
bool llama_model_loader::get_arr(const enum llm_kv kid, T & result, const bool required) {
    return get_arr(llm_kv(kid), result, required);
}

template<typename T>
bool llama_model_loader::get_key(const std::string & key, T & result, const bool required) {
    auto it = kv_overrides.find(key);

    const struct llama_model_kv_override * override =
        it != kv_overrides.end() ? &it->second : nullptr;

    const bool found = GGUFMeta::GKV<T>::set(meta, key, result, override);

    if (required && !found) {
        throw std::runtime_error(format("key not found in model: %s", key.c_str()));
    }

    return found;
}

template<typename T>
bool llama_model_loader::get_key(const enum llm_kv kid, T & result, const bool required) {
    return get_key(llm_kv(kid), result, required);
}

// get array of n <= N_MAX elements, or a single element repeated n times
template<typename T, size_t N_MAX>
bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, const bool required) {
    const int kid = gguf_find_key(meta, key.c_str());

    if (kid < 0) {
        if (required) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }
        return false;
    }

    if (n > N_MAX) {
        throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", (uint32_t) n, (uint32_t) N_MAX, key.c_str()));
    }

    if (gguf_get_kv_type(meta, kid) == GGUF_TYPE_ARRAY) {
        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(meta, kid);

        if (n != arr_info.length) {
            throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
        }

        return get_arr(key, result, required);
    } else {
        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }
}

template<typename T>
bool llama_model_loader::get_key_or_arr(const enum llm_kv kid, T & result, uint32_t n, const bool required) {
    return get_key_or_arr(llm_kv(kid), result, n, required);
}

const char * llama_model_loader::get_tensor_name(int i) const {
    return weights.at(i).tensor->name;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    for (const auto & weight : weights) {
        if (strcmp(name, weight.tensor->name) == 0) {
            return &weight;
        }
    }
    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct ggml_tensor * llama_model_loader::require_tensor_meta(const char * name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name);
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return tensor;
}

struct ggml_tensor * llama_model_loader::create_tensor_for(struct ggml_context * ctx, const struct ggml_tensor * cur, bool duplicated) {
    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                is_ok = false;
                break;
            }
        }
        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

struct ggml_tensor * llama_model_loader::create_tensor(struct ggml_context * ctx, const std::string & name,
        const std::vector<int64_t> & ne, int flags) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED));

    if (cur == NULL) {
        return NULL;
    }

    // skip unused tensors
    if (flags & TENSOR_SKIP) {
        const size_t nbytes = ggml_nbytes(cur);
        LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", name.c_str(), nbytes);

        size_data -= nbytes;
        n_created++;

        return nullptr;
    }

    return create_tensor_for(ctx, cur, flags & TENSOR_DUPLICATED);
}

struct ggml_tensor * llama_model_loader::create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base,
        const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
            dims[0], dims[1], dims[2], dims[3],
            cur->nb[1], cur->nb[2], cur->nb[3],
            offset);

    ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

void llama_model_loader::done_getting_tensors() const {
    if (n_created != n_tensors) {
        throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps, bool use_thp) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            std::unique_ptr<llama_mmap> mapping(new llama_mmap(file.get(), prefetch ? -1 : 0, ggml_is_numa(), use_thp));
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (auto & w : weights) {
        size_data += ggml_nbytes(w.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        try {
            const auto * weight = get_weight(ggml_get_name(tensor));
            if (!weight) {
                continue;
            }
            if (weight->idx != idx) {
                continue;
            }
            *first = std::min(*first, weight->offs);
            *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
        } catch(...) {
            // the tensor is not in the model
        }
    }
}

// for backwards compatibility, does not support ggml-backend
void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }
}

// Returns false if cancelled by progress_callback
bool llama_model_loader::load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data) {
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>> read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

#if defined(GGML_USE_CUDA)
    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;
    constexpr size_t buffer_size = 1 * 1024 * 1024; // 1MB

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<void*> host_ptrs;
    std::vector<ggml_backend_event_t> events;
    size_t buffer_idx = 0; // buffer to use for async loads

    ggml_backend_t cuda_backend = nullptr;
    if (!use_mmap && !check_tensors) {
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the CUDA backend is active, and if so, determine the device ID.
        ggml_backend_buffer_t buf = bufs_mmap.count(0) ? bufs_mmap.at(0) : nullptr;
        if (buf) {
            ggml_backend_buffer_type_t buffer_type = ggml_backend_buffer_get_type(buf);
            for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
                auto * cuda_buffer_type = ggml_backend_cuda_buffer_type(i);
                if (buffer_type == cuda_buffer_type) {
                    cuda_backend = ggml_backend_cuda_init(i, nullptr);
                    break;
                }
            }
        }

        // If the cuda backend is active create pinned memory buffers and events for synchronisation.
        if (cuda_backend) {
            for (size_t idx = 0; idx < n_buffers; ++idx) {
                host_buffers.emplace_back(ggml_backend_buft_alloc_buffer(llama_default_buffer_type_cpu(true), buffer_size));
                host_ptrs.emplace_back(ggml_backend_buffer_get_base(host_buffers[idx]));
                events.emplace_back(ggml_backend_event_new(cuda_backend));
            }
        }
    }
#endif

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        if (use_mmap) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs_mmap.count(weight->idx)) {
                buf_mmap = bufs_mmap.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                            return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                            }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            GGML_ASSERT(weight->idx < files.size());
            const auto & file = files.at(weight->idx);
            if (ggml_backend_buffer_is_host(cur->buffer)) {
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                                return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                                }));
                }
            } else {
#if defined(GGML_USE_CUDA)
                // If cuda_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (cuda_backend) {
                    file->seek(weight->offs, SEEK_SET);

                    size_t bytes_read = 0;

                    while (bytes_read < n_size) {
                        size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                        ggml_backend_event_synchronize(events[buffer_idx]);
                        file->read_raw(host_ptrs[buffer_idx], read_iteration);
                        ggml_backend_tensor_set_async(cuda_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                        ggml_backend_event_record(events[buffer_idx]);

                        bytes_read += read_iteration;
                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                }
                else
#endif
                {
                    read_buf.resize(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

#if defined(GGML_USE_CUDA)
    // free temporary resources used for async cuda uploads
    if (cuda_backend) {
        for (size_t idx = 0; idx < n_buffers;++idx) {
            ggml_backend_event_synchronize(events[idx]);
            ggml_backend_event_free(events[idx]);
            ggml_backend_buffer_free(host_buffers[idx]);
        }
        ggml_backend_free(cuda_backend);
    }
#endif

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

template<>
bool llama_model_loader::get_key(const enum llm_kv kid, enum llama_pooling_type & result, const bool required) {
    uint32_t tmp;
    const bool found = get_key(kid, tmp, required);
    if (found) {
        result = (enum llama_pooling_type) tmp;
    } else {
        result = LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    return found;
}
template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

template bool llama_model_loader::get_key_or_arr<std::array<int, 4>>(enum llm_kv kid, std::array<int, 4> & result, uint32_t n, bool required);
template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);

template std::enable_if<std::is_integral<unsigned int>::value, bool>::type llama_model_loader::get_arr_n<unsigned int>(enum llm_kv, unsigned int&, bool);

