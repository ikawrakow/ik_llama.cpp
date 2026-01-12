#pragma once

#include "llama.h"
#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-arch.h"

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <map>

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

static const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

struct llama_layer;

struct llama_model_loader {
    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    int64_t n_elements = 0;
    size_t  n_bytes    = 0;

    bool use_mmap = false;
    bool check_tensors;
    bool repack_tensors = false;
    bool use_thp = false;
    bool merge_qkv = false;
    bool merge_up_gate_exps = false;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const char * name, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx, name);
            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);

            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", name));
            }
        }
    };
    std::vector<llama_tensor_weight> weights;

    std::unordered_map<std::string, struct llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context * meta = NULL;
    std::vector<ggml_context *> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    llama_model_loader(const std::string & fname, bool use_mmap, bool check_tensors, bool repack_tensors, bool use_thp,
            bool merge_qkv, bool merge_up_gate_exps,
            const llama_model_kv_override * param_overrides_p,
            const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    ~llama_model_loader();

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, const bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const enum llm_kv kid, T & result, const bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, const bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, const bool required = true);

    template<typename T>
    bool get_arr(const enum llm_kv kid, T & result, const bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, const bool required = true);

    template<typename T>
    bool get_key(const enum llm_kv kid, T & result, const bool required = true);

    // get array of n <= N_MAX elements, or a single element repeated n times
    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, const bool required = true);

    template<typename T>
    bool get_key_or_arr(const enum llm_kv kid, T & result, uint32_t n, const bool required = true);

    const std::string& get_arch_name() const { return arch_name; }

    enum llm_arch get_arch() const { return llm_kv.arch; }

    const char * get_tensor_name(int i) const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight * get_weight(int i) const {
        return get_weight(get_tensor_name(i));
    }

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const char * name) const;

    struct ggml_tensor * get_tensor_meta(int i) const {
        return get_tensor_meta(get_tensor_name(i));
    }

    struct ggml_tensor * create_tensor_for(struct ggml_context * ctx, const struct ggml_tensor * cur, bool duplicated);

    const struct ggml_tensor * check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const;

    static const int TENSOR_NOT_REQUIRED = 1 << 0;
    static const int TENSOR_DUPLICATED   = 1 << 1;
    static const int TENSOR_SKIP         = 1 << 2;

    struct ggml_tensor * create_tensor(struct ggml_context * ctx, const std::string & name, const std::vector<int64_t> & ne, int flags = 0);

    struct ggml_tensor * create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base,
            const std::string & name, const std::vector<int64_t> & ne, size_t offset, bool required = true);

    void done_getting_tensors() const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr, bool use_thp = false);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // for backwards compatibility, does not support ggml-backend
    void load_data_for(struct ggml_tensor * cur) const;

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs_mmap,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);
};

void llm_load_arch(llama_model_loader & ml, llama_model & model);

void llm_load_hparams(llama_model_loader & ml, llama_model & model);

struct create_tensors_helper_interface {
    virtual ~create_tensors_helper_interface() = default;
    virtual bool create_tensors() = 0;
    virtual std::map<ggml_backend_buffer_type_t, ggml_context *> & get_ctx_map() = 0;
    virtual size_t get_ctx_size() const = 0;

    static std::unique_ptr<create_tensors_helper_interface> instance(llama_model_loader & ml, llama_model & model);
};
