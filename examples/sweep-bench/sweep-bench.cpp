#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "speculative.h"
#include "llama-vocab.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/resource.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static double get_rss_hwm_mib() {
#ifdef _WIN32
    return -1.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1.0;
    }
#ifdef __APPLE__
    return usage.ru_maxrss / (1024.0 * 1024.0);
#else
    return usage.ru_maxrss / 1024.0;
#endif
#endif
}

struct sweep_vram_tracker {
    std::vector<size_t> baseline;

    void start() {
#ifdef GGML_USE_CUDA
        const int count = ggml_backend_cuda_get_device_count();
        baseline.resize(count);
        for (int device = 0; device < count; ++device) {
            size_t free;
            size_t total;
            ggml_backend_cuda_get_device_memory(device, &free, &total);
            baseline[device] = free;
        }
#endif
    }

    double sample() {
#ifdef GGML_USE_CUDA
        if (baseline.empty()) {
            return -1.0;
        }
        size_t used = 0;
        for (int device = 0; device < (int) baseline.size(); ++device) {
            size_t free;
            size_t total;
            ggml_backend_cuda_get_device_memory(device, &free, &total);
            used += baseline[device] > free ? baseline[device] - free : 0;
        }
        return used / (1024.0 * 1024.0);
#else
        return -1.0;
#endif
    }
};

static std::string format_mib(double value, int precision, const char * missing) {
    if (value < 0.0) {
        return missing;
    }
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return buffer;
}

static void llama_selective_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    const char * skip_patterns[] = {
        "Setting default device in layer",
        "llama_model_loader: Dumping metadata",
        "llama_model_loader: - kv  ",
        "llama_model_loader: - type ",
        "validate_override:",
        "load: printing all EOG",
        "load:   - ",
        "load: special tokens cache",
        "load: token to piece cache",
        "llm_load_print_meta:",
        "print_info:",
        "------------------- Layer sizes",
        "Layer ",
        "llm_load_tensors:",
        "==========================",
        "merging up/gate in layer",
        "repacking up/gate experts weight in layer",
    };
    for (const char * pat : skip_patterns) {
        if (strstr(text, pat) != nullptr) {
            return;
        }
    }
    // Skip incomplete/continuation lines
    int i = 0;
    while (text[i] == ' ' || text[i] == '\t') {
        i++;
    }
    if (text[i] == ',' || text[i] == '(' || text[i] == ')'|| (text[i] >= '0' && text[i] <= '9')) {
        return;
    }
    LOG_TEE("%s", text);
}

static void print_usage(int argc, char ** argv) {
    gpt_params params;
    params.sweep_bench = true;
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nsweep-bench specific options:\n\n");
    LOG_TEE("  -nrep, --n-repetitions N        number of repetitions for each context size (default: 1)\n");
    LOG_TEE("         --sweep-stride N         measure every Nth sweep row (default: 1)\n");
    LOG_TEE("         --sweep-memory           report RSS high-water and sampled VRAM delta\n");
    LOG_TEE("  -wb,   --warmup-batch           run a warmup batch before measurement\n");
    LOG_TEE("         --output-format FORMAT    output format: table (default) or jsonl\n");
    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -c 8192 -b 2048 -ub 512\n", argv[0]);
    LOG_TEE("\n");
}

int main(int argc, char ** argv) {

    gpt_params params;
    params.sweep_bench = true;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv);
        return 1;
    }
    if (params.nrep < 1) params.nrep = 1;
    if (params.sweep_stride < 1) params.sweep_stride = 1;

    if (params.minilog) {
        llama_log_set(llama_selective_log_callback, nullptr);
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    sweep_vram_tracker vram_tracker;
    if (params.sweep_memory) {
        vram_tracker.start();
    }

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    const bool use_checkpoint = common_speculative_needs_checkpoint(model);

    const unsigned int n_kv_max = llama_n_ctx(ctx);


    const llama_vocab * vocab = llama_get_vocab(ctx);
    llama_token bos = vocab->token_bos();
    //llama_token eos = llama_token_eos_impl(*vocab);

    const unsigned int n_vocab  = llama_n_vocab(model);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch) {
        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            llama_synchronize(ctx);
        }

        return true;
    };

    const unsigned int pp = params.n_ubatch;
    const unsigned int tg = params.n_predict > 0 ? params.n_predict : params.n_ubatch / 4;

    if (!params.sweep_bench_output_jsonl) {
        LOG_TEE("\n");
        LOG_TEE("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG_TEE("\n");
        if (params.sweep_memory) {
            LOG_TEE("|%6s | %6s | %6s | %8s | %8s | %8s | %8s | %10s | %10s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "RSS HWM", "VRAM delta");
            LOG_TEE("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%10s-|-%10s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------", "----------", "----------");
        } else {
            LOG_TEE("|%6s | %6s | %6s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s");
            LOG_TEE("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------");
        }
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    auto pp_helper = [&](unsigned int n_kv) {
        common_batch_clear(batch);

        for (unsigned int i = 0; i < pp; ++i) {
            common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        return decode_helper(ctx, batch, ctx_params.n_batch);
    };

    // warm up
    if (params.warmup) {
        common_batch_add(batch, bos, 0, { 0 }, false);

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }
    if (params.batch_warmup) {
        // clean up KV cache after generation
        llama_kv_cache_seq_rm(ctx, 0, params.n_ubatch, -1);

        // prepare batch of pp size for prompt processing performance measurement
        common_batch_clear(batch);

        for (unsigned int i = 0; i < params.n_ubatch; ++i) {
            common_batch_add(batch, std::rand() % n_vocab, i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_ubatch)) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    common_batch_clear(batch);
    llama_kv_cache_clear(ctx);

    llama_reset_timings(ctx);

    int i_loop = 0;
    std::vector<uint8_t> checkpoint_data;

    for (unsigned int n_kv = 0; n_kv < n_kv_max; n_kv += params.n_ubatch) {
        // clean up KV cache before generation
        //llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);

        const bool measure = i_loop % params.sweep_stride == 0;
        int nrep = measure && i_loop < 1 ? params.nrep : 1;

        size_t checkpoint_size = 0;
        if (use_checkpoint && measure && n_kv > 0) {
            const size_t need = llama_state_seq_get_size(ctx, 0, 0);
            checkpoint_data.resize(need);
            checkpoint_size = llama_state_seq_get_data(ctx, checkpoint_data.data(), need, 0, 0);
            if (checkpoint_size == 0) {
                LOG_TEE("%s: failed to checkpoint sequence at %u\n", __func__, n_kv);
                return 1;
            }
            checkpoint_data.resize(checkpoint_size);
        }

        // first measure token generation performance at this context size
        int64_t t_tg_start = 0;
        int64_t t_tg_end   = 0;

        if (measure) {
            t_tg_start = ggml_time_us();
            //fprintf(stderr, "======================================== tg_start for n_kv = %u\n", n_kv);
            //printf("======================================== tg_start for n_kv = %u\n", n_kv);

            for (int irep = 0; irep < nrep; ++irep) {
                if (use_checkpoint) {
                    if (n_kv == 0) {
                        llama_kv_cache_clear(ctx);
                    }
                } else {
                    llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);
                }

                for (unsigned int i = 0; i < tg; ++i) {
                    common_batch_clear(batch);
                    common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, true);

                    if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                        LOG_TEE("%s: llama_decode() failed\n", __func__);
                        return 1;
                    }
                }
            }

            //fprintf(stderr, "======================================== tg_end for n_kv = %u\n", n_kv);
            //printf("======================================== tg_end for n_kv = %u\n", n_kv);
            t_tg_end = ggml_time_us();
        } else {
            // keep the token stream aligned with a stride-1 sweep
            for (unsigned int i = 0; i < tg; ++i) {
                (void) std::rand();
            }
        }

        if (use_checkpoint && measure) {
            if (n_kv > 0) {
                const size_t n = llama_state_seq_set_data(ctx, checkpoint_data.data(), checkpoint_data.size(), 0, 0);
                if (n != checkpoint_size) {
                    LOG_TEE("%s: failed to restore sequence (expected %zu bytes, got %zu)\n", __func__, checkpoint_size, n);
                    return 1;
                }
            } else {
                llama_kv_cache_clear(ctx);
            }
        }

        // measure prompt processing performance
        int64_t t_pp_start = 0;
        int64_t t_pp_end   = 0;

        if (measure) {
            t_pp_start = ggml_time_us();

            for (int irep = 0; irep < nrep; ++irep) {
                if (use_checkpoint) {
                    if (n_kv == 0) {
                        llama_kv_cache_clear(ctx);
                    }
                } else {
                    if (!llama_kv_cache_seq_rm(ctx, 0, n_kv, -1)) {
                        LOG_TEE("%s: failed to rewind sequence to %u\n", __func__, n_kv);
                        return 1;
                    }
                }

                if (!pp_helper(n_kv)) {
                    LOG_TEE("%s: llama_decode() failed\n", __func__);
                    return 1;
                }
            }

            t_pp_end = ggml_time_us();
        } else {
            if (!pp_helper(n_kv)) {
                LOG_TEE("%s: llama_decode() failed\n", __func__);
                return 1;
            }
        }

        if (!measure) {
            ++i_loop;
            continue;
        }

        // calculate and print metrics
        const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f / nrep;
        const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f / nrep;

        const float speed_pp = pp / t_pp;
        const float speed_tg = tg / t_tg;

        double rss_hwm_mib    = -1.0;
        double vram_delta_mib = -1.0;
        if (params.sweep_memory) {
            rss_hwm_mib    = get_rss_hwm_mib();
            vram_delta_mib = vram_tracker.sample();
        }

        if(params.sweep_bench_output_jsonl) {
            if (params.sweep_memory) {
                const std::string rss_json  = format_mib(rss_hwm_mib, 3, "null");
                const std::string vram_json = format_mib(vram_delta_mib, 3, "null");
                LOG_TEE(
                    "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                    "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"rss_hwm_mib\": %s, \"vram_delta_mib\": %s }\n",
                    n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                    pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg, rss_json.c_str(), vram_json.c_str()
                );
            } else {
                LOG_TEE(
                    "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                    "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f }\n",
                    n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                    pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg
                );
            }
        } else {
            if (params.sweep_memory) {
                const std::string rss  = format_mib(rss_hwm_mib, 1, "n/a");
                const std::string vram = format_mib(vram_delta_mib, 1, "n/a");
                LOG_TEE("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %10s | %10s |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg, rss.c_str(), vram.c_str());
            } else {
                LOG_TEE("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
            }
        }

        ++i_loop;
    }

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
