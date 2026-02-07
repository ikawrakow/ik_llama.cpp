#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "llama-vocab.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -c 8192 -b 2048 -ub 512\n", argv[0]);
    LOG_TEE("\n");
}

int main(int argc, char ** argv) {

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv);
        return 1;
    }
    if (params.nrep < 1) params.nrep = 1;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

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
        LOG_TEE("|%6s | %6s | %6s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s");
        LOG_TEE("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------");
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

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

    int i_loop = 0;

    for (unsigned int n_kv = 0; n_kv < n_kv_max; n_kv += params.n_ubatch) {
        // clean up KV cache before generation
        //llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);

        int nrep = i_loop < 1 ? params.nrep : 1;

        // first measure token generation performance at this context size
        const auto t_tg_start = ggml_time_us();

        for (int irep = 0; irep < nrep; ++irep) {

            llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);

            for (unsigned int i = 0; i < tg; ++i) {
                common_batch_clear(batch);
                common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, true);

                if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                    LOG_TEE("%s: llama_decode() failed\n", __func__);
                    return 1;
                }
            }

        }

        const auto t_tg_end = ggml_time_us();

        // measure prompt processing performance
        const auto t_pp_start = ggml_time_us();

        for (int irep = 0; irep < nrep; ++irep) {

            // clean up KV cache after generation
            llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);

            // prepare batch of pp size for prompt processing performance measurement
            common_batch_clear(batch);

            for (unsigned int i = 0; i < pp; ++i) {
                common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true;

            if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                LOG_TEE("%s: llama_decode() failed\n", __func__);
                return 1;
            }

        }

        const auto t_pp_end = ggml_time_us();

        // calculate and print metrics
        const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f / nrep;
        const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f / nrep;

        const float speed_pp = pp / t_pp;
        const float speed_tg = tg / t_tg;

        if(params.sweep_bench_output_jsonl) {
            LOG_TEE(
                "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f }\n",
                n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg
            );
        } else {
            LOG_TEE("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
        }

        ++i_loop;
    }

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
