#pragma once

#include "llama.h"

#include <cstdint>
struct llama_batch;
struct llama_context;
struct ggml_tensor;

bool llama_prepare_dsv4_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan);
void llama_reset_dsv4_state(llama_context * ctx, int32_t seq_id = -1);
bool llama_dsv4_spec_ckpt_prepare(llama_context * ctx, int mode, int max_tokens);
bool llama_dsv4_spec_ckpt_save(llama_context * ctx, bool use_gpu);
bool llama_dsv4_spec_ckpt_capture_rows(llama_context * ctx);
enum llama_spec_ckpt_restore_result llama_dsv4_spec_ckpt_restore(llama_context * ctx, bool use_gpu, int accepted_step);
void llama_dsv4_spec_ckpt_discard(llama_context * ctx);
ggml_tensor * llama_dsv4_spec_ckpt_delta(llama_context * ctx, ggml_tensor * state_tensor);
void llama_dsv4_spec_ckpt_record_plan(llama_context * ctx);
