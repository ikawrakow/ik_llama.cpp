#include "llama-model.h"
#include "llama-spec-features.h"

#include <algorithm>
#include <cstring>

static void test_tied_companion(const char * path) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.mtp = true;

    llama_model * draft = llama_model_load_from_file(path, params);
    GGML_ASSERT(draft != nullptr);
    GGML_ASSERT(draft->arch == LLM_ARCH_QWEN4EXP);
    GGML_ASSERT(draft->tok_embd != nullptr && draft->output != nullptr);
    GGML_ASSERT(draft->tok_embd->type == draft->output->type);
    GGML_ASSERT(ggml_are_same_shape(draft->tok_embd, draft->output));

    unsigned char embedding[4096], output[4096];
    const size_t size = ggml_nbytes(draft->tok_embd);
    for (size_t offset = 0; offset < size; offset += sizeof(embedding)) {
        const size_t count = std::min(sizeof(embedding), size - offset);
        ggml_backend_tensor_get(draft->tok_embd, embedding, offset, count);
        ggml_backend_tensor_get(draft->output, output, offset, count);
        GGML_ASSERT(std::memcmp(embedding, output, count) == 0);
    }

    llama_model target = {};
    target.arch = LLM_ARCH_QWEN4EXP;
    ggml_tensor target_output = {};
    target.output = &target_output;
    ggml_tensor * own_output = draft->output;
    GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(draft, &target));
    GGML_ASSERT(draft->output == own_output);
    llama_free_model(draft);
}

int main(int argc, char ** argv) {
    if (argc == 2) {
        llama_backend_init();
        test_tied_companion(argv[1]);
        llama_backend_free();
        return 0;
    }
    GGML_ASSERT(argc == 1);
    llama_model target = {};
    target.arch = LLM_ARCH_LLAMA;

    ggml_tensor tok_embd = {};
    ggml_tensor output = {};

    GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(nullptr, &target));

    for (int own_io = 0; own_io < 4; ++own_io) {
        llama_model draft = {};
        draft.arch = LLM_ARCH_QWEN4EXP;
        draft.tok_embd = own_io & 1 ? &tok_embd : nullptr;
        draft.output   = own_io & 2 ? &output   : nullptr;

        GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(&draft, nullptr));
        GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&draft, &target) == (own_io == 3));
        GGML_ASSERT(draft.tok_embd == (own_io & 1 ? &tok_embd : nullptr));
        GGML_ASSERT(draft.output   == (own_io & 2 ? &output   : nullptr));
        GGML_ASSERT(draft.bufs.empty());
    }

    target.arch = LLM_ARCH_QWEN4EXP;
    target.tok_embd = &tok_embd;
    target.output   = &output;
    tok_embd.ne[0] = output.ne[0] = 4;
    tok_embd.ne[1] = output.ne[1] = 8;

    llama_model draft = {};
    draft.arch = LLM_ARCH_QWEN4EXP;
    draft.hparams.n_embd  = 4;
    draft.hparams.n_vocab = 8;

    target.output = nullptr;
    GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(&draft, &target));
    GGML_ASSERT(draft.output == nullptr);

    target.output = &output;
    output.ne[1] = 9;
    GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(&draft, &target));

    output.ne[1] = 8;
    GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&draft, &target));
    GGML_ASSERT(draft.tok_embd == &tok_embd);
    GGML_ASSERT(draft.output == &output);
    GGML_ASSERT(draft.bufs.empty());

    return 0;
}
