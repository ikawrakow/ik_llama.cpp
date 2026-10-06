#pragma once

// quillama.cpp: sparse output projection (lm_head) for grammar-constrained decoding.
//
// When the grammar allows only a few tokens at the next position, every other logit is masked to
// -inf by the sampler anyway. ggml_quill_set_logit_subset() registers those token ids for the next
// graph evaluation: the single-row matmul with weight `w` then computes only the rows (row groups,
// for interleaved repacks) holding them, with the same kernels as the full product (bit-identical
// values), and writes -inf everywhere else. This skips reading most of a 100k-250k row output
// matrix (e.g. 255 MB per token for Qwen3 at Q6_K). n = 0 clears it.

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_quill_set_logit_subset(const struct ggml_tensor * w, const int32_t * ids, int n);

#ifdef __cplusplus
}
#endif
