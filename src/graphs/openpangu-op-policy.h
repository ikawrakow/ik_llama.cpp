#pragma once

#include "ggml-backend.h"

// Pure policy seams kept separate from graph construction so the OpenPangu-only
// automatic defaults can be tested without loading a model.
//
// Production selection is capability-gated: each fused/specialized OpenPangu op is used
// whenever the scheduled backend supports it, and falls back to the exact legacy chain
// otherwise. GGML_OPENPANGU_LEGACY_OPS is a single documented bring-up switch resolved at
// COMPILE TIME, not from the environment: build with -DGGML_OPENPANGU_LEGACY_OPS=1 to force
// every legacy chain at once (the A/B baseline). It is all-or-nothing, not per-op policy, and
// its effect is reported in the normal parameter dump. Undefined or 0 (the default) selects
// the capability-gated production path.
#ifndef GGML_OPENPANGU_LEGACY_OPS
#define GGML_OPENPANGU_LEGACY_OPS 0
#endif

static inline bool openpangu_legacy_ops_forced() {
    return GGML_OPENPANGU_LEGACY_OPS != 0;
}

static inline bool openpangu_should_use_indexer_topk(
        bool explicit_request,
        bool automatic_enabled,
        ggml_backend_t placement_backend,
        const ggml_tensor * candidate) {
    if (explicit_request) {
        return true;
    }
    return automatic_enabled && placement_backend != nullptr && candidate != nullptr &&
           ggml_backend_supports_op(placement_backend, candidate);
}
