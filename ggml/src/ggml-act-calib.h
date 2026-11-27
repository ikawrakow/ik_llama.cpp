#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Optional, CPU-only activation calibration for the FFN down projections.
//
// Set IK_ACT_CALIB=<file> to collect, per tensor and per expert, the observed
// amax of the down-projection input.  At exit a text file is written with one
// line per (tensor, expert):
//
//   tensor expert scale count median p99 p999 max mean
//
// The "scale" column is p99.9 by default and can be selected with
// IK_ACT_CALIB_QUANTILE=median|p99|p999|max.  The same file can be fed back
// through IK_ACT_SCALES (only the first three columns are read there).
bool ggml_act_calib_enabled(void);

void ggml_act_calib_observe_value(const char * name, int expert, float amax);

static inline void ggml_act_calib_observe_row(const char * name, int expert, const float * x, int64_t n) {
    float amax = 0.0f;
    for (int64_t j = 0; j < n; ++j) {
        const float v = x[j] < 0.0f ? -x[j] : x[j];
        if (v > amax) {
            amax = v;
        }
    }
    ggml_act_calib_observe_value(name, expert, amax);
}

#ifdef __cplusplus
}
#endif
