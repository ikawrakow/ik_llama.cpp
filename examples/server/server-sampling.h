#pragma once

#include "common.h"

// Sampling parameters are kept on a reusable server slot. Restore the
// server-wide defaults before applying a new request's overrides, while
// preserving the previous expiring-logit-bias parameters for state reuse.
inline std::vector<common_params_sampling::elb_param> server_apply_sampling_defaults(
        common_params_sampling & slot_sparams,
        const common_params_sampling & defaults) {
    auto previous_elb_params = slot_sparams.elb_params;
    slot_sparams = defaults;
    return previous_elb_params;
}
