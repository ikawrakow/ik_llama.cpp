#include "../examples/server/server-sampling.h"

#include "ggml.h"

static void test_request_sampling_state_is_reset() {
    common_params_sampling defaults;
    defaults.logit_bias[7] = 0.25f;
    defaults.dry_sequence_breakers = {"default"};
    defaults.elb_params.resize(1);
    defaults.elb_params[0].exitword = "default exit";

    common_params_sampling slot_sparams;
    slot_sparams.elb_params.resize(1);
    slot_sparams.elb_params[0].exitword = "previous request";
    slot_sparams.logit_bias[42] = -INFINITY;
    slot_sparams.dry_sequence_breakers = {"stale request value"};

    const auto previous_elb_params = server_apply_sampling_defaults(slot_sparams, defaults);

    GGML_ASSERT(previous_elb_params.size() == 1);
    GGML_ASSERT(previous_elb_params[0].exitword == "previous request");
    GGML_ASSERT(slot_sparams.logit_bias == defaults.logit_bias);
    GGML_ASSERT(slot_sparams.dry_sequence_breakers == defaults.dry_sequence_breakers);
    GGML_ASSERT(slot_sparams.elb_params == defaults.elb_params);
}

int main() {
    test_request_sampling_state_is_reset();
    return 0;
}
