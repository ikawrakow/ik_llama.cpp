#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <vector>

static void test_history_grows_to_capacity() {
    common_sampler sampler{};
    sampler.params.n_prev = 3;

    common_sampler_accept(&sampler, nullptr, 10, false);
    common_sampler_accept(&sampler, nullptr, 11, false);
    common_sampler_accept(&sampler, nullptr, 12, false);

    assert((sampler.prev == std::vector<llama_token>{10, 11, 12}));

    common_sampler_accept(&sampler, nullptr, 13, false);

    assert((sampler.prev == std::vector<llama_token>{11, 12, 13}));
}

static void test_zero_capacity_keeps_no_history() {
    common_sampler sampler{};
    sampler.params.n_prev = 0;

    common_sampler_accept(&sampler, nullptr, 10, false);

    assert(sampler.prev.empty());
}

int main() {
    test_history_grows_to_capacity();
    test_zero_capacity_keeps_no_history();

    return 0;
}
