#include "../src/llama-rtr-auto.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <limits>

int main() {
    uint64_t peak = 0;
    assert(llama_rtr_auto_peak_bytes({
        100, 20, 10, 8, 16, false,
    }, peak));
    assert(peak == 200); // 100 + 20 + 8 * 10

    assert(llama_rtr_auto_peak_bytes({
        100, 20, 10, 8, 16, true,
    }, peak));
    assert(peak == 328); // plus 8 * 16 CUDA staging

    assert(!llama_rtr_auto_peak_bytes({
        std::numeric_limits<uint64_t>::max(), 1, 0, 8, 0, false,
    }, peak));
    assert(!llama_rtr_auto_peak_bytes({
        0, 0, std::numeric_limits<uint64_t>::max(), 2, 0, false,
    }, peak));
    return 0;
}
