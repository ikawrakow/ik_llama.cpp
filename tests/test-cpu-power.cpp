// CPU power helpers.

#undef NDEBUG
#include <cassert>
#include <cstdint>

#include "common.h"
#include "cpu_power.h"

int main() {
    assert(cpu_power_target_freq(4600000, 400000, 100) == 4600000);
    assert(cpu_power_target_freq(3400000, 400000, 100) == 3400000);
    assert(cpu_power_target_freq(4600000, 400000, 50) == 2300000);
    assert(cpu_power_target_freq(3400000, 400000, 65) == 2210000);
    assert(cpu_power_target_freq(1000000, 800000, 10) == 800000);
    assert(cpu_power_target_freq(4600000, 400000, 0) == 400000);
    assert(cpu_power_target_freq(4600000, 400000, -5) == 400000);
    assert(cpu_power_target_freq(5000000, 0, 99) == 4950000);

    assert(common_power_sleep_us(1000.0,  50) == 1000);
    assert(common_power_sleep_us(1000.0,  25) == 3000);
    assert(common_power_sleep_us(1000.0,   1) == 99000);
    assert(common_power_sleep_us(1000.0, 100) == 0);
    assert(common_power_sleep_us(1000.0,   0) == 0);
    assert(common_power_sleep_us(   0.0,  50) == 0);

    (void) cpu_power_supported();
    (void) cpu_power_backend_name();

    return 0;
}
