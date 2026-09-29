// Unit tests for the CPU affinity argument parsers (common/common.{h,cpp}).

#undef NDEBUG
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"

using cpus_t = std::vector<int32_t>;

int main() {
    cpus_t cpus;

    // cpu_affinity_parse_mask: bitmask in hex or decimal
    assert(cpu_affinity_parse_mask("0x55", cpus));
    assert((cpus == cpus_t{0, 2, 4, 6}));

    assert(cpu_affinity_parse_mask("85", cpus)); // decimal 85 == 0x55
    assert((cpus == cpus_t{0, 2, 4, 6}));

    assert(cpu_affinity_parse_mask("1", cpus));
    assert((cpus == cpus_t{0}));

    assert(cpu_affinity_parse_mask("0x8000000000000000", cpus)); // highest bit
    assert((cpus == cpus_t{63}));

    // invalid masks
    assert(!cpu_affinity_parse_mask("",              cpus));
    assert(!cpu_affinity_parse_mask("0",             cpus)); // empty selection
    assert(!cpu_affinity_parse_mask("foo",           cpus));
    assert(!cpu_affinity_parse_mask("0x",            cpus));
    assert(!cpu_affinity_parse_mask("12abc",         cpus));
    assert(!cpu_affinity_parse_mask("-1",            cpus)); // does not wrap to ULLONG_MAX
    assert(!cpu_affinity_parse_mask("0x1FFFFFFFFFFFFFFFF", cpus)); // > 64 bits

    // cpu_affinity_parse_range: single ids and ranges
    assert(cpu_affinity_parse_range("0", cpus));
    assert((cpus == cpus_t{0}));

    assert(cpu_affinity_parse_range("0-3", cpus));
    assert((cpus == cpus_t{0, 1, 2, 3}));

    assert(cpu_affinity_parse_range("0,2,4,6", cpus));
    assert((cpus == cpus_t{0, 2, 4, 6}));

    assert(cpu_affinity_parse_range("0-1,4,6-7", cpus));
    assert((cpus == cpus_t{0, 1, 4, 6, 7}));

    // invalid ranges
    assert(!cpu_affinity_parse_range("",                cpus));
    assert(!cpu_affinity_parse_range("foo",             cpus));
    assert(!cpu_affinity_parse_range("1-",              cpus));
    assert(!cpu_affinity_parse_range("-1-3",            cpus));
    assert(!cpu_affinity_parse_range("3-1",             cpus)); // reversed
    assert(!cpu_affinity_parse_range("0,,1",            cpus)); // empty item
    assert(!cpu_affinity_parse_range("1024",            cpus)); // >= GGML_MAX_CPU_AFFINITY
    assert(!cpu_affinity_parse_range("0-2000000000",    cpus)); // must not loop/allocate

    return 0;
}
