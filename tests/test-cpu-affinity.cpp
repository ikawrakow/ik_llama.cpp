// Unit tests for the CPU affinity argument parsers (common/common.{h,cpp}).

#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "common.h"
#include "speculative.h"

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

    // P-core and E-core sets must stay disjoint
    const cpus_t pcpus = cpu_get_math_cpus();
    for (const int32_t cpu : cpu_get_efficiency_cpus()) {
        assert(std::find(pcpus.begin(), pcpus.end(), cpu) == pcpus.end());
    }

    // empty list without auto = no pinning
    assert(cpu_affinity_resolve({}, false).empty());
    assert(cpu_affinity_resolve_draft({}, false).empty());

    // --draft-params affinity is resolved and flagged, even when empty
    {
        gpt_params p;
        p.speculative.params = "--cpu-affinity";
        assert(common_speculative_prepare_startup(p, false));
        assert(p.speculative.cpu_affinity_configured);
        assert(p.speculative.cpu_affinity == cpu_affinity_resolve_draft({}, true));
    }
    {
        gpt_params p;
        p.speculative.params = "-cr 0-1";
        assert(common_speculative_prepare_startup(p, false));
        assert(p.speculative.cpu_affinity_configured);
        assert(p.speculative.cpu_affinity == cpu_affinity_resolve_draft({0, 1}, false));
    }
    {
        gpt_params p;
        p.speculative.params = "-cm 0x3"; // bitmask = CPUs 0 and 1
        assert(common_speculative_prepare_startup(p, false));
        assert(p.speculative.cpu_affinity_configured);
        assert(p.speculative.cpu_affinity == cpu_affinity_resolve_draft({0, 1}, false));
    }
    // invalid --draft-params aborts startup
    {
        gpt_params p;
        p.speculative.params = "-cr nope";
        assert(!common_speculative_prepare_startup(p, false));
    }

    return 0;
}
