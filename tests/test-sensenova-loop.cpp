// partial layer-loop plan tests

#undef NDEBUG
#include <cassert>
#include <vector>

#include "llama-hparams.h"

static std::vector<uint32_t> execution_plan(const llama_hparams & h) {
    std::vector<uint32_t> out(h.n_layer_all);
    for (uint32_t il = 0; il < h.n_layer_all; ++il) {
        out[il] = h.phys_layer(il);
    }
    return out;
}

int main() {
    llama_hparams h;
    h.n_layer = 32;

    // no loop: identity
    //assert(!h.has_layer_loop());
    for (uint32_t il = 0; il < 32; ++il) {
        assert(h.phys_layer(il) == il);
    }

    // SenseNova-Loop: [4, 28) twice
    h.n_layer_phys = 32;
    h.loop_start   = 4;
    h.loop_end     = 28;
    h.n_loops      = 2;
    h.n_layer_all  = 56;

    //assert(h.has_layer_loop());
    assert(h.n_layer_all == 56);

    const std::vector<uint32_t> expected = {
        0, 1, 2, 3,
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31,
    };
    assert(execution_plan(h) == expected);

    // official layout: the tail keeps its unrolled block index
    h.loop_tail_sparse = true;
    assert(h.stored_layer(4) == 4 && h.stored_layer(27) == 27);
    assert(h.stored_layer(28) == 52 && h.stored_layer(31) == 55);
    h.loop_tail_sparse = false;

    // full-model loop (Nanbeige-style)
    h.loop_start  = 0;
    h.loop_end    = 22;
    h.n_layer_phys = 22;
    h.n_loops     = 3;
    h.n_layer_all = 66;

    const std::vector<uint32_t> full = execution_plan(h);
    assert(full.size() == 66);
    for (uint32_t il = 0; il < 66; ++il) {
        assert(full[il] == il % 22);
    }

    return 0;
}
