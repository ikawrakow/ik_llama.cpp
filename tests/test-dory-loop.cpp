// Dory looped-nGPT execution plan tests

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

    // Dory 1.0 4B: 72 physical modules = input 16 + recurrent 40 + output 16, 2 loops.
    // The recurrent group [16, 56) executes twice; weights shared, KV slot per visit.
    h.n_layer      = 72;
    h.n_layer_phys = 72;
    h.loop_start   = 16;
    h.loop_end     = 56;
    h.n_loops      = 2;
    h.n_layer_all  = 112;

    const std::vector<uint32_t> plan = execution_plan(h);
    assert(plan.size() == 112);

    // input phase applied once: slots 0..15 -> modules 0..15
    for (int i = 0; i < 16; ++i) {
        assert(plan[i] == (uint32_t) i);
    }
    // recurrent pass 0: slots 16..55 -> modules 16..55
    for (int i = 0; i < 40; ++i) {
        assert(plan[16 + i] == (uint32_t) (16 + i));
    }
    // recurrent pass 1: slots 56..95 -> modules 16..55 (shared weights)
    for (int i = 0; i < 40; ++i) {
        assert(plan[56 + i] == (uint32_t) (16 + i));
    }
    // output phase applied once: slots 96..111 -> modules 56..71
    for (int i = 0; i < 16; ++i) {
        assert(plan[96 + i] == (uint32_t) (56 + i));
    }
    // attention/MLP alternation is preserved by the plan (even module attends, odd is MLP)
    for (int il = 0; il < 112; ++il) {
        assert(plan[il] % 2 == (uint32_t) il % 2);
    }

    // 4-loop variant: 72 + 3*40 = 192 slots
    h.n_loops     = 4;
    h.n_layer_all = 192;
    const std::vector<uint32_t> plan4 = execution_plan(h);
    assert(plan4.size() == 192);
    for (int p = 0; p < 4; ++p) {
        for (int i = 0; i < 40; ++i) {
            assert(plan4[16 + p*40 + i] == (uint32_t) (16 + i));
        }
    }
    for (int i = 0; i < 16; ++i) {
        assert(plan4[176 + i] == (uint32_t) (56 + i));
    }

    // no loop: identity
    h.n_loops     = 1;
    h.n_layer_all = 72;
    for (uint32_t il = 0; il < 72; ++il) {
        assert(h.phys_layer(il) == il);
    }

    return 0;
}
