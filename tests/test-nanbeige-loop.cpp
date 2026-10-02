// Unit tests for the Nanbeige loop semantics (llama-hparams.h).

#undef NDEBUG
#include <cassert>
#include <vector>

#include "llama-hparams.h"

static std::vector<int> loop_final_norm_layers(const llama_hparams & h, int n_layer) {
    std::vector<int> out;
    for (int il = 0; il < n_layer; ++il) {
        if (h.needs_loop_final_norm(il)) {
            out.push_back(il);
        }
    }
    return out;
}

int main() {
    llama_hparams h;
    h.n_layer = 22;

    // no loop: n_layer_phys == 0 and there is no loop boundary
    for (int il = 0; il < 22; ++il) {
        assert(!h.needs_loop_final_norm(il));
    }

    // 2 loops: norm after layer 21 only
    h.n_layer_phys = 22;
    h.n_loops      = 2;
    h.n_layer_all  = 44;
    h.n_layer      = 44;
    assert((loop_final_norm_layers(h, 44) == std::vector<int>{21}));

    // 3 loops: between consecutive passes only
    h.n_layer_phys = 22;
    h.n_loops      = 3;
    h.n_layer_all  = 66;
    h.n_layer      = 66;
    assert((loop_final_norm_layers(h, 66) == std::vector<int>{21, 43}));

    // skip_loop_final_norm disables the between-loop norm entirely
    h.skip_loop_final_norm = true;
    assert(loop_final_norm_layers(h, 66).empty());

    return 0;
}
