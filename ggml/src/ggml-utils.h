#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;

// Finds the copy the delta-net fusion can skip: the CPY that writes this node's new recurrent
// state back into the slot it was read from. Its index in cgraph, or -1 if there is none.
int ggml_delta_net_find_state_cpy(const struct ggml_cgraph * cgraph, int i);

#ifdef __cplusplus
}
#endif
