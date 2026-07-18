#pragma once

#include "ggml.h"

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
        // The CPY/PACK write-indirection pointer table is cudaMalloc'd on demand and reused
        // across replays; free it here so the device table is not leaked on graph destruction.
        if (dest_ptrs_d != nullptr) {
            CUDA_CHECK(cudaFree(dest_ptrs_d));
        }
    }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    // CPY and destination-rooted mutation ops share one graph-order pointer table.
    bool use_write_indirection = false;
    std::vector<char *> write_dest_ptrs;
    char ** dest_ptrs_d = nullptr;
    int dest_ptrs_size = 0;
    // Index of the next mutable write kernel, shared by CPY and PACK_CACHE_ROWS.
    int graph_write_index = -1;
#endif
};
