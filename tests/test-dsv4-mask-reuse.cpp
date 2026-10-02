#include "../src/graphs/dsv4-mask-view.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::cerr << "DSV4 mask reuse check failed: " << message << '\n';
        std::exit(1);
    }
}

static void set_values(ggml_tensor * tensor, const std::vector<float> & values) {
    check(ggml_is_contiguous(tensor), "input must be contiguous");
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
    } else {
        std::vector<ggml_fp16_t> half(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            half[i] = ggml_fp32_to_fp16(values[i]);
        }
        ggml_backend_tensor_set(tensor, half.data(), 0, half.size()*sizeof(ggml_fp16_t));
    }
}

static std::vector<float> get_values(ggml_tensor * tensor) {
    check(ggml_is_contiguous(tensor), "output must be contiguous");
    std::vector<float> values(ggml_nelements(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
    } else {
        std::vector<ggml_fp16_t> half(values.size());
        ggml_backend_tensor_get(tensor, half.data(), 0, half.size()*sizeof(ggml_fp16_t));
        for (size_t i = 0; i < half.size(); ++i) {
            values[i] = ggml_fp16_to_fp32(half[i]);
        }
    }
    return values;
}

static void check_crop(ggml_tensor * tensor, int x_offset, int y_offset, float shift) {
    const auto values = get_values(tensor);
    for (size_t i = 0; i < values.size(); ++i) {
        const int x = int(i % tensor->ne[0]);
        const int y = int(i / tensor->ne[0]);
        check(values[i] == shift - 12 + x + x_offset + 8*(y + y_offset), "crop or stream values");
    }
}

static void check_indexer(ggml_tensor * tensor, bool even, float shift) {
    const auto values = get_values(tensor);
    for (size_t i = 0; i < values.size(); ++i) {
        const int x = int(i % 4);
        const int y = int(i / 4);
        if ((x % 2 == 0) == even) {
            check(values[i] == shift - 12 + x + 1 + 8*(y + 1), "selected indexer values");
        } else {
            check(std::isinf(values[i]) && values[i] < 0, "unselected indexer values");
        }
    }
}

static void run(ggml_type type, ggml_backend_t backend) {
    ggml_init_params params = {4*1024*1024, nullptr, true};
    auto ctx = ggml_init(params);
    check(ctx != nullptr, "context allocation");
    dsv4_mask_view_cache cache;
    auto source = ggml_new_tensor_2d(ctx, type, 8, 6);
    ggml_set_name(source, "mask");
    ggml_set_input(source);
    auto other_source = ggml_new_tensor_2d(ctx, type, 8, 6);
    auto exact = dsv4_get_mask_view(ctx, cache, source, 8, 6, 1);
    check(exact == source && cache.empty(), "exact mask identity");
    auto exact_output = ggml_dup(ctx, exact);
    auto cropped = dsv4_get_mask_view(ctx, cache, source, 4, 4, 1);
    auto streamed = dsv4_get_mask_view(ctx, cache, source, 4, 4, 2);
    check(cropped == dsv4_get_mask_view(ctx, cache, source, 4, 4, 1), "crop identity");
    check(streamed == dsv4_get_mask_view(ctx, cache, source, 4, 4, 2), "stream identity");
    check(streamed != cropped && streamed->ne[1] == 2 && streamed->ne[3] == 2, "stream layout");
    check(dsv4_get_mask_view(ctx, cache, other_source, 4, 4, 1) != cropped, "source identity");
    auto narrower = dsv4_get_mask_view(ctx, cache, source, 3, 4, 1);
    auto shorter = dsv4_get_mask_view(ctx, cache, source, 4, 2, 1);
    check(narrower != cropped && shorter != cropped && narrower != shorter, "width/token identity");

    // Both source views have padded rows and a nonzero offset into the original input.
    auto offset_source = ggml_view_2d(ctx, source, 6, 4, source->nb[1], 9*source->nb[0]);
    auto next_source = ggml_view_2d(ctx, source, 6, 4, source->nb[1], 10*source->nb[0]);
    check(!ggml_is_contiguous(offset_source), "strided source");
    auto offset = dsv4_get_mask_view(ctx, cache, offset_source, 4, 4, 2);
    auto next = dsv4_get_mask_view(ctx, cache, next_source, 4, 4, 2);
    check(offset == dsv4_get_mask_view(ctx, cache, offset_source, 4, 4, 2), "offset identity");
    check(offset != next && offset != streamed, "different source offsets");

    auto top_a = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, 2, 1, 2);
    auto top_b = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 2, 2, 1, 2);
    auto layer_a = ggml_indexer_mask(ctx, offset, top_a);
    auto layer_b = ggml_indexer_mask(ctx, offset, top_b);
    check(layer_a != layer_b && layer_a->src[0] == layer_b->src[0], "independent layer outputs");

    // A new graph gets a new cache even when the source tensor is still alive.
    cache.clear();
    auto rebuilt = dsv4_get_mask_view(ctx, cache, source, 4, 4, 2);
    check(rebuilt != streamed, "cache reset");
    check(rebuilt == dsv4_get_mask_view(ctx, cache, source, 4, 4, 2), "rebuilt identity");
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    for (auto output : {exact_output, cropped, streamed, narrower, shorter, offset, next, layer_a, layer_b}) {
        ggml_build_forward_expand(graph, output);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "tensor allocation");
    const int32_t indices_a[] = {0, 2, 0, 2, 0, 2, 0, 2};
    const int32_t indices_b[] = {1, 3, 1, 3, 1, 3, 1, 3};
    ggml_backend_tensor_set(top_a, indices_a, 0, sizeof(indices_a));
    ggml_backend_tensor_set(top_b, indices_b, 0, sizeof(indices_b));

    for (float shift : {0.0f, 100.0f}) {
        std::vector<float> input(48);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = shift - 12 + float(i);
        }
        set_values(source, input);
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph execution");
        check(get_values(source) == input && get_values(exact_output) == input, "base mask unchanged");
        check_crop(cropped, 0, 0, shift);
        check_crop(streamed, 0, 0, shift);
        check_crop(narrower, 0, 0, shift);
        check_crop(shorter, 0, 0, shift);
        check_crop(offset, 1, 1, shift);
        check_crop(next, 2, 1, shift);
        check_indexer(layer_a, true, shift);
        check_indexer(layer_b, false, shift);
    }
    ggml_graph_clear(graph);
    ggml_build_forward_expand(graph, rebuilt);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "rebuilt graph execution");
    check_crop(rebuilt, 0, 0, 100.0f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    auto backend = ggml_backend_cpu_init();
    check(backend != nullptr, "CPU backend initialization");
    ggml_backend_cpu_set_n_threads(backend, 1);
    run(GGML_TYPE_F16, backend);
    run(GGML_TYPE_F32, backend);
    ggml_backend_free(backend);
    std::cout << "DSV4 mask reuse checks passed\n";
}
