#include "ggml-backend.h"
#include "ggml-moe-provider.h"
#include "llama-moe-cache-residency.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr float sentinel = -123456.0f;
enum class behavior { mixed, reject_dispatch, fail_collect };

struct fake_provider {
    behavior mode;
    const std::vector<float> * reference;
    size_t width;
    size_t selected;
    size_t begins = 0, plans = 0, dispatches = 0, collects = 0, ends = 0;
};

bool begin(void * data, const ggml_tensor *, void ** request) {
    auto & fake = *static_cast<fake_provider *>(data);
    ++fake.begins;
    *request = &fake;
    return true;
}

bool plan(void * data, void *, const ggml_tensor *, uint8_t * rows, size_t count) {
    auto & fake = *static_cast<fake_provider *>(data);
    ++fake.plans;
    for (size_t row = 0; row < count; ++row) rows[row] = (row & 1) == 0;
    return true;
}

bool dispatch(void * data, void *, ggml_tensor * op, const uint8_t * rows, size_t count) {
    auto & fake = *static_cast<fake_provider *>(data);
    ++fake.dispatches;
    if (fake.mode == behavior::reject_dispatch) return false;
    for (size_t row = 0; row < count; ++row) {
        if (!rows[row]) continue;
        const size_t token = row / fake.selected;
        const size_t selection = row % fake.selected;
        auto * out = reinterpret_cast<float *>(reinterpret_cast<char *>(op->data) +
                token*op->nb[2] + selection*op->nb[1]);
        std::fill(out, out + fake.width, sentinel);
    }
    return true;
}

bool collect(void * data, void *, ggml_tensor * op, const uint8_t * rows, size_t count) {
    auto & fake = *static_cast<fake_provider *>(data);
    ++fake.collects;
    if (fake.mode == behavior::fail_collect) return false;
    for (size_t row = 0; row < count; ++row) {
        const size_t token = row / fake.selected;
        const size_t selection = row % fake.selected;
        auto * out = reinterpret_cast<float *>(reinterpret_cast<char *>(op->data) +
                token*op->nb[2] + selection*op->nb[1]);
        if (rows[row]) {
            if (!std::all_of(out, out + fake.width, [](float x) { return x == sentinel; })) return false;
            std::memcpy(out, fake.reference->data() + row*fake.width, fake.width*sizeof(float));
        } else if (std::memcmp(out, fake.reference->data() + row*fake.width, fake.width*sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

void end(void * data, void *) { ++static_cast<fake_provider *>(data)->ends; }

void fill(ggml_tensor * tensor, float scale, float offset) {
    auto * values = static_cast<float *>(tensor->data);
    for (size_t i = 0; i < ggml_nelements(tensor); ++i) {
        values[i] = offset + scale*(float)((int)(i % 17) - 8);
    }
}

bool run_boundary_case(size_t batch, bool attach_provider, behavior mode) {
    constexpr size_t top_k = 2, experts = 4, input_width = 8, output_width = 6;
    ggml_context * ctx = ggml_init({ 4*1024*1024, nullptr, false });
    if (!ctx) return false;

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_width, top_k, batch);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, batch);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, input_width, output_width, experts);
    fill(input, 0.03125f, 0.25f);
    fill(weights, 0.015625f, 0.125f);
    auto * route = static_cast<int32_t *>(ids->data);
    for (size_t row = 0; row < batch*top_k; ++row) route[row] = (int32_t)(row % experts);

    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { ggml_free(ctx); return false; }
    ggml_backend_cpu_set_n_threads(cpu, 2);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_buffer_type_t bufts[] = { ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1,
            graph->n_nodes + graph->n_leafs, false);
    if (!sched) { ggml_backend_free(cpu); ggml_free(ctx); return false; }

    bool ok = ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
    const size_t rows = batch*top_k;
    std::vector<float> reference(rows*output_width);
    for (size_t row = 0; row < rows; ++row) {
        const size_t token = row/top_k, selection = row%top_k;
        const auto * src = reinterpret_cast<const float *>(reinterpret_cast<const char *>(output->data) +
                token*output->nb[2] + selection*output->nb[1]);
        std::memcpy(reference.data() + row*output_width, src, output_width*sizeof(float));
    }

    fake_provider fake{ mode, &reference, output_width, top_k };
    const ggml_moe_provider provider{ &fake, begin, plan, dispatch, collect, end };
    std::memset(output->data, 0, ggml_nbytes(output));
    if (attach_provider) ggml_backend_sched_set_moe_provider(sched, &provider);
    ok = ok && ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_sched_set_moe_provider(sched, nullptr);

    for (size_t row = 0; row < rows; ++row) {
        const size_t token = row/top_k, selection = row%top_k;
        const auto * actual = reinterpret_cast<const float *>(reinterpret_cast<const char *>(output->data) +
                token*output->nb[2] + selection*output->nb[1]);
        ok = ok && std::memcmp(actual, reference.data() + row*output_width,
                output_width*sizeof(float)) == 0;
    }

    const bool bypass = !attach_provider || batch > 32;
    if (bypass) {
        ok = ok && fake.begins == 0 && fake.plans == 0 && fake.dispatches == 0 &&
                fake.collects == 0 && fake.ends == 0;
    } else {
        ok = ok && fake.begins == 1 && fake.plans == 1 && fake.ends == 1;
        if (mode == behavior::mixed) ok = ok && fake.dispatches == 1 && fake.collects == 1;
        if (mode == behavior::reject_dispatch) ok = ok && fake.dispatches == 1 && fake.collects == 0;
        if (mode == behavior::fail_collect) ok = ok && fake.dispatches == 1 && fake.collects == 1;
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return ok;
}

bool residency_contract() {
    using residency = llama_moe_cache_residency;
    const residency::key key{ 2, 3 };
    residency state(1, 4, 8);
    uint64_t generation = 0;
    if (!state.reserve(0, key, generation)) return false;
    if (state.find(key).state == residency::slot_state::ready) return false;
    int32_t slot = -1;
    residency::pin pin[1]{};
    size_t n_pins = 0;
    if (!state.pin_rows(&key, 1, &slot, pin, 1, n_pins) || n_pins != 0) return false;
    if (state.complete(0, key, generation + 1) || state.find(key).state == residency::slot_state::ready) return false;
    if (!state.complete(0, key, generation) || state.find(key).state != residency::slot_state::ready) return false;
    if (!state.pin_rows(&key, 1, &slot, pin, 1, n_pins) || n_pins != 1) return false;
    uint64_t replacement_generation = 0;
    if (state.replace_ready(0, key, generation, { 1, 1 }, replacement_generation)) return false;
    if (!state.retire(0, key, generation) || state.inspect(0).state != residency::slot_state::retired) return false;
    uint64_t next_generation = 0;
    if (state.reserve(0, { 1, 1 }, next_generation)) return false;
    if (!state.release_pins(pin, n_pins) || !state.reusable(0) || state.live_pins() != 0) return false;
    if (!state.reserve(0, { 1, 1 }, next_generation) || next_generation <= generation) return false;
    if (state.complete(0, key, generation) || !state.cancel(0, { 1, 1 }, next_generation)) return false;
    return state.reusable(0) && state.live_pins() == 0;
}

} // namespace

int main() {
    if (!residency_contract() ||
            !run_boundary_case(4, false, behavior::mixed) ||
            !run_boundary_case(4, true, behavior::mixed) ||
            !run_boundary_case(4, true, behavior::reject_dispatch) ||
            !run_boundary_case(4, true, behavior::fail_collect) ||
            !run_boundary_case(33, true, behavior::mixed)) {
        std::fprintf(stderr, "async MoE provider regression failed\n");
        return 1;
    }
    std::puts("async MoE provider regression passed");
    return 0;
}
