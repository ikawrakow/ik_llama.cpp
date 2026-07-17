#include "common.h"
#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <initializer_list>
#include <string>
#include <vector>

static gpt_params parse(std::initializer_list<const char *> args) {
    std::vector<std::string> storage;
    storage.reserve(args.size() + 1);
    storage.emplace_back("test-rtr-params");
    for (const char * arg : args) {
        storage.emplace_back(arg);
    }

    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (std::string & arg : storage) {
        argv.push_back(&arg[0]);
    }

    gpt_params params;
    const bool ok = gpt_params_parse_ex((int) argv.size(), argv.data(), params);
    assert(ok);
    return params;
}

int main() {
    {
        const gpt_params params = parse({ "-rtr", "1", "-rtr", "auto" });
        assert(params.repack_tensors);
        assert(params.repack_tensors_auto);
        assert(params.use_mmap);
    }
    {
        const gpt_params params = parse({ "-rtr", "1", "-rtr", "0" });
        assert(!params.repack_tensors);
        assert(!params.repack_tensors_auto);
        assert(params.use_mmap);
    }
    {
        const gpt_params params = parse({ "--no-mmap", "-rtr", "auto" });
        assert(params.repack_tensors);
        assert(params.repack_tensors_auto);
        assert(!params.use_mmap);
    }
    {
        const gpt_params params = parse({ "-rtr", "auto", "-rtr", "on" });
        assert(params.repack_tensors);
        assert(!params.repack_tensors_auto);
        // The loader, not the parser, applies the legacy forced-repack coupling.
        assert(params.use_mmap);
    }
    {
        const gpt_params params = parse({ "-rtr" });
        assert(params.repack_tensors);
        assert(!params.repack_tensors_auto);
        assert(params.use_mmap);
    }
    {
        const gpt_params params = parse({ "-rtra" });
        assert(params.repack_tensors);
        assert(params.repack_tensors_auto);
        assert(params.use_mmap);
    }

    assert(!llama_model_loader_mmap_enabled(nullptr));
    assert(!llama_model_has_mmap_buffers(nullptr));
    assert(!llama_model_repack_pass_executed(nullptr));
    assert(llama_model_n_repacked(nullptr) == 0);

    return 0;
}
