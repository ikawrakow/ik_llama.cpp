#undef NDEBUG
#include "llama-arch.h"

#include <cassert>
#include <fstream>
#include <string>

int main() {
    assert(llm_arch_from_string("kimi-k3") == LLM_ARCH_KIMI_K3);
    assert(llm_arch_is_hybrid(LLM_ARCH_KIMI_K3));
    assert(!llm_arch_is_recurrent(LLM_ARCH_KIMI_K3));

    std::ifstream f("models/templates/Kimi-K3.jinja");
    assert(f.good());
    std::string tpl((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    assert(tpl.find("<|open|>")     != std::string::npos);
    assert(tpl.find("<|close|>")    != std::string::npos);
    assert(tpl.find("<|end_of_msg|>") != std::string::npos);

    return 0;
}
