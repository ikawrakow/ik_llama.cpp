#undef NDEBUG
#include "chat.h"

#include <cassert>
#include <fstream>
#include <string>

int main() {
    std::ifstream f("models/templates/Kimi-K3.jinja");
    assert(f.good());
    std::string tmpl((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto tmpls = common_chat_templates_init(nullptr, tmpl);
    common_chat_tool tool{
        "get_weather", "Get weather",
        R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
    };
    common_chat_templates_inputs inputs;
    inputs.tools = { tool };
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    common_chat_msg user;
    user.role = "user";
    user.content = "weather in Tokyo?";
    inputs.messages = { user };

    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    assert(params.format == COMMON_CHAT_FORMAT_PEG_NATIVE);
    assert(!params.parser.empty());

    const std::string out =
        "<|open|>message role=\"assistant\"<|sep|>"
        "<|open|>think<|sep|>check<|close|>think<|sep|>"
        "<|open|>response<|sep|>ok<|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        "<|open|>call tool=\"get_weather\"<|sep|>"
        "<|open|>argument key=\"city\"<|sep|>Tokyo<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|open|>call tool=\"get_weather\"<|sep|>"
        "<|open|>argument key=\"city\"<|sep|>Paris<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
        "<|close|>message<|sep|><|end_of_msg|>";

    common_chat_parser_params parser_params(params);
    parser_params.parser.load(params.parser);
    parser_params.generation_prompt.clear();
    auto msg = common_chat_parse(out, false, parser_params);
    assert(msg.tool_calls.size() == 2);
    assert(msg.tool_calls[0].name == "get_weather");
    assert(msg.tool_calls[0].arguments.find("Tokyo") != std::string::npos);
    assert(msg.tool_calls[1].arguments.find("Paris") != std::string::npos);
    return 0;
}
