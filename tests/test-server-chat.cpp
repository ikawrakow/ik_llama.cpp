#include "server-context.h"
#include "chat-peg-parser.h"
#include "testing.h"

#include <string>

std::string gen_tool_call_id() {
    return "test-tool-call-id";
}

static common_chat_parser_params make_parser_params() {
    common_chat_parser_params params;
    params.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    params.parser = build_chat_peg_parser([](common_chat_peg_builder & p) {
        return p.literal("<think>") +
               p.literal("working through the request</think>") +
               p.end();
    });
    return params;
}

static void test_update_chat_msg_forwards_partial_flag(testing & t) {
    const auto params = make_parser_params();
    // The malformed closing marker makes the PEG parser fail after consuming a
    // prefix, exercising common_chat_parse's partial-recovery branch.
    const std::string partial_output = "<think>working through the request</thx";

    server_slot partial_slot;
    partial_slot.stop = STOP_TYPE_EOS;
    partial_slot.params.chat_parser_params = params;
    partial_slot.generated_text = partial_output;

    std::vector<common_chat_msg_diff> partial_diffs;
    try {
        const auto & partial = partial_slot.update_chat_msg(true, partial_diffs);
        t.assert_true("partial update accepts parser recovery", partial.empty());
    } catch (const std::exception & ex) {
        t.assert_true(std::string("partial update should parse: ") + ex.what(), false);
    }

    server_slot final_slot;
    final_slot.stop = STOP_TYPE_NONE;
    final_slot.params.chat_parser_params = params;
    final_slot.generated_text = partial_output;

    std::vector<common_chat_msg_diff> final_diffs;
    bool final_threw = false;
    try {
        final_slot.update_chat_msg(false, final_diffs);
    } catch (const std::exception &) {
        final_threw = true;
    }
    t.assert_true("final update rejects incomplete output", final_threw);
}

int main() {
    testing t(std::cout);
    t.verbose = true;
    t.test("update_chat_msg_forwards_partial_flag", test_update_chat_msg_forwards_partial_flag);
    return t.summary();
}
