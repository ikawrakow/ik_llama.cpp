#pragma once

#include "llama-impl.h"
#include <map>
#include <regex>
#include <string>
#include <vector>

struct llama_vocab;
struct llama_sampling;

struct llama_grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;

    llama_grammar_rules rules;

    llama_grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char* src, size_t len);
    uint32_t generate_symbol_id(const std::string& base_name);

    void add_rule(uint32_t rule_id, const llama_grammar_rule& rule);

    const char* parse_alternates(
        const char* src,
        const std::string& rule_name,
        uint32_t            rule_id,
        bool                is_nested);

    const char* parse_sequence(
        const char* src,
        const std::string& rule_name,
        llama_grammar_rule& rule,
        bool               is_nested);

    const char* parse_rule(const char* src);

    bool parse(const char* src);
    void print(FILE* file);
};

struct llama_grammar_trigger_pattern {
    std::string pattern;
    std::regex  regex;
};

struct llama_grammar {
    // note: allow null vocab for testing (not great)
    const llama_vocab* vocab;

    const llama_grammar_rules  rules;  // TODO: shared ptr
    llama_grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    llama_partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<llama_grammar_trigger_pattern> trigger_patterns;
               // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                                  // string, and the grammar will be given the string from the first match group onwards.

};

//
// internal API
//
// note: needed for tests (not great)
struct llama_grammar* llama_grammar_init_impl(
    const llama_grammar_element** rules,
    size_t n_rules,
    size_t start_rule_index);

struct llama_grammar* llama_grammar_init_impl(
    const struct llama_vocab* vocab,
    const char* grammar_str,
    const char* grammar_root,
    bool lazy,
    const char** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token* trigger_tokens,
    size_t num_trigger_tokens);

void llama_grammar_free_impl(struct llama_grammar * grammar);

struct llama_grammar * llama_grammar_copy_impl(const struct llama_grammar * grammar);

void llama_grammar_sample_impl(
        const struct llama_grammar * grammar,
          const struct llama_vocab * vocab,
       const struct llama_sampling * smpl,
            llama_token_data_array * candidates);

void llama_grammar_accept_token_impl(
              struct llama_grammar * grammar,
          const struct llama_vocab * vocab,
       const struct llama_sampling * smpl,
                       llama_token   token);


void llama_grammar_accept_str(
    struct llama_grammar* grammar,
    const std::string& piece);
