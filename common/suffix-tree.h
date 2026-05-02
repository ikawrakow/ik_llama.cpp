#pragma once

#include "llama.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// A trie-based suffix tree for suffix-decoding speculative decoding.
//
// Stores all suffixes (up to max_depth) of the token history.
// Used to find matching patterns in context and generate draft tokens
// by following the most frequent continuation path.
//
// Reference: "Suffix Decoding" (Saxena et al., 2024) — arXiv:2411.04975

struct common_suffix_node {
    int64_t count = 0;
    std::unordered_map<llama_token, std::unique_ptr<common_suffix_node>> children;
};

class common_suffix_tree {
public:
    explicit common_suffix_tree(int max_depth = 64);
    ~common_suffix_tree();

    // Append tokens to the history and insert new suffixes into the trie.
    // Incremental: only processes suffixes that haven't been inserted yet.
    void extend(const llama_token * tokens, int n_tokens);

    void clear();

    // Generate draft tokens by matching the context in the trie.
    // Tries multiple context lengths and returns the draft with the best score.
    std::vector<llama_token> speculate(
            const llama_token * context, int n_context,
            int max_spec_tokens,
            float min_token_prob  = 0.1f,
            int   min_match_count = 1,
            int   min_match_len   = 5) const;

    // Load an offline corpus to pre-warm the tree before any request.
    // Supported formats (.json or .bin)
    bool load_corpus(
            const std::string & path,
            std::function<std::vector<llama_token>(const std::string &)> tokenize_fn = {});

    int  max_depth()   const { return _max_depth; }
    int  token_count() const { return (int)_tokens.size(); }

private:
    int _max_depth;
    std::unique_ptr<common_suffix_node> _root;
    std::vector<llama_token> _tokens;
    int _n_inserted = 0;

    void _insert_suffix(int start_pos);
    void _extend_suffix(int start_pos, int old_len, int new_len);
};
