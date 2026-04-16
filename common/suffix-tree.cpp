#include "suffix-tree.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

common_suffix_tree::common_suffix_tree(int max_depth)
    : _max_depth(max_depth)
    , _root(std::make_unique<common_suffix_node>())
{}

common_suffix_tree::~common_suffix_tree() = default;

void common_suffix_tree::clear() {
    _root = std::make_unique<common_suffix_node>();
    _tokens.clear();
    _n_inserted = 0;
}

void common_suffix_tree::extend(const llama_token * tokens, int n_tokens) {
    if (n_tokens <= 0) return;

    const int old_size = (int)_tokens.size();
    _tokens.insert(_tokens.end(), tokens, tokens + n_tokens);
    const int new_size = (int)_tokens.size();

    // Insert/update suffixes that are affected by the new tokens.
    // For any position i, the suffix covers tokens[i .. min(i+max_depth, end)].
    // Positions within max_depth of the old end had truncated suffixes that
    // can now be extended with new tokens.
    const int reinsert_from = std::max(0, old_size - _max_depth);

    for (int i = reinsert_from; i < new_size; ++i) {
        if (i < _n_inserted) {
            const int old_len = std::min(old_size - i, _max_depth);
            const int new_len = std::min(new_size - i, _max_depth);
            if (new_len > old_len) {
                _extend_suffix(i, old_len, new_len);
            }
        } else {
            _insert_suffix(i);
        }
    }

    _n_inserted = new_size;
}

void common_suffix_tree::_insert_suffix(int start_pos) {
    const int total = (int)_tokens.size();
    const int len = std::min(total - start_pos, _max_depth);
    if (len <= 0) return;

    common_suffix_node * node = _root.get();

    for (int i = 0; i < len; ++i) {
        const llama_token tok = _tokens[start_pos + i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            auto child = std::make_unique<common_suffix_node>();
            auto * child_ptr = child.get();
            child_ptr->count = 1;
            node->children[tok] = std::move(child);
            node = child_ptr;
        } else {
            node = it->second.get();
            node->count++;
        }
    }
}

void common_suffix_tree::_extend_suffix(int start_pos, int old_len, int new_len) {
    common_suffix_node * node = _root.get();

    for (int i = 0; i < old_len; ++i) {
        const llama_token tok = _tokens[start_pos + i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            return;
        }
        node = it->second.get();
    }

    for (int i = old_len; i < new_len; ++i) {
        const llama_token tok = _tokens[start_pos + i];
        auto it = node->children.find(tok);
        if (it == node->children.end()) {
            auto child = std::make_unique<common_suffix_node>();
            auto * child_ptr = child.get();
            child_ptr->count = 1;
            node->children[tok] = std::move(child);
            node = child_ptr;
        } else {
            node = it->second.get();
            node->count++;
        }
    }
}

std::vector<llama_token> common_suffix_tree::speculate(
        const llama_token * context, int n_context,
        int max_spec_tokens,
        float min_token_prob,
        int   min_match_count,
        int   min_match_len) const {

    std::vector<llama_token> best_draft;

    if (!_root || n_context <= 0 || max_spec_tokens <= 0) return best_draft;

    if (n_context > _max_depth) {
        context += (n_context - _max_depth);
        n_context = _max_depth;
    }

    float best_score = 0.0f;

    for (int match_len = std::max(1, min_match_len); match_len <= n_context; ++match_len) {
        const llama_token * ctx = context + (n_context - match_len);

        const common_suffix_node * node = _root.get();
        bool matched = true;
        for (int i = 0; i < match_len; ++i) {
            auto it = node->children.find(ctx[i]);
            if (it == node->children.end()) {
                matched = false;
                break;
            }
            node = it->second.get();
        }

        if (!matched) break;
        if (node->count < min_match_count) continue;
        if (node->children.empty()) continue;

        // Speculate: greedily follow highest-count child
        // Probability decays multiplicatively: prob *= child_count / parent_count
        const int draft_limit = std::min(max_spec_tokens, match_len + 8);

        std::vector<llama_token> draft;
        float score = 0.0f;
        float prob  = 1.0f;
        const common_suffix_node * cur = node;

        for (int i = 0; i < draft_limit; ++i) {
            if (cur->children.empty()) break;

            llama_token best_tok   = -1;
            int64_t     best_count = 0;
            for (const auto & [token, child] : cur->children) {
                if (child->count > best_count) {
                    best_count = child->count;
                    best_tok   = token;
                }
            }

            prob *= (float)best_count / (float)cur->count;
            if (prob < min_token_prob) break;

            score += prob;
            draft.push_back(best_tok);
            cur = cur->children.at(best_tok).get();
        }

        if (score > best_score && !draft.empty()) {
            best_score = score;
            best_draft = std::move(draft);
        }
    }

    return best_draft;
}

static void _extract_texts(const json & node, std::vector<std::string> & out) {
    if (node.is_string()) {
        const std::string s = node.get<std::string>();
        if (!s.empty()) out.push_back(s);
    } else if (node.is_array()) {
        for (const auto & item : node) {
            _extract_texts(item, out);
        }
    } else if (node.is_object()) {
        if (node.contains("content") && node["content"].is_string()) {
            const std::string s = node["content"].get<std::string>();
            if (!s.empty()) out.push_back(s);
        } else if (node.contains("messages")) {
            _extract_texts(node["messages"], out);
        }
    }
}

bool common_suffix_tree::load_corpus(
        const std::string & path,
        std::function<std::vector<llama_token>(const std::string &)> tokenize_fn) {

    bool is_json = path.size() >= 5 &&
                   path.compare(path.size() - 5, 5, ".json") == 0;

    if (is_json) {
        if (!tokenize_fn) {
            LOG_ERR("%s: JSON corpus requires a tokenizer but none was provided (path: '%s')\n",
                    __func__, path.c_str());
            return false;
        }
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_ERR("%s: failed to open corpus file '%s'\n", __func__, path.c_str());
            return false;
        }
        json root;
        try {
            f >> root;
        } catch (const json::exception & e) {
            LOG_ERR("%s: JSON parse error in '%s': %s\n", __func__, path.c_str(), e.what());
            return false;
        }
        std::vector<std::string> texts;
        _extract_texts(root, texts);
        if (texts.empty()) {
            LOG_WRN("%s: no text content found in corpus '%s'\n", __func__, path.c_str());
            return false;
        }
        size_t total_tokens = 0;
        for (const auto & text : texts) {
            auto tokens = tokenize_fn(text);
            if (!tokens.empty()) {
                extend(tokens.data(), (int)tokens.size());
                total_tokens += tokens.size();
            }
        }
        LOG_DBG("%s: loaded JSON corpus — %zu texts, %zu tokens from '%s'\n",
                __func__, texts.size(), total_tokens, path.c_str());
        return true;
    }

    // Binary format: raw int32 token IDs
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        LOG_ERR("%s: failed to open corpus file '%s'\n", __func__, path.c_str());
        return false;
    }
    std::vector<llama_token> tokens;
    int32_t tok;
    while (std::fread(&tok, sizeof(tok), 1, fp) == 1) {
        tokens.push_back(tok);
    }
    std::fclose(fp);
    if (tokens.empty()) {
        LOG_WRN("%s: suffix corpus file '%s' is empty\n", __func__, path.c_str());
        return false;
    }
    extend(tokens.data(), (int)tokens.size());
    LOG_DBG("%s: loaded binary corpus — %zu tokens from '%s'\n",
            __func__, tokens.size(), path.c_str());
    return true;
}
