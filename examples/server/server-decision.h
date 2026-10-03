#pragma once

#include "server-common.h"
#include "server-task.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// typed decision models (TypeSafe /v1/systemone API)
// the model answers each question in one forward pass, no token is generated

// typed decision models, see "<arch>.decision.type" in the model metadata
enum common_decision_type {
    COMMON_DECISION_TYPE_NONE,    // not a decision model
    COMMON_DECISION_TYPE_OPENJEV, // logits of one label token per option, read at the last prompt token
    COMMON_DECISION_TYPE_LEV,     // same as openjev, noul is read from a rating scale
    COMMON_DECISION_TYPE_NIMBLE,  // same as openjev, the prompt lists all the questions of the request
    COMMON_DECISION_TYPE_UNKNOWN, // a decision model of a type that is not supported
};

enum server_decision_question_type {
    SERVER_DECISION_QUESTION_CHOICE,
    SERVER_DECISION_QUESTION_SCORE,
    SERVER_DECISION_QUESTION_NOUL,
};

struct server_decision_option {
    std::string key;
    json description; // null if not provided
};

struct server_decision_question {
    std::string id;
    server_decision_question_type type;
    json instructions;
    std::vector<server_decision_option> options; // in the order of the model outputs
};

struct server_decision_context {
    common_decision_type type = COMMON_DECISION_TYPE_NONE;

    // read the "<arch>.decision.*" metadata, type stays NONE if the model has none
    void init(const llama_model * model);

    // throw std::invalid_argument on bad input
    std::vector<server_decision_question> parse_questions(const json & body) const;

    // number of images in the request, throw std::invalid_argument on bad input
    // images come from "images" and from the image_url parts of a state made of chat messages
    size_t count_images(const json & body) const;

    // number of prompts that are evaluated to answer this question, each one shows the options in a different order
    size_t n_variants(const server_decision_question & question) const;

    // set the prompt of one variant of this question, and where to read its result
    void fill_task(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant,
            server_task & task) const;

    // scores: the raw model outputs of each variant
    json format_answer(const server_decision_question & question, const std::vector<std::vector<float>> & scores) const;

private:
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<const common_chat_template> tmpl; // the "systemone" template

    std::map<std::string, float> temperatures; // "<type>" or "<type>.<n_options bucket>"
    size_t n_options_max   = 0;
    bool   noul_true_first = false; // noul options are [true, false] instead of [false, true]

    // OPENJEV, LEV, NIMBLE
    std::vector<llama_token> labels;
    std::vector<std::string> label_texts; // only if the label of an option is given to the template

    std::string render(
            const json & state,
            const std::vector<server_decision_question> & questions,
            const server_decision_question & question,
            size_t variant) const;
    json render_options(const server_decision_question & question, size_t variant) const;
    size_t n_outputs(const server_decision_question & question) const;

    float get_temperature(const server_decision_question & question) const;
};
