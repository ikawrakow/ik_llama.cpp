#include "server-common.h"

#include <algorithm>

using raw_buffer = std::vector<uint8_t>;


server_grammar_trigger::server_grammar_trigger(const json& in) {
    value.type = (common_grammar_trigger_type)in.at("type").get<int>();
    value.value = in.at("value").get<std::string>();
    if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        value.token = (llama_token)in.at("token").get<int>();
    }
}

json server_grammar_trigger::to_json() const {
    json out{
        {"type", (int)value.type},
        {"value", value.value},
    };
    if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
        out["token"] = (int)value.token;
    }
    return out;
}


void server_log(const char* level, const char* function, int line, const char* message, const json& extra) {
    std::stringstream ss_tid;
    ss_tid << std::this_thread::get_id();
    json log = json{
        {"tid",       ss_tid.str()},
        {"timestamp", time(nullptr)},
    };

    if (server_log_json) {
        log.merge_patch({
            {"level",    level},
            {"function", function},
            {"line",     line},
            {"msg",      message},
            });

        if (!extra.empty()) {
            log.merge_patch(extra);
        }

        printf("%s\n", log.dump(-1, ' ', false, json::error_handler_t::replace).c_str());
    }
    else {
        char buf[1024];
        snprintf(buf, 1024, "%4s [%24s] %s", level, function, message);

        if (!extra.empty()) {
            log.merge_patch(extra);
        }
        std::stringstream ss;
        ss << buf << " |";
        for (const auto& el : log.items())
        {
            const std::string value = el.value().dump(-1, ' ', false, json::error_handler_t::replace);
            ss << " " << el.key() << "=" << value;
        }

        const std::string str = ss.str();
        printf("%.*s\n", (int)str.size(), str.data());
    }
    fflush(stdout);
}

//
// chat template utils
//


bool is_base64(uint8_t c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::vector<uint8_t> base64_decode(const std::string& encoded_string) {
    int i = 0;
    int j = 0;
    int in_ = 0;

    int in_len = encoded_string.size();

    uint8_t char_array_4[4];
    uint8_t char_array_3[3];

    std::vector<uint8_t> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_]; in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }

            char_array_3[0] = ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }

            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }

        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }

        char_array_3[0] = ((char_array_4[0]) << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; j < i - 1; j++) {
            ret.push_back(char_array_3[j]);
        }
    }

    return ret;
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    std::stringstream chatcmplid;
    chatcmplid << "chatcmpl-" << random_string();

    return chatcmplid.str();
}

std::string gen_tool_call_id() {
    return random_string();
}

//
// other common utils
//
float get_slot_similarity(size_t lcp, size_t prompt_length, size_t cache_length) {
    float sim = float(lcp) * 2 / (prompt_length + cache_length);
    return sim;
}

size_t common_part(const std::vector<llama_token>& a, const std::vector<llama_token>& b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

size_t common_part(const std::string& a, const std::string& b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
}

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Check the last few bytes to see if a multi-byte character is cut off
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = text[len - i];
        // Check for start of a multi-byte sequence from the end
        if ((c & 0xE0) == 0xC0) {
            // 2-byte character start: 110xxxxx
            // Needs at least 2 bytes
            if (i < 2) return len - i;
        }
        else if ((c & 0xF0) == 0xE0) {
            // 3-byte character start: 1110xxxx
            // Needs at least 3 bytes
            if (i < 3) return len - i;
        }
        else if ((c & 0xF8) == 0xF0) {
            // 4-byte character start: 11110xxx
            // Needs at least 4 bytes
            if (i < 4) return len - i;
        }
    }

    // If no cut-off multi-byte character is found, return full length
    return len;
}

// TODO: reuse common_token_to_piece
template <class Iter>
static std::string tokens_to_str(llama_context* ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

std::string tokens_to_str(llama_context* ctx, const llama_tokens& tokens) {
    return tokens_to_str(ctx, tokens.begin(), tokens.end());
}

// format incomplete utf-8 multibyte character for output
 std::string tokens_to_output_formatted_string(const llama_context* ctx, const llama_token token) {
    std::string out = token == -1 ? "" : common_token_to_piece(ctx, token);

    // if the size is 1 and first bit is 1, meaning it's a partial character
    //   (size > 1 meaning it's already a known token)
    if (out.size() == 1 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "byte: \\x" + res;
    }

    return out;
}


common_prefix common_prefix_add(const common_prefix& a, const common_prefix& b) {
    common_prefix prefix;
    prefix.first = a.first + b.first;
    prefix.second = a.second + b.second;
    return prefix;
}

common_prefix find_common_string_prefix(const std::string& a_str, const std::string& b_str, const std::set<char>& ignore_set) {
    size_t i = 0;
    size_t j = 0;
    while (i < a_str.size() && j < b_str.size()) {
        auto a_chr = a_str[i];
        auto b_chr = b_str[j];
        if (a_chr == b_chr) {
            ++i;
            ++j;
        }
        else if (ignore_set.count(a_chr) && ignore_set.count(b_chr)) {
            ++i;
            ++j;
        }
        else if (ignore_set.count(a_chr)) {
            ++i;
        }
        else if (ignore_set.count(b_chr)) {
            ++j;
        }
        else {
            break;
        }
    }
    common_prefix string_prefix;
    string_prefix.first = i;
    string_prefix.second = j;
    return string_prefix;
}

size_t find_n_tokens_from_string(const llama_context* ctx, const llama_tokens& a, const size_t max_size, size_t start,
    std::vector<size_t>& map) {
    size_t n = 0;
    size_t string_len = 0;
    std::string str;
    auto model = llama_get_model(ctx);
    for (n = start; n < a.size(); ++n) {
        str = llama_token_to_piece(model, a[n], true);
        string_len = string_len + str.size();
        if (string_len <= max_size) {
            map.push_back(string_len);
        }
        else {
            break;
        }
    }
    return map.size();
}

std::string remove_with_set(std::string str, const std::set<char>& chars_to_remove) {
    str.erase(std::remove_if(str.begin(), str.end(),
        [&chars_to_remove](char c) { return chars_to_remove.find(c) != chars_to_remove.end(); }),
        str.end());
    return str;
}

common_prefix find_largest_common_number(const std::vector<size_t>& a_list, const std::vector<size_t>& b_list) {
    common_prefix token_prefix;
    token_prefix.first = 0;
    token_prefix.second = 0;
    int i = a_list.size() - 1; // start from end of a
    int j = b_list.size() - 1; // start from end of b
    if (i < 0 || j < 0) {
        return token_prefix;
    }
    while (i >= 0 && j >= 0) {
        if (a_list[i] == b_list[j]) {
            // found largest common value
            token_prefix.first = (size_t)i + 1;
            token_prefix.second = (size_t)j + 1;
            break;
        }
        else if (a_list[i] > b_list[j]) {
            --i;
        }
        else {
            --j;
        }
    }
    return token_prefix;
}

size_t find_n_tokens_from_string_with_ignore(const llama_context* ctx, const llama_tokens& a, const size_t max_size, size_t start, const std::set<char>& ignore_set,
    std::vector<size_t>& map) {
    bool use_ignore = ignore_set.size() > 0;
    size_t n = 0;
    size_t string_len = 0;
    size_t string_len_ignore = 0;
    std::string str;
    std::string str_ignore;
    auto model = llama_get_model(ctx);
    for (n = start; n < a.size(); ++n) {
        str = llama_token_to_piece(model, a[n], true);
        string_len = string_len + str.size();
        if (use_ignore) {
            str_ignore = remove_with_set(str, ignore_set);
        }
        else {
            str_ignore = str;
        }
        string_len_ignore = string_len_ignore + str_ignore.size();
        if (string_len <= max_size) {
            map.push_back(string_len_ignore);
        }
        else {
            break;
        }
    }
    return map.size();
}

common_prefix find_common_text_token_prefix(const llama_context* ctx, const llama_tokens& a, const llama_tokens& b,
    size_t start, bool exact) {
    common_prefix token_prefix;
    if (a.size() <= start || b.size() <= start) {
        return token_prefix;
    }
    std::set<char> ignore_set = { ' ', '\n' ,'\r' };

    llama_tokens a_sub(a.begin() + start, a.end());
    llama_tokens b_sub(b.begin() + start, b.end());

    std::string a_str = common_detokenize(ctx, a_sub, true);
    std::string b_str = common_detokenize(ctx, b_sub, true);
    common_prefix string_prefix;

    std::vector<size_t> a_list;
    std::vector<size_t> b_list;

    if (exact) {
        size_t lcp = common_part(a_str, b_str);
        string_prefix.first = lcp;
        string_prefix.second = lcp;
        token_prefix.first = find_n_tokens_from_string(ctx, a_sub, string_prefix.first, 0, a_list);
        token_prefix.second = find_n_tokens_from_string(ctx, b_sub, string_prefix.second, 0, b_list);
    }
    else {
        string_prefix = find_common_string_prefix(a_str, b_str, ignore_set);
        token_prefix.first = find_n_tokens_from_string_with_ignore(ctx, a_sub, string_prefix.first, 0, ignore_set, a_list);
        token_prefix.second = find_n_tokens_from_string_with_ignore(ctx, b_sub, string_prefix.second, 0, ignore_set, b_list);
    }

    token_prefix = find_largest_common_number(a_list, b_list);
    return token_prefix;
}



json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto& p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json{
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            });
    }
    return probs_for_token;
}

 float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string& str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}


json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output>& probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto& p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json{
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
            });
    }
    return out;
}


// convert a vector of completion_token_output to json
json probs_vector_to_json(const llama_context* ctx, const std::vector<completion_token_output>& probs) {
    json out = json::array();

    for (const auto& prob : probs) {
        json probs_for_token = json::array();

        for (const auto& p : prob.probs) {
            const std::string tok_str = tokens_to_output_formatted_string(ctx, p.tok);
            probs_for_token.push_back(json{
                {"tok_str", tok_str},
                {"prob",    p.prob},
                });
        }

        const std::string tok_str = tokens_to_output_formatted_string(ctx, prob.tok);
        out.push_back(json{
            {"content", tok_str},
            {"probs",   probs_for_token},
            });
    }

    return out;
}


// note: if data is a json array, it will be sent as multiple events, one per item
bool server_sent_event(httplib::DataSink& sink, const json& data) {
    static auto send_single = [](httplib::DataSink& sink, const json& data) -> bool {
        const std::string str =
            "data: " +
            data.dump(-1, ' ', false, json::error_handler_t::replace) +
            "\n\n"; // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).

        LOG_DBG("data stream, to_send: %s", str.c_str());
        return sink.write(str.c_str(), str.size());
    };

    if (data.is_array()) {
        for (const auto& item : data) {
            if (!send_single(sink, item)) {
                return false;
            }
        }
    }
    else {
        return send_single(sink, data);
    }

    return true;
}

bool server_sent_oai_resp_event(httplib::DataSink& sink, const json& data) {
    static auto send_single = [](httplib::DataSink& sink, const json& data) -> bool {
        const std::string str =
            "event: " + data.at("event").get<std::string>() + "\n" +
            "data: " + data.at("data").dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";

        LOG_DBG("data stream, to_send: %s", str.c_str());
        return sink.write(str.c_str(), str.size());
    };

    if (data.is_array()) {
        for (const auto& item : data) {
            if (!send_single(sink, item)) {
                return false;
            }
        }
    }
    else {
        return send_single(sink, data);
    }

    return true;
}

bool server_sent_anthropic_event(httplib::DataSink& sink, const json& data) {
    static auto send_single = [](httplib::DataSink& sink, const json& data) -> bool {
        const std::string str =
            (data.contains("event") && data.contains("data")) ?
            ("event: " + data.at("event").get<std::string>() + "\n" +
                "data: " + data.at("data").dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n") :
            ("data: " + data.at("data").dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");

        LOG_DBG("data stream, to_send: %s", str.c_str());
        return sink.write(str.c_str(), str.size());
    };

    if (data.is_array()) {
        for (const auto& item : data) {
            if (!send_single(sink, item)) {
                return false;
            }
        }
    }
    else {
        return send_single(sink, data);
    }

    return true;
}

//
// OAI utils
//
// used by /completions endpoint
json oaicompat_chat_params_parse(const json& body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({ body.at("stop").get<std::string>() });
    }
    else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Handle "logprobs" field
    int n_probs = json_value(body, "logprobs", 0);
    if (n_probs > 0) {
        llama_params["n_probs"] = n_probs;
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params{ "best_of", "suffix" };
    for (const auto& param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto& item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}


// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    const struct llama_model* model,
    json& body, /* openai api json semantics */
    const oaicompat_parser_options& opt,
    std::vector<raw_buffer>& out_files)
{
    json llama_params;

    llama_params["__oaicompat"] = true;
    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }
    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({ body.at("stop").get<std::string>() });
    }
    else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            json_schema = json_value(response_format, "schema", json::object());
        }
        else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        }
        else if (!response_type.empty() && response_type != "text") {
            json_schema = json_value(json_schema, "schema", json::object());
        }
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    json& messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::runtime_error("Expected 'messages' to be an array");
    }
    for (auto& msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::runtime_error("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::runtime_error("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json& content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::runtime_error("Expected 'content' to be a string or an array");
        }

        for (auto& p : content) {
            std::string type = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json image_url = json_value(p, "image_url", json::object());
                std::string url = json_value(image_url, "url", std::string());
                if (string_starts_with(url, "http")) {
                    // download remote image
                    // TODO @ngxson : maybe make these params configurable
                    common_remote_params params;
                    params.headers.push_back("User-Agent: ik_llama.cpp/");
                    params.max_size = 1024 * 1024 * 10; // 10MB
                    params.timeout = 10; // seconds
                    LOG_INFO("downloading image from '%s'\n", url.c_str());
                    auto res = common_remote_get_content(url, params);
                    if (200 <= res.first && res.first < 300) {
                        LOG_INFO("downloaded %ld bytes\n", res.second.size());
                        raw_buffer data;
                        data.insert(data.end(), res.second.begin(), res.second.end());
                        out_files.push_back(data);
                    }
                    else {
                        throw std::runtime_error("Failed to download image");
                    }

                }
                else {
                    // try to decode base64 image
                    std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
                    if (parts.size() != 2) {
                        throw std::runtime_error("Invalid image_url.url value");
                    }
                    else if (!string_starts_with(parts[0], "data:image/")) {
                        throw std::runtime_error("Invalid image_url.url format: " + parts[0]);
                    }
                    else if (!string_ends_with(parts[0], "base64")) {
                        throw std::runtime_error("image_url.url must be base64 encoded");
                    }
                    else {
                        auto base64_data = parts[1];
                        auto decoded_data = base64_decode(base64_data);
                        out_files.push_back(decoded_data);
                    }
                }

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("image_url");

            }
            else if (type == "input_audio") {
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json input_audio = json_value(p, "input_audio", json::object());
                std::string data = json_value(input_audio, "data", std::string());
                std::string format = json_value(input_audio, "format", std::string());
                // while we also support flac, we don't allow it here so we matches the OAI spec
                if (format != "wav" && format != "mp3") {
                    throw std::runtime_error("input_audio.format must be either 'wav' or 'mp3'");
                }
                auto decoded_data = base64_decode(data); // expected to be base64 encoded
                out_files.push_back(decoded_data);

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("input_audio");

            }
            else if (type != "text") {
                throw std::runtime_error("unsupported content[].type");
            }
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar = grammar;
    inputs.use_jinja = opt.use_jinja;
    inputs.parallel_tool_calls = json_value(body, "parallel_tool_calls", false);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format = opt.reasoning_format;
    inputs.enable_thinking = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::runtime_error("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto& item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    }
    else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    }
    else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::runtime_error("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    /*"whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\n"
        "when this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n"*/
    bool prefill_assistant_message = !inputs.messages.empty() && inputs.messages.back().role == "assistant" && opt.prefill_assistant;
    common_chat_msg last_message;
    if (prefill_assistant_message) {
        last_message = inputs.messages.back();
        inputs.messages.pop_back();

        /* sanity check, max one assistant message at the end of the list */
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            throw std::runtime_error("Cannot have 2 or more assistant messages at the end of the list.");
        }

        /* TODO: test this properly */
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;
        if (inputs.enable_thinking) {
            throw std::runtime_error("Assistant response prefill is incompatible with enable_thinking.");
        }
        inputs.add_generation_prompt = true;
    }

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls, inputs);

    /* Append assistant prefilled message */
    if (prefill_assistant_message) {
        if (!last_message.content_parts.empty()) {
            for (auto& p : last_message.content_parts) {
                chat_params.prompt += p.text;
            }
        }
        else {
            chat_params.prompt += last_message.content;
        }
    }

    llama_params["chat_format"] = static_cast<int>(chat_params.format);
    llama_params["prompt"] = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"] = chat_params.grammar;
    }
    llama_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers = json::array();
    for (const auto& trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"] = grammar_triggers;
    llama_params["preserved_tokens"] = chat_params.preserved_tokens;
    llama_params["thinking_forced_open"] = chat_params.thinking_forced_open;
    for (const auto& stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::runtime_error("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    }
    else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::runtime_error("top_logprobs requires logprobs to be set to true");
    }


    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", "tfs_z",... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto& item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

json convert_responses_to_chatcmpl(const json& response_body) {
    if (!response_body.contains("input")) {
        throw std::runtime_error("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::runtime_error("ik_llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    }
    else if (input_value.is_array()) {
        static auto exists_and_is_array = [](const json& j, const char* key) -> bool {
            return j.contains(key) && j.at(key).is_array();
        };
        static auto exists_and_is_string = [](const json& j, const char* key) -> bool {
            return j.contains(key) && j.at(key).is_string();
        };

        for (json item : input_value) {
            if (exists_and_is_string(item, "content")) {
                item["content"] = json::array({
                    json{
                        {"text", item.at("content")},
                        {"type", "input_text"},
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" || item.at("role") == "system" || item.at("role") == "developer")
            ) {
                std::vector<json> chatcmpl_content;

                for (const json& input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::runtime_error("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    }
                    else if (type == "input_image") {
                        if (!input_item.contains("image_url")) {
                            throw std::runtime_error("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json{
                                {"url", input_item.at("image_url")},
                            }},
                            {"type", "image_url"},
                        });
                    }
                    else if (type == "input_file") {
                        throw std::runtime_error("'input_file' is not supported by ik_llama.cpp at this moment");
                    }
                    else {
                        throw std::runtime_error("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            }
            else if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                std::vector<json> chatcmpl_content;

                for (const auto& output_text : item.at("content")) {
                    const std::string type = json_value(output_text, "type", std::string());
                    if (type != "output_text") {
                        throw std::runtime_error("'type' must be 'output_text'");
                    }
                    if (!exists_and_is_string(output_text, "text")) {
                        throw std::runtime_error("'Output text' requires 'text'");
                    }
                    chatcmpl_content.push_back({
                        {"text", output_text.at("text")},
                        {"type", "text"},
                    });
                }

                item.erase("status");
                item.erase("type");
                item["content"] = chatcmpl_content;
                chatcmpl_messages.push_back(item);
            }
            else if (exists_and_is_string(item, "arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                json msg = json{
                    {"role", "assistant"},
                    {"tool_calls", json::array({json{
                        {"function", json{
                            {"arguments", item.at("arguments")},
                            {"name",      item.at("name")},
                        }},
                        {"id",   item.at("call_id")},
                        {"type", "function"},
                    }})},
                };

                if (!chatcmpl_messages.empty() && chatcmpl_messages.back().contains("reasoning_content")) {
                    msg["reasoning_content"] = chatcmpl_messages.back().at("reasoning_content");
                    chatcmpl_messages.pop_back();
                }
                chatcmpl_messages.push_back(msg);
            }
            else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json{
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
                else {
                    json chatcmpl_outputs = item.at("output");
                    for (json& chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type") || chatcmpl_output.at("type") != "input_text") {
                            throw std::runtime_error("Output of tool call should be 'Input text'");
                        }
                        chatcmpl_output["type"] = "text";
                    }
                    chatcmpl_messages.push_back(json{
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            }
            else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                if (!exists_and_is_array(item, "content")) {
                    throw std::runtime_error("item['content'] is not an array");
                }
                if (item.at("content").empty()) {
                    throw std::runtime_error("item['content'] is empty");
                }
                if (!exists_and_is_string(item.at("content")[0], "text")) {
                    throw std::runtime_error("item['content']['text'] is not a string");
                }

                chatcmpl_messages.push_back(json{
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"reasoning_content", item.at("content")[0].at("text")},
                });
            }
            else {
                throw std::runtime_error("Cannot determine type of 'item'");
            }
        }
    }
    else {
        throw std::runtime_error("'input' must be a string or array of objects");
    }

    chatcmpl_messages.erase(std::remove_if(
        chatcmpl_messages.begin(),
        chatcmpl_messages.end(),
        [](const json& x) {
            return x.contains("role") &&
                x.at("role") == "assistant" &&
                x.contains("content") &&
                x.at("content") == json::array() &&
                x.contains("reasoning_content");
        }),
        chatcmpl_messages.end());

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        if (!response_body.at("tools").is_array()) {
            throw std::runtime_error("'tools' must be an array of objects");
        }
        std::vector<json> chatcmpl_tools;
        for (json resp_tool : response_body.at("tools")) {
            json chatcmpl_tool;

            if (json_value(resp_tool, "type", std::string()) != "function") {
                throw std::runtime_error("'type' of tool must be 'function'");
            }
            resp_tool.erase("type");
            chatcmpl_tool["type"] = "function";

            if (!resp_tool.contains("strict")) {
                resp_tool["strict"] = true;
            }
            chatcmpl_tool["function"] = resp_tool;
            chatcmpl_tools.push_back(chatcmpl_tool);
        }
        chatcmpl_body.erase("tools");
        chatcmpl_body["tools"] = chatcmpl_tools;
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    return chatcmpl_body;
}

json anthropic_params_from_json(
    const struct llama_model* model,
    const json& body_in, /* anthropic messages api json semantics */
    const oaicompat_parser_options& opt,
    std::vector<raw_buffer>& out_files)
{
    json body = body_in;
    json llama_params;

    if (body.contains("stop_sequences")) {
        llama_params["stop"] = body.at("stop_sequences");
    }
    else {
        llama_params["stop"] = json::array();
    }

    // handle max_tokens (required in Anthropic, but we're permissive)
    if (!body.contains("max_tokens")) {
        llama_params["n_predict"] = 4096;
    }
    else {
        llama_params["n_predict"] = body.at("max_tokens");
    }

    if (body.contains("top_k")) {
        llama_params["top_k"] = body.at("top_k");
    }

    if (body.contains("thinking")) {
        json thinking = json_value(body, "thinking", json::object());
        std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            llama_params["thinking_budget_tokens"] = budget_tokens;
        }
    }

    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            llama_params["__metadata_user_id"] = user_id;
        }
    }

    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
        }
        else if (system_param.is_array()) {
            for (const auto& block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    system_content += json_value(block, "text", std::string());
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
            });
    }

    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    json& messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::runtime_error("Expected 'messages' to be an array");
    }

    for (auto& msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::runtime_error("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content")) {
                continue;
            }
        }

        json& content = msg.at("content");

        if (content.is_string()) {
            oai_messages.push_back(msg);
            continue;
        }

        if (!content.is_array()) {
            throw std::runtime_error("Expected 'content' to be a string or an array");
        }

        json tool_calls = json::array();
        json converted_content = json::array();
        json tool_results = json::array();
        bool has_tool_calls = false;

        for (auto& block : content) {
            std::string type = json_value(block, "type", std::string());

            if (type == "text") {
                converted_content.push_back(block);
            }
            else if (type == "image") {
                json source = json_value(block, "source", json::object());
                std::string source_type = json_value(source, "type", std::string());

                if (source_type == "base64") {
                    std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                    std::string data = json_value(source, "data", std::string());

                    converted_content.push_back({
                        {"type", "image_url"},
                        {"image_url", {
                            {"url", "data:" + media_type + ";base64," + data}
                        }}
                        });
                }
                else if (source_type == "url") {
                    std::string url = json_value(source, "url", std::string());
                    converted_content.push_back({
                        {"type", "image_url"},
                        {"image_url", {
                            {"url", url}
                        }}
                        });
                }
            }
            else if (type == "tool_use") {
                tool_calls.push_back({
                    {"id", json_value(block, "id", std::string())},
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(block, "name", std::string())},
                        {"arguments", json_value(block, "input", json::object()).dump()}
                    }}
                    });
                has_tool_calls = true;
            }
            else if (type == "tool_result") {
                std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                auto result_content = json_value(block, "content", json());
                std::string result_text;
                if (result_content.is_string()) {
                    result_text = result_content.get<std::string>();
                }
                else if (result_content.is_array()) {
                    for (const auto& c : result_content) {
                        if (json_value(c, "type", std::string()) == "text") {
                            result_text += json_value(c, "text", std::string());
                        }
                    }
                }

                tool_results.push_back({
                    {"role", "tool"},
                    {"tool_call_id", tool_use_id},
                    {"content", result_text}
                    });
            }
        }

        if (!tool_results.empty()) {
            if (!converted_content.empty() || has_tool_calls) {
                json new_msg = { {"role", role} };
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                }
                else if (has_tool_calls) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                oai_messages.push_back(new_msg);
            }
            for (const auto& tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
        else {
            if (!converted_content.empty() || has_tool_calls) {
                json new_msg = { {"role", role} };
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                }
                else if (has_tool_calls) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                oai_messages.push_back(new_msg);
            }
        }
    }

    json oai_tools = json::array();
    if (body.contains("tools")) {
        json& tools = body.at("tools");
        if (tools.is_array()) {
            for (auto& tool : tools) {
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                    });
            }
        }
    }

    std::string oai_tool_choice = "auto";
    if (body.contains("tool_choice")) {
        json& tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_tool_choice = "auto";
            }
            else if (type == "any") {
                oai_tool_choice = "required";
            }
            else if (type == "tool") {
                oai_tool_choice = "required";
            }
        }
    }

    for (auto& msg : oai_messages) {
        if (!msg.contains("content")) {
            continue;
        }
        json& content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }
        if (!content.is_array()) {
            continue;
        }

        for (auto& p : content) {
            std::string type = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json image_url = json_value(p, "image_url", json::object());
                std::string url = json_value(image_url, "url", std::string());
                if (string_starts_with(url, "http")) {
                    // download remote image
                    common_remote_params params;
                    params.headers.push_back("User-Agent: ik_llama.cpp/");
                    params.max_size = 1024 * 1024 * 10; // 10MB
                    params.timeout = 10; // seconds
                    LOG_INFO("downloading image from '%s'\n", url.c_str());
                    auto res = common_remote_get_content(url, params);
                    if (200 <= res.first && res.first < 300) {
                        LOG_INFO("downloaded %ld bytes\n", res.second.size());
                        raw_buffer data;
                        data.insert(data.end(), res.second.begin(), res.second.end());
                        out_files.push_back(data);
                    }
                    else {
                        throw std::runtime_error("Failed to download image");
                    }
                }
                else {
                    // try to decode base64 image
                    std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
                    if (parts.size() != 2) {
                        throw std::runtime_error("Invalid image_url.url value");
                    }
                    else if (!string_starts_with(parts[0], "data:image/")) {
                        throw std::runtime_error("Invalid image_url.url format: " + parts[0]);
                    }
                    else if (!string_ends_with(parts[0], "base64")) {
                        throw std::runtime_error("image_url.url must be base64 encoded");
                    }
                    else {
                        auto base64_data = parts[1];
                        auto decoded_data = base64_decode(base64_data);
                        out_files.push_back(decoded_data);
                    }
                }

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("image_url");
            }
            else if (type == "input_audio") {
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json input_audio = json_value(p, "input_audio", json::object());
                std::string data = json_value(input_audio, "data", std::string());
                std::string format = json_value(input_audio, "format", std::string());
                if (format != "wav" && format != "mp3") {
                    throw std::runtime_error("input_audio.format must be either 'wav' or 'mp3'");
                }
                auto decoded_data = base64_decode(data);
                out_files.push_back(decoded_data);

                // replace this chunk with a marker
                p["type"] = "text";
                p["text"] = mtmd_default_marker();
                p.erase("input_audio");
            }
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(oai_messages);
    inputs.tools = common_chat_tools_parse_oaicompat(oai_tools);
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(oai_tool_choice);
    inputs.json_schema = "";
    inputs.grammar = "";
    inputs.use_jinja = opt.use_jinja;
    inputs.parallel_tool_calls = json_value(body, "parallel_tool_calls", false);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format = opt.reasoning_format;
    inputs.enable_thinking = opt.enable_thinking;

    if (opt.enable_thinking && opt.prefill_assistant) {
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            inputs.enable_thinking = false;
        }
    }

    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto& item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    }
    else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    }
    else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::runtime_error("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // if the assistant message appears at the end of list, we do not add end-of-turn token
    bool prefill_assistant_message = !inputs.messages.empty() && inputs.messages.back().role == "assistant" && opt.prefill_assistant;
    common_chat_msg last_message;
    if (prefill_assistant_message) {
        last_message = inputs.messages.back();
        inputs.messages.pop_back();

        // sanity check, max one assistant message at the end of the list
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            throw std::runtime_error("Cannot have 2 or more assistant messages at the end of the list.");
        }

        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

        if (inputs.enable_thinking) {
            throw std::runtime_error("Assistant response prefill is incompatible with enable_thinking.");
        }

        inputs.add_generation_prompt = true;
    }

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls, inputs);

    // Append assistant prefilled message
    if (prefill_assistant_message) {
        if (!last_message.content_parts.empty()) {
            for (auto& p : last_message.content_parts) {
                chat_params.prompt += p.text;
            }
        }
        else {
            chat_params.prompt += last_message.content;
        }
    }

    llama_params["chat_format"] = static_cast<int>(chat_params.format);
    llama_params["prompt"] = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"] = chat_params.grammar;
    }
    llama_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers = json::array();
    for (const auto& trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"] = grammar_triggers;
    llama_params["preserved_tokens"] = chat_params.preserved_tokens;
    llama_params["thinking_forced_open"] = chat_params.thinking_forced_open;
    for (const auto& stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }

    // Handle "n" field
    int n_choices = json_value(body, "n", 1);
    if (n_choices != 1) {
        throw std::runtime_error("Only one completion choice is allowed");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via Anthropic endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto& item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json& data) {
    if (data.is_array()) {
        for (const auto& e : data) {
            if (!e.is_number_integer()) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json& data) {
    bool seen_string = false;
    bool seen_number = false;
    if (data.is_array()) {
        for (const auto& e : data) {
            seen_string |= e.is_string();
            seen_number |= e.is_number_integer();
            if (seen_number && seen_string) {
                return true;
            }
        }
    }
    return false;
}

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json& data) {
    if (data.is_array()) {
        for (const auto& e : data) {
            if (e.is_number_integer()) {
                return true;
            }
        }
        return false;
    }
    return false;
}

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string>& paths, const json& js) {
    json result = json::object();

    for (const std::string& path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string& k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            }
            else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}


/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
std::vector<llama_token> tokenize_mixed(const llama_vocab* vocab, const json& json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    std::vector<llama_token> prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto& p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                std::vector<llama_token> p;
                if (first) {
                    p = llama_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                }
                else {
                    p = llama_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            }
            else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    }
    else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = llama_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

json format_tokenizer_response(const std::vector<llama_token>& tokens) {
    return json{
        {"tokens", tokens}
    };
}

json format_detokenized_response(const std::string& content) {
    return json{
        {"content", content}
    };
}

json format_error_response(const std::string& message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
    case ERROR_TYPE_INVALID_REQUEST:
        type_str = "invalid_request_error";
        code = 400;
        break;
    case ERROR_TYPE_AUTHENTICATION:
        type_str = "authentication_error";
        code = 401;
        break;
    case ERROR_TYPE_NOT_FOUND:
        type_str = "not_found_error";
        code = 404;
        break;
    case ERROR_TYPE_SERVER:
        type_str = "server_error";
        code = 500;
        break;
    case ERROR_TYPE_PERMISSION:
        type_str = "permission_error";
        code = 403;
        break;
    case ERROR_TYPE_NOT_SUPPORTED:
        type_str = "not_supported_error";
        code = 501;
        break;
    case ERROR_TYPE_UNAVAILABLE:
        type_str = "unavailable_error";
        code = 503;
        break;
    }
    return json{
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}


token_probabilities get_token_probabilities(llama_context* ctx, int idx, llama_token sampled_token_id, int n_sorted) {
    const auto* logits = llama_get_logits_ith(ctx, idx);
    const int n_vocab = llama_n_vocab(llama_get_model(ctx));
    n_sorted = std::min(n_sorted, n_vocab);

    std::vector<std::pair<float, llama_token>> sorted(n_vocab);
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) sorted[token_id] = { logits[token_id], token_id };

    std::partial_sort(sorted.begin(), sorted.begin() + n_sorted, sorted.end(), std::greater<std::pair<float, llama_token>>{});

    float max_l = sorted.front().first;
    float cum_sum = 0.0f;
    float sampled_token_p = 0.0f;
    bool sampled_token_found = false;
    std::vector<llama_token_data> cur(n_sorted);
    for (int i = 0; i < n_vocab; ++i) {
        float p = expf(sorted[i].first - max_l);
        cum_sum += p;
        if (i < n_sorted) {
            cur[i] = { sorted[i].second, sorted[i].first, p };
        }
        if (!sampled_token_found && sorted[i].second == sampled_token_id) {
            sampled_token_p = p;
            sampled_token_found = true;
        }
    }
    for (int i = n_sorted; i < n_vocab; ++i) cum_sum += expf(sorted[i].first - max_l);

    float inv_cum_sum = 1 / cum_sum;
    for (int i = 0; i < n_sorted; ++i) cur[i].p *= inv_cum_sum;
    sampled_token_p *= inv_cum_sum;

    return { sampled_token_p, cur };
}

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */

server_tokens::server_tokens(mtmd::input_chunks& mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
        for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
            push_back(mtmd_chunks[i]);
        }
    }

server_tokens::server_tokens(const llama_tokens& tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {
    }

    llama_pos server_tokens::pos_next() const {
        if (!has_mtmd) {
            return tokens.size();
        }

        llama_pos res = tokens.size();

        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
            const auto& chunk = it->second;
            res += mtmd_input_chunk_get_n_pos(chunk.get()) - mtmd_input_chunk_get_n_tokens(chunk.get());
        }

        return res;
    }

    // for debugging
    std::string server_tokens::str() const {
        std::ostringstream oss;
        oss << "tokens: ";
        for (size_t idx = 0; idx < tokens.size(); ++idx) {
            llama_token t = tokens[idx];
            oss << "idx:" << idx << " ";
            if (t == LLAMA_TOKEN_NULL) {
                oss << "<embd> ";
            }
            else {
                oss << t << " ";
            }
        }
        oss << "\n";
        oss << "image idx: ";
        for (const auto& it : map_idx_to_media) {
            oss << it.first << ", ";
        }
        return oss.str();
    }

    const mtmd::input_chunk_ptr& server_tokens::find_chunk(size_t idx) const {
        auto it = map_idx_to_media.find(idx);
        if (it != map_idx_to_media.end()) {
            return it->second;
        }
        throw std::runtime_error("Chunk not found");
    }

    void server_tokens::push_back(llama_token tok) {
        if (tok == LLAMA_TOKEN_NULL) {
            throw std::runtime_error("Invalid token");
        }
        tokens.emplace_back(tok);
    }

    // will create a copy of the chunk if it contains non-text data
    void server_tokens::push_back(const mtmd_input_chunk* chunk) {
        auto type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            GGML_ASSERT(has_mtmd);
            const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
            size_t start_idx = tokens.size();
            for (size_t i = 0; i < n_tokens; ++i) {
                tokens.emplace_back(LLAMA_TOKEN_NULL);
            }
            mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
            map_idx_to_media[start_idx] = std::move(new_chunk);
        }
        else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens;
            const auto* text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            for (size_t i = 0; i < n_tokens; ++i) {
                push_back(text_tokens[i]);
            }
        }
        else {
            GGML_ABORT("Invalid chunk type");
        }
    }

    // appends server tokens, updates the media map. copies media chunks.
    void server_tokens::push_back(server_tokens& tokens) {
        size_t start_idx = size();
        for (size_t i = 0; i < tokens.size(); i++) {
            push_back(tokens[i]);
        }
        if (tokens.has_mtmd) {
            // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
            // We could also just check, but this will prevent silently dropping MTMD data.
            GGML_ASSERT(has_mtmd);
            for (auto it = tokens.map_idx_to_media.begin(); it != tokens.map_idx_to_media.end(); ) {
                auto* chunk = tokens.map_idx_to_media[it->first].get();
                mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
                map_idx_to_media[start_idx + it->first] = std::move(new_chunk);
            }
        }
    }

    // for compatibility with context shift and prompt truncation
    void server_tokens::insert(const std::vector<llama_token>& inp_tokens) {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
    }

    // for compatibility with context shift and prompt truncation
    void server_tokens::resize(size_t size) {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        tokens.resize(size);
    }

    llama_token* server_tokens::data() {
        return tokens.data();
    }

    llama_tokens::iterator server_tokens::begin() {
        return tokens.begin();
    }

    llama_tokens::iterator server_tokens::end() {
        return tokens.end();
    }

    llama_tokens::const_iterator server_tokens::cbegin() {
        return tokens.cbegin();
    }

    llama_tokens::const_iterator server_tokens::cend() {
        return tokens.cend();
    }

    llama_tokens server_tokens::tokens_data() {
        return tokens;
    }

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const std::vector<llama_token>& server_tokens::get_text_tokens() const {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        return tokens;
    }

    // for compatibility with speculative decoding
    void server_tokens::set_token(llama_pos pos, llama_token id) {
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        tokens[pos] = id;
    }

    size_t server_tokens::size() const {
        return tokens.size();
    }

    bool server_tokens::empty() const {
        return tokens.empty();
    }

    void server_tokens::clear() {
        tokens.clear();
    }

    void server_tokens::keep_first(size_t n) {
        GGML_ASSERT(n <= tokens.size());
        if (has_mtmd) {
            if (n == tokens.size()) {
                return; // nothing to do
            }
            // we throw an error if we try to remove a token in the middle of an image
            // for ex. with input of 5 text tokens and 2 images:
            //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
            // n  1   2   3   4   5   6      7      8      9      10
            // allowed to resize      ^                    ^
            // disallowed to resize          ^      ^             ^
            if (n > 0) {
                llama_token last_token = tokens[n - 1];
                // make sure we never remove tokens in the middle of an image
                if (last_token == LLAMA_TOKEN_NULL) {
                    find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
                }
            }
            // remove all image chunks that are not used anymore
            for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ) {
                size_t idx = it->first;
                if (idx >= n) {
                    it = map_idx_to_media.erase(it);
                }
                else {
                    ++it;
                }
            }
        }
        tokens.resize(n);
    }

    std::string server_tokens::detokenize(const llama_context* ctx, bool special) const {
        llama_tokens text_tokens;
        text_tokens.reserve(tokens.size());
        for (const auto& t : tokens) {
            if (t != LLAMA_TOKEN_NULL) {
                text_tokens.push_back(t);
            }
        }
        return common_detokenize(ctx, text_tokens, special);
    }

    std::string server_tokens::detokenize(const llama_context* ctx, bool special, size_t start, size_t length) const {
        std::string str;
        if (tokens.size() <= start || length == 0) {
            return str;
        }
        llama_tokens text_tokens;
        text_tokens.reserve(tokens.size() - start);
        size_t i = 0;
        size_t count = 0;
        for (const auto& t : tokens) {
            if (t != LLAMA_TOKEN_NULL && i >= start) {
                text_tokens.push_back(t);
                ++count;
                if (count >= length) {
                    break;
                }
            }
            ++i;
        }
        return common_detokenize(ctx, text_tokens, special);
    }

    size_t server_tokens::find_n_from_tokens(const llama_context* ctx, const server_tokens& b, bool special,
        size_t start, const size_t length) {
        std::string str = detokenize(ctx, special, start, length);
        std::vector<size_t> tmp;
        size_t n = find_n_tokens_from_string(ctx, b.tokens, start, length, tmp);
        return n;
    }

    size_t server_tokens::get_common_prefix_exact(const server_tokens& b) const {
        const size_t max_idx = std::min(tokens.size(), b.tokens.size());

        if (!has_mtmd) {
            for (size_t i = 0; i < max_idx; ++i) {
                if (tokens[i] == b.tokens[i]) {
                    continue;
                }
                return i;
            }
            return max_idx;
        }

        for (size_t i = 0; i < max_idx; ++i) {
            const llama_token ai = tokens[i];
            const llama_token bi = b.tokens[i];

            if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
                const auto& a_chunk = find_chunk(i);
                const auto& b_chunk = b.find_chunk(i);

                GGML_ASSERT(a_chunk && b_chunk);

                const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
                const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

                const size_t n_tok_a = mtmd_input_chunk_get_n_tokens(a_chunk.get());
                const size_t n_tok_b = mtmd_input_chunk_get_n_tokens(b_chunk.get());

                if (id_ai == id_bi && n_tok_a == n_tok_b) {
                    GGML_ASSERT(n_tok_a > 0 && "Invalid media chunk"); // should never happen
                    i += n_tok_a - 1; // will be +1 by the for loop
                    continue;
                }

                return i;
            }

            if (ai == bi) {
                continue;
            }

            return i;
        }

        return max_idx; // all tokens are equal
    }

    llama_tokens server_tokens::get_text_tokens_exclude_think(const llama_context* ctx, const thinking_tokens& think_token) const {
        if (!think_token.exclude) {
            return get_text_tokens();
        }
        GGML_ASSERT((think_token.begin != "" && think_token.end != "") && "think tokens cannot be empty");
        std::string startStr = think_token.begin;
        std::string endStr = think_token.end;

        llama_tokens tokens = get_text_tokens();
        std::string str = common_detokenize(ctx, tokens, true);

        std::vector<std::pair<size_t, size_t>> results;
        // Find all positions of start and end
        std::vector<size_t> startPositions;
        std::vector<size_t> endPositions;

        size_t pos = 0;
        // Find all start positions
        while ((pos = str.find(startStr, pos)) != std::string::npos) {
            startPositions.push_back(pos);
            pos += startStr.length();
        }

        pos = 0;
        // Find all end positions
        while ((pos = str.find(endStr, pos)) != std::string::npos) {
            endPositions.push_back(pos + endStr.length());
            pos += endStr.length();
        }

        // For each start position, pair with all end positions that come after it
        for (size_t i = 0; i < startPositions.size(); i++) {
            for (size_t j = 0; j < endPositions.size(); j++) {
                if (results.size()) {
                    // start must be after last end
                    if (startPositions[i] > results[results.size() - 1].second && endPositions[j] > startPositions[i]) {
                        results.push_back({ startPositions[i], endPositions[j] });
                        break;
                    }
                }
                else {
                    if (endPositions[j] > startPositions[i]) {
                        results.push_back({ startPositions[i], endPositions[j] });
                        break;
                    }
                }

            }
        }
        if (!results.size()) {
            return tokens;
        }

        // Exclude tokens
        pos = 0;
        size_t n = 0;
        size_t string_len = 0;
        llama_tokens tokens_new;
        auto model = llama_get_model(ctx);
        for (n = 0; n < tokens.size(); ++n) {
            str = llama_token_to_piece(model, tokens[n], true);
            string_len = string_len + str.size();
            if (string_len <= results[pos].first) {
                tokens_new.push_back(tokens[n]);
            }
            else if (string_len <= results[pos].second) {
                continue;
            }
            else {
                tokens_new.push_back(tokens[n]);
                if (pos+1 < results.size()) {
                    pos++;
                }
            }            
        }
        return tokens_new;
    }


    common_prefix server_tokens::get_common_prefix(const llama_context* ctx, const server_tokens& b, bool exact) const {
        common_prefix token_prefix;

        size_t n = get_common_prefix_exact(b); // strict token match as a starting point
        token_prefix.first = n;
        token_prefix.second = n;

        if (!has_mtmd) {
            token_prefix = find_common_text_token_prefix(ctx, this->tokens, b.tokens, n, exact);
            token_prefix.first += n;
            token_prefix.second += n;
            return token_prefix;
        }
        size_t i = n;
        size_t j = n;
        llama_tokens a_list;
        llama_tokens b_list;
        while (i < size() && j < b.size()) {
            llama_token ai = tokens[i];
            llama_token bi = b.tokens[j];
            if (ai != LLAMA_TOKEN_NULL) {
                a_list.push_back(ai);
                ++i;
            }
            if (bi != LLAMA_TOKEN_NULL) {
                b_list.push_back(bi);
                ++j;
            }
            if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
                common_prefix prefix = find_common_text_token_prefix(ctx, a_list, b_list, 0, exact);
                // text match or empty
                if (prefix.first == a_list.size() && prefix.second == b_list.size()) {
                    a_list.clear();
                    b_list.clear();
                    const auto& a_chunk = find_chunk(i);
                    const auto& b_chunk = b.find_chunk(j);

                    GGML_ASSERT(a_chunk && b_chunk);

                    const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
                    const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

                    const size_t n_tok_a = mtmd_input_chunk_get_n_tokens(a_chunk.get());
                    const size_t n_tok_b = mtmd_input_chunk_get_n_tokens(b_chunk.get());

                    // image match
                    if (id_ai == id_bi && n_tok_a == n_tok_b) {
                        GGML_ASSERT(n_tok_a > 0 && "Invalid media chunk"); // should never happen
                        i += n_tok_a;
                        j += n_tok_a;
                        prefix.first += n_tok_a;
                        prefix.second += n_tok_a;
                        token_prefix = common_prefix_add(prefix, token_prefix);
                    }
                    else {
                        // do no include image token prefix
                        // only return text token prefix
                        token_prefix = common_prefix_add(prefix, token_prefix);
                        return token_prefix;
                    }
                }
                else {
                    // text not match
                    token_prefix = common_prefix_add(prefix, token_prefix);
                    return token_prefix;
                }
            }
        }
        common_prefix prefix = find_common_text_token_prefix(ctx, a_list, b_list, 0, exact);
        token_prefix = common_prefix_add(prefix, token_prefix);

        return token_prefix;

    }

    // take first n tokens of tokens list a
    // find the common prefix between a and b
    common_prefix server_tokens::get_common_prefix_first_n(const llama_context* ctx, const server_tokens& b, size_t n, bool exact) const {
        // not work for mtmd
        GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
        auto tokens = get_text_tokens();
        if (n > tokens.size()) {
            n = tokens.size();
        }
        llama_tokens copy(tokens.begin(), tokens.begin() + n);
        server_tokens a = server_tokens(copy, false);
        return a.get_common_prefix(ctx, b, exact);
    }

    // make sure all text tokens are within the vocab range
    bool server_tokens::validate(const struct llama_context* ctx) const {
        const llama_model* model = llama_get_model(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        for (size_t i = 0; i < tokens.size(); ++i) {
            auto& t = tokens[i];
            if (t == LLAMA_TOKEN_NULL) {
                try {
                    const auto& chunk = find_chunk(i);
                    size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk.get());
                    i += n_tokens - 1; // will be +1 by the for loop
                }
                catch (const std::exception& e) {
                    return false;
                }
            }
            else if (t < 0 || t >= n_vocab) {
                return false;
            }
        }
        return true;
    }

    // encode and decode the image chunk
    int32_t server_tokens::process_chunk(
        llama_context* ctx,
        mtmd_context* mctx,
        size_t idx,
        llama_pos pos,
        int32_t seq_id,
        size_t& n_tokens_out) const {
        const auto& chunk = find_chunk(idx);
        const char* name = mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE
            ? "image" : "audio";
        LLAMA_LOG_INFO("processing %s...\n", name);
        int32_t n_batch = llama_n_batch(ctx);
        int64_t t0 = ggml_time_ms();
        llama_pos new_n_past; // unused for now
        int32_t result = mtmd_helper_eval_chunk_single(mctx, ctx,
            chunk.get(),
            pos,
            seq_id,
            n_batch,
            true, // logits last
            &new_n_past);
        LLAMA_LOG_INFO("%s processed in %" PRId64 " ms\n", name, ggml_time_ms() - t0);
        if (result != 0) {
            LLAMA_LOG_ERROR("mtmd_helper_eval failed with status %d", result);
            n_tokens_out = 0;
            return result;
        }
        n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
        return 0;
    }

    // Keep the first n_keep and remove n_discard tokens from tokens
    void server_tokens::discard_n_tokens(int32_t n_keep, int32_t n_discard) {
        if (n_discard <= 0 || n_keep + n_discard >= size()) {
            return;
        }

        llama_tokens new_tokens = get_text_tokens(); // copy
        for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
            new_tokens[i - n_discard] = new_tokens[i];
        }
        int32_t token_size = (int32_t)size();
        new_tokens.resize(token_size - n_discard);
        clear();
        insert(new_tokens);

    }

    // Similarity between prompt and cached
    float server_tokens::get_tokens_similarity(const llama_context* ctx, const server_tokens& tokens, int n_keep, int n_discard) const {
        GGML_ASSERT(n_keep >= 0 && n_discard >= 0);
        float sim_cur = 0;
        if (n_keep == 0 && n_discard == 0) {
            auto lcp_len = get_common_prefix(ctx, tokens);
            sim_cur = get_slot_similarity(lcp_len.second, tokens.size(), size());
        }
        else {
            // remove tokens due to context shift and compare
            auto tokens_ctx_shift = server_tokens(tokens.get_text_tokens(), false); // copy cache tokens
            tokens_ctx_shift.discard_n_tokens(n_keep, n_discard);
            auto lcp_len = get_common_prefix(ctx, tokens_ctx_shift);
            sim_cur = get_slot_similarity(lcp_len.second, tokens_ctx_shift.size(), size());
        }
        return sim_cur;
    }

    // Similarity between common part and cache
    float server_tokens::get_cached_tokens_similarity(const llama_context* ctx, const server_tokens& tokens, int n_keep, int n_discard) const {
        GGML_ASSERT(n_keep >= 0 && n_discard >= 0);
        float sim_cur = 0;
        if (n_keep == 0 && n_discard == 0) {
            auto lcp_len = get_common_prefix(ctx, tokens);
            sim_cur = (float)lcp_len.first / size();
        }
        else {
            // remove tokens due to context shift and compare
            auto tokens_ctx_shift = server_tokens(tokens.get_text_tokens(), false); // copy cache tokens
            tokens_ctx_shift.discard_n_tokens(n_keep, n_discard);
            auto lcp_len = get_common_prefix(ctx, tokens_ctx_shift);
            sim_cur = (float)lcp_len.first / size();
        }
        return sim_cur;
    }


// Computes FNV-1a hash of the data
std::string fnv_hash(const uint8_t* data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return std::to_string(hash);
}

server_tokens process_mtmd_prompt(mtmd_context* mctx, std::string prompt, std::vector<raw_buffer> files) {
    mtmd::bitmaps bitmaps;
    for (auto& file : files) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size()));
        if (!bmp.ptr) {
            throw std::runtime_error("Failed to load image or audio file");
        }
        // calculate bitmap hash (for KV caching)
        std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
        bmp.set_id(hash.c_str());
        bitmaps.entries.push_back(std::move(bmp));
    }
    // process prompt
    std::vector<server_tokens> inputs;
    // multimodal
    mtmd_input_text inp_txt = {
        prompt.c_str(),
        /* add_special */   true,
        /* parse_special */ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tokenized = mtmd_tokenize(mctx,
        chunks.ptr.get(),
        &inp_txt,
        bitmaps_c_ptr.data(),
        bitmaps_c_ptr.size());
    if (tokenized != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    auto result = server_tokens(chunks, true);
    return result;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
server_tokens tokenize_input_subprompt(const llama_vocab* vocab, mtmd_context* mctx, const json& json_prompt, bool add_special, bool parse_special) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    const bool has_mtmd = mctx != nullptr;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        std::vector<llama_token> tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp, false);
    }
    else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        std::vector<llama_token> tmp = json_prompt.get<std::vector<llama_token>>();
        return server_tokens(tmp, false);
    }
    else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            if (!has_mtmd)
                throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

            // JSON object with prompt and multimodal key.
            std::vector<raw_buffer> files;
            for (const auto& entry : json_prompt.at(JSON_MTMD_DATA_KEY)) {
                files.push_back(base64_decode(entry));
            }
            return process_mtmd_prompt(mctx, json_prompt.at(JSON_STRING_PROMPT_KEY), files);
        }
        else {
            // Not multimodal, but contains a subobject.
            std::vector<llama_token> tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
            return server_tokens(tmp, false);
        }
    }
    else {
        throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
    }
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(const llama_vocab* vocab, mtmd_context* mctx, const json& json_prompt, bool add_special, bool parse_special) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto& p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, mctx, p, add_special, parse_special));
        }
    }
    else {
        result.push_back(tokenize_input_subprompt(vocab, mctx, json_prompt, add_special, parse_special));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}
// Assuming raw_buffer has .data() and .size() members
void print_files_info(const std::vector<raw_buffer>& files) {
    for (size_t i = 0; i < files.size(); ++i) {
        const auto& file = files[i];
        std::cout << "File " << i << ": Size = " << file.size() << " bytes\n";

        // Print first 16 bytes in hex
        std::cout << "First 16 bytes: ";
        for (size_t j = 0; j < std::min<size_t>(file.size(), 16); ++j) {
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(file.data()[j]) << " ";
        }
        std::cout << std::dec << "\n\n"; // Reset to decimal
    }
}

bool prompt_cache_equal(llama_context* ctx, const server_tokens& cache_tokens,
    const server_tokens& prompt_tokens, size_t start, const common_prefix& prefix) {
    std::string common_cache = cache_tokens.detokenize(ctx, true, start, prefix.first);
    std::string common_prompt = prompt_tokens.detokenize(ctx, true, start, prefix.second);
    bool equal = common_cache == common_prompt;
    return equal;
}

std::string safe_json_to_str(const json& data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}
