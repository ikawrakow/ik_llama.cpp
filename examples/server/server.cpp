#pragma warning(disable : 4996)
#include "server-context.h"
#include "server-common.h"
#include "chat.h"

#include "common.h"
#include "speculative.h"
#include "mtmd.h"
#include "sampling.h"
#include "llama.h"
#include "llama-vocab.h"

#ifndef NDEBUG
// crash the server in debug mode, otherwise send an http 500 error
#define CPPHTTPLIB_NO_EXCEPTIONS 1
#endif

#include <nlohmann/json.hpp>
#include "index.html.gz.hpp"
#include "index_llamacpp.html.gz.hpp"
#include "loading.html.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <set>
#include <mutex>
#include <thread>
#include <signal.h>
#include <memory>
#include <random>
#include <algorithm>
#include <src/llama-impl.h>
#ifdef SQLITE3_MODERN_CPP_SUPPORT
#include <sqlite_modern_cpp.h>
#include <deque>

struct DatabaseHandle {
    sqlite::database db;

    DatabaseHandle(const std::string& path) : db(path) {
        db << "CREATE TABLE IF NOT EXISTS sessions (key TEXT PRIMARY KEY, data TEXT)";
        db << "CREATE TABLE IF NOT EXISTS templates (key TEXT PRIMARY KEY, data TEXT)";
        db << "CREATE TABLE IF NOT EXISTS names (key TEXT PRIMARY KEY, data TEXT)";
    }
};
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

bool server_verbose = false;
bool server_log_json = true;


enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
    SERVER_STATE_ERROR           // An error occurred, load_model failed
};


static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
    case STOP_TYPE_EOS:   return "eos";
    case STOP_TYPE_WORD:  return "word";
    case STOP_TYPE_LIMIT: return "limit";
    default:              return "none";
    }
}


inline std::string get_model_name(std::string path)
{
    std::string filename = path.substr(path.find_last_of("/\\") + 1);
    return filename;
};


static json format_final_response_oaicompat(const json& request, json result, const std::string& completion_id, bool streaming = false) {
    bool stopped_word = result.count("stopped_word") != 0;
    bool stopped_eos = json_value(result, "stopped_eos", false);
    int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
    int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason = "length";
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }

    json choices =
        streaming ? json::array({ json{{"finish_reason", finish_reason},
                                        {"index", 0},
                                        {"delta", json::object()}} })
        : json::array({ json{{"finish_reason", finish_reason},
                              {"index", 0},
                              {"message", json{{"content", content},
                                               {"role", "assistant"}}}} });

    std::time_t t = std::time(0);

    json res = json{
        {"choices", choices},
        {"created", t},
        {"model",
            json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", streaming ? "chat.completion.chunk" : "chat.completion"},
        {"usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        }},
        {"id", completion_id}
    };

    if (server_verbose) {
        res["__verbose"] = result;
    }

    if (result.contains("completion_probabilities")) {
        res["completion_probabilities"] = json_value(result, "completion_probabilities", json::array());
    }

    return res;
}

// return value is vector as there is one case where we might need to generate two responses
static std::vector<json> format_partial_response_oaicompat(server_task_result task_result, const std::string& completion_id) {
    json result = task_result.data;
    std::cout << result.dump(4) << std::endl;
    if (!result.contains("model") || !result.contains("oaicompat_token_ctr")) {
        return std::vector<json>({ result });
    }

    bool first = json_value(result, "oaicompat_token_ctr", 0) == 0;
    std::string modelname = json_value(result, "model", std::string(DEFAULT_OAICOMPAT_MODEL));

    bool stopped_word = json_value(result, "stopped_word", false);
    bool stopped_eos = json_value(result, "stopped_eos", false);
    bool stopped_limit = json_value(result, "stopped_limit", false);
    std::string content = json_value(result, "content", std::string(""));

    std::string finish_reason;
    if (stopped_word || stopped_eos) {
        finish_reason = "stop";
    }
    if (stopped_limit) {
        finish_reason = "length";
    }

    std::time_t t = std::time(0);

    json choices;

    if (!finish_reason.empty()) {
        choices = json::array({ json{{"finish_reason", finish_reason},
                                    {"index", 0},
                                    {"delta", json::object()}} });
    }
    else {
        if (first) {
            if (content.empty()) {
                choices = json::array({ json{{"finish_reason", nullptr},
                                            {"index", 0},
                                            {"delta", json{{"role", "assistant"}}}} });
            }
            else {
                // We have to send this as two updates to conform to openai behavior
                json initial_ret = json{ {"choices", json::array({json{
                                        {"finish_reason", nullptr},
                                        {"index", 0},
                                        {"delta", json{
                                            {"role", "assistant"}
                                        }}}})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"} };

                json second_ret = json{
                            {"choices", json::array({json{{"finish_reason", nullptr},
                                                            {"index", 0},
                                                            {"delta", json{
                                                            {"content", content}}}
                                                            }})},
                            {"created", t},
                            {"id", completion_id},
                            {"model", modelname},
                            {"object", "chat.completion.chunk"} };

                return std::vector<json>({ initial_ret, second_ret });
            }
        }
        else {
            // Some idiosyncrasy in task processing logic makes several trailing calls
            // with empty content, we ignore these at the calee site.
            if (content.empty()) {
                return std::vector<json>({ json::object() });
            }

            choices = json::array({ json{
                {"finish_reason", nullptr},
                {"index", 0},
                {"delta",
                json{
                    {"content", content},
                }},
            } });
        }
    }

    json ret = json{
        {"choices", choices},
        {"created", t},
        {"id",      completion_id},
        {"model",   modelname},
        {"object",  "chat.completion.chunk"}
    };

    if (task_result.timings.prompt_n != -1) {
        ret.push_back({ "timings", task_result.timings.to_json() });
    }

    //
    if (!finish_reason.empty()) {
        int num_tokens_predicted = json_value(result, "tokens_predicted", 0);
        int num_prompt_tokens = json_value(result, "tokens_evaluated", 0);
        ret.push_back({ "usage", json {
            {"completion_tokens", num_tokens_predicted},
            {"prompt_tokens",     num_prompt_tokens},
            {"total_tokens",      num_tokens_predicted + num_prompt_tokens}
        } });
    }

    return std::vector<json>({ ret });
}


static json format_embeddings_response_oaicompat(const json& request, const json& embeddings, bool use_base64 = false) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto& elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            embedding_obj = {
                {"embedding", base64::encode(data_ptr, data_size)},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        }
        else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);
        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }
    json res = json{
        {"model", json_value(request, "model", std::string(DEFAULT_OAICOMPAT_MODEL))},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health" || req.path == "/v1/completions") {
        return;
    }

    LOG_INFO("request", {
        {"remote_addr", req.remote_addr},
        {"remote_port", req.remote_port},
        {"status",      res.status},
        {"method",      req.method},
        {"path",        req.path},
        {"params",      req.params},
    });

    LOG_VERBOSE("request", {
        {"request",  req.body},
        {"response", res.body},
    });
}

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

int main(int argc, char ** argv) {
#if SERVER_VERBOSE != 1
    log_disable();
#endif
    // own arguments required by this example
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // parse arguments from environment variables
    gpt_params_parse_from_env(params);

    // TODO: not great to use extern vars
    server_log_json = params.log_json;
    server_verbose = params.verbosity > 0;


    // struct that contains llama context and inference
    server_context ctx_server;

    if (!params.system_prompt.empty()) {
        ctx_server.system_prompt_set(params.system_prompt);
    }

    if (params.model_alias == "unknown") {
        params.model_alias = params.model;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INFO("build info", {
        {"build",  LLAMA_BUILD_NUMBER},
        {"commit", LLAMA_COMMIT}
    });

    LOG_INFO("system info", {
        {"n_threads",       params.n_threads},
        {"n_threads_batch", params.n_threads_batch},
        {"total_threads",   std::thread::hardware_concurrency()},
        {"system_info",     llama_print_system_info()},
    });

    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INFO("Running with SSL", {{"key", params.ssl_file_key}, {"cert", params.ssl_file_cert}});
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INFO("Running without SSL", {});
        svr.reset(new httplib::Server());
    }
#else
    svr.reset(new httplib::Server());
#endif

    std::atomic<server_state> state{SERVER_STATE_LOADING_MODEL};

    svr->set_default_headers({{"Server", "ik_llama.cpp"}});

    svr->set_logger(log_server_request);

    auto res_error = [](httplib::Response & res, json error_data) {
        json final_response {{"error", error_data}};
        res.set_content(final_response.dump(), "application/json; charset=utf-8");
        res.status = json_value(error_data, "code", 500);
    };

    auto res_ok = [](httplib::Response& res, const json& data) {
        res.set_content(data.dump(), "application/json; charset=utf-8");
        res.status = 200;
    };

    svr->set_exception_handler([&res_error](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message;
        try {
            std::rethrow_exception(std::move(ep));
        } catch (std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
        LOG_VERBOSE("Got exception", formatted_error);
        res_error(res, formatted_error);
    });

    svr->set_error_handler([&res_error](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    if (!svr->bind_to_port(params.hostname, params.port)) {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n", params.hostname.c_str(), params.port);
        return 1;
    }

    std::unordered_map<std::string, std::string> log_data;

    log_data["hostname"] = params.hostname;
    log_data["port"]     = std::to_string(params.port);

    if (params.api_keys.size() == 1) {
        auto key = params.api_keys[0];
        log_data["api_key"] = "api_key: ****" + key.substr(std::max((int)(key.length() - 4), 0));
    } else if (params.api_keys.size() > 1) {
        log_data["api_key"] = "api_key: " + std::to_string(params.api_keys.size()) + " keys loaded";
    }

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;
    ctx_server.cache_ram_n_min = params.cache_ram_n_min;
    ctx_server.cache_ram_similarity = params.cache_ram_similarity;
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    auto db_handle = std::make_shared<DatabaseHandle>(params.sql_save_file);
    bool sqlite_extension_loaded = false;
    if (!params.sqlite_zstd_ext_file.empty()) {
        auto* conn = db_handle->db.connection().get();
        sqlite3_enable_load_extension(conn, 1);
        char* errmsg = nullptr;
        const int rc = sqlite3_load_extension(
            conn,
            params.sqlite_zstd_ext_file.c_str(),
            nullptr,
            &errmsg
        );
        if(rc != SQLITE_OK) {
            const std::string err = errmsg ? errmsg : "Unknown extension error";
            sqlite3_free(errmsg);
            LOG_WARNING("Failed to load extension", {{"err", err}});
        }
	else {
            sqlite_extension_loaded = true;
        }
        sqlite3_enable_load_extension(conn, 0);
    }
#else
    auto db_handle = false;
#endif
    // load the model
    if (!ctx_server.load_model(params)) {
        state.store(SERVER_STATE_ERROR);
        return 1;
    } else {
        ctx_server.init();
        state.store(SERVER_STATE_READY);
    }

    LOG_INFO("model loaded", {});

    const auto model_meta = ctx_server.model_meta();

    // print sample chat example to make it clear which template is used

        LOG_INFO("chat template", {
        {"chat_template", common_chat_templates_source(ctx_server.chat_templates.get())},
    });

    LOG_INFO("chat template", {
        {"chat_example", common_chat_format_example(ctx_server.chat_templates.get(), ctx_server.params.use_jinja, {}).c_str()
        },
            {"built_in",     params.chat_template.empty()},
        });
    //
    // Middlewares
    //

    auto middleware_validate_api_key = [&params, &res_error](const httplib::Request & req, httplib::Response & res) {
        static const std::unordered_set<std::string> public_endpoints = {
            "/health",
            "/v1/health",
            "/models",
            "/v1/models",
            "/api/tags"
        };

        // If API key is not set, skip validation
        if (params.api_keys.empty()) {
            return true;
        }

        // If path is public or is static file, skip validation
        if (public_endpoints.find(req.path) != public_endpoints.end() || req.path == "/") {
            return true;
        }

        // Check for API key in the header
        auto auth_header = req.get_header_value("Authorization");

        std::string prefix = "Bearer ";
        if (auth_header.substr(0, prefix.size()) == prefix) {
            std::string received_api_key = auth_header.substr(prefix.size());
            if (std::find(params.api_keys.begin(), params.api_keys.end(), received_api_key) != params.api_keys.end()) {
                return true; // API key is valid
            }
        }

        auth_header = req.get_header_value("X-Api-Key");

        if (std::find(params.api_keys.begin(), params.api_keys.end(), auth_header) != params.api_keys.end()) {
            return true; // API key is valid
        }

        // API key is invalid or not provided
        res.status = 401;
        res.set_content(
            (json {
                {"error", {
                    {"message", "Invalid API Key"},
                    {"type", "authentication_error"},
                    {"code", 401}
                }}
            }).dump(-1, ' ', false, json::error_handler_t::replace),
            "application/json; charset=utf-8"
        );
        LOG_WARNING("Unauthorized: Invalid API Key\n", {});
        return false;
    };

    auto middleware_server_state = [&res_error, &state](const httplib::Request& req, httplib::Response& res) {
        server_state current_state = state.load();
        if (current_state == SERVER_STATE_LOADING_MODEL) {
            auto tmp = string_split<std::string>(req.path, '.');
            if (req.path == "/" || tmp.back() == "html") {
                res.set_content(reinterpret_cast<const char*>(loading_html), loading_html_len, "text/html; charset=utf-8");
                res.status = 503;
            }
            else if (req.path == "/models" || req.path == "/v1/models" || req.path == "/api/tags") {
                // allow the models endpoint to be accessed during loading
                return true;
            }
            else {
                res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
            }
            return false;
        }
        return true;
    };

    // register server middlewares
    svr->set_pre_routing_handler([&middleware_validate_api_key, &middleware_server_state](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        // If this is OPTIONS request, skip validation because browsers don't include Authorization header
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/html"); // blank response, no data
            return httplib::Server::HandlerResponse::Handled; // skip further processing
        }
        if (!middleware_server_state(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!middleware_validate_api_key(req, res)) {
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
        });

    //
    // Route handlers (or controllers)
    //

    const auto handle_health = [&](const httplib::Request & req, httplib::Response & res) {
        server_state current_state = state.load();
        switch (current_state) {
            case SERVER_STATE_READY:
                {
                    // request slots data using task queue
                    server_task task;
                    task.id   = ctx_server.queue_tasks.get_new_id();
                    task.type = SERVER_TASK_TYPE_METRICS;
                    task.id_target = -1;

                    ctx_server.queue_results.add_waiting_task_id(task.id);
                    ctx_server.queue_tasks.post(std::move(task));

                    // get the result
                    server_task_result result = ctx_server.queue_results.recv(task.id);
                    ctx_server.queue_results.remove_waiting_task_id(task.id);

                    const int n_idle_slots       = result.data.at("idle");
                    const int n_processing_slots = result.data.at("processing");

                    json health = {
                        {"status",           "ok"},
                        {"slots_idle",       n_idle_slots},
                        {"slots_processing", n_processing_slots}
                    };

                    res.status = 200; // HTTP OK
                    if (params.endpoint_slots && req.has_param("include_slots")) {
                        health["slots"] = result.data.at("slots");
                    }

                    if (n_idle_slots == 0) {
                        health["status"] = "no slot available";
                        if (req.has_param("fail_on_no_slot")) {
                            res.status = 503; // HTTP Service Unavailable
                        }
                    }

                    res.set_content(health.dump(), "application/json");
                    break;
                }
            case SERVER_STATE_LOADING_MODEL:
                {
                    res_error(res, format_error_response("Loading model", ERROR_TYPE_UNAVAILABLE));
                } break;
            case SERVER_STATE_ERROR:
                {
                    res_error(res, format_error_response("Model failed to load", ERROR_TYPE_SERVER));
                } break;
        }
    };

    const auto handle_slots = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_slots) {
            res_error(res, format_error_response("This server does not support slots endpoint.", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_multi  = -1;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(std::move(task));

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        res.set_content(result.data.at("slots").dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto handle_metrics = [&](const httplib::Request &, httplib::Response & res) {
        if (!params.endpoint_metrics) {
            res_error(res, format_error_response("This server does not support metrics endpoint.", ERROR_TYPE_NOT_SUPPORTED));
            return;
        }

        // request slots data using task queue
        server_task task;
        task.id = ctx_server.queue_tasks.get_new_id();
        task.id_multi  = -1;
        task.id_target = -1;
        task.type = SERVER_TASK_TYPE_METRICS;
        task.data.push_back({{"reset_bucket", true}});

        ctx_server.queue_results.add_waiting_task_id(task.id);
        ctx_server.queue_tasks.post(std::move(task));

        // get the result
        server_task_result result = ctx_server.queue_results.recv(task.id);
        ctx_server.queue_results.remove_waiting_task_id(task.id);

        json data = result.data;

        const uint64_t n_prompt_tokens_processed = data.at("n_prompt_tokens_processed");
        const uint64_t t_prompt_processing       = data.at("t_prompt_processing");

        const uint64_t n_tokens_predicted  = data.at("n_tokens_predicted");
        const uint64_t t_tokens_generation = data.at("t_tokens_generation");

        const int32_t kv_cache_used_cells = data.at("kv_cache_used_cells");

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) data.at("n_prompt_tokens_processed_total")}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) data.at("t_prompt_processing_total") / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) data.at("n_tokens_predicted_total")}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) data.at("t_tokens_generation_total") / 1.e3}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  n_prompt_tokens_processed ? 1.e3 / t_prompt_processing * n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  n_tokens_predicted ? 1.e3 / t_tokens_generation * n_tokens_predicted : 0.}
            },{
                    {"name",  "kv_cache_usage_ratio"},
                    {"help",  "KV-cache usage. 1 means 100 percent usage."},
                    {"value",  1. * kv_cache_used_cells / params.n_ctx}
            },{
                    {"name",  "kv_cache_tokens"},
                    {"help",  "KV-cache tokens."},
                    {"value",  (uint64_t) data.at("kv_cache_tokens_count")}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of request processing."},
                    {"value",  (uint64_t) data.at("processing")}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of request deferred."},
                    {"value",  (uint64_t) data.at("deferred")}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        const int64_t t_start = data.at("t_start");
        res.set_header("Process-Start-Time-Unix", std::to_string(t_start));

        res.set_content(prometheus.str(), "text/plain; version=0.0.4");
        res.status = 200; // HTTP OK
    };

    const auto handle_slots_save = [&ctx_server, &res_error, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_SAVE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath }
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_restore = [&ctx_server, &res_error, &params](const httplib::Request & req, httplib::Response & res, int id_slot) {
        json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res_error(res, format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        std::string filepath = params.slot_save_path + filename;

        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
        task.data = {
            { "id_slot", id_slot },
            { "filename", filename },
            { "filepath", filepath }
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_erase = [&ctx_server, &res_error](const httplib::Request & /* req */, httplib::Response & res, int id_slot) {
        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_ERASE;
        task.data = {
            { "id_slot", id_slot },
        };

        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        if (result.error) {
            res_error(res, result.data);
        } else {
            res.set_content(result.data.dump(), "application/json");
        }
    };

    const auto handle_slots_action = [&res_error, &handle_slots_save, &handle_slots_restore, &handle_slots_erase](const httplib::Request & req, httplib::Response & res) {
        std::string id_slot_str = req.path_params.at("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res_error(res, format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        std::string action = req.get_param_value("action");

        if (action == "save") {
            handle_slots_save(req, res, id_slot);
        } else if (action == "restore") {
            handle_slots_restore(req, res, id_slot);
        } else if (action == "erase") {
            handle_slots_erase(req, res, id_slot);
        } else {
            res_error(res, format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        }
    };

    const auto handle_props = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        std::string template_key = "tokenizer.chat_template", curr_tmpl;
        int32_t tlen = llama_model_meta_val_str(ctx_server.model, template_key.c_str(), nullptr, 0);
        if (tlen > 0) {
            std::vector<char> curr_tmpl_buf(tlen + 1, 0);
            if (llama_model_meta_val_str(ctx_server.model, template_key.c_str(), curr_tmpl_buf.data(), curr_tmpl_buf.size()) == tlen) {
                curr_tmpl = std::string(curr_tmpl_buf.data(), tlen);
            }
        }
        json data = {
            { "system_prompt",               ctx_server.system_prompt.c_str() },
            { "model_alias",                 ctx_server.params.model_alias },
            { "model_path",                  ctx_server.params.model},
            { "default_generation_settings", ctx_server.default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params.n_parallel },
            { "model_name",                  get_model_name(ctx_server.params.model)},
            { "chat_template",               common_chat_templates_source(ctx_server.chat_templates.get()) },
            { "bos_token",                   llama_token_to_piece(ctx_server.ctx, llama_token_bos(ctx_server.model), /* special= */ true)},
            { "eos_token",                   llama_token_to_piece(ctx_server.ctx, llama_token_eos(ctx_server.model), /* special= */ true)},
            { "model_path",                  ctx_server.params.model },
            { "modalities",                  json {
                {"vision", ctx_server.oai_parser_opt.allow_image},
                {"audio",  ctx_server.oai_parser_opt.allow_audio},
            } },
            { "n_ctx",                       ctx_server.n_ctx }

        };

        if (ctx_server.params.use_jinja) {
            if (auto tool_use_src = common_chat_templates_source(ctx_server.chat_templates.get(), "tool_use")) {
                data["chat_template_tool_use"] = tool_use_src;
            }
        }
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_props_simple = [&ctx_server](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        int n_past = 0;
        int slot_id = 0;
        for (server_slot& slot : ctx_server.slots) {
            if (slot.n_past > n_past) {
                n_past = slot.n_past;
                slot_id = slot.id;
            }
        }
        json data = {
            { "model_name",                  get_model_name(ctx_server.params.model)},
            { "model_path",                  ctx_server.params.model },
            { "modalities",                  json {
                {"vision", ctx_server.oai_parser_opt.allow_image},
                {"audio",  ctx_server.oai_parser_opt.allow_audio},
            } },
             { "n_ctx",                       ctx_server.n_ctx }
        };
        res.set_content(data.dump(), "application/json; charset=utf-8");
    };


    // handle completion-like requests (completion, chat, infill)
    // we can optionally provide a custom format for partial results and final results
    const auto handle_completions_impl = [&ctx_server, &params, &res_error, &res_ok](
        server_task_type type,
        json& data,
        const std::vector<raw_buffer>& files,
        httplib::Response& res,
        oaicompat_type oaicompat) -> void {
            GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION);
            if (ctx_server.params.embedding) {
                res_error(res, format_error_response("This server does not support completions. Start it without `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
                return;
            }

            const auto& prompt = data.at("prompt");

            // process prompt
            std::vector<server_tokens> inputs;

            if (oaicompat && ctx_server.mctx != nullptr) {
                // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
#ifndef NDEBUG
                print_files_info(files);
#endif // !NDEBUG
                inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
            }
            else {
                // Everything else, including multimodal completions.
                inputs = tokenize_input_prompts(llama_get_vocab(ctx_server.ctx), ctx_server.mctx, prompt, true, true);
            }
            const auto completion_id = gen_chatcmplid();
            const int id_task = ctx_server.queue_tasks.get_new_id();

            ctx_server.queue_results.add_waiting_task_id(id_task);
            ctx_server.request_completion(id_task, -1, data, false, false, std::move(inputs[0]));
            bool stream = json_value(data, "stream", false);
            if (!stream) {
                server_task_result result = ctx_server.queue_results.recv(id_task);
                result.oaicompat = oaicompat;
                result.oaicompat_cmpl_id = completion_id;
                json result_oai;
                if (oaicompat) {
                    if (result.final_result) {
                        result_oai = result.to_json_final();
                    }
                    else {
                        result_oai = result.to_json_partial();
                    }
                }
                else {
                    // legacy completions
                    result_oai = result.data;
                }
                if (!result.error && result.stop) {
                    res.set_content(result_oai.dump(-1, ' ', false, json::error_handler_t::replace), "application/json; charset=utf-8");
                }
                else {
                    res_error(res, result_oai);
                }
                ctx_server.queue_results.remove_waiting_task_id(id_task);
            }
else {
                // Shared state to track the currently running task ID across retries.
                auto active_task_id = std::make_shared<int>(id_task);

                // Capture 'data' by value to use as a template for retries
                const auto chunked_content_provider = [id_task, active_task_id, &ctx_server, completion_id, oaicompat, send_done = params.send_done, data](size_t, httplib::DataSink& sink) mutable {
                    bool successful_completion = false;
                    
                    const auto sse = [oaicompat, &sink](const json &res) {
                        if (oaicompat == OAICOMPAT_TYPE_ANTHROPIC) {
                            return server_sent_anthropic_event(sink, res);
                        } else {
                            return server_sent_event(sink, res);
                        }
                    };

                    // 1. Parse Configuration from Request
                    
                    // Banned Strings
                    std::vector<std::string> stop_phrases;
                    if (data.contains("banned_strings") && data["banned_strings"].is_array()) {
                        for (const auto& val : data["banned_strings"]) {
                            if (val.is_string()) {
                                std::string s = val.get<std::string>();
                                if (!s.empty()) stop_phrases.push_back(s);
                            }
                        }
                    }

                    // Logit Bias Penalty (Default: -999.0)
                    float ban_bias = -999.0f;
                    if (data.contains("banned_bias") && data["banned_bias"].is_number()) {
                        ban_bias = data["banned_bias"].get<float>();
                    }

                    // Token Limit Tracking
                    int original_n_predict = -1;
                    if (data.contains("n_predict") && data["n_predict"].is_number_integer()) {
                        original_n_predict = data["n_predict"].get<int>();
                    }
                    int total_tokens_streamed = 0;

                    // ============================================================
                    // FAST PATH: No banned strings -> No buffering
                    // ============================================================
                    if (stop_phrases.empty()) {
                        while (true) {
                            server_task_result result = ctx_server.queue_results.recv(id_task);
                            if (!result.error) {
                                result.oaicompat = oaicompat;
                                result.oaicompat_cmpl_id = completion_id;
                                json res_json;
                                if (oaicompat) {
                                    res_json = result.final_result ? result.to_json_final() : result.to_json_partial();
                                } else {
                                    res_json = result.data;
                                }

                                if (res_json.is_array()) {
                                    for (const auto& res : res_json) {
                                        if (!sse(res)) {
                                            ctx_server.request_cancel(id_task);
                                            ctx_server.queue_results.remove_waiting_task_id(id_task);
                                            return false;
                                        }
                                    }
                                    if (result.stop) {
                                        successful_completion = true;
                                        break;
                                    }
                                } else {
                                    if (!sse(res_json)) {
                                        ctx_server.request_cancel(id_task);
                                        ctx_server.queue_results.remove_waiting_task_id(id_task);
                                        return false;
                                    }
                                    if (result.stop) {
                                        successful_completion = true;
                                        break;
                                    }
                                }
                            } else {
                                sse(result.data);
                                ctx_server.queue_results.remove_waiting_task_id(id_task);
                                return false;
                            }
                        }
                    } 
                    // ============================================================
                    // SLOW PATH: Buffering and Banning Logic
                    // ============================================================
                    else {
                        // Calculate Buffer Size: Exactly the length of the longest banned string
                        size_t max_banned_char_len = 0;
                        for (const auto& phrase : stop_phrases) {
                            if (phrase.length() > max_banned_char_len) {
                                max_banned_char_len = phrase.length();
                            }
                        }

                        // Ensure at least 1 to function as a buffer
                        const size_t BUFFER_SIZE = std::max((size_t)1, max_banned_char_len);
                        
                        // Initialize Buffer & State
                        std::deque<json> token_buffer;
                        
                        int current_task_id = id_task;
                        
                        // Track bans specifically for the current "next token" to be generated.
                        std::set<int> current_step_bans;

                        // Track the text that has been confirmed/sent to the client.
                        std::string current_prompt_str = "";
                        if (data.contains("prompt") && data["prompt"].is_string()) {
                            current_prompt_str = data["prompt"].get<std::string>();
                        }

                        // Helper to extract text content
                        auto get_content_str = [](const json& j) -> std::string {
                            if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                                const auto& choice = j["choices"][0];
                                if (choice.contains("delta") && choice["delta"].contains("content")) {
                                    auto val = choice["delta"]["content"];
                                    if (val.is_string()) return val.get<std::string>();
                                }
                            }
                            if (j.contains("content")) {
                                auto val = j["content"];
                                if (val.is_string()) return val.get<std::string>();
                            }
                            return ""; 
                        };

                        // Helper to extract Token ID
                        auto get_token_id = [](const json& j) -> int {
                            if (j.contains("__raw_token_id")) return j["__raw_token_id"].get<int>();
                            if (j.contains("token")) return j["token"].get<int>();
                            if (j.contains("id")) return j["id"].get<int>();
                            return -1;
                        };

                        // Helper for case-insensitive search
                        auto to_lower_str = [](std::string s) {
                            std::transform(s.begin(), s.end(), s.begin(),
                                [](unsigned char c){ return std::tolower(c); });
                            return s;
                        };

                        // Helper to print buffer
                        auto print_debug_buffer = [&](const std::deque<json>& buf) {
                            std::cout << "Debug TokenBuffer (Size " << BUFFER_SIZE << "): [";
                            size_t print_len = std::max(buf.size(), BUFFER_SIZE);
                            for (size_t i = 0; i < print_len; ++i) {
                                if (i < buf.size()) {
                                    std::string content = get_content_str(buf[i]);
                                    std::string escaped;
                                    for (char c : content) {
                                        if (c == '\n') escaped += "\\n";
                                        else if (c == '"') escaped += "\\\"";
                                        else escaped += c;
                                    }
                                    std::cout << "\"" << escaped << "\"";
                                } else {
                                    std::cout << "\"\"";
                                }
                                if (i < print_len - 1) std::cout << ", ";
                            }
                            std::cout << "]" << std::endl;
                        };

                        while (true) {
                            // Ensure shared state matches current local state
                            *active_task_id = current_task_id;

                            // Receive from the CURRENT task ID
                            server_task_result result = ctx_server.queue_results.recv(current_task_id);
                            
                            std::vector<json> items_to_buffer;
                            json raw_item = result.data; 
                            
                            if (!result.error) {
                                result.oaicompat = oaicompat;
                                result.oaicompat_cmpl_id = completion_id;
                                json res_json;
                                if (oaicompat) {
                                    res_json = result.final_result ? result.to_json_final() : result.to_json_partial();
                                } else {
                                    res_json = result.data;
                                }

                                if (res_json.is_array()) {
                                    for (const auto& r : res_json) {
                                        json item = r;
                                        if (raw_item.contains("token")) item["__raw_token_id"] = raw_item["token"];
                                        items_to_buffer.push_back(item);
                                    }
                                } else {
                                    json item = res_json;
                                    if (raw_item.contains("token")) item["__raw_token_id"] = raw_item["token"];
                                    items_to_buffer.push_back(item);
                                }
                            } else {
                                items_to_buffer.push_back(result.data);
                            }

                            // 2. Process items into buffer
                            for (const auto& item : items_to_buffer) {
                                token_buffer.push_back(item);
                            }

                            print_debug_buffer(token_buffer);

                            // 3. Check for Stop Phrases
                            std::string buffer_text = "";
                            std::vector<size_t> token_offsets; 
                            
                            for (const auto& item : token_buffer) {
                                token_offsets.push_back(buffer_text.length());
                                buffer_text += get_content_str(item);
                            }

                            std::string buffer_lower = to_lower_str(buffer_text);
                            
                            size_t match_pos = std::string::npos;
                            std::string detected_phrase = "";

                            // Iterate over the dynamic list of stop phrases
                            for (const auto& phrase : stop_phrases) {
                                std::string target_lower = to_lower_str(phrase);
                                size_t pos = buffer_lower.find(target_lower);
                                if (pos != std::string::npos) {
                                    if (match_pos == std::string::npos || pos < match_pos) {
                                        match_pos = pos;
                                        detected_phrase = phrase;
                                    }
                                }
                            }

                            if (match_pos != std::string::npos) {
                                std::cout << "Debug: Stop phrase '" << detected_phrase << "' detected. Initiating ban logic." << std::endl;

                                // Find the guilty token
                                size_t split_index = 0;
                                bool found_split = false;
                                for (size_t i = 0; i < token_offsets.size(); ++i) {
                                    size_t token_start = token_offsets[i];
                                    std::string content = get_content_str(token_buffer[i]);
                                    size_t token_end = token_start + content.length();

                                    if (token_end > match_pos) {
                                        split_index = i;
                                        found_split = true;
                                        break;
                                    }
                                }

                                if (found_split) {
                                    // 1. Flush good tokens
                                    for (size_t i = 0; i < split_index; ++i) {
                                        json& item = token_buffer[i];
                                        if (item.contains("__raw_token_id")) item.erase("__raw_token_id");
                                        
                                        current_prompt_str += get_content_str(item);
                                        current_step_bans.clear(); 
                                        
                                        if (!sse(item)) {
                                            ctx_server.request_cancel(current_task_id);
                                            ctx_server.queue_results.remove_waiting_task_id(current_task_id);
                                            return false;
                                        }
                                        
                                        total_tokens_streamed++;
                                        if (original_n_predict > 0 && total_tokens_streamed >= original_n_predict) {
                                            ctx_server.request_cancel(current_task_id);
                                            ctx_server.queue_results.remove_waiting_task_id(current_task_id);
                                            successful_completion = true;
                                            goto cleanup;
                                        }
                                    }

                                    // 2. Identify Guilty Token & Add to Bans
                                    json& guilty_item = token_buffer[split_index];
                                    int guilty_token_id = get_token_id(guilty_item);

                                    if (guilty_token_id == -1) {
                                        std::string content = get_content_str(guilty_item);
                                        auto tokens = ctx_server.tokenize(content, false);
                                        if (!tokens.empty()) guilty_token_id = tokens[0];
                                    }

                                    if (guilty_token_id != -1) {
                                        current_step_bans.insert(guilty_token_id);
                                        std::cout << "Debug: Banning token ID " << guilty_token_id << " for this spot. Total bans: " << current_step_bans.size() << std::endl;

                                        // 3. Cancel current task
                                        ctx_server.request_cancel(current_task_id);
                                        ctx_server.queue_results.remove_waiting_task_id(current_task_id);

                                        // 4. FIX STEP: Generate 1 token with ALL current bans
                                        json fix_data = data;
                                        fix_data["prompt"] = current_prompt_str;
                                        fix_data["n_predict"] = 1; 
                                        
                                        if (!fix_data.contains("logit_bias")) fix_data["logit_bias"] = json::array();
                                        
                                        for (int banned_id : current_step_bans) {
                                            fix_data["logit_bias"].push_back(json::array({banned_id, ban_bias}));
                                        }
                                        
                                        int id_fix = ctx_server.queue_tasks.get_new_id();
                                        *active_task_id = id_fix; // Update shared state for fix task
                                        ctx_server.queue_results.add_waiting_task_id(id_fix);
                                        
                                        std::vector<server_tokens> fix_inputs = tokenize_input_prompts(
                                            llama_get_vocab(ctx_server.ctx), ctx_server.mctx, fix_data["prompt"], true, true
                                        );
                                        ctx_server.request_completion(id_fix, -1, fix_data, false, false, std::move(fix_inputs[0]));

                                        // Wait for the fix token
                                        server_task_result fix_result = ctx_server.queue_results.recv(id_fix);
                                        ctx_server.queue_results.remove_waiting_task_id(id_fix);

                                        // Process fix token
                                        json fix_token_json;
                                        json raw_fix = fix_result.data;
                                        
                                        fix_result.oaicompat = oaicompat;
                                        fix_result.oaicompat_cmpl_id = completion_id;
                                        if (oaicompat) {
                                            fix_token_json = fix_result.final_result ? fix_result.to_json_final() : fix_result.to_json_partial();
                                            if (fix_token_json.is_array() && !fix_token_json.empty()) fix_token_json = fix_token_json[0];
                                        } else {
                                            fix_token_json = fix_result.data;
                                        }
                                        
                                        if (raw_fix.contains("token")) fix_token_json["__raw_token_id"] = raw_fix["token"];

                                        std::string fix_content = get_content_str(fix_token_json);

                                        // 5. RESUME STEP: Continue generation normally
                                        json resume_data = data;
                                        bool stop_after_fix = false;
                                        
                                        if (original_n_predict > 0) {
                                            int pending = 1; // The fix token we just generated
                                            if (total_tokens_streamed + pending >= original_n_predict) {
                                                stop_after_fix = true;
                                            } else {
                                                resume_data["n_predict"] = original_n_predict - (total_tokens_streamed + pending);
                                            }
                                        }

                                        if (stop_after_fix) {
                                            // We reached the limit with the fix token. Flush it and stop.
                                            token_buffer.clear();
                                            token_buffer.push_back(fix_token_json);
                                            
                                            while (!token_buffer.empty()) {
                                                json& item = token_buffer.front();
                                                if (item.contains("__raw_token_id")) item.erase("__raw_token_id");
                                                if (!sse(item)) {
                                                    ctx_server.request_cancel(*active_task_id);
                                                    ctx_server.queue_results.remove_waiting_task_id(*active_task_id);
                                                    return false;
                                                }
                                                total_tokens_streamed++;
                                                token_buffer.pop_front();
                                            }
                                            successful_completion = true;
                                            goto cleanup;
                                        }

                                        resume_data["prompt"] = current_prompt_str + fix_content;

                                        current_task_id = ctx_server.queue_tasks.get_new_id();
                                        *active_task_id = current_task_id; // Update shared state for resume task
                                        ctx_server.queue_results.add_waiting_task_id(current_task_id);
                                        
                                        std::vector<server_tokens> resume_inputs = tokenize_input_prompts(
                                            llama_get_vocab(ctx_server.ctx), ctx_server.mctx, resume_data["prompt"], true, true
                                        );
                                        ctx_server.request_completion(current_task_id, -1, resume_data, false, false, std::move(resume_inputs[0]));

                                        // 6. Update Buffer
                                        token_buffer.clear();
                                        token_buffer.push_back(fix_token_json);
                                        
                                        continue;
                                    }
                                }
                            }

                            // 4. Standard Flush Logic
                            bool should_flush_all = result.stop || result.error;

                            if (token_buffer.size() >= BUFFER_SIZE || should_flush_all) {
                                while (!token_buffer.empty()) {
                                    if (!should_flush_all && token_buffer.size() < BUFFER_SIZE) {
                                        break;
                                    }

                                    json& item_to_send = token_buffer.front();
                                    if (item_to_send.contains("__raw_token_id")) item_to_send.erase("__raw_token_id");

                                    current_prompt_str += get_content_str(item_to_send);
                                    current_step_bans.clear(); 
                                    
                                    if (!sse(item_to_send)) {
                                        ctx_server.request_cancel(current_task_id);
                                        ctx_server.queue_results.remove_waiting_task_id(current_task_id);
                                        return false;
                                    }
                                    
                                    total_tokens_streamed++;
                                    token_buffer.pop_front();

                                    if (original_n_predict > 0 && total_tokens_streamed >= original_n_predict) {
                                        ctx_server.request_cancel(current_task_id);
                                        ctx_server.queue_results.remove_waiting_task_id(current_task_id);
                                        successful_completion = true;
                                        goto cleanup;
                                    }
                                }
                            }

                            if (result.error) {
                                ctx_server.queue_results.remove_waiting_task_id(current_task_id);
                                return false;
                            }

                            if (result.stop) {
                                successful_completion = true;
                                break;
                            }
                        }
                    }

                    cleanup:
                    bool ok = true;
                    if (successful_completion && oaicompat != OAICOMPAT_TYPE_ANTHROPIC && oaicompat != OAICOMPAT_TYPE_NONE) {
                        static const std::string done_message = "data: [DONE]\n\n";
                        LOG_VERBOSE("data stream", { {"to_send", done_message} });
                        if (!sink.write(done_message.c_str(), done_message.size())) {
                            ok = false;
                        }
                    }
                    sink.done();
                    
                    // Cleanup the active task ID (which might be different from id_task in slow path)
                    ctx_server.queue_results.remove_waiting_task_id(*active_task_id);
                    
                    return ok;
                };

                auto on_complete = [active_task_id, &ctx_server](bool) {
                    // Cancel the currently active task ID
                    int id_to_cancel = *active_task_id;
                    ctx_server.request_cancel(id_to_cancel);
                    ctx_server.queue_results.remove_waiting_task_id(id_to_cancel);
                };

                res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);
            }
    };

    const auto handle_completions = [&handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        auto data = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            res,
            OAICOMPAT_TYPE_NONE);
    };

    const auto handle_completions_oai = [&handle_completions_impl](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(body);
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            res,
            OAICOMPAT_TYPE_COMPLETION);
    };

    const auto handle_models = [&params, &model_meta](const httplib::Request & req, httplib::Response & res) {
        json models = {
            {"object", "list"},
            {"data", {
                 {
                     {"id",       params.model_alias},
                     {"object",   "model"},
                     {"created",  std::time(0)},
                     {"owned_by", "llamacpp"},
                     {"meta",     model_meta}
                 },
             }}
        };

        res.set_content(models.dump(), "application/json; charset=utf-8");
    };



    const auto handle_chat_completions = [&ctx_server, &params, &handle_completions_impl, &res_error](const httplib::Request & req, httplib::Response & res) {
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files;
        json data = oaicompat_chat_params_parse(ctx_server.model, body, ctx_server.oai_parser_opt, files);
        handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            data,
            files,
            res,
            OAICOMPAT_TYPE_CHAT);
    };

    const auto handle_anthropic_messages = [&ctx_server, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = anthropic_params_from_json(
            ctx_server.model,
            body,
            ctx_server.oai_parser_opt,
            files);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            res,
            OAICOMPAT_TYPE_ANTHROPIC);
    };

    const auto handle_anthropic_count_tokens = [&ctx_server, &handle_completions_impl, &res_ok](const httplib::Request & req, httplib::Response & res) {
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);

        // Parse the Anthropic request (max_tokens is not required for count_tokens)
        json body_parsed = anthropic_params_from_json(
            ctx_server.model,
            body,
            ctx_server.oai_parser_opt,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(llama_get_vocab(ctx_server.ctx), prompt, true, true);

        res_ok(res, {{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    const auto handle_apply_template = [&ctx_server, &params, &res_ok](const httplib::Request& req, httplib::Response& res) {
        auto body = json::parse(req.body);
        std::vector<raw_buffer> files; // dummy, unused
        json data = oaicompat_chat_params_parse(ctx_server.model, body,ctx_server.oai_parser_opt, files);
        res_ok(res, { { "prompt", std::move(data.at("prompt")) } });
    };

    const auto handle_infill = [&ctx_server, &res_error, &handle_completions_impl](const httplib::Request & req, httplib::Response & res) {
        json data = json::parse(req.body);
        const int id_task = ctx_server.queue_tasks.get_new_id();
        server_tokens token; // dummy tokens
        ctx_server.queue_results.add_waiting_task_id(id_task);
        ctx_server.request_completion(id_task, -1, data, true, false, std::move(token));
        std::vector<raw_buffer> files; // dummy
        handle_completions_impl(
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            res,
            OAICOMPAT_TYPE_NONE); // infill is not OAI compatible
    };

    const auto handle_tokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::vector<llama_token> tokens;
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            tokens = ctx_server.tokenize(body.at("content"), add_special);
        }
        const json data = format_tokenizer_response(tokens);
        return res.set_content(data.dump(), "application/json; charset=utf-8");
    };

    const auto handle_detokenize = [&ctx_server](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const std::vector<llama_token> tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens);
        }

        const json data = format_detokenized_response(content);
        return res.set_content(data.dump(), "application/json; charset=utf-8");
    };


    const auto handle_embeddings = [&ctx_server, &res_error](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);
        bool is_openai = false;

        // an input prompt can be a string or a list of tokens (integer)
        json prompt;
        if (body.count("input") != 0) {
            is_openai = true;
            prompt = body.at("input");
        } else if (body.count("content") != 0) {
            // with "content", we only support single prompt
            prompt = std::vector<std::string>{body.at("content")};
        } else {
            res_error(res, format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return;
        }

        // create and queue the task
        json responses;
        {
            const int id_task = ctx_server.queue_tasks.get_new_id();
            ctx_server.queue_results.add_waiting_task_id(id_task);
            std::vector<server_tokens> inputs;
            inputs = tokenize_input_prompts(llama_get_vocab(ctx_server.ctx), ctx_server.mctx, prompt, true, true);
            ctx_server.request_completion(id_task, -1, {{"prompt", prompt}}, false, true, std::move(inputs[0]));

            // get the result
            server_task_result result = ctx_server.queue_results.recv(id_task);
            ctx_server.queue_results.remove_waiting_task_id(id_task);
            if (!result.error) {
                if (result.data.count("results")) {
                    // result for multi-task
                    responses = result.data.at("results");
                } else {
                    // result for single task
                    responses = std::vector<json>{ result.data };
                }
            } else {
                // error received, ignore everything else
                res_error(res, result.data);
                return;
            }
        }

        // write JSON response
        json root = is_openai
            ? format_embeddings_response_oaicompat(body, responses, false)
            : responses[0];
        return res.set_content(root.dump(), "application/json; charset=utf-8");
    };

    const auto handle_lora_adapters_list = [&](const httplib::Request & req, httplib::Response & res) {
        json result = json::array();
        for (size_t i = 0; i < ctx_server.lora_adapters.size(); ++i) {
            auto & la = ctx_server.lora_adapters[i];
            result.push_back({
                {"id", i},
                {"path", la.path},
                {"scale", la.scale},
            });
        }
        res.set_content(result.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto handle_lora_adapters_apply = [&](const httplib::Request & req, httplib::Response & res) {
        const std::vector<json> body = json::parse(req.body);
        int max_idx = ctx_server.lora_adapters.size();

        // clear existing value
        for (auto & la : ctx_server.lora_adapters) {
            la.scale = 0.0f;
        }

        // set value
        for (auto entry : body) {
            int id      = entry.at("id");
            float scale = entry.at("scale");
            if (0 <= id && id < max_idx) {
                ctx_server.lora_adapters[id].scale = scale;
            } else {
                throw std::runtime_error("invalid adapter id");
            }
        }

        server_task task;
        task.type = SERVER_TASK_TYPE_SET_LORA;
        const int id_task = ctx_server.queue_tasks.post(std::move(task));
        ctx_server.queue_results.add_waiting_task_id(id_task);

        server_task_result result = ctx_server.queue_results.recv(id_task);
        ctx_server.queue_results.remove_waiting_task_id(id_task);

        res.set_content(result.data.dump(), "application/json");
        res.status = 200; // HTTP OK
    };

    const auto list_saved_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        json response = json::array();

        try {
            for (const auto& entry : fs::directory_iterator(params.slot_save_path)) {
                if (!entry.is_regular_file() || entry.file_size() < 12) {
                    continue;
                }

                std::ifstream file(entry.path(), std::ios::binary);
                if (!file) continue;

                uint32_t magic, version, n_token_count;
                file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
                file.read(reinterpret_cast<char*>(&version), sizeof(version));
                file.read(reinterpret_cast<char*>(&n_token_count), sizeof(n_token_count));

                if (magic != LLAMA_STATE_SEQ_MAGIC ||
                    version != LLAMA_STATE_SEQ_VERSION ||
                    entry.file_size() < (12 + (n_token_count * sizeof(llama_token)))) {
                    continue;
                }

                std::vector<llama_token> tokens(n_token_count);
                file.read(reinterpret_cast<char*>(tokens.data()), tokens.size() * sizeof(llama_token));

                //C++17 is not modern enough to have a nice and portable way to get the mtime of a file
                //so the following seems to be needed
                auto ftime = fs::last_write_time(entry.path());
                auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now()
                );
                std::time_t c_time = std::chrono::system_clock::to_time_t(system_time);
                std::tm tm_struct;
                #if defined(_WIN32)
                localtime_s(&tm_struct, &c_time);
                #else
                localtime_r(&c_time, &tm_struct);
                #endif
                std::ostringstream oss;
                oss << std::put_time(&tm_struct, "%Y-%m-%d %H:%M:%S");
                auto str_time = oss.str();


                response.push_back({
                    {"filename", entry.path().filename().string()},
                    {"filesize", entry.file_size()},
                    {"mtime", str_time},
                    {"token_count", n_token_count},
                    {"prompt", tokens_to_str(ctx_server.ctx, tokens)}
                });
            }
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    const auto list_slot_prompts = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res) {
        json response = json::array();
        for (server_slot & slot : ctx_server.slots) {
            response.push_back({
                {"slot_id", slot.id},
                {"token_count", slot.cache_tokens.size()},
                {"prompt", slot.cache_tokens.detokenize(ctx_server.ctx, true) }
            });
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };


    const auto delete_saved_prompt = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res)-> void {
        json response;
        namespace fs = std::filesystem;

        try {
            const json body = json::parse(req.body);
            const std::string filename_str = body.at("filename");

            // prevent directory traversal attacks
            if (filename_str.find("..") != std::string::npos || filename_str.find('/') != std::string::npos || filename_str.find('\\') != std::string::npos) {
                res.status = 400;
                response = {{"error", "Invalid filename format."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            const fs::path file_to_delete = fs::path(params.slot_save_path) / fs::path(filename_str);

            if (!fs::exists(file_to_delete) || !fs::is_regular_file(file_to_delete)) {
                res.status = 404;
                response = {{"error", "File not found."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            if (fs::remove(file_to_delete)) {
                response = {
                    {"status", "deleted"},
                    {"filename", filename_str}
                };
            } else {
                res.status = 500;
                response = {{"error", "Failed to delete the file."}};
            }
        } catch (const json::parse_error& e) {
            res.status = 400;
            response = {{"error", "Invalid JSON request body."}};
        } catch (const json::out_of_range& e) {
            res.status = 400;
            response = {{"error", "Missing 'filename' key in request body."}};
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }
        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    const auto rename_saved_prompt = [&ctx_server, &params](const httplib::Request& req, httplib::Response& res)-> void {
        json response;
        namespace fs = std::filesystem;

        try {
            const json body = json::parse(req.body);
            const std::string old_filename_str = body.at("old_filename");
            const std::string new_filename_str = body.at("new_filename");

            if (old_filename_str.find("..") != std::string::npos || old_filename_str.find_first_of("/\\") != std::string::npos ||
                new_filename_str.find("..") != std::string::npos || new_filename_str.find_first_of("/\\") != std::string::npos) {
                res.status = 400;
                response = {{"error", "Invalid filename format."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            const fs::path old_path = fs::path(params.slot_save_path) / old_filename_str;
            const fs::path new_path = fs::path(params.slot_save_path) / new_filename_str;

            if (!fs::exists(old_path) || !fs::is_regular_file(old_path)) {
                res.status = 404;
                response = {{"error", "Source file not found."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            if (fs::exists(new_path)) {
                res.status = 409;
                response = {{"error", "Destination filename already exists."}};
                res.set_content(response.dump(), "application/json; charset=utf-8");
                return;
            }

            std::error_code ec;
            fs::rename(old_path, new_path, ec);

            if (ec) {
                res.status = 500;
                response = {{"error", "Failed to rename file: " + ec.message()}};
            } else {
                response = {
                    {"status", "renamed"},
                    {"old_filename", old_filename_str},
                    {"new_filename", new_filename_str}
                };
            }

        } catch (const json::parse_error& e) {
            res.status = 400;
            response = {{"error", "Invalid JSON request body."}};
        } catch (const json::out_of_range& e) {
            res.status = 400;
            response = {{"error", "Missing 'old_filename' or 'new_filename' in request body."}};
        } catch (const std::exception& e) {
            res.status = 500;
            response = {{"error", e.what()}};
        }

        res.set_content(response.dump(), "application/json; charset=utf-8");
    };

    auto handle_static_file = [](unsigned char * content, size_t len, const char * mime_type) {
        return [content, len, mime_type](const httplib::Request &, httplib::Response & res) {
            res.set_content(reinterpret_cast<const char*>(content), len, mime_type);
            return false;
        };
    };
#ifdef SQLITE3_MODERN_CPP_SUPPORT
    const auto handle_version = [&params, sqlite_extension_loaded](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            json{{"version", 4},
            {"features", {{"sql", !params.sql_save_file.empty()}, {"zstd_compression", sqlite_extension_loaded}}}}.dump(),
            "application/json"
        );
    };
#else
    const auto handle_version = [](const httplib::Request&, httplib::Response& res)-> void {
        res.set_content(
             json{{"version", 4},
             {"features", {{"sql", false}, {"zstd_compression", false}}}}.dump(),
             "application/json"
        );
    };
#endif

#ifdef SQLITE3_MODERN_CPP_SUPPORT
    auto db_handler = [db_handle](auto func) {
        return [func, db_handle](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
	    try {
                const json body = !req.body.empty() ? json::parse(req.body) : json::object();
                func(*db_handle, body, req, res);
            } catch(const std::exception& e) {
                res.status = 500;
                res.set_content(
                    json{{"ok", false}, {"message", e.what()}}.dump(),
                    "application/json"
                );
            }
        };
    };
#else
    auto db_handler = [db_handle](auto func) {
        return [func, db_handle](const httplib::Request& req, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.status = 500;
            res.set_content(
                json{{"ok", false}, {"message", "Sqlite3 support was not enabled. Recompile with '-DLLAMA_SERVER_SQLITE3=ON'"}}.dump(),
                "application/json"
            );
        };
    };
#endif

    const auto normalize_store_name = [](const std::string& storeName) {
        if(storeName.empty()) return std::string("sessions");

        std::string normalized;
        normalized.reserve(storeName.size());

        for(char c : storeName) {
            if(std::isalpha(static_cast<unsigned char>(c))) {
                normalized.push_back(std::tolower(static_cast<unsigned char>(c)));
            }
        }

        return normalized.empty() ? "sessions" : normalized;
    };

    const auto get_key_string = [](const json& j) {
        return j.is_string() ? j.get<std::string>() : j.dump();
    };


    const auto handle_load = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        std::string data;
	const std::string store = normalize_store_name(body["storeName"]);
	db.db << "SELECT data FROM " + store + " WHERE key = ?" << get_key_string(body["key"]) >> data;
	if(data.empty()) {
            res.status = 404;
            res.set_content(json{{"ok", false}, {"message", "Key not found"}}.dump(), "application/json");
        } else {
            json response{{"ok", true}};
	    response["result"] = (store == "names") ? json(data) : json::parse(data);
            res.set_content(response.dump(), "application/json");
        }
    });

    const auto handle_save = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        const std::string store = normalize_store_name(body["storeName"]);
        const std::string data = (store == "names") ? body["data"].get<std::string>() : body["data"].dump();
        db.db << "INSERT OR REPLACE INTO " + store + " (key, data) VALUES (?, ?)" << get_key_string(body["key"]) << data;
        res.set_content(json{{"ok", true}, {"result", "Data saved successfully"}}.dump(), "application/json");
    });

    const auto handle_rename = db_handler([get_key_string](auto& db, const json& body, auto&, auto& res) {
        db.db << "UPDATE names SET data = ? WHERE key = ?"
            << body["newName"].get<std::string>()
            << get_key_string(body["key"]);
        res.set_content(json{{"ok", true}, {"result", "Session renamed successfully"}}.dump(), "application/json");
    });

    const auto handle_all = db_handler([normalize_store_name](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT key, data FROM " + normalize_store_name(body["storeName"]) >>
            [&](const std::string& key, const std::string& data) {
                result[key] = json::parse(data);
            };
        res.set_content(json{{"ok", true}, {"result", result}}.dump(), "application/json");
    });

    const auto handle_sessions = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT key, data FROM names" >> [&](const std::string& key, const std::string& data) {
            result[key] = data;
        };
        res.set_content(json{{"ok", true}, {"result", result}}.dump(), "application/json");
    });

    const auto handle_delete = db_handler([normalize_store_name, get_key_string](auto& db, const json& body, auto&, auto& res) {
        db.db << "DELETE FROM " + normalize_store_name(body["storeName"]) + " WHERE key = ?"
            << get_key_string(body["key"]);
        res.set_content(json{{"ok", true}, {"result", "Session deleted successfully"}}.dump(), "application/json");
    });

    const auto handle_vacuum = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "VACUUM";
        res.set_content(json{"ok", true}.dump(), "application/json");
    });

    const auto handle_zstd_get_configs = db_handler([](auto& db, const json& body, auto&, auto& res) {
        json result = json::object();
        db.db << "SELECT id, config FROM _zstd_configs" >> [&](const std::string id, const std::string& config) {
            result[id] = config;
        };
        res.set_content(json{{"ok", true}, {"configs", result}}.dump(), "application/json");
    });

    const auto handle_zstd_maintenance = db_handler([](auto& db, const json& body, auto&, auto& res) {
        std::string data;
        if (body["duration"].is_null()) {
            db.db << "select zstd_incremental_maintenance(?, ?)" <<  nullptr << body["db_load"].get<double>() >> data;
        }
	else {
            db.db << "select zstd_incremental_maintenance(?, ?)" << body["duration"].get<double>() << body["db_load"].get<double>() >> data;
        }
        json response{{"ok", true}};
        response["result"] = json::parse(data);
        res.set_content(response.dump(), "application/json");
    });

    const auto handle_zstd_enable = db_handler([](auto& db, const json& body, auto&, auto& res) {
        db.db << "select zstd_enable_transparent('{\"table\": \"" + body["table"].get<std::string>() + "\",\"column\": \"" + body["column"].get<std::string>() + "\", \"compression_level\": " + std::to_string(body["compression_level"].get<int>()) + ", \"dict_chooser\": \"''a''\", \"train_dict_samples_ratio\": " + std::to_string(body["train_dict_samples_ratio"].get<int>()) + "}')";
        res.set_content(json{"ok", true}.dump(), "application/json");
    });

    const auto handle_zstd_config_update = db_handler([](auto& db, const json& body, auto&, auto& res) {
        std::string patch_json = "{\"compression_level\": " + std::to_string(body["compression_level"].get<int>()) + ", \"train_dict_samples_ratio\": " + std::to_string(body["train_dict_samples_ratio"].get<int>()) + "}";
        db.db << "update _zstd_configs set config = json_patch(config, '" + patch_json + "')";
        res.set_content(json{{"ok", true}}.dump(), "application/json");
    });

    //
    // Router
    //
    if (params.webui == COMMON_WEBUI_NONE) {
        LLAMA_LOG_INFO("Web UI is disabled\n");
    }
    else {
        // register static assets routes
        if (!params.public_path.empty()) {
            // Set the base directory for serving static files
            svr->set_base_dir(params.public_path);
        }

        {
            // register static assets routes
            if (!params.public_path.empty()) {
                // Set the base directory for serving static files
                bool is_found = svr->set_mount_point("/", params.public_path);
                if (!is_found) {
                    GGML_ABORT("%s: static assets path not found: %s\n", __func__, params.public_path.c_str());
                    return 1;
                }
            }
            else {

                // using embedded static index.html
                svr->Get("/", [params](const httplib::Request& req, httplib::Response& res) {
                    if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
                        res.set_content("Error: gzip is not supported by this browser", "text/plain");
                    }
                    else {
                        res.set_header("Content-Encoding", "gzip");
                        // COEP and COOP headers, required by pyodide (python interpreter)
                        res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
                        res.set_header("Cross-Origin-Opener-Policy", "same-origin");
                        if (params.webui == COMMON_WEBUI_AUTO) {
                            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                        }
                        else if (params.webui == COMMON_WEBUI_LLAMACPP) {
                            res.set_content(reinterpret_cast<const char*>(index_llamacpp_html_gz), index_llamacpp_html_gz_len, "text/html; charset=utf-8");
                        }
                        else {
                            res.set_content(reinterpret_cast<const char*>(index_html_gz), index_html_gz_len, "text/html; charset=utf-8");
                        }
                    }
                    return false;
                    });
            }
        }
    }
    // register API routes
    svr->Get ("/health",              handle_health);
    svr->Get ("/metrics",             handle_metrics);
    svr->Get ("/props",               handle_props);
    svr->Get("/v1/props",             handle_props_simple);
    svr->Get ("/v1/models",           handle_models);
    svr->Post("/completion",          handle_completions); // legacy
    svr->Post("/completions", handle_completions); // legacy
    svr->Post("/v1/completions",     handle_completions_oai);
    svr->Post("/chat/completions",    handle_chat_completions);
    svr->Post("/v1/chat/completions", handle_chat_completions);
    svr->Post("/v1/messages",         handle_anthropic_messages);
    svr->Post("/v1/messages/count_tokens", handle_anthropic_count_tokens);
    svr->Post("/infill",              handle_infill);
    svr->Post("/embedding",           handle_embeddings); // legacy
    svr->Post("/embeddings",          handle_embeddings);
    svr->Post("/v1/embeddings",       handle_embeddings);
    svr->Post("/tokenize",            handle_tokenize);
    svr->Post("/detokenize",          handle_detokenize);
    svr->Post("/apply-template",      handle_apply_template);
    // LoRA adapters hotswap
    svr->Get ("/lora-adapters",       handle_lora_adapters_list);
    svr->Post("/lora-adapters",       handle_lora_adapters_apply);
    // Save & load slots
    svr->Get ("/slots",               handle_slots);
    svr->Get ("/slots/list",          list_slot_prompts);
    if (!params.slot_save_path.empty()) {
        // these endpoints rely on slot_save_path existing
        svr->Post("/slots/:id_slot",  handle_slots_action);
        svr->Get ("/list",            list_saved_prompts);
        svr->Post("/delete_prompt",   delete_saved_prompt);
        svr->Post("/rename_prompt",   rename_saved_prompt);

    }

    svr->Get ("/version", handle_version);
    if (!params.sql_save_file.empty()) {
        // these endpoints rely on sql_save_file existing
        svr->Post("/load", handle_load);
        svr->Post("/save", handle_save);
        svr->Post("/rename", handle_rename);
        svr->Post("/all", handle_all);
        svr->Post("/sessions", handle_sessions);
        svr->Get ("/sessions", handle_sessions);
        svr->Post("/delete", handle_delete);
        //VACUUM is there for the extension but does not require the extension
        svr->Get ("/vacuum", handle_vacuum);
#ifdef SQLITE3_MODERN_CPP_SUPPORT
        if (sqlite_extension_loaded) {
            svr->Get ("/zstd_get_configs", handle_zstd_get_configs);
            svr->Post("/zstd_incremental_maintenance", handle_zstd_maintenance);
            svr->Post("/zstd_enable_transparent", handle_zstd_enable);
            svr->Post("/zstd_update_transparent", handle_zstd_config_update);
	}
#endif
    }
    //
    // Start the server
    //
    if (params.n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        params.n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    log_data["n_threads_http"] =  std::to_string(params.n_threads_http);
    svr->new_task_queue = [&params] { return new httplib::ThreadPool(params.n_threads_http); };

    LOG_INFO("HTTP server listening", log_data);

    // run the HTTP server in a thread - see comment below
    std::thread t([&]() {
        if (!svr->listen_after_bind()) {
            state.store(SERVER_STATE_ERROR);
            return 1;
        }

        return 0;
    });

    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
        });
    ctx_server.queue_tasks.on_finish_multitask(std::bind(
        &server_context::on_finish_multitask, &ctx_server, std::placeholders::_1));
    ctx_server.queue_tasks.on_update_slots(std::bind(
        &server_context::update_slots, &ctx_server));
    ctx_server.queue_results.on_multitask_update(std::bind(
        &server_queue::update_multitask,
        &ctx_server.queue_tasks,
        std::placeholders::_1,
        std::placeholders::_2,
        std::placeholders::_3
    ));

    shutdown_handler = [&](int) {
        ctx_server.queue_tasks.terminate();
    };

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    ctx_server.queue_tasks.start_loop();

    svr->stop();
    t.join();

    llama_backend_free();

    return 0;
}
