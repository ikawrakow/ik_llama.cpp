#pragma once

#include "common.h"
#include "http.h"
#include <string>
#include <unordered_set>
#include <list>
#include <map>

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static httplib::Request prepare_proxy_req_header(const std::string & method,
    const std::string & scheme,
    const std::string & host,
    int port,
    const std::string & path,
    const std::map<std::string, std::string> & headers,
    const std::string & body,
    const httplib::FormFiles & files) {
        httplib::Request  req;
        bool has_files = !files.empty();
        req.form.files = files;
        std::string effective_body = body;
        std::string override_content_type;
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                req.set_header(key, is_default_port ? host : host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        //req.response_handler = response_handler;
        //req.content_receiver = content_receiver;
    
    return req;
}

static std::string get_param(httplib::Params params,const std::string & key, const std::string & def = "") {
    auto it = params.find("url");
    if (it != params.end()) {
        return  it->second;
    }
    return def;
}

static void proxy_request(const httplib::Request & req,
    httplib::Response & res,
    const std::string & method) {
    std::string target_url = get_param(req.params, "url");
    common_http_url parsed_url = common_http_parse_url(target_url);
    if (parsed_url.host.empty()) {
        throw std::runtime_error("invalid target URL: missing host");
    }

    if (parsed_url.path.empty()) {
        parsed_url.path = "/";
    }

    if (!parsed_url.password.empty()) {
        throw std::runtime_error("authentication in target URL is not supported");
    }

    if (parsed_url.scheme != "http" && parsed_url.scheme != "https") {
        throw std::runtime_error("unsupported URL scheme in target URL: " + parsed_url.scheme);
    }

    SRV_INF("proxying %s request to %s://%s:%i%s\n", method.c_str(), parsed_url.scheme.c_str(), parsed_url.host.c_str(), parsed_url.port, parsed_url.path.c_str());
    std::map<std::string, std::string> headers;
    for (auto [key, value] : req.headers) {
        auto new_key = key;
        if (string_starts_with(new_key, "x-proxy-header-")) {
            string_replace_all(new_key, "x-proxy-header-", "");
        }
        headers[new_key] = value;
    }

    httplib::Request proxy_req = prepare_proxy_req_header(method,
        parsed_url.scheme,
        parsed_url.host,
        parsed_url.port,
        parsed_url.path,
        headers,
        req.body,
        req.form.files);

    // Make the proxied request
    httplib::Result proxy_res;
    
    if (parsed_url.scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(parsed_url.host, parsed_url.port);
        // set timeouts, follow redirects as needed
        cli.set_connection_timeout(600);
        cli.set_read_timeout(600);
        cli.set_write_timeout(600);
        cli.set_follow_location(true);
        proxy_res = cli.send(proxy_req);
#else
        res.status = 501;
        res.set_content("HTTPS not supported (build with OpenSSL)", "text/plain");
        return;
#endif
    } else {
        httplib::Client cli(parsed_url.host, parsed_url.port);
        cli.set_connection_timeout(600);
        cli.set_read_timeout(600);
        cli.set_write_timeout(600);
        proxy_res = cli.send(std::move(proxy_req));
    }

    if (!proxy_res) {
        std::string error_data = "Proxy failed: " + httplib::to_string(proxy_res.error());
        json final_response{ {"error", error_data} };
        res.set_content(safe_json_to_str(final_response), "application/json; charset=utf-8");
        res.status = json_value(error_data, "code", 500);
        return;
    }

    res.status = proxy_res->status;
    res.set_content(proxy_res->body, proxy_res->get_header_value("Content-Type"));
    for (const auto & h : proxy_res->headers) {
        // skip hop-by-hop headers
        if (h.first != "Transfer-Encoding" && h.first != "Connection")
            res.set_header(h.first, h.second);
    }
}

static void proxy_handler_get(const httplib::Request & req, httplib::Response & res) {
    proxy_request(req, res, "GET");
}

static void proxy_handler_post(const httplib::Request & req, httplib::Response & res) {
    proxy_request(req, res, "POST");
}
