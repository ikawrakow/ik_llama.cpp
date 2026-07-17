// Internal cgroup mount resolver. Kept header-only so Linux fixtures can test
// the exact parser and resolver used by llama.cpp without touching /proc.
#pragma once

#include <algorithm>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

// cgroup paths in /proc files are absolute, lexical paths.  Never feed a
// malformed path (in particular one containing "..") back into the host
// filesystem: an unresolved cgroup must make the auto policy conservative.
static bool llama_normalize_cgroup_path(const std::string & path, std::string & result) {
    if (path.empty() || path[0] != '/') {
        return false;
    }

    std::vector<std::string> components;
    size_t begin = 1;
    while (begin <= path.size()) {
        const size_t end = path.find('/', begin);
        const std::string component = path.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (component == "..") {
            return false;
        }
        if (!component.empty() && component != ".") {
            components.push_back(component);
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }

    result = "/";
    for (const std::string & component : components) {
        if (result.size() > 1) {
            result += '/';
        }
        result += component;
    }
    return true;
}

static bool llama_cgroup_path_has_prefix(const std::string & path, const std::string & prefix) {
    return (prefix == "/" && !path.empty() && path[0] == '/') || path == prefix ||
        (prefix != "/" && path.size() > prefix.size() && path.compare(0, prefix.size(), prefix) == 0 && path[prefix.size()] == '/');
}

static std::string llama_cgroup_join_path(const std::string & base, const std::string & suffix) {
    if (suffix == "/") {
        return base;
    }
    return base == "/" ? suffix : base + suffix;
}

// Decode the four escapes permitted by proc(5) mountinfo. Reject any other
// backslash sequence rather than guessing which host path it denotes.
static bool llama_decode_mountinfo_path(const std::string & encoded, std::string & decoded) {
    decoded.clear();
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '\\') {
            decoded += encoded[i];
            continue;
        }
        if (i + 3 >= encoded.size()) {
            return false;
        }
        const std::string escape = encoded.substr(i, 4);
        if (escape == "\\040") {
            decoded += ' ';
        } else if (escape == "\\011") {
            decoded += '\t';
        } else if (escape == "\\012") {
            decoded += '\n';
        } else if (escape == "\\134") {
            decoded += '\\';
        } else {
            return false;
        }
        i += 3;
    }
    return true;
}

static bool llama_comma_list_contains(const std::string & values, const std::string & needle) {
    std::stringstream stream(values);
    std::string value;
    while (std::getline(stream, value, ',')) {
        if (value == needle) {
            return true;
        }
    }
    return false;
}

struct llama_cgroup_membership {
    bool        v2;
    std::string path;
};

struct llama_cgroup_mount {
    bool        v2;
    std::string root;
    std::string mountpoint;
};

struct llama_resolved_cgroup_mount {
    std::string mountpoint;
    std::string path;
};

static bool llama_parse_cgroup_memberships(std::istream & file, std::vector<llama_cgroup_membership> & result) {
    result.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t c1 = line.find(':');
        const size_t c2 = c1 == std::string::npos ? std::string::npos : line.find(':', c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos || c1 == 0 || c2 + 1 >= line.size()) {
            return false;
        }

        std::string path;
        if (!llama_normalize_cgroup_path(line.substr(c2 + 1), path)) {
            return false;
        }
        const std::string controllers = line.substr(c1 + 1, c2 - c1 - 1);
        if (controllers.empty()) {
            result.push_back({ true, path });
        } else if (llama_comma_list_contains(controllers, "memory")) {
            result.push_back({ false, path });
        }
    }
    return !file.bad();
}

static bool llama_read_cgroup_memberships(std::vector<llama_cgroup_membership> & result) {
    std::ifstream file("/proc/self/cgroup");
    return file.is_open() && llama_parse_cgroup_memberships(file, result);
}

static bool llama_parse_cgroup_mounts(std::istream & file, std::vector<llama_cgroup_mount> & result) {
    result.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<std::string> fields;
        std::string field;
        while (stream >> field) {
            fields.push_back(field);
        }
        if (fields.size() < 10) {
            return false;
        }
        const auto dash = std::find(fields.begin() + 6, fields.end(), "-");
        if (dash == fields.end() || std::distance(dash, fields.end()) < 4) {
            return false;
        }
        const size_t dash_index = (size_t) std::distance(fields.begin(), dash);
        const std::string & fstype = fields[dash_index + 1];
        const bool v2 = fstype == "cgroup2";
        if (!v2 && fstype != "cgroup") {
            continue;
        }
        // For v1, the controllers are listed in super options (after source).
        if (!v2 && !llama_comma_list_contains(fields[dash_index + 3], "memory")) {
            continue;
        }

        std::string root_encoded;
        std::string mountpoint_encoded;
        if (!llama_decode_mountinfo_path(fields[3], root_encoded) ||
            !llama_decode_mountinfo_path(fields[4], mountpoint_encoded)) {
            return false;
        }
        std::string root;
        std::string mountpoint;
        if (!llama_normalize_cgroup_path(root_encoded, root) ||
            !llama_normalize_cgroup_path(mountpoint_encoded, mountpoint)) {
            return false;
        }
        result.push_back({ v2, root, mountpoint });
    }
    return !file.bad();
}

static bool llama_read_cgroup_mounts(std::vector<llama_cgroup_mount> & result) {
    std::ifstream file("/proc/self/mountinfo");
    return file.is_open() && llama_parse_cgroup_mounts(file, result);
}

// Resolve one membership against its cgroup mounts. Host-relative mappings are
// preferred; among them, use the most specific root. Keep equally-specific
// duplicates (including bind mounts), whose headroom must be intersected.
static bool llama_resolve_cgroup_mounts(
        const llama_cgroup_membership & membership,
        const std::vector<llama_cgroup_mount> & mounts,
        std::vector<llama_resolved_cgroup_mount> & result) {
    result.clear();
    size_t best_root_size = 0;
    bool have_host_relative = false;
    bool have_exact = false;

    for (const llama_cgroup_mount & mount : mounts) {
        if (mount.v2 != membership.v2 || !llama_cgroup_path_has_prefix(membership.path, mount.root)) {
            continue;
        }
        const bool exact = membership.path == mount.root;
        const size_t root_size = mount.root.size();
        if (!have_host_relative || (exact && !have_exact) || (exact == have_exact && root_size > best_root_size)) {
            result.clear();
            best_root_size = root_size;
            have_host_relative = true;
            have_exact = exact;
        }
        if (exact == have_exact && root_size == best_root_size) {
            const std::string suffix = mount.root == "/" ? membership.path : membership.path.substr(mount.root.size());
            result.push_back({ mount.mountpoint, llama_cgroup_join_path(mount.mountpoint, suffix.empty() ? "/" : suffix) });
        }
    }
    if (have_host_relative) {
        return true;
    }

    // cgroup namespaces commonly expose membership paths relative to the
    // mounted hierarchy. In that case the membership is appended directly.
    for (const llama_cgroup_mount & mount : mounts) {
        if (mount.v2 == membership.v2) {
            result.push_back({ mount.mountpoint, llama_cgroup_join_path(mount.mountpoint, membership.path) });
        }
    }
    return !result.empty();
}

static bool llama_cgroup_parent_path(const std::string & path, const std::string & mountpoint, std::string & parent) {
    if (path == mountpoint) return false;
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return false;
    if (slash == 0) {
        if (mountpoint != "/") return false;
        parent = "/";
        return true;
    }
    parent = path.substr(0, slash);
    return llama_cgroup_path_has_prefix(parent, mountpoint);
}
