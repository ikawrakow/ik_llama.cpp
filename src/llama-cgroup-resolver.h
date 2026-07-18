// Internal cgroup mount resolver. Kept header-only so Linux fixtures can test
// the exact parser and resolver used by llama.cpp without touching /proc.
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

// cgroup paths in /proc files are absolute, lexical paths.  Never feed a
// malformed path (in particular one containing "..") back into the host
// filesystem: an unresolved cgroup must make the auto policy conservative.
static inline bool llama_normalize_cgroup_path(const std::string & path, std::string & result) {
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

static inline bool llama_cgroup_path_has_prefix(const std::string & path, const std::string & prefix) {
    return (prefix == "/" && !path.empty() && path[0] == '/') || path == prefix ||
        (prefix != "/" && path.size() > prefix.size() && path.compare(0, prefix.size(), prefix) == 0 && path[prefix.size()] == '/');
}

static inline std::string llama_cgroup_join_path(const std::string & base, const std::string & suffix) {
    if (suffix == "/") {
        return base;
    }
    return base == "/" ? suffix : base + suffix;
}

// Decode the four escapes permitted by proc(5) mountinfo. Reject any other
// backslash sequence rather than guessing which host path it denotes.
static inline bool llama_decode_mountinfo_path(const std::string & encoded, std::string & decoded) {
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

static inline bool llama_comma_list_contains(const std::string & values, const std::string & needle) {
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
    bool        v2;
    std::string mountpoint;
    std::string path;
};

// Intersect all visible hierarchy headrooms. The reader must fail closed for a
// mapping it cannot evaluate; ignoring an unreadable bind/full-hierarchy mount
// could otherwise turn an unknown ancestor limit into an unsafe AUTO_KEEP.
template<typename Reader>
static inline bool llama_intersect_cgroup_headrooms(
        const std::vector<llama_resolved_cgroup_mount> & mappings,
        Reader reader,
        bool & limited,
        uint64_t & bytes) {
    uint64_t headroom = UINT64_MAX;
    for (const auto & mapping : mappings) {
        bool candidate_limited = false;
        uint64_t candidate_bytes = 0;
        if (!reader(mapping, candidate_limited, candidate_bytes)) {
            return false;
        }
        if (candidate_limited) {
            headroom = std::min(headroom, candidate_bytes);
        }
    }
    limited = headroom != UINT64_MAX;
    bytes = limited ? headroom : 0;
    return true;
}

static inline bool llama_parse_cgroup_memberships(std::istream & file, std::vector<llama_cgroup_membership> & result) {
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

static inline bool llama_read_cgroup_memberships(std::vector<llama_cgroup_membership> & result) {
    std::ifstream file("/proc/self/cgroup");
    return file.is_open() && llama_parse_cgroup_memberships(file, result);
}

static inline bool llama_parse_cgroup_mounts(std::istream & file, std::vector<llama_cgroup_mount> & result) {
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

static inline bool llama_read_cgroup_mounts(std::vector<llama_cgroup_mount> & result) {
    std::ifstream file("/proc/self/mountinfo");
    return file.is_open() && llama_parse_cgroup_mounts(file, result);
}

// Resolve one membership against its cgroup mounts. Keep every host-relative
// mapping: a less-specific/full-hierarchy mount can expose an ancestor limit
// hidden by a more-specific bind mount. Namespace-relative mappings are only a
// fallback when no host-relative mapping exists.
static inline bool llama_resolve_cgroup_mounts(
        const llama_cgroup_membership & membership,
        const std::vector<llama_cgroup_mount> & mounts,
        std::vector<llama_resolved_cgroup_mount> & result) {
    result.clear();
    bool have_host_relative = false;

    auto append_unique = [&result, &membership] (const llama_cgroup_mount & mount, const std::string & path) {
        const auto duplicate = std::find_if(result.begin(), result.end(), [&mount, &path, &membership] (const auto & existing) {
            return existing.v2 == membership.v2 && existing.mountpoint == mount.mountpoint && existing.path == path;
        });
        if (duplicate == result.end()) {
            result.push_back({ membership.v2, mount.mountpoint, path });
        }
    };

    for (const llama_cgroup_mount & mount : mounts) {
        if (mount.v2 != membership.v2 || !llama_cgroup_path_has_prefix(membership.path, mount.root)) {
            continue;
        }
        have_host_relative = true;
        const std::string suffix = mount.root == "/" ? membership.path : membership.path.substr(mount.root.size());
        append_unique(mount, llama_cgroup_join_path(mount.mountpoint, suffix.empty() ? "/" : suffix));
    }
    if (have_host_relative) {
        return true;
    }

    // cgroup namespaces commonly expose membership paths relative to the
    // mounted hierarchy. In that case the membership is appended directly.
    for (const llama_cgroup_mount & mount : mounts) {
        if (mount.v2 == membership.v2) {
            append_unique(mount, llama_cgroup_join_path(mount.mountpoint, membership.path));
        }
    }
    return !result.empty();
}

static inline bool llama_cgroup_parent_path(const std::string & path, const std::string & mountpoint, std::string & parent) {
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

// A cgroup-v2 hierarchy root can be visible without exposing the memory
// interface files at that exact mountpoint (for example through a namespace or
// delegated mount).  A missing pair is safe to skip only at the mountpoint;
// below it, or when just one file is missing, the hierarchy is unreadable and
// the auto policy must fail closed.
static inline bool llama_cgroup_v2_level_files(
        bool limit_exists,
        bool usage_exists,
        bool at_mountpoint,
        bool & skip_level) {
    skip_level = false;
    if (limit_exists != usage_exists) {
        return false;
    }
    if (limit_exists) {
        return true;
    }
    if (at_mountpoint) {
        skip_level = true;
        return true;
    }
    return false;
}
