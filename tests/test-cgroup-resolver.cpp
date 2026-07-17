#include "../src/llama-cgroup-resolver.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <sstream>

int main() {
    {
        std::istringstream memberships_file("0::/tenant/workload\n");
        std::istringstream mounts_file("36 25 0:32 /tenant /custom rw,relatime - cgroup2 cgroup rw\n");
        std::vector<llama_cgroup_membership> memberships;
        std::vector<llama_cgroup_mount> mounts;
        std::vector<llama_resolved_cgroup_mount> resolved;
        assert(llama_parse_cgroup_memberships(memberships_file, memberships));
        assert(llama_parse_cgroup_mounts(mounts_file, mounts));
        assert(memberships.size() == 1 && memberships[0].v2);
        assert(llama_resolve_cgroup_mounts(memberships[0], mounts, resolved));
        assert(resolved.size() == 1);
        assert(resolved[0].mountpoint == "/custom");
        assert(resolved[0].path == "/custom/workload");
    }

    {
        std::istringstream memberships_file("29:cpu,memory:/docker root/service\n");
        std::istringstream mounts_file("35 25 0:31 /docker\\040root /cg\\040memory rw,relatime - cgroup cgroup rw,memory\n");
        std::vector<llama_cgroup_membership> memberships;
        std::vector<llama_cgroup_mount> mounts;
        std::vector<llama_resolved_cgroup_mount> resolved;
        assert(llama_parse_cgroup_memberships(memberships_file, memberships));
        assert(llama_parse_cgroup_mounts(mounts_file, mounts));
        assert(memberships.size() == 1 && !memberships[0].v2);
        assert(llama_resolve_cgroup_mounts(memberships[0], mounts, resolved));
        assert(resolved.size() == 1);
        assert(resolved[0].path == "/cg memory/service");
    }

    {
        // Namespace-relative membership: it does not share the host mount root.
        const llama_cgroup_membership membership = { true, "/slice" };
        const std::vector<llama_cgroup_mount> mounts = { { true, "/host-root", "/ns-cgroup" } };
        std::vector<llama_resolved_cgroup_mount> resolved;
        assert(llama_resolve_cgroup_mounts(membership, mounts, resolved));
        assert(resolved.size() == 1 && resolved[0].path == "/ns-cgroup/slice");
    }

    {
        std::string parent;
        assert(llama_cgroup_parent_path("/slice", "/", parent));
        assert(parent == "/");
    }

    {
        std::istringstream malformed_membership("0::relative\n");
        std::istringstream malformed_mount("36 25 0:32 / /sys/fs/cgroup rw cgroup2 cgroup rw\n");
        std::vector<llama_cgroup_membership> memberships;
        std::vector<llama_cgroup_mount> mounts;
        assert(!llama_parse_cgroup_memberships(malformed_membership, memberships));
        assert(!llama_parse_cgroup_mounts(malformed_mount, mounts));
    }
    return 0;
}
