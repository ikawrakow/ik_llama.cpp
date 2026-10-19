// Privileged helper for --cpu-power: set/reset/status only.

#include "cpu_power.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s set N    cap all CPUs to N%% (1..100, needs root)\n"
        "  %s reset    restore the hardware maximum (needs root)\n"
        "  %s status   show the current state (no root)\n", argv0, argv0, argv0);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char * cmd = argv[1];

    if (strcmp(cmd, "status") == 0) {
        printf("supported: %s\n", cpu_power_supported() ? "yes" : "no");
        printf("backend:   %s\n", cpu_power_backend_name().c_str());
        printf("temp:      %d C\n", cpu_power_max_temp_c());
        printf("freq:      %lld MHz\n", (long long) cpu_power_busy_freq_mhz({}));
        return 0;
    }

    if (strcmp(cmd, "reset") == 0) {
        const cpu_power_info info = cpu_power_apply({}, { 100, false });
        if (!info.applied) {
            fprintf(stderr, "reset failed: %s\n", info.message.c_str());
            return 1;
        }
        printf("reset: %lld MHz on %d CPUs (%s)\n",
               (long long) info.target_mhz, info.n_cpus, info.backend.c_str());
        return 0;
    }

    if (strcmp(cmd, "set") == 0) {
        if (argc != 3) {
            usage(argv[0]);
            return 1;
        }
        errno = 0;
        char * end = nullptr;
        const long value = strtol(argv[2], &end, 10);
        if (errno != 0 || end == argv[2] || *end != '\0' || value < 1 || value > 100) {
            fprintf(stderr, "invalid value: %s\n", argv[2]);
            return 1;
        }
        const cpu_power_info info = cpu_power_apply({}, { (int) value, false });
        if (!info.applied) {
            fprintf(stderr, "set failed: %s\n", info.message.c_str());
            return 1;
        }
        printf("%ld%%: %lld MHz on %d CPUs (%s)\n",
               value, (long long) info.target_mhz, info.n_cpus, info.backend.c_str());
        return 0;
    }

    usage(argv[0]);
    return 1;
}
