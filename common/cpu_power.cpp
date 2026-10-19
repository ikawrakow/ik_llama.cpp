#include "cpu_power.h"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

struct saved_freq {
    int32_t cpu;
    int64_t freq;
};

std::mutex              g_mutex;
std::vector<saved_freq> g_saved;
bool                    g_applied = false;
int                     g_percent = 100;
bool                    g_restore_on_exit = true;

#if defined(__linux__)

// Async-signal-safe restore table (SIGINT/SIGTERM/SIGHUP).
constexpr int RESTORE_MAX_ENTRIES = 512;
constexpr int RESTORE_PATH_MAX    = 128;

struct restore_entry {
    char       path[RESTORE_PATH_MAX];
    long long  value;   // restore on exit
    long long  target;  // re-apply if the process continues
};

restore_entry           g_restore[RESTORE_MAX_ENTRIES];
volatile sig_atomic_t   g_restore_count = 0;

struct saved_sigaction {
    struct sigaction act;
    bool             valid = false;
};

saved_sigaction g_prev_sigint;
saved_sigaction g_prev_sigterm;
saved_sigaction g_prev_sighup;
saved_sigaction g_prev_sigsegv;
saved_sigaction g_prev_sigabrt;
saved_sigaction g_prev_sigill;
saved_sigaction g_prev_sigbus;

saved_sigaction * saved_for(int sig) {
    switch (sig) {
        case SIGINT:  return &g_prev_sigint;
        case SIGTERM: return &g_prev_sigterm;
        case SIGHUP:  return &g_prev_sighup;
        case SIGSEGV: return &g_prev_sigsegv;
        case SIGABRT: return &g_prev_sigabrt;
        case SIGILL:  return &g_prev_sigill;
        case SIGBUS:  return &g_prev_sigbus;
        default:      return nullptr;
    }
}

// Alt stack for stack-overflow SIGSEGV (main thread only).
alignas(16) char g_alt_stack[64 * 1024];
stack_t g_prev_alt_stack;
bool    g_alt_stack_installed = false;

void install_alt_stack() {
    if (g_alt_stack_installed) {
        return;
    }
    stack_t ss{};
    ss.ss_sp = g_alt_stack;
    ss.ss_size = sizeof(g_alt_stack);
    ss.ss_flags = 0;
    if (sigaltstack(&ss, &g_prev_alt_stack) == 0) {
        g_alt_stack_installed = true;
    }
}

void uninstall_alt_stack() {
    if (!g_alt_stack_installed) {
        return;
    }
    sigaltstack(&g_prev_alt_stack, nullptr);
    g_alt_stack_installed = false;
}

bool    g_saved_max_perf_valid = false;
int64_t g_saved_max_perf_value = 0;

void add_restore_entry(int & n, const std::string & path, long long value, long long target) {
    if (n >= RESTORE_MAX_ENTRIES) {
        return;
    }
    std::snprintf(g_restore[n].path, RESTORE_PATH_MAX, "%s", path.c_str());
    g_restore[n].value  = value;
    g_restore[n].target = target;
    ++n;
}

void async_write_value(const char * path, long long value) {
    const int fd = open(path, O_WRONLY);
    if (fd < 0) {
        return;
    }
    char buf[32];
    int  len = 0;
    unsigned long long v = value < 0 ? 0ull : (unsigned long long) value;
    char tmp[32];
    int  n = 0;
    do {
        tmp[n++] = (char) ('0' + (v % 10));
        v /= 10;
    } while (v != 0 && n < (int) sizeof(tmp));
    while (n > 0) {
        buf[len++] = tmp[--n];
    }
    if (len == 0) {
        buf[len++] = '0';
    }
    const ssize_t written = write(fd, buf, (size_t) len);
    (void) written;
    close(fd);
}

void cpu_power_signal_handler(int sig, siginfo_t * info, void * uctx) {
    const int n = g_restore_count;
    std::atomic_thread_fence(std::memory_order_acquire);

    for (int i = 0; i < n; ++i) {
        async_write_value(g_restore[i].path, g_restore[i].value);
    }

    // Chain; re-apply if the old handler returns (interactive Ctrl+C).
    saved_sigaction * saved = saved_for(sig);

    auto reapply = [n]() {
        for (int i = 0; i < n; ++i) {
            async_write_value(g_restore[i].path, g_restore[i].target);
        }
    };

    if (saved && saved->valid) {
        if ((saved->act.sa_flags & SA_SIGINFO) && saved->act.sa_sigaction) {
            saved->act.sa_sigaction(sig, info, uctx);
            reapply();
            return;
        }
        if (saved->act.sa_handler == SIG_IGN) {
            reapply();
            return;
        }
        if (saved->act.sa_handler != SIG_DFL) {
            saved->act.sa_handler(sig);
            reapply();
            return;
        }
    }

    // No previous handler: re-raise with the default disposition.
    struct sigaction dfl;
    std::memset(&dfl, 0, sizeof(dfl));
    dfl.sa_handler = SIG_DFL;
    sigemptyset(&dfl.sa_mask);
    sigaction(sig, &dfl, nullptr);
    raise(sig);
    _exit(128 + sig);
}

void install_one(int sig, saved_sigaction & saved) {
    struct sigaction cur;
    std::memset(&cur, 0, sizeof(cur));
    if (sigaction(sig, nullptr, &cur) != 0) {
        return;
    }
    if ((cur.sa_flags & SA_SIGINFO) && cur.sa_sigaction == cpu_power_signal_handler) {
        return; // already ours
    }
    saved.act   = cur;
    saved.valid = true;

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = cpu_power_signal_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    sigaction(sig, &sa, nullptr);
}

void uninstall_one(int sig, saved_sigaction & saved) {
    if (saved.valid) {
        sigaction(sig, &saved.act, nullptr);
        saved.valid = false;
    }
}

#endif // __linux__

#if defined(__linux__)

bool read_int_file(const std::string & path, int64_t & value) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    f >> value;
    return !f.fail();
}

bool write_int_file(const std::string & path, int64_t value) {
    std::ofstream f(path);
    if (!f) {
        return false;
    }
    f << value;
    f.flush();
    return !f.fail();
}

std::string cpu_path(int32_t cpu, const char * leaf) {
    return "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/" + leaf;
}

std::vector<int32_t> online_cpus() {
    std::vector<int32_t> cpus;

    std::ifstream f("/sys/devices/system/cpu/online");
    if (f) {
        std::string s;
        std::getline(f, s);
        size_t i = 0;
        while (i < s.size()) {
            const size_t j = s.find(',', i);
            const std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
            try {
                const size_t dash = part.find('-');
                if (dash == std::string::npos) {
                    cpus.push_back(std::stoi(part));
                } else {
                    const int a = std::stoi(part.substr(0, dash));
                    const int b = std::stoi(part.substr(dash + 1));
                    for (int c = a; c <= b; ++c) {
                        cpus.push_back(c);
                    }
                }
            } catch (...) {
                cpus.clear();
                break;
            }
            if (j == std::string::npos) {
                break;
            }
            i = j + 1;
        }
    }

    if (cpus.empty()) {
        const long n = sysconf(_SC_NPROCESSORS_CONF);
        for (long c = 0; c < n; ++c) {
            cpus.push_back((int32_t) c);
        }
    }

    return cpus;
}

void restore_locked() {
    // Invalidate the table before restoring.
    g_restore_count = 0;
    uninstall_one(SIGINT,  g_prev_sigint);
    uninstall_one(SIGTERM, g_prev_sigterm);
    uninstall_one(SIGHUP,  g_prev_sighup);
    uninstall_one(SIGSEGV, g_prev_sigsegv);
    uninstall_one(SIGABRT, g_prev_sigabrt);
    uninstall_one(SIGILL,  g_prev_sigill);
    uninstall_one(SIGBUS,  g_prev_sigbus);
    uninstall_alt_stack();

    for (auto it = g_saved.rbegin(); it != g_saved.rend(); ++it) {
        write_int_file(cpu_path(it->cpu, "scaling_max_freq"), it->freq);
    }
    if (g_saved_max_perf_valid) {
        write_int_file("/sys/devices/system/cpu/intel_pstate/max_perf_pct", g_saved_max_perf_value);
        g_saved_max_perf_valid = false;
    }
    g_saved.clear();
    g_applied = false;
    g_percent = 100;
}

#endif // __linux__

struct exit_guard {
    ~exit_guard() {
        if (g_restore_on_exit) {
            cpu_power_restore();
        }
    }
};

exit_guard g_exit_guard;

} // namespace

int64_t cpu_power_target_freq(int64_t cpuinfo_max_freq, int64_t scaling_min_freq, int percent) {
    if (percent >= 100) {
        return cpuinfo_max_freq;
    }
    if (percent <= 0) {
        return scaling_min_freq;
    }
    const int64_t target = cpuinfo_max_freq * percent / 100;
    return std::max(scaling_min_freq, std::min(cpuinfo_max_freq, target));
}

bool cpu_power_supported() {
#if defined(__linux__)
    for (const int32_t cpu : online_cpus()) {
        int64_t value = 0;
        if (read_int_file(cpu_path(cpu, "cpuinfo_max_freq"), value)) {
            return true;
        }
    }
    return false;
#else
    return false;
#endif
}

std::string cpu_power_backend_name() {
    return cpu_power_supported() ? "cpufreq" : "none";
}

int cpu_power_max_temp_c() {
#if defined(__linux__)
    // Prefer CPU package sensors (coretemp/TCPU); fall back to the hottest zone.
    int best = -1;

    for (int h = 0; h < 16; ++h) {
        const std::string hw = "/sys/class/hwmon/hwmon" + std::to_string(h);
        std::ifstream name_file(hw + "/name");
        std::string name;
        if (!(name_file >> name)) {
            continue;
        }
        if (name != "coretemp" && name != "k10temp" && name != "zenpower") {
            continue;
        }
        for (int t = 1; t <= 64; ++t) {
            int64_t value = 0;
            if (read_int_file(hw + "/temp" + std::to_string(t) + "_input", value)) {
                const int temp = (int) (value / 1000);
                if (temp > best && temp < 150) {
                    best = temp;
                }
            }
        }
    }
    if (best >= 0) {
        return best;
    }

    for (int i = 0; i < 32; ++i) {
        const std::string base = "/sys/class/thermal/thermal_zone" + std::to_string(i);
        std::ifstream type_file(base + "/type");
        std::string type;
        if (!(type_file >> type)) {
            continue;
        }
        if (type.find("x86_pkg_temp") == std::string::npos &&
            type.find("TCPU")         == std::string::npos &&
            type.find("cpu")          == std::string::npos &&
            type.find("CPU")          == std::string::npos) {
            continue;
        }
        int64_t value = 0;
        if (read_int_file(base + "/temp", value)) {
            const int temp = (int) (value / 1000);
            if (temp > best && temp < 150) {
                best = temp;
            }
        }
    }
    if (best >= 0) {
        return best;
    }

    // No CPU sensor found: use the hottest zone.
    for (int i = 0; i < 32; ++i) {
        int64_t value = 0;
        if (read_int_file("/sys/class/thermal/thermal_zone" + std::to_string(i) + "/temp", value)) {
            const int temp = (int) (value / 1000);
            if (temp > best && temp < 150) {
                best = temp;
            }
        }
    }
    return best;
#else
    return -1;
#endif
}

int64_t cpu_power_busy_freq_mhz(const std::vector<int32_t> & cpus) {
#if defined(__linux__)
    const std::vector<int32_t> targets = cpus.empty() ? online_cpus() : cpus;
    int64_t sum = 0;
    int     n   = 0;
    for (const int32_t cpu : targets) {
        int64_t freq = 0;
        if (read_int_file(cpu_path(cpu, "scaling_cur_freq"), freq) && freq >= 800000) {
            sum += freq;
            ++n;
        }
    }
    return n > 0 ? sum / n / 1000 : 0;
#else
    (void) cpus;
    return 0;
#endif
}

int64_t cpu_power_max_freq_mhz(const std::vector<int32_t> & cpus) {
#if defined(__linux__)
    const std::vector<int32_t> targets = cpus.empty() ? online_cpus() : cpus;
    int64_t max_freq = 0;
    for (const int32_t cpu : targets) {
        int64_t freq = 0;
        if (read_int_file(cpu_path(cpu, "scaling_cur_freq"), freq) && freq > max_freq) {
            max_freq = freq;
        }
    }
    return max_freq / 1000;
#else
    (void) cpus;
    return 0;
#endif
}

void cpu_power_restore() {
    std::lock_guard<std::mutex> lock(g_mutex);
#if defined(__linux__)
    restore_locked();
#else
    g_saved.clear();
    g_applied = false;
    g_percent = 100;
#endif
}

cpu_power_info cpu_power_apply(const std::vector<int32_t> & cpus, const cpu_power_config & cfg) {
    cpu_power_info info;
    info.backend = cpu_power_backend_name();

    std::lock_guard<std::mutex> lock(g_mutex);
    g_restore_on_exit = cfg.restore_on_exit;

#if !defined(__linux__)
    (void) cpus;
    (void) cfg;
    info.message = "CPU frequency capping is not supported on this platform";
    return info;
#else
    restore_locked();

    if (cfg.percent <= 0) {
        info.message = "disabled";
        return info;
    }

    const int percent = std::min(100, cfg.percent);
    if (percent == 100) {
        info.message = "reset to hardware maximum";
    }

    const std::vector<int32_t> targets = cpus.empty() ? online_cpus() : cpus;

    int     applied           = 0;
    int64_t representative    = 0;
    bool    permission_denied = false;
    int     n_entries         = 0;

    // Do not let a signal replay a half-built table.
    g_restore_count = 0;

    for (const int32_t cpu : targets) {
        const std::string path_max = cpu_path(cpu, "cpuinfo_max_freq");
        const std::string path_min = cpu_path(cpu, "scaling_min_freq");
        const std::string path_cur = cpu_path(cpu, "scaling_max_freq");

        int64_t max_freq = 0;
        int64_t min_freq = 0;
        int64_t cur_freq = 0;
        if (!read_int_file(path_max, max_freq) ||
            !read_int_file(path_cur, cur_freq)) {
            continue;
        }
        if (!read_int_file(path_min, min_freq)) {
            min_freq = 0;
        }

        const int64_t target = cpu_power_target_freq(max_freq, min_freq, percent);
        if (target == cur_freq) {
            ++applied;
            representative = std::max(representative, target);
            continue;
        }

        if (access(path_cur.c_str(), W_OK) != 0) {
            permission_denied = true;
            continue;
        }

        g_saved.push_back({ cpu, cur_freq });
        add_restore_entry(n_entries, path_cur, cur_freq, target);
        if (!write_int_file(path_cur, target)) {
            g_saved.pop_back();
            --n_entries;
            permission_denied = true;
            continue;
        }

        // Trust a successful write: intel_pstate read-back is not immediate.
        ++applied;
        representative = std::max(representative, target);
    }

    if (applied == 0) {
        // Fallback: intel_pstate global percentage.
        const std::string path_pct = "/sys/devices/system/cpu/intel_pstate/max_perf_pct";
        int64_t cur_pct = 0;
        if (read_int_file(path_pct, cur_pct)) {
            if (access(path_pct.c_str(), W_OK) != 0) {
                permission_denied = true;
            } else if (write_int_file(path_pct, percent)) {
                g_saved_max_perf_valid = true;
                g_saved_max_perf_value = cur_pct;
                add_restore_entry(n_entries, path_pct, cur_pct, percent);
                ++applied;
                info.backend = "intel_pstate";
                int64_t max_freq = 0;
                if (!targets.empty()) {
                    read_int_file(cpu_path(targets.front(), "cpuinfo_max_freq"), max_freq);
                }
                representative = cpu_power_target_freq(max_freq, 0, percent);
            } else {
                permission_denied = true;
            }
        }
    }

    if (applied == 0) {
        info.message = permission_denied
            ? "permission denied (run with sudo, or use the systemd helper)"
            : "no writable cpufreq interface found";
        return info;
    }

    g_applied = true;
    g_percent = percent;

    // Publish the table and install handlers.
    std::atomic_thread_fence(std::memory_order_release);
    g_restore_count = n_entries;
    cpu_power_install_signal_handlers();

    info.applied    = true;
    info.n_cpus     = applied;
    info.target_mhz = representative / 1000;
    if (permission_denied) {
        info.message = "partially applied (permission denied on some CPUs)";
    }
    return info;
#endif
}

void cpu_power_install_signal_handlers() {
#if defined(__linux__)
    if (!g_restore_on_exit || g_restore_count == 0) {
        return;
    }
    install_alt_stack();
    install_one(SIGINT,  g_prev_sigint);
    install_one(SIGTERM, g_prev_sigterm);
    install_one(SIGHUP,  g_prev_sighup);
    install_one(SIGSEGV, g_prev_sigsegv);
    install_one(SIGABRT, g_prev_sigabrt);
    install_one(SIGILL,  g_prev_sigill);
    install_one(SIGBUS,  g_prev_sigbus);
#endif
}
