#pragma once

// CPU frequency cap (Linux cpufreq): per-CPU on the affinity mask, restored on exit/signals.
// SIGKILL cannot be caught; use llama-cpu-power reset.

#include <cstdint>
#include <string>
#include <vector>

struct cpu_power_config {
    int  percent = 100;          // 1..100, 100 = hardware maximum (reset)
    bool restore_on_exit = true; // false = keep the cap after this process exits
};

struct cpu_power_info {
    bool        applied   = false;
    int         n_cpus    = 0;
    int64_t     target_mhz = 0;  // highest target frequency among the selected CPUs
    std::string backend;         // "cpufreq" or empty
    std::string message;         // reason when not applied
};

// Target frequency for one CPU (pure, for tests).
int64_t cpu_power_target_freq(int64_t cpuinfo_max_freq, int64_t scaling_min_freq, int percent);

// Apply to cpus (empty = all online); restores any previous cap first.
cpu_power_info cpu_power_apply(const std::vector<int32_t> & cpus, const cpu_power_config & cfg);

// Restore saved frequencies; idempotent.
void cpu_power_restore();

// Chain on top of handlers installed later (see --cpu-power).
void cpu_power_install_signal_handlers();

bool        cpu_power_supported();
std::string cpu_power_backend_name();

// Read-only telemetry for --cpu-power-sweep (-1 / 0 = unavailable).
int     cpu_power_max_temp_c();
int64_t cpu_power_busy_freq_mhz(const std::vector<int32_t> & cpus); // mean of cores above 800 MHz
int64_t cpu_power_max_freq_mhz(const std::vector<int32_t> & cpus);  // highest current core frequency
