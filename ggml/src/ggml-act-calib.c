#include "ggml-act-calib.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// A tiny, allocation-light calibration helper.  It is only active when
// IK_ACT_CALIB is set and is fed from the single thread that owns the
// activation rows, so it needs no locking.

#define GGML_ACT_CALIB_BINS 1024
#define GGML_ACT_CALIB_LOG2_MIN (-40.0f)
#define GGML_ACT_CALIB_LOG2_MAX ( 24.0f)

struct ggml_act_calib_entry {
    char     name[64];
    int      expert;
    uint64_t count;
    double   sum;
    float    max;
    uint32_t bins[GGML_ACT_CALIB_BINS];
};

static struct ggml_act_calib_entry * g_entries;
static size_t                        g_n_entries;
static size_t                        g_cap_entries;

static size_t * g_hash;
static size_t   g_hash_cap;

static char g_path[1024];
static int  g_quantile = 999; // per mille; 1001 means "use max"
static int  g_initialized;
static int  g_enabled;

static const float g_log2_step = (GGML_ACT_CALIB_LOG2_MAX - GGML_ACT_CALIB_LOG2_MIN)/GGML_ACT_CALIB_BINS;

static int ggml_act_calib_bin(float v) {
    if (!(v > 0.0f)) {
        return 0;
    }
    const float l = log2f(v);
    if (l <= GGML_ACT_CALIB_LOG2_MIN) {
        return 0;
    }
    if (l >= GGML_ACT_CALIB_LOG2_MAX) {
        return GGML_ACT_CALIB_BINS - 1;
    }
    int bin = (int) ((l - GGML_ACT_CALIB_LOG2_MIN)/g_log2_step);
    if (bin < 0) bin = 0;
    if (bin >= GGML_ACT_CALIB_BINS) bin = GGML_ACT_CALIB_BINS - 1;
    return bin;
}

static float ggml_act_calib_bin_value(int bin) {
    // upper edge of the bin: conservative for a clamp scale
    return exp2f(GGML_ACT_CALIB_LOG2_MIN + (bin + 1)*g_log2_step);
}

static uint64_t ggml_act_calib_hash(const char * name, int expert) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char * p = (const unsigned char *) name; *p; ++p) {
        h ^= *p;
        h *= 1099511628211ULL;
    }
    h ^= (uint32_t) expert;
    h *= 1099511628211ULL;
    return h;
}

static void ggml_act_calib_rehash(size_t cap) {
    free(g_hash);
    g_hash = (size_t *) calloc(cap, sizeof(size_t));
    g_hash_cap = cap;
    if (!g_hash) {
        return;
    }
    for (size_t i = 0; i < g_n_entries; ++i) {
        size_t slot = (size_t) (ggml_act_calib_hash(g_entries[i].name, g_entries[i].expert) & (cap - 1));
        while (g_hash[slot]) {
            slot = (slot + 1) & (cap - 1);
        }
        g_hash[slot] = i + 1;
    }
}

static struct ggml_act_calib_entry * ggml_act_calib_entry(const char * name, int expert) {
    if (!g_enabled) {
        return NULL;
    }
    if (!g_hash_cap || (g_n_entries + 1)*10 >= g_hash_cap*7) {
        size_t cap = 64;
        while (cap < 2*(g_n_entries + 1)) cap <<= 1;
        ggml_act_calib_rehash(cap);
    }
    if (!g_hash) {
        return NULL;
    }
    size_t slot = (size_t) (ggml_act_calib_hash(name, expert) & (g_hash_cap - 1));
    while (g_hash[slot]) {
        struct ggml_act_calib_entry * e = &g_entries[g_hash[slot] - 1];
        if (e->expert == expert && strncmp(e->name, name, sizeof(e->name) - 1) == 0) {
            return e;
        }
        slot = (slot + 1) & (g_hash_cap - 1);
    }
    if (g_n_entries == g_cap_entries) {
        size_t cap = g_cap_entries ? 2*g_cap_entries : 64;
        void * p = realloc(g_entries, cap*sizeof(*g_entries));
        if (!p) {
            return NULL;
        }
        g_entries = (struct ggml_act_calib_entry *) p;
        g_cap_entries = cap;
    }
    struct ggml_act_calib_entry * e = &g_entries[g_n_entries];
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->expert = expert;
    g_hash[slot] = g_n_entries + 1;
    ++g_n_entries;
    return e;
}

static float ggml_act_calib_quantile(const struct ggml_act_calib_entry * e, double q) {
    if (!e->count) {
        return 0.0f;
    }
    const uint64_t target = (uint64_t) ceil(q*e->count);
    uint64_t acc = 0;
    for (int i = 0; i < GGML_ACT_CALIB_BINS; ++i) {
        acc += e->bins[i];
        if (acc >= target) {
            return ggml_act_calib_bin_value(i);
        }
    }
    return e->max;
}

static int ggml_act_calib_cmp(const void * a, const void * b) {
    const struct ggml_act_calib_entry * ea = *(const struct ggml_act_calib_entry * const *) a;
    const struct ggml_act_calib_entry * eb = *(const struct ggml_act_calib_entry * const *) b;
    const int c = strcmp(ea->name, eb->name);
    if (c != 0) {
        return c;
    }
    return ea->expert < eb->expert ? -1 : ea->expert > eb->expert ? 1 : 0;
}

static void ggml_act_calib_finish(void) {
    if (!g_enabled || !g_n_entries) {
        return;
    }
    struct ggml_act_calib_entry ** sorted = (struct ggml_act_calib_entry **) malloc(g_n_entries*sizeof(*sorted));
    if (!sorted) {
        return;
    }
    for (size_t i = 0; i < g_n_entries; ++i) {
        sorted[i] = &g_entries[i];
    }
    qsort(sorted, g_n_entries, sizeof(*sorted), ggml_act_calib_cmp);

    FILE * f = fopen(g_path, "w");
    if (!f) {
        fprintf(stderr, "ggml-act-calib: cannot write '%s'\n", g_path);
        free(sorted);
        return;
    }
    fprintf(f, "# tensor expert scale count median p99 p999 max mean\n");
    fprintf(f, "# scale is the Q8 delta (amax/127); median/p99/p999/max are raw amax values\n");
    for (size_t i = 0; i < g_n_entries; ++i) {
        const struct ggml_act_calib_entry * e = sorted[i];
        const float median = ggml_act_calib_quantile(e, 0.5);
        const float p99    = ggml_act_calib_quantile(e, 0.99);
        const float p999   = ggml_act_calib_quantile(e, 0.999);
        const float amax   = g_quantile > 1000 ? e->max
                           : g_quantile == 500  ? median
                           : g_quantile == 990  ? p99
                           :                      p999;
        const float scale  = amax / 127.0f;
        const double mean  = e->count ? e->sum/e->count : 0.0;
        fprintf(f, "%s %d %.9g %llu %.9g %.9g %.9g %.9g %.9g\n",
                e->name, e->expert, (double) scale, (unsigned long long) e->count,
                (double) median, (double) p99, (double) p999, (double) e->max, mean);
    }
    fclose(f);
    free(sorted);
    fprintf(stderr, "ggml-act-calib: wrote %zu entries to '%s'\n", g_n_entries, g_path);
}

bool ggml_act_calib_enabled(void) {
    if (!g_initialized) {
        g_initialized = 1;
        const char * path = getenv("IK_ACT_CALIB");
        if (path && *path) {
            snprintf(g_path, sizeof(g_path), "%s", path);
            const char * q = getenv("IK_ACT_CALIB_QUANTILE");
            if (q) {
                if (strcmp(q, "max") == 0) {
                    g_quantile = 1001;
                } else if (strcmp(q, "median") == 0) {
                    g_quantile = 500;
                } else if (strcmp(q, "p99") == 0) {
                    g_quantile = 990;
                } else if (strcmp(q, "p999") == 0) {
                    g_quantile = 999;
                }
            }
            g_enabled = 1;
            atexit(ggml_act_calib_finish);
        }
    }
    return g_enabled != 0;
}

void ggml_act_calib_observe_value(const char * name, int expert, float amax) {
    if (!ggml_act_calib_enabled() || !name || !*name) {
        return;
    }
    if (!(amax >= 0.0f)) {
        amax = 0.0f;
    }
    struct ggml_act_calib_entry * e = ggml_act_calib_entry(name, expert);
    if (!e) {
        return;
    }
    ++e->count;
    e->sum += (double) amax;
    if (amax > e->max) {
        e->max = amax;
    }
    ++e->bins[ggml_act_calib_bin(amax)];
}
