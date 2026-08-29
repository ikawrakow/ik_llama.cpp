// kv-cache.cpp — 磁盘 KV 块缓存实现（纯 llama.h 公开 API，不依赖 common）
//
// 结构：
//   disk_kv 命名空间 —— 不依赖 kv_cache 实例的块级函数（外部 ctx + 任意 seq_id），
//                        供 llama-server 等复用（只做命中加载/落盘/清理，不做 decode）
//   kv_cache 类       —— 保留原接口，内部调 disk_kv（主序列固定 seq 0，临时序列固定 1）

#include "kv-cache.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <stdexcept>
#include <chrono>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

// ---------------------------------------------------------------------------
// 文件工具（disk_kv 内部使用，不进公共头）
// ---------------------------------------------------------------------------
namespace {

bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}

void touch_file(const std::string & path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), FILE_WRITE_ATTRIBUTES,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL, OPEN_EXISTING, 0, NULL);
    if (h != INVALID_HANDLE_VALUE) {
        SYSTEMTIME st;
        GetSystemTime(&st);
        FILETIME ft;
        SystemTimeToFileTime(&st, &ft);
        SetFileTime(h, NULL, NULL, &ft);
        CloseHandle(h);
    }
#else
    (void) path;
#endif
}

void remove_file(const std::string & path) {
#ifdef _WIN32
    DeleteFileA(path.c_str());
#else
    std::remove(path.c_str());
#endif
}

#if !defined(_WIN32)
// --- 公共域 SHA-256（Brad Conte 的 crypto-algorithms，public domain）---
// Windows 走 BCrypt（见 disk_kv::sha256_hex）；非 Windows 用这份内嵌实现，
// 不依赖 openssl。仅用于缓存块文件命名 / 指纹，不参与密码学安全场景。
struct sha256_state {
    uint8_t  data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
};

static const uint32_t sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static void sha256_transform(sha256_state * s, const uint8_t * block) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint32_t) block[i * 4] << 24) | ((uint32_t) block[i * 4 + 1] << 16) |
               ((uint32_t) block[i * 4 + 2] << 8) | (uint32_t) block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = ((w[i-15] >> 7) | (w[i-15] << 25)) ^
                            ((w[i-15] >> 18) | (w[i-15] << 14)) ^ (w[i-15] >> 3);
        const uint32_t s1 = ((w[i-2] >> 17) | (w[i-2] << 15)) ^
                            ((w[i-2] >> 19) | (w[i-2] << 13)) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = s->state[0], b = s->state[1], c = s->state[2], d = s->state[3];
    uint32_t e = s->state[4], f = s->state[5], g = s->state[6], h = s->state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7));
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + S1 + ch + sha256_k[i] + w[i];
        const uint32_t S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10));
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    s->state[0] += a; s->state[1] += b; s->state[2] += c; s->state[3] += d;
    s->state[4] += e; s->state[5] += f; s->state[6] += g; s->state[7] += h;
}

static void sha256_init(sha256_state * s) {
    s->datalen = 0;
    s->bitlen  = 0;
    s->state[0] = 0x6a09e667; s->state[1] = 0xbb67ae85; s->state[2] = 0x3c6ef372; s->state[3] = 0xa54ff53a;
    s->state[4] = 0x510e527f; s->state[5] = 0x9b05688c; s->state[6] = 0x1f83d9ab; s->state[7] = 0x5be0cd19;
}

static void sha256_update(sha256_state * s, const uint8_t * data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        s->data[s->datalen++] = data[i];
        if (s->datalen == 64) {
            sha256_transform(s, s->data);
            s->bitlen += 512;
            s->datalen = 0;
        }
    }
}

static void sha256_final(sha256_state * s, uint8_t * hash) {
    const uint32_t i0 = s->datalen;
    if (s->datalen < 56) {
        s->data[s->datalen++] = 0x80;
        while (s->datalen < 56) s->data[s->datalen++] = 0x00;
    } else {
        s->data[s->datalen++] = 0x80;
        while (s->datalen < 64) s->data[s->datalen++] = 0x00;
        sha256_transform(s, s->data);
        memset(s->data, 0, 56);
    }
    const uint64_t bitlen = s->bitlen + i0 * 8;
    for (int j = 7; j >= 0; --j) {
        s->data[56 + j] = (uint8_t) (bitlen >> (8 * (7 - j)));
    }
    sha256_transform(s, s->data);
    for (uint32_t i = 0; i < 8; ++i) {
        hash[i * 4]     = (uint8_t) (s->state[i] >> 24);
        hash[i * 4 + 1] = (uint8_t) (s->state[i] >> 16);
        hash[i * 4 + 2] = (uint8_t) (s->state[i] >> 8);
        hash[i * 4 + 3] = (uint8_t) (s->state[i]);
    }
}
#endif // !defined(_WIN32)

// 检查序列状态文件头部是否结构完好（magic/version/token 数与块大小一致）。
// 用于区分"损坏文件"（应删除）与"环境失败"（如 KV cache 满导致恢复失败，
// 文件本身有效，不应删除）。
static bool valid_seq_state_header(const std::string & file, size_t expected_tokens) {
    FILE * f = fopen(file.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0, n_tok = 0;
    const bool ok = fread(&magic, 4, 1, f) == 1 &&
                    fread(&version, 4, 1, f) == 1 &&
                    fread(&n_tok, 4, 1, f) == 1;
    long sz = 0;
    if (ok) { fseek(f, 0, SEEK_END); sz = ftell(f); }
    fclose(f);
    if (!ok) return false;
    if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) return false;
    if (n_tok != expected_tokens) return false;
    // 合法文件 = 头部(12B) + token 数据 + 非空的状态数据；否则视为截断/损坏
    if ((long) (sizeof(uint32_t) * 3 + sizeof(llama_token) * n_tok) >= sz) return false;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// sha256（Windows BCrypt）
// ---------------------------------------------------------------------------
std::string disk_kv::sha256_hex(const void * data, size_t len) {
#ifdef _WIN32
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE h  = nullptr;
    if (!BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0))) {
        return {};
    }
    if (!BCRYPT_SUCCESS(BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0))) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return {};
    }
    // BCryptHashData 一次调用上限 ~2^32-1 字节；这里 len 远小于此
    if (!BCRYPT_SUCCESS(BCryptHashData(h, (PUCHAR) data, (ULONG) len, 0))) {
        BCryptDestroyHash(h);
        BCryptCloseAlgorithmProvider(alg, 0);
        return {};
    }
    UCHAR out[32];
    if (!BCRYPT_SUCCESS(BCryptFinishHash(h, out, sizeof(out), 0))) {
        BCryptDestroyHash(h);
        BCryptCloseAlgorithmProvider(alg, 0);
        return {};
    }
    BCryptDestroyHash(h);
    BCryptCloseAlgorithmProvider(alg, 0);

    static const char hex[] = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[i * 2]     = hex[out[i] >> 4];
        s[i * 2 + 1] = hex[out[i] & 0x0f];
    }
    return s;
#else
    sha256_state st;
    sha256_init(&st);
    sha256_update(&st, (const uint8_t *) data, len);
    uint8_t out[32];
    sha256_final(&st, out);

    static const char hex[] = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; ++i) {
        s[i * 2]     = hex[out[i] >> 4];
        s[i * 2 + 1] = hex[out[i] & 0x0f];
    }
    return s;
#endif
}

// ---------------------------------------------------------------------------
// 模型指纹 / 块文件路径
// ---------------------------------------------------------------------------
std::string disk_kv::model_fingerprint(const llama_model * model, const std::string & model_path) {
    std::string fp = sha256_hex(model_path.data(), model_path.size());
    fp += "_";
    fp += std::to_string(llama_n_ctx_train(model));
    fp += "_";
    fp += std::to_string(llama_n_vocab(model));

    // 追加模型文件大小与修改时间：同一路径换权重会改变这两者，
    // 从而让旧缓存块文件失效重算（避免静默加载旧 KV）。
    std::error_code ec;
    const uintmax_t sz = std::filesystem::file_size(model_path, ec);
    if (!ec) {
        fp += "_";
        fp += std::to_string(sz);
    } else {
        ec.clear();
    }
    const auto mtime = std::filesystem::last_write_time(model_path, ec);
    if (!ec) {
        fp += "_";
        fp += std::to_string(mtime.time_since_epoch().count());
    }
    return fp;
}

std::string disk_kv::block_path(const llama_model * model, const std::string & model_path,
                                const std::string & cache_dir,
                                const std::vector<llama_token> & prompt,
                                size_t start, size_t end) {
    const std::string h = sha256_hex(prompt.data() + start, (end - start) * sizeof(llama_token));
    const std::string fname = h + "_" + std::to_string(start) + "_" +
                              model_fingerprint(model, model_path) + ".bin";
    return (std::filesystem::path(cache_dir) / fname).string();
}

// ---------------------------------------------------------------------------
// 命中加载：文件 -> 临时序列 -> seq_cp 合并进 dst_seq（位置从文件恢复，无需搬移）
// ---------------------------------------------------------------------------
bool disk_kv::load_block(llama_context * ctx, llama_seq_id dst_seq, llama_seq_id tmp_seq,
                         const std::string & file, const std::vector<llama_token> & prompt,
                         size_t start, size_t end) {
    if (!file_exists(file)) return false;

    // 结构校验：只有确认损坏（magic/version/token 数不符或截断）才删文件。
    // 结构完好但恢复失败（如 KV cache 满，find_slot 失败）属于环境失败，
    // 保留文件，下次仍有空间时再命中，避免误删有效缓存。
    if (!valid_seq_state_header(file, end - start)) {
        remove_file(file);
        return false;
    }

    std::vector<llama_token> toks(end - start + 64);
    size_t n = 0;
    const size_t loaded = llama_state_seq_load_file(ctx, file.c_str(), tmp_seq,
                                                    toks.data(), (int32_t) toks.size(), &n);
    if (loaded == 0) {
        // 头部完好但恢复失败：大概率是 KV 满等环境原因，不删文件
        fprintf(stderr, "[disk-kv] load_block: file valid but restore failed (KV full?) - keeping %s\n", file.c_str());
        return false;
    }

    // 校验：文件里的 token 必须与请求块完全一致（哈希碰撞兜底）
    if (n != end - start) { remove_file(file); return false; }
    for (size_t i = 0; i < n; ++i) {
        if (toks[i] != prompt[start + i]) { remove_file(file); return false; }
    }

    // 合并进目标序列。hybrid 的 state 通过 seq_cp + kv_cache_update 懒拷贝。
    llama_kv_cache_seq_cp(ctx, tmp_seq, dst_seq, (llama_pos) start, (llama_pos) end);
    llama_kv_cache_update(ctx);
    llama_kv_cache_seq_rm(ctx, tmp_seq, -1, -1);

    touch_file(file); // 更新访问时间作为 LRU 依据
    return true;
}

// ---------------------------------------------------------------------------
// 落盘：src_seq [start,end) -> 临时序列 -> save（save 前必须触发 s_copy 拷 state）
// ---------------------------------------------------------------------------
bool disk_kv::save_block(llama_context * ctx, llama_seq_id src_seq, llama_seq_id tmp_seq,
                         const std::string & file, const std::vector<llama_token> & prompt,
                         size_t start, size_t end) {
    if (file_exists(file)) return true;

    llama_kv_cache_seq_cp(ctx, src_seq, tmp_seq, (llama_pos) start, (llama_pos) end);
    llama_kv_cache_update(ctx); // 触发 hybrid state 从 src 拷到 tmp
    const std::vector<llama_token> seg(prompt.begin() + start, prompt.begin() + end);

    // 原子写：先写 .tmp，成功后再 rename，避免崩溃残留半文件
    const std::string tmp_file = file + ".tmp";
    const size_t saved = llama_state_seq_save_file(ctx, tmp_file.c_str(), tmp_seq, seg.data(), seg.size());
    llama_kv_cache_seq_rm(ctx, tmp_seq, -1, -1);
    if (saved == 0) { remove_file(tmp_file); return false; }

#ifdef _WIN32
    if (!MoveFileExA(tmp_file.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        remove_file(tmp_file);
        fprintf(stderr, "[disk-kv] rename failed: %s\n", file.c_str());
        return false;
    }
#else
    if (rename(tmp_file.c_str(), file.c_str()) != 0) {
        remove_file(tmp_file);
        return false;
    }
#endif
    return true;
}

// ---------------------------------------------------------------------------
// 老化清理：删除超过 max_age_days 未访问（mtime 未刷新）的块文件 + .tmp 残留
// ---------------------------------------------------------------------------
void disk_kv::cleanup_expired(const std::string & cache_dir, int max_age_days) {
    if (max_age_days <= 0) return;

    const auto now = std::filesystem::file_time_type::clock::now();

    std::vector<std::string> expired;
    std::error_code ec;
    std::filesystem::directory_iterator it(cache_dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::filesystem::directory_entry & e = *it;
        const std::string name = e.path().filename().string();
        if (name.size() < 4) continue;
        const std::string ext = name.substr(name.size() - 4);
        if (ext != ".bin" && ext != ".tmp") continue;
        if (e.is_directory(ec)) { ec.clear(); continue; }
        const auto mtime = e.last_write_time(ec);
        if (ec) { ec.clear(); continue; }
        // 注意：不能用 file_time_type::duration(N) —— N 是时钟 tick 数，随平台粒度不同
        // （Windows 100ns / POSIX ns）。用 std::chrono::hours 表达"天数"，比较时自动换算。
        if (now - mtime > std::chrono::hours(24) * max_age_days) {
            expired.push_back(e.path().string());
        }
    }
    if (!expired.empty()) {
        fprintf(stderr, "[disk-kv] removing %zu expired blocks (age > %dd)\n",
                expired.size(), max_age_days);
        for (auto & f : expired) remove_file(f);
    }
}

// ---------------------------------------------------------------------------
// 容量 LRU：超过 max_cache_size_mb 时按 mtime 升序（最旧优先）删 .bin
// ---------------------------------------------------------------------------
void disk_kv::enforce_cache_limit(const std::string & cache_dir, size_t max_cache_size_mb) {
    if (max_cache_size_mb == 0) return;
    const int64_t limit = (int64_t) max_cache_size_mb * 1024 * 1024;

    // 收集目录下所有 .bin（含 mtime / 大小）
    std::vector<std::tuple<int64_t, std::string, int64_t>> files; // mtime, path, size
    std::error_code ec;
    std::filesystem::directory_iterator it(cache_dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::filesystem::directory_entry & e = *it;
        const std::string name = e.path().filename().string();
        if (name.size() < 4 || name.compare(name.size() - 4, 4, ".bin") != 0) continue;
        if (e.is_directory(ec)) { ec.clear(); continue; }
        const auto mtime = e.last_write_time(ec);
        if (ec) { ec.clear(); continue; }
        const int64_t sz = (int64_t) e.file_size(ec);
        if (ec) { ec.clear(); continue; }
        files.emplace_back(mtime.time_since_epoch().count(), e.path().string(), sz);
    }
    if (files.empty()) return;

    std::sort(files.begin(), files.end()); // mtime 升序
    int64_t total = 0;
    for (auto & f : files) total += std::get<2>(f);

    if (total <= limit) return;
    fprintf(stderr, "[disk-kv] cache %.1f MB > limit %zu MB, LRU evicting...\n",
            total / 1048576.0, max_cache_size_mb);
    for (auto & f : files) {
        if (total <= limit) break;
        remove_file(std::get<1>(f));
        total -= std::get<2>(f);
    }
}

// ---------------------------------------------------------------------------
// kv_cache 类实现
// ---------------------------------------------------------------------------
kv_cache::kv_cache(const char * model_path, const kv_cache_config & cfg)
    : cfg_(cfg), model_path_(model_path ? model_path : "") {
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cfg_.n_gpu_layers;
    model_ = llama_model_load_from_file(model_path, mp);
    if (!model_) {
        fprintf(stderr, "[kv-cache] failed to load model: %s\n", model_path);
        throw std::runtime_error("model load failed");
    }

    if (cfg_.n_seq_max < 2) {
        fprintf(stderr, "[kv-cache] warning: n_seq_max=%d < 2, clamping to 2 (hybrid seq state needs >= 2 slots)\n",
                cfg_.n_seq_max);
        cfg_.n_seq_max = 2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = cfg_.n_ctx;
    cp.n_batch   = cfg_.n_ctx;
    cp.n_seq_max = cfg_.n_seq_max; // hybrid qnext state 按 seq_id 索引行，必须给足槽位
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        llama_free_model(model_);
        model_ = nullptr;
        throw std::runtime_error("context init failed");
    }

    // 模型指纹：路径 + 关键尺寸 + 文件大小/修改时间，避免串模型/串权重加载缓存
    fp_ = disk_kv::model_fingerprint(model_, model_path_);

    // 确保缓存目录存在（std::filesystem，跨平台且无 system() 命令注入）
    if (!cfg_.cache_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cfg_.cache_dir, ec);
    }
    disk_kv::cleanup_expired(cfg_.cache_dir, cfg_.max_age_days); // 启动时清掉历史过期块
}

kv_cache::~kv_cache() {
    if (ctx_)   { llama_free(ctx_);   ctx_   = nullptr; }
    if (model_) { llama_free_model(model_); model_ = nullptr; }
    llama_backend_free();
}

// ---------------------------------------------------------------------------
// 批量 decode（纯 prefill，不要求 logits）；主序列固定 seq 0
// ---------------------------------------------------------------------------
bool kv_cache::decode_batch(const std::vector<llama_token> & toks, size_t pos0) {
    if (toks.empty()) return true;
    // 单次 llama_decode 的 token 数不能超过 n_batch（内部 GGML_ASSERT），
    // 这里按 n_batch 切分；llama_decode 内部会再按 n_ubatch 分块处理。
    const uint32_t max_per_decode = llama_n_batch(ctx_);
    size_t i0 = 0;
    while (i0 < toks.size()) {
        const size_t i1 = std::min(i0 + max_per_decode, toks.size());
        const int32_t n = (int32_t) (i1 - i0);

        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (size_t i = i0; i < i1; ++i) {
            batch.token[i - i0]     = toks[i];
            batch.pos[i - i0]       = (llama_pos)(pos0 + i);
            batch.n_seq_id[i - i0]  = 1;
            batch.seq_id[i - i0][0] = 0;
            batch.logits[i - i0]    = 0;
        }
        const int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "[kv-cache] llama_decode failed rc=%d\n", rc);
            return false;
        }
        i0 = i1;
    }
    return true;
}

// ---------------------------------------------------------------------------
// prefill：逐块命中 / 重算并落盘（主序列固定 seq 0，临时序列固定 1）
// ---------------------------------------------------------------------------
size_t kv_cache::prefill(const std::vector<llama_token> & prompt) {
    llama_kv_cache_seq_rm(ctx_, 0, -1, -1); // 清空主序列，模拟新请求
    hit_tokens_ = 0;
    cur_pos_    = 0;

    // 1) 命中阶段：从位置 0 连续查盘
    while (!cfg_.disable_hit && cur_pos_ + cfg_.block_tokens <= prompt.size()) {
        const size_t end = cur_pos_ + cfg_.block_tokens;
        const std::string file = disk_kv::block_path(model_, model_path_, cfg_.cache_dir, prompt, cur_pos_, end);
        if (!disk_kv::load_block(ctx_, 0, 1, file, prompt, cur_pos_, end)) break;
        cur_pos_ = end;
    }
    hit_tokens_ = cur_pos_;

    // 2) 重算阶段：剩余部分 decode + 落盘（块大小不足 BLOCK 时按实际长度）
    while (cur_pos_ < prompt.size()) {
        const size_t end = std::min(cur_pos_ + cfg_.block_tokens, prompt.size());
        const std::vector<llama_token> seg(prompt.begin() + cur_pos_, prompt.begin() + end);
        if (!decode_batch(seg, cur_pos_)) return hit_tokens_;
        if (!cfg_.disable_save) {
            const std::string file = disk_kv::block_path(model_, model_path_, cfg_.cache_dir, prompt, cur_pos_, end);
            if (!disk_kv::save_block(ctx_, 0, 1, file, prompt, cur_pos_, end)) {
                // 落盘失败（如磁盘满）：打日志继续，不中断整条 prompt 的处理
                fprintf(stderr, "[kv-cache] failed to save block [%zu, %zu)\n", cur_pos_, end);
            }
        }
        cur_pos_ = end;
    }
    // 缓存维护（老化 + 容量 LRU）整条 prompt 只跑一次，避免每块全目录扫描
    if (!cfg_.disable_save) {
        disk_kv::cleanup_expired(cfg_.cache_dir, cfg_.max_age_days);
        disk_kv::enforce_cache_limit(cfg_.cache_dir, cfg_.max_cache_size_mb);
    }
    return hit_tokens_;
}

// ---------------------------------------------------------------------------
// 单 token 续写
// ---------------------------------------------------------------------------
bool kv_cache::decode(llama_token token) {
    const std::vector<llama_token> one = { token };
    if (!decode_batch(one, cur_pos_)) return false;
    cur_pos_++;
    return true;
}

// ---------------------------------------------------------------------------
// sha256 公开静态：委托 disk_kv
// ---------------------------------------------------------------------------
std::string kv_cache::sha256_hex(const void * data, size_t len) {
    return disk_kv::sha256_hex(data, len);
}
