// kv-cache.h — 磁盘 KV 块缓存模块
//
// 把计算过的前缀 KV 按固定 token 块落盘，新请求逐块命中加载 / 未命中重算并自动落盘。
// 机制已在 Qwen2.5-0.5B（纯 transformer）与 Qwen3.8-27B Ridge（hybrid/Qwen3-Next）上验证：
// 块级拼装（load + seq_cp 合并）后续写 logits 与从头计算逐位一致（max_abs_diff = 0）。
//
// 两个 hybrid 专属前提（详见记忆 reference_hybrid-seq-state-save）：
//   1. save 前必须 llama_kv_cache_update 触发 s_copy（seq_cp 对 hybrid 是懒拷贝）
//   2. n_seq_max 必须 ≥ 2（qnext state 按 seq_id 索引行，默认 1 会让临时序列无 state 行）

#pragma once

#include "llama.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// disk_kv：不依赖 kv_cache 实例的磁盘 KV 块级函数。
// 接受外部 llama_context * + 任意主序列 seq_id，供 llama-server 等直接复用。
// 只做"命中加载 / 落盘 / 清理"，不做 decode（decode 由调用方组织自己的 batch）。
//
// 踩坑保留（务必保持）：
//   - hybrid 模型（Qwen3-Next）save 前必须 llama_kv_cache_update 触发 s_copy，
//     否则 state 不落盘（seq_cp 对 hybrid 是懒拷贝）
//   - load 到 tmp 序列后 llama_kv_cache_seq_cp(tmp→dst) 合并，再 llama_kv_cache_update
//   - 块文件命名 = sha256(块 token 字节) + "_" + start + "_" + 模型指纹 + ".bin"
//     模型指纹 = sha256(模型路径) + "_" + n_ctx_train + "_" + n_vocab +
//                "_" + 模型文件大小 + "_" + 模型文件 mtime（换权重自动失效旧缓存）
//   - load 时做 token 校验（哈希碰撞兜底），失败删坏文件；
//     结构完好但恢复失败（如 KV 满）不删文件，视为环境失败
//   - save 原子写（.tmp + MoveFileEx rename）
//   - 老化清理（mtime 超 max_age_days 删）+ 容量 LRU（超 max_cache_size_mb 删最旧）
// ---------------------------------------------------------------------------
namespace disk_kv {

    // sha256 hex（Windows 用 BCrypt；非 Windows 用内嵌公共域实现）
    std::string sha256_hex(const void * data, size_t len);

    // 模型指纹 = sha256(模型路径) + "_" + n_ctx_train + "_" + n_vocab
    //             + "_" + 模型文件大小 + "_" + 模型文件 mtime，
    // 避免串模型 / 串权重加载缓存（同路径换权重会改变大小或 mtime，旧块自动失效）
    std::string model_fingerprint(const llama_model * model, const std::string & model_path);

    // 块文件完整路径 = <cache_dir>\<sha256(块token)>_<start>_<模型指纹>.bin
    std::string block_path(const llama_model * model, const std::string & model_path,
                           const std::string & cache_dir,
                           const std::vector<llama_token> & prompt,
                           size_t start, size_t end);

    // 命中加载：把 file 里 [start,end) 的 KV 块加载到 dst_seq（经 tmp_seq 中转 + seq_cp 合并）。
    // 含 token 校验（哈希碰撞兜底），校验失败删坏文件。命中后 touch 更新 LRU 时间。
    // 注意：调用方需保证 ctx 的 n_seq_max ≥ 2，且 dst_seq 与 tmp_seq 不冲突。
    bool load_block(llama_context * ctx, llama_seq_id dst_seq, llama_seq_id tmp_seq,
                    const std::string & file, const std::vector<llama_token> & prompt,
                    size_t start, size_t end);

    // 落盘：把 src_seq 的 [start,end) 状态经 tmp_seq 中转保存到 file。
    // save 前 llama_kv_cache_update 触发 hybrid s_copy。原子写（.tmp + rename）。
    // 文件已存在直接返回 true（不重复写）。
    bool save_block(llama_context * ctx, llama_seq_id src_seq, llama_seq_id tmp_seq,
                    const std::string & file, const std::vector<llama_token> & prompt,
                    size_t start, size_t end);

    // 老化清理：删除 cache_dir 下超过 max_age_days 未访问（mtime）的 .bin 及残留 .tmp；
    // max_age_days <= 0 禁用。
    void cleanup_expired(const std::string & cache_dir, int max_age_days);

    // 容量 LRU：cache_dir 总大小超过 max_cache_size_mb（MB）时按 mtime 升序删最旧 .bin；
    // 0 = 不限。
    void enforce_cache_limit(const std::string & cache_dir, size_t max_cache_size_mb);

} // namespace disk_kv

struct kv_cache_config {
    int             n_gpu_layers = 0;    // offload 到 GPU 的层数
    int             n_ctx        = 8192; // 上下文长度
    int             n_seq_max    = 16;   // 序列槽位（必须 ≥ 2）
    size_t          block_tokens = 64;   // 每块 token 数
    std::string     cache_dir    = "kv-cache"; // 缓存目录（自动创建）
    bool            disable_hit  = false; // true 时跳过命中（全量重算，用于对照）
    bool            disable_save = false; // true 时跳过落盘（用于对照：隔离 save_block 副作用）
    size_t          max_cache_size_mb = 0; // 缓存目录容量上限（MB），0 = 不限；超限按 LRU 删最旧
    int             max_age_days      = 3; // 超过 N 天未访问的块删除（时间老化清理），<=0 禁用
};

class kv_cache {
public:
    kv_cache(const char * model_path, const kv_cache_config & cfg);
    ~kv_cache();

    // 预热完整 prompt 的 KV：从位置 0 逐块查盘，命中加载拼装，未命中重算并落盘。
    // 返回从 0 连续命中的 token 数。之后可直接 decode() 续写。
    size_t prefill(const std::vector<llama_token> & prompt);

    // 续写一个 token（追加到主序列末尾），返回是否成功
    bool decode(llama_token token);

    void set_disable_hit(bool v)  { cfg_.disable_hit  = v; }
    void set_disable_save(bool v) { cfg_.disable_save = v; }

    llama_model *   model() const { return model_; }
    llama_context * ctx()   const { return ctx_; }
    size_t hit_tokens()     const { return hit_tokens_; } // 最近一次 prefill 的命中 token 数
    size_t current_pos()    const { return cur_pos_; }
    const std::string & model_fingerprint() const { return fp_; }

    static std::string sha256_hex(const void * data, size_t len);

private:
    bool decode_batch(const std::vector<llama_token> & toks, size_t pos0);

    std::string     model_path_;
    llama_model *   model_ = nullptr;
    llama_context * ctx_   = nullptr;
    kv_cache_config cfg_;
    std::string     fp_;
    size_t          hit_tokens_ = 0;
    size_t          cur_pos_    = 0;
};
