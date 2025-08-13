#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

struct llama_file {

#if defined(_WIN32)
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    HANDLE fp_win32;
    size_t size;

private:
    std::string GetErrorMessageWin32(DWORD error_code) const;

public:

    llama_file(const char * fname, const char * mode);

    size_t tell() const;

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len) const;

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const;

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file();
#else
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode);

    size_t tell() const;

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len) const;

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const;

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file();
#endif
};
using llama_files = std::vector<std::unique_ptr<llama_file>>;

struct llama_mmap {
    void * addr;
    size_t size;
    size_t mapped_page_size = 0;

    llama_mmap(const llama_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    // list of mapped fragments (first_offset, last_offset)
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1 /* -1 = max value */, bool numa = false, bool use_thp = false);

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        // align first to the next page
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        // align last to the previous page
        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    // partially unmap the file in the range [first, last)
    void unmap_fragment(size_t first, size_t last);

#ifdef __linux__
    static int get_default_huge_page_size();
#endif

    ~llama_mmap();
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false, bool use_thp = false);

    void unmap_fragment(size_t first, size_t last);

    ~llama_mmap();
#else
    static constexpr bool SUPPORTED = false;

    llama_mmap(struct llama_file * file, size_t prefetch = -1, bool numa = false, bool use_thp = false);

    void unmap_fragment(size_t first, size_t last);
#endif
};
using llama_mmaps = std::vector<std::unique_ptr<llama_mmap>>;

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void * addr = NULL;
    size_t size = 0;

    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock &) = delete;

    ~llama_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void * ptr);

    void grow_to(size_t target_size);

    static size_t lock_granularity();

    bool raw_lock(void * ptr, size_t len) const;

    static void raw_unlock(void * ptr, size_t len);

#ifdef _POSIX_MEMLOCK_RANGE
    static constexpr bool SUPPORTED = true;
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;
#else
    static constexpr bool SUPPORTED = false;
#endif
};
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;
