#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#include "ggml-rpc.h"
#ifdef _WIN32
#  define DIRECTORY_SEPARATOR '\\'
#  define NOMINMAX
#  include <locale>
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
#else
#  define DIRECTORY_SEPARATOR '/'
#  include <unistd.h>
#  include <sys/stat.h>
#endif
#include <string>
#include <stdio.h>
#include <algorithm>
#include <thread>
#include <fstream>
#include <filesystem>
#include <codecvt>
#include <regex>

namespace fs = std::filesystem;

// NOTE: this is copied from common.cpp to avoid linking with libcommon
// returns true if successful, false otherwise

#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

static bool fs_create_directory_with_parents(const std::string& path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);
        const wchar_t* test = subpath.c_str();

        const bool success = CreateDirectoryW(test, NULL);
        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            }
            else {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        }
        else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

// NOTE: this is copied from common.cpp to avoid linking with libcommon
static std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    }
    else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        }
        else {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

struct rpc_server_params {
    std::string              host = "127.0.0.1";
    int                      port = 50052;
    bool                     use_cache = false;
    bool                     use_cpu = false;
    int                      n_threads = std::max(1U, std::thread::hardware_concurrency() / 2);
    std::vector<std::string> devices;
};

static void print_usage(int /*argc*/, char** argv, rpc_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                          show this help message and exit\n");
    fprintf(stderr, "  -t, --threads N                     number of threads for the CPU device (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -d, -dev, --device <dev1,dev2,...>  comma-separated list of devices\n");
    fprintf(stderr, "  -cpu                                enable cpu backend\n");
    fprintf(stderr, "  -h, -H, --host, --Host HOST         host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p, -P, --port, --Port PORT         port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -c, --cache                         enable local file cache\n");
    fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char** argv, rpc_server_params& params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "-h" || arg == "--host" || arg == "--Host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        }
        else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "error: invalid number of threads: %d\n", params.n_threads);
                return false;
            }
        }
        else if (arg == "-d" || arg == "-dev" || arg == "--device") {
            if (++i >= argc) {
                return false;
            }
            const std::regex regex{ R"([,/]+)" };
            std::string dev_str = argv[i];
            std::sregex_token_iterator iter(dev_str.begin(), dev_str.end(), regex, -1);
            std::sregex_token_iterator end;
            for (; iter != end; ++iter) {
                try {
                    params.devices.push_back(*iter);
                }
                catch (const std::exception&) {
                    fprintf(stderr, "error: invalid device: %s\n", iter->str().c_str());
                    return false;
                }
            }
        }
        else if (arg == "-p" || arg == "-P" || arg == "--port" || arg == "--Port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        }
        else if (arg == "-c" || arg == "--cache") {
            params.use_cache = true;
        }
        else if (arg == "-cpu") {
            params.use_cpu = true;
        }
        else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

static ggml_backend_t create_cpu_backend(const rpc_server_params& params) {   
    fprintf(stderr, "%s: using CPU backend\n", __func__);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, params.n_threads);
    return backend;
}

static ggml_backend_t create_gpu_backend(const rpc_server_params& params, uint32_t device) {
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend: CUDA%d\n", __func__, device);
    backend = ggml_backend_cuda_init(device, nullptr); // init device
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#elif GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#elif GGML_USE_VULKAN
    fprintf(stderr, "%s: using Vulkan backend\n", __func__);
    backend = ggml_backend_vk_init(device); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_vulkan_init() failed\n", __func__);
    }
#elif GGML_USE_SYCL
    fprintf(stderr, "%s: using SYCL backend\n", __func__);
    backend = ggml_backend_sycl_init(device); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_sycl_init() failed\n", __func__);
    }
#endif
    // if there aren't GPU Backends fallback to CPU backend
    //if (!backend) {
    //    fprintf(stderr, "%s: using CPU backend\n", __func__);
    //    backend = ggml_backend_cpu_init();
    //    ggml_backend_cpu_set_n_threads(backend, params.n_threads);
    //}
    return backend;
}

static int32_t find_device_idx(const std::string& str) {
    std::regex pattern(R"((\d+)$)");  // Match digits at the end
    std::smatch matches;
    int number = -1;
    if (std::regex_search(str, matches, pattern)) {
        number = std::stoi(matches[1]);
    }
    return number;
}

static size_t get_gpu_backend_count(const rpc_server_params& params) {
    size_t count = 0;
#if defined(GGML_USE_CUDA)
    count = ggml_backend_cuda_get_device_count();
#elif defined(GGML_USE_SYCL)
    count = ggml_backend_sycl_get_device_count();
#elif defined(GGML_USE_VULKAN)
    count = ggml_backend_vk_get_device_count();
#elif defined(GGML_USE_CANN)
    return ggml_backend_cann_get_device_count();
#endif
    return count;
}

static std::vector<ggml_backend_t> get_devices(const rpc_server_params& params) {
    std::vector<ggml_backend_t> devices;
    if (!params.devices.empty()) {
        for (auto device : params.devices) {
            int32_t device_id;
            ggml_backend_t dev;
            if (params.use_cpu && device == "CPU" ) {
                dev = create_cpu_backend(params);
            } else {
                device_id = find_device_idx(device);
                if (device_id < 0) {
                    fprintf(stderr, "error: unknown device: %s\n", device.c_str());
                    continue;
                }
                dev = create_gpu_backend(params, device_id);
            }
            if (dev) {
                devices.push_back(dev);
            } else {
                fprintf(stderr, "error: unknown device: %s\n", device.c_str());
            }
        }
    }
    else {
        for (size_t i = 0; i < get_gpu_backend_count(params); i++) {
            ggml_backend_t dev = create_gpu_backend(params, i);
            if (dev) {
                devices.push_back(dev);
            }
        }
        // cpu backend at last
        if (params.use_cpu || devices.empty()) {
            ggml_backend_t dev = create_cpu_backend(params);
            if (dev) {
                devices.push_back(dev);
            }
        }
    }
    return devices;
}

static void get_cpu_backend_memory(size_t * free_mem, size_t * total_mem) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total_mem = status.ullTotalPhys;
    *free_mem = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total_mem = pages * page_size;
    *free_mem = *total_mem;
#endif
}

static void get_backend_memory(uint32_t device, size_t * free_mem, size_t * total_mem) {
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_get_device_memory(device, free_mem, total_mem);
#elif GGML_USE_VULKAN
    ggml_backend_vk_get_device_memory(device, free_mem, total_mem);
#elif GGML_USE_SYCL
    ggml_backend_sycl_get_device_memory(device, free_mem, total_mem);
#else
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        *total_mem = status.ullTotalPhys;
        *free_mem = status.ullAvailPhys;
    #else
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        *total_mem = pages * page_size;
        *free_mem = *total_mem;
    #endif
#endif
}

int main(int argc, char * argv[]) {
    rpc_server_params params;
    if (!rpc_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    if (params.host != "127.0.0.1") {
        fprintf(stderr, "\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "WARNING: Host ('%s') is != '127.0.0.1'\n", params.host.c_str());
        fprintf(stderr, "         Never expose the RPC server to an open network!\n");
        fprintf(stderr, "         This is an experimental feature and is not secure!\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
    }

    auto devices = get_devices(params);
    if (devices.empty()) {
        fprintf(stderr, "No backend found\n");
        return 1;
    }

    std::string endpoint = params.host + ":" + std::to_string(params.port);
    std::vector<size_t> free_mem, total_mem;
    for (size_t i = 0; i < devices.size(); i++) {
        size_t free, total;
        const char* name = ggml_backend_name(devices[i]);
        if (std::string(name) == "CPU") {
            get_cpu_backend_memory(&free, &total);
        } else {
            int32_t idx = find_device_idx(name);
            get_backend_memory((uint32_t) idx, &free, &total);
        }
        free_mem.push_back(free);
        total_mem.push_back(total);
    }

    const char * cache_dir = nullptr;
    std::string cache_dir_str;
    if (params.use_cache) {
        cache_dir_str = fs_get_cache_directory() + "rpc/";
        if (!fs_create_directory_with_parents(cache_dir_str)) {
            fprintf(stderr, "Failed to create cache directory: %s\n", cache_dir_str.c_str());
            return 1;
        }
        cache_dir = cache_dir_str.c_str();
    }
    ggml_backend_rpc_start_server(endpoint.c_str(), cache_dir, devices.size(), devices.data(),
        free_mem.data(), total_mem.data());
    return 0;
}
