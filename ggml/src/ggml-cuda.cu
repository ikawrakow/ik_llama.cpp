//

// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "ggml-cuda.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/dmmv.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/mmq_id.cuh"
#include "ggml-cuda/quantize_id.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/multiadd.cuh"
#include "ggml-cuda/hadamard.cuh"
#include "ggml-cuda/reduce.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>

#define IK_PRINT_TIMING 0

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void ggml_cuda_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fprintf(stderr, "%s", msg);
}

ggml_log_callback ggml_cuda_log_callback = ggml_cuda_default_log_callback;
void * ggml_cuda_log_user_data = NULL;

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_cuda_log_callback = log_callback;
    ggml_cuda_log_user_data = user_data;
}

#define GGML_CUDA_LOG_INFO(...) ggml_cuda_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_CUDA_LOG_WARN(...) ggml_cuda_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_CUDA_LOG_ERROR(...) ggml_cuda_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_CUDA_LOG_DEBUG(...) ggml_cuda_log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_cuda_log(enum ggml_log_level level, const char * format, ...) {
    if (ggml_cuda_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_cuda_log_callback(level, buffer, ggml_cuda_log_user_data);
        } else {
            std::vector<char> buffer2(len + 1);  // vsnprintf adds a null terminator
            va_end(args);
            va_start(args, format);
            vsnprintf(&buffer2[0], buffer2.size(), format, args);
            ggml_cuda_log_callback(level, buffer2.data(), ggml_cuda_log_user_data);
        }
        va_end(args);
    }
}

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    GGML_CUDA_LOG_ERROR("CUDA error: %s\n", msg);
    GGML_CUDA_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_CUDA_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CUDA error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
#if defined(GGML_USE_HIPBLAS) && defined(GGML_HIP_UMA)
    auto res = hipMallocManaged(ptr, size);
    if (res == hipSuccess) {
        // if error we "need" to know why...
        CUDA_CHECK(hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
    }
    return res;
#else

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr)
    {
        err = cudaMallocManaged(ptr, size);
    }
    else
    {
        err = cudaMalloc(ptr, size);
    }
    return err;
#else
    return cudaMalloc(ptr, size);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

#endif
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    no\n", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: no\n", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        GGML_CUDA_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                prop.totalGlobalMem/(1024*1024));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm   = prop.multiProcessorCount;
        info.devices[id].smpb  = prop.sharedMemPerBlock;
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpbo = prop.sharedMemPerBlock;
        info.devices[id].cc = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

#ifdef GGML_USE_NCCL
    info.have_nccl = false;
    if (info.device_count > 1) {
        int gpu_list[GGML_CUDA_MAX_DEVICES];
        for(int i = 0; i < info.device_count; ++i) gpu_list[i] = i;
        auto status = ncclCommInitAll(info.nccl_coms, info.device_count, gpu_list);
        if (status != ncclSuccess) {
            printf("=============================== NCCL initialization failed with status %d\n", int(status));
        } else {
            printf("=============================== NCCL main communicator initialized\n");
            info.have_nccl = true;
            auto com = info.nccl_coms + info.device_count;
            if (info.device_count == 4) {
                int devs[8] = {0,1, 2,3, 0,2, 1,3};
                auto com = info.nccl_coms + info.device_count;
                for (int ip = 0; ip < 4; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            } else if (info.device_count == 3) {
                int devs[4] = {0,1, 0,2};
                for (int ip = 0; ip < 2; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            }
        }
    }
#endif
    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_CUDA_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_CUDA_LOG_WARN("Cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) (pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) (pool_addr + pool_used));
    }
};
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::ggml_backend_cuda_context(int device) :
    device(device), name(GGML_CUDA_NAME + std::to_string(device)) {
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    if (info->all_ctx[device]) {
        GGML_CUDA_LOG_WARN("%s: a context for device %d already exists?\n", __func__, device);
    }
    info->all_ctx[device] = this;
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {

    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    info->all_ctx[this->device] = nullptr;

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }

}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

GGML_CALL static const char * ggml_backend_cuda_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_buffer_get_name;
}

GGML_CALL static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    printf("%s(%s -> %s)\n", __func__, src->name, dst->name);
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_cuda_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

struct ggml_backend_cuda_split_buffer_type_context {
    //std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        //for (ggml_tensor_extra_gpu * extra : tensor_extras) {
        //    for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
        //        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
        //            if (extra->events[id][is] != nullptr) {
        //                CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
        //            }
        //        }
        //        if (extra->data_device[id] != nullptr) {
        //            CUDA_CHECK(cudaFree(extra->data_device[id]));
        //        }
        //    }
        //    delete extra;
        //}
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

GGML_CALL static const char * ggml_backend_cuda_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buffer);
}

static bool ggml_backend_buffer_is_cuda_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_cuda_split); // only used in debug builds currently, avoid unused function warning in release builds
}

GGML_CALL static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_split_buffer_init_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (!tensor->extra) return;
    //printf("%s(%s, %p)\n", __func__, tensor->name, tensor->extra);
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        auto split = extra->splits[i];
        auto ne0 = split->ne[0];
        auto size = ggml_nbytes(split);
        auto padded_size = size;
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            int nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            auto row_size = ggml_row_size(split->type, ne0);
            padded_size += padded_row_size - row_size;
        }
        ggml_cuda_set_device(i);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, padded_size, i));
        if (padded_size > size) {
            CUDA_CHECK(cudaMemset(buf + size, 0, padded_size - size));
        }
        //printf("    allocated %zu bytes for tensor %s of type %s, dim = %ld x %ld x %ld. padding: %zu\n", padded_size, split->name, ggml_type_name(split->type),
        //        split->ne[0], split->ne[1], split->ne[2], padded_size - size);
        split->data = buf;
        auto ctx = new ggml_backend_cuda_buffer_context(i, buf);
        auto buft = ggml_backend_cuda_buffer_type(i);
        split->buffer = ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, padded_size);
        ggml_backend_buffer_set_usage(split->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return;

}

GGML_CALL static void ggml_backend_cuda_split_buffer_set_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!tensor->extra) return;
    static std::map<ggml_type, int> k_map = {
        { GGML_TYPE_Q4_0_R8   , 8},
        { GGML_TYPE_Q5_0_R4   , 4},
        { GGML_TYPE_Q8_0_R8   , 8},
        { GGML_TYPE_Q2_K_R4   , 4},
        { GGML_TYPE_Q3_K_R4   , 4},
        { GGML_TYPE_Q4_K_R4   , 4},
        { GGML_TYPE_Q5_K_R4   , 4},
        { GGML_TYPE_Q6_K_R4   , 4},
        { GGML_TYPE_IQ2_XXS_R4, 4},
        { GGML_TYPE_IQ2_XS_R4 , 4},
        { GGML_TYPE_IQ3_XXS_R4, 4},
        { GGML_TYPE_IQ1_S_R4  , 4},
        { GGML_TYPE_IQ4_NL_R4 , 4},
        { GGML_TYPE_IQ3_S_R4  , 4},
        { GGML_TYPE_IQ2_S_R4  , 4},
        { GGML_TYPE_IQ4_XS_R8 , 8},
        { GGML_TYPE_IQ1_M_R4  , 4},
        { GGML_TYPE_BF16_R16  , 16},
        { GGML_TYPE_Q6_0_R4   , 4},
        { GGML_TYPE_IQ2_BN_R4 , 4},
        { GGML_TYPE_IQ2_K_R4  , 4},
        { GGML_TYPE_IQ3_K_R4  , 4},
        { GGML_TYPE_IQ4_K_R4  , 4},
        { GGML_TYPE_IQ5_K_R4  , 4},
        { GGML_TYPE_IQ4_KS_R4 , 4},
        { GGML_TYPE_IQ5_KS_R4 , 4},
        { GGML_TYPE_Q8_K_R16  , 4},
        { GGML_TYPE_Q8_KV_R8  , 4},
        { GGML_TYPE_Q8_K_R8   , 8},
    };
    //printf("%s(%s)\n", __func__, tensor->name);

    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    if (extra->split_dim < 0) {
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto nbytes = ggml_nbytes(tensor);
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(ggml_are_same_shape(tensor, split));
            GGML_ASSERT(ggml_nbytes(split) == nbytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(split->data, data, nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
    }
    else if (extra->split_dim == 0) {
        int n_interleave = 1;
        if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
        //if (tensor->type >= GGML_TYPE_Q4_0_R8) {
        //    GGML_ABORT("Dim 0 copy of row-interleaved quants is not supported yet");
        //}
        auto tt = ggml_internal_get_type_traits(tensor->type);
        std::vector<char> host_buffer;
        GGML_ASSERT(ggml_is_contiguous(tensor));
        int nrows = ggml_nrows(tensor);
        auto bs = tt.blck_size;
        auto ts = tt.type_size;
        auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        int ne = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->ne[1]%n_interleave == 0);
            ggml_cuda_set_device(i);
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT((int)ggml_nrows(split) == nrows);
            GGML_ASSERT(split->ne[0] % bs == 0);
            auto source_offset = n_interleave*(tt.row_meta_size + (ne / bs) * ts);
            auto split_row_size = ggml_row_size(split->type, split->ne[0]);
            if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
            for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                    auto dst = host_buffer.data() + (i02*split->ne[1] + i01)*split_row_size;
                    auto src = (const char *)data + i02*tensor->nb[2] + i01*tensor->nb[1];
                    if (tt.row_meta_size > 0) {
                        memcpy(dst, src, tt.row_meta_size*n_interleave);
                    }
                    memcpy(dst + tt.row_meta_size*n_interleave, src + source_offset, n_interleave*(split_row_size - tt.row_meta_size));
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            ne += split->ne[0];
        }
    }
    else if (extra->split_dim == 1) {
        if (tensor->ne[2] > 1) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<char> host_buffer;
            int ne1 = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                auto size = ggml_nbytes(split);
                if (host_buffer.size() < size) host_buffer.resize(size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    auto dst = host_buffer.data() + i02*split->ne[1]*row_size;
                    auto src = (const char *)data + i02*tensor->nb[2] + ne1*tensor->nb[1];
                    memcpy(dst, src, split->ne[1]*row_size);
                }
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne1 += split->ne[1];
            }
        } else {
            int n_interleave = 1;
            if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
            size_t cur_offset = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                GGML_ASSERT(split->ne[1]%n_interleave == 0);
                ggml_cuda_set_device(i);
                auto size = ggml_nbytes(split);
                const char * buf_host = (const char *)data + cur_offset;
                CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                cur_offset += size;
            }
        }
    }
    else {
        fprintf(stderr, "%s: not implemented for split dim %d\n", __func__, extra->split_dim == 0);
        GGML_ABORT("fatal error");
    }

    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

}

GGML_CALL static void ggml_backend_cuda_split_buffer_get_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        [[maybe_unused]] void * data, size_t offset, size_t size) {
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    GGML_ABORT("not implemented");
}

GGML_CALL static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

GGML_CALL static const char * ggml_backend_cuda_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size([[maybe_unused]] ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (!tensor->extra) return 0;
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    size_t total_size = 0;
    for (int i = 0; i < extra->n_device; ++i) {
        auto split = extra->splits[i];
        if (!split) continue;
        total_size += ggml_nbytes(split);
        auto ne0 = split->ne[0];
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            auto nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto row_size = ggml_row_size(split->type, ne0);
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            total_size += padded_row_size - row_size;
        }
    }
    return total_size;

}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * /*tensor_split*/) {
    static ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .context = */ new ggml_backend_cuda_split_buffer_type_context{}, //{tensor_split_arr},
    };
    return &buft;
}

// host buffer type

GGML_CALL static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cuda_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cuda_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x     = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel   = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi = __half2float(x[ix]);

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t rs = ggml_row_size(type, ne0);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == rs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        // TODO: this only works if the row does not contain meta data
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*rs);
            // pretend the row is a matrix with cols=1
            // TODO: this only works if the row does not contain meta data
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {

        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
        return;
    }

#ifdef GGML_CUDA_IQK_FORCE_BF16
    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src0->type);
        to_bf16_cuda_t to_bf16_cuda_1 = src1->type != GGML_TYPE_BF16 ? ggml_get_to_bf16_cuda(src1->type) : nullptr;
        if (to_bf16_cuda && (src1->type == GGML_TYPE_BF16 || to_bf16_cuda_1)) {
            size_t ne = row_diff*ne00;
            ggml_cuda_pool_alloc<nv_bfloat16> src0_as_bf16(ctx.pool(id), ne);
            to_bf16_cuda(src0_dd_i, src0_as_bf16.get(), row_diff, ne00, stream);

            ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
            if (src1->type != GGML_TYPE_BF16) {
                size_t ne = src1_ncols*ne10;
                src1_as_bf16.alloc(ne);
                to_bf16_cuda_1(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
            }
            const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
            const nv_bfloat16 * src0_ptr = src0_as_bf16.get();

            ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

            const float alpha_f32 = 1.0f;
            const float beta_f32  = 0.0f;

            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
            CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                        src1_ptr,       CUDA_R_16BF, ne10,
                        &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
            to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
            return;
        }
    }
#endif

    if (compute_capability >= CC_VOLTA && (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff, ne00, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), src1_ncols, ne10, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();
        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                                src1_ptr,       CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff, src1_ncols, stream);
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff, ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols, ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static bool ggml_cuda_set_peer_access(int main_device) {
    ggml_cuda_set_device(main_device);

    bool all_enabled = true;
    for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
        if (main_device == id_other) {
            continue;
        }

        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, main_device, id_other));
        if (can_access_peer) {
//~ #ifdef NDEBUG
            GGML_CUDA_LOG_INFO(" =========================== %s: Enabling Peer Access between Devices %d->%d\n", __func__, main_device, id_other);
//~ #endif //NDEBUG
            cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
            if (err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            } else {
                // reset the error
                (void)cudaGetLastError();
            }
        } else {
            all_enabled = false;
        }
    }
    return all_enabled;
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_rs = ggml_row_size(src0->type, ne00);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

    }

    bool quantization_done = false;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer, it may then be necessary to clear padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            const int64_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const int64_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data , 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && (src1_is_contiguous || (src1->ne[1] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float)))) {
                if (src1_is_contiguous) {
                    quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12*ne13, src1_padded_col_size, src0->type, stream);
                } else {
                    //printf("Calling quantize_tensor_q8_1_cuda for %s\n", src0->name);
                    quantize_tensor_q8_1_cuda(src1, dev[id].src1_ddq, src0->type, stream);
                }
                CUDA_CHECK(cudaGetLastError());
                quantization_done = true;
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    const int64_t src1_col_stride = ne11;
    if (quantization_done && ne11 == 1 && ne12 > 1 && ne13 == 1 && ne02 == ne12 && ne02 == dst->ne[2]) {
        //printf("invoking fast path for %s x %s\n", src0->name, src1->name);
        int id = ctx.device;
        char  *  src0_dd_i =  dev[id].src0_dd;
        float * src1_ddf_i = dev[id].src1_ddf;
        char  * src1_ddq_i = dev[id].src1_ddq;
        float *   dst_dd_i =   dev[id].dst_dd;
        cudaStream_t stream = ctx.stream(id, 0);
        ggml_cuda_op_mul_mat_vec_q_3D(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                dev[id].row_low, dev[id].row_high, ne11, src1_padded_col_size, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[id].src0_dd + (i0/i02_divisor) * ne01*src0_rs;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    if (!quantization_done) {
                        //printf("Copying %s\n", src1->name);
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                    src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                    }
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous && !quantization_done) {
                    //printf("Quantizing %s\n", src1->name);
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                    GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                    dhf_dst_i += src1_col_0*ne0;
                    CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                }

            }
        }
    }
}

static void ggml_cuda_mul_mat_vec_p021(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static __global__ void k_compute_batched_ptrs(
        const half * src0_as_f16, const half * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    int64_t i03 = i13 / r3;
    int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);

    cudaStream_t main_stream = ctx.stream();

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    void * src0_ddq = src0->data;
    half * src0_f16 = (half *) src0_ddq;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    // convert src1 to fp16
    ggml_cuda_pool_alloc<half> src1_f16_alloc(ctx.pool());
    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        to_fp16_cuda(src1_ddf, src1_f16_alloc.get(), ggml_nrows(src1), src1->ne[0], main_stream);
    }
    half * src1_f16 = src1->type == GGML_TYPE_F16 ? (half *) src1_ddf : src1_f16_alloc.get();

    ggml_cuda_pool_alloc<half> dst_f16(ctx.pool());
    char * dst_t;

    cublasComputeType_t cu_compute_type = CUBLAS_COMPUTE_16F;
    cudaDataType_t      cu_data_type    = CUDA_R_16F;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    const half  alpha_f16 = 1.0f;
    const half  beta_f16  = 0.0f;

    const float alpha_f32 = 1.0f;
    const float beta_f32  = 0.0f;

    const void * alpha = &alpha_f16;
    const void * beta  = &beta_f16;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        dst_t = (char *) dst_f16.alloc(ne_dst);

        nbd2 /= sizeof(float) / sizeof(half);
        nbd3 /= sizeof(float) / sizeof(half);
    } else {
        dst_t = (char *) dst_ddf;

        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type    = CUDA_R_32F;

        alpha = &alpha_f32;
        beta  = &beta_f32;
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

#if 0
    // use cublasGemmEx
    {
        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                int i03 = i13 / r3;
                int i02 = i12 / r2;

                CUBLAS_CHECK(
                        cublasGemmEx(g_cublas_handles[g_main_device], CUBLAS_OP_T, CUBLAS_OP_N,
                            ne01, ne11, ne10,
                            alpha, (const char *) src0_as_f16 + i02*src0->nb[2]   + i03*src0->nb[3]  , CUDA_R_16F,   nb01/sizeof(half),
                                   (const char *) src1_as_f16 + i12*src1->nb[2]/2 + i13*src1->nb[3]/2, CUDA_R_16F,   nb11/sizeof(float),
                            beta,  (      char *)       dst_t + i12*nbd2          + i13*nbd3,          cu_data_type, ne01,
                            cu_compute_type,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
    }
#else
#ifdef GGML_USE_MUSA
    GGML_ASSERT(false);
#else // !GGML_USE_MUSA
    if (r2 == 1 && r3 == 1 && ggml_is_contiguous_2(src0) && ggml_is_contiguous_2(src1)) {
        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const char *) src0_f16, CUDA_R_16F,   nb01/nb00, nb02/nb00,  // strideA
                       (const char *) src1_f16, CUDA_R_16F,   nb11/nb10, nb12/nb10,  // strideB
                beta,  (      char *)    dst_t, cu_data_type, ne01,       nb2/nb0,   // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        dim3 block_dims(ne13, ne12);
        k_compute_batched_ptrs<<<1, block_dims, 0, main_stream>>>(
                src0_f16, src1_f16, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                src1->type == GGML_TYPE_F16 ? nb12 : nb12/2,
                src1->type == GGML_TYPE_F16 ? nb13 : nb13/2,
                nbd2, nbd3,
                r2, r3);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), CUDA_R_16F,   nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), CUDA_R_16F,   nb11/nb10,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type, ne01,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
#endif // GGML_USE_MUSA
#endif

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_ddf, ggml_nrows(dst), dst->ne[0], main_stream);
    }
}

static int ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n, bool is_gemv) {

    auto stream = ctx.stream();

    auto fusion = ctx.fusion && src1->ne[1] == 1;

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*ggml_nrows(src1);
    if (!is_gemv) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    if (is_gemv) {
        quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], src1->ne[2], ne10_padded,
                src0->type, stream);
        CUDA_CHECK(cudaGetLastError());

        // The code below handles the case when Q, K, V have a bias applied after the resepctive matrix multiplication.
        // In that case the graph contains mul_mat(Q) -> mul_mat(K) -> mul_mat(V) -> add(Q) -> add(K) -> add(V)
        if (fusion && cgraph && node_n + 5 < cgraph->n_nodes &&
            cgraph->nodes[node_n+1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[node_n+2]->op == GGML_OP_MUL_MAT &&
            ggml_is_quantized(cgraph->nodes[node_n+1]->src[0]->type) &&
            ggml_is_quantized(cgraph->nodes[node_n+2]->src[0]->type) &&
            cgraph->nodes[node_n+3]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+4]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+5]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+0] == cgraph->nodes[node_n+3]->src[0] &&
            cgraph->nodes[node_n+1] == cgraph->nodes[node_n+4]->src[0] &&
            cgraph->nodes[node_n+2] == cgraph->nodes[node_n+5]->src[0]) {
            for (int i = 0; i < 3; ++i) {
                auto src0_i = cgraph->nodes[node_n+i]->src[0];
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, src0_i, src1, cgraph->nodes[node_n+i], cgraph->nodes[node_n+i+3]->src[1],
                        (const char *)src0_i->data, nullptr, src1_quantized.get(), (float *)cgraph->nodes[node_n+i]->data,
                        0, src0_i->ne[1], src1->ne[1], ne10_padded, stream);
                CUDA_CHECK(cudaGetLastError());
            }
            node_n += 5;
        } else if (fusion && cgraph && node_n + 1 < cgraph->n_nodes &&
                   cgraph->nodes[node_n+1]->op == GGML_OP_ADD &&
                   dst == cgraph->nodes[node_n+1]->src[0] &&
                   dst->ne[0] == cgraph->nodes[node_n+1]->src[1]->ne[0] &&
                   cgraph->nodes[node_n+1]->src[1]->type == GGML_TYPE_F32 &&
                   ggml_nrows(cgraph->nodes[node_n+1]->src[1]) == 1) {
            // We have a bias applied after the matrix multiplication and we can fuse it
            ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+1], cgraph->nodes[node_n+1]->src[1],
                 (const char *)dst->src[0]->data, nullptr, src1_quantized.get(), (float *)cgraph->nodes[node_n+1]->data,
                 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            ++node_n;
        } else {
            ggml_cuda_op_mul_mat_vec_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                    0, src0->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded, src0->type, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!cgraph) return node_n;

    while (node_n + 1 < cgraph->n_nodes) {
        dst = cgraph->nodes[node_n+1];
        if (ggml_is_empty(dst) || dst->op == GGML_OP_RESHAPE || dst->op == GGML_OP_TRANSPOSE || dst->op == GGML_OP_VIEW
                               || dst->op == GGML_OP_PERMUTE || dst->op == GGML_OP_NONE) {
            ++node_n; continue;
        }
        if (dst->op != GGML_OP_MUL_MAT || dst->src[1] != src1 || !ggml_is_quantized(dst->src[0]->type)) break;
        if (!is_gemv && mmq_get_q8_1_ds_layout(src0->type) != mmq_get_q8_1_ds_layout(dst->src[0]->type)) break;
        if (is_gemv) {
            if (fusion && node_n + 2 < cgraph->n_nodes &&
                cgraph->nodes[node_n+2]->op == GGML_OP_ADD &&
                dst == cgraph->nodes[node_n+2]->src[0] &&
                dst->ne[0] == cgraph->nodes[node_n+2]->src[1]->ne[0] &&
                cgraph->nodes[node_n+2]->src[1]->type == GGML_TYPE_F32 &&
                ggml_nrows(cgraph->nodes[node_n+2]->src[1]) == 1) {
                // We have a bias applied after the matrix multiplication and we can fuse it
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+2], cgraph->nodes[node_n+2]->src[1],
                        (const char *)dst->src[0]->data, nullptr, src1_quantized.get(), (float *)cgraph->nodes[node_n+2]->data,
                        0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
                ++node_n;
            } else {
                ggml_cuda_op_mul_mat_vec_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, src1_quantized.get(),
                        (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            }
        } else {
            ggml_cuda_op_mul_mat_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, src1_quantized.get(),
                    (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        ++node_n;
    }

    return node_n;

}

static int ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n) {

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_dequantize_mul_mat_vec = ggml_cuda_dmmv_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % (GGML_CUDA_DMMV_X*2) == 0 && src1->ne[1] == 1;
    bool          use_mul_mat_vec_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && ggml_cuda_mmvq_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool              use_mul_mat_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // if mmvq is available it's a better choice than dmmv:
#ifndef GGML_CUDA_FORCE_DMMV
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
#endif // GGML_CUDA_FORCE_DMMV

    bool any_gpus_with_slow_fp16 = false;

    const int cc            = ggml_cuda_info().devices[ctx.device].cc;
    use_mul_mat_q           = use_mul_mat_q           && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
    any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_available(cc);

    if ((use_mul_mat_vec_q || use_mul_mat_q) && src1->ne[2]*src1->ne[3] == 1) {
        return ggml_cuda_mul_mat_q(ctx, src0, src1, dst, cgraph, node_n, use_mul_mat_vec_q);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // FP32 precision KQ single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        // FP32 precision KQV single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if (src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16)
               && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
    return node_n;
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

template <typename data_t = float>
static __global__ void k_copy_src_to_contiguous(const char * __restrict__ src_original, char * __restrict__ src_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne10, int64_t ne11, size_t nb11, size_t nb12) {
    int32_t i = blockIdx.x;

    const int32_t i11 = row_mapping[i].i1 % ne11;
    const int32_t i12 = row_mapping[i].i2;

    data_t * src_row_contiguous = (data_t *)(src_contiguous + i*nb11);
    const data_t * src_row_original = (const data_t *)(src_original + i11*nb11 + i12*nb12);

    for (int j = threadIdx.x; j < ne10; j += blockDim.x) {
        src_row_contiguous[j] = src_row_original[j];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

//static __global__ void k_quick_add(uint32_t n, uint32_t n_per_row, const float * src1, const float * src2, float * dst) {
//
//    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
//        dst[j] = src1[j] + src2[j % n_per_row];
//    }
//}

static __global__ void k_quick_add(uint32_t n_per_row, const float * src1, const float * src2, float * dst) {

    uint32_t row = blockIdx.x;
    const float * src1_row = src1 + row*n_per_row;
    float * dst_row = dst + row*n_per_row;

    for (uint32_t j = threadIdx.x; j < n_per_row; j += blockDim.x) {
        dst_row[j] = src1_row[j] + src2[j];
    }
}

static inline bool prepare_row_mappigs(ggml_backend_cuda_context& ctx, int64_t n_as, int64_t n_ids,
        const ggml_tensor * ids, std::vector<int>& moe_counts, std::vector<int>& cum_moe_counts,
        ggml_cuda_pool_alloc<mmid_row_mapping>& dev_row_mapping) {

    GGML_ASSERT(moe_counts.empty() && cum_moe_counts.empty());

    auto stream = ctx.stream();

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<mmid_row_mapping> rmapping(ids->ne[1]*n_ids);
    moe_counts.resize(n_as, 0);
    cum_moe_counts.resize(n_as + 1);

    bool is_ser = false;
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) ++moe_counts[row_id_i];
            else is_ser = true;
        }
    }
    cum_moe_counts[0] = 0;
    for (int i = 0; i < (int)n_as; ++i) {
        cum_moe_counts[i+1] = cum_moe_counts[i] + moe_counts[i];
    }

    dev_row_mapping.alloc(cum_moe_counts[n_as]);

    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) {
                rmapping[cum_moe_counts[row_id_i]++] = {(int)id, (int)iid1};
            }
        }
    }

    for (int i = 0; i < (int)n_as; ++i) cum_moe_counts[i] -= moe_counts[i];

    CUDA_CHECK(cudaMemcpyAsync(dev_row_mapping.get(), rmapping.data(),
                cum_moe_counts[n_as]*sizeof(mmid_row_mapping), cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaStreamSynchronize(stream));

    return is_ser;
}

static bool ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * next) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    CUDA_CHECK(cudaMemsetAsync((char *)dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    if (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_quantized(src0->type) &&
        ggml_backend_buffer_is_cuda(src0->buffer) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_ctx = (ggml_backend_cuda_buffer_context *) src0->buffer->context;
        ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_ctx->device == device_id &&
            src1_ctx->device == device_id &&
            dst_ctx->device  == device_id) {
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[2] = local_dst.nb[1];

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto local_src1 = *src1;
            local_src1.ne[1] = 1;
            local_src1.nb[1] = src_1_ddq_size;
            local_src1.nb[2] = src1->ne[1] > 1 ? src_1_ddq_size : 0;
            local_src1.nb[3] = local_src1.nb[2];

            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            local_src1.data = src1_quantized.alloc(src_1_ddq_size*src1->ne[1]);
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    src1_padded_col_size, src0->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_vec_q_id(ctx, src0, &local_src1, ids, &local_dst, nullptr,
                (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], 1, src1_padded_col_size, stream);
            CUDA_CHECK(cudaGetLastError());

            if (next && next->op == GGML_OP_MUL_MAT_ID && next->src[0]->type == src0->type && src1 == next->src[1] &&
                ggml_are_same_shape(src0, next->src[0]) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ggml_backend_buffer_is_cuda(next->buffer)) {
                ggml_backend_cuda_buffer_context * next_src0_ctx = (ggml_backend_cuda_buffer_context *) next->src[0]->buffer->context;
                ggml_backend_cuda_buffer_context * next_dst_ctx  = (ggml_backend_cuda_buffer_context *) next->buffer->context;
                if (next_src0_ctx->device == device_id &&
                    next_dst_ctx->device  == device_id) {
                    local_dst.data = next->data;
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, next->src[0], &local_src1, ids, &local_dst, nullptr,
                        (const char *)next->src[0]->data, nullptr, src1_quantized.get(), (float *)next->data,
                        0, src0->ne[1], 1, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                    return true;
                }
            }

            return false;
        }
    }

    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0->type) && ggml_cuda_can_use_mmq_id(src0->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
        ggml_cuda_mul_mat_q_id(ctx, src0, src1, ids, dst, nullptr, nullptr);
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;


    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }

    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
    ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

    src1_row.data = src1_contiguous.get();
    dst_row.data  =  dst_contiguous.get();

    for (int64_t i02 = 0; i02 < n_as; i02++) {

        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) {
            continue;
        }

        size_t mapping_offset = cum_moe_counts[i02];

        {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
        }

        src0_row.data = src0_original + i02*nb02;

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = nb11;
        src1_row.nb[2] = num_src1_rows*nb11;
        src1_row.nb[3] = num_src1_rows*nb11;

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1;
        dst_row.nb[2] = num_src1_rows*nb1;
        dst_row.nb[3] = num_src1_rows*nb1;

        if (ggml_is_quantized(src0->type) &&
            ggml_cuda_should_use_mmq(src0->type, ggml_cuda_info().devices[ctx.device].cc, num_src1_rows)) {
            auto src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            auto src1_quantized_size  = src1_padded_row_size*num_src1_rows;
            if (true || num_src1_rows > MMVQ_MAX_BATCH_SIZE) {
                src1_quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_mmq_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_mul_mat_q_id(ctx, &src0_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());

                CUDA_CHECK(cudaGetLastError());
            } else {
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_row_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_op_mul_mat_vec_q(ctx, &src0_row, &src1_row, &dst_row, (const char *)src0_row.data, nullptr,
                        src1_quantized.get(), (float *)dst_row.data,
                        0, src0_row.ne[1], num_src1_rows, src1_padded_num_cols, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row, nullptr, 0);
        }

        {
            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return false;
}

static int ggml_cuda_moe_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;

    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src0 = src0_1;
    const ggml_tensor * src1 = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];

    if (src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_quantized(src0_1->type) &&
        (!src0_2 || ggml_is_quantized(src0_2->type)) &&
        ggml_backend_buffer_is_cuda(src0_1->buffer) &&
        (!src0_2 || ggml_backend_buffer_is_cuda(src0_2->buffer)) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_1_ctx = (ggml_backend_cuda_buffer_context *) src0_1->buffer->context;
        ggml_backend_cuda_buffer_context * src0_2_ctx = src0_2 ? (ggml_backend_cuda_buffer_context *) src0_2->buffer->context : nullptr;
        ggml_backend_cuda_buffer_context * src1_ctx   = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx    = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_1_ctx->device == device_id &&
            (!src0_2_ctx || src0_2_ctx->device == device_id) &&
            src1_ctx->device   == device_id &&
            dst_ctx->device    == device_id) {
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[1] = local_dst.nb[2] = local_dst.nb[3] = local_dst.ne[0]*sizeof(float);

            auto local_src1 = *src1;
            local_src1.nb[2] = local_src1.nb[3] = 0;

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            if (ggml_is_quantized(src0_1->type) || (src0_2 && ggml_is_quantized(src0_2->type))) {
                GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
                auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
                local_src1.data = src1_quantized.alloc(src_1_ddq_size);
                // Note: no use is currently made of the quantization type passed into quantize_row_q8_1_cuda.
                //       If that were to change, we would need to adjust the code to handle src0_1->type != src0_2->type
                quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], 1, 1, src1_padded_col_size,
                        src0_1->type, stream);
                CUDA_CHECK(cudaGetLastError());

                local_src1.nb[1] = src_1_ddq_size;
            }

            bool fuse_next = next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->src[0]->buffer->context)->device == device_id &&
                ggml_backend_buffer_is_cuda(next->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->buffer->context)->device == device_id;

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            if (src0_2) {
                ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, &local_src1, ids, &local_dst,
                        dst->src[4], dst->src[5],
                        (const char *)src0_1->data, src0_2 ? (const char *)src0_2->data : nullptr,
                        (const float *)src1->data, src1_quantized.get(),
                        (float *)local_dst.data, 0, src0_1->ne[1], 1, src1_padded_col_size, unary_op, stream);
            } else {
                auto local_src0_1 = *src0_1;
                local_src0_1.ne[1] /= 2;
                auto local_src0_2 = local_src0_1;
                local_src0_2.data = (char *)local_src0_1.data + local_src0_1.ne[1]*local_src0_1.nb[1];
                if (!dst->src[4]) {
                    ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, ids, &local_dst,
                            nullptr, nullptr,
                            (const char *)local_src0_1.data, (const char *)local_src0_2.data,
                            (const float *)src1->data, src1_quantized.get(),
                            (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, stream);
                } else {
                    GGML_ASSERT(!dst->src[5]);
                    auto local_bias_1 = *dst->src[4];
                    local_bias_1.ne[0] /= 2;
                    auto local_bias_2 = local_bias_1;
                    local_bias_2.data = (char *)local_bias_1.data + local_bias_1.ne[0]*local_bias_1.nb[0];
                    ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, ids, &local_dst,
                            &local_bias_1, &local_bias_2,
                            (const char *)local_src0_1.data, (const char *)local_src0_2.data,
                            (const float *)src1->data, src1_quantized.get(),
                            (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, stream);
                }
            }
            CUDA_CHECK(cudaGetLastError());

            if (!fuse_next) return i;

            const int64_t dst_padded_col_size = GGML_PAD(dst->ne[0], MATRIX_ROW_PADDING);
            GGML_ASSERT(dst->ne[0] % QK8_1 == 0);
            auto dst_row_size = dst_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto dst_ddq_size = n_ids*dst_row_size;
            ggml_cuda_pool_alloc<char> dst_quantized(ctx.pool(), dst_ddq_size);
            quantize_row_q8_1_cuda((const float *)local_dst.data, (void *)dst_quantized.get(), dst->ne[0], n_ids, 1,
                    dst_padded_col_size, next->src[0]->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_dst.ne[2] = 1;

            auto local_next = *next;
            local_next.ne[2] = local_next.ne[1];
            local_next.ne[1] = local_next.ne[3] = 1;
            local_next.nb[2] = local_next.nb[1];

            local_src1 = *next->src[1];
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;
            local_src1.nb[1] = local_src1.nb[2] = local_src1.nb[3] = dst_row_size;

            auto local_src0 = *next->src[0];
            local_src0.ne[2] = local_src0.ne[3] = 1;

            int result = i + 1;

            if (i+2 < graph->n_nodes &&
                graph->nodes[i+2]->op == GGML_OP_ADD_ID &&
                graph->nodes[i+2]->src[0] == next &&
                graph->nodes[i+2]->src[2] == ids) {
                ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, ids, &local_next, graph->nodes[i+2]->src[1],
                        (const char *)next->src[0]->data, nullptr, dst_quantized.get(), (float *)graph->nodes[i+2]->data,
                        0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                ++result;
            } else {
                ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, ids, &local_next, nullptr,
                        (const char *)next->src[0]->data, nullptr, dst_quantized.get(), (float *)next->data,
                        0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
            }
            CUDA_CHECK(cudaGetLastError());

            return result;
        }
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor dst_row = *dst;

    // The heuristics src1->ne[2] <= 32*src0->ne[2] to use the mul_mat_id implementation instead of the original version
    // is derived from
    //    * DeepSeek-Lite:  64 total, 6 active experts
    //    * GPT-OSS-20B  :  32 total, 4 active experts
    //    * Qwen3-30B-A3B: 128 total, 8 active experts
    // My original hypothesis was that it is dependent on the total/active experts ratio, but from these 3 it
    // looks like it really depends just on the total number of experts.
    // TODO: verify with more models, or perhaps make the magic constant '32' to be defined via a compile time define.
    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1 &&
        ggml_cuda_can_use_mmq_id(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {

        const int64_t ne_get_rows = ne12 * n_ids;
        ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool(), ne_get_rows + ne_get_rows + n_as + 1);
        auto ids_src1 = ids_device.get();
        auto ids_dst  = ids_src1 + ne_get_rows;
        auto expert_bounds = ids_dst + ne_get_rows;

        compute_row_ids((const int32_t *)ids->data, ids_src1, ids_dst, expert_bounds,
                ne02, ne12, n_ids, ne11, nb11, nb12, ids->nb[1], stream);

        const int64_t ne11_flat = ne12*n_ids;
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        size_t nbytes_src1_q8_1 = ne11_flat*ne10_padded * sizeof(block_q8_1)/QK8_1 +
                get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), nbytes_src1_q8_1);

        size_t ts_src1 = ggml_type_size(src1->type);
        quantize_mmq_q8_1_cuda_id((const float *)src1->data, ids_src1, src1_quantized.get(),
                src0_1->type, ne10, src1->nb[1] / ts_src1, src1->nb[2] / ts_src1, src1->nb[2] / ts_src1,
                ne10_padded, ne11_flat, 1, 1, stream);

        if (src0_2) {
        ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
        ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        dst_row.data = dst_up_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[4]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        dst_row.data = dst_gate_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_2, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[5]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[5]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
            ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst->data, ggml_nelements(dst), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
        } else {
            ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                    (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                    (float *)dst->data);
        }
        } else {

            ggml_cuda_pool_alloc<char> dst_up_gate_contiguous(ctx.pool(), 2*sizeof(float)*ggml_nelements(dst));
            ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
            dst_row.ne[0] *= 2;
            dst_row.nb[1] *= 2;
            dst_row.nb[2] *= 2;
            dst_row.nb[3] *= 2;
            dst_row.data = dst_up_gate_contiguous.get();
            ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
            if (dst->src[4]) {
                GGML_ASSERT(!dst->src[5]);
                ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                        (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                        dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
                CUDA_CHECK(cudaGetLastError());
            }

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_gate_contiguous.get() + dst->ne[0], (const float *)dst_up_gate_contiguous.get(),
                        (float *)dst->data, ggml_nelements(dst), dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst), dst->ne[0],
                        (const float *)dst_up_gate_contiguous.get(), (float *)dst->data);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        if (next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
            ggml_cuda_should_use_mmq(next->src[0]->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            //ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, (char *)ids_device.get(), nullptr);
            ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, nullptr, nullptr);
            return i+1;
        }

        return i;
    }

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ggml_tensor src0_1_row = *src0_1;
    ggml_tensor src0_2_row; if (src0_2) src0_2_row = *src0_2;
    ggml_tensor src1_row   = *src1;
    ggml_tensor final_dst;
    ggml_tensor final_src;

    char * src0_1_original = (char *) src0_1->data;
    char * src0_2_original = src0_2 ? (char *) src0_2->data : nullptr;
    char * src1_original   = (char *) src1->data;
    char * dst_original    = (char *)  dst->data;

    src0_1_row.ne[2] = 1;
    src0_1_row.ne[3] = 1;
    src0_1_row.nb[3] = nb02;
    if (src0_2) {
        src0_2_row.ne[2] = 1;
        src0_2_row.ne[3] = 1;
        src0_2_row.nb[3] = nb02;
    }

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    bool fuse_down = false;
    if (next && next->op == GGML_OP_MUL_MAT_ID) {
        fuse_down = true;
        final_dst = *next;
        final_dst.ne[1] = final_dst.ne[2] = final_dst.ne[3] = 1;
        final_dst.nb[2] = final_dst.nb[3] = final_dst.nb[1];
        final_src = *next->src[0];
        final_src.ne[2] = final_src.ne[3] = 1;
        final_src.nb[3] = final_src.nb[2];
    }

    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
    bool use_quantized_src1 = false;
    int64_t src1_padded_num_cols = 0, src1_padded_row_size = 0, src1_quantized_size = 0;
    if (ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            src1_quantized_size  = src1_padded_row_size*src1->ne[2] + get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
            src1_quantized.alloc(src1_quantized_size);
            use_quantized_src1 = true;
        }
    }
    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool());
    if (!use_quantized_src1) {
        src1_contiguous.alloc(sizeof(float)*ggml_nelements(src1));
    }
    ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool()), dst_gate_contiguous(ctx.pool());
    if (src0_2) {
        dst_up_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    } else {
        dst_up_contiguous.alloc(2*sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    }
    ggml_cuda_pool_alloc<char> final_dst_contiguous(ctx.pool());
    if (fuse_down) {
        final_dst.data = final_dst_contiguous.alloc(ggml_nelements(next));
        final_dst.src[1] = &dst_row;
    }

    src1_row.data = src1_contiguous.get();

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;

    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        if (fuse_down) {
            CUDA_CHECK(cudaMemsetAsync(next->data, 0, ggml_nbytes(next), stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        }
    }

    for (int64_t i02 = 0; i02 < n_as; i02++) {
        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) continue;
        size_t mapping_offset = cum_moe_counts[i02];

        if (use_quantized_src1) {
            quantize_mmq_q8_1_id_cuda((const float *)src1->data, src1_quantized.get(), (const char *)(dev_row_mapping.get() + mapping_offset),
                    src1->ne[0], num_src1_rows, src1_padded_num_cols, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_quantized.get();
        }
        else {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_contiguous.get();
        }

        src0_1_row.data = src0_1_original + i02*nb02;
        if (src0_2_original) src0_2_row.data = src0_2_original + i02*nb02;

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        auto nb1l = nb1;
        if (!src0_2) {
            nb1l = nb1*2;
            dst_row.ne[0] = dst->ne[0] * 2;
        }

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = use_quantized_src1 ? src1_padded_row_size : nb11;
        src1_row.nb[2] = num_src1_rows*src1_row.nb[1];
        src1_row.nb[3] = num_src1_rows*src1_row.nb[1];

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1l;
        dst_row.nb[2] = num_src1_rows*nb1l;
        dst_row.nb[3] = num_src1_rows*nb1l;

        dst_row.data  =  dst_up_contiguous.get();
        if (use_quantized_src1) {
            ggml_cuda_mul_mat_q_id(ctx, &src0_1_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
        } else {
            ggml_cuda_mul_mat(ctx, &src0_1_row, &src1_row, &dst_row, nullptr, 0);
        }
        CUDA_CHECK(cudaGetLastError());

        if (dst->src[4]) {
            GGML_ASSERT(dst_row.ne[0] == dst->src[4]->ne[0]);
            dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
            dim3 grid_dims(num_src1_rows);
            k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                    (const float *)((const char *)dst->src[4]->data + i02*dst->src[4]->nb[1]), (float *)dst_row.data);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        if (src0_2) {
            dst_row.data  = dst_gate_contiguous.get();
            if (use_quantized_src1) {
                ggml_cuda_mul_mat_q_id(ctx, &src0_2_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
            } else {
                ggml_cuda_mul_mat(ctx, &src0_2_row, &src1_row, &dst_row, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            if (dst->src[5]) {
                dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
                dim3 grid_dims(num_src1_rows);
                k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                        (const float *)((const char *)dst->src[5]->data + i02*dst->src[5]->nb[1]), (float *)dst_row.data);
                CUDA_CHECK(cudaGetLastError());
            }
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                        (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get());
            }
        } else {
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_contiguous.get() + dst->ne[0], (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row)/2, dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row)/2, dst->ne[0],
                        (const float *)dst_up_contiguous.get(), (float *)dst_gate_contiguous.get());
            }
            dst_row.data = dst_gate_contiguous.get();
            dst_row.ne[0] /= 2;
            dst_row.nb[1] /= 2;
            dst_row.nb[2] /= 2;
            dst_row.nb[3] /= 2;
        }
        CUDA_CHECK(cudaGetLastError());

        if (fuse_down) {

            final_dst.ne[1] = num_src1_rows;
            final_dst.nb[1] = final_dst.ne[0]*sizeof(float);
            final_dst.nb[2] = final_dst.nb[3] = num_src1_rows*final_dst.nb[1];
            final_src.data = (char *)next->src[0]->data + i02*next->src[0]->nb[2];
            if (ggml_is_quantized(next->src[0]->type) &&
                ggml_cuda_should_use_mmq(final_src.type, ggml_cuda_info().devices[ctx.device].cc, dst_row.ne[1])) {
                ggml_cuda_mul_mat_q_id(ctx, &final_src, &dst_row, nullptr, &final_dst, nullptr, nullptr);
            } else {
                ggml_cuda_mul_mat(ctx, &final_src, &dst_row, &final_dst, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            dim3 block_dims(std::min((unsigned int)next->ne[0], 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    (char *)next->data, final_dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    next->ne[0],
                    next->nb[1], next->nb[2]);
            CUDA_CHECK(cudaGetLastError());

        }
        else {

            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_gate_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return fuse_down ? i+1 : i;
}

static void ggml_cuda_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src1 = dst->src[2];

    GGML_ASSERT(ggml_is_quantized(src0_1->type));
    GGML_ASSERT(src0_1->type == src0_2->type);
    GGML_ASSERT(src1->ne[2] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    auto stream = ctx.stream();

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*src1->ne[1];
    if (src1->ne[1] > 8) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<float> dst_up(ctx.pool(), ggml_nelements(dst));
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    if (src1->ne[1] <= 8) {
        quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded,
                src0_1->type, stream);
        CUDA_CHECK(cudaGetLastError());

        if (src1->ne[1] == 1 && src0_1->type == src0_2->type) {
            ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, src1, nullptr, dst,
                    dst->src[4], dst->src[5],
                    (const char *)src0_1->data, (const char *)src0_2->data, (const float *)src1->data, src1_quantized.get(),
                    (float *)dst->data, 0, src0_1->ne[1], 1, ne10_padded,
                    (ggml_unary_op)dst->op_params[0], stream);
            return;
        }

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, src1_quantized.get(), dst_up.get(),
                0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0_2->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {

        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[1])) {
            quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    ne10_padded, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, src1_quantized.get(), dst_up.get(),
                    0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, src1_quantized.get(), (float *)dst->data,
                    0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto local_dst = *dst;
            local_dst.data = dst_up.get();
            ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
            ggml_cuda_mul_mat(ctx, src0_2, src1, dst, nullptr, 0);
        }
    }

    ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst),
                    (const float *)dst->data, dst_up.get(), (float *)dst->data);
    CUDA_CHECK(cudaGetLastError());

}

static inline bool ops_are_same_device(const ggml_cgraph * cgraph, int first, int last) {
    if (last <= first) return true;
    int device = ((const ggml_backend_cuda_buffer_context *)cgraph->nodes[first]->buffer->context)->device;
    for (int i = first; i <= last; ++i) {
        auto node = cgraph->nodes[i];
        if (((const ggml_backend_cuda_buffer_context *)node->buffer->context)->device != device) return false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || !node->src[j]->buffer) continue;
            if (((const ggml_backend_cuda_buffer_context *)node->src[j]->buffer->context)->device != device) return false;
        }
    }
    return true;
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst, const ggml_cgraph * cgraph, int & i) {

#if IK_PRINT_TIMING
    int64_t tim1 = ggml_time_us();
#endif

    if (ggml_is_noop(dst)) {
        return true;
    }

    // In case we forget to do that in some kernel.
    ggml_cuda_set_device(ctx.device);

    auto next = i < cgraph->n_nodes - 1 ? cgraph->nodes[i+1] : nullptr;

    auto fusion = ctx.fusion;

    //printf("%4d %s(%s) on device %d. time = %ld\n", i, ggml_op_name(dst->op), dst->name, ctx.device, ggml_time_us());
    switch (dst->op) {
        case GGML_OP_REDUCE:
            ggml_cuda_op_reduce(ctx, dst);
            break;
        case GGML_OP_FAKE_CPY:
            break;
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_HADAMARD:
            ggml_cuda_op_hadamard(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_cpy_2(ctx, dst->src[0], cgraph->nodes[i+2]->src[0], dst->src[1], cgraph->nodes[i+2]->src[1])) {
                i += 2;
            } else {
                ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            }
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ADD &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                dst->src[0]->type == GGML_TYPE_F32 &&               // with split mode "attn" we can end up having f16
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                dst == cgraph->nodes[i+1]->src[0] &&
                ggml_is_contiguous(cgraph->nodes[i+1]->src[1]) &&
                ggml_are_same_shape(dst, cgraph->nodes[i+1]->src[1]) &&
                cgraph->nodes[i+1] == cgraph->nodes[i+2]->src[0] &&
                ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_fused_add_add_rms_norm(ctx, dst, cgraph->nodes[i+1], cgraph->nodes[i+2]);
                i += 2;
            }
            else if (false && fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_FUSED_RMS_NORM &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                dst == cgraph->nodes[i+1]->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_fused_add_rms_norm(ctx, dst, cgraph->nodes[i+1]);
                ++i;
            } else {
                ggml_cuda_op_add(ctx, dst);
            }
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_MULTI_ADD:
            ggml_cuda_op_multi_add(ctx, dst);
            break;
        case GGML_OP_MUL_MULTI_ADD:
            ggml_cuda_op_mul_multi_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_FUSED_MUL_UNARY:
            ggml_cuda_op_fused_mul_unary(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (fusion && i + 5 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                        cgraph->nodes[i+4]->op == GGML_OP_VIEW &&
                        cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+5)) {
                        cuda_glm45moe_experts(ctx, cgraph->nodes[i+5], cgraph->nodes[i+4]);
                        i += 5;
                    }
                    else if (fusion && i + 4 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_GROUPED_TOPK &&
                        cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+4)) {
                        cuda_bailingmoev2_experts(ctx, cgraph->nodes[i+4], cgraph->nodes[i+3]);
                        i += 4;
                    } else if (fusion && i + 2 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD && ops_are_same_device(cgraph, i, i+2)) {
                        ggml_cuda_op_biased_sigmoid(ctx, cgraph->nodes[i+2]);
                        i += 2;
                    } else {
                        ggml_cuda_op_sigmoid(ctx, dst);
                    }
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                default:
                    return -1;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_concat_cpy(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            } else {
                ggml_cuda_op_concat(ctx, dst);
            }
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_FUSED_RMS_NORM:
            if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+3], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+2]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+3]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+1], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                dst->ne[2] == 1 && cgraph->nodes[i+2]->ne[2] == 1) {
                ggml_cuda_op_fused_rms_rms_norm(ctx, dst, cgraph->nodes[i+2]);
                i += 2;
            } else {
                ggml_cuda_op_fused_rms_norm(ctx, dst);
            }
            break;
        case GGML_OP_FUSED_NORM:
            ggml_cuda_op_fused_rms_norm(ctx, dst, true);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                GGML_CUDA_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return -1;
            } else {
                i = ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst, cgraph, i);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (ggml_cuda_mul_mat_id(ctx, dst, next)) ++i;
            break;
        case GGML_OP_MOE_FUSED_UP_GATE:
            i = ggml_cuda_moe_up_gate_unary(ctx, dst, cgraph, i);
            break;
        case GGML_OP_FUSED_UP_GATE:
            ggml_cuda_up_gate_unary(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SOFTCAP:
            ggml_cuda_op_softcap(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            if (fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ARGSORT  &&
                cgraph->nodes[i+3]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS &&
                ggml_cuda_should_use_topk_moe(cgraph->nodes[i], cgraph->nodes[i+4]) &&
                ops_are_same_device(cgraph, i, i+4)) {
                if (i + 7 < cgraph->n_nodes &&
                    cgraph->nodes[i+5]->op == GGML_OP_RESHAPE  &&
                    cgraph->nodes[i+6]->op == GGML_OP_SUM_ROWS &&
                    cgraph->nodes[i+7]->op == GGML_OP_DIV) {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+7], cgraph->nodes[i+3]);
                    i += 7;
                } else {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+4], cgraph->nodes[i+3]);
                    i += 4;
                }
            } else {
                ggml_cuda_op_soft_max(ctx, dst);
            }
            break;
        case GGML_OP_SOFT_CAP_MAX:
            ggml_cuda_op_soft_cap_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            } else {
                ggml_cuda_op_rope(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROPE_FAST:
            if (fusion && i + 3 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
               (cgraph->nodes[i+2]->op == GGML_OP_RESHAPE || cgraph->nodes[i+2]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+3])) {
                i += 3;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST   &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            }
            else {
                ggml_cuda_op_rope_fast(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_CACHE:
            ggml_cuda_op_rope_cache(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SCALE &&
                cgraph->nodes[i+2]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[1] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[1] == dst &&
                cgraph->nodes[i+1]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+1]);
                ++i;
            } else {
                ggml_cuda_op_sum_rows(ctx, dst);
            }
            break;
        case GGML_OP_ARGSORT:
            if (fusion && i + 5 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+3]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+4]->op == GGML_OP_SOFT_MAX &&
                cgraph->nodes[i+5]->op == GGML_OP_RESHAPE && ops_are_same_device(cgraph, i, i+4)) {
                cuda_openai_experts(ctx, dst, cgraph->nodes[i+4]);
                i += 5;
            } else {
                ggml_cuda_op_argsort(ctx, dst);
            }
            break;
        case GGML_OP_ARGSORT_THRESH:
            ggml_cuda_op_argsort_thresh(ctx, dst);
            break;
        case GGML_OP_GROUPED_TOPK:
            ggml_cuda_op_grouped_topk(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

#if IK_PRINT_TIMING
    int64_t tim2 = ggml_time_us();
    printf("%s(%s): %d us\n", ggml_op_name(dst->op), dst->name, (int)(tim2 - tim1));
#endif

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

GGML_CALL static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cuda_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return ggml_backend_cuda_buffer_type(cuda_ctx->device);
}

GGML_CALL static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_CUDA_LOG_WARN("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        ggml_cuda_pool_alloc<half> tmp_src(cuda_ctx_src->pool());
        ggml_cuda_pool_alloc<half> tmp_dst(cuda_ctx_dst->pool());
        bool needs_f16_f32_copy = false;
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (false && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && dst->ne[1] >= 32) {
                //
                // The goal here is to reduce traffic between GPU's, which is entirely non-negligible
                // for prompt processing.
                // We cast the tensor to be copied to f16, copy the f16 data peer-to-peer
                // and then cast back to f32 on the destination side.
                // The cost for converting to/from f16 is much lower than the cost of copying
                // two times more data over PCI-E (well, at least the 30 GB/s PCI-E I have).
                // But for some reason the following is slower.
                // Can somebody tell me why?
                //

                ggml_cuda_set_device(cuda_ctx_dst->device);
                tmp_dst.alloc(ggml_nelements(dst));

                ggml_cuda_set_device(cuda_ctx_src->device);
                tmp_src.alloc(ggml_nelements(src));

                auto src_f16 = *src;
                src_f16.type = GGML_TYPE_F16;
                for (int i = 0; i < 4; ++i) src_f16.nb[i] /= 2;
                src_f16.data = tmp_src.get();

                ggml_cuda_cpy(*cuda_ctx_src, src, &src_f16, true);

                CUDA_CHECK(cudaMemcpyPeerAsync(tmp_dst.ptr, cuda_ctx_dst->device, src_f16.data, cuda_ctx_src->device, ggml_nbytes(&src_f16), cuda_ctx_src->stream()));

                needs_f16_f32_copy = true;

            } else {
#ifdef GGML_USE_NCCL__
                auto & info = ggml_cuda_info();
                auto nbytes = ggml_nbytes(src);
                ncclGroupStart();
                ggml_cuda_set_device(cuda_ctx_src->device);
                auto status1 = ncclSend(src->data, nbytes, ncclUint8, cuda_ctx_dst->device, info.nccl_coms[cuda_ctx_src->device],
                        info.all_ctx[cuda_ctx_src->device]->stream());
                ggml_cuda_set_device(cuda_ctx_dst->device);
                auto status2 = ncclRecv(dst->data, nbytes, ncclUint8, cuda_ctx_src->device, info.nccl_coms[cuda_ctx_dst->device],
                        info.all_ctx[cuda_ctx_dst->device]->stream());
                ncclGroupEnd();
                GGML_ASSERT(status1 == ncclSuccess && status2 == ncclSuccess);
                return true;
#else
                ggml_cuda_set_device(cuda_ctx_src->device);
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
            }
#endif
        }

        // record event on src stream after the copy
        ggml_cuda_set_device(cuda_ctx_src->device);
        if (!cuda_ctx_src->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        ggml_cuda_set_device(cuda_ctx_dst->device);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
        if (needs_f16_f32_copy) {
            auto dst_f16 = *dst;
            dst_f16.type = GGML_TYPE_F16;
            for (int i = 0; i < 4; ++i) dst_f16.nb[i] /= 2;
            dst_f16.data = tmp_dst.get();
            ggml_cuda_cpy(*cuda_ctx_dst, &dst_f16, dst, true);
        }
    } else {
        // src and dst are on the same backend
        printf("Why is this being invoked?\n");
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

GGML_CALL static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH

static bool check_node_graph_compatibility_and_refresh_copy_ops(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool use_cuda_graph) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph
    cuda_ctx->cuda_graph->cpy_dest_ptrs.clear();

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if (node->src[0] && node->src[0]->buffer && ggml_backend_buft_is_cuda_split(node->src[0]->buffer->buft)) {
            use_cuda_graph = false; // Split buffers are not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to split buffer %s\n", __func__, node->src[0]->name);
#endif
        }

        if (node->op == GGML_OP_MUL_MAT_ID && (node->ne[2] != 1 || node->src[2]->ne[0] != 1)) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s(%s): disabling CUDA graphs due to unsupported node type %ld %ld\n",
                    __func__, node->src[0]->name, node->ne[2], node->src[2]->ne[0]);
#endif
        }
        if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
            auto src0_1 = node->src[0];
            auto src0_2 = node->src[1];
            auto src1   = node->src[2];
            if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src1->type != GGML_TYPE_F32 ||
                !ggml_is_quantized(src0_1->type) || (src0_2 && !ggml_is_quantized(src0_2->type))) {
                use_cuda_graph = false;
            } else {
                if (i < cgraph->n_nodes-1) {
                    auto next = cgraph->nodes[i+1];
                    if (next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type)) {
                        ++i;
                    }
                }
            }
        }

        if (node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&
            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
            strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(), ffn_moe_gate_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_up_bias_prefix.c_str(), ffn_moe_up_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_down_bias_prefix.c_str(), ffn_moe_down_bias_prefix.size()) != 0) {
            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
            // by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (node->op == GGML_OP_CPY) {

            // Store the pointers which are updated for each token, such that these can be sent
            // to the device and accessed using indirection from CUDA graph
            cuda_ctx->cuda_graph->cpy_dest_ptrs.push_back((char *) node->src[1]->data);

            // store a pointer to each copy op CUDA kernel to identify it later
            void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
            if (!ptr) {
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
            }
        }
        if (!use_cuda_graph) {
            break;
        }
    }

    if (use_cuda_graph) {
        cuda_ctx->cuda_graph->use_cpy_indirection = true;
        // copy pointers to GPU so they can be accessed via indirection within CUDA graph
        ggml_cuda_cpy_dest_ptrs_copy(cuda_ctx->cuda_graph.get(), cuda_ctx->cuda_graph->cpy_dest_ptrs.data(), cuda_ctx->cuda_graph->cpy_dest_ptrs.size(), cuda_ctx->stream());
    }

    return use_cuda_graph;
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address &&
          node->op != GGML_OP_CPY &&
          node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != graph_node_properties->src_address[i] &&
            node->op != GGML_OP_CPY &&
            node->op != GGML_OP_VIEW
        ) {
            return false;
        }
    }

    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

static bool is_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {

    bool cuda_graph_update_required = false;

    if (cuda_ctx->cuda_graph->instance == nullptr) {
        cuda_graph_update_required = true;
    }

    // Check if the graph size has changed
    if (cuda_ctx->cuda_graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        cuda_graph_update_required = true;
        cuda_ctx->cuda_graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool has_matching_properties = true;
        if (!cuda_graph_update_required) {
            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            cuda_graph_update_required = true;
        }
        set_ggml_graph_node_properties(cgraph->nodes[i], &cuda_ctx->cuda_graph->ggml_graph_properties[i]);
    }

    return cuda_graph_update_required;
}

static void update_cuda_graph_executable(ggml_backend_cuda_context * cuda_ctx) {

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_CUDA_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(cuda_ctx->cuda_graph->instance));
        cuda_ctx->cuda_graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif

static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {
    // flag used to determine whether it is an integrated_gpu
    // TODO
    [[maybe_unused]] const bool integrated = false; //ggml_cuda_info().devices[cuda_ctx->device].integrated;

    //printf("======================== %s: graph with %d nodes on device %d. time = %ld\n", __func__, cgraph->n_nodes, cuda_ctx->device, ggml_time_us());
    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                    continue;
                }

#if 0
                static bool disable_fusion = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr);
                if (!disable_fusion) {
                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
                        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i+1]);
                        i++;
                        continue;
                    }

                    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
                        i += 2;
                        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i], node);
                        continue;
                    }
                }
#endif
#ifndef NDEBUG
                //assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device));
                //for (int j = 0; j < GGML_MAX_SRC; j++) {
                //    if (node->src[j] != nullptr) {
                //        assert(node->src[j]->buffer);
                //    }
                //}
#endif // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node, cgraph, i);
                if (!ok) {
                    GGML_CUDA_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }
        }
#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (cuda_ctx->cuda_graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(cuda_ctx->cuda_graph->graph));
                cuda_ctx->cuda_graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_ctx->cuda_graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (cuda_ctx->cuda_graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&cuda_ctx->cuda_graph->instance, cuda_ctx->cuda_graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            update_cuda_graph_executable(cuda_ctx);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(cuda_ctx->cuda_graph->instance, cuda_ctx->stream()));
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

GGML_CALL static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

#ifdef USE_CUDA_GRAPH
    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    bool use_cuda_graph = !disable_cuda_graphs_due_to_env && cuda_ctx->use_cuda_graph;

    // Objects required for CUDA Graph
    if (cuda_ctx->cuda_graph == nullptr) {
        cuda_ctx->cuda_graph.reset(new ggml_cuda_graph());
    }

    bool cuda_graph_update_required = false;

    if (use_cuda_graph && cuda_ctx->cuda_graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < CC_AMPERE) {
            cuda_ctx->cuda_graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && (
        cuda_ctx->cuda_graph->disable_due_to_gpu_arch ||
        cuda_ctx->cuda_graph->disable_due_to_too_many_updates ||
        cuda_ctx->cuda_graph->disable_due_to_failed_graph_capture)) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(cuda_ctx, cgraph);

        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, cgraph, use_cuda_graph);

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph && cuda_graph_update_required) {
            cuda_ctx->cuda_graph->number_consecutive_updates++;
        } else {
            cuda_ctx->cuda_graph->number_consecutive_updates = 0;
        }

        if (cuda_ctx->cuda_graph->number_consecutive_updates >= 4) {
            cuda_ctx->cuda_graph->disable_due_to_too_many_updates = true;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    if (!use_cuda_graph) {
        cuda_ctx->cuda_graph->use_cpy_indirection = false;
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SWIGLU:
                case GGML_UNARY_OP_SWIGLU_OAI:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY: return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MOE_FUSED_UP_GATE:
        case GGML_OP_FUSED_UP_GATE:
            {
                bool is_fused_up_gate = op->op == GGML_OP_MOE_FUSED_UP_GATE || op->op == GGML_OP_FUSED_UP_GATE;
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = is_fused_up_gate ? op->src[2] : op->src[1];
                if (is_fused_up_gate && op->src[1] && a->type != op->src[1]->type) {
                    fprintf(stderr, "%s: returning false for GGML_OP_MOE_FUSED_UP_GATE because src0->type != src1->type\n", __func__);
                    return false;
                }
                //==================================================================
                //if (ggml_is_quantized(a->type) && ggml_is_quantized(b->type)) {
                //    return false;
                //}
                //==================================================================
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_is_quantized(a->type)) {
                    printf("%s: returning false for op %d because (case 1)\n", __func__, (int)op->op);
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT && a->ne[3] != b->ne[3]) {
                    return false;
                }
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q6_0:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ2_KL:
                    case GGML_TYPE_IQ3_KS:
                    case GGML_TYPE_IQ4_KS:
                    case GGML_TYPE_IQ4_KSS:
                    case GGML_TYPE_IQ5_KS:
                    case GGML_TYPE_IQ2_K:
                    case GGML_TYPE_IQ2_KS:
                    case GGML_TYPE_IQ1_KT:
                    case GGML_TYPE_IQ2_KT:
                    case GGML_TYPE_IQ3_KT:
                    case GGML_TYPE_IQ4_KT:
                    case GGML_TYPE_IQ3_K:
                    case GGML_TYPE_IQ4_K:
                    case GGML_TYPE_IQ5_K:
                    case GGML_TYPE_IQ6_K:
                    case GGML_TYPE_IQ1_BN:
                    case GGML_TYPE_IQ2_BN:
                    case GGML_TYPE_IQ2_K_R4:
                    case GGML_TYPE_IQ3_K_R4:
                    case GGML_TYPE_IQ4_K_R4:
                    case GGML_TYPE_IQ4_KS_R4:
                    case GGML_TYPE_IQ5_K_R4:
                    case GGML_TYPE_IQ5_KS_R4:
                    case GGML_TYPE_IQ1_S_R4:
                    case GGML_TYPE_IQ1_M_R4:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q6_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (ggml_is_quantized(src0_type) && (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_F32)) {
                    return true;
                }
                if (ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op->src[1])) {
                    if (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                if (ggml_are_same_shape(op->src[0], op->src[1]) && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_Q8_0) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_REDUCE:
        case GGML_OP_FAKE_CPY:
        case GGML_OP_ARGMAX:
            return true;
        case GGML_OP_HADAMARD:
            return (op->ne[0] == 64 || op->ne[0] == 128 || op->ne[0] == 256) && op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_MULTI_ADD:
        case GGML_OP_MUL_MULTI_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_FUSED_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SOFTCAP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_CAP_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ROPE_FAST:
        case GGML_OP_ROPE_CACHE:
            return true;
        case GGML_OP_FUSED_NORM:
            return ggml_is_contiguous(op->src[0]);
        //case GGML_OP_ROPE:
        //    return ggml_is_contiguous(op->src[0]);
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ARGSORT_THRESH:
        case GGML_OP_GROUPED_TOPK:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
            return (op->src[0]->ne[0] == 64 && op->src[1]->type == GGML_TYPE_F16) || op->src[0]->ne[0] == 128;
#else
            return ggml_cuda_fattn_is_supported(*cuda_ctx, op);
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cuda_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    //printf("%s(%s, %s): %d, %d\n", __func__, ggml_backend_name(backend), ggml_backend_buft_name(buft), ggml_backend_buft_is_cuda_split(buft), ggml_backend_buft_is_cuda(buft));
    if (ggml_backend_buft_is_cuda_split(buft)) {
        return true;
    }

    if (ggml_backend_buft_is_cuda(buft)) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
        ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
        return buft_ctx->device == cuda_ctx->device;
    }

    return false;
}

GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    auto ctx = (const ggml_backend_cuda_context *)backend->context;
    int min_batch_size = ctx->offload_batch_size; //originally: GGML_CUDA_MIN_BATCH_OFFLOAD;

    // Why do we want to do this? The heuristics that the batch must have more than min_batch_size tokens to be worth it
    // offloading the required model weights comes from dense models. For MoE models, the average number of tokens
    // each expert deals with in a batch is (active_experts / total_experts) * batch_size. Hence, according to the
    // learned heuristics, we need (active_experts / total_experts) * batch_size >= min_batch_size.
    // Rearranging we get
    //
    //           batch_size * active_experts >= min_batch_size * total_experts
    //
    // as the condition for offloading model weights resinding in RAM to the GPU.
    // In this case, the number of tokens is not as usual in op->ne[1] but rather in op->ne[2].
    if (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE) {
        auto ids = op->op == GGML_OP_MUL_MAT_ID ? op->src[2] : op->src[3];
        int64_t batch_size = op->ne[2];
        if (batch_size < min_batch_size) return false;
        int64_t n_experts_tot    = op->src[0]->ne[2];
        int64_t n_experts_active = ids->ne[0];
        bool should_offload = batch_size*n_experts_active >= min_batch_size*n_experts_tot;
        //printf("%s(%s): op->ne[2] = %ld, n_experts_tot = %ld, n_experts_active = %ld, ids: %s, %ld x %ld x %ld x %ld -> %d (%ld, %ld)\n", __func__, op->name, op->ne[2], n_experts_tot, n_experts_active, ids->name, ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], should_offload, batch_size*n_experts_active, min_batch_size*n_experts_tot);
        return should_offload;
    }

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;

    // Original:
    //return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
    //       (op->ne[2] >= min_batch_size && (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE));

    GGML_UNUSED(backend);
}

static ggml_backend_event_t ggml_backend_cuda_event_new(ggml_backend_t backend) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .backend = */ backend,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_event_free(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));

    delete event;
}

static void ggml_backend_cuda_event_record(ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)event->backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(event->backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_event_synchronize(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .get_default_buffer_type = */ ggml_backend_cuda_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .supports_op             = */ ggml_backend_cuda_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_offload_op,
    /* .event_new               = */ ggml_backend_cuda_event_new,
    /* .event_free              = */ ggml_backend_cuda_event_free,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .event_synchronize       = */ ggml_backend_cuda_event_synchronize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

struct cuda_params {
    int  fusion = GGML_CUDA_FUSION;
    int  offload_batch_size = GGML_CUDA_MIN_BATCH_OFFLOAD;
    int  mmq_id_thresh = 32;
#ifdef USE_CUDA_GRAPH
    bool use_cuda_graph = true;
#else
    bool use_cuda_graph = false;
#endif
    bool enable_p2p = true;
};

static std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

template <typename T> bool read_value(const std::string& val, T& result) {
    std::istringstream str(val);
    T tmp; str >> tmp;
    if (!str.fail()) {
        result = tmp;
        return true;
    }
    return false;
}

static cuda_params ggml_cuda_parse_params(const char * params_string) {
    cuda_params params{};
    if (!params_string) return params;
    auto values = string_split(std::string{params_string}, ",");
    if (values.empty()) return params;
    for (auto& value : values) {
        auto parsed = string_split(value, "=");
        bool is_good = false;
        if (parsed.size() == 2) {
            if (parsed[0] == "fusion") {
                is_good = read_value(parsed[1], params.fusion);
            }
            else if (parsed[0] == "offload-batch-size") {
                is_good = read_value(parsed[1], params.offload_batch_size);
            }
            else if (parsed[0] == "mmq-id-size") {
                is_good = read_value(parsed[1], params.mmq_id_thresh);
            }
            else if (parsed[0] == "enable-p2p") {
                is_good = read_value(parsed[1], params.enable_p2p);
            }
#ifdef USE_CUDA_GRAPH
            else if (parsed[0] == "graphs") {
                is_good = read_value(parsed[1], params.use_cuda_graph);
            }
#endif
        }
        if (!is_good) {
            GGML_CUDA_LOG_WARN("%s: invalid parameter %s (%d) -> ignored\n", __func__, value.c_str(), (int)parsed.size());
        }
    }
    return params;
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, [[maybe_unused]] const void * param_string) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_CUDA_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_CUDA_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .context   = */ ctx
    };

    bool enable_p2p = true;
    if (param_string) {
        [[maybe_unused]] auto params = ggml_cuda_parse_params((const char *)param_string);
        if (params.fusion != ctx->fusion) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fusion to %d\n", __func__, params.fusion);
            ctx->fusion             = params.fusion;
        }
        if (params.offload_batch_size != ctx->offload_batch_size) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size to %d\n", __func__, params.offload_batch_size);
            ctx->offload_batch_size = params.offload_batch_size;
        }
        if (params.mmq_id_thresh != ctx->mmq_id_thresh) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting mmq_id_thresh to %d\n", __func__, params.mmq_id_thresh);
            ctx->mmq_id_thresh      = params.mmq_id_thresh;
        }
        enable_p2p = params.enable_p2p;
#ifdef USE_CUDA_GRAPH
        if (params.use_cuda_graph != ctx->use_cuda_graph) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting use_cuda_graph to %d\n", __func__, params.use_cuda_graph);
            ctx->use_cuda_graph = params.use_cuda_graph;
        }
#endif
    }

#ifdef GGML_USE_NCCL
    if (!enable_p2p) {
        printf("================== P2P disabled, but needed for NCCL\n");
        enable_p2p = true;
    }
#endif

#if !defined(GGML_CUDA_NO_PEER_COPY)
    if (enable_p2p) {
        ctx->p2p_enabled = ggml_cuda_set_peer_access(device);
    }
#endif

    return cuda_backend;
}

GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

GGML_CALL int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11100 || defined(GGML_USE_MUSA)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();

        GGML_CUDA_LOG_WARN("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init((int) (intptr_t) user_data, nullptr);
    return cuda_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cuda_reg_devices();

GGML_CALL int ggml_backend_cuda_reg_devices() {
    int device_count = ggml_backend_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
