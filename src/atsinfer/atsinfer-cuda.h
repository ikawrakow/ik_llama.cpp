#ifndef ATSINFER_CUDA_H
#define ATSINFER_CUDA_H

#include <string>
#include <vector>
#include <cstddef>

class ATSInferCudaManager {
public:
    ATSInferCudaManager();
    ~ATSInferCudaManager();

    bool init(int device_id = 0);
    void cleanup();

    // Allocate pinned host memory
    void * alloc_pinned_host(size_t size_bytes);
    void free_pinned_host(void * ptr);

    // Allocate device memory
    void * alloc_device(size_t size_bytes);
    void free_device(void * ptr);

    // Create CUDA event handle
    void * create_event();
    void destroy_event(void * event_handle);

    // Asynchronous H2D transfer on dedicated transfer stream; records event
    bool migrate_h2d_async(
        void * device_dst,
        const void * host_src,
        size_t size_bytes,
        void * event_handle
    );

    // Make compute stream wait on transfer completion event
    bool wait_for_transfer_event(void * event_handle);

    void * get_compute_stream() const;
    void * get_transfer_stream() const;
    bool is_initialized() const;

private:
    int device;
    void * compute_stream;  // cudaStream_t
    void * transfer_stream; // cudaStream_t
    bool initialized;
};

#endif // ATSINFER_CUDA_H
