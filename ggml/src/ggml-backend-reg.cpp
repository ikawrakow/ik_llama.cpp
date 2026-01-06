// ggml/src/ggml-backend-reg.cpp
//
// Backend registration implementation.
// This file is intended to be compiled into the ggml-cpu backend DLL so
// that backend-specific registration (CUDA/Vulkan/Metal/...) is performed
// where the backend implementations are linked and available.
//
// Important: The ggml-base.dll *must not* depend on these backend-specific
// symbols at link time. The base provides the backend API (ggml-backend.cpp),
// while this file calls backend-specific registration (ggml-cuda, ggml-vulkan, ...).
//

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Ensure the base registration API is visible here.  In case the public
// headers in your tree don't expose ggml_backend_register or the typedef for
// ggml_backend_init_fn isn't visible at this point, declare the prototype
// using the explicit function-pointer type. This avoids compile errors like
// C2061/C2660 if ggml_backend_init_fn isn't defined yet.
extern GGML_CALL void ggml_backend_register(
    const char * name,
    ggml_backend_t (*init_fn)(const char * params, void * user_data),
    ggml_backend_buffer_type_t default_buffer_type,
    void * user_data);

// Forward declarations for backend registration functions.
// The exact symbol names must match the implementations in the backend modules.
// Guards ensure we only reference them when the corresponding backend is enabled.

// CPU backend registration functions (provided by the CPU backend module)
extern GGML_CALL ggml_backend_t              ggml_backend_reg_cpu_init(const char * params, void * user_data);
extern GGML_CALL ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);

// CUDA
#ifdef GGML_USE_CUDA
extern GGML_CALL void ggml_backend_cuda_reg_devices(void);
#endif

// Vulkan (mainline-style name)
#ifdef GGML_USE_VULKAN
extern GGML_CALL void ggml_backend_vulkan_reg_devices(void);
#endif

// Metal
#ifdef GGML_USE_METAL
extern GGML_CALL void ggml_backend_metal_reg_devices(void);
// If metal backend exposes init function and buffer type, declare them too:
extern GGML_CALL ggml_backend_t              ggml_backend_reg_metal_init(const char * params, void * user_data);
extern GGML_CALL ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void);
#endif

// SYCL
#ifdef GGML_USE_SYCL
extern GGML_CALL void ggml_backend_sycl_reg_devices(void);
#endif

// HIP / ROCm
#ifdef GGML_USE_HIPBLAS
extern GGML_CALL void ggml_backend_hip_reg_devices(void);
#endif

// CANN
#ifdef GGML_USE_CANN
extern GGML_CALL void ggml_backend_cann_reg_devices(void);
#endif

// MUSA
#ifdef GGML_USE_MUSA
extern GGML_CALL void ggml_backend_musa_reg_devices(void);
#endif

// Kompute
#ifdef GGML_USE_KOMPUTE
extern GGML_CALL void ggml_backend_kompute_reg_devices(void);
#endif

// RPC
#ifdef GGML_USE_RPC
extern GGML_CALL void ggml_backend_rpc_reg_devices(void);
#endif

// The exported registration entrypoint.
// Note: no 'static' here — we need external linkage so the symbol is visible from the DLL.
GGML_CALL void ggml_backend_registry_init(void) {
    static bool initialized = false;

    if (initialized) {
        return;
    }
    initialized = true;

    // register the CPU backend base entry (name / init callback / buffer type / user data)
    // The CPU registration/init functions are implemented in the CPU backend module.
    ggml_backend_register("CPU", ggml_backend_reg_cpu_init, ggml_backend_cpu_buffer_type(), NULL);

    // Register other backends only if they are enabled at compile time.
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_reg_devices();
#endif

#ifdef GGML_USE_VULKAN
    ggml_backend_vulkan_reg_devices();
#endif

#ifdef GGML_USE_METAL
    ggml_backend_metal_reg_devices();
#endif

#ifdef GGML_USE_SYCL
    ggml_backend_sycl_reg_devices();
#endif

#ifdef GGML_USE_HIPBLAS
    ggml_backend_hip_reg_devices();
#endif

#ifdef GGML_USE_CANN
    ggml_backend_cann_reg_devices();
#endif

#ifdef GGML_USE_MUSA
    ggml_backend_musa_reg_devices();
#endif

#ifdef GGML_USE_KOMPUTE
    ggml_backend_kompute_reg_devices();
#endif

#ifdef GGML_USE_RPC
    ggml_backend_rpc_reg_devices();
#endif
}

#ifdef __cplusplus
} // extern "C"
#endif
