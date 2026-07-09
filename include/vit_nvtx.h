#ifndef __VIT_NVTX_H__
#define __VIT_NVTX_H__

// Lightweight NVTX instrumentation for Nsight Systems profiling.
//
// All macros are no-ops unless the translation unit is compiled with -DUSE_NVTX
// (see the *_prof targets in the makefile). This keeps the normal benchmark
// build completely untouched: no NVTX headers, no runtime cost.
//
// Usage: place VIT_NVTX_RANGE("name") at the top of a scope (function body or an
// explicit { } block). The range is pushed on construction and popped when the
// scope exits, so its width on the Nsight Systems timeline is that scope's time.
// Ranges must sit on the main thread, outside any `#pragma omp parallel for`.

#ifdef USE_NVTX

#include <nvtx3/nvToolsExt.h>

struct VitNvtxScope {
    VitNvtxScope(const char* name) { nvtxRangePushA(name); }
    ~VitNvtxScope() { nvtxRangePop(); }
};

// Two-level indirection so __LINE__ expands before the ## paste, giving each
// scope a unique variable name.
#define VIT_NVTX_CONCAT_(a, b) a##b
#define VIT_NVTX_CONCAT(a, b) VIT_NVTX_CONCAT_(a, b)
#define VIT_NVTX_RANGE(name) VitNvtxScope VIT_NVTX_CONCAT(_vit_nvtx_scope_, __LINE__)(name)

#else

#define VIT_NVTX_RANGE(name) ((void)0)

#endif // USE_NVTX

#endif // __VIT_NVTX_H__
