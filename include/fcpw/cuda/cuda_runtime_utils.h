#pragma once

#include <cuda_runtime.h>
#include <fcpw/core/core.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace fcpw {

// Thin CUDA runtime helper layer used by the CUDA backend.
// It intentionally keeps memory/transfer behavior explicit and minimal.

inline void cudaCheck(cudaError_t err, const char *expr, const char *file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " for " << expr << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define FCPW_CUDA_CHECK(EXPR) ::fcpw::cudaCheck((EXPR), #EXPR, __FILE__, __LINE__)

template<typename T>
inline T *cudaAllocCopy(const std::vector<T>& hostData)
{
    // Keep buffers non-empty (mirrors existing GPU path behavior).
    T *devicePtr = nullptr;
    size_t count = hostData.size() == 0 ? 1 : hostData.size();
    FCPW_CUDA_CHECK(cudaMalloc(&devicePtr, count*sizeof(T)));

    if (!hostData.empty()) {
        FCPW_CUDA_CHECK(cudaMemcpy(devicePtr, hostData.data(), hostData.size()*sizeof(T),
                                   cudaMemcpyHostToDevice));
    }

    return devicePtr;
}

template<typename T>
inline T *cudaAllocZeroed(size_t count)
{
    // Used for output buffers written by kernels.
    T *devicePtr = nullptr;
    size_t allocCount = count == 0 ? 1 : count;
    FCPW_CUDA_CHECK(cudaMalloc(&devicePtr, allocCount*sizeof(T)));
    FCPW_CUDA_CHECK(cudaMemset(devicePtr, 0, allocCount*sizeof(T)));

    return devicePtr;
}

template<typename T>
inline void cudaDownload(const T *devicePtr, size_t count, std::vector<T>& hostData)
{
    // Synchronous host readback after stream synchronization.
    hostData.resize(count);
    if (count > 0) {
        FCPW_CUDA_CHECK(cudaMemcpy(hostData.data(), devicePtr, count*sizeof(T),
                                   cudaMemcpyDeviceToHost));
    }
}

template<typename T>
inline void cudaFreePtr(T *&ptr)
{
    if (ptr != nullptr) {
        FCPW_CUDA_CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}

} // namespace fcpw
