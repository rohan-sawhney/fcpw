#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fcpw/gpu/cuda/cuda_interop_structures.h>

namespace fcpw {

// Forward declaration
template<size_t DIM>
struct SceneData;

namespace cuda {

/////////////////////////////////////////////////////////////////////////////////////////////
// Error checking utilities

inline void checkCudaError(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) fcpw::cuda::checkCudaError(call, __FILE__, __LINE__)

/////////////////////////////////////////////////////////////////////////////////////////////
// CUDA context manager

class CUDAContext {
public:
    // constructor
    CUDAContext(bool printLogs_ = false);

    // destructor
    ~CUDAContext();

    // initialize CUDA context with specified device
    void initialize(int deviceId = 0);

    // getters
    int getDeviceId() const { return deviceId; }
    cudaDeviceProp getDeviceProperties() const { return deviceProps; }
    bool isInitialized() const { return initialized; }

private:
    int deviceId;
    cudaDeviceProp deviceProps;
    bool initialized;
    bool printLogs;
};

/////////////////////////////////////////////////////////////////////////////////////////////
// GPU buffer wrapper

template<typename T>
class CUDABuffer {
public:
    // constructor
    CUDABuffer(): d_data(nullptr), count(0) {}

    // destructor
    ~CUDABuffer() {
        free();
    }

    // allocate buffer on GPU
    void allocate(size_t count_) {
        if (d_data != nullptr) {
            free();
        }

        count = count_;
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&d_data, count * sizeof(T)));
        }
    }

    // upload data from CPU to GPU
    void upload(const std::vector<T>& hostData) {
        if (hostData.size() != count) {
            std::cerr << "CUDABuffer::upload: size mismatch (buffer: "
                      << count << ", data: " << hostData.size() << ")" << std::endl;
            exit(EXIT_FAILURE);
        }

        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(d_data, hostData.data(), count * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    }

    // download data from GPU to CPU
    void download(std::vector<T>& hostData) const {
        hostData.resize(count);
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(hostData.data(), d_data, count * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }
    }

    // free GPU memory
    void free() {
        if (d_data != nullptr) {
            CUDA_CHECK(cudaFree(d_data));
            d_data = nullptr;
            count = 0;
        }
    }

    // getters
    T* devicePtr() { return d_data; }
    const T* devicePtr() const { return d_data; }
    size_t size() const { return count; }
    bool isEmpty() const { return count == 0 || d_data == nullptr; }

private:
    T* d_data;
    size_t count;

    // disable copy
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;
};

/////////////////////////////////////////////////////////////////////////////////////////////
// BVH buffers for CUDA

struct CUDABvhBuffers {
    // BVH node buffers (one active based on BVH type)
    CUDABuffer<GPUBvhNode> bvhNodes;
    CUDABuffer<GPUSnchNode> snchNodes;

    // Primitive buffers (one active based on geometry type)
    CUDABuffer<GPULineSegment> lineSegments;
    CUDABuffer<GPUTriangle> triangles;

    // Silhouette buffers (one active based on dimension)
    CUDABuffer<GPUVertex> vertices;
    CUDABuffer<GPUEdge> edges;
    CUDABuffer<GPUNoSilhouette> noSilhouettes;

    // Refit data
    CUDABuffer<uint32_t> refitNodeIndices;

    // Query/result buffers (allocated per query)
    CUDABuffer<GPURay> rays;
    CUDABuffer<GPUBoundingSphere> boundingSpheres;
    CUDABuffer<float3> randNums;
    CUDABuffer<uint32_t> flipNormalOrientation;
    CUDABuffer<GPUInteraction> interactions;

    // allocate buffers from CPU scene data
    template<size_t DIM>
    void allocate(SceneData<DIM>* sceneData,
                  bool allocateGeometry,
                  bool allocateSilhouettes,
                  bool allocateRefitData);

    // free all buffers
    void free();
};

/////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launch utilities

struct KernelLaunchConfig {
    dim3 blockDim;    // threads per block
    dim3 gridDim;     // blocks per grid
    size_t sharedMem;
    cudaStream_t stream;

    // compute launch configuration for given number of queries
    static KernelLaunchConfig compute(uint32_t nQueries,
                                      uint32_t threadsPerBlock = 256);
};

/////////////////////////////////////////////////////////////////////////////////////////////
// Timing utilities

class CUDATimer {
public:
    // constructor
    CUDATimer();

    // destructor
    ~CUDATimer();

    // start timing
    void start();

    // stop timing
    void stop();

    // get elapsed time in milliseconds
    float elapsedMilliseconds() const;

private:
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    mutable float elapsed;
};

} // namespace cuda
} // namespace fcpw
