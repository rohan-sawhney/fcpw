# CUDA Backend

The CUDA backend provides native GPU acceleration for FCPW's geometric queries on NVIDIA GPUs. It compiles BVH traversal kernels offline with `nvcc`, avoiding the runtime shader compilation used by the Slang/GPU backend.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (provides `nvcc`, `cuda_runtime.h`, and `cudart`)
- CMake 3.18+ (for `CUDA_ARCHITECTURES` support)

The build system auto-detects the GPU architecture on the build machine via `CUDA_ARCHITECTURES native`.

## Building

### With CUDA support only

```bash
mkdir build && cd build
cmake -DFCPW_ENABLE_CUDA_SUPPORT=ON -DFCPW_BUILD_TESTS=ON ..
make -j8
```

### With both CUDA and Slang/GPU backends

The CUDA and Slang/GPU backends can coexist in the same build:

```bash
cmake -DFCPW_ENABLE_CUDA_SUPPORT=ON -DFCPW_ENABLE_GPU_SUPPORT=ON -DFCPW_BUILD_TESTS=ON ..
make -j8
```

### CMake details

Enabling `FCPW_ENABLE_CUDA_SUPPORT` does three things:

1. Compiles `include/fcpw/cuda/cuda_kernels.cu` into a static library (`fcpw_cuda_kernels`) with separable compilation enabled.
2. Adds the `FCPW_USE_CUDA` compile definition to the `fcpw` interface target.
3. Links `fcpw_cuda_kernels` and `CUDA::cudart` to the `fcpw` interface target.

## Usage

Include `<fcpw/fcpw_cuda.h>` and use `CUDAScene<DIM>`:

```cpp
#include <fcpw/fcpw.h>
#include <fcpw/fcpw_cuda.h>

using namespace fcpw;

// 1. Build a non-vectorized BVH on the CPU
Scene<3> scene;
scene.setObjectCount(1);
scene.setObjectVertices(vertices, 0);
scene.setObjectTriangles(triangles, 0);
// For silhouette queries, call scene.computeSilhouettes() before build
scene.build(AggregateType::Bvh_OverlapSurfaceArea, false /* vectorize must be false */);

// 2. Transfer to GPU
CUDAScene<3> cudaScene;
cudaScene.transferToGPU(scene);

// 3. Run queries
std::vector<CUDABoundingSphere> spheres = /* ... */;
std::vector<CUDAInteraction> interactions;
cudaScene.findClosestPoints(spheres, interactions);
```

### Supported queries

| Method | Description |
|---|---|
| `intersect(rays, interactions)` | Ray intersection (closest hit or occlusion) |
| `intersect(spheres, randNums, interactions)` | Sphere intersection with random primitive sampling |
| `findClosestPoints(spheres, interactions)` | Closest point on geometry |
| `findClosestSilhouettePoints(spheres, flip, interactions)` | Closest point on visibility silhouette |

Each query method has two overloads: one accepting `std::vector` of CUDA types directly, and one accepting `Eigen::MatrixXf`/`Eigen::VectorXf` for convenience (the Eigen overloads convert internally with multi-threaded packing).

### BVH refit

After modifying geometry on the CPU (via `Scene::updateObjectVertex`), refit the GPU BVH without a full rebuild:

```cpp
scene.updateObjectVertex(/* ... */);
cudaScene.refit(scene);  // uploads updated geometry and refits bounding boxes
```

If geometry is updated directly on the GPU (in user CUDA kernels), pass `updateGeometry=false`:

```cpp
cudaScene.refit(scene, false);  // refits bounding boxes only
```

### Limitations

- Single object scenes only (no CSG trees, instancing, or nested aggregates).
- The CPU BVH must be non-vectorized (`vectorize=false` in `Scene::build`).
- GPU queries are **not thread-safe** -- do not call query methods concurrently from multiple host threads.

## Running tests

Build the test executable:

```bash
cd build
cmake -DFCPW_ENABLE_CUDA_SUPPORT=ON -DFCPW_BUILD_TESTS=ON ..
make cuda_tests -j8
```

### 3D triangle mesh

```bash
./tests/cuda_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1024
```

### 3D with silhouette queries

```bash
./tests/cuda_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1024 --computeSilhouettes
```

### 2D line segments

```bash
./tests/cuda_tests --dim=2 --lFile ../tests/input/spiral.obj --nQueries=1024
```

### 3D with BVH refit

```bash
./tests/cuda_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1024 --refitBvh
```

### Performance benchmark

```bash
./tests/cuda_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1048576
```

All tests compare CUDA results against CPU results and print `done` on success. Any mismatches are printed with full query details.

### Test flags

| Flag | Description |
|---|---|
| `--dim=N` | Scene dimension: 2 or 3 (required) |
| `--tFile PATH` | Triangle mesh OBJ file |
| `--lFile PATH` | Line segment OBJ file |
| `--nQueries=N` | Number of queries (default: 1048576) |
| `--computeSilhouettes` | Enable SNCH silhouette queries |
| `--refitBvh` | Test BVH refit path |
| `--vizScene` | Visualize with Polyscope |
| `--plotInteriorPoints` | Show interior points in visualization |

## File overview

| File | Description |
|---|---|
| `cuda_types.h` | Host-side POD types (`CUDAFloat3`, `CUDABvhNode`, etc.) with no `cuda_runtime.h` dependency |
| `cuda_interop_structures.h` | Extracts CPU BVH data into CUDA-compatible arrays |
| `cuda_bvh_device.cuh` | Device code: math helpers, BVH traversal, primitive intersection, refit |
| `cuda_kernels.h` | Host-callable kernel launch declarations |
| `cuda_kernels.cu` | CUDA kernel definitions and launch wrappers (compiled by nvcc) |
| `fcpw_cuda.h` | Public API: `CUDAScene<DIM>` class declaration |
| `fcpw_cuda.inl` | `CUDAScene` implementation: GPU memory management, query dispatch |

## CUDA vs Slang/GPU backend

| | CUDA backend | Slang/GPU backend |
|---|---|---|
| CMake option | `FCPW_ENABLE_CUDA_SUPPORT` | `FCPW_ENABLE_GPU_SUPPORT` |
| Shader language | CUDA C++ | Slang |
| Compilation | Offline (nvcc) | Runtime (Slang-RHI) |
| GPU vendors | NVIDIA only | NVIDIA, AMD, Intel, Apple |
| Dependency | CUDA Toolkit | Slang-RHI submodule |
| API class | `CUDAScene<DIM>` | `GPUScene<DIM>` |
