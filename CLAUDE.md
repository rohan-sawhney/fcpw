# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FCPW (Fastest Closest Points in the West) is a header-only C++ library for high-performance geometric queries on 2D line segment meshes and 3D triangle meshes. It provides both CPU vectorization (via Enoki SIMD) and GPU acceleration (via Slang shading language).

**Key Query Types:**
- Closest point queries
- Ray intersections
- Silhouette queries (visibility-based)
- Containment testing
- Line-of-sight checks
- CSG operations (union, intersection, difference)

## Build Commands

### Initial Setup
```bash
# Clone with submodules
git submodule update --init --recursive

# Clone optional dependencies for demos
git clone --recurse-submodules https://github.com/nmwsharp/polyscope.git deps/polyscope
git clone --recurse-submodules https://github.com/wjakob/nanobind.git deps/nanobind
```

### C++ Build
```bash
mkdir build && cd build

# Basic build with demos and tests
cmake -DFCPW_BUILD_DEMO=ON -DFCPW_BUILD_TESTS=ON ..

# Build with GPU support enabled
cmake -DFCPW_BUILD_DEMO=ON -DFCPW_BUILD_TESTS=ON -DFCPW_ENABLE_GPU_SUPPORT=ON ..

# Build without Enoki (falls back to Eigen, no vectorization)
cmake -DFCPW_USE_ENOKI=OFF ..

# Build with 8-wide branching (default is 4-wide)
cmake -DFCPW_USE_EIGHT_WIDE_BRANCHING=ON ..

make -j8
```

**CMake Options:**
- `FCPW_USE_ENOKI` (ON/OFF) - Enable CPU SIMD vectorization
- `FCPW_ENABLE_GPU_SUPPORT` (ON/OFF) - Enable GPU acceleration
- `FCPW_BUILD_DEMO` (ON/OFF) - Build interactive demo
- `FCPW_BUILD_TESTS` (ON/OFF) - Build test suite
- `FCPW_BUILD_BINDINGS` (ON/OFF) - Build Python bindings
- `FCPW_USE_EIGHT_WIDE_BRANCHING` (ON/OFF) - Use 8-wide MBVH instead of 4-wide

### Python Build
```bash
# Install from PyPI
pip install fcpw

# Build locally (without GPU support by default)
pip install .

# Build with GPU support
pip install . --config-settings=cmake.define.FCPW_ENABLE_GPU_SUPPORT=ON
```

## Running Tests

### C++ Tests (from build/ directory)

**3D Triangle Mesh Tests - Correctness:**
```bash
./tests/aggregate_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1024 --checkCorrectness --plotInteriorPoints --computeSilhouettes --vizScene
./tests/aggregate_tests --dim=3 --tFile ../tests/input/armadillo.obj --nQueries=1024 --checkCorrectness
```

**3D Triangle Mesh Tests - Performance:**
```bash
./tests/aggregate_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1048576 --checkPerformance
```

**2D Line Segment Tests:**
```bash
./tests/aggregate_tests --dim=2 --lFile ../tests/input/plus-shape.obj --nQueries=1024 --checkCorrectness --vizScene
./tests/aggregate_tests --dim=2 --lFile ../tests/input/spiral.obj --nQueries=1048576 --checkPerformance
```

**CSG Tests:**
```bash
./tests/csg_tests --dim=3 --tFile ../tests/input/armadillo.obj --tFile ../tests/input/bunny.obj --tFile ../tests/input/kitten.obj --csgFile ../tests/input/csg.txt --instanceFile ../tests/input/instances3d.txt
```

**GPU Tests:**
```bash
./tests/gpu_tests --dim=3 --tFile ../tests/input/bunny.obj --nQueries=1048576 --plotInteriorPoints --vizScene
./tests/gpu_tests --dim=2 --lFile ../tests/input/spiral.obj --nQueries=1048576 --vizScene
```

### Python Tests (from tests/ directory)
```bash
# Basic CPU tests
python3 fcpw_tests.py --file_path=input/bunny.obj --dim=3 --n_queries=1024 --compare_with_cpu_baseline --visualize_scene

# GPU tests
python3 fcpw_tests.py --file_path=input/bunny.obj --dim=3 --n_queries=1048576 --run_gpu_queries --compare_with_warp --visualize_scene
```

### Running Demos
```bash
# From build/ directory
./demos/demo [--useGpu]

# Python demo (from demos/ directory, requires polyscope)
python -m pip install polyscope
python demo.py [--use_gpu]
```

### Compiling Slang Shaders (for GPU development)
```bash
# Compile Slang module
slangc include/fcpw/gpu/bvh.slang -o bvh.slang-module

# Compile compute shaders to GLSL
slangc include/fcpw/gpu/bvh-traversal.cs.slang -entry rayIntersection -target glsl -o bvh-traversal-ray-intersection.glsl
slangc include/fcpw/gpu/bvh-traversal.cs.slang -entry closestPoint -target glsl -o bvh-traversal-closest-point.glsl
```

## Architecture Overview

### Header-Only Template Design

FCPW is primarily header-only with template implementations in `.inl` files paired with `.h` headers. The template parameter `DIM` enables 2D and 3D operations with the same codebase.

**Compile-time Configuration:**
- `FCPW_USE_ENOKI` - Enables SIMD vectorization
- `FCPW_SIMD_WIDTH` - Auto-set to 4/8/16 based on CPU ISA (SSE/AVX/AVX512)
- `FCPW_USE_GPU` - Enables GPU code paths
- `FCPW_USE_EIGHT_WIDE_BRANCHING` - Switches MBVH between 4-way and 8-way branching

### Core Architecture Layers

**1. Core Primitives (`include/fcpw/core/`)**
- `core.h` - Type definitions, SIMD abstractions (Enoki packets), Vector/Transform types
- `primitive.h` - Abstract base classes: `Primitive`, `GeometricPrimitive`, `Aggregate`
- `ray.h` - Ray structure with origin, direction, inverse direction
- `bounding_volumes.h` - AABB and BoundingSphere with overlap/intersection tests
- `interaction.h` - Query result container (distance, point, normal, UV, indices)
- `wide_query_operations.h` - Vectorized intersection/overlap tests for SIMD

**2. Acceleration Structures (`include/fcpw/aggregates/`)**
- `bvh.h/inl` - Binary Bounding Volume Hierarchy with SAH construction
- `mbvh.h/inl` - Multi-level BVH for vectorized queries (4-way or 8-way branching)
- `baseline.h/inl` - Naive brute-force structure (for testing/validation)
- `csg_node.h/inl` - CSG tree nodes for boolean operations

**BVH Construction Heuristics:**
- `Bvh_SurfaceArea` - Surface Area Heuristic (recommended for optimal query performance)
- `Bvh_OverlapSurfaceArea` - SAH considering spatial overlap
- `Bvh_LongestAxisCenter` - Simple midpoint split on longest axis
- `Bvh_Volume` / `Bvh_OverlapVolume` - Volume-based heuristics

**3. Geometric Primitives (`include/fcpw/geometry/`)**
- `polygon_soup.h` - Generic indexed mesh container
- `triangles.h/inl` - 3D triangle with Möller-Trumbore ray intersection
- `line_segments.h/inl` - 2D line segments (embedded in 3D)
- `silhouette_edges.h/inl` - 3D silhouette edges for visibility queries
- `silhouette_vertices.h/inl` - 2D silhouette vertices

**4. Scene Management (`include/fcpw/utilities/`)**
- `scene_data.h` - Geometry loading and storage
- High-level API for setting vertices, indices, instance transforms, CSG trees

**5. GPU Code (`include/fcpw/gpu/`)**
- `.slang` files - Slang shaders for CUDA/HLSL/Vulkan/Metal
- `bvh.slang` - BVH traversal and geometric queries
- `geometry.slang` - Primitive intersection code
- `bvh-traversal.cs.slang` - Compute shaders for batch queries
- `bvh.cs.slang` - Compute shader for BVH refitting
- C++ interop structures for CPU-GPU data transfer

### Key Data Flow

**Building a Scene:**
1. `Scene<DIM>::setObjectCount(n)` - Initialize geometry
2. `Scene<DIM>::setObjectVertices()` / `setObjectTriangles()` - Load data
3. `Scene<DIM>::build(aggregateType, vectorize)` - Construct BVH using SAH
   - Creates `BvhNode<DIM>` hierarchy or `MbvhNode<DIM, WIDTH>` for vectorization
   - Primitives stored in `PolygonSoup` with indices into vertices
4. (Optional) `Scene<DIM>::computeSilhouettes()` - Precompute SNCH data

**Query Execution (CPU):**
1. Single queries: `scene.findClosestPoint(point, interaction)`
   - Traverses BVH depth-first with distance-ordered node visits
   - Prunes nodes where `box.computeSquaredDistance(point) > bestDistance^2`
2. Batch queries: `scene.findClosestPoints(points[], interactions[])`
   - Uses MBVH with Enoki SIMD packets (4-wide or 8-wide)
   - Parallel traversal of multiple query points simultaneously
   - Intel TBB for multi-threading across batches

**Query Execution (GPU):**
1. `GPUScene<DIM>::transferToGPU(scene)` - Upload BVH and geometry
2. `gpuScene.findClosestPoints(spheres[], interactions[])` - Launch compute shader
   - Thread groups of 64+ threads process queries in parallel
   - Each thread traverses BVH independently
   - Results written to output buffer and read back to CPU

### CPU vs GPU Code Paths

**CPU Path:**
- Enoki SIMD types (`FloatP<WIDTH>`, `IntP<WIDTH>`) for vectorization
- `MbvhNode<DIM>` with 4-way or 8-way branching
- Intel TBB for multi-core parallelism
- Eigen for linear algebra

**GPU Path:**
- Slang compute shaders compiled to CUDA/HLSL/Vulkan/Metal
- Slang-RHI library for cross-API resource management
- CPU builds BVH, GPU consumes read-only structure
- GPU supports refitting via `bvh.cs.slang` compute shader

### Advanced Features

**Silhouette Queries (SNCH - Silhouette Normal Cone Hierarchy):**
- `computeSilhouettes()` identifies boundary edges/vertices
- Normal cones stored per BVH node enable cone-based pruning
- `findClosestSilhouettePoint()` finds closest visibility boundary point

**CSG Operations:**
- `CsgTreeNode` wraps multiple aggregates with boolean operations
- Supports union, intersection, difference
- Hierarchical evaluation during queries

**Instancing:**
- `setObjectInstanceTransforms()` creates multiple transform copies
- Single BVH shared across all instances
- Transforms applied during query traversal

**Dynamic Refitting:**
- `updateObjectVertex()` / `updateObjectVertices()` modify geometry
- `refit()` updates BVH bounding boxes without full rebuild
- GPU refitting supported via compute shader

## Development Guidelines

### Adding New Primitive Types

1. Create new primitive class in `include/fcpw/geometry/` inheriting from `GeometricPrimitive<DIM>`
2. Implement required methods:
   - `intersect(ray, interaction)` - Ray intersection
   - `findClosestPoint(point, interaction)` - Closest point query
   - `computeBoundingBox()` - AABB computation
   - `getCentroid()` - Centroid for BVH partitioning
   - `getSurfaceArea()` / `getVolume()` - For SAH heuristic
3. Add vectorized variants in `wide_query_operations.h` if using MBVH
4. Update `PrimitiveType` enum and factory methods in `scene_data.h`

### Modifying BVH Construction

BVH construction logic is in `include/fcpw/aggregates/bvh.inl`:
- `buildRecursive()` - Main recursive builder
- `computeObjectSplit()` - SAH split computation
- Key parameters: `leafSize` (max primitives per leaf), `nBuckets` (SAH buckets)

### GPU Shader Development

1. Modify `.slang` files in `include/fcpw/gpu/`
2. BVH traversal kernels in `bvh-traversal.cs.slang`
3. Test with `gpu_tests` executable
4. Slang compilation happens at runtime via Slang-RHI

### Testing Strategy

- `aggregate_tests.cpp` - Correctness and performance testing for BVH
- `gpu_tests.cpp` - Validates GPU results match CPU results
- `csg_tests.cpp` - Boolean operations testing
- Use `--checkCorrectness` to compare against brute-force baseline
- Use `--checkPerformance` for timing measurements
- Use `--vizScene` for Polyscope visualization

## Common Pitfalls

**GPU Support on Windows:**
- May require downloading `dxil.dll` and `dxcompiler.dll` from DirectX Shader Compiler releases
- Copy to `C:\Windows\System32\` (64-bit) or `C:\Windows\SysWOW64\` (32-bit)

**Vectorized BVH:**
- GPU requires non-vectorized BVH: `scene.build(aggregateType, false)`
- CPU can use vectorized: `scene.build(aggregateType, true)`
- Cannot transfer vectorized (MBVH) to GPU

**Dependencies:**
- Eigen is always required (in `deps/eigen-git-mirror/`)
- Enoki is optional but recommended (enables SIMD)
- Polyscope required for demos and tests
- nanobind required for Python bindings
- Slang-RHI required for GPU support

**Template Instantiation:**
- Most code is in `.inl` files paired with `.h` headers
- Explicitly instantiate templates if creating new compilation units
- Common instantiations: `Scene<2>`, `Scene<3>`

## File Naming Conventions

- `.h` files - Interface declarations
- `.inl` files - Template implementations (included by `.h` files)
- `.slang` files - GPU shader code (Slang language)
- `.cs.slang` files - Compute shaders

## External References

- **Full API Documentation:** See `include/fcpw/fcpw.h` for complete Scene API
- **GPU API Documentation:** See `include/fcpw/fcpw_gpu.h` for GPUScene API
- **Slang Language:** https://shader-slang.com/slang/user-guide/
- **Enoki SIMD:** https://github.com/mitsuba-renderer/enoki
- **Polyscope Visualization:** https://polyscope.run

## Future Roadmap (from roadmap.txt)

- Multiple geometry instances on GPU backend
- Additional primitive types: spheres, beziers, NURBS, subdivision surfaces
- Traversal optimizations: stackless traversal, quantized bounding boxes
- SNCH improvements: unoriented meshes, tighter normal cones
- Oriented bounding boxes as alternative to AABBs
- Spatial split BVH
- Packet queries for batch distance calculations
