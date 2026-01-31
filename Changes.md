## Summary of Changes

### C++ Implementation

1. **`include/fcpw/core/ray.h`**
   - Added `WatertightRayData` struct containing precomputed data for watertight intersection (axis permutation indices, shear constants, conservative reciprocal directions)
   - Added specialization of `Ray<3>` with `getWatertightData()` method

2. **`include/fcpw/core/bounding_volumes.h`**
   - Added `intersectBoxConservative()` function implementing conservative ray-box intersection from the paper

3. **`include/fcpw/geometry/triangles.h` / `triangles.inl`**
   - Added `intersectWatertight()` methods implementing the watertight algorithm from the paper, with an improvement to the edge test precision (see "Implementation Note" below)
   - Added helper functions `intersectPrimitive`, `intersectPrimitiveVector`, and `intersectBox` for templated dispatch

4. **`include/fcpw/fcpw.h` / `fcpw.inl`**
   - Added `watertight` optional parameter (default `false`) to `Scene::intersect()` methods

5. **`include/fcpw/core/primitive.h`**
   - Added `watertight` parameter to `Aggregate::intersect()` and `intersectFromNode()` virtual methods
   - Updated `TransformedAggregate` to pass watertight flag through

6. **`include/fcpw/aggregates/bvh.h` / `bvh.inl`**
   - Updated `intersectFromNode()` and `processSubtreeForIntersection()` to accept watertight flag
   - When watertight is enabled, uses conservative box intersection and watertight triangle intersection

7. **`include/fcpw/aggregates/baseline.h` / `baseline.inl`**
   - Updated `intersectFromNode()` methods with watertight parameter

### GPU (Slang) Implementation

**Note:** We were unable to run this test locally due to lack of GPU hardware.  Probably a good idea for the maintainer(s) to either run the test, or reject the GPU-related changes.

8. **`include/fcpw/gpu/ray.slang`**
   - Added `WatertightRayData` struct with constructor to compute watertight data from a ray

9. **`include/fcpw/gpu/bounding-volumes.slang`**
   - Added `intersectConservative()` method to `BoundingBox` struct

10. **`include/fcpw/gpu/geometry.slang`**
    - Added `intersectTriangleWatertight()` function implementing the watertight algorithm
    - Added `intersectWatertight()` method to `IPrimitive` interface
    - Added `intersectWatertight()` to `Triangle` and `LineSegment` structs

11. **`include/fcpw/gpu/aggregate.slang`**
    - Added `intersectWatertight()` to `IAggregate` interface and `TransformedAggregate`

12. **`include/fcpw/gpu/bvh.slang`**
    - Added `intersectWatertight()` method using conservative box and watertight triangle intersection

13. **`include/fcpw/gpu/bvh.cs.slang`**
    - Added `rayIntersectionWatertight` compute shader entry point

14. **`include/fcpw/fcpw_gpu.h` / `fcpw_gpu.inl`**
    - Added `watertight` parameter to `GPUScene::intersect()` methods
    - Added `rayIntersectionWatertightShader` member
    - Dispatch to watertight shader when flag is set (3D only)

## Usage

The watertight mode is opt-in and backward compatible:

```cpp
// C++ CPU
Scene<3> scene;
// ... setup scene ...
Interaction<3> i;
Ray<3> r(origin, direction);
scene.intersect(r, i, false, true);  // last param enables watertight

// C++ GPU
GPUScene<3> gpuScene("PATH_TO_FCPW");
// ... setup ...
gpuScene.intersect(rays, interactions, false, true);  // last param enables watertight
```

## Validation / Testing

We wrote a test `watertight_tests.cpp` to verify that (i) the watertight implementation agrees with the existing (Möller-Trumbore) algorithm in typical cases and that (ii) it correctly handles edge cases needed to make ray tracing watertight.  We also added a mesh `convex.obj` to `tests/input/` to help with debugging, since this mesh eliminates many possible sources of error/confusion (e.g., exactly one intersection along any ray starting on the mesh interior).

The tests work as follows:

**Part I (Typical Behavior):**
- Generates N rays from the mesh centroid in random directions
- Compares default vs watertight intersection results
- Expected: Results should be largely similar (both methods work well for random rays)

**Part II (Edge Behavior):**
- For each of N rays, picks a random triangle, selects one of its edges, and picks a random point on that edge
- Shoots a ray from the centroid toward this edge point
- Compares default vs watertight intersection results
- Expected: The watertight method should catch significantly more intersections because shooting rays at exact edge points triggers the edge cases that the watertight algorithm handles

On test meshes `armadillo.obj`, `bunny.obj`, `convex.obj`, and `kitten.obj`, shooting N=100000 rays, the watertight method agrees with the standard method 100% of the time for ordinary intersections (Part I), and improves the hit rate from around 89% to 100% on edge intersections (Part II).

## Implementation Note: Double-Precision Edge Tests

### Original Algorithm

The original paper ("Watertight Ray/Triangle Intersection" by Woop, Benthin, and Wald) computes the edge function values U, V, W using single-precision floating-point arithmetic, with a fallback to double precision when any of these values is **exactly zero**:

```cpp
float U = Cx*By - Cy*Bx;
float V = Ax*Cy - Ay*Cx;
float W = Bx*Ay - By*Ax;

if (U == 0.0f || V == 0.0f || W == 0.0f) {
    // Recompute using double precision
}
```

The paper's reasoning is that IEEE 754 guarantees deterministic behavior: when two floating-point values are sufficiently close, their subtraction yields exactly 0.0f. Thus, an exact zero indicates the ray passes through or very near an edge, triggering the more precise fallback.

### The Failure Case

This approach has a subtle failure mode. The edge function values (U, V, W) are computed as **differences of products** (e.g., `Cx*By - Cy*Bx`), not simple subtractions. Even when a ray passes very close to a triangle edge (where, mathematically, one of U, V, W should be zero), the accumulated floating-point rounding errors across multiple operations can produce a **tiny non-zero value with an unpredictable sign**.

For example, when a ray grazes an edge:
- The mathematically correct value might be U = 0
- Single-precision computes U ≈ +1e-10 (a tiny positive number)
- V and W are both negative

The edge test checks whether all three values have the same sign:
```cpp
if ((U < 0 || V < 0 || W < 0) && (U > 0 || V > 0 || W > 0))
    return false;  // Signs disagree → reject
```

With U positive (even if infinitesimally so) and V, W negative, this test fails—the ray is incorrectly rejected. Critically, because U is not **exactly** 0.0f, the double-precision fallback never triggers.

### Our Solution

We unconditionally compute the edge function values using double precision:

```cpp
double Ud = (double)Cx * (double)By - (double)Cy * (double)Bx;
double Vd = (double)Ax * (double)Cy - (double)Ay * (double)Cx;
double Wd = (double)Bx * (double)Ay - (double)By * (double)Ax;

float U = (float)Ud;
float V = (float)Vd;
float W = (float)Wd;
```

Double precision provides approximately 15-16 significant decimal digits (versus ~7 for single precision). This dramatically reduces the likelihood of sign errors in near-edge cases. The performance impact is minimal on modern hardware, and the robustness improvement is significant—in testing, this change completely eliminated false-negative intersections on a test where every ray from the interior is shot directly toward a point on an edge.

