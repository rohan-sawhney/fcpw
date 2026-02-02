## Overview

This PR implements **watertight ray-triangle intersection** for the fcpw library, based on the algorithm described in:

> Sven Woop, Carsten Benthin, and Ingo Wald. "Watertight Ray/Triangle Intersection." *Journal of Computer Graphics Techniques (JCGT)*, Vol. 2, No. 1, 2013.

The watertight algorithm guarantees that rays passing through triangle edges or vertices will always report a hit on exactly one of the adjacent triangles, eliminating the "cracks" that can occur with standard intersection tests due to floating-point precision issues.

The implementation is:
- **Opt-in**: Enabled via an optional `watertight` parameter (default `false`)
- **Backward compatible**: Existing code continues to work unchanged
- **CPU only**: A GPU implementation was not included in this PR

---

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

8. **`include/fcpw/aggregates/mbvh.h` / `mbvh.inl`**
   - Updated `intersectFromNode()` methods to propagate watertight flag to primitive intersections

9. **`include/fcpw/aggregates/csg_node.h` / `csg_node.inl`**
   - Updated `intersectFromNode()` methods with watertight parameter

### Python Bindings

10. **`python/fcpw_py.cpp`**
    - Added `watertight` parameter (default `false`) to ray intersection methods for `scene_3D` (also present in `scene_2D` for API consistency, but has no effect there)
    - Affects single-ray intersection, multi-hit intersection, and bundled ray intersection APIs

### Python Tests

11. **`tests/fcpw_tests.py`**
    - Added `--test_watertight` command-line option to run watertight intersection tests
    - Added `test_watertight_intersection()` function that generates rays toward edge points and compares default vs watertight hit rates
    - Made `warp` and `polyscope` imports optional, allowing tests to run on macOS without NVIDIA GPU

---

## Usage

The watertight mode is opt-in and backward compatible.

### C++

```cpp
Scene<3> scene;
// ... setup scene ...
Interaction<3> i;
Ray<3> r(origin, direction);
scene.intersect(r, i, false, true);  // last param enables watertight
```

### Python

```python
import fcpw

scene = fcpw.scene_3D()
# ... setup scene ...

# Single ray
ray = fcpw.ray_3D(origin, direction)
interaction = fcpw.interaction_3D()
hit = scene.intersect(ray, interaction, False, True)  # last param enables watertight

# Bundled rays
interactions = fcpw.interaction_3D_list()
scene.intersect(ray_origins, ray_directions, ray_bounds, interactions, False, True)
```

To run the Python watertight test:
```bash
cd tests
python fcpw_tests.py --file_path input/bunny.obj --dim 3 --n_queries 1000 --test_watertight
```

---

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

---

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
