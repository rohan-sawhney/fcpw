<p align="center">
<img src="logo.png" height="140" width="100">
</p>
<h1 align="center"><em>FCPW: Fastest Closest Points in the West</em></h1>

*FCPW* is a lightweight, header-only C++ library for performing fast geometric queries such as ray intersection, closest point and closest silhouette point queries ([1](http://www.rohansawhney.io/WalkOnStars.pdf), [2](https://dl.acm.org/doi/abs/10.1145/364338.364380)). It is about 3x faster than <a href="https://www.embree.org">Embree</a> for closest point queries and is only slightly slower for ray intersection queries (0.8x).

*FCPW* uses a wide BVH with vectorized traversal to accelerate queries to geometric primitives, with additional support for constructive solid geometry (CSG) and instancing. Geometric primitives currently supported include triangles, line segments or a mixture of the two. Support for other types of geometry (e.g., beziers) can be added by extending the [GeometricPrimitive](https://github.com/rohan-sawhney/fcpw/blob/master/include/fcpw/core/primitive.h) API.

*FCPW* uses <a href="https://github.com/mitsuba-renderer/enoki">Enoki</a> for vectorization, but falls back to <a href="http://eigen.tuxfamily.org/index.php?title=Main_Page">Eigen</a> if Enoki is not included in the project. In the latter case, *FCPW* performs queries with a non-vectorized binary BVH.

# Including *FCPW*

Add the following lines to your CMakeLists.txt file

```
add_subdirectory(fcpw)
target_link_libraries(YOUR_TARGET fcpw)
target_include_directories(YOUR_TARGET PUBLIC ${FCPW_EIGEN_INCLUDES})
target_include_directories(YOUR_TARGET PUBLIC ${FCPW_ENOKI_INCLUDES})
```

to include *FCPW* in your project. Then, simply add

```
#include <fcpw/fcpw.h>
```

in the file where you want to use *FCPW*. If you are not using Cmake, then you'll have to define a few extra variables

```
#define FCPW_USE_ENOKI
#define FCPW_SIMD_WIDTH 4 // change based on the SIMD width supported by your machine
#include <fcpw/fcpw.h>
```

and include Eigen and Enoki (optional) independently into your project. If you plan on building and running the tests, clone the following projects into the `deps` folder

```
git clone https://github.com/embree/embree.git deps/embree
git clone https://github.com/wjakob/tbb.git deps/tbb
git clone --recurse-submodules https://github.com/nmwsharp/polyscope.git deps/polyscope
```

# API

The easiest and most direct way to use *FCPW* is through its <a href="https://github.com/rohan-sawhney/fcpw/blob/master/include/fcpw/fcpw.h">Scene</a> API which provides methods to load the geometry, build the acceleration structure and perform geometric queries. Here is an example on a scene consisting of only triangles:

```c++
// initialize a 3d scene
Scene<3> scene;

// set the PrimitiveType for each object in the scene;
// in this case, we have a single object consisting of triangles
scene.setObjectTypes({{PrimitiveType::Triangle}});

// set the vertex and triangle count of the (0th) object
scene.setObjectVertexCount(nVertices, 0);
scene.setObjectTriangleCount(nTriangles, 0);

// specify the vertex positions
for (int i = 0; i < nVertices; i++) {
	scene.setObjectVertex(positions[i], i, 0);
}

// specify the triangle indices
for (int i = 0; i < nTriangles; i++) {
	scene.setObjectTriangle(&indices[3*i], i, 0);
}

// compute vertex & edge normals (optional)
scene.computeObjectNormals(0);

// compute silhouette data (required only for closest silhouette point queries)
scene.computeSilhouettes();

// now that the geometry has been specified, build the acceleration structure
scene.build(AggregateType::Bvh_SurfaceArea, true); // the second boolean argument enables vectorization

// perform a closest point query
Interaction<3> cpqInteraction;
scene.findClosestPoint(queryPoint, cpqInteraction);

// perform a closest silhouette point query
Interaction<3> cspqInteraction;
scene.findClosestSilhouettePoint(queryPoint, cspqInteraction);

// perform a ray intersection query
std::vector<Interaction<3>> rayInteractions;
scene.intersect(queryRay, rayInteractions, false, true); // don't check for occlusion, and record all hits
```

Notice that `Scene` is templated on dimension, which enables *FCPW* to work with geometric data in any dimension out of the box as long geometric primitives are defined for the dimension of interest. The <a href="https://github.com/rohan-sawhney/fcpw/blob/master/include/fcpw/core/interaction.h">Interaction</a> object stores information relevant to the query, such as the distance to the geometric primitive and the closest/intersection point on the primitive.

*FCPW* can additionally compute the normal at the closest/intersection point, though this must be explicitly requested through the `computeObjectNormals` method in the `Scene` class. Furthermore, it is possible to load multiple objects, possibly with mixed primitives and instance transforms, into a scene. A CSG tree can also be built via the `setCsgTreeNode` method. More details can be found in <a href="https://github.com/rohan-sawhney/fcpw/blob/master/include/fcpw/fcpw.h">fcpw.h</a>.

If your scene contains multiple objects with the same type of primitives (e.g. triangles), it is better to "flatten" those objects into a single object before loading the geometry into *FCPW*. In the latter case, *FCPW* builds a single acceleration structure over all geometric primitives in the scene, while in the former it builds a hierarchy of acceleration structures, with an acceleration structure for each object in the scene.

# Benchmarks

See this [report](https://thenumb.at/assets/projects/fcpw-report.pdf) by [Max Slater](https://thenumb.at).

# Author
[Rohan Sawhney](http://www.rohansawhney.io), with support from [Max Slater](https://thenumb.at), Ruihao Ye, [Keenan Crane](https://www.cs.cmu.edu/~kmcrane/) and Johann Korndoerfer.

# License

Released under the [MIT License](https://opensource.org/licenses/MIT).
