#FCPW

FCPW is a header only C++ library for fast closest point and ray intersection queries. It is about 3x faster than <a href="https://www.embree.org">Embree</a> for closest point queries and is only slightly slower for ray
intersection queries (0.8x) (see [Benchmarks](#Benchmarks)). FCPW uses a wide BVH with vectorized traversal to accelerate queries to geometric primitives, with additional support for constructive solid geometry and instancing. The geometric primitives currently supported are triangles, line segments and/or a mixture of the two, though it is fairly easy to add support for other types of primitives in the library. FCPW uses <a href="http://rgl.epfl.ch/people/wjakob">Wenzel Jakob</a>'s amazing <a href="https://github.com/mitsuba-renderer/enoki">Enoki</a> library for vectorization, though it falls back to <a href="http://eigen.tuxfamily.org/index.php?title=Main_Page">Eigen</a> if Enoki is not included in the project. In the latter case, FCPW performs queries with a non-vectorized binary BVH.

#Including FCPW

FCPW does not require compiling or installing, you just need to

```
#define FCPW_USE_ENOKI ON
#define FCPW_SIMD_WIDTH 8
#include <fcpw/fcpw.h>
```

in your code. Alternatively, you can add the following lines to your CMakeLists.txt file

```
add_subdirectory(fcpw)
target_link_libraries(YOUR_TARGET fcpw)
target_include_directories(YOUR_TARGET PUBLIC ${FCPW_EIGEN_INCLUDES})
if(FCPW_USE_ENOKI)
	target_include_directories(YOUR_TARGET PUBLIC ${FCPW_ENOKI_INCLUDES})
endif()
```

#API

TODO

#Benchmarks

TODO

#Author
[Rohan Sawhney](http://www.rohansawhney.io)<br/>, with support from Ruihao Ye and Johann Korndoerfer

#License

Released under the [MIT License](https://opensource.org/licenses/MIT)
