cmake_minimum_required(VERSION 3.1.0)
project(fcpw_test)

option(FCPW_BENCHMARK_EMBREE "Benchmark embree" OFF)


################################################################################
# c++ version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

################################################################################
# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
	"MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
	add_definitions(-DNDEBUG)
endif()

if(FCPW_BENCHMARK_EMBREE)
	add_definitions(-DFCPW_BENCHMARK_EMBREE)
	set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "Enable ispc functionality")
	set(EMBREE_TUTORIALS OFF CACHE BOOL "Enable embree tutorials")
	add_subdirectory(deps/embree)
	set(EMBREE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/embree)

	message(STATUS "Embree SSE2: " ${EMBREE_ISA_SSE2})
	message(STATUS "Embree SSE42: " ${EMBREE_ISA_SSE42})
	message(STATUS "Embree AVX: " ${EMBREE_ISA_AVX})
	message(STATUS "Embree AXV2: " ${EMBREE_ISA_AVX2})
	message(STATUS "Embree KNL: " ${EMBREE_ISA_AVX512KNL})
	message(STATUS "Embree SKX: " ${EMBREE_ISA_AVX512SKX})
endif()

add_subdirectory(deps/polyscope)

# build test to time and check aggregate correctness
add_executable(aggregate_tests aggregate_tests.cpp)
target_link_libraries(aggregate_tests fcpw)
target_include_directories(aggregate_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
target_include_directories(aggregate_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
target_include_directories(aggregate_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps/ThreadPool)
if(LINK_ENOKI)
  target_include_directories(aggregate_tests PUBLIC ${ENOKI_INCLUDES})
endif()
if(BENCHMARK_EMBREE)
  target_include_directories(aggregate_tests PUBLIC ${EMBREE_INCLUDES})
  target_link_libraries(aggregate_tests polyscope embree)
else()
  target_link_libraries(aggregate_tests polyscope)
endif()

# build csg test
add_executable(csg_tests csg_tests.cpp)
target_link_libraries(aggregate_tests fcpw)
target_include_directories(csg_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
target_include_directories(csg_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
target_include_directories(csg_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/deps/ThreadPool)
if(LINK_ENOKI)
  target_include_directories(csg_tests PUBLIC ${ENOKI_INCLUDES})
endif()
target_link_libraries(csg_tests polyscope)

