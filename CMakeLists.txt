cmake_minimum_required(VERSION 3.1.0)
project(fcpw)

option(BUILD_ENOKI "Build enoki" ON)
option(USE_EIGHT_WIDE_BRANCHING "Use 8 wide branching (default 4)" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BENCHMARK_EMBREE "Benchmark embree" OFF)
option(PROFILE "Profile" OFF)

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

################################################################################
# fcpw dependencies
if(BENCHMARK_EMBREE)
	add_definitions(-DBENCHMARK_EMBREE)
	set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "Enable ispc functionality")
	set(EMBREE_TUTORIALS OFF CACHE BOOL "Enable embree tutorials")
	add_subdirectory(deps/embree)

	message(STATUS "Embree SSE2: " ${EMBREE_ISA_SSE2})
	message(STATUS "Embree SSE42: " ${EMBREE_ISA_SSE42})
	message(STATUS "Embree AVX: " ${EMBREE_ISA_AVX})
	message(STATUS "Embree AXV2: " ${EMBREE_ISA_AVX2})
	message(STATUS "Embree KNL: " ${EMBREE_ISA_AVX512KNL})
	message(STATUS "Embree SKX: " ${EMBREE_ISA_AVX512SKX})
endif()

if(PROFILE)
	add_definitions(-DPROFILE)
	file(GLOB PROFILER_SOURCE deps/high-performance-cplusplus-profiler/profiler/*)
	add_library(profiler STATIC ${PROFILER_SOURCE})
endif()

if(BUILD_ENOKI)
	set(CMAKE_CXX_STANDARD 17)
	add_definitions(-DBUILD_ENOKI)
	set(ENOKI_AUTODIFF ON CACHE BOOL "Enable enoki autodiff")
	add_subdirectory(deps/enoki)

	# define SIMD width
	string(TOUPPER "${ENOKI_ARCH_FLAGS}" ENOKI_ARCH_FLAGS_UPPER)
	message(STATUS "Enoki Max ISA: " ${ENOKI_ARCH_FLAGS_UPPER})
	if(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SSE")
		add_definitions(-DSIMD_WIDTH=4)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX2")
		add_definitions(-DSIMD_WIDTH=8)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX")
		add_definitions(-DSIMD_WIDTH=8)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "KNL")
		add_definitions(-DSIMD_WIDTH=16)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SKX")
		add_definitions(-DSIMD_WIDTH=16)
	else()
		add_definitions(-DSIMD_WIDTH=4)
	endif()

	if(USE_EIGHT_WIDE_BRANCHING)
		add_definitions(-DUSE_EIGHT_WIDE_BRANCHING)
	endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

################################################################################
# fcpw header and source files
set(FCPW_CORE_SOURCE
	)

set(FCPW_CORE_HEADERS
	core/vector_operations.h
	core/core.h
	core/ray.h
	core/bounding_volumes.h
	core/interaction.h
	core/primitive.h
	core/csg_node.h
	core/wide_query_operations.h
	)

file(GLOB FCPW_SOURCE
	 geometry/*
	 accelerators/*
	 utilities/*
	 )

################################################################################
# fcpw library
add_library(fcpw STATIC
			${FCPW_CORE_SOURCE}
			${FCPW_CORE_HEADERS}
			${FCPW_SOURCE}
			)

target_include_directories(fcpw PUBLIC deps/eigen-git-mirror)

if(BUILD_ENOKI)
	target_include_directories(fcpw PUBLIC deps/enoki/include)
	list(APPEND ALL_FCPW_LIBS enoki-autodiff)
endif()

if(BENCHMARK_EMBREE)
	target_include_directories(fcpw PUBLIC deps/embree)
	list(APPEND ALL_FCPW_LIBS embree)
endif()

if(PROFILE)
	target_include_directories(fcpw PUBLIC deps/high-performance-cplusplus-profiler)
	list(APPEND ALL_FCPW_LIBS profiler)
endif()

list(APPEND ALL_FCPW_LIBS fcpw)

################################################################################
if(BUILD_TESTS)
	target_include_directories(fcpw PUBLIC deps/ThreadPool)
	add_subdirectory(deps/polyscope)

	add_executable(aggregate_tests tests/aggregate_tests.cpp)
	target_link_libraries(aggregate_tests ${ALL_FCPW_LIBS} polyscope)

	add_executable(csg_tests tests/csg_tests.cpp)
	target_link_libraries(csg_tests ${ALL_FCPW_LIBS} polyscope)
endif()
