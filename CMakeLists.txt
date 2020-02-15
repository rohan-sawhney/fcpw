cmake_minimum_required(VERSION 3.1.0)
project(fcpw)

option(BUILD_TESTS "Build tests" ON)
option(BENCHMARK_EMBREE "Benchmark Embree" ON)
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

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/deps/glog/cmake")
	message(FATAL_ERROR "The glog submodule directory is missing! "
	"You probably did not clone the project with --recursive, or you first checked out "
	"fcpw before it was added. It is possible to recover by running "
	"\"git submodule update --init --recursive\"")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
	add_definitions(-DNDEBUG)
endif()

################################################################################
# fcpw dependencies
set(WITH_GFLAGS OFF CACHE BOOL "Use gflags")
add_subdirectory(deps/glog)

if(BENCHMARK_EMBREE)
	set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "enable ispc functionality")
	set(EMBREE_TUTORIALS OFF CACHE BOOL "enable embree tutorials")
	add_subdirectory(deps/embree)
endif()

if(PROFILE)
	add_definitions(-DPROFILE)
	file(GLOB PROFILER_SOURCE deps/high-performance-cplusplus-profiler/profiler/*)
	add_library(profiler STATIC ${PROFILER_SOURCE})
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

################################################################################
# fcpw header and source files
set(FCPW_CORE_SOURCE
	)

set(FCPW_CORE_HEADERS
	core/core.h
	core/ray.h
	core/bounding_volumes.h
	core/interaction.h
	core/primitive.h
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

target_include_directories(fcpw PUBLIC deps/glog/src ${CMAKE_BINARY_DIR}/deps/glog)
target_include_directories(fcpw PUBLIC deps/eigen-git-mirror)
target_include_directories(fcpw PUBLIC deps/ThreadPool)
target_include_directories(fcpw PUBLIC core)

if(BENCHMARK_EMBREE)
	target_include_directories(fcpw PUBLIC deps/embree)
	list(APPEND ALL_FCPW_LIBS embree)
endif()

if(PROFILE)
	target_include_directories(fcpw PUBLIC deps/high-performance-cplusplus-profiler)
	list(APPEND ALL_FCPW_LIBS profiler)
endif()

list(APPEND ALL_FCPW_LIBS fcpw glog)

################################################################################
if(BUILD_TESTS)
	add_subdirectory(deps/polyscope)

	add_executable(aggregate_tests tests/aggregate_tests.cpp)
	target_link_libraries(aggregate_tests ${ALL_FCPW_LIBS} polyscope)

	add_executable(csg_tests tests/csg_tests.cpp)
	target_link_libraries(csg_tests ${ALL_FCPW_LIBS} polyscope)
endif()
