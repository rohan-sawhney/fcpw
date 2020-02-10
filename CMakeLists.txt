cmake_minimum_required(VERSION 3.1.0)
project(fcpw)

option(BUILD_TESTS "Build tests" OFF)
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

set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "enable ispc functionality")
set(EMBREE_TUTORIALS OFF CACHE BOOL "enable embree tutorials")
add_subdirectory(deps/embree)

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
	core/shape.h
	)

file(GLOB FCPW_SOURCE
	 geometry/*
	 accelerators/*
	 constructive-solid-geometry/*
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
target_include_directories(fcpw PUBLIC ${EMBREE_INCLUDE_DIRS})
target_include_directories(fcpw PUBLIC core)

if(PROFILE)
	target_include_directories(fcpw PUBLIC deps/high-performance-cplusplus-profiler)
	list(APPEND ALL_FCPW_LIBS profiler)
endif()

list(APPEND ALL_FCPW_LIBS fcpw glog ${EMBREE_LIBRARY})

################################################################################
if(BUILD_TESTS)
	add_subdirectory(deps/polyscope)

	add_executable(geometry_tests tests/geometry_tests.cpp)
	target_link_libraries(geometry_tests ${ALL_FCPW_LIBS} polyscope)

	add_executable(csg_tests tests/csg_tests.cpp)
	target_link_libraries(csg_tests ${ALL_FCPW_LIBS} polyscope)
endif()
