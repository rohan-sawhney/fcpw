cmake_minimum_required(VERSION 3.1.0)
project(fcpw)

option(FCPW_USE_ENOKI "Build enoki" ON)
option(FCPW_USE_EIGHT_WIDE_BRANCHING "Use 8 wide branching (default 4)" OFF)	
option(FCPW_BUILD_TESTS "Build tests" ON)
option(FCPW_TESTS_BENCHMARK_EMBREE "Benchmark embree" OFF)

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
# add dependencies
get_directory_property(hasParent PARENT_DIRECTORY)

if(hasParent)
	set(FCPW_EIGEN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen-git-mirror PARENT_SCOPE)
else()
	set(FCPW_EIGEN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen-git-mirror)
endif()

if(FCPW_USE_ENOKI)
	add_subdirectory(deps/enoki)
	if(hasParent)
		set(FCPW_ENOKI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/enoki/include PARENT_SCOPE)
	else()
		set(FCPW_ENOKI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/enoki/include)
	endif()

	# Update the compilation flags
	enoki_set_compile_flags()
	enoki_set_native_flags()
endif()

################################################################################
# build library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

if(FCPW_USE_ENOKI)
	target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_USE_ENOKI)

	# define SIMD width
	string(TOUPPER "${ENOKI_ARCH_FLAGS}" ENOKI_ARCH_FLAGS_UPPER)
	message(STATUS "Enoki Max ISA: " ${ENOKI_ARCH_FLAGS_UPPER})
	if(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SSE")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_SIMD_WIDTH=4)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX2")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_SIMD_WIDTH=8)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_SIMD_WIDTH=8)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "KNL")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_SIMD_WIDTH=16)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SKX")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_SIMD_WIDTH=16)
	else()
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_SIMD_WIDTH=4)
	endif()

	if(FCPW_USE_EIGHT_WIDE_BRANCHING)
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_USE_EIGHT_WIDE_BRANCHING)
	endif()
endif()

################################################################################
# build tests
if(FCPW_BUILD_TESTS)
	add_subdirectory(tests)
endif()
