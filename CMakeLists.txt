cmake_minimum_required(VERSION 3.1.0)
project(fcpw)

option(FCPW_LINK_ENOKI "Build enoki" ON)
option(FCPW_BUILD_TESTS "Build tests" OFF)
option(FCPW_USE_EIGHT_WIDE_BRANCHING "Use 8 wide branching (default 4)" OFF)

################################################################################
# dependencies
set(FCPW_EIGEN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen-git-mirror)

if(FCPW_LINK_ENOKI)
	set(CMAKE_CXX_STANDARD 17)
	add_definitions(-DFCPW_LINK_ENOKI)
	add_subdirectory(deps/enoki)

	# define SIMD width
	string(TOUPPER "${ENOKI_ARCH_FLAGS}" ENOKI_ARCH_FLAGS_UPPER)
	message(STATUS "Enoki Max ISA: " ${ENOKI_ARCH_FLAGS_UPPER})
	if(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SSE")
		add_definitions(-DFCPW_SIMD_WIDTH=4)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX2")
		add_definitions(-DFCPW_SIMD_WIDTH=8)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "AVX")
		add_definitions(-DFCPW_SIMD_WIDTH=8)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "KNL")
		add_definitions(-DFCPW_SIMD_WIDTH=16)
	elseif(${ENOKI_ARCH_FLAGS_UPPER} MATCHES "SKX")
		add_definitions(-DFCPW_SIMD_WIDTH=16)
	else()
		add_definitions(-DFCPW_SIMD_WIDTH=4)
	endif()

	# Update the compilation flags
	enoki_set_compile_flags()
	enoki_set_native_flags()

	if(FCPW_USE_EIGHT_WIDE_BRANCHING)
		add_definitions(-DFCPW_USE_EIGHT_WIDE_BRANCHING)
	endif()
	set(FCPW_ENOKI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/enoki/include)
endif()

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${${PROJECT_NAME}_SOURCE_DIR}/include)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

################################################################################
if(FCPW_BUILD_TESTS)
  add_subdirectory(test)
endif()
