cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw)

option(FCPW_USE_ENOKI "Build enoki" ON)
option(FCPW_USE_EIGHT_WIDE_BRANCHING "Use 8 wide branching (default 4)" OFF)
option(FCPW_ENABLE_GPU_SUPPORT "Enable GPU support" OFF)
option(FCPW_BUILD_BINDINGS "Build bindings" OFF)
option(FCPW_BUILD_DEMO "Build demo" OFF)
option(FCPW_BUILD_TESTS "Build tests" OFF)

################################################################################
if(FCPW_BUILD_BINDINGS)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
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
list(APPEND FCPW_DEPS_INCLUDES ${FCPW_EIGEN_INCLUDES})

if(FCPW_ENABLE_GPU_SUPPORT)
    set(SLANG_RHI_INSTALL             OFF CACHE BOOL " " FORCE)
    set(SLANG_RHI_ENABLE_WGPU         OFF CACHE BOOL "Disable WebGPU" FORCE)
    set(SLANG_RHI_FETCH_SLANG_VERSION "2026.1.1" CACHE STRING "Slang version to fetch")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/slang-rhi)

    if(hasParent)
        set(FCPW_SLANG_RHI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/slang-rhi/include PARENT_SCOPE)
        set(FCPW_SLANG_RHI_LIBRARY slang-rhi PARENT_SCOPE)
    else()
        set(FCPW_SLANG_RHI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/slang-rhi/include)
        set(FCPW_SLANG_RHI_LIBRARY slang-rhi)
    endif()
    list(APPEND FCPW_DEPS_INCLUDES ${FCPW_SLANG_RHI_INCLUDES})
endif()

if(FCPW_USE_ENOKI)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/enoki)
    if(hasParent)
        set(FCPW_ENOKI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/enoki/include PARENT_SCOPE)
    else()
        set(FCPW_ENOKI_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/enoki/include)
    endif()
    list(APPEND FCPW_DEPS_INCLUDES ${FCPW_ENOKI_INCLUDES})

    # Update the compilation flags
    enoki_set_compile_flags()
    enoki_set_native_flags()
endif()

################################################################################
# build library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${FCPW_DEPS_INCLUDES})
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

if(FCPW_ENABLE_GPU_SUPPORT)
    target_compile_definitions(${PROJECT_NAME} INTERFACE -DFCPW_USE_GPU)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${FCPW_SLANG_RHI_LIBRARY})
endif()

################################################################################
# build Python bindings
if(FCPW_BUILD_BINDINGS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/nanobind)
    add_subdirectory(python)
endif()

################################################################################
# build demo and tests
if(FCPW_BUILD_DEMO OR FCPW_BUILD_TESTS)
    set(FCPW_ARGS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/args)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope)
endif()

if(FCPW_BUILD_DEMO)
    add_subdirectory(demos)
endif()

if(FCPW_BUILD_TESTS)
    add_subdirectory(tests)
endif()
