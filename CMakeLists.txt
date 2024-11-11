cmake_minimum_required(VERSION 3.14...3.27)
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

if(FCPW_ENABLE_GPU_SUPPORT)
    # download slang
    set(FCPW_SLANG_VERSION "2024.14.5")
    set(FCPW_SLANG_URL_BASE "https://github.com/shader-slang/slang/releases/download/v${FCPW_SLANG_VERSION}")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(FCPW_SLANG_FILENAME "slang-${FCPW_SLANG_VERSION}-linux-x86_64-glibc-2.17.tar.gz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(FCPW_SLANG_FILENAME "slang-${FCPW_SLANG_VERSION}-windows-x86_64.zip")
    else()
        # unfortunately macOS is not currently officially supported by gfx
        message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
    endif()

    set(FCPW_SLANG_URL "${FCPW_SLANG_URL_BASE}/${FCPW_SLANG_FILENAME}")
    set(FCPW_SLANG_DOWNLOAD_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${FCPW_SLANG_FILENAME}")
    set(FCPW_SLANG_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/slang")

    if(NOT EXISTS ${FCPW_SLANG_OUTPUT_DIR})
        file(DOWNLOAD ${FCPW_SLANG_URL} ${FCPW_SLANG_DOWNLOAD_PATH})
        file(MAKE_DIRECTORY ${FCPW_SLANG_OUTPUT_DIR})

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            execute_process(COMMAND tar -xzf ${FCPW_SLANG_DOWNLOAD_PATH} -C ${FCPW_SLANG_OUTPUT_DIR}
                            WORKING_DIRECTORY ${FCPW_SLANG_OUTPUT_DIR})
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
            find_program(UNZIP_EXECUTABLE unzip)
            if(UNZIP_EXECUTABLE)
                execute_process(COMMAND ${UNZIP_EXECUTABLE} -o ${FCPW_SLANG_DOWNLOAD_PATH} -d ${FCPW_SLANG_OUTPUT_DIR}
                                WORKING_DIRECTORY ${FCPW_SLANG_OUTPUT_DIR})
            else()
                message(FATAL_ERROR "Unzip utility not found!")
            endif()
        endif()
    else()
        message(STATUS "Slang already downloaded and extracted at ${FCPW_SLANG_OUTPUT_DIR}. Skipping download.")
    endif()

    file(REMOVE ${FCPW_SLANG_DOWNLOAD_PATH})

    # set the include directory
    if(hasParent)
        set(FCPW_SLANG_INCLUDES ${FCPW_SLANG_OUTPUT_DIR}/include PARENT_SCOPE)
    else()
        set(FCPW_SLANG_INCLUDES ${FCPW_SLANG_OUTPUT_DIR}/include)
    endif()

    # find the .lib/.so files
    set(SLANG_LIBRARY_PATH ${FCPW_SLANG_OUTPUT_DIR}/lib)
    find_library(FCPW_SLANG_LIBRARY NAMES slang PATHS ${SLANG_LIBRARY_PATH})
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        find_library(FCPW_SLANG_GLSLANG_LIBRARY NAMES slang-glslang PATHS ${SLANG_LIBRARY_PATH})
    else()
        set(FCPW_SLANG_GLSLANG_LIBRARY "")
    endif()
    find_library(FCPW_GFX_LIBRARY NAMES gfx PATHS ${SLANG_LIBRARY_PATH})

    # find the corresponding .dll files
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(SLANG_BINARY_PATH ${FCPW_SLANG_OUTPUT_DIR}/bin)
        find_file(FCPW_SLANG_BINARY NAMES slang.dll PATHS ${SLANG_BINARY_PATH})
        find_file(FCPW_SLANG_GLSLANG_BINARY NAMES slang-glslang.dll PATHS ${SLANG_BINARY_PATH})
        find_file(FCPW_GFX_BINARY NAMES gfx.dll PATHS ${SLANG_BINARY_PATH})
    endif()

    message(STATUS "SLANG LIBRARY: " ${FCPW_SLANG_LIBRARY})
    message(STATUS "SLANG GLSLANG LIBRARY: " ${FCPW_SLANG_GLSLANG_LIBRARY})
    message(STATUS "GFX LIBRARY: " ${FCPW_GFX_LIBRARY})
endif()

if(FCPW_BUILD_TESTS)
    set(TBB_BUILD_SHARED          ON  CACHE BOOL " " FORCE)
    set(TBB_BUILD_STATIC          OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TESTS           OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC       OFF CACHE BOOL " " FORCE)
    set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
    add_subdirectory(deps/tbb)
    set_property(TARGET tbb tbb_def_files PROPERTY FOLDER "dependencies")
    if(hasParent)
        set(FCPW_TESTS_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include PARENT_SCOPE)
    else()
        set(FCPW_TESTS_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
    endif()
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
# build Python bindings
if(FCPW_BUILD_BINDINGS)
    add_subdirectory(deps/nanobind)
    add_subdirectory(python)
endif()

################################################################################
# build demo and tests
if(FCPW_BUILD_DEMO OR FCPW_BUILD_TESTS)
    set(FCPW_ARGS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/args)
    add_subdirectory(deps/polyscope)
endif()

if(FCPW_BUILD_DEMO)
    add_subdirectory(demos)
endif()

if(FCPW_BUILD_TESTS)
    add_subdirectory(tests)
endif()
