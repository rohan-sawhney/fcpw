cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-tests)

################################################################################
# build tests
list(APPEND FCPW_TESTS_INCLUDES ${FCPW_ARGS_INCLUDES})

# watertight_tests doesn't need TBB or polyscope
add_executable(watertight_tests watertight_tests.cpp)
target_link_libraries(watertight_tests fcpw)
target_include_directories(watertight_tests PRIVATE ${FCPW_TESTS_INCLUDES})

# GPU watertight tests - needs GPU support but not TBB or polyscope
# NOTE: GPU compute shaders currently do not work on Metal (macOS).
# To enable this test on platforms with working GPU support (Vulkan/CUDA),
# set FCPW_BUILD_GPU_TESTS=ON in addition to FCPW_ENABLE_GPU_SUPPORT=ON.
if(FCPW_ENABLE_GPU_SUPPORT AND FCPW_BUILD_GPU_TESTS)
    add_executable(gpu_watertight_tests gpu_watertight_tests.cpp)
    target_link_libraries(gpu_watertight_tests fcpw)
    target_include_directories(gpu_watertight_tests PRIVATE ${FCPW_TESTS_INCLUDES})
endif()

# Other tests require TBB - only build if submodule is available
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/CMakeLists.txt)
    set(TBB_TEST          OFF CACHE BOOL " " FORCE)
    set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)

    list(APPEND FCPW_TESTS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)

    add_executable(aggregate_tests aggregate_tests.cpp)
    target_link_libraries(aggregate_tests fcpw polyscope TBB::tbb)
    target_include_directories(aggregate_tests PRIVATE ${FCPW_TESTS_INCLUDES})

    add_executable(csg_tests csg_tests.cpp)
    target_link_libraries(csg_tests fcpw polyscope TBB::tbb)
    target_include_directories(csg_tests PRIVATE ${FCPW_TESTS_INCLUDES})

    if(FCPW_ENABLE_GPU_SUPPORT)
        add_executable(gpu_tests gpu_tests.cpp)
        target_link_libraries(gpu_tests fcpw polyscope TBB::tbb)
        target_include_directories(gpu_tests PRIVATE ${FCPW_TESTS_INCLUDES})
    endif()
else()
    message(STATUS "TBB submodule not found - skipping aggregate_tests, csg_tests, gpu_tests")
endif()
