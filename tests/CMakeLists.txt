cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-tests)

################################################################################
# build tests
if(FCPW_BUILD_TESTS)
    set(TBB_TEST          OFF CACHE BOOL " " FORCE)
    set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)

    set(FCPW_TESTS_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
endif()

add_executable(aggregate_tests aggregate_tests.cpp)
target_link_libraries(aggregate_tests fcpw)
target_include_directories(aggregate_tests PRIVATE ${FCPW_TESTS_TBB_INCLUDES})
target_include_directories(aggregate_tests PRIVATE ${FCPW_ARGS_INCLUDES})
target_link_libraries(aggregate_tests polyscope TBB::tbb)

add_executable(csg_tests csg_tests.cpp)
target_link_libraries(csg_tests fcpw)
target_include_directories(csg_tests PRIVATE ${FCPW_TESTS_TBB_INCLUDES})
target_include_directories(csg_tests PRIVATE ${FCPW_ARGS_INCLUDES})
target_link_libraries(csg_tests polyscope TBB::tbb)

if(FCPW_ENABLE_GPU_SUPPORT)
    add_executable(gpu_tests gpu_tests.cpp)
    target_link_libraries(gpu_tests fcpw)
    target_include_directories(gpu_tests PRIVATE ${FCPW_TESTS_TBB_INCLUDES})
    target_include_directories(gpu_tests PRIVATE ${FCPW_ARGS_INCLUDES})
    target_link_libraries(gpu_tests polyscope TBB::tbb)
endif()
