cmake_minimum_required(VERSION 3.1.0)
project(fcpw-tests)

################################################################################
# build tests
add_executable(aggregate_tests aggregate_tests.cpp)
target_link_libraries(aggregate_tests fcpw)
target_include_directories(aggregate_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
if(FCPW_USE_ENOKI)
    target_include_directories(aggregate_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
endif()

target_include_directories(aggregate_tests PUBLIC ${FCPW_TESTS_TBB_INCLUDES})
target_include_directories(aggregate_tests PUBLIC ${FCPW_ARGS_INCLUDES})

target_link_libraries(aggregate_tests polyscope tbb)


add_executable(csg_tests csg_tests.cpp)
target_link_libraries(csg_tests fcpw)
target_include_directories(csg_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
if(FCPW_USE_ENOKI)
    target_include_directories(csg_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
endif()

target_include_directories(csg_tests PUBLIC ${FCPW_TESTS_TBB_INCLUDES})
target_include_directories(csg_tests PUBLIC ${FCPW_ARGS_INCLUDES})

target_link_libraries(csg_tests polyscope tbb)
