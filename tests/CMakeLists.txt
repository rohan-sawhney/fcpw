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
if(FCPW_TESTS_BENCHMARK_EMBREE)
	target_include_directories(aggregate_tests PUBLIC ${FCPW_TESTS_EMBREE_INCLUDES})
	target_link_libraries(aggregate_tests embree)
	target_compile_definitions(aggregate_tests PUBLIC -DFCPW_TESTS_BENCHMARK_EMBREE)
endif()

target_include_directories(aggregate_tests PUBLIC ${FCPW_TESTS_TBB_INCLUDES})
target_include_directories(aggregate_tests PUBLIC ${ARGS_INCLUDES})

target_link_libraries(aggregate_tests polyscope tbb)


add_executable(csg_tests csg_tests.cpp)
target_link_libraries(csg_tests fcpw)
target_include_directories(csg_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
if(FCPW_USE_ENOKI)
	target_include_directories(csg_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
endif()

target_include_directories(csg_tests PUBLIC ${FCPW_TESTS_TBB_INCLUDES})
target_include_directories(csg_tests PUBLIC ${ARGS_INCLUDES})

target_link_libraries(csg_tests polyscope tbb)
