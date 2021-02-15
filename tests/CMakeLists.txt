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
