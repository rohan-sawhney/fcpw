cmake_minimum_required(VERSION 3.1.0)
project(fcpw-tests)

################################################################################
# build tests

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(cpq_tests cpq_tests.cpp)
target_link_libraries(cpq_tests fcpw Threads::Threads)

target_include_directories(cpq_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
if(FCPW_USE_ENOKI)
	target_include_directories(cpq_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
endif()


add_executable(int_tests int_tests.cpp)
target_link_libraries(int_tests fcpw Threads::Threads)

target_include_directories(int_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
if(FCPW_USE_ENOKI)
	target_include_directories(int_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
endif()

