cmake_minimum_required(VERSION 3.1.0)
project(fcpw-tests)

################################################################################
# build tests

option(FCPW_TEST_AGG "Build Aggregate tests" ON)
option(FCPW_TEST_BB "Build OBB test" ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(FCPW_TEST_AGG) 
	add_executable(aggregate_tests aggregate_tests.cpp)
	target_link_libraries(aggregate_tests fcpw)

	target_include_directories(aggregate_tests PUBLIC ${FCPW_EIGEN_INCLUDES})
	if(FCPW_USE_ENOKI)
		target_include_directories(aggregate_tests PUBLIC ${FCPW_ENOKI_INCLUDES})
	endif()

	target_link_libraries(aggregate_tests Threads::Threads)

	if(MSVC) 
		target_compile_options(aggregate_tests PRIVATE /bigobj)
		endif()
endif()		

if(FCPW_TEST_BB)
	add_executable(bb_test bb_test.cpp)
	target_link_libraries(bb_test fcpw polyscope)

	target_include_directories(bb_test PUBLIC ${FCPW_EIGEN_INCLUDES})
	if(FCPW_USE_ENOKI)
		target_include_directories(bb_test PUBLIC ${FCPW_ENOKI_INCLUDES})
	endif()

	if(MSVC)
		target_compile_options(bb_test PRIVATE /bigobj /wd4267)
	endif()
endif()
