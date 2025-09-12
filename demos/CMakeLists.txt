cmake_minimum_required(VERSION 3.15...3.27)
project(fcpw-demo)

add_executable(demo demo.cpp)
target_link_libraries(demo fcpw)
target_link_libraries(demo polyscope)
target_include_directories(demo PRIVATE ${FCPW_ARGS_INCLUDES})
