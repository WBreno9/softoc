cmake_minimum_required(VERSION 3.12.1)
project(softoc)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.1")
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable (softoc main.c)
