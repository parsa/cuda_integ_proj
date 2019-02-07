cmake_minimum_required(VERSION 3.10)
project(cuda_integ_proj CXX CUDA)

#add_executable(cip_sa add_standalone.cu)

#add_executable(cip_i prog.cpp add_integ.cu)

add_library(sub_k sub_integ.cu)
add_library(add_k add_integ.cu)
add_executable(cip_i_2 prog.cpp)
target_link_libraries(cip_i_2 add_k sub_k)

#if(BUILD_TESTING)
#  add_executable(cip_test_1 test_add_integ.cu add_integ.cu)
#
#  add_test(NAME add_integ COMMAND cip_test_1)
#endif()