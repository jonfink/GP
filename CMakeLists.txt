cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


#common commands for building c++ executables and libraries
rosbuild_add_library(${PROJECT_NAME} 
  src/GP.cc 
  src/KernelFunction.cc
  src/MeanFunction.cc
  src/CGOptimizer.cc)
target_link_libraries(${PROJECT_NAME} 
  armadillo
  gsl)
if(APPLE)
  set_target_properties(${PROJECT_NAME}
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate -framework vecLib")
endif()

rosbuild_add_library(SPGP
  src/SPGP.cc 
  src/GP.cc 
  src/KernelFunction.cc
  src/MeanFunction.cc
  src/CGOptimizer.cc)
target_link_libraries(SPGP
  armadillo
  gsl)
if(APPLE)
  set_target_properties(SPGP
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate -framework vecLib")
endif()

rosbuild_add_executable(test src/test.cc)
target_link_libraries(test armadillo ${PROJECT_NAME})
if(APPLE)
  set_target_properties(test
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate")
endif()

rosbuild_add_executable(test_spgp src/test_spgp.cc)
target_link_libraries(test_spgp armadillo SPGP)
if(APPLE)
  set_target_properties(test_spgp
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate")
endif()