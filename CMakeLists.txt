cmake_minimum_required(VERSION 2.8.3)
project(gp)
set(CMAKE_BUILD_TYPE Debug)

find_package(ARMADILLO REQUIRED)

include_directories(
 ${ARMADILLO_INCLUDE_DIRS})

#common commands for building c++ executables and libraries
add_library(${PROJECT_NAME}
  src/GP.cc
  src/KernelFunction.cc
  src/MeanFunction.cc
  src/CGOptimizer.cc)
target_link_libraries(${PROJECT_NAME}
  ${ARMADILLO_LIBRARIES}
  gsl)
if(APPLE)
  set_target_properties(${PROJECT_NAME}
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate -framework vecLib")
endif()

add_library(SPGP
  src/SPGP.cc
  src/GP.cc
  src/KernelFunction.cc
  src/MeanFunction.cc
  src/CGOptimizer.cc)
target_link_libraries(SPGP
  ${ARMADILLO_LIBRARIES}
  gsl)
if(APPLE)
  set_target_properties(SPGP
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate -framework vecLib")
endif()

add_executable(test_gp src/test.cc)
target_link_libraries(test_gp ${ARMADILLO_LIBRARIES} ${PROJECT_NAME})
if(APPLE)
  set_target_properties(test_gp
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate")
endif()

add_executable(test_spgp src/test_spgp.cc)
target_link_libraries(test_spgp ${ARMADILLO_LIBRARIES} SPGP)
if(APPLE)
  set_target_properties(test_spgp
    PROPERTIES LINK_FLAGS "${LINK_FLAGS} -framework Accelerate")
endif()
