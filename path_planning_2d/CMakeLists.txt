cmake_minimum_required(VERSION 2.8.3)
project(path_planning_2d)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  nav_msgs
  visualization_msgs
  dummy_simulator
  jps3d
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS timer)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(jps3d REQUIRED)

#set(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS ON)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  message("${PROJECT_NAME} built in ${CMAKE_BUILD_TYPE} mode.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -Wall -std=c++11")
  set(${CUDA_NVCC_FLAGS} "${CUDA_NVCC_FLAGS};-O0")
else()
  message("${PROJECT_NAME} built in ${CMAKE_BUILD_TYPE} mode.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
  set(${CUDA_NVCC_FLAGS} "${CUDA_NVCC_FLAGS};-O3")
endif()

list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
#list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
#list(APPEND CUDA_NVCC_FLAGS -lcublas -lcublas_device -lcudadevrt)
#list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC)
# Too many code unreachable warnings in the compiler output.
list(APPEND CUDA_NVCC_FLAGS -Xcudafe "--diag_suppress=code_is_unreachable")

################################################
## Declare ROS messages, services and actions ##
################################################


################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES path_planning_2d
  CATKIN_DEPENDS
    roscpp std_msgs std_srvs nav_msgs visualization_msgs dummy_simulator
  DEPENDS
    Boost OpenCV EIGEN3 JPS3D
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${JPS3D_INCLUDE_DIRS}
  # This should be automatically be included with cuda_add_<target>()
  ${CUDA_INCLUDE_DIRS}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${catkin_LIBRARY_DIRS}
)


## MDP node
cuda_add_executable(mdp_path_planning_2d_node
  src/mdp/path_planning_2d.cu
  src/mdp/path_planning_2d_cuda.cu
  src/mdp/path_planning_2d_node.cpp
)
target_link_libraries(mdp_path_planning_2d_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
add_dependencies(mdp_path_planning_2d_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## A* node
add_executable(astar_path_planning_2d_node
  src/astar/path_planning_2d.cpp
  src/astar/path_planning_2d_node.cpp
)
target_link_libraries(astar_path_planning_2d_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${JPS3D_LIBRARIES}
)
add_dependencies(astar_path_planning_2d_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

## POMDP node
cuda_add_executable(pomdp_path_planning_2d_node
  src/pomdp/path_planning_2d.cu
  src/pomdp/model_generation_cuda.cu
  src/pomdp/fast_informed_bound_cuda.cu
  src/pomdp/point_based_value_iteration_cuda.cu
  src/pomdp/search_tree_cuda.cu
  src/pomdp/path_planning_2d_node.cpp
)
#CUDA_ADD_CUBLAS_TO_TARGET(pomdp_path_planning_2d_node)
target_link_libraries(pomdp_path_planning_2d_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
)
add_dependencies(pomdp_path_planning_2d_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_path_planning_2d.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
