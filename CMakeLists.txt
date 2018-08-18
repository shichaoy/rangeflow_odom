cmake_minimum_required(VERSION 2.8.3)
project(rangeflow_odom)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  eigen_conversions
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  cv_bridge
  image_geometry
  message_generation
  cmake_modules
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
     DIRECTORY msg
     FILES vo_state.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES 
   sensor_msgs
   std_msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS 
  LIBRARIES 
  CATKIN_DEPENDS pcl_conversions pcl_ros roscpp sensor_msgs std_msgs tf cv_bridge image_geometry eigen_conversions message_runtime
  DEPENDS eigen Boost
)

###########
## Build ##
###########

# set(CMAKE_BUILD_TYPE Release)
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -std=gnu++0x")

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a cpp executable
add_executable (rangeflow_odom src/rangeflow.cpp)
add_executable (refine src/refine.cpp)

add_dependencies(rangeflow_odom  ${rangeflow_odom_EXPORTED_TARGETS})

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(rangeflow_odom_node rangeflow_odom_generate_messages_cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(rangeflow_odom ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(refine ${catkin_LIBRARIES} ${Boost_LIBRARIES})
