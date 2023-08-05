cmake_minimum_required(VERSION 3.0.2)
project(mild_hri_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  rospy
  sensor_msgs
  std_msgs
  tf2
  tf2_msgs
  tf2_ros
  visualization_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES mild_hri_ros
#  CATKIN_DEPENDS image_transport rospy sensor_msgs std_msgs tf2 tf2_msgs tf2_ros visualization_msgs
#  DEPENDS system_lib
)

