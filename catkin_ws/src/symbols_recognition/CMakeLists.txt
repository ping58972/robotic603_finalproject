cmake_minimum_required(VERSION 3.0.2)
project(symbols_recognition)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  gazebo_msgs
  gazebo_ros
  geometry_msgs
  image_transport
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    gazebo_msgs
    gazebo_ros
    geometry_msgs
    image_transport
    nav_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/P2D1_wallFollower.py
  scripts/position_publisher.py
  scripts/real_wall_follower.py
  scripts/symbol_classifier.py
  scripts/target_symbol_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  models
  rviz
  worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

add_subdirectory(plugins)
