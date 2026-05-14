#!/usr/bin/env bash

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

roslaunch symbols_recognition real_robot.launch "$@"
