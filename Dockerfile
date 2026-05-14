FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release

# Add the ROS repository and update apt sources
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update

# Install all the things
RUN apt-get install -y \
  ros-noetic-desktop-full \
  python3-rosdep \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool \
  build-essential \
  python3-dev \
  python3-catkin-tools \
  ros-noetic-rosserial-arduino \
  ros-noetic-rosserial \
  ros-noetic-librealsense2 \
  ros-noetic-realsense2-camera \
  ros-noetic-realsense2-description \
  ros-noetic-usb-cam \
  ros-noetic-rplidar-ros \
  ros-noetic-camera-info-manager \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-image-transport-plugins \
  ros-noetic-vision-opencv \
  tmux \
  python3-pip \
  python3-numpy \
  python3-opencv \
  python3-pil \
  ros-noetic-teleop-twist-keyboard

RUN python3 -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  typing-extensions==4.12.2 \
  torch==2.4.1 \
  torchvision==0.19.1

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

RUN mkdir /catkin_ws
WORKDIR /catkin_ws

SHELL ["/bin/bash", "-c"] 
COPY ./catkin_ws/src /catkin_ws/src
COPY ./setup.bash /setup.bash
RUN chmod +x /setup.bash

RUN source /opt/ros/noetic/setup.bash && \
  cd /catkin_ws && \
  catkin build
