cmake_minimum_required(VERSION 3.0.2)
project(bike_rl_agent)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
)

## Declare a catkin package
catkin_package()

## Install Python scripts
install(PROGRAMS
  scripts/train.py
  scripts/bike_env.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)