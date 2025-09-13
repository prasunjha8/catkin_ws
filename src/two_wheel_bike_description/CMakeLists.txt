cmake_minimum_required(VERSION 3.0.2)
project(two_wheel_bike_description)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch worlds urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
