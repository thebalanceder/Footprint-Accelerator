cmake_minimum_required(VERSION 3.5)
project(soft2_vo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# ✅ Correct Ceres detection
find_package(Ceres REQUIRED PATHS /usr/lib/cmake/Ceres NO_DEFAULT_PATH)

add_executable(soft2_node src/soft2_node.cpp)

target_include_directories(soft2_node PUBLIC
  ${rclcpp_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${nav_msgs_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(soft2_node
  ${rclcpp_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${nav_msgs_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${image_transport_LIBRARIES}
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
  ${CERES_LIBRARIES}
)

ament_target_dependencies(soft2_node
  rclcpp cv_bridge sensor_msgs nav_msgs tf2 tf2_ros image_transport OpenCV Eigen3
)

install(TARGETS soft2_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
ament_package()

