cmake_minimum_required(VERSION 3.0.2)
project(myboat_gazebo)

find_package(catkin REQUIRED COMPONENTS
  gazebo_dev
  roscpp
  rospy
  sensor_msgs
  message_generation
  xacro
  wave_gazebo_plugins
  usv_msgs
  quadrotor_msgs
  tf
  visualization_msgs
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS message_runtime gazebo_dev roscpp sensor_msgs wave_gazebo_plugins usv_msgs quadrotor_msgs tf visualization_msgs
)

# Plugins require c++11
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

include_directories( include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
link_directories(
  ${catkin_LIBRARY_DIRS}
)

# 推进器
add_library(myboat_thrust_plugin
  src/myboat_thrust_plugin.cpp
)
target_link_libraries(myboat_thrust_plugin
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
)
install(TARGETS myboat_thrust_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 动力学
add_library(myboat_dynamics_plugin
  src/myboat_dynamics_plugin.cpp
)
target_link_libraries(myboat_dynamics_plugin
  ${catkin_LIBRARIES}
  ${Eigen_LIBRARIES}
)
install(TARGETS myboat_dynamics_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Python nodes / scripts
catkin_install_python(PROGRAMS
  src/odom_to_pose.py
  src/myboat_odom_tf.py
  src/myboat_odom_relay.py
  src/vla_mission_node.py
  src/collision_target_cruise_node.py
  src/collision_ego_cruise_node.py
  src/collision_perception_node.py
  src/collision_state_estimator_node.py
  src/collision_risk_assessor_node.py
  src/colregs_llm_decision_node.py
  src/collision_behavior_fsm_node.py
  src/collision_cmd_mux_node.py
  src/collision_trajectory_planner_node.py
  src/pairwise_distance_debug_node.py
  src/trajectory_recorder_node.py
  src/rviz_trigger_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# test cpp
add_executable(test_cpp_myboat_control src/test_cpp_myboat_control.cpp)
target_link_libraries(test_cpp_myboat_control
  ${catkin_LIBRARIES}
)

# exploration to myboat adapter
add_executable(exploration_to_myboat src/exploration_to_myboat.cpp)
target_link_libraries(exploration_to_myboat
  ${catkin_LIBRARIES}
)

# lidar 点云水面噪声仿真节点
add_executable(lidar_water_noise_node src/lidar_water_noise_node.cpp)
target_link_libraries(lidar_water_noise_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# debug: publish test PositionCommand to /planning/pos_cmd
add_executable(test_pos_cmd_publisher src/test_pos_cmd_publisher.cpp)
target_link_libraries(test_pos_cmd_publisher
  ${catkin_LIBRARIES}
)

# # 设置依赖（可选但推荐）
# add_dependencies(myboat_control_node
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS}
# )
