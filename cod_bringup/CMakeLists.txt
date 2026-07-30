cmake_minimum_required(VERSION 3.8)
project(cod_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Skip copyright / cpplint checks for now
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

# Install launch/config/map/rviz/waypoint/BT files
install(
  DIRECTORY
    launch
    params
    rviz
    maps
    wps
    behavior_trees
  DESTINATION share/${PROJECT_NAME}
)

# Install Python executable nodes
install(
  PROGRAMS
    scripts/mission_manager.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
