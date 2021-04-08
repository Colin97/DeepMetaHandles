cmake_minimum_required(VERSION 3.1)
project(minghua)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

find_package(LIBIGL REQUIRED QUIET)

# Add your project files
file(GLOB SRCFILES *.cpp)
add_executable(sample_key_points_bin sample_key_points.cpp)
add_executable(calc_weight_bin calc_weight.cpp)
add_executable(sample_surface_points_bin sample_surface_points.cpp)
add_executable(normalize_bin normalize.cpp)
target_link_libraries(sample_key_points_bin igl::core)
target_link_libraries(calc_weight_bin igl::core igl::opengl igl::opengl_glfw)
target_link_libraries(sample_surface_points_bin igl::core)
target_link_libraries(normalize_bin igl::core)