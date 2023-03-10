cmake_minimum_required(VERSION 3.0)
project(miniWeatherML)

enable_language(Fortran)
enable_language(CXX)
enable_language(C)
if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

set(YAKL_HOME ${CMAKE_CURRENT_SOURCE_DIR}/YAKL)
set(YAKL_BIN  ${CMAKE_CURRENT_BINARY_DIR}/yakl)
add_subdirectory(${YAKL_HOME} ${YAKL_BIN})

add_executable(driver driver.cpp)
include(${YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(driver)
target_link_libraries(driver "${MW_LINK_FLAGS}")


add_executable(driver_without_yakl driver_without_yakl.cpp)
set_source_files_properties(driver_without_yakl.cpp PROPERTIES COMPILE_FLAGS "${YAKL_COMPILER_FLAGS}")
target_link_libraries(driver_without_yakl "${MW_LINK_FLAGS}")

