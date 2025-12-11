cmake_minimum_required(VERSION 3.16.0)
phantom_project(phantom_driver CXX)

###########
## Build ##
###########

message("####################")
message("Phantom Driver")
message("####################")

# Git information
# See more at phantom-os-2/src/phantom_ai/core/CmakeLists.txt.

execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_PERCEPTION_BRANCH
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_PERCEPTION_SHA
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git log -n 1 --pretty=%an
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_PERCEPTION_AUTHOR
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

string(
  REPLACE
  "\"" "'"
  GIT_COMMIT_PERCEPTION_AUTHOR ${GIT_COMMIT_PERCEPTION_AUTHOR})

execute_process(
  COMMAND git log -n 1 --pretty=%ad
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_PERCEPTION_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git log -n 1 --pretty=%s
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_PERCEPTION_MSG
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND git log -n 10 --format="%h <%ae> %s"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_PERCEPTION_HISTORY
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

string(
  REGEX REPLACE
  "[^a-zA-Z0-9 /_-]" ""
  GIT_COMMIT_PERCEPTION_BRANCH ${GIT_COMMIT_PERCEPTION_BRANCH})

string(
  REGEX REPLACE
  "[^a-zA-Z0-9 /_-]" ""
  GIT_COMMIT_PERCEPTION_SHA ${GIT_COMMIT_PERCEPTION_SHA})

string(
  REGEX REPLACE
  "[^a-zA-Z0-9 /_-]" ""
  GIT_COMMIT_PERCEPTION_AUTHOR ${GIT_COMMIT_PERCEPTION_AUTHOR})

string(
  REGEX REPLACE
  "[^a-zA-Z0-9 /_-]" ""
  GIT_COMMIT_PERCEPTION_DATE ${GIT_COMMIT_PERCEPTION_DATE})

string(
  REGEX REPLACE
  "[^a-zA-Z0-9 /_-]" ""
  GIT_COMMIT_PERCEPTION_MSG ${GIT_COMMIT_PERCEPTION_MSG})

string(
  REGEX REPLACE
  "[^a-zA-Z0-9 -/_<>@\n]" ""
  GIT_COMMIT_PERCEPTION_HISTORY ${GIT_COMMIT_PERCEPTION_HISTORY})

string(
  REGEX REPLACE
  "\n" "|"
  GIT_COMMIT_PERCEPTION_HISTORY ${GIT_COMMIT_PERCEPTION_HISTORY})

string(
  REGEX REPLACE
  "[\"#]" ""
  GIT_COMMIT_PERCEPTION_HISTORY ${GIT_COMMIT_PERCEPTION_HISTORY})

message("Git Perception Branch: ${GIT_COMMIT_PERCEPTION_BRANCH}")
message("Git Perception SHA: ${GIT_COMMIT_PERCEPTION_SHA}")
message("Commit Author: ${GIT_COMMIT_PERCEPTION_AUTHOR}")
message("Commit Date: ${GIT_COMMIT_PERCEPTION_DATE}")
message("Commit Message: ${GIT_COMMIT_PERCEPTION_MSG}")

# Projects
add_executable(phantom_driver)

# Target Properties #

list(
  APPEND
  phantom_driver_target_properties_private
  # C
  C_EXTENSIONS OFF
  C_STANDARD 17
  C_STANDARD_REQUIRED ON
  C_VISIBILITY_INLINES_HIDDEN ON
  C_VISIBILITY_PRESET hidden
  # C++
  CXX_EXTENSIONS OFF
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_INLINES_HIDDEN ON
  CXX_VISIBILITY_PRESET hidden
  # General
  POSITION_INDEPENDENT_CODE ON)

# Compile Definitions #

list(
  APPEND
  phantom_driver_compile_definitions_private)

# Compile Options #

list(
  APPEND
  phantom_driver_compile_options_private
  # Warnings #
  -Wall
  -Werror
  -Wextra
  -Wpedantic
  # Code Generation #
  -fdata-sections
  -ffunction-sections
  -fstack-clash-protection
  -fstack-protector-strong
      ## Build ##
      $<$<STREQUAL:${CMAKE_BUILD_TYPE},Debug>:-fno-omit-frame-pointer -g3>
      $<$<STREQUAL:${CMAKE_BUILD_TYPE},RelWithDebInfo>:-pg>)

# Include Directories #

list(
  APPEND
  phantom_driver_include_directories_private
  # External
  ${Eigen-3.3.7_INCLUDE_DIR}
  ${embedded_network_INCLUDE_DIR}
  ${HAL_INCLUDE_DIR}
  ${PHANTOM_AI_CAN_INCLUDE_DIR}
  ${PHANTOM_AI_COMMON_INCLUDE_DIR}
  ${PHANTOM_AI_CORE_INCLUDE_DIR}
  ${PHANTOM_AI_DBC_INCLUDE_DIR}
  ${PHANTOM_AI_PHANTOM_VISION2_INCLUDE_DIR}
  ${PHANTOM_AI_TOOLS_INCLUDE_DIR}
  ${PHANTOM_AI_UTILS_INCLUDE_DIR}
  ${PHANTOM_AI_VEHICLE_STATE_PARSER_INCLUDE_DIR}
  ${PHANTOM_AI_WRAPPERS_INCLUDE_DIR}
  $ENV{TI_PSDK_RTOS}/vision_apps/apps/phantom_vision_hal/interface
  # Internal
  ${CMAKE_CURRENT_SOURCE_DIR})

# Sources #

list(
  APPEND
  phantom_driver_sources_private
  # Internal
  src/main.cpp)

# Link Options #

list(
  APPEND
  phantom_driver_link_options_private
  # Warnings #
  LINKER:--fatal-warnings
  LINKER:--warn-shared-textrel
  # Code Generation #
  LINKER:--gc-sections
  LINKER:--no-undefined
  LINKER:-z,defs
  LINKER:-z,noexecstack
  LINKER:-z,now
  LINKER:-z,relro
    ## Build ##
    $<$<STREQUAL:${CMAKE_BUILD_TYPE},Release>:LINKER:--strip-all>)

# Link Libraries #

list(
  APPEND
  phantom_driver_link_libraries_private
  # Internal
  can_wrapper
  camera_wrapper
  hal_process_lib
  network_rx_wrapper
  phantom_vision2
  phantom_vision3
  utils
  vehicle_state_parser)

##################
## Dependencies ##
##################

# OpenCV #
find_package(OpenCV REQUIRED)

list(
  APPEND
  phantom_driver_include_directories_private
  ${OpenCV_INCLUDE_DIRS})

list(
  APPEND
  phantom_driver_link_libraries_private
  ${OpenCV_LIBS})

############
## Target ##
############

# Target Properties #

set_target_properties(
  phantom_driver
  PROPERTIES
  ${phantom_driver_target_properties_private})

# Compile Definitions #

target_compile_definitions(
  phantom_driver
  PRIVATE
  ${phantom_driver_compile_definitions_private})

# Compile Options #

target_compile_options(
  phantom_driver
  PRIVATE
  ${phantom_driver_compile_options_private})

# Include Directories #

target_include_directories(
  phantom_driver
  PRIVATE
  ${phantom_driver_include_directories_private})

# Sources #

target_sources(
  phantom_driver
  PRIVATE
  ${phantom_driver_sources_private})

# Link Options #

target_link_options(
  phantom_driver
  PRIVATE
  ${phantom_driver_link_options_private})

# Link Libraries #

target_link_libraries(
  phantom_driver
  PRIVATE
  ${phantom_driver_link_libraries_private})

# Install #

install(
  TARGETS phantom_driver
  COMPONENT perception
  RUNTIME DESTINATION ${PHANTOM_OS_INSTALL_BIN_DIR}
  LIBRARY DESTINATION ${PHANTOM_OS_INSTALL_LIB_DIR})
