cmake_minimum_required(VERSION 3.10)
project(fast_ecc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(OpenCV REQUIRED)

if(MSVC)
  add_compile_options(/utf-8)   # sources are UTF-8 (avoids C4819 on Japanese locales)
endif()

# --- the library -----------------------------------------------------------
add_library(fast_ecc src/fast_ecc.cpp)
target_include_directories(fast_ecc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(fast_ecc PUBLIC ${OpenCV_LIBS})
target_include_directories(fast_ecc PUBLIC ${OpenCV_INCLUDE_DIRS})

# --- tools (optional) ------------------------------------------------------
option(FAST_ECC_BUILD_BENCH    "Build the benchmark"        ON)
option(FAST_ECC_BUILD_EXAMPLES "Build the examples"         ON)
option(FAST_ECC_BUILD_TESTS    "Build the equivalence test" ON)

if(FAST_ECC_BUILD_BENCH)
  add_executable(bench bench/bench.cpp)
  target_link_libraries(bench PRIVATE fast_ecc ${OpenCV_LIBS})
endif()

if(FAST_ECC_BUILD_EXAMPLES)
  add_executable(align_pair examples/align_pair.cpp)
  target_link_libraries(align_pair PRIVATE fast_ecc ${OpenCV_LIBS})
endif()

if(FAST_ECC_BUILD_TESTS)
  enable_testing()
  add_executable(test_equivalence test/test_equivalence.cpp)
  target_link_libraries(test_equivalence PRIVATE fast_ecc ${OpenCV_LIBS})
  add_test(NAME equivalence COMMAND test_equivalence)
endif()
