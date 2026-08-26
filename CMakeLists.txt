cmake_minimum_required(VERSION 3.10)
project(fast_ecc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Name the modules explicitly.  ${OpenCV_LIBS} expands to every module the OpenCV
# package was *configured* with, so a build whose config lists (say) gapi without
# having built it fails to link -- even though fast-ecc never touches it.
find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
set(FASTECC_CV opencv_core opencv_imgproc opencv_video)

if(MSVC)
  add_compile_options(/utf-8)   # sources are UTF-8 (avoids C4819 on Japanese locales)
endif()

# --- the library -----------------------------------------------------------
add_library(fast_ecc src/fast_ecc.cpp)
target_include_directories(fast_ecc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(fast_ecc PUBLIC ${FASTECC_CV})
target_include_directories(fast_ecc PUBLIC ${OpenCV_INCLUDE_DIRS})

# The vector paths are written for OpenCV's universal intrinsics at whatever
# width the headers expose, which outside OpenCV's own build is 128 bits
# (SSE2 / NEON).  This option compiles them for AVX2 with FMA instead --
# eight lanes, fused multiply-adds, gathers -- and makes the library require
# an AVX2 CPU (2013 on).  See the top of src/fast_ecc.cpp.
option(FAST_ECC_AVX2 "Compile the vector paths for AVX2 + FMA (needs an AVX2 CPU)" OFF)
if(FAST_ECC_AVX2)
  target_compile_definitions(fast_ecc PRIVATE FASTECC_AVX2=1)
  if(MSVC)
    target_compile_options(fast_ecc PRIVATE /arch:AVX2)
  else()
    target_compile_options(fast_ecc PRIVATE -mavx2 -mfma)
  endif()
endif()

# --- tools (optional) ------------------------------------------------------
option(FAST_ECC_BUILD_BENCH    "Build the benchmark"        ON)
option(FAST_ECC_BUILD_EXAMPLES "Build the examples"         ON)
option(FAST_ECC_BUILD_TESTS    "Build the equivalence test" ON)

if(FAST_ECC_BUILD_BENCH)
  add_executable(bench bench/bench.cpp)
  target_link_libraries(bench PRIVATE fast_ecc ${FASTECC_CV})
endif()

if(FAST_ECC_BUILD_EXAMPLES)
  find_package(OpenCV REQUIRED COMPONENTS imgcodecs)
  add_executable(align_pair examples/align_pair.cpp)
  target_link_libraries(align_pair PRIVATE fast_ecc ${FASTECC_CV} opencv_imgcodecs)
endif()

if(FAST_ECC_BUILD_TESTS)
  enable_testing()
  add_executable(test_equivalence test/test_equivalence.cpp)
  target_link_libraries(test_equivalence PRIVATE fast_ecc ${FASTECC_CV})
  add_test(NAME equivalence COMMAND test_equivalence)
endif()
