cmake_minimum_required(VERSION 3.28)
project(QuantAlpha CXX)

# ---------------------------------------------------------------------------
# C++26 Standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Compiler flags (Google Style: warnings as errors in CI)
# ---------------------------------------------------------------------------
add_compile_options(
  -Wall -Wextra -Wpedantic -Werror
  -O3 -march=native
  -fno-omit-frame-pointer
)

# ---------------------------------------------------------------------------
# Dependencies via CMake FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# Eigen 3.4
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Eigen)

# GoogleTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Interface library (header-only quant modules)
# ---------------------------------------------------------------------------
add_library(quant_alpha INTERFACE)
target_include_directories(quant_alpha INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)
target_link_libraries(quant_alpha INTERFACE Eigen3::Eigen)

# ---------------------------------------------------------------------------
# Test executable
# ---------------------------------------------------------------------------
enable_testing()

add_executable(quant_tests
  src/cpp/tests/test_quant.cpp
)

target_link_libraries(quant_tests
  PRIVATE
    quant_alpha
    GTest::gtest_main
    Eigen3::Eigen
)

include(GoogleTest)
gtest_discover_tests(quant_tests)

# ---------------------------------------------------------------------------
# Coverage (enabled via -DENABLE_COVERAGE=ON)
# ---------------------------------------------------------------------------
option(ENABLE_COVERAGE "Enable GCC/Clang coverage" OFF)
if(ENABLE_COVERAGE)
  target_compile_options(quant_tests PRIVATE --coverage -O0)
  target_link_options(quant_tests PRIVATE --coverage)
endif()
