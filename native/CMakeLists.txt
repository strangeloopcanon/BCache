cmake_minimum_required(VERSION 3.20)
project(bodocache_copy_engine LANGUAGES CXX)

option(USE_CUDA "Build with CUDA backend" OFF)
option(USE_HIP  "Build with HIP backend"  OFF)
option(USE_L0   "Build with Level Zero backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

if (USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(bodocache_copy_engine MODULE copy_engine_native_cuda.cu)
  target_compile_definitions(bodocache_copy_engine PRIVATE USE_CUDA_BACKEND=1)
  target_include_directories(bodocache_copy_engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  set_target_properties(bodocache_copy_engine PROPERTIES CUDA_ARCHITECTURES native)
elseif(USE_HIP)
  find_package(HIP REQUIRED)
  hip_add_library(bodocache_copy_engine MODULE copy_engine_native_hip.cpp)
  target_compile_definitions(bodocache_copy_engine PRIVATE USE_HIP_BACKEND=1)
elseif(USE_L0)
  find_path(LEVEL_ZERO_INCLUDE_DIRS level_zero/ze_api.h)
  find_library(LEVEL_ZERO_LIB ze_loader)
  if (NOT LEVEL_ZERO_INCLUDE_DIRS OR NOT LEVEL_ZERO_LIB)
    message(FATAL_ERROR "Level Zero headers or loader not found")
  endif()
  add_library(bodocache_copy_engine MODULE copy_engine_native_l0.cpp)
  target_compile_definitions(bodocache_copy_engine PRIVATE USE_L0_BACKEND=1)
  target_include_directories(bodocache_copy_engine PRIVATE ${LEVEL_ZERO_INCLUDE_DIRS})
  target_link_libraries(bodocache_copy_engine PRIVATE ${LEVEL_ZERO_LIB})
else()
  message(FATAL_ERROR "Select a backend: -DUSE_CUDA=ON or -DUSE_HIP=ON")
endif()

target_link_libraries(bodocache_copy_engine PRIVATE pybind11::module Python3::Module)

# Python module naming (no 'lib' prefix, .so/.pyd suffix)
set_target_properties(bodocache_copy_engine PROPERTIES
  PREFIX ""
  OUTPUT_NAME "bodocache_agent_copy_engine"
)

# Optional io_uring reader module
option(USE_URING "Build io_uring reader module" OFF)
if (USE_URING)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(LIBURING REQUIRED IMPORTED_TARGET liburing)
  add_library(bodocache_io_uring MODULE io_uring_reader.cpp)
  target_link_libraries(bodocache_io_uring PRIVATE PkgConfig::LIBURING pybind11::module Python3::Module)
  set_target_properties(bodocache_io_uring PROPERTIES PREFIX "" OUTPUT_NAME "bodocache_agent_io_uring")
endif()
