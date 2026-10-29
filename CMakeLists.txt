cmake_minimum_required(VERSION 3.25)
project(FORGE VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FORGE_BUILD_TESTS "Build FORGE tests" ON)
option(FORGE_BUILD_APPS  "Build FORGE apps"  ON)
option(FORGE_SANITIZE    "Build with ASan/UBSan" OFF)

if(FORGE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---- spec codegen: .fspec -> op table header + docs + oracle test list ----
file(GLOB FORGE_FSPECS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/spec/ops/*.fspec)
set(FORGE_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
add_custom_command(
  OUTPUT ${FORGE_GEN_DIR}/forge/spec/ops.gen.hpp ${FORGE_GEN_DIR}/ops.md
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/spec/gen/fspec_gen.py
          --out ${FORGE_GEN_DIR} ${FORGE_FSPECS}
  DEPENDS ${FORGE_FSPECS} ${CMAKE_SOURCE_DIR}/spec/gen/fspec_gen.py
  COMMENT "Generating op registry from spec/ops/*.fspec")
add_custom_target(forge_spec_gen DEPENDS ${FORGE_GEN_DIR}/forge/spec/ops.gen.hpp)

# ---- Tier R: raw intrinsic wrappers generated from spec/intrinsics ----
add_custom_command(
  OUTPUT ${FORGE_GEN_DIR}/forge/lane/raw/x86.gen.hpp
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/spec/gen/intrin_gen.py
          --db ${CMAKE_SOURCE_DIR}/spec/intrinsics/x86.json
          --out ${FORGE_GEN_DIR}/forge/lane/raw/x86.gen.hpp
  DEPENDS ${CMAKE_SOURCE_DIR}/spec/intrinsics/x86.json ${CMAKE_SOURCE_DIR}/spec/gen/intrin_gen.py
  COMMENT "Generating lane Tier R (x86)")
add_custom_target(forge_raw_gen DEPENDS ${FORGE_GEN_DIR}/forge/lane/raw/x86.gen.hpp)

add_library(forge_core INTERFACE)
target_include_directories(forge_core INTERFACE core/include)
add_library(forge::core ALIAS forge_core)

add_library(forge_spec INTERFACE)
target_include_directories(forge_spec INTERFACE spec/include ${FORGE_GEN_DIR})
target_link_libraries(forge_spec INTERFACE forge_core)
add_dependencies(forge_spec forge_spec_gen)

add_library(forge_lane INTERFACE)
target_include_directories(forge_lane INTERFACE lane/include ${FORGE_GEN_DIR})
target_link_libraries(forge_lane INTERFACE forge_core)
add_dependencies(forge_lane forge_raw_gen)

add_library(forge_verify INTERFACE)
target_include_directories(forge_verify INTERFACE verify/include)
target_link_libraries(forge_verify INTERFACE forge_spec)

add_library(forge_grid INTERFACE)
target_include_directories(forge_grid INTERFACE grid/include support/include)
target_link_libraries(forge_grid INTERFACE forge_lane)

add_library(forge_flow INTERFACE)
target_include_directories(forge_flow INTERFACE flow/include)
target_link_libraries(forge_flow INTERFACE forge_spec forge_lane forge_grid)

add_library(forge_pack INTERFACE)
target_include_directories(forge_pack INTERFACE pack/include)
target_link_libraries(forge_pack INTERFACE forge_flow)

add_library(forge_mind INTERFACE)
target_include_directories(forge_mind INTERFACE mind/include)
target_link_libraries(forge_mind INTERFACE forge_pack)
target_include_directories(forge_mind INTERFACE mind/include)
add_library(forge_c SHARED mind/capi/forge_c.cpp)
target_link_libraries(forge_c PRIVATE forge_mind forge_verify forge_lane_mv)
set_target_properties(forge_c PROPERTIES OUTPUT_NAME forge VERSION 0.1.0 SOVERSION 1 CXX_VISIBILITY_PRESET default)
target_compile_definitions(forge_mind INTERFACE FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Multiversioned lane kernels: one TU per target, own flags, own namespace.
add_library(forge_lane_mv STATIC
  lane/src/kernels_scalar.cpp
  lane/src/kernels_avx2.cpp
  lane/src/kernels_avx512.cpp
  lane/src/dispatch.cpp)
target_link_libraries(forge_lane_mv PUBLIC forge_lane)
set_source_files_properties(lane/src/kernels_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
set_source_files_properties(lane/src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")

if(FORGE_BUILD_APPS)
  add_executable(forge apps/forge.cpp)
  target_link_libraries(forge PRIVATE forge_mind forge_verify forge_lane_mv)
endif()

if(FORGE_BUILD_TESTS)
  enable_testing()
  file(GLOB FORGE_TESTS CONFIGURE_DEPENDS tests/test_*.cpp)
  foreach(t ${FORGE_TESTS})
    get_filename_component(n ${t} NAME_WE)
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE forge_mind forge_verify forge_lane_mv)
    target_include_directories(${n} PRIVATE tests)
    if(n STREQUAL "test_lane_raw")
      target_compile_options(${n} PRIVATE -mavx2 -mfma -mf16c)
    endif()
    add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
  add_executable(capi_smoke tests/capi_smoke.c)
  target_include_directories(capi_smoke PRIVATE mind/include)
  target_link_libraries(capi_smoke PRIVATE forge_c)
  if(FORGE_BUILD_APPS)
    add_test(NAME golden_tiny_llama
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/verify/golden/tiny_llama.py
              --forge $<TARGET_FILE:forge> --work ${CMAKE_BINARY_DIR}/golden)
    add_test(NAME capi_smoke COMMAND capi_smoke ${CMAKE_BINARY_DIR}/golden/tiny_f32.fpk)
    set_tests_properties(capi_smoke PROPERTIES DEPENDS golden_tiny_llama)
  endif()
endif()
