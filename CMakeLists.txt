cmake_minimum_required(VERSION 3.25)
project(FORGE VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FORGE_BUILD_TESTS "Build FORGE tests" ON)
option(FORGE_BUILD_APPS  "Build FORGE apps"  ON)
option(FORGE_SANITIZE    "Build with ASan/UBSan" OFF)
option(FORGE_LIBFUZZER   "Link fuzz targets with libFuzzer (clang -fsanitize=fuzzer) instead of the built-in driver" OFF)
set(FORGE_FUZZ_ITERS 4000 CACHE STRING "Mutated inputs per fuzz target in ctest")

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


# ---- GPU kernels generated from spec/ops (FORGE_PLAN B.4): GLSL -> SPIR-V (embedded), CUDA/HIP compile checks ----
set(FORGE_GPU_DIR ${FORGE_GEN_DIR}/gpu)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/spec/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR} --list
                OUTPUT_VARIABLE FORGE_GPU_KERNELS OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _gpu_rc)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/spec/gen/gpu_gen.py)
find_program(FORGE_GLSLANG glslangValidator)
find_program(FORGE_SPIRV_VAL spirv-val)
find_path(FORGE_VULKAN_INCLUDE vulkan/vulkan.h)
set(FORGE_GPU_SOURCES)
foreach(k ${FORGE_GPU_KERNELS})
  list(APPEND FORGE_GPU_SOURCES ${FORGE_GPU_DIR}/${k}.comp)
endforeach()
add_custom_command(OUTPUT ${FORGE_GPU_SOURCES} ${FORGE_GPU_DIR}/kernels.cu
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/spec/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR}
  DEPENDS ${FORGE_FSPECS} ${CMAKE_SOURCE_DIR}/spec/gen/gpu_gen.py COMMENT "Generating GPU kernels from spec/ops")
if(_gpu_rc EQUAL 0 AND FORGE_GLSLANG AND FORGE_VULKAN_INCLUDE)
  set(FORGE_SPV)
  foreach(k ${FORGE_GPU_KERNELS})
    set(_val "")
    if(FORGE_SPIRV_VAL)
      set(_val COMMAND ${FORGE_SPIRV_VAL} --target-env vulkan1.1 ${FORGE_GPU_DIR}/${k}.spv)
    endif()
    add_custom_command(OUTPUT ${FORGE_GPU_DIR}/${k}.spv
      COMMAND ${FORGE_GLSLANG} -V --target-env vulkan1.1 -o ${FORGE_GPU_DIR}/${k}.spv ${FORGE_GPU_DIR}/${k}.comp ${_val}
      DEPENDS ${FORGE_GPU_DIR}/${k}.comp COMMENT "SPIR-V ${k}")
    list(APPEND FORGE_SPV ${FORGE_GPU_DIR}/${k}.spv)
  endforeach()
  add_custom_command(OUTPUT ${FORGE_GEN_DIR}/forge/grid/spirv.gen.hpp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/spec/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR}
            --embed ${FORGE_GEN_DIR}/forge/grid/spirv.gen.hpp
    DEPENDS ${FORGE_SPV} ${CMAKE_SOURCE_DIR}/spec/gen/gpu_gen.py COMMENT "Embedding SPIR-V kernels")
  add_custom_target(forge_gpu_gen ALL DEPENDS ${FORGE_GEN_DIR}/forge/grid/spirv.gen.hpp)
  # Copy only the Vulkan headers so the system include dir is not put ahead of the C++ standard library.
  file(COPY ${FORGE_VULKAN_INCLUDE}/vulkan DESTINATION ${FORGE_GEN_DIR}/vkinc)
  if(EXISTS ${FORGE_VULKAN_INCLUDE}/vk_video)
    file(COPY ${FORGE_VULKAN_INCLUDE}/vk_video DESTINATION ${FORGE_GEN_DIR}/vkinc)
  endif()
  set(FORGE_HAVE_VULKAN ON)
  message(STATUS "FORGE: Vulkan backend enabled (${FORGE_GLSLANG})")
else()
  add_custom_target(forge_gpu_gen ALL DEPENDS ${FORGE_GPU_DIR}/kernels.cu)
  message(STATUS "FORGE: Vulkan backend disabled (need glslangValidator + vulkan/vulkan.h)")
endif()
add_library(forge_grid INTERFACE)
target_include_directories(forge_grid INTERFACE grid/include support/include)
target_include_directories(forge_grid INTERFACE ${FORGE_GEN_DIR})
if(FORGE_HAVE_VULKAN)
  target_include_directories(forge_grid INTERFACE ${FORGE_GEN_DIR}/vkinc)
endif()
add_dependencies(forge_grid forge_gpu_gen)
target_link_libraries(forge_grid INTERFACE forge_lane)

add_library(forge_flow INTERFACE)
target_include_directories(forge_flow INTERFACE flow/include)
target_link_libraries(forge_flow INTERFACE forge_spec forge_lane forge_grid)

# Vendored Monocypher 4.0.2 (BSD-2-Clause / CC0): Ed25519 for .fpk signing.
add_library(forge_trust STATIC third_party/monocypher/monocypher.c third_party/monocypher/monocypher-ed25519.c)
target_include_directories(forge_trust PUBLIC third_party/monocypher)
set_target_properties(forge_trust PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)

add_library(forge_pack INTERFACE)
target_include_directories(forge_pack INTERFACE pack/include)
target_link_libraries(forge_pack INTERFACE forge_flow forge_trust)

add_library(forge_mind INTERFACE)
target_include_directories(forge_mind INTERFACE mind/include)
target_link_libraries(forge_mind INTERFACE forge_pack)
target_include_directories(forge_mind INTERFACE mind/include)
add_library(forge_c SHARED mind/capi/forge_c.cpp)
target_link_libraries(forge_c PRIVATE forge_mind forge_verify forge_lane_mv)
set_target_properties(forge_c PROPERTIES OUTPUT_NAME forge VERSION 0.1.0 SOVERSION 1 CXX_VISIBILITY_PRESET default)
target_compile_definitions(forge_mind INTERFACE FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Multiversioned lane kernels: one TU per target, own flags, own namespace.
include(CheckCXXSourceCompiles)
# First flag set that compiles a fixed-length (256-bit) scalable-vector probe: GCC/Clang spelling, then zig's.
function(forge_probe_flags out header type attr)
  foreach(cand IN LISTS ARGN)
    string(REPLACE "|" ";" _fl "${cand}")
    set(CMAKE_REQUIRED_FLAGS "${cand}")
    string(REPLACE "|" " " CMAKE_REQUIRED_FLAGS "${cand}")
    unset(_ok CACHE)
    check_cxx_source_compiles("#include <${header}>\ntypedef ${type} t __attribute__((${attr}));\nstatic_assert(sizeof(t)*8==${FORGE_PROBE_BITS});\nint main(){return 0;}" _ok)
    if(_ok)
      set(${out} "${_fl}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out} "" PARENT_SCOPE)
endfunction()

# Scalable vectors: one fixed-length kernel table per VL (128..1024 bits); dispatch picks the hardware's.
set(FORGE_SCALABLE_BITS 128 256 512 1024)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(FORGE_ARCH aarch64)
  add_library(forge_lane_mv STATIC lane/src/kernels_scalar.cpp lane/src/kernels_neon.cpp lane/src/dispatch.cpp)
  set(_defs)
  foreach(b ${FORGE_SCALABLE_BITS})
    set(FORGE_PROBE_BITS ${b})
    forge_probe_flags(FORGE_SVE${b}_FLAGS arm_sve.h svfloat32_t "arm_sve_vector_bits(${b})"
      "-march=armv8.2-a+sve|-msve-vector-bits=${b}" "-mcpu=generic+sve|-msve-vector-bits=${b}")
    if(FORGE_SVE${b}_FLAGS)
      add_library(forge_lane_sve${b} OBJECT lane/src/kernels_sve.cpp)
      target_compile_options(forge_lane_sve${b} PRIVATE ${FORGE_SVE${b}_FLAGS})
      target_link_libraries(forge_lane_sve${b} PRIVATE forge_lane)
      target_sources(forge_lane_mv PRIVATE $<TARGET_OBJECTS:forge_lane_sve${b}>)
      list(APPEND _defs FORGE_LANE_SVE${b}=1)
    endif()
  endforeach()
  set_source_files_properties(lane/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${_defs}")
  message(STATUS "FORGE: SVE kernel tables: ${_defs}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
  set(FORGE_ARCH riscv64)
  add_library(forge_lane_mv STATIC lane/src/kernels_scalar.cpp lane/src/dispatch.cpp)
  set(_defs)
  foreach(b ${FORGE_SCALABLE_BITS})
    set(FORGE_PROBE_BITS ${b})
    forge_probe_flags(FORGE_RVV${b}_FLAGS riscv_vector.h vfloat32m1_t "riscv_rvv_vector_bits(${b})"
      "-march=rv64gcv_zvl${b}b|-mrvv-vector-bits=${b}" "-mcpu=baseline_rv64+v+zvl${b}b|-mrvv-vector-bits=${b}")
    if(FORGE_RVV${b}_FLAGS)
      add_library(forge_lane_rvv${b} OBJECT lane/src/kernels_rvv.cpp)
      target_compile_options(forge_lane_rvv${b} PRIVATE ${FORGE_RVV${b}_FLAGS})
      target_link_libraries(forge_lane_rvv${b} PRIVATE forge_lane)
      target_sources(forge_lane_mv PRIVATE $<TARGET_OBJECTS:forge_lane_rvv${b}>)
      list(APPEND _defs FORGE_LANE_RVV${b}=1)
    endif()
  endforeach()
  set_source_files_properties(lane/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${_defs}")
  message(STATUS "FORGE: RVV kernel tables: ${_defs}")
else()
  set(FORGE_ARCH x86_64)
  add_library(forge_lane_mv STATIC
    lane/src/kernels_scalar.cpp
    lane/src/kernels_avx2.cpp
    lane/src/kernels_avx512.cpp
    lane/src/dispatch.cpp)
  set_source_files_properties(lane/src/kernels_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(lane/src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")
endif()
target_link_libraries(forge_lane_mv PUBLIC forge_lane)

if(FORGE_BUILD_APPS)
  add_executable(forge apps/forge.cpp)
  target_link_libraries(forge PRIVATE forge_mind forge_verify forge_lane_mv)
  add_executable(forge-serve apps/forge_serve.cpp)
  target_link_libraries(forge-serve PRIVATE forge_mind forge_verify forge_lane_mv)
endif()

if(FORGE_BUILD_TESTS)
  enable_testing()
  file(GLOB FORGE_TESTS CONFIGURE_DEPENDS tests/test_*.cpp)
  foreach(t ${FORGE_TESTS})
    get_filename_component(n ${t} NAME_WE)
    if(n STREQUAL "test_lane_raw" AND NOT FORGE_ARCH STREQUAL "x86_64")
      continue()
    endif()
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE forge_mind forge_verify forge_lane_mv)
    target_include_directories(${n} PRIVATE tests)
    if(n STREQUAL "test_lane_raw")
      target_compile_options(${n} PRIVATE -mavx2 -mfma -mf16c)
    endif()
    add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endforeach()
  set_tests_properties(test_vulkan PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  if(FORGE_HAVE_VULKAN)
    target_include_directories(test_vulkan PRIVATE ${FORGE_GEN_DIR}/vkinc)
  endif()
  # CUDA / HIP: compile the generated kernels to PTX / AMDGCN with clang (no SDK needed); hardware runs need a GPU.
  find_program(FORGE_GPU_CLANG NAMES clang-19 clang-18 clang-20 clang)
  if(FORGE_GPU_CLANG)
    add_test(NAME gpu_compile_cuda COMMAND ${FORGE_GPU_CLANG} -x cuda --cuda-device-only -nocudainc -nocudalib
             --cuda-gpu-arch=sm_80 -Werror -S -o ${FORGE_GPU_DIR}/kernels.ptx ${FORGE_GPU_DIR}/kernels.cu)
    add_test(NAME gpu_compile_hip COMMAND ${FORGE_GPU_CLANG} -x hip --cuda-device-only -nogpuinc -nogpulib
             --offload-arch=gfx90a -Werror -S -o ${FORGE_GPU_DIR}/kernels.amdgcn.s ${FORGE_GPU_DIR}/kernels.cu)
    set_tests_properties(gpu_compile_cuda gpu_compile_hip PROPERTIES LABELS gpu)
  endif()
  # Cross builds (e.g. aarch64 under qemu) are typically static: skip the shared-library smoke test there.
  if(NOT CMAKE_CROSSCOMPILING)
    add_executable(capi_smoke tests/capi_smoke.c)
    target_include_directories(capi_smoke PRIVATE mind/include)
    target_link_libraries(capi_smoke PRIVATE forge_c)
  endif()
  if(FORGE_BUILD_APPS)
    add_test(NAME pack_signing COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/sign_test.py
             $<TARGET_FILE:forge> ${CMAKE_BINARY_DIR}/golden/tiny_f32.fpk)
    set_tests_properties(pack_signing PROPERTIES DEPENDS golden_tiny_llama)
    add_test(NAME onnx_import COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/onnx_test.py $<TARGET_FILE:forge>)
    add_test(NAME golden_moe_mla
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/verify/golden/tiny_moe.py
              --forge $<TARGET_FILE:forge> --work ${CMAKE_BINARY_DIR}/golden_moe)
    add_test(NAME golden_tiny_llama
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/verify/golden/tiny_llama.py
              --forge $<TARGET_FILE:forge> --work ${CMAKE_BINARY_DIR}/golden)
    if(FORGE_HAVE_VULKAN AND NOT CMAKE_CROSSCOMPILING)
      # Same goldens with the Vulkan lowerings (ops without a GPU kernel run on the CPU).
      add_test(NAME golden_tiny_llama_vulkan COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${CMAKE_BINARY_DIR}/golden_vk)
      add_test(NAME golden_moe_mla_vulkan COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/verify/golden/tiny_moe.py
               --forge $<TARGET_FILE:forge> --work ${CMAKE_BINARY_DIR}/golden_moe_vk)
      set_tests_properties(golden_tiny_llama_vulkan golden_moe_mla_vulkan PROPERTIES ENVIRONMENT "FORGE_DEVICE=vulkan" LABELS gpu)
    endif()
    if(CMAKE_CROSSCOMPILING_EMULATOR)
      list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " _forge_runner)
      set_tests_properties(golden_tiny_llama golden_moe_mla pack_signing onnx_import PROPERTIES ENVIRONMENT "FORGE_RUNNER=${_forge_runner}")
    endif()
    if(TARGET capi_smoke)
      add_test(NAME capi_smoke COMMAND capi_smoke ${CMAKE_BINARY_DIR}/golden/tiny_f32.fpk)
      set_tests_properties(capi_smoke PROPERTIES DEPENDS golden_tiny_llama)
      add_test(NAME python_bindings COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python_bindings.py
               ${CMAKE_BINARY_DIR}/golden/tiny_f32.fpk $<TARGET_FILE:forge> ${CMAKE_BINARY_DIR}/golden/tiny_q8_0.fpk)
      set_tests_properties(python_bindings PROPERTIES DEPENDS golden_tiny_llama ENVIRONMENT "FORGE_LIB=$<TARGET_FILE:forge_c>")
      add_test(NAME serve_http COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/serve_test.py
               $<TARGET_FILE:forge-serve> $<TARGET_FILE:forge> ${CMAKE_BINARY_DIR}/golden/tiny_f32.fpk)
      set_tests_properties(serve_http PROPERTIES DEPENDS golden_tiny_llama)
    endif()
  endif()
endif()

# ---- fuzz targets (fuzz/): libFuzzer entry points + a built-in mutational driver for CI ----
if(FORGE_BUILD_TESTS)
  foreach(ft json blueprint fpk safetensors gguf tokenizer grammar onnx)
    if(FORGE_LIBFUZZER)
      add_executable(fuzz_${ft} fuzz/fuzz_${ft}.cpp)
      target_compile_options(fuzz_${ft} PRIVATE -fsanitize=fuzzer)
      target_link_options(fuzz_${ft} PRIVATE -fsanitize=fuzzer)
    else()
      add_executable(fuzz_${ft} fuzz/fuzz_${ft}.cpp fuzz/driver.cpp)
    endif()
    target_link_libraries(fuzz_${ft} PRIVATE forge_mind forge_verify forge_lane_mv)
    if(NOT FORGE_LIBFUZZER)
      add_test(NAME fuzz_${ft} COMMAND fuzz_${ft} --iters ${FORGE_FUZZ_ITERS} ${CMAKE_SOURCE_DIR}/fuzz/corpus/${ft}
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
      set_tests_properties(fuzz_${ft} PROPERTIES TIMEOUT 600 LABELS fuzz)
    endif()
  endforeach()
endif()
