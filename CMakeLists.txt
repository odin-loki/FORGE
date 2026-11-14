cmake_minimum_required(VERSION 3.25)
project(FORGE VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# clang 18 and older are not supported (their std::expected workaround is gone). CI builds with clang 23; versions
# 19 to 22 (zig cc, the NDK, FreeBSD's base clang) still configure, with a note.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
    message(FATAL_ERROR "FORGE needs clang 23 (found ${CMAKE_CXX_COMPILER_VERSION}; 18 and older cannot build it). Install clang-23 from apt.llvm.org.")
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 23)
    message(STATUS "clang ${CMAKE_CXX_COMPILER_VERSION}: FORGE is built and tested with clang 23")
  endif()
endif()
# libc++ < 20 keeps std::jthread / std::stop_token behind -fexperimental-library: probed in
# cmake/forge_cxx_compat.cmake. The installed package runs the same probe for its consumers.
include(cmake/forge_cxx_compat.cmake)
forge_cxx_compat(_forge_compat_cxx _forge_compat_link)
if(_forge_compat_cxx)
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${_forge_compat_cxx}>")
endif()
if(_forge_compat_link)
  add_link_options("$<$<LINK_LANGUAGE:CXX>:${_forge_compat_link}>")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FORGE_BUILD_TESTS "Build FORGE tests" ON)
option(FORGE_BUILD_APPS  "Build FORGE apps"  ON)
option(FORGE_SANITIZE    "Build with ASan/UBSan" OFF)
option(FORGE_TSAN        "Build with ThreadSanitizer (scripting runtime threads, task pool)" OFF)
option(FORGE_REPRODUCIBLE "Strip build paths and timestamps from artefacts (bit-identical rebuilds)" ON)
if(WIN32)
  link_libraries(ws2_32 bcrypt userenv)   # userenv: AppContainer profiles (os/sandbox.hpp)
  # Windows' default 1 MiB main-thread stack overflows the script interpreter's recursion tests; match Linux's 8 MiB.
  if(MSVC)
    add_link_options(/STACK:8388608)
    # Sources and narrow literals are UTF-8 ("\u2026", collation and charset tables); without this cl.exe
    # re-encodes them to the ANSI code page.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
    # Do not rely on CMake's default MSVC flags (they were empty on a CMake 4.3 + VS 18 configure): without /EHsc
    # destructors do not run during unwinding (a lock_guard left a mutex locked after a throw), and without /O2 the
    # script interpreter suite ran >15 min. /bigobj for the large generated/script TUs.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>
                        $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CONFIG:Release,RelWithDebInfo,MinSizeRel>>:/O2>)
  else()
    add_link_options(-Wl,--stack,8388608)
    # Win64 GCC cannot align the stack past 16 bytes, so its aligned AVX/AVX-512 spills can fault (seen on a VNNI
    # runner); the assembler then emits unaligned moves, which cost nothing on aligned data.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wa,-muse-unaligned-vector-move>)
    endif()
  endif()
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
option(FORGE_JNI          "Build libforge_jni (Android / JVM Kotlin bindings, bindings/android)" OFF)
if(ANDROID OR IOS OR CMAKE_SYSTEM_NAME MATCHES "iOS|tvOS|visionOS")
  set(FORGE_MOBILE ON)
endif()
if(ANDROID)
  # The NDK's libc++ gates parts of C++20/23 that FORGE uses (std::jthread / stop_token, parts of <format>,
  # <chrono> zones) behind -fexperimental-library, which also links libc++experimental.a.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fexperimental-library>)
  add_link_options(-fexperimental-library)
endif()
option(FORGE_LIBFUZZER   "Link fuzz targets with libFuzzer (clang -fsanitize=fuzzer) instead of the built-in driver" OFF)
set(FORGE_FUZZ_ITERS 4000 CACHE STRING "Mutated inputs per fuzz target in ctest")

if(FORGE_REPRODUCIBLE)
  if(NOT MSVC)   # cl.exe has no prefix map (it warns D9002 and ignores the flag)
    add_compile_options(-ffile-prefix-map=${PROJECT_SOURCE_DIR}=. -ffile-prefix-map=${PROJECT_BINARY_DIR}=build)
  endif()
  if(NOT APPLE AND NOT MSVC)   # ld64 and link.exe have no --build-id
    add_link_options(-Wl,--build-id=none)
  endif()
endif()
if(FORGE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
if(FORGE_TSAN)
  if(FORGE_SANITIZE)
    message(FATAL_ERROR "FORGE_TSAN and FORGE_SANITIZE cannot be combined")
  endif()
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()
# ---- build hardening (FSS-BLD-1, docs/security-standard.md) ----
# Each flag is probed and skipped when the toolchain rejects it; the result is reported by
# forge::build_hardening() (core/include/forge/core/build_info.hpp). Must stay after the sanitizer options.
option(FORGE_HARDENING "Hardening compile/link flags: stack protector, FORTIFY_SOURCE, PIE, RELRO/BIND_NOW, CET/BTI, /guard:cf" ON)
include(cmake/forge_hardening.cmake)
# ---- end build hardening ----

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# ---- spec codegen: .fspec -> op table header + docs + oracle test list ----
file(GLOB FORGE_FSPECS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/ops/defs/*.fspec)
set(FORGE_GEN_DIR ${PROJECT_BINARY_DIR}/generated)
add_custom_command(
  OUTPUT ${FORGE_GEN_DIR}/forge/ops/ops.gen.hpp ${FORGE_GEN_DIR}/ops.md
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ops/gen/fspec_gen.py
          --out ${FORGE_GEN_DIR} ${FORGE_FSPECS}
  DEPENDS ${FORGE_FSPECS} ${PROJECT_SOURCE_DIR}/ops/gen/fspec_gen.py
  COMMENT "Generating op registry from ops/defs/*.fspec")
add_custom_target(forge_ops_gen DEPENDS ${FORGE_GEN_DIR}/forge/ops/ops.gen.hpp)

# ---- Tier R: raw intrinsic wrappers generated from ops/intrinsics ----
# One header per ISA, every one includable (and empty) on every platform; forge/simd/raw.hpp picks the
# target's. riscv_policy holds the RVV tail/mask policy variants (opt-in, not included by raw.hpp).
set(FORGE_RAW_ISAS x86 arm riscv riscv_policy loongarch ppc wasm s390x mips)
set(FORGE_RAW_HEADERS)
foreach(isa ${FORGE_RAW_ISAS})
  set(_count intrinsic_count)
  if(isa STREQUAL "riscv_policy")
    set(_count policy_intrinsic_count)
  endif()
  add_custom_command(
    OUTPUT ${FORGE_GEN_DIR}/forge/simd/raw/${isa}.gen.hpp
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ops/gen/intrin_gen.py
            --db ${PROJECT_SOURCE_DIR}/ops/intrinsics/${isa}.json
            --out ${FORGE_GEN_DIR}/forge/simd/raw/${isa}.gen.hpp --count-name ${_count}
    DEPENDS ${PROJECT_SOURCE_DIR}/ops/intrinsics/${isa}.json ${PROJECT_SOURCE_DIR}/ops/gen/intrin_gen.py
    COMMENT "Generating simd Tier R (${isa})")
  list(APPEND FORGE_RAW_HEADERS ${FORGE_GEN_DIR}/forge/simd/raw/${isa}.gen.hpp)
endforeach()
add_custom_target(forge_raw_gen DEPENDS ${FORGE_RAW_HEADERS})

include(GNUInstallDirs)
# Include directories of a FORGE library: source/build tree paths while building FORGE, nothing in the installed
# package (install(TARGETS ... INCLUDES DESTINATION) adds <prefix>/include there), so the exported targets carry no
# absolute path. Relative directories are relative to the source root.
function(forge_include_dirs target scope)
  foreach(d ${ARGN})
    cmake_path(ABSOLUTE_PATH d BASE_DIRECTORY ${PROJECT_SOURCE_DIR} NORMALIZE)
    target_include_directories(${target} ${scope} $<BUILD_INTERFACE:${d}>)
  endforeach()
endfunction()
# Vendored headers are installed under <prefix>/include/forge/third_party.
set(FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/forge/third_party)

add_library(forge_core INTERFACE)
forge_include_dirs(forge_core INTERFACE core/include)
add_library(forge::core ALIAS forge_core)

add_library(forge_os INTERFACE)   # unified OS layer
forge_include_dirs(forge_os INTERFACE os/include)
find_package(Threads)   # FreeBSD/older glibc need -pthread for std::thread users of the OS layer
if(Threads_FOUND)
  target_link_libraries(forge_os INTERFACE Threads::Threads)
endif()
if(WIN32)
  target_link_libraries(forge_os INTERFACE ws2_32 bcrypt advapi32)   # advapi32: fs.hpp check_private / private DACLs
elseif(UNIX AND NOT APPLE)
  target_link_libraries(forge_os INTERFACE ${CMAKE_DL_LIBS})
endif()
target_link_libraries(forge_os INTERFACE forge_trust)   # keystore.hpp file backend: Monocypher AEAD (forge_trust is defined below)
target_link_libraries(forge_core INTERFACE forge_os)
if(WIN32)
  target_link_libraries(forge_core INTERFACE advapi32)   # log.hpp: RegisterEventSourceW / ReportEventW
endif()
# Compile-time log strip (core/log.hpp): calls below this level compile to nothing in every target linking forge_core.
set(FORGE_LOG_MIN_LEVEL "none" CACHE STRING "Strip log calls below this level at compile time: trace debug info warn error fatal off, or none")
set_property(CACHE FORGE_LOG_MIN_LEVEL PROPERTY STRINGS none trace debug info warn error fatal off)
include(cmake/forge_log_level.cmake)
forge_log_level_value("${FORGE_LOG_MIN_LEVEL}" _forge_log_min)
if(_forge_log_min STREQUAL "INVALID")
  message(FATAL_ERROR "FORGE_LOG_MIN_LEVEL='${FORGE_LOG_MIN_LEVEL}': use none, trace, debug, info, warn, error, fatal or off")
elseif(NOT _forge_log_min STREQUAL "")
  target_compile_definitions(forge_core INTERFACE FORGE_LOG_MIN_LEVEL=${_forge_log_min})
  message(STATUS "FORGE: log calls below '${FORGE_LOG_MIN_LEVEL}' are stripped (FORGE_LOG_MIN_LEVEL=${_forge_log_min})")
endif()

add_library(forge_loadbalancer INTERFACE)   # load balancer + L4/L7 proxy
forge_include_dirs(forge_loadbalancer INTERFACE loadbalancer/include)
target_link_libraries(forge_loadbalancer INTERFACE forge_os)
find_package(Threads)
if(Threads_FOUND)
  target_link_libraries(forge_loadbalancer INTERFACE Threads::Threads)
endif()

# forge::os::mobile — sensors, touch, haptics, display, cameras, permissions (docs/mobile.md).
set(FORGE_MOBILE_SRCS os/src/mobile/mobile.cpp os/src/mobile/linux.cpp os/src/mobile/android.cpp
  os/src/mobile/services.cpp os/src/mobile/windows.cpp os/src/mobile/freebsd.cpp)
if(APPLE)
  enable_language(OBJCXX)
  list(APPEND FORGE_MOBILE_SRCS os/src/mobile/apple.mm)
endif()
add_library(forge_mobile STATIC ${FORGE_MOBILE_SRCS})
forge_include_dirs(forge_mobile PUBLIC os/include core/include support/include)
target_link_libraries(forge_mobile PUBLIC forge_core)
if(Threads_FOUND)
  target_link_libraries(forge_mobile PUBLIC Threads::Threads)
endif()
set_target_properties(forge_mobile PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ANDROID)
  target_link_libraries(forge_mobile PUBLIC android camera2ndk ${CMAKE_DL_LIBS})
elseif(WIN32)
  target_link_libraries(forge_mobile PUBLIC shell32 user32)
elseif(APPLE)
  set_source_files_properties(os/src/mobile/apple.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    target_link_libraries(forge_mobile PUBLIC "-framework UIKit" "-framework CoreMotion" "-framework CoreLocation" "-framework AVFoundation"
      "-framework AudioToolbox" "-framework UserNotifications" "-framework LocalAuthentication" "-framework Foundation")
  else()
    # macOS: AVFoundation (camera enumeration/permission), CoreLocation (location), LocalAuthentication (Touch ID)
    target_link_libraries(forge_mobile PUBLIC "-framework Foundation" "-framework AppKit" "-framework IOKit" "-framework UserNotifications"
      "-framework AVFoundation" "-framework CoreLocation" "-framework LocalAuthentication")
  endif()
endif()

# forge::os::text / forge::os::locale: FORGE's own Unicode, encodings and locale library (docs/text-locale.md).
file(GLOB FORGE_TEXT_SRCS CONFIGURE_DEPENDS os/src/text/*.cpp)
if(APPLE)
  enable_language(OBJCXX)
  list(APPEND FORGE_TEXT_SRCS os/src/text/locale_apple.mm)
  set_source_files_properties(os/src/text/locale_apple.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()
add_library(forge_text STATIC ${FORGE_TEXT_SRCS})
forge_include_dirs(forge_text PUBLIC os/include core/include support/include)
target_link_libraries(forge_text PUBLIC forge_core)
set_target_properties(forge_text PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(APPLE)
  target_link_libraries(forge_text PUBLIC "-framework CoreFoundation" "-framework Foundation")
endif()
option(FORGE_FETCH_UCD "Download the Unicode conformance test files for test_text_conformance" OFF)
if(FORGE_FETCH_UCD)
  include(cmake/fetch_ucd.cmake)
endif()

add_library(forge_ops INTERFACE)
forge_include_dirs(forge_ops INTERFACE ops/include ${FORGE_GEN_DIR})
target_link_libraries(forge_ops INTERFACE forge_core)
add_dependencies(forge_ops forge_ops_gen)

add_library(forge_simd INTERFACE)
forge_include_dirs(forge_simd INTERFACE simd/include ${FORGE_GEN_DIR})
target_link_libraries(forge_simd INTERFACE forge_core)
add_dependencies(forge_simd forge_raw_gen)

add_library(forge_verify INTERFACE)
forge_include_dirs(forge_verify INTERFACE verify/include)
target_link_libraries(forge_verify INTERFACE forge_ops)


# ---- GPU kernels generated from spec/ops (FORGE_PLAN B.4): GLSL -> SPIR-V (embedded), CUDA/HIP compile checks ----
set(FORGE_GPU_DIR ${FORGE_GEN_DIR}/gpu)
execute_process(COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR} --list
                OUTPUT_VARIABLE FORGE_GPU_KERNELS OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _gpu_rc)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py)
find_program(FORGE_GLSLANG glslangValidator)
find_program(FORGE_SPIRV_VAL spirv-val)
find_path(FORGE_VULKAN_INCLUDE vulkan/vulkan.h)
set(FORGE_GPU_SOURCES)
foreach(k ${FORGE_GPU_KERNELS})
  list(APPEND FORGE_GPU_SOURCES ${FORGE_GPU_DIR}/${k}.comp)
endforeach()
add_custom_command(OUTPUT ${FORGE_GPU_SOURCES} ${FORGE_GPU_DIR}/kernels.cu ${FORGE_GPU_DIR}/kernels.cl ${FORGE_GPU_DIR}/kernels_cl_mock.inc ${FORGE_GPU_DIR}/kernels_nn_mock.inc ${FORGE_GPU_DIR}/kernels.metal
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR}
  DEPENDS ${FORGE_FSPECS} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py COMMENT "Generating GPU kernels from spec/ops")
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
  add_custom_command(OUTPUT ${FORGE_GEN_DIR}/forge/gpu/spirv.gen.hpp
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR}
            --embed ${FORGE_GEN_DIR}/forge/gpu/spirv.gen.hpp
    DEPENDS ${FORGE_SPV} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py COMMENT "Embedding SPIR-V kernels")
  add_custom_target(forge_gpu_gen ALL DEPENDS ${FORGE_GEN_DIR}/forge/gpu/spirv.gen.hpp)
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
# CUDA / HIP kernel images, built from the generated kernels.cu with clang (no SDK) and embedded for the
# dlopen-based host runtime (gpu/cuda.hpp). HIP code objects need lld; zig's bundled ld.lld works too.
# When the host compiler is clang, the GPU images use that same toolchain (one LLVM for host and device);
# otherwise the first clang on PATH. Override with -DFORGE_GPU_CLANG=... / -DFORGE_LLD=...
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
  # REALPATH: /usr/bin/clang++-23 is a link into /usr/lib/llvm-23/bin; /usr/bin/clang can be an older clang
  get_filename_component(_host_cxx_real "${CMAKE_CXX_COMPILER}" REALPATH)
  get_filename_component(_host_llvm_bin "${_host_cxx_real}" DIRECTORY)
  find_program(FORGE_GPU_CLANG NAMES clang PATHS ${_host_llvm_bin} NO_DEFAULT_PATH)
  find_program(FORGE_LLD NAMES ld.lld lld PATHS ${_host_llvm_bin} NO_DEFAULT_PATH)
endif()
find_program(FORGE_GPU_CLANG NAMES clang-23 clang)
find_program(FORGE_LLD NAMES ld.lld ld.lld-23 lld)
if(NOT FORGE_LLD)
  find_program(FORGE_ZIG NAMES zig)
  if(NOT FORGE_ZIG)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import ziglang, os; print(os.path.join(os.path.dirname(ziglang.__file__), 'zig'))"
                    OUTPUT_VARIABLE _zig OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _zr)
    if(_zr EQUAL 0 AND EXISTS "${_zig}")
      set(FORGE_ZIG ${_zig} CACHE FILEPATH "zig (for ld.lld)" FORCE)
    endif()
  endif()
endif()
set(FORGE_HIP_ARCHS gfx90a gfx942 gfx1100 CACHE STRING "AMDGPU targets embedded for the HIP runtime")
set(FORGE_GPU_IMAGES ${FORGE_GPU_DIR}/kernels.cl ${FORGE_GPU_DIR}/kernels.metal)
if(FORGE_MOBILE)   # no CUDA/HIP/Level Zero on phones; the NDK's lld can't link AMDGPU either
  set(FORGE_GPU_CLANG_IMAGES OFF)
else()
  set(FORGE_GPU_CLANG_IMAGES ${FORGE_GPU_CLANG})
endif()   # OpenCL C, built by the driver at run time
# Distro clangs (e.g. FreeBSD base clang) may be built without the NVPTX/AMDGPU backends.
set(FORGE_GPU_HAS_NVPTX OFF)
set(FORGE_GPU_HAS_AMDGPU OFF)
if(FORGE_GPU_CLANG_IMAGES)
  execute_process(COMMAND ${FORGE_GPU_CLANG} -print-targets OUTPUT_VARIABLE _gpu_targets ERROR_QUIET)
  if(_gpu_targets MATCHES "nvptx64")
    set(FORGE_GPU_HAS_NVPTX ON)
  endif()
  if(_gpu_targets MATCHES "amdgcn")
    set(FORGE_GPU_HAS_AMDGPU ON)
  endif()
  if(NOT FORGE_GPU_HAS_NVPTX OR NOT FORGE_GPU_HAS_AMDGPU)
    message(STATUS "FORGE: ${FORGE_GPU_CLANG} lacks GPU backends (nvptx=${FORGE_GPU_HAS_NVPTX} amdgpu=${FORGE_GPU_HAS_AMDGPU}); skipping those images")
  endif()
endif()
# Level Zero wants SPIR-V: OpenCL C -> LLVM IR (clang, spir64) -> SPIR-V (llvm-spirv), when the translator exists.
find_program(FORGE_LLVM_SPIRV NAMES llvm-spirv llvm-spirv-23)
if(FORGE_GPU_CLANG_IMAGES AND FORGE_LLVM_SPIRV)
  add_custom_command(OUTPUT ${FORGE_GPU_DIR}/kernels.spv
    COMMAND ${FORGE_GPU_CLANG} -x cl -cl-std=CL1.2 --target=spir64 -O2 -emit-llvm -c -o ${FORGE_GPU_DIR}/kernels.cl.bc ${FORGE_GPU_DIR}/kernels.cl
    COMMAND ${FORGE_LLVM_SPIRV} ${FORGE_GPU_DIR}/kernels.cl.bc -o ${FORGE_GPU_DIR}/kernels.spv
    DEPENDS ${FORGE_GPU_DIR}/kernels.cl COMMENT "SPIR-V for Level Zero")
  list(APPEND FORGE_GPU_IMAGES ${FORGE_GPU_DIR}/kernels.spv)
endif()
if(FORGE_GPU_CLANG_IMAGES AND FORGE_GPU_HAS_NVPTX)
  add_custom_command(OUTPUT ${FORGE_GPU_DIR}/kernels.ptx
    COMMAND ${FORGE_GPU_CLANG} -x cuda --cuda-device-only -nocudainc -nocudalib -Wno-unknown-cuda-version --cuda-gpu-arch=sm_70 -O3 -Werror
            -S -o ${FORGE_GPU_DIR}/kernels.ptx ${FORGE_GPU_DIR}/kernels.cu
    DEPENDS ${FORGE_GPU_DIR}/kernels.cu COMMENT "PTX (sm_70)")
  list(APPEND FORGE_GPU_IMAGES ${FORGE_GPU_DIR}/kernels.ptx)
  # sm_80+ image: the same kernels plus the int8 tensor-core prefill matmul; the runtime tries it first.
  add_custom_command(OUTPUT ${FORGE_GPU_DIR}/kernels.sm80.ptx
    COMMAND ${FORGE_GPU_CLANG} -x cuda --cuda-device-only -nocudainc -nocudalib -Wno-unknown-cuda-version --cuda-gpu-arch=sm_80 -O3 -Werror
            -S -o ${FORGE_GPU_DIR}/kernels.sm80.ptx ${FORGE_GPU_DIR}/kernels.cu
    DEPENDS ${FORGE_GPU_DIR}/kernels.cu COMMENT "PTX (sm_80)")
  list(APPEND FORGE_GPU_IMAGES ${FORGE_GPU_DIR}/kernels.sm80.ptx)
endif()
if(FORGE_GPU_CLANG_IMAGES AND FORGE_GPU_HAS_AMDGPU)
  set(_ld)
  if(FORGE_LLD)
    get_filename_component(_lldir ${FORGE_LLD} DIRECTORY)
    set(_ld --ld-path=${FORGE_LLD})
    set(_ldpath ${_lldir})
  elseif(FORGE_ZIG)
    # clang only accepts an --ld-path for AMDGPU whose file is named like lld
    file(COPY ${PROJECT_SOURCE_DIR}/cmake/tools/zig/lld DESTINATION ${PROJECT_BINARY_DIR}/tools
         FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    set(_ld --ld-path=${PROJECT_BINARY_DIR}/tools/lld)
    set(_ldpath ${PROJECT_BINARY_DIR}/tools)   # clang runs "lld" from PATH for the LTO link at -O3
  endif()
  if(_ld)
    foreach(arch ${FORGE_HIP_ARCHS})
      add_custom_command(OUTPUT ${FORGE_GPU_DIR}/kernels.${arch}.hsaco
        COMMAND ${CMAKE_COMMAND} -E env FORGE_ZIG=${FORGE_ZIG} "PATH=${_ldpath}:$ENV{PATH}" ${FORGE_GPU_CLANG} -x hip --cuda-device-only --no-gpu-bundle-output
                -nogpuinc -nogpulib -cuid=forge_kernels --offload-arch=${arch} -O3 -Werror ${_ld} -o ${FORGE_GPU_DIR}/kernels.${arch}.hsaco ${FORGE_GPU_DIR}/kernels.cu
        DEPENDS ${FORGE_GPU_DIR}/kernels.cu COMMENT "HIP code object (${arch})" VERBATIM)   # PATH may hold "(x86)" (WSL)
      list(APPEND FORGE_GPU_IMAGES ${FORGE_GPU_DIR}/kernels.${arch}.hsaco)
    endforeach()
  endif()
endif()
add_custom_command(OUTPUT ${FORGE_GEN_DIR}/forge/gpu/gpu_images.gen.hpp
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py --ops ${FORGE_FSPECS} --out ${FORGE_GPU_DIR}
          --embed-images ${FORGE_GEN_DIR}/forge/gpu/gpu_images.gen.hpp ${FORGE_GPU_IMAGES}
  DEPENDS ${FORGE_GPU_IMAGES} ${PROJECT_SOURCE_DIR}/ops/gen/gpu_gen.py COMMENT "Embedding CUDA/HIP kernel images")
add_custom_target(forge_gpu_images ALL DEPENDS ${FORGE_GEN_DIR}/forge/gpu/gpu_images.gen.hpp)
add_dependencies(forge_gpu_images forge_gpu_gen)
message(STATUS "FORGE: CUDA/HIP images: ${FORGE_GPU_IMAGES}")
add_library(forge_gpu INTERFACE)
forge_include_dirs(forge_gpu INTERFACE gpu/include support/include third_party)
target_link_libraries(forge_gpu INTERFACE forge_os)   # task/pool.hpp -> task/engine.hpp -> task/sched.hpp uses os/cpu_topology.hpp, numa.hpp, sync.hpp
forge_include_dirs(forge_gpu INTERFACE ${FORGE_GEN_DIR})
if(FORGE_HAVE_VULKAN)
  forge_include_dirs(forge_gpu INTERFACE ${FORGE_GEN_DIR}/vkinc)
endif()
target_include_directories(forge_gpu INTERFACE $<INSTALL_INTERFACE:${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}>)   # <level-zero/ze_api.h>
add_dependencies(forge_gpu forge_gpu_gen forge_gpu_images)
if(APPLE)   # Metal runtime (Objective-C++); HARDWARE-UNTESTED
  enable_language(OBJCXX)
  add_library(forge_metal STATIC gpu/src/metal.mm)
  target_compile_options(forge_metal PRIVATE -fobjc-arc)
  target_compile_definitions(forge_metal PUBLIC FORGE_GPU_METAL=1)
  forge_include_dirs(forge_metal PUBLIC gpu/include support/include third_party third_party/monocypher ${FORGE_GEN_DIR} core/include os/include ops/include simd/include)
  add_dependencies(forge_metal forge_gpu_gen forge_gpu_images)
  target_link_libraries(forge_metal PUBLIC "-framework Metal" "-framework Foundation")
  target_link_libraries(forge_gpu INTERFACE forge_metal)
endif()
target_link_libraries(forge_gpu INTERFACE forge_simd)

add_library(forge_graph INTERFACE)
forge_include_dirs(forge_graph INTERFACE graph/include)
target_link_libraries(forge_graph INTERFACE forge_ops forge_simd forge_gpu forge_trust forge_security)   # security: rpc key memory (CRY-2), key file checks (FIO-2)
# TLS for rpc-serve (graph/rpc.hpp). OpenSSL is optional; without it rpc stays plain TCP (auth still works).
option(FORGE_TLS "TLS for forge rpc-serve via OpenSSL when available" ON)
if(FORGE_TLS)
  # Pass -DOPENSSL_ROOT_DIR=<prefix> when OpenSSL lives outside the toolchain's sysroot.
  find_package(OpenSSL QUIET)
  if(OpenSSL_FOUND)
    # Opt-in per target (forge, forge_script, test_rpc): rpc.hpp uses an ABI-tagged inline namespace, so
    # TLS and plain builds of it can coexist in one program without ODR clashes.
    add_library(forge_rpc_tls INTERFACE)
    target_link_libraries(forge_rpc_tls INTERFACE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(forge_rpc_tls INTERFACE FORGE_HAVE_TLS=1)
    message(STATUS "FORGE: rpc TLS via OpenSSL ${OPENSSL_VERSION}")
  else()
    message(STATUS "FORGE: OpenSSL not found; rpc TLS disabled")
  endif()
endif()

# Vendored Monocypher 4.0.2 (BSD-2-Clause / CC0): Ed25519 for .fpk signing, XChaCha20-Poly1305 for the os keystore file backend.
add_library(forge_trust STATIC third_party/monocypher/monocypher.c third_party/monocypher/monocypher-ed25519.c)
forge_include_dirs(forge_trust PUBLIC third_party/monocypher)
target_include_directories(forge_trust PUBLIC $<INSTALL_INTERFACE:${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}/monocypher>)
set_target_properties(forge_trust PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 99)

add_library(forge_modelfile INTERFACE)
forge_include_dirs(forge_modelfile INTERFACE modelfile/include)
# forge_text: the tokenizer (modelfile/tokenizer.hpp) uses its Unicode tables, normalization forms and regex engine.
# Every consumer of the header-only layers above (apps, tests, forge_c, the JNI and xcframework builds, the exported
# forge:: targets) links it from here.
target_link_libraries(forge_modelfile INTERFACE forge_graph forge_trust forge_text)

# forge::security: FORGE Security Standard policy, enforce, audit, lockdown, check (docs/security-api.md). Header-only.
add_library(forge_security INTERFACE)
forge_include_dirs(forge_security INTERFACE security/include support/include)
target_link_libraries(forge_security INTERFACE forge_core forge_trust)   # forge_trust: Ed25519 for detached signatures

add_library(forge_inference INTERFACE)
forge_include_dirs(forge_inference INTERFACE inference/include)
target_link_libraries(forge_inference INTERFACE forge_modelfile forge_security)   # FSS model integrity / fs scope at load
# forge::compress: model compression at import time - pruning (magnitude / Wanda / RIA / SparseGPT, N:M), masked
# refits, layer-wise allocation (OWL / EBA), structured width / depth pruning, MoE expert pruning / merging and
# MoE -> dense conversion, calibrated on FORGE graphs; evaluation on FORGE sessions (docs/compress.md). Header-only.
add_library(forge_compress INTERFACE)
forge_include_dirs(forge_compress INTERFACE compress/include)
target_link_libraries(forge_compress INTERFACE forge_inference)
forge_include_dirs(forge_inference INTERFACE inference/include)
# forge::script managed scripting runtime (docs/scripting.md). Safe C++23; libc++ hardening when available.
# Shell backends: one file each in script/src/shells/ defining forge_shell_register_<stem>(); the registry TU is generated.
file(GLOB FORGE_SHELL_SRCS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/script/src/shells/*.cpp)
set(_shell_decl "") 
set(_shell_call "")
foreach(f ${FORGE_SHELL_SRCS})
  get_filename_component(_stem ${f} NAME_WE)
  string(APPEND _shell_decl "void forge_shell_register_${_stem}(forge::script::shell_registry&);\n")
  string(APPEND _shell_call "        forge_shell_register_${_stem}(r);\n")
endforeach()
file(CONFIGURE OUTPUT ${PROJECT_BINARY_DIR}/generated/shells_registry.cpp CONTENT
"// generated by CMake from script/src/shells/*.cpp\n#include \"shells/shell.hpp\"\n${_shell_decl}namespace forge::script {\nconst shell_registry& shells() {\n    static const shell_registry reg = [] {\n        shell_registry r;\n${_shell_call}        return r;\n    }();\n    return reg;\n}\n}\n")
add_library(forge_script STATIC ${FORGE_SHELL_SRCS} ${PROJECT_BINARY_DIR}/generated/shells_registry.cpp script/src/heap.cpp script/src/parser.cpp script/src/runtime.cpp script/src/tree.cpp
  script/src/vm.cpp script/src/builtins.cpp script/src/cppstd.cpp script/src/automation.cpp script/src/ai.cpp script/src/remote.cpp script/src/mobile.cpp script/src/text.cpp script/src/reflect.cpp script/src/emit.cpp script/src/debug.cpp script/src/engine.cpp script/src/fsc.cpp script/src/check.cpp
  script/src/aot_codegen.cpp script/src/aot_compile.cpp script/src/aot_rt.cpp)   # ahead-of-time compilation (docs/aot.md)
forge_include_dirs(forge_script PUBLIC script/include)
target_include_directories(forge_script PRIVATE script/src)
target_link_libraries(forge_script PUBLIC forge_modelfile forge_mobile forge_text PRIVATE forge_verify forge_trust forge_security)
if(TARGET forge_rpc_tls)
  target_link_libraries(forge_script PUBLIC forge_rpc_tls)
endif()
find_package(Threads)
if(Threads_FOUND)
  target_link_libraries(forge_script PUBLIC Threads::Threads)
endif()
target_compile_options(forge_script PRIVATE $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall -Wextra>)
set_target_properties(forge_script PROPERTIES POSITION_INDEPENDENT_CODE ON)
# ---- forge script --aot (docs/aot.md) --------------------------------------------------------------------------
# forge_script_rt: the AOT runtime library that natively compiled scripts link against. It is forge_script plus the
# static libraries forge_script needs, in one archive, so an installed `forge` can compile scripts without the
# source tree. forge-script-run: the runner stub embed mode copies. forge-aot-runtime.cfg tells the AOT driver
# (script/src/aot_compile.cpp) where the headers and the library are and which system libraries to link.
set(_aot_objs "")
set(_aot_ld "")
set(_aot_seen "")
function(_forge_aot_walk tgt)
  if(tgt IN_LIST _aot_seen)
    return()
  endif()
  list(APPEND _aot_seen ${tgt})
  get_target_property(_type ${tgt} TYPE)
  get_target_property(_imp ${tgt} IMPORTED)
  set(_deps "")
  if(_type STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_deps ${tgt} INTERFACE_LINK_LIBRARIES)
  elseif(_imp)
    list(APPEND _aot_ld "$<TARGET_LINKER_FILE:${tgt}>")
    get_target_property(_deps ${tgt} INTERFACE_LINK_LIBRARIES)
  elseif(_type STREQUAL "STATIC_LIBRARY")
    list(APPEND _aot_objs "$<TARGET_OBJECTS:${tgt}>")
    get_target_property(_d1 ${tgt} LINK_LIBRARIES)
    get_target_property(_d2 ${tgt} INTERFACE_LINK_LIBRARIES)
    foreach(_d IN LISTS _d1 _d2)
      if(_d)
        list(APPEND _deps ${_d})
      endif()
    endforeach()
  endif()
  foreach(_d IN LISTS _deps)
    if(NOT _d OR _d MATCHES "NOTFOUND$")
      continue()
    endif()
    string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" _d "${_d}")
    if(TARGET ${_d})
      _forge_aot_walk(${_d})
    elseif(_d MATCHES "^\\$<" OR _d MATCHES "^-" OR IS_ABSOLUTE "${_d}")
      list(APPEND _aot_ld "${_d}")
    elseif(MSVC)
      list(APPEND _aot_ld "${_d}.lib")
    else()
      list(APPEND _aot_ld "-l${_d}")
    endif()
  endforeach()
  set(_aot_objs "${_aot_objs}" PARENT_SCOPE)
  set(_aot_ld "${_aot_ld}" PARENT_SCOPE)
  set(_aot_seen "${_aot_seen}" PARENT_SCOPE)
endfunction()
_forge_aot_walk(forge_script)
list(REMOVE_DUPLICATES _aot_ld)
add_library(forge_script_rt STATIC ${_aot_objs})
set_target_properties(forge_script_rt PROPERTIES LINKER_LANGUAGE CXX)
add_executable(forge-script-run script/rt/stub_main.cpp)
target_link_libraries(forge-script-run PRIVATE forge_script)
# Every embed-mode executable is a copy of this stub, so it is linked without symbols and debug information whatever
# the build type (MSVC keeps those in the .pdb already).
if(APPLE)
  target_link_options(forge-script-run PRIVATE -Wl,-S -Wl,-x)
elseif(NOT MSVC)
  target_link_options(forge-script-run PRIVATE -s)
endif()
# Headers a generated program includes: forge_aot_rt.hpp, the interpreter's internal headers it builds on, and the
# core / os headers those need. Build tree: straight from the source tree. Install tree: see the install block.
set(FORGE_AOT_RT_HEADERS script/rt/forge_aot_rt.hpp script/src/aot.hpp script/src/interp.hpp script/src/ast.hpp script/src/objects.hpp script/src/value.hpp)
set(FORGE_AOT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/script/rt ${PROJECT_SOURCE_DIR}/script/src ${PROJECT_SOURCE_DIR}/core/include ${PROJECT_SOURCE_DIR}/os/include
    ${PROJECT_SOURCE_DIR}/third_party/monocypher)   # os/locked_memory.hpp wipes with Monocypher
set(_aot_cfg "# AOT runtime for `forge script --aot` (docs/aot.md). Relative paths are relative to this file.\nabi=1\n")
if(CMAKE_CXX_COMPILER_TARGET)
  string(APPEND _aot_cfg "triple=${CMAKE_CXX_COMPILER_TARGET}\n")
endif()
if(UNIX AND NOT APPLE)
  string(APPEND _aot_cfg "cxxflag=-pthread\n")
endif()
foreach(_f IN LISTS _aot_ld)
  string(APPEND _aot_cfg "ldflag=${_f}\n")
endforeach()
if(UNIX AND NOT APPLE AND NOT ANDROID)
  string(APPEND _aot_cfg "ldflag=-pthread\n")
endif()
# A sanitizer build's runtime library is instrumented, so the link of an AOT binary needs the sanitizer runtime
# (without it every native-mode compile in the build tree failed with undefined __asan_* / __ubsan_* symbols and
# test_script_aot could not pass in the linux-asan job). Build-tree configuration only.
set(_aot_cfg_san "")
if(FORGE_SANITIZE)
  set(_aot_cfg_san "ldflag=-fsanitize=address,undefined\n")
elseif(FORGE_TSAN)
  set(_aot_cfg_san "ldflag=-fsanitize=thread\n")
endif()
set(_aot_cfg_build "${_aot_cfg}${_aot_cfg_san}lib=$<TARGET_FILE:forge_script_rt>\nstub=$<TARGET_FILE:forge-script-run>\n")
foreach(_i IN LISTS FORGE_AOT_INCLUDE_DIRS)
  string(APPEND _aot_cfg_build "include=${_i}\n")
endforeach()
# Next to the executables, where find_runtime's <exe dir>/aot-runtime finds it: the build root, or <config>/ with a
# multi-config generator (Xcode, Visual Studio), one file per configuration since the paths in it differ. No absolute
# path is compiled in: it would make the forge binary depend on the build directory (the repro job compares two).
get_property(_aot_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(_aot_multi)
  file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/$<CONFIG>/aot-runtime/forge-aot-runtime.cfg CONTENT "${_aot_cfg_build}")
else()
  file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/aot-runtime/forge-aot-runtime.cfg CONTENT "${_aot_cfg_build}")
endif()
# Example native plugin (C ABI in script/include/forge/script/plugin.h); also used by test_script.
add_library(forge_script_example_plugin MODULE examples/script_plugin/hello_plugin.c)
target_include_directories(forge_script_example_plugin PRIVATE script/include)
set_target_properties(forge_script_example_plugin PROPERTIES OUTPUT_NAME hello_plugin C_VISIBILITY_PRESET hidden)
add_library(forge_c SHARED inference/capi/forge_c.cpp)
target_link_libraries(forge_c PRIVATE forge_inference forge_verify forge_simd_mv)
target_compile_definitions(forge_c PRIVATE FORGE_C_BUILD)
if(MSVC)   # forge.exe and forge.dll would both link to forge.pdb
  set_target_properties(forge_c PROPERTIES PDB_NAME forge_c)
endif()
set_target_properties(forge_c PROPERTIES OUTPUT_NAME forge VERSION ${PROJECT_VERSION} SOVERSION 1 CXX_VISIBILITY_PRESET default)
if(FORGE_JNI)
  find_package(JNI QUIET)
  add_library(forge_jni SHARED bindings/android/jni/forge_jni.c)
  target_include_directories(forge_jni PRIVATE inference/include)
  if(NOT ANDROID AND JNI_FOUND)
    target_include_directories(forge_jni PRIVATE ${JNI_INCLUDE_DIRS})
  endif()
  target_link_libraries(forge_jni PRIVATE forge_c)
  if(ANDROID)
    target_link_libraries(forge_jni PRIVATE log)
  endif()
endif()
# Import templates (forge/import/hf.hpp template_dir()): $FORGE_TEMPLATES, then <exe dir>/<FORGE_TEMPLATES_RELDIR>
# (an installed tree), then the source tree. The source path is a build-tree definition only: it is not exported.
target_compile_definitions(forge_inference INTERFACE $<BUILD_INTERFACE:FORGE_SOURCE_DIR="${PROJECT_SOURCE_DIR}">)
set(FORGE_INSTALL_TEMPLATEDIR ${CMAKE_INSTALL_DATADIR}/forge/templates)
if(NOT IS_ABSOLUTE "${CMAKE_INSTALL_BINDIR}" AND NOT IS_ABSOLUTE "${FORGE_INSTALL_TEMPLATEDIR}")
  file(RELATIVE_PATH _forge_tpl_rel "/p/${CMAKE_INSTALL_BINDIR}" "/p/${FORGE_INSTALL_TEMPLATEDIR}")
  if(NOT _forge_tpl_rel STREQUAL "../share/forge/templates")   # the header's default
    target_compile_definitions(forge_modelfile INTERFACE FORGE_TEMPLATES_RELDIR="${_forge_tpl_rel}")
  endif()
endif()

# Multiversioned simd kernels: one TU per target, own flags, own namespace.
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

# One table from simd/src/kernels_gn.cpp (the vec_gn.hpp backends): forge_simd_gn_table(<name> <header> <cpp condition>
# <flags>...). Each <flags> is a |-separated candidate ("none" = no flags), tried in order (GCC/Clang spelling, then
# zig's); the first with which <header> compiles and <condition> holds is used. Appends FORGE_SIMD_HAVE_<NAME>=1
# to FORGE_SIMD_GN_DEFS (for dispatch.cpp) and sets FORGE_SIMD_<NAME>_FLAGS. Call after forge_simd_mv exists.
function(forge_simd_gn_table name header cond)
  string(TOUPPER "${name}" _up)
  set(FORGE_SIMD_${_up}_FLAGS "" PARENT_SCOPE)
  foreach(cand IN LISTS ARGN)
    if(cand STREQUAL "none")
      set(cand "")
    endif()
    string(REPLACE "|" " " CMAKE_REQUIRED_FLAGS "${cand}")
    string(REPLACE "|" ";" _fl "${cand}")
    unset(_ok CACHE)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)   # compile only: the flags are for one TU, never for libc
    check_cxx_source_compiles("#include <${header}>\n#if !(${cond})\n#error not enabled\n#endif\nint main(){return 0;}" _ok)
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    if(_ok)
      add_library(forge_simd_${name} OBJECT simd/src/kernels_gn.cpp)
      target_sources(forge_simd_${name} PRIVATE simd/src/isa_extra.cpp)   # Tier A, seq and int8 tables of the same ISA
      target_compile_options(forge_simd_${name} PRIVATE ${_fl})
      if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        target_compile_options(forge_simd_${name} PRIVATE -ffp-contract=off)   # see vec_gn.hpp
      endif()
      if(name STREQUAL "neon32")
        # AArch32 NEON arithmetic flushes subnormals; vec_gn.hpp repairs its own operations, and the scalar code
        # of these files must stay VFP instructions, so the compiler may not turn it into NEON ones.
        target_compile_options(forge_simd_${name} PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)
      endif()
      set_target_properties(forge_simd_${name} PROPERTIES POSITION_INDEPENDENT_CODE ON)   # linked into libforge.so
      target_link_libraries(forge_simd_${name} PRIVATE forge_simd)
      target_sources(forge_simd_mv PRIVATE $<TARGET_OBJECTS:forge_simd_${name}>)
      set(FORGE_SIMD_GN_DEFS ${FORGE_SIMD_GN_DEFS} FORGE_SIMD_HAVE_${_up}=1 PARENT_SCOPE)
      if(_fl)
        set(FORGE_SIMD_${_up}_FLAGS "${_fl}" PARENT_SCOPE)
      else()
        set(FORGE_SIMD_${_up}_FLAGS "none" PARENT_SCOPE)
      endif()
      return()
    endif()
  endforeach()
endfunction()
set(FORGE_SIMD_GN_DEFS)
# simd/src/isa_extra.cpp: the Tier A, seq and int8 tables of one Tier P backend, compiled beside each kernel
# table with the same flags. No contraction of a*b+c: the seq and int8 kernels must give the scalar bits.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(simd/src/isa_extra.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
else()
  set_source_files_properties(simd/src/isa_extra.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

# Scalable vectors: one fixed-length kernel table per VL (128..1024 bits); dispatch picks the hardware's.
set(FORGE_SCALABLE_BITS 128 256 512 1024)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"))
  set(FORGE_ARCH aarch64)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/kernels_neon.cpp simd/src/isa_extra.cpp simd/src/dispatch.cpp)
  set(_defs)
  foreach(b ${FORGE_SCALABLE_BITS})
    set(FORGE_PROBE_BITS ${b})
    forge_probe_flags(FORGE_SVE${b}_FLAGS arm_sve.h svfloat32_t "arm_sve_vector_bits(${b})"
      "-march=armv8.2-a+sve|-msve-vector-bits=${b}" "-mcpu=generic+sve|-msve-vector-bits=${b}")
    if(FORGE_SVE${b}_FLAGS)
      add_library(forge_simd_sve${b} OBJECT simd/src/kernels_sve.cpp simd/src/isa_extra.cpp)
      target_compile_options(forge_simd_sve${b} PRIVATE ${FORGE_SVE${b}_FLAGS})
      set_target_properties(forge_simd_sve${b} PROPERTIES POSITION_INDEPENDENT_CODE ON)   # linked into libforge.so
      target_link_libraries(forge_simd_sve${b} PRIVATE forge_simd)
      target_sources(forge_simd_mv PRIVATE $<TARGET_OBJECTS:forge_simd_sve${b}>)
      list(APPEND _defs FORGE_SIMD_SVE${b}=1)
    endif()
  endforeach()
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${_defs}")
  message(STATUS "FORGE: SVE kernel tables: ${_defs}")
  # Integer matvec kernels (simd/qdot_i8.hpp): NEON always; the dot-product extension and SVE (vector-length
  # agnostic, one object for every VL) when the compiler has them. No fused multiply-adds beyond the explicit
  # fma: every path gives the scalar reference's bits.
  target_sources(forge_simd_mv PRIVATE simd/src/qi8_neon.cpp simd/src/qi8_dotprod.cpp simd/src/qi8_sve.cpp)
  set(_qi8_defs)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_qi8_fp "")
  else()
    set(_qi8_fp "-ffp-contract=off")
    set(FORGE_QI8_DOTPROD_FLAGS "")
    foreach(_f "-march=armv8.2-a+dotprod" "-mcpu=generic+dotprod")
      if(NOT FORGE_QI8_DOTPROD_FLAGS)
        set(CMAKE_REQUIRED_FLAGS "${_f}")
        unset(FORGE_CAN_DOTPROD CACHE)
        check_cxx_source_compiles("#include <arm_neon.h>\nint main(){int8x16_t a=vdupq_n_s8(1);return vaddvq_s32(vdotq_s32(vdupq_n_s32(0),a,a));}" FORGE_CAN_DOTPROD)
        if(FORGE_CAN_DOTPROD)
          set(FORGE_QI8_DOTPROD_FLAGS "${_f}")
        endif()
      endif()
    endforeach()
    set(FORGE_QI8_SVE_FLAGS "")
    if(NOT APPLE)   # no Apple core has SVE outside streaming mode
      foreach(_f "-march=armv8.2-a+sve" "-mcpu=generic+sve")
        if(NOT FORGE_QI8_SVE_FLAGS)
          set(CMAKE_REQUIRED_FLAGS "${_f}")
          unset(FORGE_CAN_QI8_SVE CACHE)
          check_cxx_source_compiles("#include <arm_sve.h>\nint main(){svint8_t a=svdup_n_s8(1);return (int)svaddv_s32(svptrue_b32(),svdot_s32(svdup_n_s32(0),a,a));}" FORGE_CAN_QI8_SVE)
          if(FORGE_CAN_QI8_SVE)
            set(FORGE_QI8_SVE_FLAGS "${_f}")
          endif()
        endif()
      endforeach()
    endif()
    unset(CMAKE_REQUIRED_FLAGS)
    if(FORGE_QI8_DOTPROD_FLAGS)
      set_source_files_properties(simd/src/qi8_dotprod.cpp PROPERTIES COMPILE_OPTIONS "${FORGE_QI8_DOTPROD_FLAGS};${_qi8_fp}" COMPILE_DEFINITIONS FORGE_QI8_HAVE_DOTPROD=1)
      list(APPEND _qi8_defs FORGE_QI8_HAVE_DOTPROD=1)
    endif()
    if(FORGE_QI8_SVE_FLAGS)
      set_source_files_properties(simd/src/qi8_sve.cpp PROPERTIES COMPILE_OPTIONS "${FORGE_QI8_SVE_FLAGS};${_qi8_fp}" COMPILE_DEFINITIONS FORGE_QI8_HAVE_SVE=1)
      list(APPEND _qi8_defs FORGE_QI8_HAVE_SVE=1)
    endif()
  endif()
  set_source_files_properties(simd/src/qi8_neon.cpp PROPERTIES COMPILE_OPTIONS "${_qi8_fp}")
  set_source_files_properties(simd/src/qi8_dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${_qi8_defs}")
  message(STATUS "FORGE: integer matvec kernels: neon ${_qi8_defs}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
  set(FORGE_ARCH riscv64)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  set(_defs)
  foreach(b ${FORGE_SCALABLE_BITS})
    set(FORGE_PROBE_BITS ${b})
    forge_probe_flags(FORGE_RVV${b}_FLAGS riscv_vector.h vfloat32m1_t "riscv_rvv_vector_bits(${b})"
      "-march=rv64gcv_zvl${b}b|-mrvv-vector-bits=${b}" "-march=rv64gcv_zvl${b}b|-mrvv-vector-bits=zvl" "-mcpu=baseline_rv64+v+zvl${b}b|-mrvv-vector-bits=${b}")
    if(FORGE_RVV${b}_FLAGS)
      add_library(forge_simd_rvv${b} OBJECT simd/src/kernels_rvv.cpp simd/src/isa_extra.cpp)
      target_compile_options(forge_simd_rvv${b} PRIVATE ${FORGE_RVV${b}_FLAGS})
      set_target_properties(forge_simd_rvv${b} PROPERTIES POSITION_INDEPENDENT_CODE ON)   # linked into libforge.so
      target_link_libraries(forge_simd_rvv${b} PRIVATE forge_simd)
      target_sources(forge_simd_mv PRIVATE $<TARGET_OBJECTS:forge_simd_rvv${b}>)
      list(APPEND _defs FORGE_SIMD_RVV${b}=1)
    endif()
  endforeach()
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${_defs}")
  message(STATUS "FORGE: RVV kernel tables: ${_defs}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64")
  # LoongArch64: LSX (128-bit) and LASX (256-bit) tables; dispatch reads AT_HWCAP.
  set(FORGE_ARCH loongarch64)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  forge_simd_gn_table(lsx lsxintrin.h "defined(__loongarch_sx) && !defined(__loongarch_asx)" "-mlsx|-mno-lasx" "-mlsx")
  forge_simd_gn_table(lasx lasxintrin.h "defined(__loongarch_asx)" "-mlasx")
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${FORGE_SIMD_GN_DEFS}")
  message(STATUS "FORGE: LoongArch kernel tables: ${FORGE_SIMD_GN_DEFS}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64le|powerpc64le)$")
  # ppc64le: one VSX table at the POWER8 baseline of the ELFv2 ABI.
  set(FORGE_ARCH ppc64le)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  forge_simd_gn_table(vsx altivec.h "defined(__VSX__) && defined(_ARCH_PWR8)" "-mcpu=power8|-mvsx" "-mcpu=pwr8" "-mvsx" "none")
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${FORGE_SIMD_GN_DEFS}")
  message(STATUS "FORGE: Power kernel tables: ${FORGE_SIMD_GN_DEFS}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
  # s390x: one table for the z14 vector enhancements facility (binary32 vectors); dispatch reads AT_HWCAP.
  set(FORGE_ARCH s390x)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  forge_simd_gn_table(vxe vecintrin.h "defined(__VEC__) && __VEC__ >= 10302" "-march=z14|-mzvector" "-mcpu=z14|-mzvector")
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${FORGE_SIMD_GN_DEFS}")
  message(STATUS "FORGE: s390x kernel tables: ${FORGE_SIMD_GN_DEFS}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "wasm" OR EMSCRIPTEN)
  # WebAssembly: a module either uses SIMD128 or does not, so the choice is made here (-DFORGE_WASM_SIMD=OFF for
  # engines without it) and the table is compiled in or left out. There is no run-time detection.
  set(FORGE_ARCH wasm)
  option(FORGE_WASM_SIMD "Build the WebAssembly SIMD128 kernel table" ON)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  if(FORGE_WASM_SIMD)
    forge_simd_gn_table(wasm128 wasm_simd128.h "defined(__wasm_simd128__)" "-msimd128")
  endif()
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${FORGE_SIMD_GN_DEFS}")
  message(STATUS "FORGE: WebAssembly kernel tables: ${FORGE_SIMD_GN_DEFS}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^mips64")
  # MIPS64: MSA table; dispatch reads AT_HWCAP.
  set(FORGE_ARCH mips64)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  forge_simd_gn_table(msa msa.h "defined(__mips_msa)" "-mmsa" "-mcpu=mips64r5+msa")
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${FORGE_SIMD_GN_DEFS}")
  message(STATUS "FORGE: MIPS kernel tables: ${FORGE_SIMD_GN_DEFS}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7|armv8l)")
  # 32-bit Arm: NEON with VFPv4 (fused multiply-add); dispatch reads AT_HWCAP.
  set(FORGE_ARCH arm)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  forge_simd_gn_table(neon32 arm_neon.h "defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)" "-mfpu=neon-vfpv4" "-mcpu=cortex_a7" "none")
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${FORGE_SIMD_GN_DEFS}")
  message(STATUS "FORGE: Arm (32-bit) kernel tables: ${FORGE_SIMD_GN_DEFS}")
elseif(CMAKE_SYSTEM_PROCESSOR AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|EM64T|i[3-6]86|i86pc|x86|X86)$" AND NOT APPLE AND NOT MSVC)
  # Anything else: the scalar table only.
  set(FORGE_ARCH generic)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  message(STATUS "FORGE: no SIMD kernel tables for ${CMAKE_SYSTEM_PROCESSOR}: scalar only")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
  # 32-bit x86: the SSE tables only (the AVX tables, integer matvec and tile kernels are x86-64 code).
  set(FORGE_ARCH x86)
  add_library(forge_simd_mv STATIC simd/src/kernels_scalar.cpp simd/src/dispatch.cpp)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    forge_simd_gn_table(sse2 emmintrin.h "1" "/arch:SSE2")
    forge_simd_gn_table(sse42 smmintrin.h "1" "/arch:SSE2|/DFORGE_SIMD_SSE41=1|/DFORGE_SIMD_TABLE_SSE42=1")
  else()
    forge_simd_gn_table(sse2 emmintrin.h "defined(__SSE2__)" "-msse2|-mfpmath=sse")
    forge_simd_gn_table(sse42 smmintrin.h "defined(__SSE4_2__)" "-msse4.2|-mpopcnt|-mfpmath=sse|-DFORGE_SIMD_TABLE_SSE42=1")
  endif()
  if(TARGET forge_simd_sse2 AND TARGET forge_simd_sse42)
    set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "FORGE_SIMD_HAVE_SSE=1")
  endif()
  message(STATUS "FORGE: x86 (32-bit) kernel tables: ${FORGE_SIMD_GN_DEFS}")
else()
  set(FORGE_ARCH x86_64)
  add_library(forge_simd_mv STATIC
    simd/src/kernels_scalar.cpp
    simd/src/kernels_avx2.cpp
    simd/src/kernels_avx512.cpp
    simd/src/dispatch.cpp)
  # SSE tier (vec_gn.hpp): an SSE2 table at the x86-64 baseline and an SSE4.2 (x86-64-v2) table for pre-AVX2 CPUs.
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    forge_simd_gn_table(sse2 emmintrin.h "1" "none")
    forge_simd_gn_table(sse42 smmintrin.h "1" "/DFORGE_SIMD_SSE41=1|/DFORGE_SIMD_TABLE_SSE42=1")
  else()
    forge_simd_gn_table(sse2 emmintrin.h "defined(__SSE2__)" "none")
    forge_simd_gn_table(sse42 smmintrin.h "defined(__SSE4_2__)" "-msse4.2|-mpopcnt|-DFORGE_SIMD_TABLE_SSE42=1")
  endif()
  set_source_files_properties(simd/src/dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "FORGE_SIMD_HAVE_AVX2=1;FORGE_SIMD_HAVE_SSE=1")
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(simd/src/kernels_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(simd/src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(simd/src/kernels_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(simd/src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")
  endif()
  # Integer matvec kernels (simd/qdot_i8.hpp): AVX2 always, AVX-VNNI / AVX-512 VNNI when the compiler has them.
  target_sources(forge_simd_mv PRIVATE simd/src/qi8_avx2.cpp simd/src/qi8_avxvnni.cpp simd/src/qi8_avx512vnni.cpp)
  set(_qi8_defs)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(simd/src/qi8_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(simd/src/qi8_avxvnni.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2" COMPILE_DEFINITIONS FORGE_QI8_HAVE_AVXVNNI=1)
    set_source_files_properties(simd/src/qi8_avx512vnni.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512" COMPILE_DEFINITIONS FORGE_QI8_HAVE_AVX512VNNI=1)
    list(APPEND _qi8_defs FORGE_QI8_HAVE_AVXVNNI=1 FORGE_QI8_HAVE_AVX512VNNI=1)
  else()
    include(CheckCXXSourceCompiles)
    set(_qi8_v3 "-mavx2;-mfma;-mf16c;-mbmi2;-ffp-contract=off")   # no fused multiply-adds: every path gives the scalar reference's bits
    set_source_files_properties(simd/src/qi8_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_qi8_v3}")
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c -mavxvnni")
    check_cxx_source_compiles("#include <immintrin.h>\nint main(){__m256i a=_mm256_setzero_si256();a=_mm256_dpbusd_avx_epi32(a,a,a);return _mm256_extract_epi32(a,0);}" FORGE_CAN_AVXVNNI)
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512vnni")
    check_cxx_source_compiles("#include <immintrin.h>\nint main(){__m256i a=_mm256_setzero_si256();a=_mm256_dpbusd_epi32(a,a,a);return _mm256_extract_epi32(a,0);}" FORGE_CAN_AVX512VNNI)
    unset(CMAKE_REQUIRED_FLAGS)
    if(FORGE_CAN_AVXVNNI)
      set_source_files_properties(simd/src/qi8_avxvnni.cpp PROPERTIES COMPILE_OPTIONS "${_qi8_v3};-mavxvnni" COMPILE_DEFINITIONS FORGE_QI8_HAVE_AVXVNNI=1)
      list(APPEND _qi8_defs FORGE_QI8_HAVE_AVXVNNI=1)
    endif()
    if(FORGE_CAN_AVX512VNNI)
      set_source_files_properties(simd/src/qi8_avx512vnni.cpp PROPERTIES COMPILE_OPTIONS "${_qi8_v3};-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni" COMPILE_DEFINITIONS FORGE_QI8_HAVE_AVX512VNNI=1)
      list(APPEND _qi8_defs FORGE_QI8_HAVE_AVX512VNNI=1)
    endif()
  endif()
  set_source_files_properties(simd/src/qi8_dispatch.cpp PROPERTIES COMPILE_DEFINITIONS "${_qi8_defs}")
  message(STATUS "FORGE: integer matvec kernels: avx2 ${_qi8_defs}")
endif()
target_sources(forge_simd_mv PRIVATE simd/src/qi8_dispatch.cpp)
target_link_libraries(forge_simd_mv PUBLIC forge_simd)
set_target_properties(forge_simd_mv PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Matrix tile units (simd/tile.hpp): AMX on x86_64, SME on aarch64. Each file stubs out when its unit can't be compiled.
include(CheckCXXSourceCompiles)
target_sources(forge_simd_mv PRIVATE simd/src/tile_amx.cpp simd/src/tile_sme.cpp)
if(FORGE_ARCH STREQUAL "x86_64")
  set(CMAKE_REQUIRED_FLAGS "-mamx-tile -mamx-bf16 -mamx-int8")
  check_cxx_source_compiles("#include <immintrin.h>\nint main(){_tile_zero(0);_tile_release();return 0;}" FORGE_CAN_AMX)
  unset(CMAKE_REQUIRED_FLAGS)
  if(FORGE_CAN_AMX)
    set_source_files_properties(simd/src/tile_amx.cpp PROPERTIES COMPILE_OPTIONS "-mamx-tile;-mamx-bf16;-mamx-int8" COMPILE_DEFINITIONS FORGE_HAVE_AMX=1)
  endif()
elseif(FORGE_ARCH STREQUAL "aarch64")
  foreach(_f "-march=armv9-a+sme" "-mcpu=generic+sme")
    set(CMAKE_REQUIRED_FLAGS "${_f}")
    unset(FORGE_CAN_SME CACHE)
    # Compile-only: some runtimes (zig/musl, older libgcc) lack the SME ABI support routines; tile_sme.cpp
    # carries weak fallbacks for those.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    check_cxx_source_compiles("#include <arm_sme.h>\n__arm_new(\"za\") void f() __arm_streaming { svzero_za(); }\nint main(){return 0;}" FORGE_CAN_SME)
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
    unset(CMAKE_REQUIRED_FLAGS)
    if(FORGE_CAN_SME)
      set_source_files_properties(simd/src/tile_sme.cpp PROPERTIES COMPILE_OPTIONS "${_f}" COMPILE_DEFINITIONS FORGE_HAVE_SME=1)
      break()
    endif()
  endforeach()
endif()
message(STATUS "FORGE: tile units AMX=${FORGE_CAN_AMX} SME=${FORGE_CAN_SME}")
# Sequence-mixing kernels (simd/seq.hpp: chunked attention, gated delta rule): scalar everywhere, AVX2 and
# AVX-512 tables on x86_64. No contraction of a*b+c anywhere, so every table returns identical bits.
target_sources(forge_simd_mv PRIVATE simd/src/seq_scalar.cpp simd/src/seq_dispatch.cpp)
if(FORGE_ARCH STREQUAL "x86_64")
  target_sources(forge_simd_mv PRIVATE simd/src/seq_avx2.cpp simd/src/seq_avx512.cpp)
endif()
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set_source_files_properties(simd/src/seq_scalar.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
  set_source_files_properties(simd/src/seq_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:precise")
  set_source_files_properties(simd/src/seq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/fp:precise")
else()
  set_source_files_properties(simd/src/seq_scalar.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  set_source_files_properties(simd/src/seq_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
  set_source_files_properties(simd/src/seq_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma;-ffp-contract=off")
endif()
# Tier A algorithm tables (simd/tier_a.hpp: search, sort, scan, text, bitsets): scalar everywhere, AVX2 and
# AVX-512 tables on x86_64.
target_sources(forge_simd_mv PRIVATE simd/src/tiera_scalar.cpp simd/src/tiera_dispatch.cpp)
if(FORGE_ARCH STREQUAL "x86_64")
  target_sources(forge_simd_mv PRIVATE simd/src/tiera_avx2.cpp simd/src/tiera_avx512.cpp)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(simd/src/tiera_avx2.cpp   PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(simd/src/tiera_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(simd/src/tiera_avx2.cpp   PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(simd/src/tiera_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")
  endif()
endif()

# Hardware video decode (vision/video.hpp): NVDEC when nvcuvid.h is available, VideoToolbox on Apple, MediaCodec on Android.
add_library(forge_video STATIC support/src/video.cpp)
target_link_libraries(forge_video PUBLIC forge_inference)
set(FORGE_NVCUVID_INCLUDE "" CACHE PATH "Directory with nvcuvid.h/cuviddec.h (NVIDIA Video Codec SDK) and cuda.h")
find_path(FORGE_NVCUVID_H nvcuvid.h HINTS ${FORGE_NVCUVID_INCLUDE} /usr/local/cuda/include)
if(FORGE_NVCUVID_H)
  target_include_directories(forge_video PRIVATE ${FORGE_NVCUVID_H})
  target_compile_definitions(forge_video PRIVATE FORGE_HAVE_NVCUVID=1)
endif()
if(APPLE)
  target_sources(forge_video PRIVATE support/src/video_vt.mm)
  set_source_files_properties(support/src/video_vt.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_link_libraries(forge_video PUBLIC "-framework VideoToolbox" "-framework CoreMedia" "-framework CoreVideo" "-framework Foundation")
elseif(ANDROID)
  target_link_libraries(forge_video PUBLIC mediandk)
elseif(UNIX)
  target_link_libraries(forge_video PUBLIC ${CMAKE_DL_LIBS})
endif()

if(FORGE_BUILD_APPS)
  add_executable(forge apps/forge.cpp)
  target_link_libraries(forge PRIVATE forge_inference forge_compress forge_verify forge_simd_mv forge_script)
  target_link_libraries(forge PRIVATE forge_security)   # forge security check
  add_executable(forge-serve apps/forge_serve.cpp)
  target_link_libraries(forge-serve PRIVATE forge_inference forge_verify forge_simd_mv forge_loadbalancer)
  if(TARGET forge_rpc_tls)   # --tls-cert / --tls-key / --tls-client-ca; without OpenSSL forge-serve is plain HTTP only
    target_link_libraries(forge-serve PRIVATE forge_rpc_tls)
  endif()
endif()

if(FORGE_BUILD_TESTS)
  enable_testing()
  file(GLOB FORGE_TESTS CONFIGURE_DEPENDS tests/test_*.cpp)
  # Extra directories searched for shells by the shell/emit golden tests (user-local installs, never apt).
  set(_shells_default "")
  if(EXISTS "${PROJECT_SOURCE_DIR}/../opt/shells/bin")
    get_filename_component(_shells_default "${PROJECT_SOURCE_DIR}/../opt/shells/bin" ABSOLUTE)
  endif()
  set(FORGE_TEST_SHELL_PATH "${_shells_default}" CACHE STRING "Extra PATH entries for shell backend tests")
  foreach(t ${FORGE_TESTS})
    get_filename_component(n ${t} NAME_WE)
    if(n STREQUAL "test_simd_raw")   # Tier R: header-only, so it builds (and cross-runs) without the libraries
      add_executable(${n} ${t})
      target_link_libraries(${n} PRIVATE forge_simd)
      if(FORGE_ARCH STREQUAL "x86_64")   # x86 asks simd::detect() (CPUID + XGETBV) before executing AVX2 code
        target_link_libraries(${n} PRIVATE forge_simd_mv)
      endif()
      target_include_directories(${n} PRIVATE tests)
      add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(${n} PROPERTIES SKIP_RETURN_CODE 77)
      # The wrappers of an optional ISA only exist in a translation unit compiled for it; the test checks
      # the CPU before it runs any (and x86 never executes past what CPUID reports).
      set(_raw_flags)
      set(_raw_variant)
      if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
          set(_raw_flags /arch:AVX2)
        endif()
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"))
        set(_raw_variant sve "-march=armv8.2-a+sve")
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
        set(_raw_variant rvv "-march=rv64gcv")
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64")
        set(_raw_variant lsx "-mlsx")
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
        set(_raw_variant zvector "-march=z13;-mzvector")
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips64")
        set(_raw_variant msa "-mmsa")
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64")
        set(_raw_flags -maltivec -mvsx)
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "wasm")
        set(_raw_flags -msimd128)
      elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|amd64")
        set(_raw_flags -mavx2 -mfma -mf16c)
      endif()
      target_compile_options(${n} PRIVATE ${_raw_flags})
      if(_raw_variant)   # second build of the same test for the ISA the default flags leave off
        list(GET _raw_variant 0 _rv_name)
        list(GET _raw_variant 1 _rv_flags)
        include(CheckCXXCompilerFlag)
        string(REPLACE ";" " " _rv_flag_str "${_rv_flags}")
        check_cxx_compiler_flag("${_rv_flag_str}" FORGE_RAW_TEST_${_rv_name})
        if(FORGE_RAW_TEST_${_rv_name})
          add_executable(${n}_${_rv_name} ${t})
          target_link_libraries(${n}_${_rv_name} PRIVATE forge_simd)
          target_include_directories(${n}_${_rv_name} PRIVATE tests)
          target_compile_options(${n}_${_rv_name} PRIVATE ${_rv_flags})
          add_test(NAME ${n}_${_rv_name} COMMAND ${n}_${_rv_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
          set_tests_properties(${n}_${_rv_name} PROPERTIES SKIP_RETURN_CODE 77)
        endif()
      endif()
      continue()
    endif()
    if(n MATCHES "^test_(text|locale)")   # only needs forge_text
      add_executable(${n} ${t})
      target_link_libraries(${n} PRIVATE forge_text)
      add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(${n} PROPERTIES SKIP_RETURN_CODE 77)
      continue()
    endif()
    if(n STREQUAL "test_os_dynlib")   # os::dynlib: a fixture library to load, plus foreign-format copies to parse
      add_library(forge_test_fixture SHARED tests/dynlib_fixture.c)
      set_target_properties(forge_test_fixture PROPERTIES OUTPUT_NAME forge_fixture PREFIX "lib" C_VISIBILITY_PRESET hidden
                            LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dynlib_fixture RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dynlib_fixture)
      if(WIN32)
        set_target_properties(forge_test_fixture PROPERTIES PREFIX "")   # os::library::decorate("forge_fixture") == forge_fixture.dll
      endif()
      add_executable(${n} ${t})
      target_link_libraries(${n} PRIVATE forge_os)
      target_include_directories(${n} PRIVATE tests)
      add_dependencies(${n} forge_test_fixture)
      target_compile_definitions(${n} PRIVATE FORGE_TEST_FIXTURE_DIR="${PROJECT_BINARY_DIR}/dynlib_fixture")
      find_program(FORGE_FIXTURE_CLANG NAMES clang clang-23)
      if(FORGE_FIXTURE_CLANG AND NOT CMAKE_CROSSCOMPILING)
        find_program(FORGE_FIXTURE_LLD_LINK NAMES lld-link lld-link-23)
        find_program(FORGE_FIXTURE_LD64 NAMES ld64.lld ld64.lld-23)
        find_program(FORGE_FIXTURE_LIPO NAMES llvm-lipo llvm-lipo-23)
        find_program(FORGE_FIXTURE_OBJCOPY NAMES llvm-objcopy llvm-objcopy-23)
        set(_xf ${PROJECT_BINARY_DIR}/dynlib_fixture/xfmt)
        add_custom_command(OUTPUT ${_xf}/stamp
          COMMAND ${CMAKE_COMMAND} -DCLANG=${FORGE_FIXTURE_CLANG} -DLLD_LINK=${FORGE_FIXTURE_LLD_LINK} -DLD64=${FORGE_FIXTURE_LD64}
                  -DLIPO=${FORGE_FIXTURE_LIPO} -DOBJCOPY=${FORGE_FIXTURE_OBJCOPY} -DSRC=${PROJECT_SOURCE_DIR}/tests/dynlib_fixture.c
                  -DOUT=${_xf} -P ${PROJECT_SOURCE_DIR}/cmake/dynlib_fixtures.cmake
          DEPENDS ${PROJECT_SOURCE_DIR}/tests/dynlib_fixture.c ${PROJECT_SOURCE_DIR}/cmake/dynlib_fixtures.cmake
          COMMENT "Cross-linking os::dynlib parser fixtures (PE, Mach-O, ELF variants)" VERBATIM)
        add_custom_target(forge_test_fixture_xfmt DEPENDS ${_xf}/stamp)
        add_dependencies(${n} forge_test_fixture_xfmt)
        target_compile_definitions(${n} PRIVATE FORGE_TEST_XFMT_DIR="${_xf}")
      endif()
      add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      continue()
    endif()
    if(n STREQUAL "test_simd_std")   # interop.hpp: also built against libc++ where that is the library with <mdspan>
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CROSSCOMPILING AND NOT MSVC AND NOT APPLE)
        set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++")
        check_cxx_source_compiles("#include <mdspan>\n#include <vector>\n#ifndef __cpp_lib_mdspan\n#error no mdspan\n#endif\nint main(){std::vector<int> v(4);std::mdspan<int,std::dextents<unsigned long,1>> m(v.data(),4);return m[0];}" FORGE_LIBCXX_MDSPAN)
        unset(CMAKE_REQUIRED_FLAGS)
        if(FORGE_LIBCXX_MDSPAN)
          # Everything in this executable is compiled against libc++: the scalar Tier A table from source and
          # the test's own tier_a_active() (the kernel library is built against the default standard library).
          add_executable(${n}_libcxx ${t} simd/src/tiera_scalar.cpp)
          target_compile_options(${n}_libcxx PRIVATE -stdlib=libc++)
          target_compile_definitions(${n}_libcxx PRIVATE FORGE_TEST_OWN_TIERA_DISPATCH=1)
          target_link_options(${n}_libcxx PRIVATE -stdlib=libc++)
          target_link_libraries(${n}_libcxx PRIVATE forge_simd)
          target_include_directories(${n}_libcxx PRIVATE tests)
          add_test(NAME ${n}_libcxx COMMAND ${n}_libcxx WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
          set_tests_properties(${n}_libcxx PROPERTIES PASS_REGULAR_EXPRESSION "mdspan overloads: yes")
        endif()
      endif()
    endif()
    if(n STREQUAL "test_simd_tables" OR n STREQUAL "test_simd_types" OR n STREQUAL "test_simd_std")   # simd only: cross-builds without the rest of the tree
      add_executable(${n} ${t})
      target_link_libraries(${n} PRIVATE forge_simd_mv)
      target_include_directories(${n} PRIVATE tests)
      add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      continue()
    endif()
    if(n STREQUAL "test_mobile")   # only needs the device layer
      add_executable(${n} ${t})
      target_link_libraries(${n} PRIVATE forge_mobile)
      add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      continue()
    endif()
    add_executable(${n} ${t})
    target_link_libraries(${n} PRIVATE forge_inference forge_compress forge_verify forge_simd_mv forge_loadbalancer)
    if(n STREQUAL "test_video")
      target_link_libraries(${n} PRIVATE forge_video)
    endif()
    if(n STREQUAL "test_rpc" AND TARGET forge_rpc_tls)
      target_link_libraries(${n} PRIVATE forge_rpc_tls)
    endif()
    if(n MATCHES "^test_script" OR n STREQUAL "test_signing_detached")   # test_signing_detached: scripts + the example plugin
      target_link_libraries(${n} PRIVATE forge_script)
      if(TARGET forge_script_example_plugin)
        add_dependencies(${n} forge_script_example_plugin)
        target_compile_definitions(${n} PRIVATE FORGE_TEST_PLUGIN="$<TARGET_FILE:forge_script_example_plugin>")
      endif()
    endif()
    if(n STREQUAL "test_script_aot")   # forge script --aot (docs/aot.md): needs the runtime library and the runner stub
      add_dependencies(${n} forge_script_rt forge-script-run)
      target_include_directories(${n} PRIVATE tests)
      target_compile_definitions(${n} PRIVATE FORGE_AOT_TEST_PCH_DIR="${PROJECT_BINARY_DIR}/aot-test-pch")
      add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(${n} PROPERTIES TIMEOUT 1500)
      # Every corpus script through every host compiler (clang++-23, g++): minutes, so not in the default run.
      add_test(NAME ${n}_native_sweep COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(${n}_native_sweep PROPERTIES ENVIRONMENT "FORGE_AOT_SWEEP=full" LABELS slow SKIP_RETURN_CODE 77 TIMEOUT 3600)
      continue()
    endif()
    if(n STREQUAL "test_script_fsc")   # the .fsc encoder/decoder is internal (script/src/fsc.hpp)
      target_include_directories(${n} PRIVATE script/src)
    endif()
    target_include_directories(${n} PRIVATE tests)
    add_test(NAME ${n} COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    if(n MATCHES "^test_script" AND FORGE_TEST_SHELL_PATH)
      set_tests_properties(${n} PROPERTIES ENVIRONMENT "FORGE_TEST_SHELL_PATH=${FORGE_TEST_SHELL_PATH}")
    endif()
    if(n STREQUAL "test_rpc")   # the TLS / mTLS / channel-binding cases alone; exits 77 without OpenSSL
      add_test(NAME test_rpc_tls COMMAND ${n} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(test_rpc_tls PROPERTIES SKIP_RETURN_CODE 77 ENVIRONMENT "FORGE_TEST_ONLY=rpc_tls")
    endif()
  endforeach()
  if(TARGET test_security)   # forge::security (re-runs itself for the lockdown child cases)
    target_link_libraries(test_security PRIVATE forge_security)
    set_tests_properties(test_security PROPERTIES LABELS "security")
  endif()
  if(TARGET test_signing_detached)   # FSS INT-2 / INT-3: detached signatures, plugin and script integrity
    set_tests_properties(test_signing_detached PROPERTIES LABELS "security")
  endif()
  set_tests_properties(test_log PROPERTIES SKIP_RETURN_CODE 77)   # skips when built with FORGE_LOG_MIN_LEVEL
  add_test(NAME log_min_level_cmake COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/tests/log_min_level.cmake)
  # Tier R: every generated wrapper of every architecture, compiled against the real target headers with
  # clang's cross targets (no foreign sysroot needed). Skipped (77) when that clang is not installed.
  set(FORGE_RAW_CHECK_CLANG "clang++-23" CACHE STRING "clang++ (23 or newer, all targets) for the Tier R cross syntax check")
  # The engine on the work-stealing scheduler (FORGE_TASK_SCHEDULER=sched, task/engine.hpp): the executor contract
  # on both executors (test_engine itself runs with the default, the pool), then the engine's own tests again
  # under `sched`: their determinism checks (batch vs single row, prefill vs decode, golden logits) must hold there too.
  if(TARGET test_engine)
    add_test(NAME engine_sched COMMAND test_engine WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(engine_sched PROPERTIES ENVIRONMENT "FORGE_TASK_SCHEDULER=sched" PASS_REGULAR_EXPRESSION "engine: sched" FAIL_REGULAR_EXPRESSION "FAIL")
    add_test(NAME engine_pool COMMAND test_engine WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(engine_pool PROPERTIES ENVIRONMENT "FORGE_TASK_SCHEDULER=pool" PASS_REGULAR_EXPRESSION "engine: pool" FAIL_REGULAR_EXPRESSION "FAIL")
    foreach(_t test_graph test_inference test_kvquant test_sessions test_aot)
      if(TARGET ${_t})
        add_test(NAME sched_${_t} COMMAND ${_t} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(sched_${_t} PROPERTIES ENVIRONMENT "FORGE_TASK_SCHEDULER=sched" LABELS sched)
      endif()
    endforeach()
  endif()
  # vec_types.hpp on every Tier P backend: one executable per kernel table, tests/simd_types_isa.cpp compiled with
  # that table's flags and a baseline main that returns 77 (skipped) when the CPU cannot run the table. Not for
  # MSVC, where every target uses the portable arrays that test_simd_types already covers.
  function(forge_types_test name)   # ARGN: the table's compile options ("none" for the baseline)
    add_library(forge_types_obj_${name} OBJECT tests/simd_types_isa.cpp)
    if(NOT "${ARGN}" STREQUAL "none")
      target_compile_options(forge_types_obj_${name} PRIVATE ${ARGN})
    endif()
    target_link_libraries(forge_types_obj_${name} PRIVATE forge_simd)
    target_include_directories(forge_types_obj_${name} PRIVATE tests)
    add_executable(test_simd_types_${name} tests/simd_types_main.cpp $<TARGET_OBJECTS:forge_types_obj_${name}>)
    target_link_libraries(test_simd_types_${name} PRIVATE forge_simd_mv)
    add_test(NAME types_${name} COMMAND test_simd_types_${name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    set_tests_properties(types_${name} PROPERTIES SKIP_RETURN_CODE 77)
  endfunction()
  if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND EXISTS "${PROJECT_SOURCE_DIR}/tests/simd_types_isa.cpp")
    foreach(_t sse2 sse42 lsx lasx vsx vxe wasm128 msa neon32)
      string(TOUPPER "${_t}" _up)
      if(TARGET forge_simd_${_t})
        forge_types_test(${_t} ${FORGE_SIMD_${_up}_FLAGS})
      endif()
    endforeach()
    if(FORGE_ARCH STREQUAL "x86_64")
      forge_types_test(avx2 -mavx2 -mfma -mf16c -DFORGE_TYPES_AVX2=1)
      forge_types_test(avx512 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -DFORGE_TYPES_AVX512=1)
    elseif(FORGE_ARCH STREQUAL "aarch64")
      forge_types_test(neon none)
    endif()
    foreach(b ${FORGE_SCALABLE_BITS})
      if(TARGET forge_simd_sve${b})
        forge_types_test(sve${b} ${FORGE_SVE${b}_FLAGS})
      endif()
      if(TARGET forge_simd_rvv${b})
        forge_types_test(rvv${b} ${FORGE_RVV${b}_FLAGS})
      endif()
    endforeach()
  endif()
  # Every simd test executable, for builds that want just those (the emulated CI rows, cross builds).
  set(_simd_tests)
  foreach(_t test_simd test_simd_tables test_simd_tiera test_simd_types test_simd_std)
    if(TARGET ${_t})
      list(APPEND _simd_tests ${_t})
    endif()
  endforeach()
  foreach(_t sse2 sse42 lsx lasx vsx vxe wasm128 msa neon32 avx2 avx512 neon sve128 sve256 sve512 sve1024 rvv128 rvv256 rvv512 rvv1024)
    if(TARGET test_simd_types_${_t})
      list(APPEND _simd_tests test_simd_types_${_t})
    endif()
  endforeach()
  add_custom_target(forge_simd_tests DEPENDS ${_simd_tests})
  add_test(NAME raw_intrinsics_cross
           COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/check_raw_intrinsics.py
                   --clang ${FORGE_RAW_CHECK_CLANG} --out ${PROJECT_BINARY_DIR}/raw_check)
  set_tests_properties(raw_intrinsics_cross PROPERTIES SKIP_RETURN_CODE 77 LABELS "raw_cross" TIMEOUT 3600)
  # Scalable vectors at many lengths under qemu: the sve/rvv lane must be active at every VL, including the
  # ones without a power-of-two table (lowered at load time) and ones above 1024 bits.
  if(CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
    list(GET CMAKE_CROSSCOMPILING_EMULATOR 0 _emu)
    set(_vls)
    if(_emu MATCHES "qemu-aarch64" AND FORGE_SVE256_FLAGS)
      set(_lane sve)
      foreach(b 128 256 384 512 640 1024 1536 2048)
        math(EXPR _bytes "${b} / 8")
        list(APPEND _vls "${b}|max,sve${b}=on,sve-default-vector-length=${_bytes}")
      endforeach()
    elseif(_emu MATCHES "qemu-riscv64" AND FORGE_RVV256_FLAGS)
      set(_lane rvv)
      foreach(b 128 256 512 1024)
        list(APPEND _vls "${b}|rv64,v=true,vlen=${b}")
      endforeach()
    endif()
    foreach(v ${_vls})
      string(REPLACE "|" ";" v "${v}")
      list(GET v 0 b)
      list(GET v 1 cpu)
      add_test(NAME lane_${_lane}_vl${b} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(lane_${_lane}_vl${b} PROPERTIES PASS_REGULAR_EXPRESSION "active: ${_lane}" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      # the Tier A, seq and int8 tables of the same lane (isa_extra.cpp): bit identity with scalar at this length
      add_test(NAME tables_${_lane}_vl${b} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd_tables> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tables_${_lane}_vl${b} PROPERTIES PASS_REGULAR_EXPRESSION "seq tables: ${_lane}" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      add_test(NAME tiera_${_lane}_vl${b} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd_tiera> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tiera_${_lane}_vl${b} PROPERTIES PASS_REGULAR_EXPRESSION "active: ${_lane}" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      if(TARGET test_simd_types_${_lane}${b})   # vec_types.hpp on the table built for exactly this length
        add_test(NAME types_${_lane}_vl${b} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd_types_${_lane}${b}> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(types_${_lane}_vl${b} PROPERTIES PASS_REGULAR_EXPRESSION "0 failure" FAIL_REGULAR_EXPRESSION "CHECK failed" LABELS emu)
      endif()
    endforeach()
    if(_emu MATCHES "qemu-aarch64")   # NEON alone (an ARMv8.0 core, then one with dotprod and no SVE)
      foreach(v "a57|cortex-a57" "nosve|max,sve=off")
        string(REPLACE "|" ";" v "${v}")
        list(GET v 0 n)
        list(GET v 1 cpu)
        add_test(NAME tables_neon_${n} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd_tables> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(tables_neon_${n} PROPERTIES PASS_REGULAR_EXPRESSION "seq tables: neon" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
        add_test(NAME tiera_neon_${n} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd_tiera> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(tiera_neon_${n} PROPERTIES PASS_REGULAR_EXPRESSION "active: neon" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
        if(TARGET test_simd_types_neon)
          add_test(NAME types_neon_${n} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_simd_types_neon> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
          set_tests_properties(types_neon_${n} PROPERTIES PASS_REGULAR_EXPRESSION "0 failure" FAIL_REGULAR_EXPRESSION "CHECK failed" LABELS emu)
        endif()
      endforeach()
    endif()
    # vec_gn.hpp tables under qemu: test_simd again with each table forced (it must become the active one).
    set(_gn)
    if(_emu MATCHES "qemu-loongarch64")
      set(_gn lasx lsx)
    elseif(_emu MATCHES "qemu-ppc64le")
      set(_gn vsx)
    elseif(_emu MATCHES "qemu-s390x")
      set(_gn vxe)
    elseif(_emu MATCHES "qemu-mips64")
      set(_gn msa)
    elseif(_emu MATCHES "qemu-arm$")
      set(_gn neon)
    endif()
    foreach(t ${_gn})
      add_test(NAME lane_${t} COMMAND test_simd WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(lane_${t} PROPERTIES ENVIRONMENT "FORGE_SIMD_TARGET=${t}" PASS_REGULAR_EXPRESSION "active: ${t}" FAIL_REGULAR_EXPRESSION "FAIL")
      add_test(NAME tables_${t} COMMAND test_simd_tables WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tables_${t} PROPERTIES PASS_REGULAR_EXPRESSION "qi8 tables:[a-z0-9. ]* ${t} " FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      add_test(NAME tiera_${t} COMMAND test_simd_tiera WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tiera_${t} PROPERTIES ENVIRONMENT "FORGE_SIMD_TARGET=${t}" PASS_REGULAR_EXPRESSION "active: ${t}" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
    endforeach()
    if(_emu MATCHES "qemu-arm$")   # 32-bit Arm: the same four on a second core model (the toolchain file's is a Cortex-A7)
      add_test(NAME lane_neon_a15 COMMAND ${_emu} -cpu cortex-a15 $<TARGET_FILE:test_simd> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(lane_neon_a15 PROPERTIES PASS_REGULAR_EXPRESSION "active: neon" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      add_test(NAME tables_neon_a15 COMMAND ${_emu} -cpu cortex-a15 $<TARGET_FILE:test_simd_tables> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tables_neon_a15 PROPERTIES PASS_REGULAR_EXPRESSION "qi8 tables:[a-z0-9. ]* neon " FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      add_test(NAME tiera_neon_a15 COMMAND ${_emu} -cpu cortex-a15 $<TARGET_FILE:test_simd_tiera> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tiera_neon_a15 PROPERTIES PASS_REGULAR_EXPRESSION "active: neon" FAIL_REGULAR_EXPRESSION "FAIL" LABELS emu)
      if(TARGET test_simd_types_neon32)
        add_test(NAME types_neon32_a15 COMMAND ${_emu} -cpu cortex-a15 $<TARGET_FILE:test_simd_types_neon32> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(types_neon32_a15 PROPERTIES PASS_REGULAR_EXPRESSION "0 failure" FAIL_REGULAR_EXPRESSION "CHECK failed" LABELS emu)
      endif()
    endif()
    # Integer matvec tables (simd/qdot_i8.hpp) against the scalar reference: an ARMv8.0 core (neon only), then
    # dotprod and the vector-length agnostic SVE table at several VLs (FORGE_SVE_KEEP_VL: 384 and 2048 stay as they are).
    if(_emu MATCHES "qemu-aarch64" AND TARGET test_graph)
      set(_qi8_cpus "a57|cortex-a57|scalar neon\n" "nosve|max,sve=off|scalar neon dotprod\n")
      if(FORGE_QI8_SVE_FLAGS)
        foreach(b 128 256 384 512 2048)
          math(EXPR _bytes "${b} / 8")
          list(APPEND _qi8_cpus "sve${b}|max,sve${b}=on,sve-default-vector-length=${_bytes}|scalar neon dotprod sve\n")
        endforeach()
      endif()
      foreach(v ${_qi8_cpus})
        string(REPLACE "|" ";" v "${v}")
        list(GET v 0 n)
        list(GET v 1 cpu)
        list(GET v 2 want)
        add_test(NAME qi8_aarch64_${n} COMMAND ${_emu} -cpu ${cpu} $<TARGET_FILE:test_graph> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(qi8_aarch64_${n} PROPERTIES ENVIRONMENT "FORGE_TEST_ONLY=qdot_i8;FORGE_SVE_KEEP_VL=1" LABELS emu
          PASS_REGULAR_EXPRESSION "tables run: ${want}" FAIL_REGULAR_EXPRESSION "FAIL")
      endforeach()
    endif()
    if(_emu MATCHES "qemu-aarch64" AND FORGE_CAN_SME)   # SME outer-product tiles at several streaming VLs
      foreach(b 128 256 512 2048)
        add_test(NAME tile_sme_svl${b} COMMAND ${_emu} -cpu max,sme=on,sme${b}=on $<TARGET_FILE:test_tile> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(tile_sme_svl${b} PROPERTIES PASS_REGULAR_EXPRESSION "SME SVL" FAIL_REGULAR_EXPRESSION "FAIL|failed" LABELS emu)
      endforeach()
    endif()
  endif()
  # x86-64 SSE tier: test_simd with the SSE2 table forced (always usable on x86-64; test_simd itself also covers the
  # SSE4.2 table whenever the CPU has it). MSVC builds have no CPUID detection, so only `scalar` is usable there.
  if(FORGE_ARCH STREQUAL "x86_64" AND NOT CMAKE_CROSSCOMPILING AND NOT MSVC)
    foreach(t sse2)
      add_test(NAME lane_${t} COMMAND test_simd WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(lane_${t} PROPERTIES ENVIRONMENT "FORGE_SIMD_TARGET=${t}" PASS_REGULAR_EXPRESSION "active: ${t}" FAIL_REGULAR_EXPRESSION "FAIL")
    endforeach()
    # Tier A with each lower x86 table forced as the active one (test_simd_tiera / test_simd_tables already run
    # every usable table against scalar; these also cover the dispatch path).
    foreach(t sse2 sse4.2 avx2)
      add_test(NAME tiera_${t} COMMAND test_simd_tiera WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tiera_${t} PROPERTIES ENVIRONMENT "FORGE_SIMD_TARGET=${t}" PASS_REGULAR_EXPRESSION "0 failure" FAIL_REGULAR_EXPRESSION "FAIL")
    endforeach()
  endif()
  # forge::os / forge::loadbalancer cross-OS compile+link checks (zig ships FreeBSD, macOS and MinGW libcs).
  find_program(FORGE_ZIG NAMES zig HINTS $ENV{ZIG_PATH})
  if(NOT FORGE_ZIG)   # the ziglang wheel (pip install ziglang) carries the binary inside the package
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_EXECUTABLE)
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import ziglang, os; print(os.path.join(os.path.dirname(ziglang.__file__), 'zig'))"
                      OUTPUT_VARIABLE _zig OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _zr)
      if(_zr EQUAL 0 AND EXISTS "${_zig}")
        set(FORGE_ZIG ${_zig} CACHE FILEPATH "zig (cross-OS checks)" FORCE)
      endif()
    endif()
  endif()
  if(FORGE_ZIG)
    # Every os test (tests/test_os*.cpp) and the load balancer test, compiled and linked per target. The C source
    # (Monocypher) is compiled separately: cmake/xos_link.cmake.
    file(GLOB _xos_tests ${PROJECT_SOURCE_DIR}/tests/test_os*.cpp)
    list(APPEND _xos_tests ${PROJECT_SOURCE_DIR}/tests/test_loadbalancer.cpp)
    foreach(tg x86_64-freebsd aarch64-macos x86_64-macos x86_64-windows-gnu aarch64-linux-musl)
      set(_lib "")
      if(tg MATCHES windows)
        set(_lib "-lws2_32|-lbcrypt|-ladvapi32")
      endif()
      foreach(t ${_xos_tests})
        get_filename_component(src ${t} NAME_WE)
        add_test(NAME xos_${src}_${tg} COMMAND ${CMAKE_COMMAND} -DZIG=${FORGE_ZIG} -DTARGET=${tg} -DSRC=${t}
                 -DCSRCS=${PROJECT_SOURCE_DIR}/third_party/monocypher/monocypher.c -DOUT=${PROJECT_BINARY_DIR}/xos/${src}.${tg}
                 "-DINCS=-I${PROJECT_SOURCE_DIR}/os/include|-I${PROJECT_SOURCE_DIR}/loadbalancer/include|-I${PROJECT_SOURCE_DIR}/tests|-I${PROJECT_SOURCE_DIR}/third_party/monocypher"
                 "-DLIBS=${_lib}" -P ${PROJECT_SOURCE_DIR}/cmake/xos_link.cmake)
        set_tests_properties(xos_${src}_${tg} PROPERTIES LABELS "cross;slow")
      endforeach()
    endforeach()
    # Every os header on its own (one translation unit each), so no header leans on what another one included.
    file(GLOB _xos_hdrs RELATIVE ${PROJECT_SOURCE_DIR}/os/include ${PROJECT_SOURCE_DIR}/os/include/forge/os/*.hpp)
    set(_xos_hdr_srcs "")
    foreach(h ${_xos_hdrs})
      get_filename_component(hn ${h} NAME_WE)
      set(_hsrc ${PROJECT_BINARY_DIR}/xos/hdr/${hn}.cpp)
      if(NOT EXISTS ${_hsrc})
        file(WRITE ${_hsrc} "#include <${h}>\n")
      endif()
      list(APPEND _xos_hdr_srcs ${_hsrc})
    endforeach()
    string(REPLACE ";" "|" _xos_hdr_list "${_xos_hdr_srcs}")
    foreach(tg x86_64-freebsd aarch64-macos x86_64-windows-gnu)
      add_test(NAME xos_headers_${tg} COMMAND ${CMAKE_COMMAND} -DZIG=${FORGE_ZIG} -DTARGET=${tg} -DOUT=${PROJECT_BINARY_DIR}/xos/hdr_${tg}
               "-DINCS=-I${PROJECT_SOURCE_DIR}/os/include|-I${PROJECT_SOURCE_DIR}/core/include|-I${PROJECT_SOURCE_DIR}/support/include|-I${PROJECT_SOURCE_DIR}/third_party/monocypher"
               "-DSRCS=${_xos_hdr_list}" -P ${PROJECT_SOURCE_DIR}/cmake/xos_compile_each.cmake)
      set_tests_properties(xos_headers_${tg} PROPERTIES LABELS "cross;slow" TIMEOUT 7200)
    endforeach()
    # The script emitter + shell backends (host_shell() has per-OS branches): syntax-checked for every target.
    file(GLOB _emit_srcs ${PROJECT_SOURCE_DIR}/script/src/emit.cpp ${PROJECT_SOURCE_DIR}/script/src/shells/*.cpp)
    foreach(tg x86_64-freebsd aarch64-macos x86_64-windows-gnu aarch64-linux-musl)
      string(REPLACE ";" "|" _emit_list "${_emit_srcs}")
      add_test(NAME xos_emit_${tg} COMMAND ${CMAKE_COMMAND} -DZIG=${FORGE_ZIG} -DTARGET=${tg} -DOUT=${PROJECT_BINARY_DIR}/xos/emit_${tg}
               "-DINCS=-I${PROJECT_SOURCE_DIR}/script/include|-I${PROJECT_SOURCE_DIR}/script/src|-I${PROJECT_SOURCE_DIR}/os/include|-I${PROJECT_BINARY_DIR}"
               "-DSRCS=${_emit_list}" -P ${PROJECT_SOURCE_DIR}/cmake/xos_compile_each.cmake)
      set_tests_properties(xos_emit_${tg} PROPERTIES LABELS "cross;slow")
    endforeach()
    # forge::os::mobile backends (every non-Apple-ObjC source; apple.mm needs an Apple SDK -> CI on macOS).
    set(_mob_list "${PROJECT_SOURCE_DIR}/os/src/mobile/mobile.cpp|${PROJECT_SOURCE_DIR}/os/src/mobile/services.cpp|${PROJECT_SOURCE_DIR}/os/src/mobile/linux.cpp|${PROJECT_SOURCE_DIR}/os/src/mobile/windows.cpp|${PROJECT_SOURCE_DIR}/os/src/mobile/freebsd.cpp|${PROJECT_SOURCE_DIR}/os/src/mobile/android.cpp")
    foreach(tg x86_64-freebsd aarch64-macos x86_64-windows-gnu aarch64-linux-musl)
      add_test(NAME xos_mobile_${tg} COMMAND ${CMAKE_COMMAND} -DZIG=${FORGE_ZIG} -DTARGET=${tg} -DOUT=${PROJECT_BINARY_DIR}/xos/mobile_${tg}
               "-DINCS=-I${PROJECT_SOURCE_DIR}/os/include|-I${PROJECT_SOURCE_DIR}/core/include|-I${PROJECT_SOURCE_DIR}/support/include|-I${PROJECT_SOURCE_DIR}/third_party/monocypher"
               "-DSRCS=${_mob_list}" -P ${PROJECT_SOURCE_DIR}/cmake/xos_compile_each.cmake)
      set_tests_properties(xos_mobile_${tg} PROPERTIES LABELS "cross;slow")
    endforeach()
    # forge::os::text / forge::os::locale (every C++ source; locale_apple.mm needs an Apple SDK -> CI on macOS).
    file(GLOB _text_srcs ${PROJECT_SOURCE_DIR}/os/src/text/*.cpp)
    string(REPLACE ";" "|" _text_list "${_text_srcs}")
    foreach(tg x86_64-freebsd aarch64-macos x86_64-windows-gnu aarch64-linux-musl)
      add_test(NAME xos_text_${tg} COMMAND ${CMAKE_COMMAND} -DZIG=${FORGE_ZIG} -DTARGET=${tg} -DOUT=${PROJECT_BINARY_DIR}/xos/text_${tg}
               "-DINCS=-I${PROJECT_SOURCE_DIR}/os/include|-I${PROJECT_SOURCE_DIR}/core/include|-I${PROJECT_SOURCE_DIR}/support/include|-I${PROJECT_SOURCE_DIR}/third_party/monocypher"
               "-DSRCS=${_text_list}" -P ${PROJECT_SOURCE_DIR}/cmake/xos_compile_each.cmake)
      set_tests_properties(xos_text_${tg} PROPERTIES LABELS "cross;slow")
    endforeach()
    file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/xos)
  endif()
  # Intel SDE: AMX tiles on an emulated Sapphire Rapids / Granite Rapids, and the skip path on Ice Lake.
  find_program(FORGE_SDE NAMES sde64 HINTS $ENV{SDE_PATH})
  if(FORGE_SDE AND FORGE_CAN_AMX)
    foreach(c spr gnr icl)
      add_test(NAME tile_amx_sde_${c} COMMAND ${FORGE_SDE} -${c} -- $<TARGET_FILE:test_tile> WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
      set_tests_properties(tile_amx_sde_${c} PROPERTIES LABELS emu FAIL_REGULAR_EXPRESSION "FAIL")
    endforeach()
  endif()
  set_tests_properties(test_vulkan PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  add_test(NAME test_vulkan_align256 COMMAND test_vulkan WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(test_vulkan_align256 PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu ENVIRONMENT FORGE_VK_SSBO_ALIGN=256)
  set_tests_properties(test_cuda_hip PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  add_test(NAME test_gpu_cuda COMMAND test_cuda_hip cuda)
  add_test(NAME test_gpu_hip COMMAND test_cuda_hip hip)
  add_test(NAME test_gpu_features_cuda COMMAND test_cuda_hip features cuda)
  add_test(NAME test_gpu_features_hip COMMAND test_cuda_hip features hip)
  add_test(NAME test_gpu_metal COMMAND test_cuda_hip metal)
  set_tests_properties(test_gpu_metal PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  foreach(m opencl ze)   # real runtimes: skip (77) without a driver/device
    add_test(NAME test_gpu_${m} COMMAND test_cuda_hip ${m})
    add_test(NAME test_gpu_features_${m} COMMAND test_cuda_hip features ${m})
    set_tests_properties(test_gpu_${m} test_gpu_features_${m} PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  endforeach()
  set_tests_properties(test_gpu_features_cuda test_gpu_features_hip PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  set_tests_properties(test_gpu_cuda test_gpu_hip PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  if(NOT CMAKE_CROSSCOMPILING)
    # The CUDA host runtime against a CPU mock of libcuda.so.1 (argument packing, memcpy, launch sizing).
    add_library(mock_cuda SHARED tests/mock_cuda.c)
    set_target_properties(mock_cuda PROPERTIES OUTPUT_NAME cuda SOVERSION 1 LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/mockcuda)
    target_link_libraries(mock_cuda PRIVATE $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>)
    target_include_directories(mock_cuda PRIVATE ${FORGE_GPU_DIR})   # kernels_nn_mock.inc (generated)
    add_dependencies(mock_cuda forge_gpu_gen)
    add_test(NAME test_gpu_cuda_mock COMMAND test_cuda_hip cuda)
    set_tests_properties(test_gpu_cuda_mock PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockcuda" LABELS gpu SKIP_RETURN_CODE 77)
    add_dependencies(test_cuda_hip mock_cuda)
    add_library(mock_hip SHARED tests/mock_cuda.c)
    target_compile_definitions(mock_hip PRIVATE MOCK_HIP)
    set_target_properties(mock_hip PROPERTIES OUTPUT_NAME amdhip64 SOVERSION 6 LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/mockhip)
    target_link_libraries(mock_hip PRIVATE $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>)
    target_include_directories(mock_hip PRIVATE ${FORGE_GPU_DIR})
    add_dependencies(mock_hip forge_gpu_gen)
    add_dependencies(test_cuda_hip mock_hip)
    foreach(m opencl ze)
      if(m STREQUAL "opencl")
        add_library(mock_${m} SHARED tests/mock_opencl.c)
        set_target_properties(mock_${m} PROPERTIES OUTPUT_NAME OpenCL SOVERSION 1)
      else()
        add_library(mock_${m} SHARED tests/mock_ze.c)
        set_target_properties(mock_${m} PROPERTIES OUTPUT_NAME ze_loader SOVERSION 1)
      endif()
      set_target_properties(mock_${m} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/mock${m})
      target_include_directories(mock_${m} PRIVATE ${FORGE_GPU_DIR} ${PROJECT_SOURCE_DIR}/third_party)
      target_link_libraries(mock_${m} PRIVATE $<$<NOT:$<C_COMPILER_ID:MSVC>>:m>)
      add_dependencies(mock_${m} forge_gpu_gen)
      add_dependencies(test_cuda_hip mock_${m})
      add_test(NAME test_gpu_${m}_mock COMMAND test_cuda_hip ${m})
      add_test(NAME test_gpu_features_${m}_mock COMMAND test_cuda_hip features ${m})
      set_tests_properties(test_gpu_${m}_mock test_gpu_features_${m}_mock PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
        ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m};MOCK_DEVICES=2;FORGE_ZE_TEST_IMAGE=opencl-c")
    endforeach()
    add_test(NAME test_gpu_hip_mock COMMAND test_cuda_hip hip)
    set_tests_properties(test_gpu_hip_mock PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockhip" LABELS gpu SKIP_RETURN_CODE 77)
    add_test(NAME test_gpu_features_cuda_mock COMMAND test_cuda_hip features cuda)
    set_tests_properties(test_gpu_features_cuda_mock PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockcuda;MOCK_DEVICES=2" LABELS gpu SKIP_RETURN_CODE 77)
    add_test(NAME test_gpu_features_hip_mock COMMAND test_cuda_hip features hip)
    set_tests_properties(test_gpu_features_hip_mock PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockhip;MOCK_DEVICES=2" LABELS gpu SKIP_RETURN_CODE 77)
    # Device memory pools: allocation accounting, LRU eviction under a small budget, driver out-of-memory retry.
    foreach(m cuda hip opencl ze)
      if(NOT m STREQUAL "cuda" AND NOT m STREQUAL "hip")   # kernel program cache (gpu/kernel_cache.hpp)
        add_test(NAME test_gpu_kernel_cache_${m}_mock COMMAND test_cuda_hip cache ${m})
        set_tests_properties(test_gpu_kernel_cache_${m}_mock PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
          ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m};MOCK_DEVICES=1;FORGE_ZE_TEST_IMAGE=opencl-c")
      endif()
      add_test(NAME test_gpu_stale_${m}_mock COMMAND test_cuda_hip stale ${m})   # dead plans leave no device copies / pins
      set_tests_properties(test_gpu_stale_${m}_mock PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
        ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m};FORGE_ZE_TEST_IMAGE=opencl-c")
      add_test(NAME test_gpu_select_${m}_mock COMMAND test_cuda_hip select ${m})   # FORGE_DEVICE=auto (gpu/select.hpp)
      set_tests_properties(test_gpu_select_${m}_mock PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
        ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m};MOCK_DEVICES=2;FORGE_ZE_TEST_IMAGE=opencl-c")
      add_test(NAME test_gpu_determinism_${m}_mock COMMAND test_cuda_hip determinism ${m})   # gpu/determinism.hpp
      set_tests_properties(test_gpu_determinism_${m}_mock PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
        ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m};FORGE_ZE_TEST_IMAGE=opencl-c")
      add_test(NAME test_gpu_memory_${m}_mock COMMAND test_cuda_hip memory ${m})
      set_tests_properties(test_gpu_memory_${m}_mock PROPERTIES LABELS gpu SKIP_RETURN_CODE 77
        ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m};FORGE_ZE_TEST_IMAGE=opencl-c")
    endforeach()
    if(FORGE_BUILD_APPS)
      add_test(NAME golden_tiny_llama_cuda_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_cumock)
      set_tests_properties(golden_tiny_llama_cuda_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=cuda;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockcuda")
      # "pipe:auto" over two mock CUDA devices (gpu::plan_pipeline): the layers split over cuda:0 and cuda:1
      add_test(NAME golden_tiny_llama_pipe_auto_cuda_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_pipeauto_cumock)
      set_tests_properties(golden_tiny_llama_pipe_auto_cuda_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=pipe:auto;MOCK_DEVICES=2;FORGE_DEVICE_ALLOW_SOFTWARE=1;FORGE_DEVICE_BENCH=0;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockcuda")
      add_test(NAME golden_tiny_llama_opencl_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_clmock)
      set_tests_properties(golden_tiny_llama_opencl_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=opencl;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockopencl")
      add_test(NAME golden_tiny_llama_ze_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_zemock)
      set_tests_properties(golden_tiny_llama_ze_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=level_zero;FORGE_ZE_TEST_IMAGE=opencl-c;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockze")
      # MoE / MLA+MoE goldens with the device mixture-of-experts path (run_moe: forge_moe_hidden / forge_moe_down)
      foreach(m cuda hip opencl ze)
        set(_moe_dev ${m})
        if(m STREQUAL "ze")
          set(_moe_dev level_zero)
        endif()
        add_test(NAME golden_moe_mla_${m}_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_moe.py
                 --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_moe_${m}mock)
        set_tests_properties(golden_moe_mla_${m}_mock PROPERTIES LABELS gpu
          ENVIRONMENT "FORGE_DEVICE=${_moe_dev};FORGE_ZE_TEST_IMAGE=opencl-c;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mock${m}")
      endforeach()
      add_test(NAME golden_tiny_llama_hip_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_hipmock)
      set_tests_properties(golden_tiny_llama_hip_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=hip;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockhip")
      # The other template families: attention_bias over the KV state, conv1d / conv2d, layer_norm and gelu run on the
      # device (the PORT_NN kernels); what has no device kernel (transpose, l2_normalize, masked_mean, ...) must run on
      # the CPU between device steps and give the same result.
      foreach(_g gpt2 gemma phi3 bert vit whisper)
        add_test(NAME golden_tiny_${_g}_cuda_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_${_g}.py
                 --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_${_g}_cumock)
        set_tests_properties(golden_tiny_${_g}_cuda_mock PROPERTIES LABELS gpu
          ENVIRONMENT "FORGE_DEVICE=cuda;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockcuda")
        add_test(NAME golden_tiny_${_g}_opencl_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_${_g}.py
                 --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_${_g}_clmock)
        set_tests_properties(golden_tiny_${_g}_opencl_mock PROPERTIES LABELS gpu
          ENVIRONMENT "FORGE_DEVICE=opencl;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockopencl")
      endforeach()
      # RoPE frequency scaling end to end on the mock devices (host-built inverse-frequency tables; dynamic NTK on the CPU)
      add_test(NAME golden_rope_scaling_cuda_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_rope.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_rope_cumock)
      set_tests_properties(golden_rope_scaling_cuda_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=cuda;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockcuda")
      add_test(NAME golden_rope_scaling_opencl_mock COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_rope.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_rope_clmock)
      set_tests_properties(golden_rope_scaling_opencl_mock PROPERTIES LABELS gpu
        ENVIRONMENT "FORGE_DEVICE=opencl;LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/mockopencl")
    endif()
  endif()
  if(FORGE_HAVE_VULKAN)
    target_include_directories(test_vulkan PRIVATE ${FORGE_GEN_DIR}/vkinc)
  endif()
  # CUDA / HIP: compile the generated kernels to PTX / AMDGCN with clang (no SDK needed); hardware runs need a GPU.
  if(FORGE_GPU_CLANG)
    # Apple clang has no NVPTX backend and some Windows clang builds no AMDGPU one: a missing backend is a
    # skip (77) with the reason printed, not a failure. The skip script asks the clang that would run the test.
    execute_process(COMMAND ${FORGE_GPU_CLANG} -print-targets OUTPUT_VARIABLE _forge_gpu_targets ERROR_QUIET)
    foreach(_t cuda:nvptx64 hip:amdgcn)
      string(REPLACE ":" ";" _tp ${_t})
      list(GET _tp 0 _lang)
      list(GET _tp 1 _be)
      string(FIND "${_forge_gpu_targets}" " ${_be} " _has)
      set(FORGE_GPU_HAS_${_lang} ON)
      if(_has LESS 0)
        set(FORGE_GPU_HAS_${_lang} OFF)
      endif()
    endforeach()
    if(FORGE_GPU_HAS_cuda)
      add_test(NAME gpu_compile_cuda COMMAND ${FORGE_GPU_CLANG} -x cuda --cuda-device-only -nocudainc -nocudalib -Wno-unknown-cuda-version
               --cuda-gpu-arch=sm_80 -Werror -S -o ${FORGE_GPU_DIR}/kernels.check_sm80.ptx ${FORGE_GPU_DIR}/kernels.cu)
    else()
      add_test(NAME gpu_compile_cuda COMMAND ${CMAKE_COMMAND} -E echo "skip: ${FORGE_GPU_CLANG} has no nvptx64 backend")
      set_tests_properties(gpu_compile_cuda PROPERTIES SKIP_REGULAR_EXPRESSION "^skip: ")
    endif()
    if(FORGE_GPU_HAS_hip)
      add_test(NAME gpu_compile_hip COMMAND ${FORGE_GPU_CLANG} -x hip --cuda-device-only -nogpuinc -nogpulib
               --offload-arch=gfx90a -Werror -S -o ${FORGE_GPU_DIR}/kernels.check_gfx90a.s ${FORGE_GPU_DIR}/kernels.cu)
    else()
      add_test(NAME gpu_compile_hip COMMAND ${CMAKE_COMMAND} -E echo "skip: ${FORGE_GPU_CLANG} has no amdgcn backend")
      set_tests_properties(gpu_compile_hip PROPERTIES SKIP_REGULAR_EXPRESSION "^skip: ")
    endif()
    add_test(NAME gpu_compile_opencl COMMAND ${FORGE_GPU_CLANG} -x cl -cl-std=CL1.2 --target=spir64 -Werror -fsyntax-only ${FORGE_GPU_DIR}/kernels.cl)
    add_test(NAME gpu_compile_opencl_nofp64 COMMAND ${FORGE_GPU_CLANG} -x cl -cl-std=CL3.0 --target=spir64 -Xclang -cl-ext=-cl_khr_fp64,-__opencl_c_fp64
             -Werror -fsyntax-only ${FORGE_GPU_DIR}/kernels.cl)
    # Mobile: NDK cross-builds (skip without ANDROID_NDK_HOME), device + iOS runs (skip without adb/Xcode).
    foreach(abi arm64-v8a x86_64)
      add_test(NAME mobile_android_${abi} COMMAND sh ${PROJECT_SOURCE_DIR}/tests/mobile_build.sh android ${abi})
      set_tests_properties(mobile_android_${abi} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800 LABELS "mobile;slow"
        ENVIRONMENT "FORGE_MOBILE_BUILD_DIR=${PROJECT_BINARY_DIR}/mobile")
    endforeach()
    add_test(NAME mobile_android_device COMMAND sh ${PROJECT_SOURCE_DIR}/tests/mobile_build.sh android-device)
    add_test(NAME mobile_ios COMMAND sh ${PROJECT_SOURCE_DIR}/tests/mobile_build.sh ios)
    set_tests_properties(mobile_android_device mobile_ios PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 LABELS "mobile;slow")
    find_path(FORGE_JNI_H jni.h HINTS $ENV{JAVA_HOME}/include NO_CMAKE_SYSTEM_PATH)  # JDK only; the NDK path is covered by mobile_android_*
    if(FORGE_JNI_H)  # host syntax check of the JNI bridge
      file(GLOB _jni_md "${FORGE_JNI_H}/linux" "${FORGE_JNI_H}/darwin" "${FORGE_JNI_H}/win32")
      set(_jni_inc -I${FORGE_JNI_H})
      foreach(d ${_jni_md})
        list(APPEND _jni_inc -I${d})
      endforeach()
      if(MSVC)   # cl / clang-cl spelling of the same check
        add_test(NAME jni_syntax COMMAND ${CMAKE_C_COMPILER} /nologo /c /Fo${PROJECT_BINARY_DIR}/jni_syntax.obj /WX /W3 ${_jni_inc}
          -I${PROJECT_SOURCE_DIR}/inference/include ${PROJECT_SOURCE_DIR}/bindings/android/jni/forge_jni.c)
      else()
        add_test(NAME jni_syntax COMMAND ${CMAKE_C_COMPILER} -c -o ${PROJECT_BINARY_DIR}/jni_syntax.o -Werror -Wall -Wno-unused-command-line-argument ${_jni_inc}
          -I${PROJECT_SOURCE_DIR}/inference/include ${PROJECT_SOURCE_DIR}/bindings/android/jni/forge_jni.c)
      endif()
    endif()
    add_test(NAME gpu_compile_metal_syntax COMMAND ${FORGE_GPU_CLANG} --driver-mode=g++ -std=c++17 -fsyntax-only -Wno-unknown-attributes -Werror
             -include ${PROJECT_SOURCE_DIR}/tests/shims/msl_host_shim.hpp -x c++ ${FORGE_GPU_DIR}/kernels.metal)
    # gpu/src/metal.mm (the Metal host code, Objective-C++) parsed against stub declarations of the Metal API on Linux,
    # where no Apple SDK exists: catches C++ / Objective-C errors before the macOS build does. Not a check of Metal usage.
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT CMAKE_CROSSCOMPILING)
      add_test(NAME gpu_compile_metal_host_syntax COMMAND ${FORGE_GPU_CLANG} --driver-mode=g++ -x objective-c++ -std=c++23 -fobjc-arc -fobjc-runtime=gnustep-2.0
               -fsyntax-only -Werror -Wno-objc-root-class -I${PROJECT_SOURCE_DIR}/tests/shims/metal_stub -I${PROJECT_SOURCE_DIR}/gpu/include
               -I${PROJECT_SOURCE_DIR}/support/include -I${PROJECT_SOURCE_DIR}/third_party -I${PROJECT_SOURCE_DIR}/third_party/monocypher -I${FORGE_GEN_DIR}
               -I${PROJECT_SOURCE_DIR}/core/include -I${PROJECT_SOURCE_DIR}/os/include -I${PROJECT_SOURCE_DIR}/ops/include -I${PROJECT_SOURCE_DIR}/simd/include
               ${PROJECT_SOURCE_DIR}/gpu/src/metal.mm)
      set_tests_properties(gpu_compile_metal_host_syntax PROPERTIES LABELS gpu)
    endif()
    find_program(FORGE_XCRUN xcrun)
    if(FORGE_XCRUN)   # real Metal compiler (macOS with Xcode)
      add_test(NAME gpu_compile_metal COMMAND ${FORGE_XCRUN} -sdk macosx metal -c ${FORGE_GPU_DIR}/kernels.metal -o ${PROJECT_BINARY_DIR}/kernels.air)
      set_tests_properties(gpu_compile_metal PROPERTIES LABELS gpu)
    endif()
    set_tests_properties(gpu_compile_metal_syntax gpu_compile_cuda gpu_compile_hip gpu_compile_opencl gpu_compile_opencl_nofp64 PROPERTIES LABELS gpu)
  endif()
  # Cross builds (e.g. aarch64 under qemu) are typically static: skip the shared-library smoke test there.
  if(NOT CMAKE_CROSSCOMPILING)
    add_executable(capi_smoke tests/capi_smoke.c)
    target_include_directories(capi_smoke PRIVATE inference/include)
    target_link_libraries(capi_smoke PRIVATE forge_c)
    add_executable(doc_example_capi_generate docs/examples/capi_generate.c)
    target_include_directories(doc_example_capi_generate PRIVATE inference/include)
    target_link_libraries(doc_example_capi_generate PRIVATE forge_c)
    # libforge.so needs libstdc++; link the C consumers with the C++ driver so toolchains with a private
    # sysroot (conda clang) can resolve it.
    set_target_properties(capi_smoke doc_example_capi_generate PROPERTIES LINKER_LANGUAGE CXX)
    if(FORGE_BUILD_APPS)
      add_test(NAME doc_example_capi_generate COMMAND doc_example_capi_generate ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk)
      set_tests_properties(doc_example_capi_generate PROPERTIES DEPENDS golden_tiny_llama)
    endif()
  endif()
  # Documentation examples (docs/examples): compiled and run so the guides cannot drift from the API.
  add_executable(doc_example_simd_dispatch docs/examples/simd_dispatch.cpp)
  target_link_libraries(doc_example_simd_dispatch PRIVATE forge_simd_mv)
  add_test(NAME doc_example_simd_dispatch COMMAND doc_example_simd_dispatch)
  add_executable(doc_example_gpu_graph docs/examples/gpu_graph.cpp)
  target_link_libraries(doc_example_gpu_graph PRIVATE forge_gpu forge_simd_mv)
  add_test(NAME doc_example_gpu_graph COMMAND doc_example_gpu_graph)
  add_executable(doc_example_modelfile_roundtrip docs/examples/modelfile_roundtrip.cpp)
  target_link_libraries(doc_example_modelfile_roundtrip PRIVATE forge_modelfile forge_simd_mv)
  add_test(NAME doc_example_modelfile_roundtrip COMMAND doc_example_modelfile_roundtrip)
  add_executable(doc_example_graph_build docs/examples/graph_build.cpp)
  target_link_libraries(doc_example_graph_build PRIVATE forge_graph forge_verify forge_simd_mv)
  add_test(NAME doc_example_graph_build COMMAND doc_example_graph_build)
  add_executable(doc_example_os_features docs/examples/os_features.cpp)
  target_link_libraries(doc_example_os_features PRIVATE forge_os)
  add_test(NAME doc_example_os_features COMMAND doc_example_os_features)
  add_executable(doc_example_loadbalancer_pool docs/examples/loadbalancer_pool.cpp)
  target_link_libraries(doc_example_loadbalancer_pool PRIVATE forge_loadbalancer)
  add_test(NAME doc_example_loadbalancer_pool COMMAND doc_example_loadbalancer_pool)
  add_executable(doc_example_inference_generate docs/examples/inference_generate.cpp)
  target_link_libraries(doc_example_inference_generate PRIVATE forge_inference forge_verify forge_simd_mv)
  if(FORGE_BUILD_APPS)
    add_test(NAME doc_example_inference_generate COMMAND doc_example_inference_generate ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk "Hi")
    set_tests_properties(doc_example_inference_generate PROPERTIES DEPENDS golden_tiny_llama)
  endif()
  if(FORGE_BUILD_APPS)
    add_test(NAME modelfile_signing COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/sign_test.py
             $<TARGET_FILE:forge> ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk)
    set_tests_properties(modelfile_signing PROPERTIES DEPENDS golden_tiny_llama)
    # detached signatures (forge sign / verify on any file), forge script and rpc-serve under integrity.scripts
    add_test(NAME signing_cli COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/signing_cli_test.py $<TARGET_FILE:forge>)
    set_tests_properties(signing_cli PROPERTIES LABELS security)
    add_test(NAME onnx_import COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/onnx_test.py $<TARGET_FILE:forge>)
    add_test(NAME golden_moe_mla
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_moe.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_moe)
    # Python with torch/transformers/gguf (verify/golden/requirements.txt) for the golden tests below.
    set(FORGE_GOLDEN_PYTHON "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with torch/transformers/gguf for golden tests")
    add_test(NAME golden_quant_inplace
      COMMAND ${FORGE_GOLDEN_PYTHON} ${PROJECT_SOURCE_DIR}/verify/golden/quant_inplace.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_quant_inplace)
    # Qwen3.5 (Gated DeltaNet) against transformers; skips without torch + transformers >= 5. Set
    # FORGE_GOLDEN_PYTHON to an interpreter that has them (e.g. a venv) to run it.
    add_test(NAME golden_tiny_qwen3_5
      COMMAND ${FORGE_GOLDEN_PYTHON} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_qwen3_5.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_tiny_qwen3_5)
    add_test(NAME golden_tiny_qwen3
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_qwen3.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_qwen3)
    add_test(NAME golden_tiny_llama
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden)
    # Tiny random models of the other template families (GPT-2, Gemma 1/2/3, Phi-3 / Phi-4-mini, BERT / RoBERTa
    # embedding, ViT / CLIP / SigLIP vision, Whisper encoder) against float64 numpy references.
    set(FORGE_GOLDEN_FAMILIES gpt2 gemma phi3 bert vit whisper sessions)   # sessions: diffusion_loop and streaming models (verify/golden/tiny_sessions.py)
    foreach(_g ${FORGE_GOLDEN_FAMILIES})
      add_test(NAME golden_tiny_${_g}
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_${_g}.py
                --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_${_g})
      if(CMAKE_CROSSCOMPILING_EMULATOR)
        list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " _forge_runner_g)
        set_tests_properties(golden_tiny_${_g} PROPERTIES ENVIRONMENT "FORGE_RUNNER=${_forge_runner_g}")
      endif()
    endforeach()
    # Qwen2 (biases on the Q/K/V projections) and RoPE frequency scaling (linear, dynamic, YaRN, llama3), HF and GGUF,
    # against float64 numpy references
    add_test(NAME golden_tiny_qwen2
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_qwen2.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_qwen2)
    add_test(NAME golden_rope_scaling
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_rope.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_rope)
    # GGUF tensor types against gguf-py (exits 77 = skipped without the gguf package); AWQ / GPTQ checkpoints against
    # the packers in the test script.
    add_test(NAME golden_gguf_types
      COMMAND ${FORGE_GOLDEN_PYTHON} ${PROJECT_SOURCE_DIR}/verify/golden/gguf_types.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_gguf_types)
    set_tests_properties(golden_gguf_types PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME golden_hf_quant
      COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/hf_quant.py
              --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_hf_quant)
    if(FORGE_HAVE_VULKAN AND NOT CMAKE_CROSSCOMPILING)
      # Same goldens with the Vulkan lowerings (ops without a GPU kernel run on the CPU).
      add_test(NAME golden_tiny_llama_vulkan COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_llama.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_vk)
      add_test(NAME golden_moe_mla_vulkan COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_moe.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_moe_vk)
      # scaled rope has no GLSL kernel: vk::supports sends it to the CPU, the rest of the graph stays on the device
      add_test(NAME golden_rope_scaling_vulkan COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/verify/golden/tiny_rope.py
               --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_rope_vk)
      set_tests_properties(golden_tiny_llama_vulkan golden_moe_mla_vulkan golden_rope_scaling_vulkan PROPERTIES ENVIRONMENT "FORGE_DEVICE=vulkan" LABELS gpu)
      # Quantised weights kept encoded on the device (the GLSL ports of forge_qmatmul_t / forge_qglu / forge_qgather:
      # q8_0 HF imports, in-place quantisation, AWQ / GPTQ checkpoints), QK-norm, and the other template families
      # (their attention_bias / conv / layer_norm steps run on the CPU between device steps).
      set(_vk_goldens hf_quant:${Python3_EXECUTABLE} quant_inplace:${FORGE_GOLDEN_PYTHON} tiny_qwen2:${Python3_EXECUTABLE} tiny_qwen3:${Python3_EXECUTABLE})
      foreach(_g ${FORGE_GOLDEN_FAMILIES})
        list(APPEND _vk_goldens tiny_${_g}:${Python3_EXECUTABLE})
      endforeach()
      foreach(_e ${_vk_goldens})
        string(REGEX REPLACE "^([^:]+):(.*)$" "\\1" _g "${_e}")
        string(REGEX REPLACE "^([^:]+):(.*)$" "\\2" _py "${_e}")
        add_test(NAME golden_${_g}_vulkan COMMAND ${_py} ${PROJECT_SOURCE_DIR}/verify/golden/${_g}.py
                 --forge $<TARGET_FILE:forge> --work ${PROJECT_BINARY_DIR}/golden_${_g}_vk)
        set_tests_properties(golden_${_g}_vulkan PROPERTIES ENVIRONMENT "FORGE_DEVICE=vulkan" LABELS gpu)
      endforeach()
    endif()
    if(CMAKE_CROSSCOMPILING_EMULATOR)
      list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " _forge_runner)
      set_tests_properties(golden_tiny_llama golden_tiny_qwen2 golden_rope_scaling golden_tiny_qwen3 golden_tiny_qwen3_5 golden_quant_inplace golden_moe_mla golden_gguf_types golden_hf_quant modelfile_signing signing_cli onnx_import PROPERTIES ENVIRONMENT "FORGE_RUNNER=${_forge_runner}")
    endif()
    if(TARGET capi_smoke)
      # ABI 1-4: the generative golden, the tiny BERT for forge_embed_*, and a work dir for the adapter / plan sidecar it writes
      add_test(NAME capi_smoke COMMAND capi_smoke ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk ${PROJECT_BINARY_DIR}/golden_bert/bert_f32.fpk ${PROJECT_BINARY_DIR})
      set_tests_properties(capi_smoke PROPERTIES DEPENDS "golden_tiny_llama;golden_tiny_bert")
      # Not in sanitizer builds: an uninstrumented python cannot dlopen a libforge.so that needs the ASan / TSan
      # runtime (it would have to be preloaded, and then conflicts with the sanitized forge the test also starts).
      # capi_smoke covers the same C API there.
      if(NOT FORGE_SANITIZE AND NOT FORGE_TSAN)
        add_test(NAME python_bindings COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/python_bindings.py
                 ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk $<TARGET_FILE:forge> ${PROJECT_BINARY_DIR}/golden/tiny_q8_0.fpk
                 ${PROJECT_BINARY_DIR}/golden_bert/bert_f32.fpk ${PROJECT_BINARY_DIR})
        set_tests_properties(python_bindings PROPERTIES DEPENDS "golden_tiny_llama;golden_tiny_bert" ENVIRONMENT "FORGE_LIB=$<TARGET_FILE:forge_c>")
      endif()
      add_test(NAME serve_http COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/serve_test.py
               $<TARGET_FILE:forge-serve> $<TARGET_FILE:forge> ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk)
      set_tests_properties(serve_http PROPERTIES DEPENDS golden_tiny_llama)
      # The JNI surface: the Kotlin externals, their Java mirror and forge_jni.c declare the same natives
      add_test(NAME jni_decl_matches COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/jni_decl_check.py ${PROJECT_SOURCE_DIR})
      # ... and the bridge runs on a host JVM (every native, -Xcheck:jni). Linux hosts with a JDK only: the one place
      # it has been run. The Kotlin classes on top are not compiled by this (docs/bindings.md).
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT FORGE_SANITIZE)
        find_package(JNI QUIET)
        find_package(Java QUIET COMPONENTS Runtime Development)
        if(JNI_FOUND AND Java_FOUND)
          if(NOT TARGET forge_jni)
            add_library(forge_jni SHARED bindings/android/jni/forge_jni.c)
            target_include_directories(forge_jni PRIVATE inference/include ${JNI_INCLUDE_DIRS})
            target_link_libraries(forge_jni PRIVATE forge_c)
          endif()
          add_test(NAME jni_smoke COMMAND ${CMAKE_COMMAND} -DJAVAC=${Java_JAVAC_EXECUTABLE} -DJAVA=${Java_JAVA_EXECUTABLE}
                   -DSRC=${PROJECT_SOURCE_DIR}/tests/jni -DOUT=${PROJECT_BINARY_DIR}/jni_smoke -DLIBDIR=$<TARGET_FILE_DIR:forge_jni>
                   -DFPK=${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk -DBERT=${PROJECT_BINARY_DIR}/golden_bert/bert_f32.fpk
                   -P ${PROJECT_SOURCE_DIR}/tests/jni_smoke.cmake)
          set_tests_properties(jni_smoke PROPERTIES DEPENDS "golden_tiny_llama;golden_tiny_bert" TIMEOUT 300)
        endif()
      endif()
      # request fields, tool calling, HTTP behaviour (keep-alive, chunked bodies, limits, shutdown) and --lora
      add_test(NAME serve_api COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/serve_api_test.py
               $<TARGET_FILE:forge-serve> $<TARGET_FILE:forge> ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk ${PROJECT_BINARY_DIR}/golden/hf ${PROJECT_BINARY_DIR}/serve_api)
      set_tests_properties(serve_api PROPERTIES DEPENDS golden_tiny_llama TIMEOUT 600)
      # API keys, rate limits and quotas; TLS and mutual TLS (skipped parts are printed when the build has no OpenSSL)
      add_test(NAME serve_auth COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/serve_auth_test.py
               $<TARGET_FILE:forge-serve> ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk ${PROJECT_BINARY_DIR}/serve_auth)
      set_tests_properties(serve_auth PROPERTIES DEPENDS golden_tiny_llama TIMEOUT 300)
      # /v1/embeddings on a `session embedding` model (the tiny BERT the bert golden imports), 400 on a generative one
      add_test(NAME serve_embeddings COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/serve_embeddings_test.py
               $<TARGET_FILE:forge-serve> $<TARGET_FILE:forge> ${PROJECT_BINARY_DIR}/golden_bert/bert_f32.fpk
               ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk ${PROJECT_BINARY_DIR}/serve_embeddings)
      set_tests_properties(serve_embeddings PROPERTIES DEPENDS "golden_tiny_bert;golden_tiny_llama")
      # forge run --chat and /v1/chat/completions through a model file's own chat template (a checked-in tiny GGUF)
      add_test(NAME chat_template_http COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/chat_template_test.py
               $<TARGET_FILE:forge-serve> $<TARGET_FILE:forge> ${PROJECT_SOURCE_DIR}/tests/data/tokenizer/gguf_tiny_llama.gguf ${PROJECT_BINARY_DIR}/chat_template_http)
      add_test(NAME security_cli COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/security_cli_test.py
               $<TARGET_FILE:forge> $<TARGET_FILE:forge-serve> ${PROJECT_BINARY_DIR}/golden/tiny_f32.fpk)
      set_tests_properties(security_cli PROPERTIES DEPENDS golden_tiny_llama LABELS security)
    endif()
    # Python-module prerequisites, probed once at configure time: a test whose modules are missing is
    # reported by ctest as "Not Run (Disabled)" instead of failing (install them to run it; see docs/building.md).
    function(forge_py_needs mods)
      string(REPLACE ";" "," _imp "${mods}")
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import ${_imp}" RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
      if(NOT _rc EQUAL 0)
        message(STATUS "FORGE: python lacks one of {${_imp}}; disabled tests: ${ARGN}")
        foreach(t IN LISTS ARGN)
          if(TEST ${t})
            set_tests_properties(${t} PROPERTIES DISABLED TRUE)
          endif()
        endforeach()
      endif()
    endfunction()
    forge_py_needs("numpy;safetensors" golden_tiny_llama golden_tiny_qwen2 golden_rope_scaling golden_rope_scaling_cuda_mock
      golden_rope_scaling_opencl_mock golden_rope_scaling_vulkan golden_tiny_qwen3 golden_tiny_qwen3_5 golden_moe_mla golden_quant_inplace
      golden_gguf_types golden_hf_quant
      golden_tiny_llama_vulkan golden_moe_mla_vulkan golden_tiny_llama_cuda_mock golden_tiny_llama_pipe_auto_cuda_mock golden_tiny_llama_opencl_mock
      golden_tiny_llama_ze_mock golden_tiny_llama_hip_mock golden_moe_mla_cuda_mock golden_moe_mla_hip_mock golden_moe_mla_opencl_mock golden_moe_mla_ze_mock doc_example_inference_generate doc_example_capi_generate
      capi_smoke python_bindings serve_http serve_api serve_auth security_cli modelfile_signing)
    if(TEST jni_smoke)
      forge_py_needs("numpy;safetensors" jni_smoke)   # it runs on the golden models
    endif()
    foreach(_g ${FORGE_GOLDEN_FAMILIES})   # the family goldens (and the mock-device runs of them, where defined)
      forge_py_needs("numpy;safetensors" golden_tiny_${_g} golden_tiny_${_g}_cuda_mock golden_tiny_${_g}_opencl_mock)
    endforeach()
    forge_py_needs("numpy;safetensors" serve_embeddings)
    forge_py_needs("cryptography" modelfile_signing)
    forge_py_needs("numpy;onnx" onnx_import)
    if(Python3_EXECUTABLE)
      add_test(NAME docs_up_to_date COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/gen_docs.py --check)
      # relative links, anchors and backticked repository paths of README.md and docs/*.md resolve
      add_test(NAME docs_links COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/check_docs_links.py --root ${PROJECT_SOURCE_DIR})
      # the workflow GitHub runs and the two copies kept under ci/ (ci/install-workflows.sh installs from ci/github) are one file
      if(EXISTS ${PROJECT_SOURCE_DIR}/.github/workflows/ci.yml)
        add_test(NAME ci_workflow_copies COMMAND ${CMAKE_COMMAND} -E compare_files ${PROJECT_SOURCE_DIR}/.github/workflows/ci.yml ${PROJECT_SOURCE_DIR}/ci/github/ci.yml)
        add_test(NAME ci_workflow_copy_flat COMMAND ${CMAKE_COMMAND} -E compare_files ${PROJECT_SOURCE_DIR}/.github/workflows/ci.yml ${PROJECT_SOURCE_DIR}/ci/github-actions-ci.yml)
      endif()
      add_test(NAME sbom_spdx COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/sbom_test.py ${PROJECT_SOURCE_DIR})
      # docs/reference.md's op x backend table is read from the sources by gen_docs.py; this compares it with
      # the registries of the code as built (tools/coverage_dump.cpp walks them).
      if(NOT CMAKE_CROSSCOMPILING)   # the Python side starts the executable directly
        add_executable(forge_coverage_dump tools/coverage_dump.cpp)
        target_link_libraries(forge_coverage_dump PRIVATE forge_inference forge_verify forge_simd_mv)
        add_test(NAME coverage_table COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/gen_docs.py --check-coverage $<TARGET_FILE:forge_coverage_dump>)
      endif()
      # forge script against the real compiler: every tests/script_conformance/*.cpp is built with the host
      # compiler and run, then run by `forge script` on the tree-walker and the VM; the outputs must be equal.
      # Skips (77) when the host compiler has no C++23 <print>. About a minute: one host compile per program.
      if(TARGET forge AND NOT CMAKE_CROSSCOMPILING)
        add_test(NAME script_conformance COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/script_conformance/run.py
                 --forge $<TARGET_FILE:forge> --cxx ${CMAKE_CXX_COMPILER} --work ${PROJECT_BINARY_DIR}/script_conformance)
        set_tests_properties(script_conformance PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
      endif()
    endif()
  endif()
endif()

# ---- ahead-of-time model compilation (docs/aot-models.md) ----
# `forge aot --aot-mode native` compiles generated C++ against this build. forge_aot_sdk.txt records what such a
# translation unit needs: the compiler, the flags that change how the headers compile, the include directories and
# definitions of forge_inference / forge_verify, and the static libraries behind them (found by walking the link
# interface, so new include directories are picked up without touching this block).
function(_forge_aot_walk t)
  get_property(_seen GLOBAL PROPERTY FORGE_AOT_SEEN)
  if("${t}" IN_LIST _seen)
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY FORGE_AOT_SEEN "${t}")
  if(t STREQUAL "Threads::Threads")
    set_property(GLOBAL APPEND PROPERTY FORGE_AOT_SYS "-pthread")
    return()
  endif()
  if(NOT TARGET ${t})
    if(t MATCHES "^[A-Za-z0-9_]+$")   # a system library by name (dl, ws2_32, ...)
      set_property(GLOBAL APPEND PROPERTY FORGE_AOT_SYS "-l${t}")
    elseif(t MATCHES "^-")
      set_property(GLOBAL APPEND PROPERTY FORGE_AOT_SYS "${t}")
    endif()
    return()
  endif()
  get_target_property(_imp ${t} IMPORTED)
  if(_imp)
    return()
  endif()
  get_target_property(_type ${t} TYPE)
  if(_type STREQUAL "STATIC_LIBRARY")
    set_property(GLOBAL APPEND PROPERTY FORGE_AOT_LIB "$<TARGET_FILE:${t}>")
  endif()
  get_target_property(_inc ${t} INTERFACE_INCLUDE_DIRECTORIES)
  foreach(d IN LISTS _inc)
    string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" d "${d}")
    if(d AND NOT d MATCHES "^\\$<")
      get_filename_component(d "${d}" ABSOLUTE BASE_DIR "${PROJECT_SOURCE_DIR}")
      set_property(GLOBAL APPEND PROPERTY FORGE_AOT_INC "${d}")
    endif()
  endforeach()
  get_target_property(_def ${t} INTERFACE_COMPILE_DEFINITIONS)
  foreach(d IN LISTS _def)
    string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" d "${d}")
    if(d AND NOT d MATCHES "^\\$<")
      set_property(GLOBAL APPEND PROPERTY FORGE_AOT_DEF "${d}")
    endif()
  endforeach()
  get_target_property(_lnk ${t} INTERFACE_LINK_LIBRARIES)
  foreach(l IN LISTS _lnk)
    if(l)
      string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" l "${l}")
      if(NOT l MATCHES "^\\$<")
        _forge_aot_walk("${l}")
      endif()
    endif()
  endforeach()
endfunction()
if(FORGE_BUILD_APPS AND TARGET forge)
  foreach(_t forge_inference forge_verify forge_simd_mv)
    _forge_aot_walk(${_t})
  endforeach()
  get_property(_aot_inc GLOBAL PROPERTY FORGE_AOT_INC)
  get_property(_aot_def GLOBAL PROPERTY FORGE_AOT_DEF)
  get_property(_aot_lib GLOBAL PROPERTY FORGE_AOT_LIB)
  get_property(_aot_sys GLOBAL PROPERTY FORGE_AOT_SYS)
  list(REMOVE_DUPLICATES _aot_inc)
  list(APPEND _aot_inc "${PROJECT_SOURCE_DIR}/inference/include")
  list(REMOVE_DUPLICATES _aot_inc)
  list(REMOVE_DUPLICATES _aot_def)
  list(REMOVE_DUPLICATES _aot_sys)
  set(_aot_flags ${CMAKE_CXX23_STANDARD_COMPILE_OPTION})
  separate_arguments(_aot_user NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
  list(APPEND _aot_flags ${_aot_user})
  if(FORGE_STD_STOP_TOKEN_EXPERIMENTAL OR ANDROID)
    list(APPEND _aot_flags -fexperimental-library)
  endif()
  if(FORGE_SANITIZE)
    list(APPEND _aot_flags -fsanitize=address,undefined -fno-omit-frame-pointer)
  endif()
  if(WIN32)
    list(APPEND _aot_def _USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN)
    list(APPEND _aot_sys -lws2_32 -lbcrypt -luserenv)
  endif()
  set(_aot_txt "# FORGE ${PROJECT_VERSION}: what `forge aot --aot-mode native` needs to compile against this build (docs/aot-models.md)\n")
  string(APPEND _aot_txt "cxx=${CMAKE_CXX_COMPILER}\ncxx_id=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\nar=${CMAKE_AR}\n")
  foreach(x IN LISTS _aot_flags)
    string(APPEND _aot_txt "flag=${x}\n")
  endforeach()
  foreach(x IN LISTS _aot_inc)
    string(APPEND _aot_txt "include=${x}\n")
  endforeach()
  foreach(x IN LISTS _aot_def)
    string(APPEND _aot_txt "define=${x}\n")
  endforeach()
  foreach(x IN LISTS _aot_lib)
    string(APPEND _aot_txt "lib=${x}\n")
  endforeach()
  foreach(x IN LISTS _aot_sys)
    string(APPEND _aot_txt "syslib=${x}\n")
  endforeach()
  # next to forge, where aot_load_sdk looks: <config>/ with a multi-config generator (one file per configuration)
  get_property(_sdk_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(_sdk_multi)
    set(_aot_sdk_file "${PROJECT_BINARY_DIR}/$<CONFIG>/forge_aot_sdk.txt")
  else()
    set(_aot_sdk_file "${PROJECT_BINARY_DIR}/forge_aot_sdk.txt")
  endif()
  file(GENERATE OUTPUT "${_aot_sdk_file}" CONTENT "${_aot_txt}")
  # The same for an installed tree (<prefix>/share/forge/forge_aot_sdk.txt, found next to ../bin/forge): paths are
  # written relative to the prefix as ${prefix}/..., which the reader resolves from the file's own location.
  include(GNUInstallDirs)
  if(NOT IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}" AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_DATADIR}")
    set(_aot_itxt "# FORGE ${PROJECT_VERSION}: what `forge aot --aot-mode native` needs to compile against this installation (docs/aot-models.md)\n")
    string(APPEND _aot_itxt "cxx=${CMAKE_CXX_COMPILER}\ncxx_id=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\nar=${CMAKE_AR}\n")
    foreach(x IN LISTS _aot_flags)
      string(APPEND _aot_itxt "flag=${x}\n")
    endforeach()
    foreach(x "" /forge/third_party /forge/third_party/monocypher)
      string(APPEND _aot_itxt "include=\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}${x}\n")
    endforeach()
    foreach(x IN LISTS _aot_def)
      if(NOT x MATCHES "^FORGE_SOURCE_DIR=")
        string(APPEND _aot_itxt "define=${x}\n")
      endif()
    endforeach()
    foreach(t forge_trust forge_simd_mv)
      string(APPEND _aot_itxt "lib=\${prefix}/${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:${t}>\n")
    endforeach()
    foreach(x IN LISTS _aot_sys)
      string(APPEND _aot_itxt "syslib=${x}\n")
    endforeach()
    file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/forge_aot_sdk.install.txt" CONTENT "${_aot_itxt}")
    install(FILES "${PROJECT_BINARY_DIR}/forge_aot_sdk.install.txt" DESTINATION ${CMAKE_INSTALL_DATADIR}/forge RENAME forge_aot_sdk.txt COMPONENT dev)
  endif()
  # forge finds forge_aot_sdk.txt next to itself; no FORGE_AOT_SDK_FILE: an absolute path would make the binary depend
  # on the build directory (the repro job builds in two and compares)
  add_dependencies(forge forge_simd_mv forge_trust)
  if(FORGE_BUILD_TESTS AND TARGET test_aot_native)
    # generate + compile + load with the host compiler; exits 77 without one (or without process spawning)
    target_compile_definitions(test_aot_native PRIVATE FORGE_AOT_SDK_FILE="${_aot_sdk_file}"
                               FORGE_TEST_FORGE_BIN="$<TARGET_FILE:forge>" FORGE_TEST_BINARY_DIR="${PROJECT_BINARY_DIR}")
    add_dependencies(test_aot_native forge)
    set_tests_properties(test_aot_native PROPERTIES LABELS slow SKIP_RETURN_CODE 77 TIMEOUT 3000)
  endif()
endif()
if(FORGE_BUILD_TESTS AND TARGET test_aot)
  # the golden MoE / quantised models (built by the golden tests) when they are there
  target_compile_definitions(test_aot PRIVATE FORGE_TEST_BINARY_DIR="${PROJECT_BINARY_DIR}")
  # ... and the models of the other template families: run after the golden tests that write them (order only; the
  # cases skip a model that is not there)
  set(_forge_aot_after golden_tiny_llama golden_moe_mla golden_tiny_qwen2 golden_rope_scaling)
  foreach(_g gpt2 gemma phi3 bert vit whisper)
    list(APPEND _forge_aot_after golden_tiny_${_g})
  endforeach()
  set(_forge_aot_deps "")
  foreach(_t ${_forge_aot_after})
    if(TEST ${_t})
      list(APPEND _forge_aot_deps ${_t})
    endif()
  endforeach()
  if(_forge_aot_deps)
    set_property(TEST test_aot APPEND PROPERTY DEPENDS ${_forge_aot_deps})
    if(TEST test_aot_native)
      set_property(TEST test_aot_native APPEND PROPERTY DEPENDS ${_forge_aot_deps})
    endif()
  endif()
endif()

# ---- fuzz targets (fuzz/): libFuzzer entry points + a built-in mutational driver for CI ----
if(FORGE_BUILD_TESTS)
  foreach(ft json blueprint fpk safetensors gguf tokenizer chat_template grammar onnx serve_http proxy_http rpc json_schema policy signature aot png bmp pnm inflate reflect dbus)
    if(FORGE_LIBFUZZER)
      add_executable(fuzz_${ft} fuzz/fuzz_${ft}.cpp)
      target_compile_options(fuzz_${ft} PRIVATE -fsanitize=fuzzer)
      target_link_options(fuzz_${ft} PRIVATE -fsanitize=fuzzer)
    else()
      add_executable(fuzz_${ft} fuzz/fuzz_${ft}.cpp fuzz/driver.cpp)
    endif()
    if(ft STREQUAL "dbus")   # the os::dbus codec is header-only: no need to build the inference library for it
      target_link_libraries(fuzz_${ft} PRIVATE forge_core)
    else()
      target_link_libraries(fuzz_${ft} PRIVATE forge_inference forge_verify forge_simd_mv)
    endif()
    if(ft STREQUAL "serve_http")       # the reader lives next to forge-serve (apps/serve_http.hpp)
      target_include_directories(fuzz_${ft} PRIVATE apps)
    elseif(ft STREQUAL "proxy_http")
      target_link_libraries(fuzz_${ft} PRIVATE forge_loadbalancer)
    elseif(ft STREQUAL "rpc" AND TARGET forge_rpc_tls)
      target_link_libraries(fuzz_${ft} PRIVATE forge_rpc_tls)
    endif()
    if(NOT FORGE_LIBFUZZER)
      set(_fz_iters ${FORGE_FUZZ_ITERS})
      if(ft STREQUAL "rpc" AND APPLE)
        # each input (one loopback connection) seems to wait out the client's 2 s receive timeout on the macOS runners
        # instead of seeing the server close (test_rpc takes 78 s there, 1 s on Linux; cause not found yet): 4000 and
        # then 500 inputs ran past the 600 s limit, so 1/40 of them here. Linux runs them all.
        math(EXPR _fz_iters "${FORGE_FUZZ_ITERS} / 40")
      endif()
      add_test(NAME fuzz_${ft} COMMAND fuzz_${ft} --iters ${_fz_iters} ${PROJECT_SOURCE_DIR}/fuzz/corpus/${ft}
               WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
      set_tests_properties(fuzz_${ft} PROPERTIES TIMEOUT 600 LABELS fuzz)
    endif()
  endforeach()
endif()
if(FORGE_BUILD_TESTS)   # seed writer for fuzz_signature / fuzz_policy (run by hand: see fuzz/make_sig_corpus.cpp)
  add_executable(fuzz_make_sig_corpus EXCLUDE_FROM_ALL fuzz/make_sig_corpus.cpp)
  target_link_libraries(fuzz_make_sig_corpus PRIVATE forge_inference forge_verify forge_simd_mv)
  add_executable(fuzz_make_fpk_corpus EXCLUDE_FROM_ALL fuzz/make_fpk_corpus.cpp)   # seeds of fuzz_fpk (run by hand)
  target_link_libraries(fuzz_make_fpk_corpus PRIVATE forge_inference forge_verify forge_simd_mv)
endif()
if(FORGE_BUILD_TESTS AND TARGET forge_script)   # script front end (parser + compiler + verifier)
  if(FORGE_LIBFUZZER)
    add_executable(fuzz_script fuzz/fuzz_script.cpp)
    target_compile_options(fuzz_script PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_script PRIVATE -fsanitize=fuzzer)
  else()
    add_executable(fuzz_script fuzz/fuzz_script.cpp fuzz/driver.cpp)
    add_test(NAME fuzz_script COMMAND fuzz_script --iters ${FORGE_FUZZ_ITERS} ${PROJECT_SOURCE_DIR}/fuzz/corpus/script
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(fuzz_script PROPERTIES TIMEOUT 600 LABELS fuzz)
  endif()
  target_link_libraries(fuzz_script PRIVATE forge_script)
  target_include_directories(fuzz_script PRIVATE script/src)
  # forge script --aot code generator (script/src/aot_codegen.cpp): translate, never execute; FORGE_FUZZ_AOT_DIFF adds
  # the slow differential mode (fuzz/fuzz_script_aot.cpp)
  if(FORGE_LIBFUZZER)
    add_executable(fuzz_script_aot fuzz/fuzz_script_aot.cpp)
    target_compile_options(fuzz_script_aot PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_script_aot PRIVATE -fsanitize=fuzzer)
  else()
    add_executable(fuzz_script_aot fuzz/fuzz_script_aot.cpp fuzz/driver.cpp)
    add_test(NAME fuzz_script_aot COMMAND fuzz_script_aot --iters ${FORGE_FUZZ_ITERS} ${PROJECT_SOURCE_DIR}/fuzz/corpus/script
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(fuzz_script_aot PROPERTIES TIMEOUT 600 LABELS fuzz)
  endif()
  target_link_libraries(fuzz_script_aot PRIVATE forge_script)
  target_include_directories(fuzz_script_aot PRIVATE script/src)
  add_dependencies(fuzz_script_aot forge_script_rt forge-script-run)
  # .fsc cache loader (script/src/fsc.cpp): header/checksum/signature checks and the payload decoder
  if(FORGE_LIBFUZZER)
    add_executable(fuzz_fsc fuzz/fuzz_fsc.cpp)
    target_compile_options(fuzz_fsc PRIVATE -fsanitize=fuzzer)
    target_link_options(fuzz_fsc PRIVATE -fsanitize=fuzzer)
  else()
    add_executable(fuzz_fsc fuzz/fuzz_fsc.cpp fuzz/driver.cpp)
    add_test(NAME fuzz_fsc COMMAND fuzz_fsc --iters ${FORGE_FUZZ_ITERS} ${PROJECT_SOURCE_DIR}/fuzz/corpus/fsc
             WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
    set_tests_properties(fuzz_fsc PROPERTIES TIMEOUT 600 LABELS fuzz)
  endif()
  target_link_libraries(fuzz_fsc PRIVATE forge_script)
  target_include_directories(fuzz_fsc PRIVATE script/src)
endif()

# App integrations (Part F.2): runnable embedding examples for FUSE, Cypha, CameraDetector, seatbelt, NeuralVFX.
option(FORGE_BUILD_EXAMPLES "Build examples/integrations" ON)
if(FORGE_BUILD_EXAMPLES)
  file(GLOB FORGE_EXAMPLES CONFIGURE_DEPENDS examples/integrations/*.cpp)
  foreach(e ${FORGE_EXAMPLES})
    get_filename_component(n ${e} NAME_WE)
    add_executable(example_${n} ${e})
    target_link_libraries(example_${n} PRIVATE forge_inference forge_verify forge_simd_mv)
    if(FORGE_BUILD_TESTS)
      add_test(NAME example_${n} COMMAND example_${n})
      set_tests_properties(example_${n} PROPERTIES LABELS integration)
    endif()
  endforeach()
endif()

# ---- Install + packaging (v1.0 release tooling; tools/release.py drives this) -------------------------
# Layout under the prefix:
#   include/                      every public header (forge.h, forge/<layer>/...), the generated *.gen.hpp
#   include/forge/third_party/    vendored headers the public headers include (monocypher, level-zero)
#   lib/                          libforge (C ABI) and the static libraries
#   lib/cmake/forge/              forgeConfig.cmake: find_package(forge CONFIG) -> forge::<name> targets
#   lib/pkgconfig/forge.pc        the C ABI for pkg-config
#   share/forge/templates/        import blueprint templates (found relative to the executable at run time)
#   bin/                          forge, forge-serve
include(CMakePackageConfigHelpers)

# The generated headers must exist before "install": tie their generators to the default target.
add_custom_target(forge_gen_headers ALL)
add_dependencies(forge_gen_headers forge_ops_gen forge_raw_gen forge_gpu_gen forge_gpu_images)

install(DIRECTORY core/include/ os/include/ simd/include/ ops/include/ gpu/include/ graph/include/ modelfile/include/ inference/include/ compress/include/
                  support/include/ verify/include/ security/include/ loadbalancer/include/ script/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev)
install(DIRECTORY ${FORGE_GEN_DIR}/forge/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/forge COMPONENT dev
        FILES_MATCHING PATTERN "*.gen.hpp")
install(FILES third_party/monocypher/monocypher.h third_party/monocypher/monocypher-ed25519.h
        DESTINATION ${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}/monocypher COMPONENT dev)
install(FILES third_party/level-zero/ze_api.h DESTINATION ${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}/level-zero COMPONENT dev)
install(FILES third_party/monocypher/LICENCE.md DESTINATION ${CMAKE_INSTALL_DOCDIR}/third_party/monocypher COMPONENT runtime)

# Exported targets: forge_<name> here is forge::<name> in the package (and as an alias in this build).
set(FORGE_EXPORT_TARGETS forge_core forge_os forge_ops forge_simd forge_verify forge_gpu forge_graph forge_modelfile forge_security
    forge_inference forge_compress forge_loadbalancer forge_simd_mv forge_trust forge_text forge_mobile forge_script forge_video forge_c)
foreach(t forge_metal forge_rpc_tls)   # Apple only / OpenSSL found
  if(TARGET ${t})
    list(APPEND FORGE_EXPORT_TARGETS ${t})
  endif()
endforeach()
foreach(t ${FORGE_EXPORT_TARGETS})
  string(REGEX REPLACE "^forge_" "" _n ${t})
  set_target_properties(${t} PROPERTIES EXPORT_NAME ${_n})
  if(NOT TARGET forge::${_n})
    add_library(forge::${_n} ALIAS ${t})
  endif()
endforeach()
install(TARGETS ${FORGE_EXPORT_TARGETS} EXPORT forgeTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
foreach(t forge forge-serve)
  if(TARGET ${t})
    install(TARGETS ${t} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
  endif()
endforeach()
install(DIRECTORY import/templates/ DESTINATION ${FORGE_INSTALL_TEMPLATEDIR} COMPONENT runtime FILES_MATCHING PATTERN "*.fbp")
install(FILES LICENSE NOTICE COMMERCIAL-LICENCE.md SECURITY.md README.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)

# forge script --aot (docs/aot.md): the AOT runtime under <prefix>/lib/forge/aot, where an installed `forge` looks
# for it (../lib/forge/aot next to the executable; else --aot-runtime DIR or $FORGE_AOT_RUNTIME). The library is also
# exported as forge::script_rt. Generated programs include forge_aot_rt.hpp and the interpreter's internal headers
# from lib/forge/aot/include, and forge/core + forge/os from the package's own include directory.
if(TARGET forge_script_rt)
  set(FORGE_AOT_INSTALL_DIR lib/forge/aot)
  set_target_properties(forge_script_rt PROPERTIES EXPORT_NAME script_rt)
  if(NOT TARGET forge::script_rt)
    add_library(forge::script_rt ALIAS forge_script_rt)
  endif()
  install(TARGETS forge_script_rt EXPORT forgeTargets ARCHIVE DESTINATION ${FORGE_AOT_INSTALL_DIR} COMPONENT dev)
  install(TARGETS forge-script-run RUNTIME DESTINATION ${FORGE_AOT_INSTALL_DIR} BUNDLE DESTINATION ${FORGE_AOT_INSTALL_DIR} COMPONENT dev)
  install(FILES ${FORGE_AOT_RT_HEADERS} DESTINATION ${FORGE_AOT_INSTALL_DIR}/include COMPONENT dev)
  if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(_aot_pkg_inc "${CMAKE_INSTALL_INCLUDEDIR}")
  else()
    file(RELATIVE_PATH _aot_pkg_inc "/p/${FORGE_AOT_INSTALL_DIR}" "/p/${CMAKE_INSTALL_INCLUDEDIR}")
  endif()
  if(IS_ABSOLUTE "${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}")
    set(_aot_pkg_tp "${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}/monocypher")
  else()
    file(RELATIVE_PATH _aot_pkg_tp "/p/${FORGE_AOT_INSTALL_DIR}" "/p/${FORGE_INSTALL_THIRD_PARTY_INCLUDEDIR}/monocypher")
  endif()
  file(GENERATE OUTPUT ${PROJECT_BINARY_DIR}/aot-runtime-install/forge-aot-runtime.cfg
       CONTENT "${_aot_cfg}lib=$<TARGET_FILE_NAME:forge_script_rt>\nstub=$<TARGET_FILE_NAME:forge-script-run>\ninclude=include\ninclude=${_aot_pkg_inc}\ninclude=${_aot_pkg_tp}\n")
  install(FILES ${PROJECT_BINARY_DIR}/aot-runtime-install/forge-aot-runtime.cfg DESTINATION ${FORGE_AOT_INSTALL_DIR} COMPONENT dev)
endif()

# CMake package.
set(FORGE_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/forge)
set(FORGE_PKG_HAVE_THREADS OFF)
if(Threads_FOUND)
  set(FORGE_PKG_HAVE_THREADS ON)
endif()
set(FORGE_PKG_HAVE_TLS OFF)
if(TARGET forge_rpc_tls)
  set(FORGE_PKG_HAVE_TLS ON)
endif()
set(FORGE_PKG_HAVE_VULKAN OFF)
if(FORGE_HAVE_VULKAN)
  set(FORGE_PKG_HAVE_VULKAN ON)
endif()
install(EXPORT forgeTargets NAMESPACE forge:: DESTINATION ${FORGE_INSTALL_CMAKEDIR} COMPONENT dev)
configure_package_config_file(cmake/forgeConfig.cmake.in ${PROJECT_BINARY_DIR}/forgeConfig.cmake
  INSTALL_DESTINATION ${FORGE_INSTALL_CMAKEDIR} PATH_VARS CMAKE_INSTALL_INCLUDEDIR FORGE_INSTALL_TEMPLATEDIR)
# 0.x: a minor release may change the C++ headers.
write_basic_package_version_file(${PROJECT_BINARY_DIR}/forgeConfigVersion.cmake VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/forgeConfig.cmake ${PROJECT_BINARY_DIR}/forgeConfigVersion.cmake cmake/forge_cxx_compat.cmake
        DESTINATION ${FORGE_INSTALL_CMAKEDIR} COMPONENT dev)

# pkg-config file for the C ABI (forge.h + libforge). Relocatable when the install directories are relative.
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  set(FORGE_PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
  cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_PREFIX}" OUTPUT_VARIABLE FORGE_PC_LIBDIR)
  cmake_path(ABSOLUTE_PATH CMAKE_INSTALL_INCLUDEDIR BASE_DIRECTORY "${CMAKE_INSTALL_PREFIX}" OUTPUT_VARIABLE FORGE_PC_INCLUDEDIR)
else()
  file(RELATIVE_PATH _forge_pc_rel "/p/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/p")
  string(REGEX REPLACE "/$" "" _forge_pc_rel "${_forge_pc_rel}")
  set(FORGE_PC_PREFIX "\${pcfiledir}/${_forge_pc_rel}")
  set(FORGE_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
  set(FORGE_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
configure_file(cmake/forge.pc.in ${PROJECT_BINARY_DIR}/forge.pc @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/forge.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig COMPONENT dev)

# install_consumer: install into a scratch prefix, then configure, build and run tests/install_consumer against it
# with find_package(forge). Needs the default target built. Not registered for cross builds (the consumer could not
# run) or sanitizer builds (the consumer would have to repeat the sanitizer flags).
if(FORGE_BUILD_TESTS AND NOT CMAKE_CROSSCOMPILING AND NOT FORGE_SANITIZE AND NOT FORGE_TSAN AND NOT FORGE_LIBFUZZER)
  add_test(NAME install_consumer COMMAND ${CMAKE_COMMAND}
    -DFORGE_BUILD_DIR=${PROJECT_BINARY_DIR} -DFORGE_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DFORGE_CONFIG=$<CONFIG>
    -DFORGE_WORK_DIR=${PROJECT_BINARY_DIR}/install_consumer
    -DFORGE_BINDIR=${CMAKE_INSTALL_BINDIR} -DFORGE_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR} -DFORGE_TEMPLATEDIR=${FORGE_INSTALL_TEMPLATEDIR}
    -DFORGE_GENERATOR=${CMAKE_GENERATOR} -DFORGE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM} -DFORGE_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
    -DFORGE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM} -DFORGE_C_COMPILER=${CMAKE_C_COMPILER} -DFORGE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DFORGE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DFORGE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
    -P ${PROJECT_SOURCE_DIR}/tests/install_consumer/run.cmake)
  set_tests_properties(install_consumer PROPERTIES LABELS "install" TIMEOUT 1800)
endif()


set(CPACK_PACKAGE_NAME forge)
set(CPACK_PACKAGE_VENDOR "Odin Loch")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "FORGE: one C++23 stack for SIMD, GPU and model inference")
set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/LICENSE)
set(CPACK_PACKAGE_FILE_NAME "forge-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "forge-${PROJECT_VERSION}-src")
set(CPACK_SOURCE_IGNORE_FILES "/build/" "/\\\\.git/" "/\\\\.venv/" "__pycache__")
if(WIN32)
  set(CPACK_GENERATOR ZIP)
  set(CPACK_SOURCE_GENERATOR ZIP)
else()
  set(CPACK_GENERATOR TGZ)
  set(CPACK_SOURCE_GENERATOR TGZ)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND CPACK_GENERATOR DEB)
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Odin Loch")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
  endif()
endif()
include(CPack)
