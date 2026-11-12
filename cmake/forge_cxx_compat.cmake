# Standard-library workaround FORGE's headers need on some toolchains. Used by the top-level CMakeLists.txt (applied
# to every target) and by the installed forgeConfig.cmake (applied to forge::os, so consumers get it too).
#
# forge_cxx_compat(<compile-options-var> <link-options-var>) sets the two variables to lists (possibly empty):
#   - libc++ < 20 (FreeBSD 14 base clang, older Xcode) keeps std::jthread / std::stop_token behind
#     -fexperimental-library.
# The probe results are cached (FORGE_STD_STOP_TOKEN_OK, FORGE_STD_STOP_TOKEN_EXPERIMENTAL).
include(CheckCXXSourceCompiles)
function(forge_cxx_compat out_compile out_link)
  set(_c "")
  set(_l "")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)   # compile-only: no -pthread/link needed for the probe
    set(CMAKE_REQUIRED_FLAGS -std=c++23)
    set(_src "#include <stop_token>\n#include <thread>\nint main() { std::stop_source s; std::jthread t([](std::stop_token) {}); return s.stop_requested(); }")
    check_cxx_source_compiles("${_src}" FORGE_STD_STOP_TOKEN_OK)
    if(NOT FORGE_STD_STOP_TOKEN_OK)
      set(CMAKE_REQUIRED_FLAGS "-std=c++23 -fexperimental-library")
      check_cxx_source_compiles("${_src}" FORGE_STD_STOP_TOKEN_EXPERIMENTAL)
      if(FORGE_STD_STOP_TOKEN_EXPERIMENTAL)
        list(APPEND _c -fexperimental-library)
        list(APPEND _l -fexperimental-library)
      endif()
    endif()
  endif()
  set(${out_compile} "${_c}" PARENT_SCOPE)
  set(${out_link} "${_l}" PARENT_SCOPE)
endfunction()
