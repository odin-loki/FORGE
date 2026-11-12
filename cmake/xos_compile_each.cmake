# Compile each source in SRCS (;-list, passed as |-list) for TARGET with zig, objects into OUT. Every source is
# tried; the failures are listed at the end.
# Used instead of -fsyntax-only, which zig 0.17 breaks ("failed to rename tmp/... .o").
string(REPLACE "|" ";" _srcs "${SRCS}")
string(REPLACE "|" ";" _inc "${INCS}")
file(MAKE_DIRECTORY "${OUT}")
set(_failed "")
foreach(s ${_srcs})
  get_filename_component(n "${s}" NAME_WE)
  execute_process(COMMAND "${ZIG}" c++ -target ${TARGET} -std=c++23 -c ${_inc} "${s}" -o "${OUT}/${n}.o" RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    list(APPEND _failed "${n}")
  endif()
endforeach()
if(_failed)
  message(FATAL_ERROR "failed for ${TARGET}: ${_failed}")
endif()
