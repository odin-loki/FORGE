# Install script for directory: /workspace/FORGE-q35

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES
    "/workspace/FORGE-q35/core/include/"
    "/workspace/FORGE-q35/lane/include/"
    "/workspace/FORGE-q35/spec/include/"
    "/workspace/FORGE-q35/grid/include/"
    "/workspace/FORGE-q35/flow/include/"
    "/workspace/FORGE-q35/pack/include/"
    "/workspace/FORGE-q35/mind/include/"
    "/workspace/FORGE-q35/support/include/"
    "/workspace/FORGE-q35/verify/include/"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libforge.so.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libforge.so.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/workspace/FORGE-q35/build-t/libforge.so.0.1.0"
    "/workspace/FORGE-q35/build-t/libforge.so.1"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libforge.so.0.1.0"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libforge.so.1"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/workspace/FORGE-q35/build-t/libforge.so")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/workspace/FORGE-q35/build-t/libforge_lane_mv.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/workspace/FORGE-q35/build-t/libforge_trust.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/workspace/FORGE-q35/build-t/forge")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge-serve" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge-serve")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge-serve"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/workspace/FORGE-q35/build-t/forge-serve")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge-serve" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge-serve")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/forge-serve")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "runtime" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/FORGE" TYPE FILE FILES
    "/workspace/FORGE-q35/LICENSE"
    "/workspace/FORGE-q35/NOTICE"
    "/workspace/FORGE-q35/COMMERCIAL-LICENCE.md"
    "/workspace/FORGE-q35/SECURITY.md"
    "/workspace/FORGE-q35/README.md"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/workspace/FORGE-q35/build-t/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
if(CMAKE_INSTALL_COMPONENT)
  if(CMAKE_INSTALL_COMPONENT MATCHES "^[a-zA-Z0-9_.+-]+$")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
  else()
    string(MD5 CMAKE_INST_COMP_HASH "${CMAKE_INSTALL_COMPONENT}")
    set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INST_COMP_HASH}.txt")
    unset(CMAKE_INST_COMP_HASH)
  endif()
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/workspace/FORGE-q35/build-t/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
