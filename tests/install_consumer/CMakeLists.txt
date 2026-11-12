# Out-of-tree consumer of an installed FORGE (the install_consumer ctest; driven by run.cmake).
# Configure with -DCMAKE_PREFIX_PATH=<install prefix>.
cmake_minimum_required(VERSION 3.25)
project(forge_install_consumer LANGUAGES C CXX)

find_package(forge 0.1 CONFIG REQUIRED)
message(STATUS "forge ${forge_VERSION}: include ${FORGE_INCLUDE_DIR}, templates ${FORGE_TEMPLATES_DIR}")

# C++: headers from several layers (os, simd, security, modelfile, inference) and the compiled simd kernels.
add_executable(consumer_cpp consumer.cpp)
target_link_libraries(consumer_cpp PRIVATE forge::inference forge::simd_mv)

# C: forge.h and libforge only.
add_executable(consumer_c consumer.c)
target_link_libraries(consumer_c PRIVATE forge::c)

# run.cmake installs both next to FORGE's own binaries: that puts forge.dll beside them on Windows and lets
# consumer_cpp check that the templates are found relative to the executable.
set_target_properties(consumer_c consumer_cpp PROPERTIES INSTALL_RPATH "$<TARGET_FILE_DIR:forge::c>")
set(CONSUMER_BINDIR bin CACHE STRING "Where FORGE's own binaries live under the prefix")
install(TARGETS consumer_cpp consumer_c RUNTIME DESTINATION ${CONSUMER_BINDIR})
