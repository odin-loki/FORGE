// forge-script-run: the runner stub of `forge script --aot` embed mode (docs/aot.md). On its own it does nothing;
// `forge script --aot out script.cpp --aot-mode embed` copies it and appends the program, and the copy runs that
// program on the bytecode VM under the policy baked in with it.
#include <forge/script/aot.hpp>
#include <cstdio>

int main(int argc, char** argv) {
    if (auto rc = forge::script::aot_run_embedded(argc, argv)) return *rc;
    std::fprintf(stderr, "forge-script-run: this is the bare runner stub; it carries no program.\n"
                         "Build one with: forge script --aot <out> <script> --aot-mode embed\n");
    return 2;
}
