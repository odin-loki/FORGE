#include "shell.hpp"
void forge_shell_register_elvish(forge::script::shell_registry& r) {
    forge::script::shell_backend b; b.name = "elvish"; b.exes = {"elvish"}; b.family = "elvish"; b.golden_tested = true;
    b.argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> { return {exe, "-norc", "-c", sc}; };
    b.quote = [](const std::string& s) { return forge::script::shellq::doubled(s, '\''); };
    r.add(std::move(b));
}
