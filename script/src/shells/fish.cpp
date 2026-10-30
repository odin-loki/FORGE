#include "shell.hpp"
void forge_shell_register_fish(forge::script::shell_registry& r) {
    forge::script::shell_backend b; b.name = "fish"; b.exes = {"fish"}; b.family = "fish"; b.golden_tested = true;
    b.argv = forge::script::shellq::dash_c;
    b.quote = [](const std::string& s) { std::string o = "'"; for (char c : s) { if (c == '\'' || c == '\\') o += '\\'; o += c; } return o + "'"; };
    r.add(std::move(b));
}
