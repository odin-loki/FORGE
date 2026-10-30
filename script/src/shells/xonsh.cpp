// xonsh: arguments are Python string literals.
#include "shell.hpp"
void forge_shell_register_xonsh(forge::script::shell_registry& r) {
    forge::script::shell_backend b; b.name = "xonsh"; b.exes = {"xonsh"}; b.family = "python"; b.golden_tested = true;
    b.argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> { return {exe, "--no-rc", "-c", sc}; };
    b.quote = [](const std::string& s) { std::string o = "'"; for (char c : s) { if (c == '\'' || c == '\\') o += '\\'; if (c == '\n') { o += "\\n"; continue; } o += c; } return o + "'"; };
    r.add(std::move(b));
}
