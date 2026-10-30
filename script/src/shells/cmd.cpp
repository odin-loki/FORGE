#include "shell.hpp"
void forge_shell_register_cmd(forge::script::shell_registry& r) {
    forge::script::shell_backend b; b.name = "cmd"; b.exes = {"cmd.exe", "cmd"}; b.family = "cmd"; b.notes = "Windows only (wine in CI)";
    b.argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> { return {exe, "/d", "/s", "/c", sc}; };
    b.quote = [](const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"') o += "\"\""; else if (c == '%') o += "%%"; else o += c; } return o + "\""; };
    r.add(std::move(b));
}
