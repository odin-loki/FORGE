// Nushell: raw strings r#'...'# need no escaping; use enough '#' to avoid collisions.
#include "shell.hpp"
void forge_shell_register_nushell(forge::script::shell_registry& r) {
    forge::script::shell_backend b; b.name = "nu"; b.aliases = {"nushell"}; b.exes = {"nu"}; b.family = "nu"; b.golden_tested = true;
    b.argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> { return {exe, "--no-config-file", "-c", sc}; };
    b.quote = [](const std::string& s) { std::string h = "#"; while (s.find("'" + h) != std::string::npos) h += '#'; return "r" + h + "'" + s + "'" + h; };
    r.add(std::move(b));
}
