// csh / tcsh. Single quotes stop everything except history '!' (escaped) and newlines (backslash-continued).
#include "shell.hpp"
void forge_shell_register_csh(forge::script::shell_registry& r) {
    auto q = [](const std::string& s) { std::string o = "'"; for (char c : s) { if (c == '\'') o += "'\\''"; else if (c == '!') o += "\\!"; else if (c == '\n') o += "\\\n"; else o += c; } return o + "'"; };
    for (auto [n, e, t] : {std::tuple{"tcsh", "tcsh", true}, std::tuple{"csh", "csh", false}}) {
        forge::script::shell_backend b; b.name = n; b.exes = {e}; if (std::string(n) == "csh") b.exes.push_back("tcsh");
        b.family = "csh"; b.golden_tested = t; b.quote = q; b.notes = std::string(n) == "csh" ? "falls back to tcsh" : "";
        b.argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> { return {exe, "-f", "-c", sc}; };
        r.add(std::move(b));
    }
}
