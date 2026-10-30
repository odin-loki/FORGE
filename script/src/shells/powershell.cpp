// pwsh = PowerShell 7+ (falls back to Windows PowerShell); powershell = Windows PowerShell 5.1 (falls back to pwsh).
#include "shell.hpp"
void forge_shell_register_powershell(forge::script::shell_registry& r) {
    auto argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> { return {exe, "-NoProfile", "-NonInteractive", "-Command", sc}; };
    auto q = [](const std::string& s) { std::string o = "'"; for (char c : s) { o += c; if (c == '\'') o += '\''; } return o + "'"; };
    forge::script::shell_backend p; p.name = "pwsh"; p.aliases = {"ps"}; p.exes = {"pwsh", "powershell"}; p.family = "powershell"; p.golden_tested = true; p.argv = argv; p.quote = q;
    r.add(p);
    forge::script::shell_backend w; w.name = "powershell"; w.aliases = {"powershell5", "winps"}; w.exes = {"powershell", "pwsh"}; w.family = "powershell"; w.argv = argv; w.quote = q;
    w.notes = "Windows PowerShell 5.1 (Windows only; tested via the pwsh fallback elsewhere)";
    r.add(w);
}
