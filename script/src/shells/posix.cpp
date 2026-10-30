// POSIX-family shells: same -c invocation and '...' quoting.
#include "shell.hpp"
namespace fs = forge::script;
void forge_shell_register_posix(fs::shell_registry& r) {
    struct S { const char* name; std::vector<std::string> aliases, exes; bool tested; const char* notes; };
    const S list[] = {
        {"sh", {}, {"sh"}, true, "system POSIX sh"},
        {"bash", {}, {"bash"}, true, ""},
        {"zsh", {}, {"zsh"}, true, ""},
        {"dash", {}, {"dash"}, true, "Debian Almquist shell"},
        {"ash", {"busybox"}, {"ash", "busybox"}, true, "BusyBox ash (runs `busybox sh` when no ash binary)"},
        {"ksh", {"ksh93"}, {"ksh", "ksh93"}, false, "AT&T ksh93"},
        {"mksh", {}, {"mksh"}, false, "MirBSD Korn shell"},
        {"oksh", {"pdksh"}, {"oksh", "pdksh"}, false, "OpenBSD ksh"},
        {"yash", {}, {"yash"}, true, ""},
        {"posh", {}, {"posh"}, false, "Policy-compliant Ordinary SHell"},
        {"osh", {"oils"}, {"osh"}, false, "Oils, POSIX/bash-compatible mode"},
        {"ysh", {}, {"ysh"}, false, "Oils new shell; accepts '...' literals"},
        {"hush", {}, {"hush"}, false, "BusyBox hush"},
    };
    for (auto& s : list) {
        fs::shell_backend b; b.name = s.name; b.aliases = s.aliases; b.exes = s.exes; b.family = "posix"; b.golden_tested = s.tested; b.notes = s.notes;
        b.quote = fs::shellq::posix;
        b.argv = [](const std::string& exe, const std::string& sc) -> std::vector<std::string> {
            if (exe.ends_with("busybox")) return {exe, "sh", "-c", sc};
            return {exe, "-c", sc};
        };
        r.add(std::move(b));
    }
}
