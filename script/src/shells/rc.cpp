// Plan 9 rc (and Byron Rakitzis' rc), plus es: '' doubling inside single quotes.
#include "shell.hpp"
void forge_shell_register_rc(forge::script::shell_registry& r) {
    for (auto n : {"rc", "es"}) {
        forge::script::shell_backend b; b.name = n; b.exes = {n}; b.family = "rc";
        b.argv = forge::script::shellq::dash_c; b.quote = [](const std::string& s) { return forge::script::shellq::doubled(s, '\''); };
        r.add(std::move(b));
    }
}
