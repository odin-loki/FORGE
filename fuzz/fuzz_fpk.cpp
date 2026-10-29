#include "fuzz.hpp"
#include <forge/pack/sign.hpp>
FORGE_FUZZ({
    forge::pack::reader r(forge::fuzz::as_file(data, size, ".fpk"));
    (void)r.verify();
    (void)forge::pack::verify_signature(r);
    if (r.has_section("manifest")) (void)r.manifest();
    for (auto& [n, e] : r.tensors()) (void)r.load(n);
})
