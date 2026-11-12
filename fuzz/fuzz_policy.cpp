#include "fuzz.hpp"
#include <forge/security/policy.hpp>
#include <stdexcept>
// Security policy files (security/policy.hpp). Rejections are policy_violation; a policy that parses must
// survive to_json -> parse_policy unchanged.
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    namespace sec = forge::security;
    std::string_view text{reinterpret_cast<const char*>(data), size};
    sec::policy p;
    try { p = sec::parse_policy(text, "fuzz"); }
    catch (const sec::policy_violation&) { return 0; }
    catch (const forge::forge_exception&) { return 0; }
    std::string j = sec::to_json(p);
    sec::policy q;
    try { q = sec::parse_policy(j, "fuzz-roundtrip"); }
    catch (const std::exception& e) { throw std::logic_error(std::string("policy: to_json output does not parse: ") + e.what() + "\n" + j); }
    if (sec::to_json(q) != j) throw std::logic_error("policy: to_json/parse_policy round trip changed the policy\n" + j + "\n" + sec::to_json(q));
    return 0;
}
