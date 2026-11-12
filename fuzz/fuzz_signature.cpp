#include "fuzz.hpp"
#include <forge/modelfile/bundle.hpp>
#include <forge/security/integrity.hpp>
#include <filesystem>
#include <stdexcept>
// Signature manifests that arrive next to untrusted artefacts: byte 0 even = a bundle's FORGE-SIGNATURE.json
// (modelfile/bundle.hpp verify_bundle, over a small fixed directory), odd = a detached .sig
// (security/integrity.hpp detached_verify). Neither may accept a fuzzed manifest as trusted. (.fpk signature
// sections are covered by fuzz_fpk.)
namespace {
const std::string& bundle_dir() {
    static const std::string d = [] {
        namespace fs = std::filesystem;
        auto p = fs::temp_directory_path() / ("forge_fuzz_bundle_" + std::to_string(FORGE_FUZZ_GETPID()));
        fs::create_directories(p / "sub");
        std::ofstream(p / "config.json", std::ios::binary) << "{\"a\":1}";
        std::ofstream(p / "sub" / "w.bin", std::ios::binary) << "weights";
        return p.string();
    }();
    return d;
}
}
FORGE_FUZZ({
    if (size < 1) return;
    std::string_view text{reinterpret_cast<const char*>(data + 1), size - 1};
    // a key nobody holds: no manifest may verify against it
    static const std::vector<std::string> trusted = {"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"};
    if (data[0] % 2 == 0) {
        std::ofstream(std::filesystem::path(bundle_dir()) / std::string(forge::modelfile::bundle_manifest_name), std::ios::binary | std::ios::trunc) << text;
        auto rep = forge::modelfile::verify_bundle(bundle_dir(), trusted);
        if (rep.ok() || rep.trusted) throw std::logic_error("bundle: fuzzed manifest verified against a key without a private half");
        (void)forge::modelfile::verify_bundle(bundle_dir(), {}, true);
    } else {
        static const std::string payload = "print(1)\n";
        auto bytes = std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
        auto rep = forge::security::detached_verify(bytes, "x.fs", text, trusted);
        if (rep.trusted || rep.st == forge::security::detached_report::status::ok) throw std::logic_error("detached signature: fuzzed text verified against a key without a private half");
        (void)forge::security::detached_verify(bytes, "x.fs", text);
    }
})
