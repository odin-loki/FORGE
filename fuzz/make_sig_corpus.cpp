// Writes the seed corpus of fuzz_signature and fuzz_policy with the code under test, so the seeds are valid
// documents (real BLAKE3 digests, real Ed25519 signatures by a fixed throwaway key, every policy profile):
//   cmake --build <build> --target fuzz_make_sig_corpus && <build>/fuzz_make_sig_corpus fuzz/corpus
// The bundle seeds describe the directory fuzz_signature.cpp builds (config.json, sub/w.bin); the detached
// seeds sign its payload "print(1)\n" as x.fs. regress_* files are left alone.
#include <forge/modelfile/bundle.hpp>
#include <forge/security/integrity.hpp>
#include <forge/security/policy.hpp>
#include <cstdio>
#include <filesystem>
#include <fstream>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    using namespace forge;
    if (argc != 2) { std::fprintf(stderr, "usage: fuzz_make_sig_corpus <corpus dir>\n"); return 2; }
    fs::path out = argv[1];
    fs::create_directories(out / "signature"); fs::create_directories(out / "policy");
    auto put = [&](const char* target, const std::string& name, const std::string& data) {
        std::ofstream(out / target / name, std::ios::binary | std::ios::trunc) << data;
    };
    std::array<std::uint8_t, 32> seed{}, seed2{};
    for (std::size_t i = 0; i < 32; ++i) { seed[i] = std::uint8_t(i + 1); seed2[i] = std::uint8_t(0xA0 + i); }
    auto k1 = modelfile::keypair_from_seed(seed), k2 = modelfile::keypair_from_seed(seed2);

    // ---- bundle manifests (first byte 0 = verify_bundle in the harness)
    auto dir = fs::temp_directory_path() / "forge_make_sig_corpus";
    fs::remove_all(dir); fs::create_directories(dir / "sub");
    std::ofstream(dir / "config.json", std::ios::binary) << "{\"a\":1}";
    std::ofstream(dir / "sub" / "w.bin", std::ios::binary) << "weights";
    auto manifest = [&] { return *os::read_file((dir / std::string(modelfile::bundle_manifest_name)).string()); };
    modelfile::sign_bundle(dir.string(), k1);
    std::string one = manifest();
    put("signature", "bundle_signed", std::string(1, '\0') + one);
    modelfile::sign_bundle(dir.string(), k2);
    put("signature", "bundle_two_signatures", std::string(1, '\0') + manifest());
    std::string bad = one; if (auto p = bad.find("\"size\":"); p != std::string::npos) bad.insert(p + 7, "9");
    put("signature", "bundle_wrong_size", std::string(1, '\0') + bad);
    put("signature", "bundle_unsafe_path", std::string(1, '\0') +
        R"({"alg":"ed25519","scheme":"forge-bundle-v1","files":[{"path":"../x","size":1,"blake3":""}],"digest":"","signatures":[]})");
    fs::remove_all(dir);

    // ---- detached signatures (first byte 1 = detached_verify)
    const std::string payload = "print(1)\n";
    auto bytes = std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size());
    std::string sig = security::detached_sign(bytes, "x.fs", seed);
    put("signature", "detached_signed", std::string(1, '\1') + sig);
    put("signature", "detached_two_signatures", std::string(1, '\1') + security::detached_sign(bytes, "x.fs", seed2, sig));
    put("signature", "detached_other_name", std::string(1, '\1') + security::detached_sign(bytes, "y.fs", seed));
    const std::string other = "print(2)\n";
    put("signature", "detached_other_content", std::string(1, '\1') +
        security::detached_sign(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(other.data()), other.size()), "x.fs", seed));

    // ---- policies: every profile's defaults as to_json writes them, and a hand-written partial document
    for (auto p : {security::profile::none, security::profile::strict, security::profile::standard, security::profile::permissive})
        put("policy", std::string("defaults_") + security::name(p) + ".json", security::to_json(security::profile_defaults(p)));
    put("policy", "partial.json", R"({"profile": "standard", "net": {"mode": "loopback"}, "allow_root": false})");
    std::printf("wrote %s/signature and %s/policy\n", argv[1], argv[1]);
    return 0;
}
