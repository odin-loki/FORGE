#include "fuzz.hpp"
#include <forge/modelfile/sign.hpp>
// The .fpk reader on an untrusted file: section table, tensor index, hashes, signature, and both encryption schemes.
// Sealed seeds (fuzz/make_fpk_corpus.cpp) are written with the fixed key below, so mutated ciphertext, tags, headers
// and indexes reach the AEAD paths instead of stopping at "wrong key"; each input is read once without a key (the
// locked state) and once with it.
static const forge::modelfile::model_key& fuzz_key() {
    static const forge::modelfile::model_key k = [] { forge::modelfile::model_key x{}; for (std::size_t i = 0; i < 32; ++i) x[i] = std::uint8_t(0x40 + i); return x; }();
    return k;
}
FORGE_FUZZ({
    auto path = forge::fuzz::as_file(data, size, ".fpk");
    for (int pass = 0; pass < 2; ++pass) {
        try {
            forge::modelfile::reader r(path, pass ? std::optional<forge::modelfile::model_key>(fuzz_key()) : std::nullopt);
            (void)r.verify();
            (void)forge::modelfile::verify_signature(r);
            if (r.has_section("manifest")) { try { (void)r.manifest(); } catch (const forge::forge_exception&) {} }
            for (auto& [n, e] : r.tensors()) {
                try { (void)r.load(n); } catch (const forge::forge_exception&) {}
                try { (void)r.tensor_bytes(n); } catch (const forge::forge_exception&) {}
            }
            (void)r.plaintext_resident();
            r.release_plaintext();
        } catch (const forge::forge_exception&) {}
        if (pass == 0) {   // no encryption section: the second pass would repeat the first
            static const char tag[] = "encryption";
            if (std::string_view(reinterpret_cast<const char*>(data), size).find(tag) == std::string_view::npos) break;
        }
    }
})
