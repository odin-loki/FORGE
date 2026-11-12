#include "fuzz.hpp"
#include <forge/import/gguf.hpp>
#include <cstdlib>
// The header parser, then every tensor: the size checks of its ggml type (bytes), the transcoders (stored) and the
// block decoder of its codec on whatever bytes the file holds.
FORGE_FUZZ({
    forge::import::gguf_file g(forge::fuzz::as_file(data, size, ".gguf"));
    (void)g.get("general.architecture");
    for (auto& t : g.tensors()) {
        try {
            (void)g.bytes(t);
            auto s = g.stored(t);
            const auto* c = forge::graph::codecs::find(s.codec);
            std::size_t n = std::size_t(g.count(t));
            if (!c || c->bytes(n) != s.bytes.size()) std::abort();   // a stored tensor always matches its codec
            if (n <= (std::size_t(1) << 22)) { std::vector<float> x(n); c->decode(s.bytes.data(), n, x.data()); }
        } catch (const forge::forge_exception&) {}
    }
})
