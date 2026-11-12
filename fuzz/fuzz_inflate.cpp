#include "fuzz.hpp"
#include <forge/vision/codec.hpp>
#include <cstdlib>
// zlib_decompress on arbitrary bytes (output capped at 1 MiB), then compress -> decompress of the input itself.
FORGE_FUZZ({
    auto z = forge::vision::zlib_compress(data, size);
    auto back = forge::vision::zlib_decompress(z.data(), z.size(), size);
    if (back.size() != size || (size && std::memcmp(back.data(), data, size) != 0)) std::abort();
    (void)forge::vision::zlib_decompress(data, size, std::size_t(1) << 20);
})
