#include "fuzz.hpp"
#include <forge/vision/codec.hpp>
#include <cstdlib>
FORGE_FUZZ({
    auto r = forge::vision::decode_pnm({data, size}, {.max_pixels = std::size_t(1) << 20});
    auto again = forge::vision::decode_pnm(forge::vision::encode_pnm(r));
    if (again.w != r.w || again.h != r.h || again.c != r.c || again.depth != r.depth || again.u8 != r.u8 || again.u16 != r.u16) std::abort();
})
