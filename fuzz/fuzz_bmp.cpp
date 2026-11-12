#include "fuzz.hpp"
#include <forge/vision/codec.hpp>
#include <cstdlib>
FORGE_FUZZ({
    auto r = forge::vision::decode_bmp({data, size}, {.max_pixels = std::size_t(1) << 20});
    auto again = forge::vision::decode_bmp(forge::vision::encode_bmp(r));
    if (again.w != r.w || again.h != r.h || again.c != r.c || again.u8 != r.u8) std::abort();
})
