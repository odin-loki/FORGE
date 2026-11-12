#pragma once
// Recorded graphs and eviction (docs/gpu.md "Eviction and recorded graphs"). A recorded kernel launch holds device
// buffers by handle. When one of them is evicted, released or re-created, the handle is dead or names other data, so
// every device buffer carries a generation (a number never reused within a runtime) and every recorded buffer
// argument remembers what it referred to: the host view or encoded weight, and the generation it resolved to. A
// replay first checks that each referenced buffer still exists with that generation; if not, the launch (OpenCL,
// Level Zero) or the whole captured graph (CUDA, HIP) is re-recorded from the host views instead of replayed.
#include <cstddef>
#include <cstdint>

namespace forge::gpu {

struct buffer_ref {
    enum kind_t : std::uint8_t { none, tensor, weight };
    std::uint8_t kind = none;
    std::uint8_t mode = 0;          // tensor: 0 read, 1 written, 2 both
    const void* host = nullptr;     // tensor: start of the host view; weight: the encoded bytes on the host
    std::size_t count = 0;          // tensor: floats in the view; weight: bytes
    std::uint64_t gen = 0;          // generation of the device buffer the argument resolved to; 0 = always re-record
};

} // namespace forge::gpu
