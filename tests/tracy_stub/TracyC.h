// A stand-in for Tracy's <tracy/TracyC.h>, declaring only the entry points prof.hpp's FORGE_PROF_TRACY bridge
// calls, with the signatures of Tracy 0.9 / 0.10 as recalled when this was written (NOT copied from Tracy, and
// not checked against a Tracy release). Each call is appended to a log the test inspects. Compile test only.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ___tracy_c_zone_context { std::uint32_t id; int active; };
typedef struct ___tracy_c_zone_context TracyCZoneCtx;

inline std::vector<std::string>& tracy_stub_log() { static std::vector<std::string> l; return l; }
inline std::vector<std::string>& tracy_stub_names() { static std::vector<std::string> n; return n; }

inline std::uint64_t ___tracy_alloc_srcloc_name(std::uint32_t, const char*, std::size_t, const char*, std::size_t, const char* name, std::size_t name_sz) {
    tracy_stub_names().emplace_back(name, name_sz);
    return tracy_stub_names().size();   // 1-based handle
}
inline TracyCZoneCtx ___tracy_emit_zone_begin_alloc(std::uint64_t srcloc, int active) {
    tracy_stub_log().push_back("begin " + tracy_stub_names().at(std::size_t(srcloc - 1)));
    return TracyCZoneCtx{std::uint32_t(srcloc), active};
}
inline void ___tracy_emit_zone_end(TracyCZoneCtx ctx) { tracy_stub_log().push_back("end " + tracy_stub_names().at(ctx.id - 1)); }
inline void ___tracy_emit_plot(const char* name, double val) { tracy_stub_log().push_back(std::string("plot ") + name + " " + std::to_string(val)); }
inline void ___tracy_emit_memory_alloc_named(const void*, std::size_t size, int, const char* name) { tracy_stub_log().push_back(std::string("alloc ") + name + " " + std::to_string(size)); }
inline void ___tracy_emit_memory_free_named(const void*, int, const char* name) { tracy_stub_log().push_back(std::string("free ") + name); }
