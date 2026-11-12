// Precompiled plan blobs (graph/aot.hpp) are untrusted input: parse the directory, then rebuild every entry as a
// session would. Rejected blobs raise forge_exception. A blob that is accepted must give a plan whose buffers all lie
// inside its arena; anything else is reported as a finding. The blob's own BLAKE3 is not required here (a mutated
// input never has it), so the fuzzer reaches the checks behind it. Seed: fuzz/corpus/aot (written by test_aot with
// FORGE_AOT_WRITE_SEED=<path>).
#include "fuzz.hpp"
#include <forge/graph/aot.hpp>
#include <stdexcept>
FORGE_FUZZ({
    using namespace forge; using namespace forge::graph;
    std::span<const std::uint8_t> blob(data, size);
    try { aot::image hashed(blob); } catch (const forge_exception&) {}
    aot::image img(blob, false);
    static const std::vector<float> zeros(64, 0.f);
    aot::binder b;
    b.weight = [](const std::string&) { return std::pair<shape_t, const float*>{shape_t{8, 8}, zeros.data()}; };
    for (auto& e : img.entries()) {
        if (e.size > (64u << 10)) continue;
        state_store st; st.capacity = 16; st.page_rows = 1;
        std::unique_ptr<plan> p;
        try { p = aot::instantiate(img, e, b, &st, {}, 64u << 10); } catch (const forge_exception&) { continue; }
        const float* lo = p->arena.data(); const float* hi = lo + p->arena.size();
        for (std::size_t i = 0; i < p->g.size(); ++i) {
            value_id r = value_id(i);
            while (p->mem.slots[r].kind == buffer_slot::alias) r = p->mem.slots[r].root;
            if (p->mem.slots[r].kind != buffer_slot::arena) continue;
            const auto& v = p->views[i];
            if (v.data < lo || v.size() > std::size_t(hi - v.data)) throw std::runtime_error("accepted plan has a buffer outside its arena");
        }
        for (auto& s : p->steps) {
            if (s.out >= p->g.size()) throw std::runtime_error("accepted plan has a step for no node");
            for (auto in : s.in) if (in >= s.out) throw std::runtime_error("accepted plan has a step reading a later value");
            if (s.kind == 0 && !s.low) throw std::runtime_error("accepted plan has a kernel step without a kernel");
        }
    }
})
