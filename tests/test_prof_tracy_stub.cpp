// prof's FORGE_PROF_TRACY bridge, compiled against tests/tracy_stub/TracyC.h. This shows the bridge compiles
// and makes the calls in the right order; it says nothing about a real Tracy client (none is in the tree).
#define FORGE_PROF_TRACY 1
#define FORGE_PROF_TRACY_INCLUDE "tracy_stub/TracyC.h"
#include "check.hpp"
#include <forge/prof/prof.hpp>
using namespace forge;

TEST(prof_tracy_bridge_calls_stub) {
    auto& r = prof::recorder::get(); r.clear(); tracy_stub_log().clear();
    int x = 0;
    {   // forwarded whether or not the recorder is on (Tracy does its own gating)
        FORGE_ZONE("outer"); { FORGE_ZONE("inner"); }
        prof::note_alloc("pool", &x, 48); prof::note_free("pool", &x, 48);
    }
    CHECK((tracy_stub_log() == std::vector<std::string>{"begin outer", "begin inner", "end inner", "alloc pool 48", "free pool", "end outer"}));
    CHECK(r.events().empty());
    tracy_stub_log().clear(); r.on = true;
    { FORGE_ZONE("z"); FORGE_COUNTER("n", 5); FORGE_GAUGE("g", 0.5); }
    r.on = false;
    CHECK((tracy_stub_log() == std::vector<std::string>{"begin z", "plot n 5.000000", "plot g 0.500000", "end z"}));
    CHECK(r.events().size() == 3);   // and recorded as usual
    r.clear();
}
