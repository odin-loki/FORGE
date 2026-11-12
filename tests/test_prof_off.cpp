// prof with FORGE_PROF_DISABLE: the macros compile to nothing (their arguments are not even evaluated) while
// the functions stay available.
#define FORGE_PROF_DISABLE 1
#include "check.hpp"
#include <forge/prof/hw.hpp>
using namespace forge;

static int side_effects = 0;
static const char* name_with_side_effect() { ++side_effects; return "x"; }

TEST(prof_compile_time_off) {
    auto& r = prof::recorder::get(); r.clear(); r.on = true;
    {
        FORGE_ZONE(name_with_side_effect()); FORGE_COUNTER("c_disabled", ++side_effects); FORGE_GAUGE("g_disabled", ++side_effects);
        FORGE_INSTANT(name_with_side_effect()); FORGE_HW_ZONE(name_with_side_effect());
    }
    CHECK(r.events().empty()); CHECK(side_effects == 0);
    for (auto& c : r.counters()) CHECK(c.name != "c_disabled");
    { prof::zone z("direct"); }   // the classes still work when used directly
    r.on = false;
    CHECK(r.events().size() == 1);
    r.clear();
}
