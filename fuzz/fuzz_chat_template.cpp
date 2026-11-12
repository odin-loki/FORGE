#include "fuzz.hpp"
#include <forge/inference/chat_template.hpp>
#include <cstdlib>
// input = Jinja chat template [\x01 messages JSON]. A template comes out of a model file, so it is untrusted:
// whatever it is, rendering has to end with text or a forge_exception within the (reduced) limits below.
FORGE_FUZZ({
    std::string_view s{reinterpret_cast<const char*>(data), size};
    auto cut = s.find('\x01');
    static const forge::json::value dflt = forge::json::parse(
        R"([{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"},)"
        R"({"role":"assistant","content":null,"tool_calls":[{"id":"call00001","type":"function","function":{"name":"f","arguments":{"a":1,"b":[true,null,2.5]}}}]},)"
        R"({"role":"tool","tool_call_id":"call00001","content":"ok"},{"role":"user","content":"<think>\nx\n</think>\n\nBye"}])");
    static const forge::json::value tools = forge::json::parse(R"([{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"integer"}}}}}])");
    forge::inference::chat_options o;
    o.limits.max_steps = 200000; o.limits.max_output = 1u << 18; o.limits.max_template = 1u << 16;
    o.bos_token = "<s>"; o.eos_token = "</s>"; o.tools = tools; o.now = 1700000000;
    o.add_generation_prompt = size % 2 == 0;
    forge::inference::chat_template t(s.substr(0, cut), o.limits);
    auto out = t.render(cut == std::string_view::npos ? dflt : forge::json::parse(s.substr(cut + 1)), o);
    if (out.size() > o.limits.max_output) std::abort();
})
