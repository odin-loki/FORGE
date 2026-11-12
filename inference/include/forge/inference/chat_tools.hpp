#pragma once
// Chat-model helpers next to the chat template (chat_template.hpp), used by forge-serve, `forge run --chat` and the
// C ABI:
//
//   end_of_turn_tokens(model)   the token ids that end an assistant turn: EOS, the manifest's "eos_token_ids" (what the
//                               importers record from GGUF eos / eot / eom ids and HF generation_config.json), and the
//                               end-of-turn markers the model's chat template writes (<|im_end|>, <|eot_id|>, ...)
//   detect_tool_format(tmpl)    which tool-call syntax the model was trained on, read off its chat template
//   tool_call_grammar(...)      a GBNF grammar that forces a tool call in that syntax (tool_choice required / named)
//   tool_call_parser            finds tool calls in generated text, incrementally (streaming) or at the end
//
// Formats:  hermes   <tool_call>\n{"name": "f", "arguments": {...}}\n</tool_call>      (Hermes, Qwen, ChatML tool models)
//           llama3   <|python_tag|>{"name": "f", "parameters": {...}}  or the bare JSON object (Llama 3.1 / 3.2)
//           mistral  [TOOL_CALLS][{"name": "f", "arguments": {...}}, ...]
//           generic  {"name": "f", "arguments": {...}} as the whole reply (templates with no tool syntax of their own)
// Model output is untrusted text: the scanner is iterative (no recursion), reads at most the text it is given and
// never throws on malformed JSON (an unfinished or broken call is simply not reported as done).
#include <forge/inference/json_schema.hpp>
#include <forge/json/json.hpp>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace forge::inference {

// ---------------------------------------------------------------------------------------------- end of turn
namespace detail_chat {
inline constexpr std::string_view turn_end_markers[] = {
    "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>", "<|end_of_text|>", "</s>",
    "<|eot|>", "<|END_OF_TURN_TOKEN|>", "<\xEF\xBD\x9C" "end\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>", "<|return|>"};
}
// `chat` false: only what ends any generation (EOS and the manifest's list), for raw completions.
template <class Model> std::vector<int> end_of_turn_tokens(const Model& m, bool chat = true) {
    std::vector<int> ids;
    auto add = [&](double d) {
        if (!(d >= 0) || d >= double(m.vocab() > 0 ? m.vocab() : 2147483647) || ids.size() >= 64) return;
        int id = int(d);
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    };
    auto* tok = m.tokenizer();
    if (tok) add(tok->eos());
    if (auto* l = m.manifest().find("eos_token_ids"); l && l->is_arr())
        for (auto& x : l->arr()) if (x.is_num()) add(x.num());
    if (chat && tok && !m.chat_template().empty())
        for (auto mk : detail_chat::turn_end_markers)
            if (int id = tok->token_id(mk); id >= 0 && m.chat_template().find(mk) != std::string::npos) add(id);
    return ids;
}

// ---------------------------------------------------------------------------------------------- tool calls
enum class tool_format { generic, hermes, llama3, mistral };
inline const char* tool_format_name(tool_format f) { return f == tool_format::hermes ? "hermes" : f == tool_format::llama3 ? "llama3" : f == tool_format::mistral ? "mistral" : "generic"; }
inline tool_format detect_tool_format(std::string_view tmpl) {
    auto has = [&](std::string_view n) { return tmpl.find(n) != std::string_view::npos; };
    if (has("<tool_call>")) return tool_format::hermes;
    if (has("[TOOL_CALLS]")) return tool_format::mistral;
    if (has("<|python_tag|>") || has("<|eom_id|>") || (has("<|start_header_id|>") && has("tools"))) return tool_format::llama3;
    return tool_format::generic;
}

// The declared functions of an OpenAI "tools" array: [{"type": "function", "function": {"name", "parameters"}}].
// Entries of another type are skipped. Raises on a malformed list (it comes from a request).
struct tool_decl { std::string name; const json::value* parameters = nullptr; };
inline std::vector<tool_decl> tool_decls(const json::value& tools) {
    std::vector<tool_decl> out;
    if (tools.is_null()) return out;
    if (!tools.is_arr()) raise(errc::invalid_argument, "\"tools\" must be an array");
    if (tools.arr().size() > 128) raise(errc::invalid_argument, "\"tools\": at most 128 tools, got {}", tools.arr().size());
    for (auto& t : tools.arr()) {
        if (!t.is_obj()) raise(errc::invalid_argument, "\"tools\": every entry must be an object");
        if (t.str_or("type", "function") != "function") continue;
        auto* f = t.find("function");
        if (!f || !f->is_obj() || !f->find("name") || !(*f)["name"].is_str() || (*f)["name"].str().empty())
            raise(errc::invalid_argument, "\"tools\": a function tool needs \"function\": {{\"name\": ...}}");
        auto& name = (*f)["name"].str();
        if (name.size() > 256) raise(errc::invalid_argument, "\"tools\": function name longer than 256 bytes");
        for (auto& d : out) if (d.name == name) raise(errc::invalid_argument, "\"tools\": function \"{}\" is declared twice", name);
        auto* p = f->find("parameters");
        out.push_back({name, p && p->is_obj() ? p : nullptr});
    }
    return out;
}

// GBNF forcing tool calls in format `f` (compact JSON). `only`: the one function that may be called ("" = any declared
// one); `parallel`: one or more calls instead of exactly one (hermes and mistral; the other formats carry one call).
// The arguments follow each function's "parameters" JSON Schema (json_schema.hpp); no schema = any JSON object.
inline std::string tool_call_grammar(tool_format f, const json::value& tools, const std::string& only = {}, bool parallel = false) {
    auto decls = tool_decls(tools);
    std::string rules, any;
    const char* args_key = f == tool_format::llama3 ? "parameters" : "arguments";
    static const json::value any_object = json::parse(R"({"type":"object"})");
    int i = 0;
    for (auto& d : decls) {
        if (!only.empty() && d.name != only) continue;
        std::string args = json_schema_rules(d.parameters ? *d.parameters : any_object, std::format("t{}x", i), rules);
        std::string head = "{\"name\":" + json::dump(json::value(d.name)) + ",\"" + args_key + "\":";
        rules += std::format("tcall{} ::= {} {} \"}}\"\n", i, detail_schema::lit(head), args);
        any += (any.empty() ? "" : " | ") + std::format("tcall{}", i);
        ++i;
    }
    if (any.empty() && only.empty()) raise(errc::invalid_argument, "tool_choice needs at least one function in \"tools\"");
    if (any.empty()) raise(errc::invalid_argument, "tool_choice names a function that is not in \"tools\": {}", only);
    std::string root;
    if (f == tool_format::hermes) {
        root = "\"<tool_call>\\n\" anycall \"\\n</tool_call>\"";
        if (parallel) root += " (\"\\n<tool_call>\\n\" anycall \"\\n</tool_call>\")*";
    } else if (f == tool_format::mistral) root = parallel ? "\"[TOOL_CALLS][\" anycall (\",\" anycall)* \"]\"" : "\"[TOOL_CALLS][\" anycall \"]\"";
    else root = "anycall";
    return "root ::= " + root + "\nanycall ::= " + any + "\n" + rules + json_schema_base_rules();
}

struct tool_call {
    std::string name;
    std::string arguments;   // JSON text as the model wrote it (so far, while streaming)
    bool named = false;      // the name is known
    bool done = false;       // the call's JSON object is complete
};

// Finds the tool calls in the text generated so far. update() takes the whole text each time (it rescans the call
// region only); content_end() is how much of the text is ordinary reply content that can be shown already.
class tool_call_parser {
public:
    // names: the declared functions. A bare JSON object (llama3 / generic) only counts as a call to one of them.
    tool_call_parser(tool_format f, std::vector<std::string> names) : f_(f), names_(std::move(names)) {}
    tool_format format() const { return f_; }
    const std::vector<tool_call>& calls() const { return calls_; }
    bool in_call() const { return state_ == st::calls; }
    std::size_t content_end() const { return content_end_; }

    void update(std::string_view text, bool final) {
        if (state_ == st::content_only) { content_end_ = text.size(); return; }
        if (state_ == st::search) {
            std::string_view mk = marker();
            if (!mk.empty()) {
                // the marker can only start in the unseen tail (minus a marker length, for one split over updates)
                std::size_t from = seen_ > mk.size() ? seen_ - mk.size() : 0;
                if (auto p = text.find(mk, from); p != std::string_view::npos) { state_ = st::calls; start_ = p; content_end_ = p; }
            }
            if (state_ == st::search && bare_ok() && !bare_rejected_) {
                std::size_t p = 0;
                while (p < text.size() && is_ws(text[p])) ++p;
                if (p < text.size() && text[p] == '{') {
                    auto o = scan_object(text, p);
                    if (o.named && declared(o.name)) { state_ = st::calls; start_ = p; content_end_ = 0; }
                    else if (o.named || o.complete || final) bare_rejected_ = true;
                    else { content_end_ = 0; seen_ = text.size(); return; }   // undecided: hold the text back
                } else if (p < text.size()) bare_rejected_ = true;
                else if (!final) { content_end_ = 0; seen_ = text.size(); return; }   // only blanks so far
            }
            seen_ = text.size();
            if (state_ == st::search) {
                if (mk.empty() && bare_rejected_) state_ = st::content_only;
                // keep back a tail that could be the beginning of the marker
                std::size_t hold = 0;
                if (!final && !mk.empty())
                    for (std::size_t k = std::min(mk.size() - 1, text.size()); k > 0; --k)
                        if (text.substr(text.size() - k) == mk.substr(0, k)) { hold = k; break; }
                content_end_ = text.size() - hold;
                return;
            }
        }
        // the call region: objects separated by blanks, list punctuation and the format's markers
        calls_.clear();
        std::size_t i = start_;
        while (i < text.size() && calls_.size() < 64) {
            if (is_ws(text[i]) || text[i] == ',' || text[i] == ';' || text[i] == ']') { ++i; continue; }
            bool skipped = false;
            for (std::string_view mk : {std::string_view("<tool_call>"), std::string_view("</tool_call>"), std::string_view("<|python_tag|>"), std::string_view("[TOOL_CALLS]")})
                if (text.substr(i, mk.size()) == mk) { i += mk.size(); skipped = true; break; }
            if (skipped) continue;
            if (text[i] == '[') { ++i; continue; }
            if (text[i] != '{') break;   // anything else ends the calls (ignored)
            auto o = scan_object(text, i);
            tool_call c; c.name = std::move(o.name); c.named = o.named; c.done = o.complete;
            if (o.has_args) c.arguments = std::string(text.substr(o.ab, o.ae - o.ab));
            calls_.push_back(std::move(c));
            if (!o.complete) break;
            i = o.end;
        }
    }
    // After the last update(text, true): the finished calls with their arguments as JSON object text ("{}" when the
    // model gave none; a JSON string that itself holds JSON is unwrapped). Unfinished or unnamed calls are dropped.
    std::vector<tool_call> finished() const {
        std::vector<tool_call> out;
        for (auto& c : calls_) {
            if (!c.done || !c.named) continue;
            tool_call r = c;
            if (r.arguments.empty()) r.arguments = "{}";
            else if (r.arguments[0] == '"') {
                try { auto v = json::parse(r.arguments); if (v.is_str()) { auto in = json::parse(v.str()); if (in.is_obj()) r.arguments = v.str(); } }
                catch (const std::exception&) {}
            }
            out.push_back(std::move(r));
        }
        return out;
    }

    // One JSON object starting at text[i] == '{', possibly cut off: its "name" string and the span of its "arguments"
    // (or "parameters") value. No recursion; strings and escapes are honoured.
    struct object_scan { bool complete = false, named = false, has_args = false; std::size_t end = 0, ab = 0, ae = 0; std::string name; };
    static object_scan scan_object(std::string_view s, std::size_t i) {
        object_scan r;
        enum { want_key, want_colon, want_value, in_value, after_value } st = want_key;
        std::size_t depth = 0, str_b = 0; bool in_str = false, esc = false, args_open = false;
        int which = 0;   // the value being read at depth 1: 1 = name, 2 = arguments
        std::string key;
        auto unquote = [&](std::size_t b, std::size_t e) {
            try { auto v = json::parse(s.substr(b, e - b)); return v.is_str() ? v.str() : std::string(); } catch (const std::exception&) { return std::string(); }
        };
        auto end_value = [&](std::size_t at) { if (which == 2 && args_open) { r.ae = at; args_open = false; } st = after_value; };
        for (std::size_t j = i; j < s.size(); ++j) {
            char c = s[j];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') {
                    in_str = false;
                    if (depth == 1 && st == want_key) { key = unquote(str_b, j + 1); st = want_colon; }
                    else if (depth == 1 && st == in_value) {
                        if (which == 1) { r.name = unquote(str_b, j + 1); r.named = !r.name.empty(); }
                        end_value(j + 1);
                    }
                }
                continue;
            }
            if (c == '"') {
                in_str = true; str_b = j;
                if (depth == 1 && st == want_value) { st = in_value; if (which == 2) { r.ab = j; r.has_args = args_open = true; } }
                continue;
            }
            if (c == '{' || c == '[') {
                if (depth == 1 && st == want_value) { st = in_value; if (which == 2) { r.ab = j; r.has_args = args_open = true; } }
                if (++depth == 1) st = want_key;
                continue;
            }
            if (c == '}' || c == ']') {
                if (depth == 1) { if (st == in_value) end_value(j); r.complete = true; r.end = j + 1; break; }   // a scalar ran up to the brace
                if (--depth == 1 && st == in_value) end_value(j + 1);
                continue;
            }
            if (depth != 1) continue;
            if (c == ':' && st == want_colon) { st = want_value; which = key == "name" ? 1 : (key == "arguments" || key == "parameters") && !r.has_args ? 2 : 0; }
            else if (c == ',') { if (st == in_value) end_value(j); st = want_key; }
            else if (!is_ws(c) && st == want_value) { st = in_value; if (which == 2) { r.ab = j; r.has_args = args_open = true; } }   // number, true, null
        }
        if (args_open) r.ae = s.size();
        return r;
    }
private:
    enum class st { search, calls, content_only };
    static bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
    std::string_view marker() const { return f_ == tool_format::hermes ? "<tool_call>" : f_ == tool_format::mistral ? "[TOOL_CALLS]" : f_ == tool_format::llama3 ? "<|python_tag|>" : ""; }
    bool bare_ok() const { return f_ == tool_format::llama3 || f_ == tool_format::generic; }
    bool declared(const std::string& n) const { return std::find(names_.begin(), names_.end(), n) != names_.end(); }
    tool_format f_;
    std::vector<std::string> names_;
    st state_ = st::search;
    std::size_t start_ = 0, seen_ = 0, content_end_ = 0;
    bool bare_rejected_ = false;
    std::vector<tool_call> calls_;
};

// OpenAI clients send an assistant message's tool_calls[].function.arguments as a JSON string; chat templates expect
// the object (they print it with tojson). Returns the messages with such strings parsed (left alone when not JSON).
inline json::value normalize_tool_messages(const json::value& messages) {
    if (!messages.is_arr()) return messages;
    json::array out;
    for (auto& m : messages.arr()) {
        auto* tc = m.is_obj() ? m.find("tool_calls") : nullptr;
        if (!tc || !tc->is_arr()) { out.push_back(m); continue; }
        json::object mo;
        for (auto& [k, v] : m.obj()) {
            if (k != "tool_calls") { mo.emplace_back(k, v); continue; }
            json::array calls;
            for (auto& c : v.arr()) {
                auto* fn = c.is_obj() ? c.find("function") : nullptr;
                auto* a = fn && fn->is_obj() ? fn->find("arguments") : nullptr;
                if (!a || !a->is_str()) { calls.push_back(c); continue; }
                json::value parsed;
                try { parsed = json::parse(a->str()); } catch (const std::exception&) { calls.push_back(c); continue; }
                json::object co;
                for (auto& [ck, cv] : c.obj()) {
                    if (ck != "function") { co.emplace_back(ck, cv); continue; }
                    json::object fo;
                    for (auto& [fk, fv] : cv.obj()) fo.emplace_back(fk, fk == "arguments" ? parsed : fv);
                    co.emplace_back(ck, json::value(std::move(fo)));
                }
                calls.push_back(json::value(std::move(co)));
            }
            mo.emplace_back(k, json::value(std::move(calls)));
        }
        out.push_back(json::value(std::move(mo)));
    }
    return json::value(std::move(out));
}

} // namespace forge::inference
