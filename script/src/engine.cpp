// forge::script engine: public API, REPL and Ctrl-C handling.
#include <forge/script/script.hpp>
#include "interp.hpp"
#include "parser.hpp"
#include "vm.hpp"
#include <forge/os/os.hpp>
#include <csignal>
#include <filesystem>
#include <format>
#include <iostream>

namespace forge::script {

struct engine::impl {
    std::shared_ptr<Shared> sh = std::make_shared<Shared>();
    std::unique_ptr<Interp> in;
    void fresh() { in = std::make_unique<Interp>(sh); in->file = "<repl>"; }
    result fail(std::exception_ptr ep) {
        result r; r.ok = false;
        ScriptError e = in->from_cpp(ep);
        if (auto* i = as<Instance>(e.v)) {
            r.error_type = i->cls->name;
            if (auto it = i->fields.find("line"); it != i->fields.end()) if (auto* l = std::get_if<std::int64_t>(&it->second)) r.line = int(*l);
            r.error = in->error_message(e.v);
            if (auto it = i->fields.find("trace"); it != i->fields.end()) { auto t = in->str(it->second); if (!t.empty()) r.error += "\n" + t; }
        } else { r.error_type = "Error"; r.error = "uncaught: " + in->repr(e.v); }
        if (!r.line) r.line = in->line;
        in->depth = 0; in->frames.clear();
        return r;
    }
    result exec(std::string_view src, const std::string& name, bool vm, bool want_value) {
        result r;
        std::shared_ptr<const FnDecl> top;
        try { top = parse_program(src, name); }
        catch (const syntax_error& e) { r.ok = false; r.exit_code = 2; r.error_type = "SyntaxError"; r.line = e.line; r.error = std::string("SyntaxError: ") + e.what(); return r; }
        sh->keep.push_back(top);
        sh->start_ns = os::monotonic_ns();
        in->deadline_ns = sh->limits.timeout_ms ? sh->start_ns + sh->limits.timeout_ms * 1000000 : 0;
        sh->interrupt = false;
        std::string saved = in->file; in->file = name;
        try {
            Value v = in->run_top(top, vm);
            in->run_finalizers();   // destructors of main()'s locals
            if (top->has_main) { if (auto* c = std::get_if<std::int64_t>(&v)) r.exit_code = int(*c); }
            else if (want_value && !is_nil(v)) r.value = in->repr(v);
        } catch (...) {
            r = fail(std::current_exception()); r.exit_code = 1;
            if (r.error_type == "SystemExit") {   // std::exit(n)
                auto p = r.error.find(':'); std::string n = p == std::string::npos ? r.error : r.error.substr(p + 1);
                try { r.exit_code = std::stoi(n); r.ok = true; r.error.clear(); r.error_type.clear(); } catch (...) {}
            }
        }
        if (name != "<repl>") in->file = saved;
        return r;
    }
};

namespace { std::atomic<bool>* g_interrupt = nullptr; extern "C" void on_sigint(int) { if (g_interrupt) g_interrupt->store(true); } }

engine::engine(options o) : p_(std::make_unique<impl>()) {
    auto& sh = *p_->sh;
    sh.limits.max_steps = o.max_steps; sh.limits.timeout_ms = o.timeout_ms; sh.limits.heap_bytes = o.heap_bytes; sh.limits.max_depth = o.max_depth;
    sh.caps = Caps{o.allow_fs, o.allow_net, o.allow_proc, o.allow_env};
    sh.use_vm = o.use_vm; sh.jit_threshold = std::max(1, o.jit_threshold);
    Heap::instance().debug = o.debug; debug::stall_ms = o.deadlock_ms;
    sh.argv = o.argv;
    script_argv() = o.argv;
    if (o.out) sh.out = o.out;
    sh.ai = o.ai; sh.confirm = o.confirm;
    if (o.heap_bytes) Heap::instance().set_soft_cap(Heap::instance().live_bytes() + o.heap_bytes);
    install_builtins(sh);
    auto l = make<List>(); for (auto& a : o.argv) l->push(obj(make<String>(a)));
    sh.builtins["args"] = obj(l);
    p_->fresh();
}
engine::~engine() { { std::lock_guard g(p_->sh->cb_m); p_->sh->callbacks.clear(); } p_->in.reset(); p_->sh->modules.clear(); Heap::instance().collect(); }

result engine::run_file(const std::string& path) {
    auto src = os::read_file(path);
    if (!src) { result r; r.ok = false; r.error_type = "IOError"; r.error = "IOError: " + src.err.what; return r; }
    std::error_code ec;
    auto abs = std::filesystem::weakly_canonical(path, ec);
    p_->sh->base_dir = (ec ? std::filesystem::path(path) : abs).parent_path().string();
    std::string s = src.value;
    if (s.starts_with("#!")) { auto i = s.find('\n'); s = i == std::string::npos ? "" : "#" + s.substr(i); }   // keep line numbers
    return p_->exec(s, ec ? path : abs.string(), p_->sh->use_vm, false);
}
result engine::run(std::string_view src, const std::string& name) { return p_->exec(src, name, p_->sh->use_vm, false); }
result engine::eval(std::string_view src) { return p_->exec(src, "<repl>", false, true); }
void engine::interrupt() { p_->sh->interrupt = true; }
void engine::reset() { p_->fresh(); Heap::instance().collect(); }

std::string engine::disassemble(std::string_view src) {
    try {
        auto top = parse_program(src, "<dis>");
        std::string out = script::disassemble(*compile(*top));
        std::function<void(const NodeP&)> walk = [&](const NodeP& n) {
            if (!n) return;
            if (n->fn && (n->k == NK::FnStmt || n->k == NK::Lambda)) { out += "\n" + script::disassemble(*compile(*n->fn)); walk(n->fn->body); }
            if (n->cls) for (auto& m : n->cls->methods) { out += "\n" + script::disassemble(*compile(*m)); walk(m->body); }
            for (auto& k : n->kids) walk(k);
            walk(n->extra);
        };
        walk(top->body);
        return out;
    } catch (const std::exception& e) { return std::string("error: ") + e.what() + "\n"; }
}

std::string engine::type_of(std::string_view expr) {
    try {
        auto top = parse_program(expr, "<type>");
        p_->sh->keep.push_back(top);
        Value v = p_->in->run_top(top, false);
        if (auto* i = as<Instance>(v)) return i->cls->name;
        if (as<Function>(v) || as<Native>(v) || as<Bound>(v)) return "fn";
        return Interp::type_of(v);
    } catch (...) { return p_->fail(std::current_exception()).error; }
}

std::string engine::stats() {
    auto& h = Heap::instance();
    return std::format("heap {} bytes, {} objects, {} collections ({} objects collected), steps {}",
                       h.live_bytes(), h.live_objects(), h.collections(), h.collected(), p_->sh->steps.load());
}

int engine::repl() {
    g_interrupt = &p_->sh->interrupt;
    std::signal(SIGINT, on_sigint);
    std::cout << "forge script — :help for commands, :quit to leave\n";
    std::string buf, line;
    for (;;) {
        std::cout << (buf.empty() ? "> " : ". ") << std::flush;
        if (!std::getline(std::cin, line)) { std::cout << "\n"; break; }
        p_->sh->interrupt = false;
        if (buf.empty() && line.starts_with(":")) {
            auto sp = line.find(' ');
            std::string cmd = line.substr(0, sp), arg = sp == std::string::npos ? "" : line.substr(sp + 1);
            if (cmd == ":quit" || cmd == ":q" || cmd == ":exit") break;
            else if (cmd == ":help") std::cout << ":type <expr>  type of an expression\n:dis <code>   bytecode for code\n:mem          heap statistics\n:gc           run the cycle collector\n:reset        clear all globals\n:vm on|off    compile hot functions\n:quit         leave\n";
            else if (cmd == ":type") std::cout << type_of(arg) << "\n";
            else if (cmd == ":dis") std::cout << disassemble(arg);
            else if (cmd == ":mem") std::cout << stats() << "\n";
            else if (cmd == ":gc") std::cout << "collected " << Heap::instance().collect() << " objects\n";
            else if (cmd == ":reset") { reset(); std::cout << "globals cleared\n"; }
            else if (cmd == ":vm") { p_->sh->use_vm = arg != "off"; std::cout << "vm " << (p_->sh->use_vm ? "on" : "off") << "\n"; }
            else std::cout << "unknown command " << cmd << " (:help)\n";
            continue;
        }
        buf += line; buf += "\n";
        try { parse_program(buf, "<repl>"); }
        catch (const syntax_error& e) { if (e.incomplete && !line.empty()) continue; }
        auto r = eval(buf);
        buf.clear();
        if (!r.ok) std::cout << r.error << "\n";
        else if (!r.value.empty()) std::cout << r.value << "\n";
    }
    std::signal(SIGINT, SIG_DFL);
    g_interrupt = nullptr;
    return 0;
}

void engine::define(const std::string& name, int min_args, int max_args, host_fn fn) {
    p_->sh->builtins[name] = host_native(*p_->sh, name, min_args, max_args, std::move(fn));
}
result engine::call(std::uint64_t id, const std::string& args_json) {
    Value f;
    { std::lock_guard g(p_->sh->cb_m); auto it = p_->sh->callbacks.find(id); if (it != p_->sh->callbacks.end()) f = it->second; }
    result r;
    if (is_nil(f)) { r.ok = false; r.error_type = "ValueError"; r.error = "ValueError: no callback " + std::to_string(id); return r; }
    auto& in = *p_->in;
    const int depth = in.depth; auto frames = in.frames; const int line = in.line;
    const bool nested = depth > 0 || !frames.empty();
    if (!nested) { p_->sh->interrupt = false; }
    try {
        Value args = host_args(args_json);
        std::vector<Value> a = as<List>(args)->items;
        Value v = in.call(f, a);
        if (!is_nil(v)) r.value = in.repr(v);
    } catch (...) {
        r = p_->fail(std::current_exception());
        in.depth = depth; in.frames = std::move(frames);
    }
    in.line = line;
    return r;
}
void engine::release(std::uint64_t id) { std::lock_guard g(p_->sh->cb_m); p_->sh->callbacks.erase(id); }
std::size_t engine::callbacks() const { std::lock_guard g(p_->sh->cb_m); return p_->sh->callbacks.size(); }

} // namespace forge::script
