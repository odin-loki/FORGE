#pragma once
// RPC device: run lowerings on a remote FORGE process over TCP ("rpc:host:port").
// The server (`forge rpc-serve`, or rpc::serve()) executes each op with its own device registry (CPU, Vulkan, CUDA...),
// so a machine without a GPU can borrow one over the network. Every op is still oracle-gated on the client.
// Wire format (little endian): request  = "FRPC" u32 op-name-len name u32 nattrs {key, tag, value} u32 nin {u32 rank, i64 dims, f32 data} u32 rank i64 out-dims
//                              response = u32 status(0 ok) then f32 out data, or u32 len + error text.
// Not optimised: weights travel with every op (no remote residency yet).
#include <forge/flow/kernels.hpp>
#include <cstring>
#include <string>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <forge/os/os.hpp>

namespace forge::flow::rpc {
namespace detail {
inline void wsa() { os::net_init(); }
inline bool send_all(os::socket& s, const void* p, std::size_t n) { return s.send_all(p, n); }
inline bool recv_all(os::socket& s, void* p, std::size_t n) { return s.recv_all(p, n); }
struct writer {
    std::vector<char> b;
    void raw(const void* p, std::size_t n) { auto* c = static_cast<const char*>(p); b.insert(b.end(), c, c + n); }
    void u32(std::uint32_t v) { raw(&v, 4); }
    void i64(std::int64_t v) { raw(&v, 8); }
    void str(std::string_view s) { u32(std::uint32_t(s.size())); raw(s.data(), s.size()); }
    void shape(const shape_t& s) { u32(std::uint32_t(s.size())); for (auto d : s) i64(d); }
};
struct reader {
    os::socket& s; bool ok = true;
    void raw(void* p, std::size_t n) { if (ok) ok = recv_all(s, p, n); }
    std::uint32_t u32() { std::uint32_t v = 0; raw(&v, 4); return v; }
    std::int64_t i64() { std::int64_t v = 0; raw(&v, 8); return v; }
    std::string str(std::size_t cap = 1 << 20) { auto n = u32(); if (n > cap) { ok = false; return {}; } std::string r(n, '\0'); raw(r.data(), n); return r; }
    shape_t shape() { auto r = u32(); shape_t s; if (r > 16) { ok = false; return s; } for (std::uint32_t i = 0; i < r; ++i) s.push_back(i64()); return s; }
};
inline void put_attrs(writer& w, const spec::attrs& a) {
    w.u32(std::uint32_t(a.v.size()));
    for (auto& [k, v] : a.v) {
        w.str(k); w.u32(std::uint32_t(v.index()));
        if (auto* d = std::get_if<double>(&v)) w.raw(d, 8);
        else if (auto* iv = std::get_if<std::vector<i64>>(&v)) { w.u32(std::uint32_t(iv->size())); for (auto x : *iv) w.i64(x); }
        else w.str(std::get<std::string>(v));
    }
}
inline spec::attrs get_attrs(reader& r) {
    spec::attrs a; auto n = r.u32(); if (n > 4096) { r.ok = false; return a; }
    for (std::uint32_t i = 0; i < n && r.ok; ++i) {
        auto k = r.str(); auto tag = r.u32();
        if (tag == 0) { double d; r.raw(&d, 8); a.v[k] = d; }
        else if (tag == 1) { auto m = r.u32(); if (m > (1u << 20)) { r.ok = false; break; } std::vector<i64> iv(m); for (auto& x : iv) x = r.i64(); a.v[k] = iv; }
        else if (tag == 2) a.v[k] = r.str();
        else r.ok = false;
    }
    return a;
}
inline bool valid_shape(const shape_t& s) { i64 n = 1; for (auto d : s) { if (d < 0 || d > (i64(1) << 32)) return false; n *= std::max<i64>(d, 1); if (n > (i64(1) << 32)) return false; } return true; }
inline int op_index(std::string_view name) {
    for (std::size_t i = 0; i < spec::op_count; ++i) if (spec::info(spec::op(i)).name == name) return int(i);
    return -1;
}
} // namespace detail

// Connection to one server; serialises calls (one op in flight).
class client {
public:
    static std::shared_ptr<client> connect(const std::string& host, const std::string& port, std::string* why) {
        detail::wsa();
        int pn = std::atoi(port.c_str());
        auto r = os::socket::connect(host, pn, 10000);
        if (!r) { if (why) *why = "rpc: cannot connect to " + host + ":" + port + " (" + r.err.what + ")"; return nullptr; }
        r->set_nodelay();
        auto c = std::shared_ptr<client>(new client); c->s_ = std::move(r.value); return c;
    }
    std::uint64_t calls() const { return calls_; }
    void run(std::string_view op, std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) {
        std::lock_guard g(m_);
        detail::writer w; w.raw("FRPC", 4); w.str(op); detail::put_attrs(w, a);
        w.u32(std::uint32_t(in.size()));
        for (auto& t : in) { w.shape(t.shape); w.raw(t.data, t.size() * 4); }
        w.shape(out.shape);
        if (!detail::send_all(s_, w.b.data(), w.b.size())) raise(errc::io_error, "rpc: send failed");
        detail::reader r{s_};
        auto st = r.u32();
        if (!r.ok) raise(errc::io_error, "rpc: connection lost");
        if (st != 0) raise(errc::unsupported, "rpc server: {}", r.str());
        r.raw(out.data, out.size() * 4);
        if (!r.ok) raise(errc::io_error, "rpc: short reply");
        ++calls_;
    }
private:
    client() = default;
    os::socket s_; std::mutex m_; std::uint64_t calls_ = 0;
};

// Registry whose every op runs remotely ("rpc.<op>", priority 20). Ops the server lacks a lowering for run there on its reference.
inline std::unique_ptr<registry> rpc_registry(std::shared_ptr<client> c) {
    auto r = registry::make();
    for (std::size_t i = 0; i < spec::op_count; ++i) {
        auto o = spec::op(i); std::string name(spec::info(o).name);
        r->add({"rpc." + name, o, 20, nullptr,
                [c, name](std::span<const tensor_view> in, tensor_view out, const spec::attrs& a) { c->run(name, in, out, a); }});
    }
    return r;
}
// "rpc:host:port" (host may be [v6]).
inline std::unique_ptr<registry> open(const std::string& addr, std::string* why) {
    auto hp = addr; auto colon = hp.rfind(':');
    if (colon == std::string::npos) { if (why) *why = "rpc: want rpc:host:port"; return nullptr; }
    auto host = hp.substr(0, colon), port = hp.substr(colon + 1);
    if (host.size() > 1 && host.front() == '[') host = host.substr(1, host.size() - 2);
    auto c = client::connect(host, port, why);
    return c ? rpc_registry(c) : nullptr;
}

// Server. `reg` (null = CPU) picks the lowering per op exactly like a local plan would.
class server {
public:
    // port 0 = ephemeral; see port(). bind = "127.0.0.1" by default (pass "0.0.0.0" to expose; there is no auth).
    server(registry* reg, int port = 0, const std::string& bind = "127.0.0.1") : reg_(reg ? reg : &registry::cpu()) {
        detail::wsa();
        auto l = os::socket::listen(bind, port, 8);
        if (!l) raise(errc::io_error, "rpc: cannot listen on {}:{}: {}", bind, port, l.err.what);
        ls_ = std::move(l.value); port_ = ls_.local_port();
    }
    ~server() { stop(); }
    int port() const { return port_; }
    std::uint64_t ops() const { return ops_; }
    // Accepts connections until stop(); one thread per connection.
    void run() {
        while (!stop_) {
            auto c = ls_.accept();
            if (!c) { if (stop_) break; continue; }
            auto sp = std::make_shared<os::socket>(std::move(c.value));
            std::lock_guard g(m_);
            workers_.emplace_back([this, sp] { serve_conn(*sp); sp->close(); });
        }
    }
    void start() { th_ = std::thread([this] { run(); }); }
    void stop() {
        if (stop_.exchange(true)) return;
        ls_.shutdown_both(); ls_.close();
        if (th_.joinable()) th_.join();
        std::lock_guard g(m_);
        for (auto& w : workers_) if (w.joinable()) w.detach();   // connections end when their clients close
    }
private:
    void serve_conn(os::socket& s) {
        for (;;) {
            detail::reader r{s};
            char magic[4]; r.raw(magic, 4);
            if (!r.ok) return;
            if (std::memcmp(magic, "FRPC", 4) != 0) return;
            auto name = r.str(256); auto a = detail::get_attrs(r);
            auto n = r.u32(); if (n > 64) return;
            std::vector<tensor> ins; std::vector<tensor_view> views; std::vector<shape_t> shapes;
            for (std::uint32_t i = 0; i < n && r.ok; ++i) {
                auto sh = r.shape(); if (!detail::valid_shape(sh)) return;
                ins.emplace_back(sh); r.raw(ins.back().data.data(), ins.back().data.size() * 4);
            }
            auto os = r.shape(); if (!r.ok || !detail::valid_shape(os)) return;
            for (auto& t : ins) { views.push_back({t.data.data(), t.shape}); shapes.push_back(t.shape); }
            detail::writer w;
            int oi = detail::op_index(name);
            try {
                if (oi < 0) raise(errc::unsupported, "unknown op '{}'", name);
                tensor out(os);
                auto& l = reg_->select(spec::op(oi), shapes, os, a);
                auto* h = l.on_device ? reg_->hooks.get() : nullptr;
                std::lock_guard dg(dev_m_);   // device runtimes are single-stream
                if (h) h->begin_run();
                try { l.fn(views, {out.data.data(), os}, a); if (h) { h->to_host(out.data.data(), out.data.size()); h->end_run(); } }
                catch (...) { if (h) h->end_run(); throw; }
                w.u32(0); w.raw(out.data.data(), out.data.size() * 4); ++ops_;
            } catch (const std::exception& e) { w.b.clear(); w.u32(1); w.str(e.what()); }
            if (!detail::send_all(s, w.b.data(), w.b.size())) return;
        }
    }
    registry* reg_; os::socket ls_; int port_ = 0;
    std::atomic<bool> stop_{false}; std::atomic<std::uint64_t> ops_{0};
    std::thread th_; std::mutex m_, dev_m_; std::vector<std::thread> workers_;
};
} // namespace forge::flow::rpc
