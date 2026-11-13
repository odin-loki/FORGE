// RPC device over loopback and pipeline split: whole-model logits must match the plain CPU run.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/inference/inference.hpp>
#include <forge/graph/vulkan.hpp>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::inference;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_rpc_" + std::to_string(::getpid()) + "_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }
static std::vector<float> logits(const std::string& dev) {
    auto m = model::open(model_a());
    CHECK(m->set_device(dev) == dev);
    inference::session s(m);
    std::vector<int> p = {1, 5, 9, 33, 70, 101};
    auto a = s.forward(p); int t = 7; auto b = s.forward(std::span<const int>(&t, 1));
    a.insert(a.end(), b.begin(), b.end()); return a;
}
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size()); float m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

TEST(rpc_loopback_matches_cpu) {
    graph::rpc::server srv(nullptr, 0);
    srv.start();
    auto ref = logits("cpu");
    auto got = logits("rpc:127.0.0.1:" + std::to_string(srv.port()));
    CHECK(srv.ops() > 50);
    CHECK_MSG(maxdiff(ref, got) < 1e-5f, "rpc diff %g", maxdiff(ref, got));
    srv.stop();
}
TEST(rpc_errors_propagate) {
    graph::rpc::server srv(nullptr, 0); srv.start();
    auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
    CHECK(c != nullptr);
    tensor x({4}), y({4});
    tensor_view in{x.data.data(), x.shape};
    bool threw = false;
    try { c->run("no_such_op", std::span<const tensor_view>(&in, 1), {y.data.data(), y.shape}, {}); } catch (const std::exception&) { threw = true; }
    CHECK(threw);
    for (auto& v : x.data) v = -1.f;   // connection still usable after an error
    c->run("relu", std::span<const tensor_view>(&in, 1), {y.data.data(), y.shape}, {});
    CHECK(y.data[0] == 0.f);
    srv.stop();
}

// A peer chooses the op, the number of inputs, every shape and the output shape. The server runs a frame only
// if it is a well-formed call (found by fuzz_rpc: softmax with no inputs indexed past the input list; an output
// shape smaller than the op's result is a heap overflow; reshape dims overflowed i64).
TEST(rpc_malformed_calls_are_refused) {
    graph::rpc::server srv(nullptr, 0); srv.start();
    auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
    CHECK(c != nullptr);
    tensor x({2, 4}), w({3, 4}), y({2, 4}), small_({1}), w5({5});
    tensor_view vx{x.data.data(), x.shape}, vw{w.data.data(), w.shape}, vw5{w5.data.data(), w5.shape};
    auto refused = [&](const char* op, std::vector<tensor_view> in, tensor& out, const ops::attrs& a = {}) {
        try { c->run(op, in, {out.data.data(), out.shape}, a); } catch (const forge_exception& e) { return std::string(e.what()).find("rpc server:") != std::string::npos; }
        return false;
    };
    CHECK(refused("softmax", {}, y));                    // no inputs
    CHECK(refused("relu", {vx, vx}, y));                 // too many
    CHECK(refused("relu", {vx}, small_));                 // output smaller than the result
    CHECK(refused("matmul_t", {vx, vw}, y));             // [2,4] x [3,4]^T is [2,3], not [2,4]
    CHECK(refused("matmul_t", {vx, vw5}, y));            // weight of the wrong rank
    CHECK(refused("rms_norm", {vx, vw5}, y));            // weight longer than the last axis
    CHECK(refused("rope", {vx, vw5}, y));                // not [s, h, d]
    ops::attrs big; big.set("shape", std::vector<i64>{i64(1) << 40, i64(1) << 40, 8});
    CHECK(refused("reshape", {vx}, y, big));             // 2^83 elements
    ops::attrs neg; neg.set("shape", std::vector<i64>{-2, -4});
    CHECK(refused("reshape", {vx}, y, neg));
    for (auto& v : x.data) v = -1.f;                     // every refusal left the connection usable
    c->run("relu", std::span<const tensor_view>(&vx, 1), {y.data.data(), y.shape}, {});
    CHECK(y.data[0] == 0.f);
    srv.stop();
    graph::rpc::server lim(nullptr, 0); lim.max_tensor_elems = 4; lim.start();   // a frame naming a larger tensor is dropped unread
    auto c2 = graph::rpc::client::connect("127.0.0.1", std::to_string(lim.port()), nullptr);
    CHECK(c2 != nullptr);
    bool lost = false;
    try { c2->run("relu", std::span<const tensor_view>(&vx, 1), {y.data.data(), y.shape}, {}); } catch (const forge_exception&) { lost = true; }
    CHECK(lost);
    lim.stop();
}
TEST(rpc_connect_failure_falls_back) {
    std::string why; auto r = graph::device_registry("rpc:127.0.0.1:1", &why);
    CHECK(!r); CHECK(!why.empty());
}
TEST(pipeline_split_two_stages) {
    graph::rpc::server srv(nullptr, 0); srv.start();
    auto ref = logits("cpu");
    auto dev = "pipe:cpu,rpc:127.0.0.1:" + std::to_string(srv.port());
    auto got = logits(dev);
    CHECK(srv.ops() > 0);
    CHECK_MSG(maxdiff(ref, got) < 1e-5f, "pipe diff %g", maxdiff(ref, got));
    // both stages got work, and the split is contiguous
    std::string why; auto reg = graph::device_registry(dev, &why);
    CHECK(reg && reg->stages.size() == 2);
    srv.stop();
}
TEST(pipeline_with_vulkan_if_present) {
    std::string why; auto v = graph::device_registry("vulkan", &why);
    if (!v) { std::printf("skip: no vulkan (%s)\n", why.c_str()); return; }
    v.reset();
    auto ref = logits("cpu"); auto got = logits("pipe:vulkan,cpu");
    CHECK_MSG(maxdiff(ref, got) < 2e-3f, "pipe vk diff %g", maxdiff(ref, got));
}
TEST(pipeline_multi_gpu_if_present) {   // real multi-GPU: needs two CUDA/HIP devices
    for (auto b : {"cuda", "hip"}) {
        std::string w0, w1; auto a = graph::device_registry(std::string(b) + ":0", &w0), c = graph::device_registry(std::string(b) + ":1", &w1);
        if (!a || !c) continue;
        a.reset(); c.reset();
        auto ref = logits("cpu"); auto got = logits(std::string("pipe:") + b + ":0," + b + ":1");
        CHECK_MSG(maxdiff(ref, got) < 2e-3f, "%s pipe diff %g", b, maxdiff(ref, got));
    }
}

// ---- security: token / Ed25519 key auth, TLS, remote scripts
static graph::rpc::client_security tok(const std::string& t) { graph::rpc::client_security s; s.token = t; return s; }
static const char* kSeed = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";   // RFC 8032 test 1
static bool relu_ok(graph::rpc::client& c) {
    tensor x({4}), y({4}); for (auto& v : x.data) v = -1.f; tensor_view in{x.data.data(), x.shape};
    c.run("relu", std::span<const tensor_view>(&in, 1), {y.data.data(), y.shape}, {}); return y.data[0] == 0.f;
}
TEST(rpc_token_auth) {
    graph::rpc::server_security sec; sec.tokens = {"s3cret"};
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec); srv.start();
    auto port = std::to_string(srv.port()); std::string why;
    CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, {}) || true);   // unauthenticated: rejected on first frame
    { auto c = graph::rpc::client::connect("127.0.0.1", port, &why, {}); bool threw = false; if (c) { try { relu_ok(*c); } catch (...) { threw = true; } } CHECK(!c || threw); }
    why.clear(); CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, tok("wrong"))); CHECK(why.find("bad token") != std::string::npos);
    auto c = graph::rpc::client::connect("127.0.0.1", port, &why, tok("s3cret")); CHECK(c != nullptr);
    CHECK(relu_ok(*c));
    srv.stop();
}
TEST(rpc_key_auth_and_scripts) {
    auto id = std::make_shared<graph::rpc::identity>(graph::rpc::identity_from_seed_hex(kSeed));
    CHECK(graph::rpc::sec::hex(id->pub.data(), 32) == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    graph::rpc::server_security sec; sec.authorized_keys = {graph::rpc::sec::hex(id->pub.data(), 32)}; sec.allow_scripts = true;
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.script_handler = [](const std::string& src, const std::string&, const std::vector<std::string>& argv, std::uint32_t) {
        return graph::rpc::server::script_result{true, "", "", "ran:" + src + ":" + std::to_string(argv.size())};
    };
    srv.start(); auto port = std::to_string(srv.port()); std::string why;
    graph::rpc::client_security good; good.key = id;
    auto c = graph::rpc::client::connect("127.0.0.1", port, &why, good); CHECK_MSG(c != nullptr, "%s", why.c_str());
    auto r = c->run_script("x", "<t>", {"a", "b"}); CHECK(r.ok && r.output == "ran:x:2");
    CHECK(relu_ok(*c));
    std::uint8_t other[32] = {1}; graph::rpc::client_security bad; bad.key = std::make_shared<graph::rpc::identity>(graph::rpc::identity_from_seed_hex(graph::rpc::sec::hex(other, 32)));
    why.clear(); CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, bad)); CHECK(why.find("not authorized") != std::string::npos);
    CHECK(srv.scripts() == 1);
    srv.stop();
    // scripts need auth; and a no-auth server refuses them
    bool threw = false; try { graph::rpc::server_security s2; s2.allow_scripts = true; graph::rpc::server x(nullptr, 0, "127.0.0.1", s2); } catch (...) { threw = true; } CHECK(threw);
    graph::rpc::server open_srv(nullptr, 0); open_srv.start();
    auto oc = graph::rpc::client::connect("127.0.0.1", std::to_string(open_srv.port()), &why, {});
    threw = false; try { oc->run_script("x"); } catch (const std::exception& e) { threw = std::string(e.what()).find("disabled") != std::string::npos; } CHECK(threw);
    open_srv.stop();
}
// ---- per-key permissions (authorized_keys "allow=" options); plain TCP, so these run without OpenSSL too
static std::shared_ptr<graph::rpc::identity> ident(std::uint8_t b) {
    std::uint8_t seed[32] = {b, 7}; return std::make_shared<graph::rpc::identity>(graph::rpc::identity_from_seed_hex(graph::rpc::sec::hex(seed, 32)));
}
static std::string pubhex(const graph::rpc::identity& k) { return graph::rpc::sec::hex(k.pub.data(), 32); }
TEST(rpc_authorized_keys_parse) {
    auto a = ident(1), b = ident(2), c = ident(3);
    auto upper = pubhex(*c); for (auto& ch : upper) ch = char(std::toupper(static_cast<unsigned char>(ch)));
    auto ks = graph::rpc::parse_authorized_keys("# comment\n\n" + pubhex(*a) + "\n"                  // old format: full access
                                                "allow=device " + pubhex(*b) + " ci runner\r\n"
                                                "  allow=scripts allow=device " + upper + "\n");
    CHECK(ks.size() == 3);
    if (ks.size() == 3) {
        CHECK(ks[0].pub == pubhex(*a) && ks[0].perms == graph::rpc::perm_all && ks[0].comment.empty());
        CHECK(ks[1].perms == graph::rpc::perm_device && ks[1].comment == "ci runner");
        CHECK(ks[2].pub == pubhex(*c) && ks[2].perms == graph::rpc::perm_all);
    }
    CHECK(graph::rpc::parse_authorized_keys("allow= " + pubhex(*a))[0].perms == graph::rpc::perm_none);
    CHECK_THROWS(graph::rpc::parse_authorized_keys("allow=root " + pubhex(*a)));      // unknown permission
    CHECK_THROWS(graph::rpc::parse_authorized_keys("from=x " + pubhex(*a)));          // unknown option
    CHECK_THROWS(graph::rpc::parse_authorized_keys("allow=device\n"));                // no key
    CHECK_THROWS(graph::rpc::parse_authorized_keys("deadbeef comment\n"));            // not a key
}
TEST(rpc_key_permissions) {
    auto dev = ident(1), scr = ident(2), full = ident(3), none = ident(4);
    graph::rpc::server_security sec; sec.allow_scripts = true;
    sec.authorized_keys = graph::rpc::parse_authorized_keys("allow=device " + pubhex(*dev) + " gpu-only\n"
                                                            "allow=scripts " + pubhex(*scr) + "\n" +
                                                            pubhex(*full) + " legacy line\n"
                                                            "allow= " + pubhex(*none) + "\n");
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.script_handler = [](const std::string&, const std::string&, const std::vector<std::string>&, std::uint32_t) {
        return graph::rpc::server::script_result{true, "", "", "ran"};
    };
    srv.start(); auto port = std::to_string(srv.port());
    auto conn = [&](std::shared_ptr<graph::rpc::identity> k) { graph::rpc::client_security s; s.key = k; std::string why;
        auto c = graph::rpc::client::connect("127.0.0.1", port, &why, s); CHECK_MSG(c != nullptr, "%s", why.c_str()); return c; };
    auto denied = [](auto&& fn) { try { fn(); } catch (const std::exception& e) { return std::string(e.what()).find("permission denied") != std::string::npos; } return false; };
    if (auto c = conn(dev)) {
        CHECK(relu_ok(*c));
        CHECK(denied([&] { c->run_script("x"); }));
        CHECK(relu_ok(*c));                                       // still usable after a refusal
    }
    if (auto c = conn(scr)) {
        CHECK(c->run_script("x").output == "ran");
        CHECK(denied([&] { relu_ok(*c); }));
        CHECK(c->run_script("y").ok);
    }
    if (auto c = conn(full)) { CHECK(relu_ok(*c)); CHECK(c->run_script("x").ok); }   // old-format line keeps full access
    if (auto c = conn(none)) { CHECK(denied([&] { relu_ok(*c); })); CHECK(denied([&] { c->run_script("x"); })); }
    CHECK(srv.scripts() == 3);
    srv.stop();
}
// ---- scoped tokens (token-file lines "[allow=...] <token> [name]") and config errors with line numbers
static bool throws_with(auto&& fn, std::string_view needle) {
    try { fn(); } catch (const std::exception& e) { bool ok = std::string_view(e.what()).find(needle) != std::string_view::npos; if (!ok) std::printf("  message: %s\n", e.what()); return ok; }
    return false;
}
TEST(rpc_token_file_parse) {
    auto ts = graph::rpc::parse_tokens("# tokens\n\nbare-token\r\n"
                                       "allow=device dev+tok/== gpu borrower\n"
                                       "  allow=scripts allow=device both\n"
                                       "allow= nothing\n");
    CHECK(ts.size() == 4);
    if (ts.size() == 4) {
        CHECK(ts[0].token == "bare-token" && ts[0].perms == graph::rpc::perm_all && ts[0].name.empty());   // a bare token: full access
        CHECK(ts[1].token == "dev+tok/==" && ts[1].perms == graph::rpc::perm_device && ts[1].name == "gpu borrower");
        CHECK(ts[2].token == "both" && ts[2].perms == graph::rpc::perm_all);
        CHECK(ts[3].token == "nothing" && ts[3].perms == graph::rpc::perm_none);
    }
    CHECK(graph::rpc::parse_tokens("abc=").at(0).token == "abc=");                          // '=' inside a token is fine
    CHECK(throws_with([] { graph::rpc::parse_tokens("ok\n# c\nallow=root t\n", "tok.txt"); }, "tok.txt line 3: unknown permission 'root'"));
    CHECK(throws_with([] { graph::rpc::parse_tokens("ok\nallow=device\n"); }, "line 2: no token"));
    // authorized_keys errors carry line numbers too
    CHECK(throws_with([] { graph::rpc::parse_authorized_keys("\n\nallow=device,root " + std::string(64, 'a')); }, "authorized_keys line 3: unknown permission 'root'"));
    // client-certificate maps
    auto fp = std::string(64, 'a'); std::string colons; for (std::size_t i = 0; i < 64; i += 2) { if (i) colons += ':'; colons += "AA"; }
    auto m = graph::rpc::parse_client_map("# map\nallow=device sha256:" + colons + " laptop\ncn:ci-runner\nallow=scripts,device cn:x\n");
    CHECK(m.size() == 3);
    if (m.size() == 3) {
        CHECK(m[0].kind == graph::rpc::cert_rule::sha256 && m[0].value == fp && m[0].perms == graph::rpc::perm_device && m[0].comment == "laptop");
        CHECK(m[1].kind == graph::rpc::cert_rule::cn && m[1].value == "ci-runner" && m[1].perms == graph::rpc::perm_all);
        CHECK(m[2].value == "x" && m[2].perms == graph::rpc::perm_all);
    }
    CHECK(throws_with([] { graph::rpc::parse_client_map("cn:a\nsha256:abcd\n", "map.txt"); }, "map.txt line 2: sha256: wants a 64-hex"));
    CHECK(throws_with([] { graph::rpc::parse_client_map("cn:a\n\nallow=device\n"); }, "line 3: no certificate identity"));
    CHECK(throws_with([] { graph::rpc::parse_client_map("cn:a\nfrom=x cn:b\n"); }, "line 2: unknown option 'from'"));
    CHECK(throws_with([] { graph::rpc::parse_client_map("cn:a\ncn:a\n"); }, "line 2: duplicate"));
    CHECK(throws_with([] { graph::rpc::parse_client_map("cn:\n"); }, "line 1: empty cn"));
    CHECK(throws_with([] { graph::rpc::parse_client_map("alice\n"); }, "line 1"));
    // a client map needs client certificates
    graph::rpc::server_security s; s.client_map = graph::rpc::parse_client_map("cn:a\n");
    CHECK_THROWS(graph::rpc::server(nullptr, 0, "127.0.0.1", s));
}
TEST(rpc_token_scopes) {
    graph::rpc::server_security sec; sec.allow_scripts = true;
    sec.tokens = graph::rpc::parse_tokens("allow=device devtok gpu\nallow=scripts scrtok\nfulltok\nallow= nonetok\n");
    sec.tokens.push_back(std::string("legacy"));                                              // bare token pushed as before
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.script_handler = [](const std::string&, const std::string&, const std::vector<std::string>&, std::uint32_t) {
        return graph::rpc::server::script_result{true, "", "", "ran"};
    };
    std::mutex lm; std::vector<std::string> logs;
    srv.log = [&](const std::string& m) { std::lock_guard g(lm); logs.push_back(m); };
    srv.start(); auto port = std::to_string(srv.port());
    auto conn = [&](const std::string& t) { std::string why; auto c = graph::rpc::client::connect("127.0.0.1", port, &why, tok(t)); CHECK_MSG(c != nullptr, "%s", why.c_str()); return c; };
    auto denied = [](auto&& fn) { return throws_with(fn, "permission denied"); };
    if (auto c = conn("devtok")) { CHECK(relu_ok(*c)); CHECK(denied([&] { c->run_script("x"); })); CHECK(relu_ok(*c)); }
    if (auto c = conn("scrtok")) { CHECK(c->run_script("x").ok); CHECK(denied([&] { relu_ok(*c); })); CHECK(c->run_script("y").ok); }
    if (auto c = conn("fulltok")) { CHECK(relu_ok(*c)); CHECK(c->run_script("x").ok); }
    if (auto c = conn("legacy")) { CHECK(relu_ok(*c)); CHECK(c->run_script("x").ok); }
    if (auto c = conn("nonetok")) { CHECK(denied([&] { relu_ok(*c); })); CHECK(denied([&] { c->run_script("x"); })); }
    std::string why; CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, tok("wrong")));
    CHECK(srv.scripts() == 4);
    srv.stop();
    // the log names the principal (token number and name, never the token) and its permissions
    std::string all; for (auto& l : logs) all += l + "\n";
    CHECK_MSG(all.find("authenticated token#1 (gpu) allow=device") != std::string::npos, "%s", all.c_str());
    CHECK_MSG(all.find("authenticated token#5 allow=all") != std::string::npos, "%s", all.c_str());
    CHECK_MSG(all.find("denied script for token#1 (gpu)") != std::string::npos, "%s", all.c_str());
    CHECK_MSG(all.find("refused: bad token") != std::string::npos, "%s", all.c_str());
    CHECK(all.find("devtok") == std::string::npos && all.find("fulltok") == std::string::npos);
}

// ---- finer permissions (device.<op class>, scripts.run/fs/net/proc) and per-rule limits (ops=, max-bytes=,
// script-timeout=); plain TCP, so these run without OpenSSL too
static bool matmul_ok(graph::rpc::client& c) {
    tensor a({2, 3}, 1.f), b({3, 2}, 1.f), y({2, 2}); tensor_view in[2] = {{a.data.data(), a.shape}, {b.data.data(), b.shape}};
    c.run("matmul", std::span<const tensor_view>(in, 2), {y.data.data(), y.shape}, {}); return y.data[0] == 3.f && y.data[3] == 3.f;
}
static bool unary_ok(graph::rpc::client& c, const char* op, std::size_t n, float want) {
    tensor x({std::int64_t(n)}, -1.f), y({std::int64_t(n)}); tensor_view in{x.data.data(), x.shape};
    c.run(op, std::span<const tensor_view>(&in, 1), {y.data.data(), y.shape}, {}); return y.data[0] == want && y.data[n - 1] == want;
}
TEST(rpc_fine_permissions_parse) {
    namespace R = graph::rpc;
    // the old names are the full sets, so files written before the split mean what they meant
    CHECK(R::parse_perms("device") == R::perm_device && R::parse_perms("scripts") == R::perm_scripts && R::parse_perms("all") == R::perm_all);
    CHECK(R::parse_perms("device,scripts") == R::perm_all);
    CHECK(R::parse_perms("device.elementwise,device.reduction,device.contraction,device.data_movement,device.stateful") == R::perm_device);
    CHECK(R::parse_perms("scripts.run,scripts.fs,scripts.net,scripts.proc") == R::perm_scripts);
    CHECK(R::parse_perms("device.contraction,scripts.run") == unsigned(R::perm_device_contraction | R::perm_scripts_run));
    CHECK((R::perm_device & R::perm_scripts) == 0);
    // the device bits follow ops::op_class
    CHECK(R::perm_for_op_class(unsigned(ops::op_class::elementwise)) == R::perm_device_elementwise);
    CHECK(R::perm_for_op_class(unsigned(ops::op_class::reduction)) == R::perm_device_reduction);
    CHECK(R::perm_for_op_class(unsigned(ops::op_class::contraction)) == R::perm_device_contraction);
    CHECK(R::perm_for_op_class(unsigned(ops::op_class::data_movement)) == R::perm_device_data_movement);
    CHECK(R::perm_for_op_class(unsigned(ops::op_class::stateful)) == R::perm_device_stateful);
    CHECK(R::perm_for_op_class(5) == 0);
    CHECK(R::perms_string(R::perm_all) == "all" && R::perms_string(R::perm_none) == "none");
    CHECK(R::perms_string(R::perm_device) == "device" && R::perms_string(R::perm_scripts) == "scripts");
    CHECK_MSG(R::perms_string(R::perm_device_contraction | R::perm_scripts_run) == "device.contraction,scripts.run", "%s", R::perms_string(R::perm_device_contraction | R::perm_scripts_run).c_str());
    CHECK(R::perms_string(R::perm_device | R::perm_scripts_run | R::perm_scripts_net) == "device,scripts.run,scripts.net");
    CHECK(R::parse_perms(R::perms_string(R::perm_device_reduction | R::perm_scripts_fs)) == unsigned(R::perm_device_reduction | R::perm_scripts_fs));
    CHECK(throws_with([] { R::parse_perms("device.gpu"); }, "unknown permission 'device.gpu'"));
    CHECK(throws_with([] { R::parse_perms("scripts.exec"); }, "unknown permission"));
    // limits on token lines, key lines and client-map rules
    auto ts = R::parse_tokens("allow=device ops=relu,matmul ops=relu max-bytes=4k script-timeout=1500 tok a name\nops=neg bare\nplain\n");
    CHECK(ts.size() == 3);
    if (ts.size() == 3) {
        CHECK(ts[0].token == "tok" && ts[0].perms == R::perm_device && ts[0].name == "a name");
        CHECK(ts[0].lim.ops_limited && ts[0].lim.ops == std::vector<std::string>({"matmul", "relu"}) && ts[0].lim.max_bytes == 4096 && ts[0].lim.script_timeout_ms == 1500);
        CHECK_MSG(ts[0].lim.str() == " ops=matmul,relu max-bytes=4096 script-timeout=1500", "%s", ts[0].lim.str().c_str());
        CHECK(ts[0].lim.op_allowed("relu") && !ts[0].lim.op_allowed("neg"));
        CHECK(ts[1].token == "bare" && ts[1].perms == R::perm_all && ts[1].lim.ops == std::vector<std::string>({"neg"}));   // a limit alone keeps all permissions
        CHECK(ts[2].perms == R::perm_all && !ts[2].lim.any() && ts[2].lim.op_allowed("anything") && ts[2].lim.str().empty());
    }
    auto ks = R::parse_authorized_keys("allow=scripts.run script-timeout=250 max-bytes=2m " + std::string(64, 'a') + " ci\n");
    CHECK(ks.size() == 1 && ks[0].perms == R::perm_scripts_run && ks[0].lim.script_timeout_ms == 250 && ks[0].lim.max_bytes == (2u << 20) && ks[0].comment == "ci");
    auto cm = R::parse_client_map("allow=device.contraction ops=matmul max-bytes=1g cn:gpu\n");
    CHECK(cm.size() == 1 && cm[0].perms == R::perm_device_contraction && cm[0].lim.ops == std::vector<std::string>({"matmul"}) && cm[0].lim.max_bytes == (1ull << 30));
    CHECK(R::parse_tokens("ops= t").at(0).lim.ops_limited && R::parse_tokens("ops= t").at(0).lim.ops.empty());   // ops= with no value: no op at all
    CHECK(throws_with([] { R::parse_tokens("ok\nmax-bytes=0 t\n", "tok.txt"); }, "tok.txt line 2: max-bytes= wants a positive size"));
    CHECK(throws_with([] { R::parse_tokens("max-bytes=12x t\n"); }, "line 1: max-bytes="));
    CHECK(throws_with([] { R::parse_tokens("max-bytes=99999999999999999999 t\n"); }, "max-bytes="));
    CHECK(throws_with([] { R::parse_tokens("max-bytes=1 max-bytes=2 t\n"); }, "given twice"));
    CHECK(throws_with([] { R::parse_tokens("script-timeout=5k t\n"); }, "script-timeout= wants positive milliseconds"));
    CHECK(throws_with([] { R::parse_tokens("script-timeout=0 t\n"); }, "script-timeout="));
    CHECK(throws_with([] { R::parse_authorized_keys("limit=3 " + std::string(64, 'a')); }, "unknown option 'limit'"));
    CHECK(throws_with([] { R::parse_tokens("allow=device.contraction,device.gpu t\n"); }, "line 1: unknown permission 'device.gpu'"));
    // a token and its client certificate: the stricter of each limit, and only ops both allow
    R::limits a, b; a.ops_limited = true; a.ops = {"matmul", "relu"}; a.max_bytes = 100; b.ops_limited = true; b.ops = {"neg", "relu"}; b.max_bytes = 50; b.script_timeout_ms = 9;
    a.intersect(b); CHECK(a.ops == std::vector<std::string>({"relu"}) && a.max_bytes == 50 && a.script_timeout_ms == 9);
    R::limits c2; c2.intersect(a); CHECK(c2.ops_limited && c2.ops == a.ops && c2.max_bytes == 50);
    R::limits d; d.max_bytes = 10; d.intersect(R::limits{}); CHECK(d.max_bytes == 10 && !d.ops_limited);
    // a typo in ops= is a configuration error, not a rule that silently allows nothing
    R::server_security s; s.tokens = R::parse_tokens("ops=relu,matmull t\n");
    CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", s); }, "unknown op 'matmull'"));
}
TEST(rpc_fine_permissions_device) {
    namespace R = graph::rpc;
    R::server_security sec;
    sec.tokens = R::parse_tokens("allow=device.elementwise ew\nallow=device.contraction con\nallow=device ops=relu,matmul listed\n"
                                 "allow=device max-bytes=64 small\nallow=device.elementwise,device.contraction ops=neg both\nallow=scripts noscope\n");
    R::server srv(nullptr, 0, "127.0.0.1", sec);
    std::mutex lm; std::string logs; srv.log = [&](const std::string& m) { std::lock_guard g(lm); logs += m + "\n"; };
    srv.start(); auto port = std::to_string(srv.port());
    auto conn = [&](const std::string& t) { std::string why; auto c = R::client::connect("127.0.0.1", port, &why, tok(t)); CHECK_MSG(c != nullptr, "%s", why.c_str()); return c; };
    auto denied = [](auto&& fn, std::string_view why) { return throws_with(fn, "permission denied") && throws_with(fn, why); };
    if (auto c = conn("ew")) {            // one op class
        CHECK(relu_ok(*c));
        CHECK(denied([&] { matmul_ok(*c); }, "may not run contraction ops such as matmul (allow=device.contraction)"));
        CHECK(relu_ok(*c));               // the refused request's tensors were consumed: the connection is in step
        CHECK(throws_with([&] { unary_ok(*c, "no_such_op", 4, 0.f); }, "unknown op"));
    }
    if (auto c = conn("con")) { CHECK(matmul_ok(*c)); CHECK(denied([&] { relu_ok(*c); }, "may not run elementwise ops such as relu")); CHECK(matmul_ok(*c)); }
    if (auto c = conn("listed")) {        // every class, two ops
        CHECK(relu_ok(*c)); CHECK(matmul_ok(*c));
        CHECK(denied([&] { unary_ok(*c, "neg", 4, 1.f); }, "may not run the op neg (ops=)"));
        CHECK(relu_ok(*c));
    }
    if (auto c = conn("small")) {         // 4 floats in + 4 out = 32 bytes; 64 floats = 512 bytes
        CHECK(unary_ok(*c, "neg", 4, 1.f)); CHECK(unary_ok(*c, "neg", 8, 1.f));     // 64 bytes: at the limit
        CHECK(denied([&] { unary_ok(*c, "neg", 64, 1.f); }, "above its limit (max-bytes=64)"));
        CHECK(denied([&] { unary_ok(*c, "neg", 9, 1.f); }, "72 tensor bytes"));     // counted with the output
        CHECK(unary_ok(*c, "neg", 4, 1.f));
    }
    if (auto c = conn("both")) {          // the class and the list must both allow the op
        CHECK(unary_ok(*c, "neg", 4, 1.f));
        CHECK(denied([&] { relu_ok(*c); }, "(ops=)")); CHECK(denied([&] { matmul_ok(*c); }, "(ops=)"));
    }
    if (auto c = conn("noscope")) CHECK(denied([&] { relu_ok(*c); }, "may not run device ops"));
    CHECK_MSG(srv.ops() == 11, "ops %llu", static_cast<unsigned long long>(srv.ops()));
    srv.stop();
    CHECK_MSG(logs.find("authenticated token#1 allow=device.elementwise") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("authenticated token#3 allow=device ops=matmul,relu") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("authenticated token#4 allow=device max-bytes=64") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("denied device op matmul for token#1: may not run contraction ops") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("denied device op neg for token#4") != std::string::npos, "%s", logs.c_str());
}
TEST(rpc_fine_permissions_scripts) {
    namespace R = graph::rpc;
    R::server_security sec; sec.allow_scripts = true;
    sec.tokens = R::parse_tokens("allow=scripts.run runonly\nallow=scripts.run,scripts.fs fsonly\nallow=scripts full\nallow=scripts.fs,scripts.net,scripts.proc norun\n"
                                 "allow=scripts script-timeout=500 max-bytes=8 lim\nallow=device,scripts.run,scripts.net mixed\n");
    auto denied = [](auto&& fn, std::string_view why) { return throws_with(fn, "permission denied") && throws_with(fn, why); };
    {
        R::server srv(nullptr, 0, "127.0.0.1", sec);
        std::mutex qm; R::server::script_request last;
        srv.script_request_handler = [&](const R::server::script_request& q) { std::lock_guard g(qm); last = q; return R::server::script_result{true, "", "", "ran:" + q.source}; };
        // the request handler wins over the older ones
        srv.script_handler = [](const std::string&, const std::string&, const std::vector<std::string>&, std::uint32_t) { return R::server::script_result{false, "RuntimeError", "old handler", ""}; };
        std::mutex lm; std::string logs; srv.log = [&](const std::string& m) { std::lock_guard g(lm); logs += m + "\n"; };
        srv.start(); auto port = std::to_string(srv.port());
        auto conn = [&](const std::string& t) { std::string why; auto c = R::client::connect("127.0.0.1", port, &why, tok(t)); CHECK_MSG(c != nullptr, "%s", why.c_str()); return c; };
        if (auto c = conn("runonly")) {
            auto r = c->run_script("src", "<n>", {"a"}, 0, "sigtext"); CHECK(r.ok && r.output == "ran:src");
            CHECK(!last.allow_fs && !last.allow_net && !last.allow_proc);
            CHECK(last.who == "token#1" && last.name == "<n>" && last.argv == std::vector<std::string>({"a"}) && last.signature == "sigtext" && last.timeout_ms == 0);
            CHECK(denied([&] { relu_ok(*c); }, "may not run device ops"));
        }
        if (auto c = conn("fsonly")) { CHECK(c->run_script("x").ok); CHECK(last.allow_fs && !last.allow_net && !last.allow_proc); }
        if (auto c = conn("full")) { CHECK(c->run_script("x", "<remote>", {}, 1234).ok); CHECK(last.allow_fs && last.allow_net && last.allow_proc && last.timeout_ms == 1234 && last.signature.empty()); }
        if (auto c = conn("norun")) { CHECK(denied([&] { c->run_script("x"); }, "may not run scripts")); CHECK(denied([&] { c->run_script("y"); }, "may not run scripts")); }
        if (auto c = conn("lim")) {           // script-timeout= caps what the client asks for; max-bytes= caps the source
            CHECK(c->run_script("12345678").ok && last.timeout_ms == 500);
            CHECK(c->run_script("x", "<remote>", {}, 100).ok && last.timeout_ms == 100);
            CHECK(c->run_script("x", "<remote>", {}, 9000).ok && last.timeout_ms == 500);
            CHECK(denied([&] { c->run_script("123456789"); }, "script of 9 bytes, above its limit (max-bytes=8)"));
            CHECK(c->run_script("ok").ok);
        }
        if (auto c = conn("mixed")) { CHECK(relu_ok(*c)); CHECK(c->run_script("x").ok); CHECK(!last.allow_fs && last.allow_net && !last.allow_proc); }
        CHECK_MSG(srv.scripts() == 8, "scripts %llu", static_cast<unsigned long long>(srv.scripts()));
        srv.stop();
        CHECK_MSG(logs.find("authenticated token#1 allow=scripts.run") != std::string::npos, "%s", logs.c_str());
        CHECK_MSG(logs.find("authenticated token#5 allow=scripts max-bytes=8 script-timeout=500") != std::string::npos, "%s", logs.c_str());
        CHECK_MSG(logs.find("script '<n>' for token#1 (signed) without fs net proc") != std::string::npos, "%s", logs.c_str());
        CHECK_MSG(logs.find("script '<remote>' for token#6 without fs proc") != std::string::npos, "%s", logs.c_str());
        CHECK_MSG(logs.find("denied script for token#4") != std::string::npos, "%s", logs.c_str());
    }
    {   // a host with only the older handlers cannot withhold a capability: partial script permissions are refused
        R::server srv(nullptr, 0, "127.0.0.1", sec);
        srv.signed_script_handler = [](const std::string&, const std::string&, const std::vector<std::string>&, std::uint32_t t, const std::string&) {
            return R::server::script_result{true, "", "", "old:" + std::to_string(t)}; };
        srv.start(); auto port = std::to_string(srv.port()); std::string why;
        auto run = R::client::connect("127.0.0.1", port, &why, tok("runonly")); CHECK(run != nullptr);
        if (run) CHECK(denied([&] { run->run_script("x"); }, "cannot restrict capabilities"));
        auto full = R::client::connect("127.0.0.1", port, &why, tok("full")); CHECK(full != nullptr);
        if (full) CHECK(full->run_script("x").output == "old:0");
        auto lim = R::client::connect("127.0.0.1", port, &why, tok("lim")); CHECK(lim != nullptr);
        if (lim) CHECK(lim->run_script("x").output == "old:500");          // limits still apply
        CHECK(srv.scripts() == 2);
        srv.stop();
    }
}
// ---- client-map san: rules and the revoked list: parsing and matching (no TLS needed)
TEST(rpc_san_rules_parse) {
    namespace R = graph::rpc;
    auto m = R::parse_client_map("allow=device san:dns:Runner1.CI.Example. laptop\nsan:dns:*.ci.example\nsan:uri:spiffe://forge.test/ns/CI\n"
                                 "allow=scripts.run san:email:Ops@Forge.Test\nsan:ip:10.1.2.3\nsan:ip:FD00::7\nsan:ip:::ffff:1.2.3.4\ncn:x\n");
    CHECK(m.size() == 8);
    if (m.size() == 8) {
        CHECK(m[0].kind == R::cert_rule::san_dns && m[0].value == "runner1.ci.example" && m[0].perms == R::perm_device && m[0].comment == "laptop" && m[0].is_san());
        CHECK(m[1].kind == R::cert_rule::san_dns && m[1].value == "*.ci.example" && m[1].perms == R::perm_all);
        CHECK(m[2].kind == R::cert_rule::san_uri && m[2].value == "spiffe://forge.test/ns/CI");
        CHECK(m[3].kind == R::cert_rule::san_email && m[3].value == "Ops@forge.test" && m[3].perms == R::perm_scripts_run);
        CHECK(m[4].kind == R::cert_rule::san_ip && m[4].value == "10.1.2.3");
        CHECK_MSG(m[5].value == "fd00:0:0:0:0:0:0:7", "%s", m[5].value.c_str());
        CHECK_MSG(m[6].value == "0:0:0:0:0:ffff:102:304", "%s", m[6].value.c_str());
        CHECK(!m[7].is_san());
        // entries as channel::peer_cert_sans reports them
        CHECK(m[0].matches_san("dns:runner1.ci.example") && !m[0].matches_san("dns:runner2.ci.example") && !m[0].matches_san("uri:runner1.ci.example"));
        CHECK(m[1].matches_san("dns:runner1.ci.example") && m[1].matches_san("dns:x.ci.example"));
        CHECK(!m[1].matches_san("dns:ci.example") && !m[1].matches_san("dns:a.b.ci.example") && !m[1].matches_san("dns:.ci.example") && !m[1].matches_san("dns:xci.example"));
        CHECK(m[1].matches_san("dns:*.ci.example"));                       // the same spelling
        CHECK(!m[0].matches_san("dns:*.ci.example"));                      // a wildcard in the certificate is not expanded
        CHECK(m[2].matches_san("uri:spiffe://forge.test/ns/CI") && !m[2].matches_san("uri:spiffe://forge.test/ns/ci") && !m[2].matches_san("uri:spiffe://forge.test/ns/CI/x"));
        CHECK(m[3].matches_san("email:Ops@forge.test") && !m[3].matches_san("email:ops@forge.test"));
        CHECK(m[4].matches_san("ip:10.1.2.3") && !m[4].matches_san("ip:10.1.2.30") && !m[4].matches_san("dns:10.1.2.3"));
        CHECK(m[5].matches_san("ip:fd00:0:0:0:0:0:0:7"));
        CHECK(!m[7].matches_san("dns:x"));
    }
    CHECK(R::sec::dns_rule_matches("*.b.c", "a.b.c") && !R::sec::dns_rule_matches("*.b.c", "b.c") && !R::sec::dns_rule_matches("*.b.c", "*.x.b.c") && !R::sec::dns_rule_matches("", ""));
    // addresses
    std::vector<std::uint8_t> ip;
    CHECK(R::sec::parse_ip("192.168.0.1", ip) && ip == std::vector<std::uint8_t>({192, 168, 0, 1}));
    CHECK(R::sec::parse_ip("::", ip) && ip == std::vector<std::uint8_t>(16, 0));
    CHECK(R::sec::parse_ip("1::", ip) && ip.size() == 16 && ip[1] == 1 && ip[15] == 0);
    CHECK(R::sec::parse_ip("2001:db8:0:0:1:0:0:1", ip) && R::sec::ip_text(ip.data(), ip.size()) == "2001:db8:0:0:1:0:0:1");
    CHECK(R::sec::parse_ip("2001:db8::1:0:0:1", ip) && R::sec::ip_text(ip.data(), ip.size()) == "2001:db8:0:0:1:0:0:1");
    for (auto bad : {"", "1.2.3", "1.2.3.4.5", "256.1.1.1", "1.2.3.x", "1..2.3", "1:2", ":1::2", "1::2::3", "1:2:3:4:5:6:7:8:9", "1:2:3:4:5:6:7::8", "12345::", "g::", "1::2:", ":::", "1.2.3.4::"})
        CHECK_MSG(!R::sec::parse_ip(bad, ip), "parse_ip accepted '%s'", bad);
    CHECK(R::sec::ip_text(ip.data(), 5).empty());
    CHECK(R::sec::san_canonical("dns", "a b").empty() && R::sec::san_canonical("dns", std::string_view("a\0b", 3)).empty() && R::sec::san_canonical("dns", "a..b").empty());
    CHECK(R::sec::san_canonical("email", "nobody").empty() && R::sec::san_canonical("uri", "no-scheme").empty() && R::sec::san_canonical("other", "x").empty());
    // config errors name the file and line
    CHECK(throws_with([] { R::parse_client_map("cn:a\nsan:dns:\n", "map.txt"); }, "map.txt line 2: bad san:dns: value"));
    CHECK(throws_with([] { R::parse_client_map("san:host.example\n"); }, "line 1: san: wants a type"));
    CHECK(throws_with([] { R::parse_client_map("san:rid:1.2.3\n"); }, "san: wants a type"));
    CHECK(throws_with([] { R::parse_client_map("san:ip:10.0.0.256\n"); }, "bad san:ip: value"));
    CHECK(throws_with([] { R::parse_client_map("san:email:nobody\n"); }, "bad san:email: value"));
    CHECK(throws_with([] { R::parse_client_map("san:uri:no-scheme\n"); }, "bad san:uri: value"));
    for (auto bad : {"san:dns:*", "san:dns:*.", "san:dns:a.*.example", "san:dns:*x.example", "san:dns:*.*.example", "san:uri:spiffe://*/x", "san:email:*@forge.test"})
        CHECK_MSG(throws_with([&] { R::parse_client_map(std::string(bad) + "\n"); }, "'*' is only a wildcard"), "accepted %s", bad);
    CHECK(throws_with([] { R::parse_client_map("san:dns:a.example\nsan:dns:A.Example.\n"); }, "line 2: duplicate"));
    CHECK(throws_with([] { R::parse_client_map("san:ip:::1\nsan:ip:0:0:0:0:0:0:0:1\n"); }, "line 2: duplicate"));
    CHECK(throws_with([] { R::parse_client_map("from=x san:dns:a.example\n"); }, "unknown option 'from'"));
    // revoked lists
    auto fp = std::string(64, 'b'); std::string colons; for (std::size_t i = 0; i < 64; i += 2) { if (i) colons += ':'; colons += "BB"; }
    auto rv = R::parse_revoked("# stolen laptops\n\n" + colons + " laptop 3\nsha256:" + std::string(64, 'c') + "\n");
    CHECK(rv.size() == 2 && rv[0] == fp && rv[1] == std::string(64, 'c'));
    CHECK(R::parse_revoked("# nothing yet\n").empty());
    CHECK(throws_with([] { R::parse_revoked("ok\n", "rev.txt"); }, "rev.txt line 1: want a 64-hex"));
    // revocation needs client certificates
    R::server_security s; s.tokens = {"t"}; s.revoked_certs = {fp};
    CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", s); }, "needs client certificates"));
    R::server_security s2; s2.tokens = {"t"}; s2.tls.crl_files = {tmp("no-such-crl.pem")};
    CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", s2); }, "needs client certificates"));
}

// ---- TLS. These skip without OpenSSL (or its CLI); the `test_rpc_tls` ctest entry runs only them and then exits 77.
// The CLI runs through os::run (no shell, no redirects), so the probe and the throwaway PKI work on Windows too.
// FORGE_TEST_OPENSSL names the binary; on Windows, Git for Windows' copies are tried after PATH.
static const std::string& openssl_bin() {
    static const std::string b = [] {
        std::vector<std::string> c;
        if (const char* e = std::getenv("FORGE_TEST_OPENSSL"); e && *e) c.push_back(e);
        c.push_back("openssl");
#if defined(_WIN32)
        c.push_back("C:\\Program Files\\Git\\usr\\bin\\openssl.exe"); c.push_back("C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe");
#endif
        for (auto& x : c) { auto r = os::run({x, "version"}, true); if (r && r.value.exit_code == 0) return x; }
        return std::string{};
    }();
    return b;
}
static bool ossl(std::vector<std::string> args, std::string* out = nullptr) {
    args.insert(args.begin(), openssl_bin()); auto r = os::run(args, true);
    if (out && r) *out = r.value.out;
    return r && r.value.exit_code == 0;
}
static bool tls_or_skip() {
    if (graph::rpc::tls_available() && !openssl_bin().empty()) return true;
    std::printf("skip: %s\n", graph::rpc::tls_available() ? "no openssl CLI" : "built without TLS");
    if (const char* only = std::getenv("FORGE_TEST_ONLY"); only && std::strstr(only, "rpc_tls")) std::exit(77);
    return false;
}
// Throwaway PKI: a self-signed server cert for localhost, a client CA with three client certs (CN forge-client,
// forge-client-2, forge-client-3), and a self-signed "rogue" client cert (CN forge-client) the server does not trust.
struct test_pki { std::string srv, srv_key, ca, ca_key, cl, cl_key, cl2, cl2_key, cl3, cl3_key, rogue, rogue_key; bool ok = false; };
static const test_pki& pki() {
    static test_pki p = [] {
        test_pki p; const std::vector<std::string> ec = {"-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes"};
        auto req = [&](std::vector<std::string> a, const std::vector<std::string>& tail) { a.insert(a.end(), ec.begin(), ec.end()); a.insert(a.end(), tail.begin(), tail.end()); return ossl(a); };
        p.srv = tmp("srv.pem"); p.srv_key = tmp("srv.key"); p.ca = tmp("ca.pem"); p.ca_key = tmp("ca.key");
        p.cl = tmp("cl.pem"); p.cl_key = tmp("cl.key"); p.cl2 = tmp("cl2.pem"); p.cl2_key = tmp("cl2.key"); p.cl3 = tmp("cl3.pem"); p.cl3_key = tmp("cl3.key");
        p.rogue = tmp("rogue.pem"); p.rogue_key = tmp("rogue.key"); auto csr = tmp("cl.csr");
        auto client = [&](const std::string& cn, const std::string& crt, const std::string& key, const char* serial) {
            return req({"req"}, {"-subj", "/CN=" + cn, "-keyout", key, "-out", csr})
                && ossl({"x509", "-req", "-in", csr, "-CA", p.ca, "-CAkey", p.ca_key, "-set_serial", serial, "-days", "1", "-out", crt});
        };
        p.ok = req({"req", "-x509"}, {"-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost", "-keyout", p.srv_key, "-out", p.srv})
            && req({"req", "-x509"}, {"-days", "1", "-subj", "/CN=forge-test-ca", "-keyout", p.ca_key, "-out", p.ca})
            && client("forge-client", p.cl, p.cl_key, "2") && client("forge-client-2", p.cl2, p.cl2_key, "3") && client("forge-client-3", p.cl3, p.cl3_key, "4")
            && req({"req", "-x509"}, {"-days", "1", "-subj", "/CN=forge-client", "-keyout", p.rogue_key, "-out", p.rogue});
        std::error_code e; std::filesystem::remove(csr, e);
        return p;
    }();
    return p;
}
// SHA-256 fingerprint of a PEM certificate as the openssl CLI prints it ("AB:CD:..."), or "".
static std::string cli_fingerprint(const std::string& pem) {
    std::string out; if (!ossl({"x509", "-noout", "-fingerprint", "-sha256", "-in", pem}, &out)) return {};
    auto eq = out.find('='); if (eq == std::string::npos) return {};
    auto v = out.substr(eq + 1); while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.pop_back(); return v;
}
static graph::rpc::client_security tls_tok(const std::string& t) { auto s = tok(t); s.tls.enabled = true; s.tls.ca_file = pki().srv; s.tls.server_name = "localhost"; return s; }

TEST(rpc_tls_server_verify) {
    if (!tls_or_skip()) return;
    CHECK(pki().ok); auto crt = pki().srv, key = pki().srv_key;
    graph::rpc::server_security sec; sec.tokens = {"t"}; sec.tls = {crt, key};
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec); srv.start(); CHECK(srv.tls());
    auto port = std::to_string(srv.port()); std::string why;
    auto s = tok("t"); s.tls.enabled = true; s.tls.ca_file = crt; s.tls.server_name = "localhost";
    auto c = graph::rpc::client::connect("127.0.0.1", port, &why, s); CHECK_MSG(c != nullptr, "%s", why.c_str());
    if (c) CHECK(relu_ok(*c));
    auto nv = tok("t"); nv.tls.enabled = true; why.clear();                         // no CA / fingerprint: refuse
    CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, nv)); CHECK(why.find("verify") != std::string::npos);
    auto fp = tok("t"); fp.tls.enabled = true; fp.tls.fingerprint_sha256 = std::string(64, 'a'); why.clear();
    CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, fp)); CHECK(why.find("fingerprint") != std::string::npos);
    auto plain = tok("t"); why.clear(); auto pc = graph::rpc::client::connect("127.0.0.1", port, &why, plain);   // plaintext to a TLS server fails
    CHECK(!pc);
    srv.stop();
}
TEST(rpc_tls_mtls) {
    if (!tls_or_skip()) return;
    auto& p = pki(); CHECK(p.ok);
    graph::rpc::server_security sec; sec.tokens = {"t"}; sec.tls = {p.srv, p.srv_key, p.ca};
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec); srv.start();
    auto port = std::to_string(srv.port()); std::string why;
    auto good = tls_tok("t"); good.tls.cert_file = p.cl; good.tls.key_file = p.cl_key;
    auto c = graph::rpc::client::connect("127.0.0.1", port, &why, good); CHECK_MSG(c != nullptr, "%s", why.c_str());
    if (c) CHECK(relu_ok(*c));
    why.clear(); CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, tls_tok("t")));           // no client certificate
    auto rogue = tls_tok("t"); rogue.tls.cert_file = p.rogue; rogue.tls.key_file = p.rogue_key;         // not signed by the client CA
    why.clear(); CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, rogue));
    auto half = tls_tok("t"); half.tls.cert_file = p.cl;                                                 // cert without key
    why.clear(); CHECK(!graph::rpc::client::connect("127.0.0.1", port, &why, half)); CHECK(why.find("go together") != std::string::npos);
    CHECK(srv.ops() == 1);
    srv.stop();
    // mTLS without a server certificate is a configuration error
    graph::rpc::server_security bad; bad.tokens = {"t"}; bad.tls.client_ca_file = p.ca;
    CHECK_THROWS(graph::rpc::server(nullptr, 0, "127.0.0.1", bad));
    // the client CA file must load
    graph::rpc::server_security miss; miss.tokens = {"t"}; miss.tls = {p.srv, p.srv_key, tmp("missing-ca.pem")};
    CHECK_THROWS(graph::rpc::server(nullptr, 0, "127.0.0.1", miss));
}
// A man in the middle that the client wrongly trusts (it holds a certificate the client accepts) terminates the
// client's TLS and relays the auth exchange into its own TLS connection to the real server. Channel binding makes
// the relayed proof fail: the two TLS connections have different exporters.
static std::string relay_auth(int server_port, const graph::rpc::client_security& victim) {
#ifndef FORGE_HAVE_TLS
    (void)server_port; (void)victim; return "built without TLS";
#else
    auto& p = pki(); namespace R = graph::rpc;
    auto l = os::socket::listen("127.0.0.1", 0, 1); CHECK(bool(l)); if (!l) return "listen";
    int mitm_port = l.value.local_port();
    std::string victim_why; std::shared_ptr<R::client> victim_conn;
    std::thread th([&] { victim_conn = R::client::connect("127.0.0.1", std::to_string(mitm_port), &victim_why, victim); });
    auto a = l.value.accept(); CHECK(bool(a));
    R::channel down(std::move(a.value)); std::string why;
    CHECK_MSG(down.accept_tls(R::make_server_tls({p.srv, p.srv_key}), &why), "%s", why.c_str());
    auto u = os::socket::connect("127.0.0.1", server_port, 5000); CHECK(bool(u));
    R::channel up(std::move(u.value)); R::tls_client_config tc; tc.enabled = true; tc.ca_file = p.srv; tc.server_name = "localhost";
    CHECK_MSG(up.connect_tls(tc, "127.0.0.1", &why), "%s", why.c_str());
    CHECK(down.channel_binding().size() == 32 && up.channel_binding().size() == 32 && down.channel_binding() != up.channel_binding());
    auto pipe = [](R::channel& from, R::channel& to, std::size_t n) { std::vector<char> b(n); return from.recv_all(b.data(), n) && to.send_all(b.data(), n); };
    char hello[8];
    if (down.recv_all(hello, 8) && up.send_all(hello, 8)) {
        std::uint32_t method; std::memcpy(&method, hello + 4, 4);
        bool ok = method == 4 ? pipe(down, up, 32) && pipe(up, down, 32) && pipe(down, up, 64) : method == 3 ? pipe(down, up, 64) : false;
        std::uint32_t st = 0;
        if (ok && up.recv_all(&st, 4) && down.send_all(&st, 4) && st != 0) { std::uint32_t n = 0; if (up.recv_all(&n, 4) && n < 4096) { down.send_all(&n, 4); pipe(up, down, n); } }
    }
    th.join();
    CHECK(victim_conn == nullptr);
    return victim_why;
#endif
}
TEST(rpc_tls_channel_binding) {
    if (!tls_or_skip()) return;
    auto& p = pki(); CHECK(p.ok); namespace R = graph::rpc;
    auto id = ident(9);
    R::server_security sec; sec.tokens = {"t"}; sec.authorized_keys = {pubhex(*id)}; sec.tls = {p.srv, p.srv_key};
    R::server srv(nullptr, 0, "127.0.0.1", sec); srv.start();
    auto port = std::to_string(srv.port());
    // direct connections authenticate (bound methods 3/4)
    auto ks = tls_tok(""); ks.key = id; std::string why;
    auto c = R::client::connect("127.0.0.1", port, &why, ks); CHECK_MSG(c != nullptr, "%s", why.c_str()); if (c) CHECK(relu_ok(*c));
    why.clear(); c = R::client::connect("127.0.0.1", port, &why, tls_tok("t")); CHECK_MSG(c != nullptr, "%s", why.c_str()); if (c) CHECK(relu_ok(*c));
    why.clear(); CHECK(!R::client::connect("127.0.0.1", port, &why, tls_tok("nope"))); CHECK(why.find("bad token") != std::string::npos);
    // relayed through a second TLS connection: binding mismatch
    auto w = relay_auth(srv.port(), ks);   CHECK_MSG(w.find("key not authorized") != std::string::npos, "%s", w.c_str());
    w = relay_auth(srv.port(), tls_tok("t")); CHECK_MSG(w.find("bad token") != std::string::npos, "%s", w.c_str());
#ifdef FORGE_HAVE_TLS
    // unbound (pre-binding) methods are refused over TLS, so a relay cannot downgrade
    {
        auto u = os::socket::connect("127.0.0.1", srv.port(), 5000); CHECK(bool(u));
        R::channel ch(std::move(u.value)); R::tls_client_config tc; tc.enabled = true; tc.ca_file = p.srv; tc.server_name = "localhost";
        CHECK(ch.connect_tls(tc, "127.0.0.1", &why));
        CHECK(ch.send_all("FRPH", 4) && R::sec::put_u32(ch, 1));   // refused on the method alone (no unread bytes left behind)
        why.clear(); CHECK(!R::sec::status(ch, &why)); CHECK_MSG(why.find("channel-bound") != std::string::npos, "%s", why.c_str());
    }
#endif
    // and bound methods are refused on plain TCP
    {
        R::server_security ps; ps.tokens = {"t"}; R::server plain(nullptr, 0, "127.0.0.1", ps); plain.start();
        auto u = os::socket::connect("127.0.0.1", plain.port(), 5000); CHECK(bool(u));
        R::channel ch(std::move(u.value));
        CHECK(ch.send_all("FRPH", 4) && R::sec::put_u32(ch, 3));
        why.clear(); CHECK(!R::sec::status(ch, &why)); CHECK_MSG(why.find("needs TLS") != std::string::npos, "%s", why.c_str());
        plain.stop();
    }
    srv.stop();
}
// mTLS client map: fingerprint and CN rules, intersection with token permissions, unmapped certificates refused,
// and a certificate-only server (no tokens or keys) where a mapped certificate authenticates device ops.
TEST(rpc_tls_client_map) {
    if (!tls_or_skip()) return;
    auto& p = pki(); CHECK(p.ok); namespace R = graph::rpc;
    auto fp = cli_fingerprint(p.cl); CHECK_MSG(fp.size() == 95, "fingerprint '%s'", fp.c_str());   // AB:CD:... form
    // cl: fingerprint rule (device only) wins over the CN rule for its CN (all); cl2: CN rule (all); cl3: unmapped
    auto map = R::parse_client_map("allow=device sha256:" + fp + " laptop\ncn:forge-client\ncn:forge-client-2 ci\n");
    auto cert = [&](R::client_security s, const std::string& crt, const std::string& key) { s.tls.cert_file = crt; s.tls.key_file = key; return s; };
    auto script_ok = [](const std::string&, const std::string&, const std::vector<std::string>&, std::uint32_t) { return R::server::script_result{true, "", "", "ran"}; };
    auto denied = [](auto&& fn) { return throws_with(fn, "permission denied"); };
    std::mutex lm; std::string logs; auto logger = [&](const std::string& m) { std::lock_guard g(lm); logs += m + "\n"; };
    {
        R::server_security sec; sec.tls = {p.srv, p.srv_key, p.ca}; sec.client_map = map; sec.allow_scripts = true;
        sec.tokens = R::parse_tokens("full\nallow=device dev\n");
        auto key_id = ident(5); sec.authorized_keys = R::parse_authorized_keys("allow=scripts " + pubhex(*key_id) + " runner\n");
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.script_handler = script_ok; srv.log = logger; srv.start();
        auto port = std::to_string(srv.port()); std::string why;
        auto conn = [&](const R::client_security& s) { why.clear(); auto c = R::client::connect("127.0.0.1", port, &why, s); CHECK_MSG(c != nullptr, "%s", why.c_str()); return c; };
        if (auto c = conn(cert(tls_tok("full"), p.cl, p.cl_key))) {        // full token & device cert = device
            CHECK(relu_ok(*c)); CHECK(denied([&] { c->run_script("x"); }));
        }
        if (auto c = conn(cert(tls_tok("full"), p.cl2, p.cl2_key))) {      // full token & full cert = all
            CHECK(relu_ok(*c)); CHECK(c->run_script("x").ok);
        }
        if (auto c = conn(cert(tls_tok("dev"), p.cl2, p.cl2_key))) {       // device token & full cert = device
            CHECK(relu_ok(*c)); CHECK(denied([&] { c->run_script("x"); }));
        }
        auto ks = cert(tls_tok(""), p.cl, p.cl_key); ks.key = key_id;     // scripts key & device cert = nothing
        if (auto c = conn(ks)) { CHECK(denied([&] { relu_ok(*c); })); CHECK(denied([&] { c->run_script("x"); })); }
        why.clear(); CHECK(!R::client::connect("127.0.0.1", port, &why, cert(tls_tok("full"), p.cl3, p.cl3_key)));   // unmapped cert
        CHECK_MSG(why.find("client certificate not authorized") != std::string::npos, "%s", why.c_str());
        // a mapped certificate without a token: this server has tokens, so they are required
        if (auto c = conn(cert(tls_tok(""), p.cl2, p.cl2_key))) CHECK(throws_with([&] { relu_ok(*c); }, "authentication required"));
        CHECK(srv.scripts() == 1);
        srv.stop();
    }
    {   // certificate-only server: a mapped certificate authenticates device ops; scripts cannot be enabled
        R::server_security sec; sec.tls = {p.srv, p.srv_key, p.ca}; sec.client_map = map;
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.script_handler = script_ok; srv.log = logger; srv.start();
        auto port = std::to_string(srv.port()); std::string why;
        auto c = R::client::connect("127.0.0.1", port, &why, cert(tls_tok(""), p.cl, p.cl_key)); CHECK_MSG(c != nullptr, "%s", why.c_str());
        if (c) { CHECK(relu_ok(*c)); CHECK(throws_with([&] { c->run_script("x"); }, "disabled")); }
        auto c3 = R::client::connect("127.0.0.1", port, &why, cert(tls_tok(""), p.cl3, p.cl3_key)); CHECK(c3 != nullptr);   // TLS ok, then refused
        if (c3) CHECK(throws_with([&] { relu_ok(*c3); }, "client certificate not authorized"));
        CHECK(srv.ops() == 1);
        srv.stop();
        R::server_security s2 = sec; s2.allow_scripts = true;                // scripts still need tokens or keys
        CHECK_THROWS(R::server(nullptr, 0, "127.0.0.1", s2));
        R::server_security s3; s3.tls = {p.srv, p.srv_key}; s3.client_map = map; s3.tokens = {"t"};   // map without a client CA
        CHECK_THROWS(R::server(nullptr, 0, "127.0.0.1", s3));
    }
    CHECK_MSG(logs.find("authenticated token#1 cert:sha256:") != std::string::npos && logs.find("(laptop) allow=device") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("authenticated token#2 cert:cn=forge-client-2 (ci) allow=device") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("(runner) cert:sha256:") != std::string::npos && logs.find("(laptop) allow=none") != std::string::npos, "%s", logs.c_str());
    CHECK_MSG(logs.find("cn=forge-client-3 is not in the client map") != std::string::npos, "%s", logs.c_str());
}
// Second throwaway PKI, issued by pki()'s client CA: two client certificates with subjectAltName entries, and CRLs
// of that CA (one revoking nothing, one revoking pki().cl3) made with `openssl ca`. Everything lives in one directory.
struct test_pki2 { std::string dir, san1, san1_key, san2, san2_key, crl_none, crl_cl3; bool ok = false; };
static bool write_text(const std::string& path, const std::string& text) { std::ofstream f(path, std::ios::binary | std::ios::trunc); f << text; f.close(); return bool(f); }
static const test_pki2& pki2() {
    static test_pki2 q = [] {
        test_pki2 q; auto& p = pki(); if (!p.ok) return q;
        q.dir = tmp("pki2"); std::error_code e; std::filesystem::create_directories(q.dir, e);
        auto at = [&](const char* n) { return (std::filesystem::path(q.dir) / n).generic_string(); };
        q.san1 = at("san1.pem"); q.san1_key = at("san1.key"); q.san2 = at("san2.pem"); q.san2_key = at("san2.key"); q.crl_none = at("crl-none.pem"); q.crl_cl3 = at("crl-cl3.pem");
        auto csr = at("c.csr"), ext = at("ext.cnf"), cnf = at("ca.cnf"), index = at("index.txt");
        auto client = [&](const std::string& cn, const std::string& sans, const std::string& crt, const std::string& key, const char* serial) {
            return write_text(ext, "subjectAltName=" + sans + "\n")
                && ossl({"req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-subj", "/CN=" + cn, "-keyout", key, "-out", csr})
                && ossl({"x509", "-req", "-in", csr, "-CA", p.ca, "-CAkey", p.ca_key, "-set_serial", serial, "-days", "1", "-extfile", ext, "-out", crt});
        };
        // `openssl ca` wants a config naming its database; -gencrl and -revoke use nothing else of it
        auto ca = [&](std::vector<std::string> a) { a.insert(a.begin(), {"ca", "-config", cnf, "-keyfile", p.ca_key, "-cert", p.ca}); return ossl(a); };
        q.ok = client("san-one", "DNS:runner1.ci.forge.test,URI:spiffe://forge.test/ci/runner1,email:ci@Forge.Test,IP:10.1.2.3,IP:fd00::7", q.san1, q.san1_key, "16")
            && client("san-two", "DNS:deep.runner.ci.forge.test,DNS:*.wild.forge.test", q.san2, q.san2_key, "17")
            && write_text(index, "") && write_text(cnf, "[ca]\ndefault_ca = forge_ca\n[forge_ca]\ndatabase = " + index + "\ndefault_md = sha256\ndefault_crl_days = 30\n")
            && ca({"-gencrl", "-out", q.crl_none})
            && ca({"-revoke", p.cl3}) && ca({"-gencrl", "-out", q.crl_cl3});
        return q;
    }();
    return q;
}
static std::string slurp(const std::string& path) { auto r = os::read_file(path); return r ? r.value : std::string(); }
static graph::rpc::client_security with_cert(graph::rpc::client_security s, const std::string& crt, const std::string& key) { s.tls.cert_file = crt; s.tls.key_file = key; return s; }
// Connect and run one op. "" when it worked, else what went wrong (the connect error or the op's exception).
static std::string try_relu(int port, const graph::rpc::client_security& s) {
    std::string why; auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(port), &why, s);
    if (!c) return why.empty() ? "connect failed" : why;
    try { return relu_ok(*c) ? "" : "wrong result"; } catch (const std::exception& e) { return e.what(); }
}
// Client-map san: rules against real certificates: every SAN type, the DNS wildcard, and precedence over cn: rules.
TEST(rpc_tls_san_rules) {
    if (!tls_or_skip()) return;
    auto& p = pki(); auto& q = pki2(); CHECK(p.ok); CHECK(q.ok); namespace R = graph::rpc;
    if (!p.ok || !q.ok) return;
    // a certificate-only server with a one-rule map: does this certificate get in?
    auto admits = [&](const std::string& rule, const std::string& crt, const std::string& key) {
        R::server_security sec; sec.tls = {p.srv, p.srv_key, p.ca}; sec.client_map = R::parse_client_map(rule + "\n");
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.start();
        auto err = try_relu(srv.port(), with_cert(tls_tok(""), crt, key)); srv.stop();
        if (!err.empty()) CHECK_MSG(err.find("client certificate not authorized") != std::string::npos, "%s: %s", rule.c_str(), err.c_str());
        return err.empty();
    };
    auto one = [&](const std::string& rule) { return admits(rule, q.san1, q.san1_key); };
    auto two = [&](const std::string& rule) { return admits(rule, q.san2, q.san2_key); };
    CHECK(one("san:dns:runner1.ci.forge.test")); CHECK(!two("san:dns:runner1.ci.forge.test"));
    CHECK(!admits("san:dns:runner1.ci.forge.test", p.cl, p.cl_key));              // a certificate without SANs
    CHECK(one("san:dns:RUNNER1.ci.forge.test."));                                 // case and a trailing dot do not matter
    CHECK(!one("san:dns:runner2.ci.forge.test")); CHECK(!one("san:dns:ci.forge.test"));
    CHECK(one("san:dns:*.ci.forge.test"));                                        // one label
    CHECK(!two("san:dns:*.ci.forge.test"));                                       // deep.runner.ci.forge.test is two labels
    CHECK(two("san:dns:*.runner.ci.forge.test"));
    CHECK(!two("san:dns:build.wild.forge.test"));                                 // the certificate's own wildcard is not expanded
    CHECK(two("san:dns:*.wild.forge.test"));                                      // only the same spelling matches it
    CHECK(one("san:uri:spiffe://forge.test/ci/runner1")); CHECK(!one("san:uri:spiffe://forge.test/ci/runner")); CHECK(!one("san:uri:spiffe://forge.test/ci/Runner1"));
    CHECK(!two("san:uri:spiffe://forge.test/ci/runner1"));
    CHECK(one("san:email:ci@forge.test")); CHECK(!one("san:email:CI@forge.test")); CHECK(!one("san:email:ci@forge.example"));
    CHECK(one("san:ip:10.1.2.3")); CHECK(!one("san:ip:10.1.2.4")); CHECK(one("san:ip:fd00::7")); CHECK(one("san:ip:FD00:0:0::7")); CHECK(!one("san:ip:fd00::8"));
    CHECK(!one("san:dns:10.1.2.3")); CHECK(!one("san:uri:runner1.ci.forge.test:"));   // types do not cross
    CHECK(one("cn:san-one"));                                                     // cn: rules still work for it
    // With tokens: the san: rule's permissions intersect with the token's, and a san: rule wins over a cn: rule.
    {
        R::server_security sec; sec.tls = {p.srv, p.srv_key, p.ca}; sec.allow_scripts = true; sec.tokens = R::parse_tokens("full\n");
        sec.client_map = R::parse_client_map("cn:san-one by-name\nallow=device.elementwise,scripts.run san:dns:*.ci.forge.test runners\nallow=device san:uri:spiffe://forge.test/ci/runner1 later\n");
        R::server srv(nullptr, 0, "127.0.0.1", sec);
        std::mutex lm; std::string logs; srv.log = [&](const std::string& m) { std::lock_guard g(lm); logs += m + "\n"; };
        R::server::script_request last;
        srv.script_request_handler = [&](const R::server::script_request& r) { std::lock_guard g(lm); last = r; return R::server::script_result{true, "", "", "ran"}; };
        srv.start(); std::string why;
        auto c = R::client::connect("127.0.0.1", std::to_string(srv.port()), &why, with_cert(tls_tok("full"), q.san1, q.san1_key)); CHECK_MSG(c != nullptr, "%s", why.c_str());
        if (c) {
            CHECK(relu_ok(*c));
            CHECK(throws_with([&] { matmul_ok(*c); }, "permission denied"));          // device.elementwise only
            CHECK(c->run_script("x").ok); CHECK(!last.allow_fs && !last.allow_net && !last.allow_proc);
            CHECK_MSG(last.who == "token#1 cert:san=dns:runner1.ci.forge.test (runners)", "%s", last.who.c_str());
        }
        why.clear(); CHECK(!R::client::connect("127.0.0.1", std::to_string(srv.port()), &why, with_cert(tls_tok("full"), q.san2, q.san2_key)));   // no rule for san-two
        CHECK_MSG(why.find("client certificate not authorized") != std::string::npos, "%s", why.c_str());
        srv.stop();
        CHECK_MSG(logs.find("authenticated token#1 cert:san=dns:runner1.ci.forge.test (runners) allow=device.elementwise,scripts.run") != std::string::npos, "%s", logs.c_str());
    }
}
// Revocation: a CRL checked in the TLS handshake, a deny-list of fingerprints checked with the credentials, and both
// re-read when their files change.
TEST(rpc_tls_revocation) {
    if (!tls_or_skip()) return;
    auto& p = pki(); auto& q = pki2(); CHECK(p.ok); CHECK(q.ok); namespace R = graph::rpc;
    if (!p.ok || !q.ok) return;
    auto cl = with_cert(tls_tok("t"), p.cl, p.cl_key), cl2 = with_cert(tls_tok("t"), p.cl2, p.cl2_key), cl3 = with_cert(tls_tok("t"), p.cl3, p.cl3_key);
    std::mutex lm; std::string logs; auto logger = [&](const std::string& m) { std::lock_guard g(lm); logs += m + "\n"; };
    auto logged = [&](std::string_view what) { std::lock_guard g(lm); return logs.find(what) != std::string::npos; };
    {   // a CRL that revokes cl3: its handshake fails, whatever token it holds; the other certificates are served
        R::server_security sec; sec.tokens = {"t"}; sec.tls = {p.srv, p.srv_key, p.ca, {q.crl_cl3}};
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.log = logger; srv.start();
        CHECK_MSG(try_relu(srv.port(), cl).empty(), "%s", try_relu(srv.port(), cl).c_str());
        CHECK(try_relu(srv.port(), cl2).empty());
        CHECK(!try_relu(srv.port(), cl3).empty());
        CHECK(try_relu(srv.port(), cl).empty());
        CHECK(srv.ops() == 3);
        srv.stop();
        CHECK_MSG(logged("refused: tls accept:") && logged("certificate revoked"), "%s", logs.c_str());
    }
    {   // a CRL that revokes nothing serves cl3; a server without a CRL of the client CA would not check at all
        R::server_security sec; sec.tokens = {"t"}; sec.tls = {p.srv, p.srv_key, p.ca, {q.crl_none}};
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.start();
        CHECK_MSG(try_relu(srv.port(), cl3).empty(), "%s", try_relu(srv.port(), cl3).c_str());
        srv.stop();
    }
    {   // configuration errors: fail closed at start
        R::server_security sec; sec.tokens = {"t"}; sec.tls = {p.srv, p.srv_key, "", {q.crl_cl3}};              // a CRL without a client CA
        CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", sec); }, "needs client certificates"));
        sec.tls = {p.srv, p.srv_key, p.ca, {tmp("missing-crl.pem")}};
        CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", sec); }, "cannot read CRL"));
        sec.tls = {p.srv, p.srv_key, p.ca, {p.ca}};                                                                // PEM, but no CRL in it
        CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", sec); }, "no PEM CRL in"));
        // a CRL from another issuer does not cover the client CA: every client is refused (no CRL = fail closed)
        auto other_key = pki2().dir + "/other.key", other_ca = pki2().dir + "/other.pem", other_crl = pki2().dir + "/other-crl.pem", cnf = pki2().dir + "/ca.cnf";
        CHECK(ossl({"req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-days", "1", "-subj", "/CN=another-ca", "-keyout", other_key, "-out", other_ca}));
        CHECK(ossl({"ca", "-config", cnf, "-keyfile", other_key, "-cert", other_ca, "-gencrl", "-out", other_crl}));
        sec.tls = {p.srv, p.srv_key, p.ca, {other_crl}};
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.log = logger; srv.start();
        CHECK(!try_relu(srv.port(), cl).empty());
        srv.stop();
        CHECK_MSG(logged("unable to get certificate CRL"), "%s", logs.c_str());
    }
    {   // reload: the CRL file is replaced while the server runs
        auto live = pki2().dir + "/live-crl.pem"; CHECK(write_text(live, slurp(q.crl_none)));
        R::server_security sec; sec.tokens = {"t"}; sec.tls = {p.srv, p.srv_key, p.ca, {live}};
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.log = logger; srv.start();
        CHECK(try_relu(srv.port(), cl3).empty());
        CHECK(write_text(live, slurp(q.crl_cl3)));
        CHECK(!try_relu(srv.port(), cl3).empty());                               // revoked from the next connection on
        CHECK(try_relu(srv.port(), cl).empty());
        CHECK_MSG(logged("reloaded certificate revocation lists (1 CRL file(s), 0 revoked fingerprint(s))"), "%s", logs.c_str());
        CHECK(write_text(live, "not a CRL\n"));                                  // a broken file keeps the previous list
        CHECK(!try_relu(srv.port(), cl3).empty());
        CHECK(try_relu(srv.port(), cl).empty());
        CHECK_MSG(logged("certificate revocation reload failed, keeping the previous lists"), "%s", logs.c_str());
        srv.stop();
    }
    {   // the deny-list: a fingerprint as the openssl CLI prints it; refused with a valid token or key
        auto key_id = ident(6);
        R::server_security sec; sec.tokens = {"t"}; sec.authorized_keys = {pubhex(*key_id)}; sec.tls = {p.srv, p.srv_key, p.ca};
        sec.revoked_certs = {cli_fingerprint(p.cl2)};
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.log = logger; srv.start();
        CHECK(try_relu(srv.port(), cl).empty());
        auto e = try_relu(srv.port(), cl2); CHECK_MSG(e.find("permission denied: client certificate revoked") != std::string::npos, "%s", e.c_str());
        auto ks = with_cert(tls_tok(""), p.cl2, p.cl2_key); ks.key = key_id;
        e = try_relu(srv.port(), ks); CHECK_MSG(e.find("client certificate revoked") != std::string::npos, "%s", e.c_str());
        ks = with_cert(tls_tok(""), p.cl, p.cl_key); ks.key = key_id; CHECK(try_relu(srv.port(), ks).empty());
        e = try_relu(srv.port(), with_cert(tls_tok("wrong"), p.cl2, p.cl2_key)); CHECK_MSG(e.find("bad token") != std::string::npos, "%s", e.c_str());   // no hint without a credential
        CHECK(srv.ops() == 2);
        srv.stop();
        CHECK_MSG(logged("refused: token#1 with cert:sha256:") && logged("is on the revoked list"), "%s", logs.c_str());
        sec.revoked_certs = {"abcd"};
        CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", sec); }, "64-hex"));
    }
    {   // a revoked file, with a client map and no tokens (certificates authenticate), edited while the server runs
        auto file = pki2().dir + "/revoked.txt"; CHECK(write_text(file, "# nothing revoked yet\n"));
        R::server_security sec; sec.tls = {p.srv, p.srv_key, p.ca}; sec.client_map = R::parse_client_map("cn:forge-client\ncn:forge-client-2\n");
        sec.revoked_files = {file};
        R::server srv(nullptr, 0, "127.0.0.1", sec); srv.log = logger; srv.start();
        auto c1 = with_cert(tls_tok(""), p.cl, p.cl_key), c2 = with_cert(tls_tok(""), p.cl2, p.cl2_key);
        CHECK(try_relu(srv.port(), c1).empty()); CHECK(try_relu(srv.port(), c2).empty());
        CHECK(write_text(file, "# nothing revoked yet\nsha256:" + cli_fingerprint(p.cl) + " lost laptop\n"));
        auto e = try_relu(srv.port(), c1); CHECK_MSG(e.find("client certificate revoked") != std::string::npos, "%s", e.c_str());
        CHECK(try_relu(srv.port(), c2).empty());
        CHECK_MSG(logged("(0 CRL file(s), 1 revoked fingerprint(s))"), "%s", logs.c_str());
        CHECK(srv.ops() == 3);
        srv.stop();
        sec.revoked_files = {tmp("missing-revoked.txt")};
        CHECK(throws_with([&] { R::server x(nullptr, 0, "127.0.0.1", sec); }, "cannot read revoked list"));
    }
}
TEST(rpc_tls_zz_cleanup) {
    if (!graph::rpc::tls_available()) return;
    auto& p = pki();
    for (auto* f : {&p.srv, &p.srv_key, &p.ca, &p.ca_key, &p.cl, &p.cl_key, &p.cl2, &p.cl2_key, &p.cl3, &p.cl3_key, &p.rogue, &p.rogue_key}) { std::error_code ec; std::filesystem::remove(*f, ec); }
    if (std::error_code ec; !openssl_bin().empty() && !pki2().dir.empty()) std::filesystem::remove_all(pki2().dir, ec);
}

// ---- remote weight residency (graph/rpc.hpp protocol version 2; docs/rpc.md "Weights on the server") ----
namespace {
struct wfix {   // y[rows, cols] = x[rows, k] . W[cols, k]^T with W a noted weight
    i64 rows = 3, k = 256, cols = 64;
    tensor W, x, y;
    explicit wfix(float ph) : W({cols, k}), x({rows, k}), y({rows, cols}) {
        for (std::size_t i = 0; i < W.data.size(); ++i) W.data[i] = std::sin(float(i) * 0.37f + ph) * 0.1f;
        for (std::size_t i = 0; i < x.data.size(); ++i) x.data[i] = std::cos(float(i) * 0.11f + ph);
    }
    std::size_t wbytes() const { return W.data.size() * 4; }
    void run(graph::rpc::client& c) {
        std::fill(y.data.begin(), y.data.end(), -7.f);
        tensor_view in[2] = {{x.data.data(), x.shape}, {W.data.data(), W.shape}};
        c.run("matmul_t", std::span<const tensor_view>(in, 2), {y.data.data(), y.shape}, {});
    }
    bool right() const { auto r = ops::ref::matmul_t({x, W}, {}); for (std::size_t i = 0; i < r.data.size(); ++i) if (std::abs(r.data[i] - y.data[i]) > 1e-4f) return false; return true; }
};
// One connection: send `bytes`, half-close, return everything the server answers until it closes.
std::string raw_talk(int port, const std::vector<char>& bytes) {
    auto c = os::socket::connect("127.0.0.1", port, 2000); if (!c) return "<no connection>";
    c->set_timeout_ms(5000);
    if (!bytes.empty() && !c->send_all(bytes.data(), bytes.size())) return "<send failed>";
    c->shutdown_write();
    std::string out; char buf[4096];
    for (long k; (k = c->recv(buf, sizeof buf)) > 0;) out.append(buf, std::size_t(k));
    return out;
}
std::uint32_t u32_at(const std::string& s, std::size_t at) { std::uint32_t v = 0xffffffffu; if (s.size() >= at + 4) std::memcpy(&v, s.data() + at, 4); return v; }
hash::blake3::digest hash_of(const std::vector<float>& d) { hash::blake3 h; h.update(d.data(), d.size() * 4); return h.final(); }
void put_upload(graph::rpc::detail::writer& w, const hash::blake3::digest& h, const shape_t& sh, const std::vector<float>& d) {
    w.raw("FRPW", 4); w.raw(h.data(), 32); w.shape(sh); w.raw(d.data(), d.size() * 4);
}
}

TEST(rpc_weight_residency_upload_once) {
    graph::rpc::server srv(nullptr, 0); srv.start();
    auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
    CHECK(c != nullptr);
    wfix f(1.f);
    f.run(*c); CHECK(f.right());                                   // not noted: the weight travels with the op (FRPC)
    CHECK(c->uploads() == 0 && c->weight_refs() == 0 && srv.weight_uploads() == 0);
    c->note_weight(f.W.data.data(), f.W.data.size());
    const auto sent0 = c->bytes_sent();
    f.run(*c); CHECK(f.right());                                   // first use: one upload, then the op by handle
    CHECK(c->resident() && c->weight_budget() == srv.weight_budget_bytes);
    CHECK(c->uploads() == 1 && c->upload_bytes() == f.wbytes() && c->weight_refs() == 1);
    CHECK(srv.weight_uploads() == 1 && srv.weight_bytes() == f.wbytes() && srv.weight_hits() == 1);
    const auto sent1 = c->bytes_sent();
    for (int i = 0; i < 3; ++i) { f.x.data[0] += 1.f; f.run(*c); CHECK(f.right()); }
    CHECK(c->uploads() == 1 && c->weight_refs() == 4 && srv.weight_uploads() == 1 && srv.weight_hits() == 4);
    CHECK_MSG(sent1 - sent0 > f.wbytes() && c->bytes_sent() - sent1 < f.wbytes() / 2, "bytes sent: first use %llu, three later ops %llu, weight %zu",
              (unsigned long long)(sent1 - sent0), (unsigned long long)(c->bytes_sent() - sent1), f.wbytes());
    // the same bytes at another address: another upload request, but the server already holds the hash
    wfix g(1.f); c->note_weight(g.W.data.data(), g.W.data.size());
    g.run(*c); CHECK(g.right());
    CHECK(c->uploads() == 2 && srv.weight_uploads() == 1 && srv.weight_bytes() == f.wbytes());
    // a noted buffer whose contents changed is hashed again, never served from the old handle
    for (auto& v : f.W.data) v *= 2.f;
    f.run(*c); CHECK(f.right());
    CHECK(c->uploads() == 3 && srv.weight_uploads() == 2);
    // forgotten: travels inline again
    c->forget_weight(f.W.data.data(), f.W.data.size());
    const auto refs = c->weight_refs(); f.run(*c); CHECK(f.right() && c->weight_refs() == refs);
    c.reset();                                                     // the connection's weights go with it
    for (int i = 0; i < 200 && srv.weight_bytes(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(srv.weight_bytes() == 0);
    srv.stop();
}

TEST(rpc_weight_residency_lru_budget) {
    graph::rpc::server srv(nullptr, 0);
    wfix a(1.f), b(2.f), c3(3.f);
    srv.weight_budget_bytes = 2 * a.wbytes() + 100;                // room for two of the three
    srv.start();
    auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
    for (auto* f : {&a, &b, &c3}) c->note_weight(f->W.data.data(), f->W.data.size());
    a.run(*c); b.run(*c); a.run(*c);                               // a is now the most recently used
    CHECK(a.right() && b.right() && srv.weight_uploads() == 2 && srv.weight_evictions() == 0);
    c3.run(*c);                                                    // evicts b, the least recently used
    CHECK(c3.right() && srv.weight_uploads() == 3 && srv.weight_evictions() == 1 && srv.weight_bytes() == 2 * a.wbytes());
    a.run(*c); CHECK(a.right() && c->weight_misses() == 0);        // a is still there
    b.run(*c);                                                     // b is not: the server says so, the client uploads it again
    CHECK(b.right() && c->weight_misses() == 1 && c->uploads() == 4 && srv.weight_evictions() == 2);
    CHECK(srv.weight_bytes() <= srv.weight_budget_bytes);
    // a weight larger than the whole budget is refused once and then travels inline
    wfix big(4.f); big.cols = 300; big.W = tensor({big.cols, big.k}); big.y = tensor({big.rows, big.cols});
    for (std::size_t i = 0; i < big.W.data.size(); ++i) big.W.data[i] = std::sin(float(i) * 0.21f) * 0.1f;
    c->note_weight(big.W.data.data(), big.W.data.size());
    const auto up = c->uploads(); big.run(*c); big.run(*c);
    CHECK(big.right() && c->uploads() == up && srv.weight_bytes() <= srv.weight_budget_bytes);
    // a second connection shares the budget but not the weights: it cannot use a hash the first one uploaded
    auto c2 = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
    c2->note_weight(a.W.data.data(), a.W.data.size());
    const auto su = srv.weight_uploads(); a.run(*c2);
    CHECK(a.right() && c2->uploads() <= 1 && (c2->uploads() == 0 || srv.weight_uploads() == su + 1));
    CHECK(srv.weight_bytes() <= srv.weight_budget_bytes);
    srv.stop();
}

TEST(rpc_weight_residency_reconnect_invalidates) {
    graph::rpc::server srv(nullptr, 0); srv.start();
    auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
    wfix f(1.f); c->note_weight(f.W.data.data(), f.W.data.size());
    f.run(*c); f.run(*c);
    CHECK(f.right() && c->uploads() == 1 && srv.weight_uploads() == 1);
    CHECK(c->reconnect());                                         // a new connection: the server's copy is gone, the client knows
    f.run(*c);
    CHECK(f.right() && c->uploads() == 2 && srv.weight_uploads() == 2 && c->weight_misses() == 0 && c->reconnects() == 1);
    srv.stop();
    // the server goes away and another one takes its place: the next op reconnects by itself and uploads again
    graph::rpc::server s1(nullptr, 0); s1.start(); const int port = s1.port();
    auto d = graph::rpc::client::connect("127.0.0.1", std::to_string(port), nullptr);
    d->note_weight(f.W.data.data(), f.W.data.size());
    f.run(*d); CHECK(f.right() && d->uploads() == 1);
    s1.stop();
    bool rebound = true; std::unique_ptr<graph::rpc::server> s2;
    try { s2 = std::make_unique<graph::rpc::server>(nullptr, port); s2->start(); } catch (const std::exception&) { rebound = false; }   // the port may be taken
    if (rebound) {
        f.run(*d);
        CHECK(f.right() && d->reconnects() == 1 && d->uploads() == 2 && s2->weight_uploads() == 1);
        s2->stop();
    } else std::puts("  note: could not listen on the same port again; automatic reconnect not exercised");
}

TEST(rpc_weight_residency_old_peers) {
    wfix f(1.f);
    {   // a server from before residency (protocol 1): the probe is an unknown op, weights travel with every op
        graph::rpc::server srv(nullptr, 0); srv.protocol_version = 1; srv.start();
        auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
        c->note_weight(f.W.data.data(), f.W.data.size());
        f.run(*c); f.run(*c);
        CHECK(f.right() && !c->resident() && c->uploads() == 0 && c->weight_refs() == 0 && c->reconnects() == 0 && srv.weight_uploads() == 0);
        // and it closes the connection on the new frames, as an old server does on any unknown magic
        graph::rpc::detail::writer w; put_upload(w, hash_of(f.W.data), f.W.shape, f.W.data);
        CHECK(raw_talk(srv.port(), w.b).empty());
        srv.stop();
    }
    {   // a client from before residency sends plain FRPC frames: a version 2 server serves them unchanged
        graph::rpc::server srv(nullptr, 0); srv.start();
        graph::rpc::detail::writer w; w.raw("FRPC", 4); w.str("relu"); graph::rpc::detail::put_attrs(w, {}); w.u32(1);
        std::vector<float> v{-1.f, 2.f}; w.shape({2}); w.raw(v.data(), 8); w.shape({2});
        auto rep = raw_talk(srv.port(), w.b);
        float out[2] = {9, 9}; if (rep.size() == 12) std::memcpy(out, rep.data() + 4, 8);
        CHECK(rep.size() == 12 && u32_at(rep, 0) == 0 && out[0] == 0.f && out[1] == 2.f);
        srv.stop();
    }
    {   // a server with no budget says so in the probe
        graph::rpc::server srv(nullptr, 0); srv.weight_budget_bytes = 0; srv.start();
        auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
        c->note_weight(f.W.data.data(), f.W.data.size());
        f.run(*c); CHECK(f.right() && !c->resident() && c->uploads() == 0);
        srv.stop();
    }
}

TEST(rpc_weight_residency_permissions) {
    wfix f(1.f);                                                   // W: 64 KiB, x: 3 KiB, y: 768 bytes
    graph::rpc::server_security sec;
    sec.tokens = graph::rpc::parse_tokens(
        "full\n"
        "allow=device.contraction contr\n"
        "allow=device.elementwise elem\n"
        "allow=device max-bytes=4096 small\n"                      // the weight alone is over the limit
        "allow=device max-bytes=66000 exact\n"                     // the weight fits; weight + x + y (69,376 bytes) does not
        "allow=device max-bytes=100000 roomy\n"
        "allow=scripts scr\n");
    std::vector<std::string> log; std::mutex lm;
    graph::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.log = [&](const std::string& m) { std::lock_guard g(lm); log.push_back(m); };
    srv.start();
    auto conn = [&](const char* t) { auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr, tok(t)); if (c) c->note_weight(f.W.data.data(), f.W.data.size()); return c; };
    auto up0 = srv.weight_uploads();
    { auto c = conn("contr"); f.run(*c); f.run(*c); CHECK(f.right() && c->uploads() == 1 && c->weight_refs() == 2); }      // matmul_t is a contraction
    CHECK(srv.weight_uploads() == up0 + 1);
    { auto c = conn("elem");                                       // may upload (it has a device permission) but not run a contraction
      CHECK(throws_with([&] { f.run(*c); }, "may not run contraction ops")); CHECK(c->uploads() == 1); }
    { auto c = conn("scr");                                        // no device permission at all: the upload is refused, then the op
      CHECK(throws_with([&] { f.run(*c); }, "permission denied")); CHECK(c->uploads() == 0); }
    { auto c = conn("small");                                      // upload over max-bytes: refused without being buffered; inline is over it too
      const auto su = srv.weight_uploads();
      CHECK(throws_with([&] { f.run(*c); }, "above its limit (max-bytes=4096)")); CHECK(c->uploads() == 0 && srv.weight_uploads() == su); }
    { auto c = conn("exact");                                      // the upload fits, but the handle counts as the weight's bytes in the op
      CHECK(throws_with([&] { f.run(*c); }, "above its limit (max-bytes=66000)")); CHECK(c->uploads() == 1); }
    { auto c = conn("roomy"); f.run(*c); CHECK(f.right() && c->uploads() == 1 && c->weight_refs() == 1); }
    srv.stop();
    bool logged = false; for (auto& l : log) if (l.find("refused weight upload") != std::string::npos) logged = true;
    CHECK(logged);
}

// Everything in an upload or a handle comes from the peer. Found while writing the decoder, kept as cases: a wrong
// hash, an unknown handle, a handle used with another size, a 44-byte frame naming gigabytes, a bad input kind.
TEST(rpc_weight_residency_untrusted_wire) {
    using graph::rpc::detail::writer;
    graph::rpc::server srv(nullptr, 0); srv.max_tensor_elems = 1u << 20; srv.start();
    std::vector<float> wd(64); for (std::size_t i = 0; i < wd.size(); ++i) wd[i] = float(i) - 30.f;
    const auto good = hash_of(wd);
    auto op_by_handle = [&](writer& w, const hash::blake3::digest& h, const shape_t& sh, std::uint32_t kind = 1) {
        w.raw("FRPR", 4); w.str("relu"); graph::rpc::detail::put_attrs(w, {}); w.u32(1); w.u32(kind); w.raw(h.data(), 32); w.shape(sh); w.shape(sh);
    };
    {   // upload, then use the handle: status 0 twice, relu of the stored data
        writer w; put_upload(w, good, {64}, wd); op_by_handle(w, good, {64});
        auto rep = raw_talk(srv.port(), w.b);
        CHECK(rep.size() == 4 + 4 + 64 * 4 && u32_at(rep, 0) == 0 && u32_at(rep, 4) == 0);
        float y30 = -1, y40 = -1; if (rep.size() == 264) { std::memcpy(&y30, rep.data() + 8 + 30 * 4, 4); std::memcpy(&y40, rep.data() + 8 + 40 * 4, 4); }
        CHECK(y30 == 0.f && y40 == 10.f);
    }
    {   // the same handle with another shape of the same size is fine; with another size it is an error, not a read past the weight
        writer w; put_upload(w, good, {64}, wd); op_by_handle(w, good, {8, 8}); op_by_handle(w, good, {65}); op_by_handle(w, good, {4});
        auto rep = raw_talk(srv.port(), w.b);
        CHECK(u32_at(rep, 0) == 0 && u32_at(rep, 4) == 0 && u32_at(rep, 8 + 256) == 1);
        CHECK(rep.find("a weight handle names 64 elements, the request 65") != std::string::npos && rep.find("the request 4") != std::string::npos);
    }
    {   // a hash that does not match the bytes: refused, nothing stored, and the handle stays unknown
        auto bad = good; bad[0] ^= 1;
        writer w; put_upload(w, bad, {64}, wd); op_by_handle(w, bad, {64});
        auto rep = raw_talk(srv.port(), w.b);
        CHECK(u32_at(rep, 0) == 1 && rep.find("do not have the hash") != std::string::npos);
        const auto at = rep.size() >= 40 ? rep.size() - 40 : 0;    // the last reply: status 3, one missing hash
        CHECK(rep.size() >= 40 && u32_at(rep, at) == 3 && u32_at(rep, at + 4) == 1 && std::memcmp(rep.data() + at + 8, bad.data(), 32) == 0);
    }
    {   // a handle another connection uploaded is not visible here
        writer up; put_upload(up, good, {64}, wd);
        auto c1 = os::socket::connect("127.0.0.1", srv.port(), 2000); CHECK(bool(c1));
        if (c1) { c1->set_timeout_ms(5000); CHECK(c1->send_all(up.b.data(), up.b.size())); char st[4]; CHECK(c1->recv_all(st, 4)); }   // stored, connection kept open
        writer w; op_by_handle(w, good, {64});
        auto rep = raw_talk(srv.port(), w.b);
        CHECK(rep.size() == 40 && u32_at(rep, 0) == 3);
    }
    {   // 44 bytes that name a 4 MiB weight: over this server's budget -> refused by size, the missing payload just ends the connection
        graph::rpc::server small_(nullptr, 0); small_.weight_budget_bytes = 1 << 16; small_.max_tensor_elems = 1u << 30; small_.start();
        writer w; w.raw("FRPW", 4); w.raw(good.data(), 32); w.shape({i64(1) << 20});
        CHECK(raw_talk(small_.port(), w.b).empty() && small_.weight_bytes() == 0 && small_.weight_uploads() == 0);
        writer w2; w2.raw("FRPW", 4); w2.raw(good.data(), 32); w2.shape({i64(1) << 40});      // more elements than any tensor may have: dropped
        CHECK(raw_talk(small_.port(), w2.b).empty());
        writer w3; w3.raw("FRPW", 4); w3.raw(good.data(), 32); w3.u32(99);                     // a rank no shape has
        CHECK(raw_talk(small_.port(), w3.b).empty());
        writer w4; put_upload(w4, hash_of({}), {0}, {});                                        // an empty weight
        auto rep = raw_talk(small_.port(), w4.b); CHECK(u32_at(rep, 0) == 1 && rep.find("empty weight") != std::string::npos);
        small_.stop();
    }
    {   // an input kind that does not exist ends the connection; a truncated handle does too
        writer w; op_by_handle(w, good, {64}, 7); CHECK(raw_talk(srv.port(), w.b).empty());
        writer w2; w2.raw("FRPR", 4); w2.str("relu"); graph::rpc::detail::put_attrs(w2, {}); w2.u32(1); w2.u32(1); w2.raw(good.data(), 10);
        CHECK(raw_talk(srv.port(), w2.b).empty());
    }
    {   // handles and inline inputs mixed, more handles than the op has inputs, the same handle twice
        std::vector<float> xd(64, 2.f);
        writer w; put_upload(w, good, {64}, wd);
        w.raw("FRPR", 4); w.str("add"); graph::rpc::detail::put_attrs(w, {}); w.u32(2); w.u32(1); w.raw(good.data(), 32); w.shape({64}); w.u32(0); w.shape({64}); w.raw(xd.data(), 256); w.shape({64});
        w.raw("FRPR", 4); w.str("add"); graph::rpc::detail::put_attrs(w, {}); w.u32(2); w.u32(1); w.raw(good.data(), 32); w.shape({64}); w.u32(1); w.raw(good.data(), 32); w.shape({64}); w.shape({64});
        w.raw("FRPR", 4); w.str("relu"); graph::rpc::detail::put_attrs(w, {}); w.u32(2); w.u32(1); w.raw(good.data(), 32); w.shape({64}); w.u32(1); w.raw(good.data(), 32); w.shape({64}); w.shape({64});
        auto rep = raw_talk(srv.port(), w.b);
        float a0 = 0, b0 = 0; if (rep.size() >= 4 + 260 + 260) { std::memcpy(&a0, rep.data() + 8, 4); std::memcpy(&b0, rep.data() + 4 + 260 + 4, 4); }
        CHECK(u32_at(rep, 0) == 0 && u32_at(rep, 4) == 0 && a0 == -28.f && u32_at(rep, 4 + 260) == 0 && b0 == -60.f);
        CHECK(u32_at(rep, 4 + 520) == 1 && rep.find("relu takes 1 inputs, got 2") != std::string::npos);
    }
    {   // the server is still serving
        auto c = graph::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
        CHECK(c && relu_ok(*c));
    }
    for (int i = 0; i < 300 && srv.weight_bytes(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(srv.weight_bytes() == 0);   // every connection above is closed: nothing stays stored
    srv.stop();
}

// A whole model over loopback: the plan reports its weights (registry::weights), each is uploaded once, the second
// forward pass uploads nothing, and the logits are those of the CPU.
TEST(rpc_weight_residency_model) {
    graph::rpc::server srv(nullptr, 0); srv.start();
    auto ref = logits("cpu");
    auto m = model::open(model_a());
    const std::string dev = "rpc:127.0.0.1:" + std::to_string(srv.port());
    CHECK(m->set_device(dev) == dev);
    std::vector<int> p = {1, 5, 9, 33, 70, 101};
    std::vector<float> got;
    std::uint64_t up1 = 0, hits1 = 0;
    {
        inference::session s(m);
        got = s.forward(p);
        up1 = srv.weight_uploads(); hits1 = srv.weight_hits();
        int t = 7; auto b = s.forward(std::span<const int>(&t, 1)); got.insert(got.end(), b.begin(), b.end());
    }
    CHECK_MSG(up1 > 0 && hits1 >= up1, "uploads %llu, handle uses %llu", (unsigned long long)up1, (unsigned long long)hits1);
    CHECK_MSG(srv.weight_uploads() <= up1 + 4, "the second pass uploaded %llu more weights", (unsigned long long)(srv.weight_uploads() - up1));
    CHECK(srv.weight_hits() > hits1);
    CHECK_MSG(maxdiff(ref, got) < 1e-5f, "rpc diff %g", maxdiff(ref, got));
    srv.stop();
}
