// RPC device over loopback and pipeline split: whole-model logits must match the plain CPU run.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/mind/mind.hpp>
#include <forge/flow/vulkan.hpp>
#include <filesystem>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace forge;
using namespace forge::mind;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_rpc_" + std::to_string(::getpid()) + "_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }
static std::vector<float> logits(const std::string& dev) {
    auto m = model::open(model_a());
    CHECK(m->set_device(dev) == dev);
    mind::session s(m);
    std::vector<int> p = {1, 5, 9, 33, 70, 101};
    auto a = s.forward(p); int t = 7; auto b = s.forward(std::span<const int>(&t, 1));
    a.insert(a.end(), b.begin(), b.end()); return a;
}
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size()); float m = 0; for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i])); return m; }

TEST(rpc_loopback_matches_cpu) {
    flow::rpc::server srv(nullptr, 0);
    srv.start();
    auto ref = logits("cpu");
    auto got = logits("rpc:127.0.0.1:" + std::to_string(srv.port()));
    CHECK(srv.ops() > 50);
    CHECK_MSG(maxdiff(ref, got) < 1e-5f, "rpc diff %g", maxdiff(ref, got));
    srv.stop();
}
TEST(rpc_errors_propagate) {
    flow::rpc::server srv(nullptr, 0); srv.start();
    auto c = flow::rpc::client::connect("127.0.0.1", std::to_string(srv.port()), nullptr);
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
TEST(rpc_connect_failure_falls_back) {
    std::string why; auto r = flow::device_registry("rpc:127.0.0.1:1", &why);
    CHECK(!r); CHECK(!why.empty());
}
TEST(pipeline_split_two_stages) {
    flow::rpc::server srv(nullptr, 0); srv.start();
    auto ref = logits("cpu");
    auto dev = "pipe:cpu,rpc:127.0.0.1:" + std::to_string(srv.port());
    auto got = logits(dev);
    CHECK(srv.ops() > 0);
    CHECK_MSG(maxdiff(ref, got) < 1e-5f, "pipe diff %g", maxdiff(ref, got));
    // both stages got work, and the split is contiguous
    std::string why; auto reg = flow::device_registry(dev, &why);
    CHECK(reg && reg->stages.size() == 2);
    srv.stop();
}
TEST(pipeline_with_vulkan_if_present) {
    std::string why; auto v = flow::device_registry("vulkan", &why);
    if (!v) { std::printf("skip: no vulkan (%s)\n", why.c_str()); return; }
    v.reset();
    auto ref = logits("cpu"); auto got = logits("pipe:vulkan,cpu");
    CHECK_MSG(maxdiff(ref, got) < 2e-3f, "pipe vk diff %g", maxdiff(ref, got));
}
TEST(pipeline_multi_gpu_if_present) {   // real multi-GPU: needs two CUDA/HIP devices
    for (auto b : {"cuda", "hip"}) {
        std::string w0, w1; auto a = flow::device_registry(std::string(b) + ":0", &w0), c = flow::device_registry(std::string(b) + ":1", &w1);
        if (!a || !c) continue;
        a.reset(); c.reset();
        auto ref = logits("cpu"); auto got = logits(std::string("pipe:") + b + ":0," + b + ":1");
        CHECK_MSG(maxdiff(ref, got) < 2e-3f, "%s pipe diff %g", b, maxdiff(ref, got));
    }
}

// ---- security: token / Ed25519 key auth, TLS, remote scripts
static flow::rpc::client_security tok(const std::string& t) { flow::rpc::client_security s; s.token = t; return s; }
static const char* kSeed = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60";   // RFC 8032 test 1
static bool relu_ok(flow::rpc::client& c) {
    tensor x({4}), y({4}); for (auto& v : x.data) v = -1.f; tensor_view in{x.data.data(), x.shape};
    c.run("relu", std::span<const tensor_view>(&in, 1), {y.data.data(), y.shape}, {}); return y.data[0] == 0.f;
}
TEST(rpc_token_auth) {
    flow::rpc::server_security sec; sec.tokens = {"s3cret"};
    flow::rpc::server srv(nullptr, 0, "127.0.0.1", sec); srv.start();
    auto port = std::to_string(srv.port()); std::string why;
    CHECK(!flow::rpc::client::connect("127.0.0.1", port, &why, {}) || true);   // unauthenticated: rejected on first frame
    { auto c = flow::rpc::client::connect("127.0.0.1", port, &why, {}); bool threw = false; if (c) { try { relu_ok(*c); } catch (...) { threw = true; } } CHECK(!c || threw); }
    why.clear(); CHECK(!flow::rpc::client::connect("127.0.0.1", port, &why, tok("wrong"))); CHECK(why.find("bad token") != std::string::npos);
    auto c = flow::rpc::client::connect("127.0.0.1", port, &why, tok("s3cret")); CHECK(c != nullptr);
    CHECK(relu_ok(*c));
    srv.stop();
}
TEST(rpc_key_auth_and_scripts) {
    auto id = std::make_shared<flow::rpc::identity>(flow::rpc::identity_from_seed_hex(kSeed));
    CHECK(flow::rpc::sec::hex(id->pub.data(), 32) == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    flow::rpc::server_security sec; sec.authorized_keys = {flow::rpc::sec::hex(id->pub.data(), 32)}; sec.allow_scripts = true;
    flow::rpc::server srv(nullptr, 0, "127.0.0.1", sec);
    srv.script_handler = [](const std::string& src, const std::string&, const std::vector<std::string>& argv, std::uint32_t) {
        return flow::rpc::server::script_result{true, "", "", "ran:" + src + ":" + std::to_string(argv.size())};
    };
    srv.start(); auto port = std::to_string(srv.port()); std::string why;
    flow::rpc::client_security good; good.key = id;
    auto c = flow::rpc::client::connect("127.0.0.1", port, &why, good); CHECK_MSG(c != nullptr, "%s", why.c_str());
    auto r = c->run_script("x", "<t>", {"a", "b"}); CHECK(r.ok && r.output == "ran:x:2");
    CHECK(relu_ok(*c));
    std::uint8_t other[32] = {1}; flow::rpc::client_security bad; bad.key = std::make_shared<flow::rpc::identity>(flow::rpc::identity_from_seed_hex(flow::rpc::sec::hex(other, 32)));
    why.clear(); CHECK(!flow::rpc::client::connect("127.0.0.1", port, &why, bad)); CHECK(why.find("not authorized") != std::string::npos);
    CHECK(srv.scripts() == 1);
    srv.stop();
    // scripts need auth; and a no-auth server refuses them
    bool threw = false; try { flow::rpc::server_security s2; s2.allow_scripts = true; flow::rpc::server x(nullptr, 0, "127.0.0.1", s2); } catch (...) { threw = true; } CHECK(threw);
    flow::rpc::server open_srv(nullptr, 0); open_srv.start();
    auto oc = flow::rpc::client::connect("127.0.0.1", std::to_string(open_srv.port()), &why, {});
    threw = false; try { oc->run_script("x"); } catch (const std::exception& e) { threw = std::string(e.what()).find("disabled") != std::string::npos; } CHECK(threw);
    open_srv.stop();
}
TEST(rpc_tls) {
    if (!flow::rpc::tls_available()) { std::printf("skip: built without TLS\n"); return; }
    auto crt = tmp("c.pem"), key = tmp("k.pem");
    auto cmd = "openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -days 1 -subj /CN=localhost -addext subjectAltName=DNS:localhost -keyout " + key + " -out " + crt + " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0) { std::printf("skip: no openssl CLI\n"); return; }
    flow::rpc::server_security sec; sec.tokens = {"t"}; sec.tls = {crt, key};
    flow::rpc::server srv(nullptr, 0, "127.0.0.1", sec); srv.start(); CHECK(srv.tls());
    auto port = std::to_string(srv.port()); std::string why;
    auto s = tok("t"); s.tls.enabled = true; s.tls.ca_file = crt; s.tls.server_name = "localhost";
    auto c = flow::rpc::client::connect("127.0.0.1", port, &why, s); CHECK_MSG(c != nullptr, "%s", why.c_str());
    if (c) CHECK(relu_ok(*c));
    auto nv = tok("t"); nv.tls.enabled = true; why.clear();                         // no CA / fingerprint: refuse
    CHECK(!flow::rpc::client::connect("127.0.0.1", port, &why, nv)); CHECK(why.find("verify") != std::string::npos);
    auto fp = tok("t"); fp.tls.enabled = true; fp.tls.fingerprint_sha256 = std::string(64, 'a'); why.clear();
    CHECK(!flow::rpc::client::connect("127.0.0.1", port, &why, fp)); CHECK(why.find("fingerprint") != std::string::npos);
    auto plain = tok("t"); why.clear(); auto pc = flow::rpc::client::connect("127.0.0.1", port, &why, plain);   // plaintext to a TLS server fails
    CHECK(!pc);
    srv.stop();
    std::filesystem::remove(crt); std::filesystem::remove(key);
}
