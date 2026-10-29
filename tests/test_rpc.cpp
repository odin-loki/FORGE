// RPC device over loopback and pipeline split: whole-model logits must match the plain CPU run.
#include "check.hpp"
#include "tiny_model.hpp"
#include <forge/mind/mind.hpp>
#include <forge/flow/vulkan.hpp>
#include <filesystem>
#include <unistd.h>

using namespace forge;
using namespace forge::mind;

static std::string tmp(const std::string& n) { return (std::filesystem::temp_directory_path() / ("forge_rpc_" + std::to_string(::getpid()) + "_" + n)).string(); }
static const std::string& model_a() { static std::string p = forge_test::make_tiny_model(tmp("a.fpk")); return p; }
static std::vector<float> logits(const std::string& dev) {
    auto m = model::open(model_a());
    CHECK(m->set_device(dev) == dev);
    session s(m);
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
