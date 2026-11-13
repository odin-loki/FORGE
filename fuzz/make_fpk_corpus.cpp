// Writes the seed corpus of fuzz_fpk with the code under test, so the seeds are valid containers:
//   cmake --build <build> --target fuzz_make_fpk_corpus && <build>/fuzz_make_fpk_corpus fuzz/corpus/fpk
// A model of three small tensors (f32, f16, q8_0), then the same file signed, sealed with the old single-AEAD
// scheme, sealed with the chunked scheme (64-byte chunks, so a tensor spans several), and sealed + signed.
// sections_aligned.fpk carries the optional sections (state, adapters, calib) and 4096-byte tensor alignment.
// sharded.fpk is the main file of a sharded model (packed, 128-byte blobs) without its blobs.
// The sealed seeds use the fixed key of fuzz/fuzz_fpk.cpp. regress_* files are left alone.
#include <forge/modelfile/sign.hpp>
#include <forge/modelfile/shard.hpp>
#include <forge/modelfile/adapters.hpp>
#include <forge/modelfile/calib.hpp>
#include <forge/modelfile/state_schema.hpp>
#include <cstdio>
#include <filesystem>

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    using namespace forge;
    if (argc != 2) { std::fprintf(stderr, "usage: fuzz_make_fpk_corpus <corpus dir>\n"); return 2; }
    fs::path out = argv[1];
    fs::create_directories(out);
    auto at = [&](const char* n) { return (out / n).string(); };
    modelfile::model_key key{}; for (std::size_t i = 0; i < 32; ++i) key[i] = std::uint8_t(0x40 + i);
    std::array<std::uint8_t, 32> seed{}; for (std::size_t i = 0; i < 32; ++i) seed[i] = std::uint8_t(i + 1);
    auto kp = modelfile::keypair_from_seed(seed);

    modelfile::writer w;
    std::vector<float> a(16), b(8), c(64);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = float(i) * 0.25f - 2;
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = float(i) - 3.5f;
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = float(int(i * 37 % 64) - 32) / 16;
    w.add_tensor("w", {4, 4}, a, "f32"); w.add_tensor("e", {8}, b, "f16"); w.add_tensor("q", {2, 32}, c, "q8_0");
    w.set_manifest(json::value(json::object{{"format", "fpk"}, {"forge_format_version", 1.0}, {"architecture", "seed"}, {"session", "one_shot"}}));
    w.set_section("blueprint", "blueprint \"seed\" opset 1\ninput x : f32[4]\ngraph { y = linear(x, W.w) }\noutput y : f32[4]\nsession one_shot\n");
    w.write(at("plain.fpk"));
    modelfile::sign_file(at("plain.fpk"), at("signed.fpk"), kp);
    modelfile::encrypt_file(at("plain.fpk"), at("sealed_single.fpk"), key, {.single_aead = true});
    modelfile::encrypt_file(at("plain.fpk"), at("sealed_chunked.fpk"), key, {.chunk_bytes = 64});
    modelfile::sign_file(at("sealed_chunked.fpk"), at("sealed_chunked_signed.fpk"), kp);
    {   // the optional sections: a declared state slot, an embedded adapter, calibration statistics; page-aligned tensors
        modelfile::writer x; x.set_alignment(4096);
        x.add_tensor("w", {4, 4}, a, "f32");
        x.add_tensor("adapter.s.w.lora_a", {2, 4}, std::span<const float>(b.data(), 8), "f32"); x.add_tensor("adapter.s.w.lora_b", {4, 2}, std::span<const float>(b.data(), 8), "f16");
        x.set_manifest(json::value(json::object{{"format", "fpk"}, {"forge_format_version", 1.0}, {"architecture", "seed"}, {"session", "one_shot"}}));
        x.set_section("blueprint", "blueprint \"seed\" opset 1\ninput x : f32[4]\ngraph { y = linear(x, W.w) }\noutput y : f32[4]\nsession one_shot\n");
        modelfile::state_schema sc; sc.slots.push_back({"kv.0.k", false, "", {2, 8}, "f32"}); sc.slots.push_back({"s.0", true, "causal_conv", {3, 4}, "f32"});
        x.set_section(modelfile::state_section, sc.dump());
        modelfile::adapter_info ad; ad.name = "s"; ad.alpha = 4; ad.rank = 2; ad.targets = {"w"};
        x.set_section(modelfile::adapters_section, modelfile::dump_adapters({ad}));
        modelfile::calib_table ct; ct.tensors["w"] = modelfile::tensor_stats::of(a);
        x.set_section(modelfile::calib_section, ct.dump());
        x.write(at("sections_aligned.fpk"));
        modelfile::reader r(at("sections_aligned.fpk"));
        if (modelfile::read_adapters(r).size() != 1 || !r.misaligned_tensors().empty()) { std::fprintf(stderr, "sections_aligned.fpk is not as intended\n"); return 1; }
    }
    {   // a sharded main file (its blobs are not part of the corpus: the harness sees them as missing)
        auto td = out / "shard_tmp"; fs::remove_all(td); fs::create_directories(td);
        modelfile::shard_split(at("plain.fpk"), (td / "sharded.fpk").string(), {.shard_bytes = 128});
        fs::copy_file(td / "sharded.fpk", out / "sharded.fpk", fs::copy_options::overwrite_existing);
        fs::remove_all(td);
        std::printf("%-28s %zu bytes\n", "sharded.fpk", std::size_t(fs::file_size(out / "sharded.fpk")));
    }
    for (auto* n : {"plain.fpk", "signed.fpk", "sealed_single.fpk", "sealed_chunked.fpk", "sealed_chunked_signed.fpk", "sections_aligned.fpk"}) {
        modelfile::reader r(at(n), key);
        if (!r.verify().empty()) { std::fprintf(stderr, "%s does not verify\n", n); return 1; }
        std::printf("%-28s %zu bytes\n", n, r.file_size());
    }
    return 0;
}
