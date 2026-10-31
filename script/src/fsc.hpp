#pragma once
// ForgeScript .fsc cache: the parsed and resolved program (the input to the bytecode compiler and the
// tree-walker), serialised so a cached run skips lexing, parsing and resolution. See docs/scripting.md.
//
// Layout (little-endian):
//   "FSC1" u32 vm_version  u8[32] key  u64 payload_len  payload  u8 signed  [u8[32] pub  u8[64] sig]
//   key = BLAKE3("forge-fsc\0" || u32 vm_version || file name || '\0' || source)
//   sig = Ed25519 over "forge-fsc-v1" || key || BLAKE3(payload)
// A cache file is used only when its key equals the key recomputed from the current source, the VM
// version matches, the payload decodes with every bound checked, and — when it is signed or the engine
// requires signatures — the signature verifies against a trusted public key. Anything else is ignored and
// the source is parsed normally (a stale or tampered cache never changes behaviour, only speed).
// Bytecode chunks are still compiled from the decoded tree and pass the VM verifier before running.
#include "ast.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace forge::script::fsc {

inline constexpr std::uint32_t vm_version = 1;   // bump when Node/FnDecl/ClassDecl or the compiler changes meaning

using key_t = std::array<std::uint8_t, 32>;
key_t key_of(std::string_view src, const std::string& file);
std::string key_hex(const key_t& k);

struct trust {
    std::vector<std::array<std::uint8_t, 32>> pubkeys;   // accepted signers
    bool require_signed = false;                         // reject unsigned caches
};

// Encode `top`. With a 32-byte seed the file is signed.
std::vector<std::uint8_t> encode(const FnDecl& top, const key_t& key, const std::array<std::uint8_t, 32>* seed = nullptr);
// Decode; nullopt + why on any mismatch, bound violation or failed signature.
std::optional<std::shared_ptr<const FnDecl>> decode(std::span<const std::uint8_t> bytes, const key_t& expect, const trust& t, std::string* why = nullptr);

// Cache directory: options::fsc_cache_dir, else $FORGE_SCRIPT_CACHE, else empty (cache off).
std::string path_for(const std::string& dir, const key_t& k);

} // namespace forge::script::fsc
