#pragma once
// forge::script ahead-of-time compilation, internal interface (docs/aot.md).
//
// An AOT binary carries a *payload* (the script source, the sources of its statically resolved imports and the
// policy baked in at compile time) and, in native mode, one natively compiled C++ function per script function.
// At start-up the runtime parses the payload with the same parser the interpreter uses, numbers the nodes of the
// tree with index_program() (the code generator numbered them the same way, and a fingerprint proves it) and
// attaches the native bodies to their FnDecl. Interp::call_function / run_top then call the native body instead
// of the tree-walker or the VM. Everything else (values, heap, built-ins, errors, limits, capabilities, the OS
// sandbox) is the interpreter's own code, so the semantics are shared, not re-implemented.
#include "interp.hpp"
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace forge::script { struct aot_options; }
namespace forge::script::aot {

// Bump when the generated code's contract with the runtime changes (helpers in rt/forge_aot_rt.hpp, the node
// numbering, the payload encoding). A binary whose generated code was built for another value refuses to start.
inline constexpr std::uint32_t abi_version = 1;

using native_fn = Value (*)(Interp&, const Ref<Env>&);

// Deterministic numbering of one parsed program: every Node and FnDecl reachable from `top`, in the parser's own
// pre-order (kids, extra, fn (parameter defaults, body), class (field initialisers, methods)). Shared subtrees are
// numbered once. `fingerprint` covers every field the code generator reads.
struct program_index {
    std::vector<const Node*> nodes;
    std::vector<const FnDecl*> fns;                 // fns[0] is the program's top-level function
    std::uint64_t fingerprint = 0;
    int node_of(const Node& n) const;               // -1 when the node is not part of this program
    int fn_of(const FnDecl& f) const;
    std::unordered_map<const Node*, int> node_ids;
    std::unordered_map<const FnDecl*, int> fn_ids;
};
program_index index_program(const FnDecl& top);

// One compiled unit: the main script or a bundled module. The generated code defines one of these per unit; the
// runtime fills `*nodes` when it binds the unit.
struct unit {
    const char* name;                               // unit name (payload::name / module::key)
    std::uint64_t fingerprint;
    std::uint32_t n_nodes, n_fns;
    const native_fn* fns;                           // n_fns entries, indexed like program_index::fns
    const Node* const** nodes;                      // set to the bound program's node table
};

// What `forge script --aot` bakes into a binary. Run-time options can narrow it, never widen it, unless
// allow_widen was set at compile time.
struct baked_policy {
    bool allow_fs = true, allow_net = true, allow_proc = true, allow_env = true;
    std::uint64_t max_steps = 0, timeout_ms = 0, heap_bytes = 0;
    std::int32_t max_depth = 800;
    std::uint8_t os_sandbox = 1;                    // options::os_sandbox_mode
    bool static_check = true, strict_check = false, debug = false;
    bool allow_widen = false;
    std::string policy_json, policy_origin;         // FSS policy text baked with --policy (empty: none)
};

struct payload {
    std::string name;                               // main script name, as given at compile time
    std::string source;
    struct module { std::string importer, path, key, source; };   // importer: unit name; path: as written; key: unit name of the module
    std::vector<module> modules;
    baked_policy baked;
};
std::vector<std::uint8_t> encode_payload(const payload& p);
bool decode_payload(const std::uint8_t* data, std::size_t size, payload& out, std::string* why = nullptr);

// Trailer appended (after the payload) to a runner stub in embed mode:
//   payload | "FORGEAOT" u32 version u64 payload_len u8[32] BLAKE3(payload) "TOAEGROF"
inline constexpr std::size_t trailer_size = 8 + 4 + 8 + 32 + 8;
std::vector<std::uint8_t> make_trailer(const std::vector<std::uint8_t>& payload_bytes);
// Reads the payload appended to `exe`. 0 = none, 1 = found (out filled), -1 = present but damaged (why filled).
int read_embedded(const std::string& exe, std::vector<std::uint8_t>& out, std::string* why = nullptr);

// ---- compile side (aot_codegen.cpp, used by aot_compile.cpp)
struct translated { std::string cxx; payload pl; int delegated = 0, unboxed = 0; std::set<std::string> delegated_kinds; };
// Parse, check, resolve static imports and build the payload; with want_cxx also generate the C++ source.
translated translate_program(std::string_view src, const std::string& file, const aot_options& o, bool want_cxx);

// ---- hooks called by the interpreter (no-ops unless this process is an AOT binary)
// After a program is parsed: attach native bodies when `name` is a compiled unit of this binary.
void bind_program(const std::shared_ptr<const FnDecl>& top, const std::string& name);
// import_module: a module bundled at compile time? Fills key (the module's unit name) and source.
bool bundled_module(const Interp& in, const std::string& path, std::string& key, std::string& source);
// Call a bound native body (FnDecl::aot_body).
inline Value call_native(const FnDecl& d, Interp& in, const Ref<Env>& env) { return reinterpret_cast<native_fn>(d.aot_body)(in, env); }

// ---- entry points for generated code and the runner stub (declared again in rt/forge_aot_rt.hpp)
int run_native(int argc, char** argv, const unsigned char* payload_bytes, std::size_t payload_size, const unit* units, std::size_t n_units, std::uint32_t abi);

} // namespace forge::script::aot
