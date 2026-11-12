// forge::os::dbus codec: bytes from a peer must never crash, over-read or balloon; whatever decodes must
// re-encode to a message that decodes again and re-encodes to the same bytes (the encoder is canonical).
#include "fuzz.hpp"
#include <forge/os/dbus.hpp>
#include <cstdlib>
FORGE_FUZZ({
    namespace db = forge::os::dbus;
    db::limits lim; lim.max_message = 1u << 16; lim.max_nodes = 4096;
    std::string_view text(reinterpret_cast<const char*>(data), size);
    (void)db::parse_address(text); (void)db::valid_signature(text); (void)db::valid_single_type(text); (void)db::valid_object_path(text);
    auto sz = db::message_size(data, size, lim);
    if (!sz || !sz.value || sz.value > size) { (void)db::decode(data, size, lim); return; }
    auto m = db::decode(data, sz.value, lim);
    if (!m) return;
    auto e = db::encode(*m);
    if (!e) std::abort();                       // a decoded message is always encodable
    auto m2 = db::decode(reinterpret_cast<const std::uint8_t*>(e->data()), e->size(), lim);
    if (!m2) {                                   // only the size cap may differ: re-encoding drops padding, never adds
        if (e->size() > lim.max_message) return;
        std::abort();
    }
    auto e2 = db::encode(*m2);
    if (!e2 || *e2 != *e) std::abort();
})
