#pragma once
// forge::os::service control: define, install, start, stop, query and uninstall a per-user background service.
//   Linux/FreeBSD with systemd: a user unit in $XDG_CONFIG_HOME/systemd/user driven by `systemctl --user`.
//   macOS: a LaunchAgent plist in ~/Library/LaunchAgents driven by `launchctl`.
//   Windows: a Service Control Manager service via `sc.exe` (needs an elevated process).
//   Android / iOS: unsupported.
// The unit-file text is produced by render_*() so it can be inspected or tested without touching the system;
// set service_spec::dir to install somewhere other than the per-user default (also used by tests).
// Guide: docs/os-wrapper.md.
#include <forge/os/process.hpp>
#include <forge/os/service.hpp>
#include <forge/os/users.hpp>
#include <fstream>

namespace forge::os::service {
struct service_spec {
    std::string name, description;
    std::vector<std::string> argv;               // program and arguments
    std::string working_dir;
    std::map<std::string, std::string> env;
    bool restart_on_failure = true, start_at_login = true;
    std::string dir;                             // override the unit directory (empty = per-user default)
};
enum class service_state { not_installed, stopped, running, failed, unknown };
inline const char* state_name(service_state s) {
    switch (s) { case service_state::not_installed: return "not_installed"; case service_state::stopped: return "stopped"; case service_state::running: return "running"; case service_state::failed: return "failed"; default: return "unknown"; }
}
inline bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 128) return false;
    for (char c : n) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.' || c == '@')) return false;
    return true;
}
inline std::string quote_systemd_(const std::string& a) {
    bool plain = !a.empty(); for (char c : a) plain &= !(c == ' ' || c == '"' || c == '\\' || c == '\'' || c == '$' || c == '%' || c == '\t');
    if (plain) return a;
    std::string o = "\""; for (char c : a) { if (c == '"' || c == '\\') o += '\\'; if (c == '%') { o += "%%"; continue; } if (c == '$') { o += "$$"; continue; } o += c; } return o + "\"";
}
inline std::string xml_esc_(const std::string& s) { std::string o; for (char c : s) { if (c == '&') o += "&amp;"; else if (c == '<') o += "&lt;"; else if (c == '>') o += "&gt;"; else if (c == '"') o += "&quot;"; else o += c; } return o; }
inline std::string render_systemd_unit(const service_spec& s) {
    std::string u = "[Unit]\nDescription=" + (s.description.empty() ? s.name : s.description) + "\n\n[Service]\nType=simple\nExecStart=";
    for (std::size_t i = 0; i < s.argv.size(); ++i) u += (i ? " " : "") + quote_systemd_(s.argv[i]);
    u += "\n";
    if (!s.working_dir.empty()) u += "WorkingDirectory=" + quote_systemd_(s.working_dir) + "\n";
    for (auto& [k, v] : s.env) u += "Environment=" + quote_systemd_(k + "=" + v) + "\n";
    u += std::string("Restart=") + (s.restart_on_failure ? "on-failure" : "no") + "\n";
    u += "\n[Install]\nWantedBy=default.target\n";
    return u;
}
inline std::string render_launchd_plist(const service_spec& s) {
    std::string p = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n";
    p += "  <key>Label</key><string>" + xml_esc_(s.name) + "</string>\n  <key>ProgramArguments</key>\n  <array>\n";
    for (auto& a : s.argv) p += "    <string>" + xml_esc_(a) + "</string>\n";
    p += "  </array>\n";
    if (!s.working_dir.empty()) p += "  <key>WorkingDirectory</key><string>" + xml_esc_(s.working_dir) + "</string>\n";
    if (!s.env.empty()) { p += "  <key>EnvironmentVariables</key>\n  <dict>\n"; for (auto& [k, v] : s.env) p += "    <key>" + xml_esc_(k) + "</key><string>" + xml_esc_(v) + "</string>\n"; p += "  </dict>\n"; }
    p += std::string("  <key>RunAtLoad</key><") + (s.start_at_login ? "true" : "false") + "/>\n";
    p += std::string("  <key>KeepAlive</key>") + (s.restart_on_failure ? "<dict><key>SuccessfulExit</key><false/></dict>" : "<false/>") + "\n";
    p += "</dict>\n</plist>\n";
    return p;
}
inline std::string unit_dir(const service_spec& s) {
    if (!s.dir.empty()) return s.dir;
#if defined(__APPLE__)
    return home_dir() + "/Library/LaunchAgents";
#else
    return config_dir() + "/systemd/user";
#endif
}
inline std::string unit_path(const service_spec& s) {
#if defined(__APPLE__)
    return unit_dir(s) + "/" + s.name + ".plist";
#else
    return unit_dir(s) + "/" + s.name + ".service";
#endif
}
inline result<proc_result> ctl_(std::vector<std::string> argv) { spawn_spec sp; sp.argv = std::move(argv); sp.timeout_ms = 30000; return spawn_wait(sp); }
inline status ctl_ok_(std::vector<std::string> argv) {
    auto what = argv.empty() ? std::string() : argv[0] + (argv.size() > 1 ? " " + argv[1] : "") + (argv.size() > 2 ? " " + argv[2] : "");
    auto r = ctl_(std::move(argv)); if (!r) return {{}, r.err};
    if (r->exit_code != 0) return {{}, {r->exit_code, what + ": " + (r->err.empty() ? r->out : r->err)}};
    return {};
}
inline bool systemd_user_available_() { return !find_executable("systemctl").empty() && (!getenv("XDG_RUNTIME_DIR").empty() || !getenv("DBUS_SESSION_BUS_ADDRESS").empty()); }

// Write the unit (and register it with the manager unless spec.dir overrides the location).
inline status install(const service_spec& s) {
    if (!valid_name(s.name)) return {{}, {EINVAL, "service: invalid name '" + s.name + "'"}};
    if (s.argv.empty()) return {{}, {EINVAL, "service: argv is empty"}};
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {{}, unsupported("service install")};
#elif defined(FORGE_OS_WINDOWS)
    std::string bin; for (auto& a : s.argv) bin += (bin.empty() ? "" : " ") + (a.find(' ') != std::string::npos ? "\"" + a + "\"" : a);
    if (auto st = ctl_ok_({"sc.exe", "create", s.name, "binPath=", bin, "start=", s.start_at_login ? "auto" : "demand", "DisplayName=", s.description.empty() ? s.name : s.description}); !st) return st;
    if (s.restart_on_failure) ctl_ok_({"sc.exe", "failure", s.name, "reset=", "86400", "actions=", "restart/5000"});
    return {};
#else
    std::error_code ec; std::filesystem::create_directories(unit_dir(s), ec);
    if (ec) return {{}, {ec.value(), "service: mkdir " + unit_dir(s) + ": " + ec.message()}};
#if defined(__APPLE__)
    auto text = render_launchd_plist(s);
#else
    auto text = render_systemd_unit(s);
#endif
    if (auto w = fs::write_file_atomic(unit_path(s), text); !w) return w;
    if (!s.dir.empty()) return {};
#if defined(__APPLE__)
    return {};
#else
    if (!systemd_user_available_()) return {};   // file written; the manager picks it up at next login
    if (auto st = ctl_ok_({"systemctl", "--user", "daemon-reload"}); !st) return st;
    if (s.start_at_login) return ctl_ok_({"systemctl", "--user", "enable", s.name + ".service"});
    return {};
#endif
#endif
}
inline status start(const service_spec& s) {
#if defined(FORGE_OS_WINDOWS)
    return ctl_ok_({"sc.exe", "start", s.name});
#elif defined(__APPLE__) && !defined(FORGE_OS_IOS)
    return ctl_ok_({"launchctl", "load", "-w", unit_path(s)});
#elif defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    (void)s; return {{}, unsupported("service start")};
#else
    if (!systemd_user_available_()) return {{}, unsupported("service start (no systemd user session)")};
    return ctl_ok_({"systemctl", "--user", "start", s.name + ".service"});
#endif
}
inline status stop(const service_spec& s) {
#if defined(FORGE_OS_WINDOWS)
    return ctl_ok_({"sc.exe", "stop", s.name});
#elif defined(__APPLE__) && !defined(FORGE_OS_IOS)
    return ctl_ok_({"launchctl", "unload", unit_path(s)});
#elif defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    (void)s; return {{}, unsupported("service stop")};
#else
    if (!systemd_user_available_()) return {{}, unsupported("service stop (no systemd user session)")};
    return ctl_ok_({"systemctl", "--user", "stop", s.name + ".service"});
#endif
}
inline service_state query(const service_spec& s) {
#if defined(FORGE_OS_WINDOWS)
    auto r = ctl_({"sc.exe", "query", s.name}); if (!r) return service_state::unknown;
    if (r->exit_code == 1060) return service_state::not_installed;
    if (r->out.find("RUNNING") != std::string::npos) return service_state::running;
    if (r->out.find("STOPPED") != std::string::npos) return service_state::stopped;
    return service_state::unknown;
#elif defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    (void)s; return service_state::unknown;
#else
    if (!fs::exists(unit_path(s))) return service_state::not_installed;
#if defined(__APPLE__)
    auto r = ctl_({"launchctl", "list", s.name}); if (!r) return service_state::unknown;
    return r->exit_code == 0 ? service_state::running : service_state::stopped;
#else
    if (!systemd_user_available_() || !s.dir.empty()) return service_state::stopped;
    auto r = ctl_({"systemctl", "--user", "is-active", s.name + ".service"}); if (!r) return service_state::unknown;
    auto o = r->out; while (!o.empty() && (o.back() == '\n' || o.back() == ' ')) o.pop_back();
    return o == "active" || o == "activating" || o == "reloading" ? service_state::running : o == "failed" ? service_state::failed : service_state::stopped;
#endif
#endif
}
inline status uninstall(const service_spec& s) {
#if defined(FORGE_OS_WINDOWS)
    ctl_({"sc.exe", "stop", s.name}); return ctl_ok_({"sc.exe", "delete", s.name});
#elif defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    (void)s; return {{}, unsupported("service uninstall")};
#else
    if (s.dir.empty()) {
#if defined(__APPLE__)
        ctl_({"launchctl", "unload", unit_path(s)});
#else
        if (systemd_user_available_()) { ctl_({"systemctl", "--user", "disable", "--now", s.name + ".service"}); }
#endif
    }
    std::error_code ec; std::filesystem::remove(fs::u8path(unit_path(s)), ec);
    if (ec) return {{}, {ec.value(), "service: remove " + unit_path(s) + ": " + ec.message()}};
#if !defined(__APPLE__)
    if (s.dir.empty() && systemd_user_available_()) ctl_({"systemctl", "--user", "daemon-reload"});
#endif
    return {};
#endif
}
} // namespace forge::os::service
