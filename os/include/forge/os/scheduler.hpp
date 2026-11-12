#pragma once
// forge::os::schedule: run a command for the current user on a schedule, through the OS's own scheduler.
//   Linux with a systemd user session: <name>.timer + <name>.service in $XDG_CONFIG_HOME/systemd/user
//                  (`systemctl --user enable --now <name>.timer`).  Otherwise (and on FreeBSD): the user's crontab,
//                  one line tagged "# forge-schedule:<name>" (crontab -l / crontab -).
//   macOS:         a LaunchAgent with StartInterval or StartCalendarInterval, bootstrapped into gui/<uid>.
//   Windows:       a Task Scheduler task from render_task_xml(), registered with `schtasks /Create /XML`.
//   Android / iOS: unsupported from native code (WorkManager / BGTaskScheduler belong to the app; see
//                  background.hpp for keeping work alive).
// A task runs either every interval_s seconds, or at calendar times: minute / hour / day (of month) / month /
// weekday, -1 = every (cron semantics; weekday 0 = Sunday). To keep the meaning identical on the four schedulers:
// minute is required once a coarser field is set, day and weekday are not combined, and for cron an interval must
// be a whole number of minutes that divides an hour, or of hours that divides a day.
// The definitions come from render_*() so they can be inspected and tested on any OS; task::dir writes them into
// a directory without registering anything.
#include <forge/os/service_control.hpp>

namespace forge::os::schedule {
struct task {
    std::string name;                 // 1-64 characters of [A-Za-z0-9._-]
    std::vector<std::string> argv;    // argv[0] should be an absolute path (schedulers run with a minimal PATH)
    std::int64_t interval_s = 0;      // > 0: every interval_s seconds (minimum 60); the calendar fields must then be -1
    int minute = -1, hour = -1, day = -1, month = -1, weekday = -1;
    std::string description, working_dir;
    std::string dir;                  // "" = register with the scheduler; else only write the definition files here
};
inline bool valid_name(const std::string& n) {
    if (n.empty() || n.size() > 64 || n[0] == '.' || n[0] == '-') return false;
    for (unsigned char c : n) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return false;
    return true;
}
inline status validate(const task& t) {
    auto bad = [](const std::string& m) { return status{{}, errno_error(EINVAL, "schedule: " + m)}; };
    if (!valid_name(t.name)) return bad("invalid name '" + t.name + "'");
    if (t.argv.empty() || t.argv[0].empty()) return bad("empty command");
    for (auto& a : t.argv) if (service::has_ctl_(a)) return bad("control character in command");
    if (service::has_ctl_(t.description) || service::has_ctl_(t.working_dir)) return bad("control character in description or working directory");
    const bool cal = t.minute >= 0 || t.hour >= 0 || t.day >= 0 || t.month >= 0 || t.weekday >= 0;
    if (t.interval_s > 0) {
        if (cal) return bad("interval_s and calendar fields are exclusive");
        if (t.interval_s < 60) return bad("interval below 60 s");
        return {};
    }
    if (t.interval_s < 0) return bad("negative interval");
    if (t.minute > 59 || t.hour > 23 || t.day == 0 || t.day > 31 || t.month == 0 || t.month > 12 || t.weekday > 6 || t.minute < -1 || t.hour < -1 || t.day < -1 || t.month < -1 || t.weekday < -1) return bad("calendar field out of range");
    if ((t.hour >= 0 || t.day >= 0 || t.month >= 0 || t.weekday >= 0) && t.minute < 0) return bad("minute is required when hour, day, month or weekday is set");
    if ((t.day >= 0 || t.month >= 0 || t.weekday >= 0) && t.hour < 0) return bad("hour is required when day, month or weekday is set");
    if (t.day >= 0 && t.weekday >= 0) return bad("day and weekday cannot be combined");
    if (t.month >= 0 && t.day < 0) return bad("month needs a day");
    return {};
}
namespace detail_schedule {
inline std::string marker(const std::string& name) { return "# forge-schedule:" + name; }
inline std::string field(int v) { return v < 0 ? "*" : std::to_string(v); }
inline std::string two(int v) { char b[8]; std::snprintf(b, sizeof b, "%02d", v); return b; }
}
// One crontab line (no trailing newline); "" when the interval cannot be expressed in cron.
inline std::string render_cron_line(const task& t) {
    std::string when;
    if (t.interval_s > 0) {
        if (t.interval_s % 60) return {};
        std::int64_t m = t.interval_s / 60;
        if (m < 60) { if (60 % m) return {}; when = m == 1 ? "* * * * *" : "*/" + std::to_string(m) + " * * * *"; }
        else { if (m % 60) return {}; std::int64_t h = m / 60; if (h > 24 || 24 % h) return {}; when = h == 1 ? "0 * * * *" : h == 24 ? "0 0 * * *" : "0 */" + std::to_string(h) + " * * *"; }
    } else {
        using detail_schedule::field;
        when = field(t.minute) + " " + field(t.hour) + " " + field(t.day) + " " + field(t.month) + " " + field(t.weekday);
    }
    std::string cmd; if (!t.working_dir.empty()) cmd = "cd " + service::sh_quote_(t.working_dir) + " && ";
    for (std::size_t i = 0; i < t.argv.size(); ++i) cmd += (i ? " " : "") + service::sh_quote_(t.argv[i]);
    std::string esc; for (char c : cmd) { if (c == '%') esc += '\\'; esc += c; }   // an unescaped % ends the command in a crontab
    return when + " " + esc + " " + detail_schedule::marker(t.name);
}
inline std::string render_systemd_service(const task& t) {
    std::string u = "[Unit]\nDescription=" + service::systemd_verbatim_(t.description.empty() ? t.name : t.description) + "\n\n[Service]\nType=oneshot\nExecStart=";
    for (std::size_t i = 0; i < t.argv.size(); ++i) u += (i ? " " : "") + service::quote_systemd_(t.argv[i]);
    u += "\n";
    if (!t.working_dir.empty()) u += "WorkingDirectory=" + service::systemd_verbatim_(t.working_dir) + "\n";
    return u;
}
inline std::string render_systemd_timer(const task& t) {
    std::string u = "[Unit]\nDescription=" + service::systemd_verbatim_(t.description.empty() ? t.name : t.description) + " (timer)\n\n[Timer]\n";
    if (t.interval_s > 0) u += "OnActiveSec=" + std::to_string(t.interval_s) + "s\nOnUnitActiveSec=" + std::to_string(t.interval_s) + "s\nAccuracySec=1s\n";
    else {
        static const char* const dow[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
        using detail_schedule::two;
        std::string c = t.weekday >= 0 ? std::string(dow[t.weekday]) + " " : std::string();
        c += "*-" + (t.month >= 0 ? two(t.month) : std::string("*")) + "-" + (t.day >= 0 ? two(t.day) : std::string("*")) + " ";
        c += (t.hour >= 0 ? two(t.hour) : std::string("*")) + ":" + (t.minute >= 0 ? two(t.minute) : std::string("*")) + ":00";
        u += "OnCalendar=" + c + "\nPersistent=true\n";
    }
    u += "Unit=" + t.name + ".service\n\n[Install]\nWantedBy=timers.target\n";
    return u;
}
inline std::string render_launchd_plist(const task& t) {
    std::string p = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n";
    p += "  <key>Label</key><string>" + service::xml_esc_(t.name) + "</string>\n  <key>ProgramArguments</key>\n  <array>\n";
    for (auto& a : t.argv) p += "    <string>" + service::xml_esc_(a) + "</string>\n";
    p += "  </array>\n";
    if (!t.working_dir.empty()) p += "  <key>WorkingDirectory</key><string>" + service::xml_esc_(t.working_dir) + "</string>\n";
    if (t.interval_s > 0) p += "  <key>StartInterval</key><integer>" + std::to_string(t.interval_s) + "</integer>\n";
    else {
        p += "  <key>StartCalendarInterval</key>\n  <dict>\n";
        auto f = [&](const char* k, int v) { if (v >= 0) p += std::string("    <key>") + k + "</key><integer>" + std::to_string(v) + "</integer>\n"; };
        f("Minute", t.minute); f("Hour", t.hour); f("Day", t.day); f("Month", t.month); f("Weekday", t.weekday);
        p += "  </dict>\n";
    }
    p += "  <key>RunAtLoad</key><false/>\n</dict>\n</plist>\n";
    return p;
}
// Task Scheduler 1.2 XML (declared UTF-16: install() writes it as UTF-16LE with a BOM).
inline std::string render_task_xml(const task& t) {
    using detail_schedule::two;
    static const char* const dow[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    static const char* const mon[] = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
    std::string trig;
    auto rep = [](const std::string& iso) { return "      <Repetition><Interval>" + iso + "</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>\n"; };
    if (t.interval_s > 0) {
        std::string iso = t.interval_s % 3600 == 0 ? "PT" + std::to_string(t.interval_s / 3600) + "H" : t.interval_s % 60 == 0 ? "PT" + std::to_string(t.interval_s / 60) + "M" : "PT" + std::to_string(t.interval_s) + "S";
        trig = "    <TimeTrigger>\n" + rep(iso) + "      <StartBoundary>2020-01-01T00:00:00</StartBoundary>\n      <Enabled>true</Enabled>\n    </TimeTrigger>\n";
    } else if (t.hour < 0) {   // every minute, or hourly at :minute
        trig = "    <TimeTrigger>\n" + rep(t.minute < 0 ? "PT1M" : "PT1H") + "      <StartBoundary>2020-01-01T00:" + two(t.minute < 0 ? 0 : t.minute) + ":00</StartBoundary>\n      <Enabled>true</Enabled>\n    </TimeTrigger>\n";
    } else {
        trig = "    <CalendarTrigger>\n      <StartBoundary>2020-01-01T" + two(t.hour) + ":" + two(t.minute) + ":00</StartBoundary>\n      <Enabled>true</Enabled>\n";
        if (t.weekday >= 0) trig += std::string("      <ScheduleByWeek><DaysOfWeek><") + dow[t.weekday] + " /></DaysOfWeek><WeeksInterval>1</WeeksInterval></ScheduleByWeek>\n";
        else if (t.day >= 0) {
            trig += "      <ScheduleByMonth><DaysOfMonth><Day>" + std::to_string(t.day) + "</Day></DaysOfMonth><Months>";
            for (int m = 1; m <= 12; ++m) if (t.month < 0 || t.month == m) trig += std::string("<") + mon[m - 1] + " />";
            trig += "</Months></ScheduleByMonth>\n";
        } else trig += "      <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay>\n";
        trig += "    </CalendarTrigger>\n";
    }
    std::string x = "<?xml version=\"1.0\" encoding=\"UTF-16\"?>\n<Task version=\"1.2\" xmlns=\"http://schemas.microsoft.com/windows/2004/02/mit/task\">\n";
    x += "  <RegistrationInfo>\n    <Description>" + service::xml_esc_(t.description.empty() ? t.name : t.description) + "</Description>\n  </RegistrationInfo>\n  <Triggers>\n" + trig + "  </Triggers>\n";
    x += "  <Settings>\n    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n"
         "    <StartWhenAvailable>true</StartWhenAvailable>\n    <Enabled>true</Enabled>\n    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n  </Settings>\n";
    x += "  <Actions Context=\"Author\">\n    <Exec>\n      <Command>" + service::xml_esc_(t.argv.empty() ? std::string() : t.argv[0]) + "</Command>\n";
    if (t.argv.size() > 1) x += "      <Arguments>" + service::xml_esc_(service::win_cmdline_(t.argv, 1)) + "</Arguments>\n";
    if (!t.working_dir.empty()) x += "      <WorkingDirectory>" + service::xml_esc_(t.working_dir) + "</WorkingDirectory>\n";
    x += "    </Exec>\n  </Actions>\n</Task>\n";
    return x;
}

// systemd, cron, launchd, schtasks, or none (nothing usable: Android, iOS, a Linux without systemd session and crontab).
inline std::string backend() {
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return "none";
#elif defined(FORGE_OS_WINDOWS)
    return "schtasks";
#elif defined(__APPLE__)
    return "launchd";
#elif defined(FORGE_OS_FREEBSD)
    return find_executable("crontab").empty() ? "none" : "cron";
#else
    if (service::systemd_user_available_() && os::stat(getenv("XDG_RUNTIME_DIR") + "/systemd").exists) return "systemd";
    return find_executable("crontab").empty() ? "none" : "cron";
#endif
}
namespace detail_schedule {
inline status write_text(const std::string& path, const std::string& text) {
    std::error_code ec; std::filesystem::create_directories(fs::u8path(dirname(path)), ec);
    if (ec) return {{}, {ec.value(), "schedule: mkdir " + dirname(path) + ": " + ec.message()}};
    return fs::write_file_atomic(path, text);
}
inline std::string utf16_bom(const std::string& utf8) {
    std::string bytes = "\xFF\xFE";
    for (std::size_t i = 0; i < utf8.size();) {   // UTF-8 -> UTF-16LE (the same on every host: tests read it back)
        unsigned char c = static_cast<unsigned char>(utf8[i]); std::uint32_t cp = 0xFFFD; std::size_t n = 1;
        if (c < 0x80) cp = c; else if ((c >> 5) == 6 && i + 1 < utf8.size()) { cp = ((c & 0x1fu) << 6) | (static_cast<unsigned char>(utf8[i + 1]) & 0x3fu); n = 2; }
        else if ((c >> 4) == 14 && i + 2 < utf8.size()) { cp = ((c & 0x0fu) << 12) | ((static_cast<unsigned char>(utf8[i + 1]) & 0x3fu) << 6) | (static_cast<unsigned char>(utf8[i + 2]) & 0x3fu); n = 3; }
        else if ((c >> 3) == 30 && i + 3 < utf8.size()) { cp = ((c & 0x07u) << 18) | ((static_cast<unsigned char>(utf8[i + 1]) & 0x3fu) << 12) | ((static_cast<unsigned char>(utf8[i + 2]) & 0x3fu) << 6) | (static_cast<unsigned char>(utf8[i + 3]) & 0x3fu); n = 4; }
        i += n;
        auto put = [&](std::uint32_t u) { bytes.push_back(char(u & 0xff)); bytes.push_back(char((u >> 8) & 0xff)); };
        if (cp >= 0x10000) { cp -= 0x10000; put(0xD800 + (cp >> 10)); put(0xDC00 + (cp & 0x3ff)); } else put(cp);
    }
    return bytes;
}
// The user's crontab without the line tagged for `name` (and whether such a line was there).
inline std::string cron_without(const std::string& crontab, const std::string& name, bool* found = nullptr) {
    std::string out, tag = marker(name); std::size_t i = 0; if (found) *found = false;
    while (i < crontab.size()) {
        auto e = crontab.find('\n', i); std::string line = crontab.substr(i, e == std::string::npos ? std::string::npos : e - i); i = e == std::string::npos ? crontab.size() : e + 1;
        bool hit = line.size() >= tag.size() && line.compare(line.size() - tag.size(), tag.size(), tag) == 0;
        if (hit) { if (found) *found = true; continue; }
        out += line; out += '\n';
    }
    return out;
}
#if defined(FORGE_OS_POSIX) && !defined(FORGE_OS_ANDROID) && !defined(FORGE_OS_IOS)
inline result<std::string> crontab_read() {
    spawn_spec sp; sp.argv = {"crontab", "-l"}; sp.timeout_ms = 15000;
    auto r = spawn_wait(sp); if (!r) return {{}, r.err};
    return {r->exit_code == 0 ? r->out : std::string(), {}};   // "no crontab for user" exits 1
}
inline status crontab_write(const std::string& text) {
    spawn_spec sp; sp.argv = {"crontab", "-"}; sp.stdin_data = text; sp.timeout_ms = 15000;
    auto r = spawn_wait(sp); if (!r) return {{}, r.err};
    if (r->exit_code != 0) return {{}, {r->exit_code, "schedule: crontab: " + (r->err.empty() ? r->out : r->err)}};
    return {};
}
#endif
}
// The files install() writes for a task with dir set (and, for systemd / launchd, without).
inline std::vector<std::string> definition_files(const task& t) {
#if defined(FORGE_OS_WINDOWS)
    return t.dir.empty() ? std::vector<std::string>{} : std::vector<std::string>{t.dir + "\\" + t.name + ".xml"};
#elif defined(__APPLE__)
    return {(t.dir.empty() ? home_dir() + "/Library/LaunchAgents" : t.dir) + "/" + t.name + ".plist"};
#elif defined(FORGE_OS_FREEBSD) || defined(FORGE_OS_ANDROID)
    return t.dir.empty() ? std::vector<std::string>{} : std::vector<std::string>{t.dir + "/" + t.name + ".cron"};
#else
    if (!t.dir.empty()) return {t.dir + "/" + t.name + ".timer", t.dir + "/" + t.name + ".service", t.dir + "/" + t.name + ".cron"};
    if (backend() != "systemd") return {};
    std::string d = config_dir() + "/systemd/user"; return {d + "/" + t.name + ".timer", d + "/" + t.name + ".service"};
#endif
}
inline status install(const task& t) {
    if (auto v = validate(t); !v) return v;
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {{}, unsupported("schedule (use WorkManager / BGTaskScheduler from the app)")};
#elif defined(FORGE_OS_WINDOWS)
    std::string xml_path = t.dir.empty() ? temp_dir() + "\\forge-schedule-" + t.name + "-" + std::to_string(pid()) + ".xml" : t.dir + "\\" + t.name + ".xml";
    if (auto w = detail_schedule::write_text(xml_path, detail_schedule::utf16_bom(render_task_xml(t))); !w) return w;
    if (!t.dir.empty()) return {};
    auto r = service::ctl_ok_({"schtasks.exe", "/Create", "/TN", t.name, "/XML", xml_path, "/F"});
    os::remove(xml_path); return r;
#elif defined(__APPLE__)
    auto files = definition_files(t);
    if (auto w = detail_schedule::write_text(files[0], render_launchd_plist(t)); !w) return w;
    if (!t.dir.empty()) return {};
    std::string domain = "gui/" + std::to_string(::getuid());
    service::ctl_({"launchctl", "bootout", domain + "/" + t.name});   // replace a loaded older definition
    return service::ctl_ok_({"launchctl", "bootstrap", domain, files[0]});
#else
    std::string cron = render_cron_line(t);
    if (!t.dir.empty()) {
#if defined(FORGE_OS_LINUX)
        if (auto w = detail_schedule::write_text(t.dir + "/" + t.name + ".timer", render_systemd_timer(t)); !w) return w;
        if (auto w = detail_schedule::write_text(t.dir + "/" + t.name + ".service", render_systemd_service(t)); !w) return w;
#endif
        return detail_schedule::write_text(t.dir + "/" + t.name + ".cron", cron.empty() ? std::string("# interval not expressible in cron\n") : cron + "\n");
    }
    std::string be = backend();
#if defined(FORGE_OS_LINUX)
    if (be == "systemd") {
        auto files = definition_files(t);
        if (auto w = detail_schedule::write_text(files[0], render_systemd_timer(t)); !w) return w;
        if (auto w = detail_schedule::write_text(files[1], render_systemd_service(t)); !w) return w;
        if (auto st = service::ctl_ok_({"systemctl", "--user", "daemon-reload"}); !st) return st;
        return service::ctl_ok_({"systemctl", "--user", "enable", "--now", t.name + ".timer"});
    }
#endif
    if (be != "cron") return {{}, unsupported("schedule (no systemd user session and no crontab)")};
    if (cron.empty()) return {{}, errno_error(EINVAL, "schedule: cron cannot run every " + std::to_string(t.interval_s) + " s (use whole minutes dividing an hour, or hours dividing a day)")};
    auto cur = detail_schedule::crontab_read(); if (!cur) return {{}, cur.err};
    return detail_schedule::crontab_write(detail_schedule::cron_without(*cur, t.name) + cron + "\n");
#endif
}
inline result<bool> installed(const task& t) {
    if (!valid_name(t.name)) return {false, errno_error(EINVAL, "schedule: invalid name '" + t.name + "'")};
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {false, unsupported("schedule")};
#else
    if (!t.dir.empty()) { auto f = definition_files(t); return {!f.empty() && os::stat(f[0]).exists, {}}; }
#if defined(FORGE_OS_WINDOWS)
    auto r = service::ctl_({"schtasks.exe", "/Query", "/TN", t.name}); if (!r) return {false, r.err};
    return {r->exit_code == 0, {}};
#elif defined(__APPLE__)
    return {os::stat(definition_files(t)[0]).exists, {}};
#else
    std::string be = backend();
    if (be == "systemd") return {os::stat(definition_files(t)[0]).exists, {}};
    if (be != "cron") return {false, unsupported("schedule (no systemd user session and no crontab)")};
    auto cur = detail_schedule::crontab_read(); if (!cur) return {false, cur.err};
    bool found = false; detail_schedule::cron_without(*cur, t.name, &found); return {found, {}};
#endif
#endif
}
// Removing a task that is not installed succeeds.
inline status uninstall(const task& t) {
    if (!valid_name(t.name)) return {{}, errno_error(EINVAL, "schedule: invalid name '" + t.name + "'")};
#if defined(FORGE_OS_ANDROID) || defined(FORGE_OS_IOS)
    return {{}, unsupported("schedule")};
#else
    if (!t.dir.empty()) { for (auto& f : definition_files(t)) if (os::stat(f).exists) os::remove(f); return {}; }
#if defined(FORGE_OS_WINDOWS)
    auto in = installed(t); if (!in) return {{}, in.err};
    if (!*in) return {};
    return service::ctl_ok_({"schtasks.exe", "/Delete", "/TN", t.name, "/F"});
#elif defined(__APPLE__)
    auto f = definition_files(t)[0]; if (!os::stat(f).exists) return {};
    service::ctl_({"launchctl", "bootout", "gui/" + std::to_string(::getuid()) + "/" + t.name});
    return os::remove(f);
#else
    std::string be = backend();
    if (be == "systemd") {
        auto files = definition_files(t); if (!os::stat(files[0]).exists && !os::stat(files[1]).exists) return {};
        service::ctl_({"systemctl", "--user", "disable", "--now", t.name + ".timer"});
        for (auto& f : files) if (os::stat(f).exists) os::remove(f);
        service::ctl_({"systemctl", "--user", "daemon-reload"}); return {};
    }
    if (be != "cron") return {{}, unsupported("schedule (no systemd user session and no crontab)")};
    auto cur = detail_schedule::crontab_read(); if (!cur) return {{}, cur.err};
    bool found = false; std::string rest = detail_schedule::cron_without(*cur, t.name, &found);
    if (!found) return {};
    return detail_schedule::crontab_write(rest);
#endif
#endif
}
} // namespace forge::os::schedule
