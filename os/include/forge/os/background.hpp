#pragma once
// Background execution: keep long work (a model download, a long generation) alive when the app is backgrounded
// or the machine wants to sleep.
//   iOS: UIApplication beginBackgroundTaskWithName:expirationHandler: / endBackgroundTask: (objc runtime).
//   Android: there is no NDK API — the app's JNI glue installs hooks (startForegroundService with a notification,
//   or WorkManager) via background::set_hooks(); without hooks begin() returns unsupported.
//   Desktop: prevents idle sleep while the task runs (Windows SetThreadExecutionState, macOS IOPMAssertion,
//   Linux: nothing to do — returns kind "not_required").
#include <forge/os/detail.hpp>

namespace forge::os {
class background {
public:
    struct hooks {   // Android JNI glue
        std::function<bool(const std::string& name, const std::string& title, const std::string& text)> begin;
        std::function<void(const std::string& name)> end;
    };
    static void set_hooks(hooks h) { std::lock_guard<std::mutex> g(m_()); hooks_() = std::move(h); }
    // Returns a token (>0) to pass to end(); kind() reports what the OS actually did.
    static result<std::uint64_t> begin(const std::string& name, const std::string& title = "FORGE", const std::string& text = "Working…") {
        std::lock_guard<std::mutex> g(m_()); std::uint64_t id = ++next_();
        task t; t.name = name;
#if defined(FORGE_OS_IOS)
        auto& o = detail::objc_api::get(); void* app = o.ok() ? o.send(o.cls("UIApplication"), "sharedApplication") : nullptr;
        if (!app) return {0, unsupported("UIApplication (not running inside an app)")};
        auto& cf = detail::cf_api::get(); auto nm = cf.str(name.c_str());
        t.native = o.send<std::uintptr_t, const void*, void*>(app, "beginBackgroundTaskWithName:expirationHandler:", nm, nullptr); cf.CFRelease(nm);
        if (t.native == 0 /*UIBackgroundTaskInvalid*/) return {0, {1, "beginBackgroundTask refused"}};
        t.kind = "ios-background-task";
#elif defined(__ANDROID__)
        if (!hooks_().begin) return {0, unsupported("android foreground service (install background::set_hooks from JNI)")};
        if (!hooks_().begin(name, title, text)) return {0, {1, "foreground service start refused"}};
        t.kind = "android-foreground-service";
#elif defined(FORGE_OS_WINDOWS)
        (void)title; (void)text;
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED); t.kind = "execution-state";
#elif defined(FORGE_OS_MACOS)
        (void)title; (void)text;
        if (void* io = ::dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY)) {
            auto create = reinterpret_cast<int (*)(const void*, std::uint32_t, const void*, std::uint32_t*)>(::dlsym(io, "IOPMAssertionCreateWithName"));
            auto& cf = detail::cf_api::get();
            if (create && cf.ok) { auto ty = cf.str("PreventUserIdleSystemSleep"), nm = cf.str(name.c_str()); std::uint32_t aid = 0;
                if (create(ty, 255, nm, &aid) == 0) { t.native = aid; t.kind = "iopm-assertion"; } cf.CFRelease(ty); cf.CFRelease(nm); }
        }
        if (t.kind.empty()) t.kind = "not_required";
#else
        (void)title; (void)text; t.kind = "not_required";
#endif
        tasks_()[id] = t; return {id, {}};
    }
    static status end(std::uint64_t token) {
        std::lock_guard<std::mutex> g(m_()); auto it = tasks_().find(token); if (it == tasks_().end()) return {{}, {22, "background: unknown token"}};
        auto t = it->second; tasks_().erase(it);
#if defined(FORGE_OS_IOS)
        auto& o = detail::objc_api::get(); if (void* app = o.send(o.cls("UIApplication"), "sharedApplication")) o.send<void, std::uintptr_t>(app, "endBackgroundTask:", t.native);
#elif defined(__ANDROID__)
        if (hooks_().end) hooks_().end(t.name);
#elif defined(FORGE_OS_WINDOWS)
        if (tasks_().empty()) SetThreadExecutionState(ES_CONTINUOUS);
#elif defined(FORGE_OS_MACOS)
        if (t.kind == "iopm-assertion") if (void* io = ::dlopen("/System/Library/Frameworks/IOKit.framework/IOKit", RTLD_LAZY))
            if (auto rel = reinterpret_cast<int (*)(std::uint32_t)>(::dlsym(io, "IOPMAssertionRelease"))) rel(std::uint32_t(t.native));
#endif
        return {};
    }
    static std::string kind(std::uint64_t token) { std::lock_guard<std::mutex> g(m_()); auto it = tasks_().find(token); return it == tasks_().end() ? "" : it->second.kind; }
    static std::size_t active() { std::lock_guard<std::mutex> g(m_()); return tasks_().size(); }
private:
    struct task { std::string name, kind; std::uintptr_t native = 0; };
    static std::mutex& m_() { static std::mutex m; return m; }
    static hooks& hooks_() { static hooks h; return h; }
    static std::uint64_t& next_() { static std::uint64_t n = 0; return n; }
    static std::map<std::uint64_t, task>& tasks_() { static std::map<std::uint64_t, task> t; return t; }
};
// RAII helper.
struct background_scope {
    std::uint64_t token = 0; explicit background_scope(const std::string& n) { auto r = background::begin(n); if (r) token = r.value; }
    ~background_scope() { if (token) background::end(token); }
};
} // namespace forge::os
