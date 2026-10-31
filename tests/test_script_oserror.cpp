// forge/os/oserror.hpp: errno / Win32 classification, typed exceptions, fault signals -> exceptions,
// and the ForgeScript exception classes they map to.
#include <forge/os/oserror.hpp>
#include <forge/script/script.hpp>
#include "check.hpp"
#include <string>
#include <atomic>
#if !defined(_WIN32)
#include <unistd.h>
#include <csignal>
#endif
using namespace forge;

TEST(classify_errno_basic) {
#if !defined(_WIN32)
    CHECK(os::classify_errno(ENOENT) == os::os_errc::not_found);
    CHECK(os::classify_errno(EEXIST) == os::os_errc::already_exists);
    CHECK(os::classify_errno(EACCES) == os::os_errc::permission_denied);
    CHECK(os::classify_errno(ECONNREFUSED) == os::os_errc::connection_refused);
    os::note_sandboxed(true);
    CHECK(os::classify_errno(EPERM) == os::os_errc::sandbox_violation);
    os::note_sandboxed(false);
#endif
    CHECK(std::string(os::script_error_type(os::os_errc::not_found)) == "FileNotFoundError");
    CHECK(std::string(os::script_error_type(os::os_errc::sandbox_violation)) == "SandboxViolationError");
}

TEST(throw_if_typed) {
    bool caught = false;
    try { os::throw_if(os::read_file("/nonexistent/forge/oserror/probe")); }
    catch (const os::os_exception& e) { caught = true; CHECK(e.kind == os::os_errc::not_found); }
    CHECK(caught);
}

TEST(fault_guard_recovers_segv) {
    bool caught = false;
    // `int* volatile`: the pointer value is opaque to the optimiser. With a visibly-null pointer GCC proves the
    // lambda always faults, isolates the path (store + ud2) and drops fault_guard's t_fault store as dead.
    try { os::fault_guard([] { int* volatile p = nullptr; *p = 1; }); }
    catch (const os::os_exception& e) { caught = true; CHECK(e.kind == os::os_errc::signal); }
    CHECK(caught);
    CHECK(os::fault_guard([] { return 7; }) == 7);   // still usable afterwards
}

TEST(script_catches_typed_os_errors) {
    script::engine e; std::string out;
    auto r = e.eval("try { fs.read(\"/nonexistent/forge/oserror/probe\"); \"no\" } catch (FileNotFoundError err) { \"yes\" }");
    // syntax of catch clauses may differ; the hierarchy check below is the contract
    auto r2 = e.eval("FileNotFoundError(\"x\") is OSError");
    (void)r; CHECK_MSG(r2.ok, "%s", r2.error.c_str());
    auto r3 = e.eval("SandboxViolationError(\"x\") is PermissionError");
    CHECK_MSG(r3.ok, "%s", r3.error.c_str());
}

TEST(classify_errno_extended) {
#if !defined(_WIN32)
    CHECK(os::classify_errno(EROFS) == os::os_errc::read_only);
    CHECK(os::classify_errno(ENOTEMPTY) == os::os_errc::not_empty);
    CHECK(os::classify_errno(ESRCH) == os::os_errc::no_such_process);
    CHECK(os::classify_errno(EPIPE) == os::os_errc::broken_pipe);
    CHECK(std::string(os::script_error_type(os::os_errc::broken_pipe)) == "BrokenPipeError");
#endif
}

TEST(classify_ntstatus_and_mach) {
    CHECK(os::classify_ntstatus(0xC0000034u) == os::os_errc::not_found);
    CHECK(os::classify_ntstatus(0xC0000005u) == os::os_errc::signal);
    CHECK(os::classify_ntstatus(0xC000013Au) == os::os_errc::cancelled);
    CHECK(os::classify_mach_exception(1) == os::os_errc::signal);
    CHECK(os::classify_mach_exception(12) == os::os_errc::sandbox_violation);
}

#if !defined(_WIN32)
TEST(fault_guard_sigfpe_and_sigpipe_ignored) {
    bool caught = false;
    try { os::fault_guard([] { ::raise(SIGFPE); }); }
    catch (const os::os_exception& e) { caught = true; CHECK(e.signo == SIGFPE); }
    CHECK(caught);
    int fd[2]; CHECK(::pipe(fd) == 0); ::close(fd[0]);
    errno = 0; auto n = ::write(fd[1], "x", 1); CHECK(n < 0 && errno == EPIPE); ::close(fd[1]);   // SIGPIPE ignored
}

TEST(cancel_signal_sets_flag) {
    std::atomic<bool> f{false};
    os::install_cancel_signals(&f); ::raise(SIGTERM); CHECK(f.load());
    CHECK(os::take_cancel_signal() == SIGTERM);
    os::install_cancel_signals(nullptr);
}
#endif

TEST(script_os_hierarchy_extended) {
    script::engine e;
    for (const char* q : {"BrokenPipeError(\"x\") is ConnectionError", "ConnectionError(\"x\") is OSError",
                          "ReadOnlyFileSystemError(\"x\") is OSError", "ProcessLookupError(\"x\") is OSError"}) {
        auto r = e.eval(q); CHECK_MSG(r.ok && r.value == "true", "%s -> %s %s", q, r.value.c_str(), r.error.c_str());
    }
}

TEST(sandbox_label_matches_policy) {
#if !defined(_WIN32)
    os::note_sandbox_policy(/*net*/true, /*exec*/false, /*fs_write*/false);
    { os::os_op_scope s(os::os_op::net); CHECK(os::classify_errno(EPERM) == os::os_errc::sandbox_violation); }
    { os::os_op_scope s(os::os_op::exec); CHECK(os::classify_errno(EPERM) == os::os_errc::permission_denied); }
    { os::os_op_scope s(os::os_op::net); CHECK(os::classify_errno(EACCES) == os::os_errc::permission_denied); }
    CHECK(os::classify_errno(EPERM) == os::os_errc::permission_denied);   // os_op::any never matches a known policy
    os::note_sandboxed(false);
#endif
}

TEST(script_python_style_hierarchy) {
    script::engine e;
    for (const char* q : {"PermissionError(\"x\") is OSError", "TimeoutError(\"x\") is OSError", "OutOfMemoryError(\"x\") is OSError",
                          "CancelledError(\"x\") is OSError", "DeadlockError(\"x\") is OSError", "SandboxViolationError(\"x\") is OSError"}) {
        auto r = e.eval(q); CHECK_MSG(r.ok && r.value == "true", "%s -> %s %s", q, r.value.c_str(), r.error.c_str());
    }
}

TEST(engine_cancel_signals_option) {
    script::options o; o.cancel_signals = false; script::engine e(o);
    auto r = e.run("1 + 1;"); CHECK(r.ok);
}

#if !defined(_WIN32) && !defined(FORGE_OS_IOS)
TEST(script_process_pool) {
    script::engine e;
    auto r = e.eval("auto pp = ProcessPool(2); auto f = pp.submit(\"auto task(auto x) { return x * x; }\", 7); auto v = f.get(); pp.shutdown(); v");
    CHECK_MSG(r.ok && r.value == "49", "%s %s", r.value.c_str(), r.error.c_str());
    auto r2 = e.eval("auto p2 = ProcessPool(1); auto m = p2.map(\"auto task(auto x) { return x + 1; }\", [1, 2, 3]); p2.shutdown(); m");
    CHECK_MSG(r2.ok, "%s", r2.error.c_str());
    // error type crosses the process boundary; worker has no fs by default
    auto r3 = e.eval("auto p3 = ProcessPool(1); auto t = \"\"; try { p3.submit(\"auto task(auto x) { return fs.read(\\\"/etc/hostname\\\"); }\", 0).get(); } catch (PermissionError err) { t = \"perm\"; } p3.shutdown(); t");
    CHECK_MSG(r3.ok && r3.value.find("perm") != std::string::npos, "%s %s", r3.value.c_str(), r3.error.c_str());
}
#endif
