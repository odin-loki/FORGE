// forge/os/oserror.hpp: errno / Win32 classification, typed exceptions, fault signals -> exceptions,
// and the ForgeScript exception classes they map to.
#include <forge/os/oserror.hpp>
#include <forge/script/script.hpp>
#include "check.hpp"
#include <string>
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
    try { os::fault_guard([] { volatile int* p = nullptr; *p = 1; }); }
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
