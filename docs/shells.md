# Shell emit targets

Every shell is a backend: one file in `script/src/shells/` that defines `forge_shell_register_<file>(shell_registry&)`.
CMake globs that directory and generates the registry, so **adding a shell takes one file**. For each backend and each alias,
`forge::<name>(template, args...)` substitutes `{}` with the shell's own quoting and runs it, using the first executable it finds.
`forge::shells()` lists the backends with `{name, family, available, exe, tested, aliases}`.

| Backend | Aliases | Invocation | Golden-tested |
|---|---|---|---|
| sh, bash, zsh, dash | | `-c` | yes |
| ash | busybox | `busybox sh -c` | yes |
| yash | | `-c` | yes |
| ksh | ksh93 | `-c` | **untested** (not installed on the box) |
| mksh, oksh (pdksh), posh, hush | | `-c` | **untested** |
| osh / ysh | oils | `-c` | **untested** |
| fish | | `-c` | yes |
| tcsh | | `-f -c`, `!` escaped | yes |
| csh | | falls back to tcsh | yes, via tcsh; real BSD csh **untested** |
| nu | nushell | `--no-config-file -c`, raw strings `r#'…'#` | yes |
| xonsh | | `--no-rc -c`, Python literals | yes |
| elvish | | `-norc -c` | yes |
| pwsh (PowerShell 7) | ps | `-NoProfile -Command` | yes |
| powershell (Windows PowerShell 5.1) | powershell5, winps | prefers `powershell.exe` | **untested** on 5.1 (on Linux it falls back to pwsh) |
| cmd | | `cmd /c` | **untested** (Windows/wine only) |
| rc, es | | `-c` | **untested** |

Golden test: `shell_backends_golden` in `tests/test_script.cpp`. It round-trips a hostile literal through each available
shell and skips any that are missing. Set `FORGE_TEST_SHELL_PATH=/workspace/opt/shells/bin` to pick up the user-local shells.
