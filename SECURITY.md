# Security

Model files are untrusted input. The `.fpk` reader bounds-checks every section and tensor
against the file size before use. Report vulnerabilities privately to the author.
FORGE is offline by default: no network or telemetry code exists in the runtime.
