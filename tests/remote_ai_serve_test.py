#!/usr/bin/env python3
"""Remote AI client loopback: start forge-serve on the tiny golden model, then run test_remote_ai against it
(FORGE_TEST_SERVE_URL), so inference/remote.hpp is checked against FORGE's own OpenAI-compatible server."""
import os, subprocess, sys
serve, test_bin, fpk = sys.argv[1], sys.argv[2], sys.argv[3]
p = subprocess.Popen([serve, fpk, "--port", "0", "--max-active", "2"], stdout=subprocess.PIPE, text=True)
try:
    line = p.stdout.readline()
    port = int(line.rsplit(":", 1)[1].split()[0])
    env = dict(os.environ, FORGE_TEST_SERVE_URL=f"http://127.0.0.1:{port}/v1", FORGE_TEST_ONLY="serve_loopback")
    r = subprocess.run([test_bin], env=env, capture_output=True, text=True, timeout=300)
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr)
    if "skipped" in r.stdout or "[ ok ] serve_loopback" not in r.stdout:
        sys.exit(r.returncode or 1)
    sys.exit(r.returncode)
finally:
    p.terminate()
    try: p.wait(timeout=10)
    except subprocess.TimeoutExpired: p.kill()
