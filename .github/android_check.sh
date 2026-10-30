#!/bin/sh
# Runs the --emit android goldens on the emulator's /system/bin/sh and the native test_script binary.
set -eu
d=/data/local/tmp/forge; adb shell "rm -rf $d && mkdir -p $d"; fail=0
adb push emit $d/ >/dev/null
for g in functions flat batch; do
  got=$(adb shell "cd $d/emit && /system/bin/sh ./$g.sh; echo RC=\$?" | tr -d '\r')
  rc=$(echo "$got" | sed -n 's/^RC=//p'); got=$(echo "$got" | sed '/^RC=/d')
  if [ "$got" = "$(cat emit/$g.want)" ] && [ "$rc" = "$(cat emit/$g.rc)" ]; then echo "PASS emit android $g"; else echo "FAIL emit android $g (rc $rc)"; echo "$got"; fail=1; fi
done
libcxx=$(find -L "$ANDROID_NDK_LATEST_HOME/" -path "*x86_64-linux-android/libc++_shared.so" | head -1)
adb push "$libcxx" build/android-x86_64/test_script $d/ >/dev/null
adb push tests $d/ >/dev/null
adb shell "cd $d && LD_LIBRARY_PATH=$d TMPDIR=$d ./test_script; echo RC=\$?" | tr -d '\r' | tee out.txt
grep -q '^RC=0$' out.txt || fail=1
exit $fail
