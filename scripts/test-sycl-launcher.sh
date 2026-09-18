#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
launcher="$project_dir/snap/local/singstone-launcher"
test_root=$(mktemp -d)
trap 'rm -rf "$test_root"' EXIT HUP INT TERM
mkdir -p "$test_root/bin"

cat >"$test_root/bin/singstone" <<'EOF'
#!/bin/sh
printf 'cpu:%s:%s\n' "${SINGSTONE_WHISPER_BACKEND:-}" "$*"
EOF
cat >"$test_root/bin/singstone-sycl" <<'EOF'
#!/bin/sh
printf 'sycl:%s:%s:%s:%s\n' \
  "${SINGSTONE_WHISPER_BACKEND:-}" "${SINGSTONE_WHISPER_DEVICE:-}" \
  "${ONEAPI_DEVICE_SELECTOR:-}" "$*"
EOF
cat >"$test_root/bin/probe-ok" <<'EOF'
#!/bin/sh
printf '%s\n' 'Intel(R) Iris(R) Xe Graphics'
exit 0
EOF
cat >"$test_root/bin/probe-fail" <<'EOF'
#!/bin/sh
exit 1
EOF
cat >"$test_root/bin/probe-crash" <<'EOF'
#!/bin/sh
kill -SEGV $$
EOF
chmod +x "$test_root/bin/"*

assert_output() {
  expected=$1
  shift
  actual=$($launcher "$@")
  if [ "$actual" != "$expected" ]; then
    printf 'expected: %s\nactual:   %s\n' "$expected" "$actual" >&2
    exit 1
  fi
}

SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-ok" \
  SNAP_USER_COMMON="$test_root/state-ok" \
  assert_output 'sycl:intel-sycl:Intel(R) Iris(R) Xe Graphics:level_zero:gpu:one two words' one 'two words'
SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-fail" \
  SNAP_USER_COMMON="$test_root/state-fail" \
  assert_output 'cpu:cpu:one two words' one 'two words'
SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-ok" \
  SNAP_USER_COMMON="$test_root/state-disabled" \
  SINGSTONE_DISABLE_GPU=1 \
  assert_output 'cpu:cpu:one two words' one 'two words'
SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-ok" \
  SNAP_USER_COMMON="$test_root/state-selector" \
  ONEAPI_DEVICE_SELECTOR='level_zero:0' \
  assert_output 'sycl:intel-sycl:Intel(R) Iris(R) Xe Graphics:level_zero:0:one two words' one 'two words'

crash_stderr="$test_root/crash.stderr"
actual=$(SNAP="$test_root" SNAP_USER_COMMON="$test_root/state-crash" \
  SINGSTONE_SYCL_PROBE="$test_root/bin/probe-crash" \
  "$launcher" 2>"$crash_stderr")
[ "$actual" = 'cpu:cpu:' ] || {
  printf 'expected crash fallback to CPU, got: %s\n' "$actual" >&2
  exit 1
}
[ ! -s "$crash_stderr" ] || {
  printf 'probe crash escaped to launcher stderr:\n' >&2
  cat "$crash_stderr" >&2
  exit 1
}
grep -q '^status=139 backend=level_zero:gpu$' \
  "$test_root/state-crash/gpu-probe.log"
