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
  assert_output 'sycl:intel-sycl:Intel(R) Iris(R) Xe Graphics:level_zero:gpu:one two words' one 'two words'
SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-fail" \
  assert_output 'cpu:cpu:one two words' one 'two words'
SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-ok" \
  SINGSTONE_DISABLE_GPU=1 \
  assert_output 'cpu:cpu:one two words' one 'two words'
SNAP="$test_root" SINGSTONE_SYCL_PROBE="$test_root/bin/probe-ok" \
  ONEAPI_DEVICE_SELECTOR='level_zero:0' \
  assert_output 'sycl:intel-sycl:Intel(R) Iris(R) Xe Graphics:level_zero:0:one two words' one 'two words'
