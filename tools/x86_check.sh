#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: tools/x86_check.sh <workload> [--avx2] [<out-dir>]" >&2
    exit 2
fi

workload=$1
shift
avx2=
if [ "${1:-}" = --avx2 ]; then
    avx2=--avx2
    shift
fi
if [ "$#" -gt 1 ]; then
    echo "usage: tools/x86_check.sh <workload> [--avx2] [<out-dir>]" >&2
    exit 2
fi
output=${1:-target/x86-check/$workload${avx2:+-avx2}}
set --
if [ -n "$avx2" ]; then
    set -- --avx2
fi
case "$output" in
    /|.)
        echo "refusing to remove output directory $output" >&2
        exit 2
        ;;
esac
rm -rf -- "$output"
mkdir -p "$(dirname "$output")"

if ! cargo run --offline --release --example x86_check -- "$workload" "$output" "$@"; then
    echo "cargo run failed; check that LLVM_SYS_211_PREFIX is exported" >&2
    exit 1
fi

status=0
variants=0
while IFS=' ' read -r kind variant extra; do
    if [ "$kind" != variant ]; then
        continue
    fi
    variants=$((variants + 1))
    if [ -n "$variant" ] && [ -z "$extra" ] &&
        clang -arch x86_64 -c "$output/$variant.s" -o "$output/$variant.o" &&
        clang -arch x86_64 -O1 -Wl,-no_pie tools/x86_check.c "$output/$variant.o" \
            -o "$output/$variant.check" &&
        arch -x86_64 "$output/$variant.check" "$output"
    then
        echo "PASS $variant"
    else
        echo "FAIL $variant"
        status=1
    fi
done < "$output/manifest.txt"

if [ "$variants" -eq 0 ]; then
    echo "manifest has no variants" >&2
    exit 2
fi
exit "$status"
