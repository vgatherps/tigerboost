#!/bin/sh
# Prints the LLVM 21 prefix for llvm-sys, or nothing (llvm-sys then searches PATH).
# Keeps LLVM_SYS_211_PREFIX if set. Darwin: Homebrew llvm@21, asking the brew in Homebrew's
# default prefix for this CPU first (an Intel Homebrew earlier on PATH would hand an arm64
# build x86_64 LLVM). Linux: llvm-config-21, else llvm-config reporting 21.x.
if [ -n "${LLVM_SYS_211_PREFIX:-}" ]; then
    echo "$LLVM_SYS_211_PREFIX"
    exit 0
fi
case "$(uname -s)" in
Darwin)
    case "$(uname -m)" in
    arm64) brew=/opt/homebrew/bin/brew ;;
    *) brew=/usr/local/bin/brew ;;
    esac
    if [ ! -x "$brew" ]; then
        brew="$(command -v brew)"
    fi
    if [ -n "$brew" ]; then
        prefix="$("$brew" --prefix llvm@21 2>/dev/null)"
        if [ -n "$prefix" ] && [ -d "$prefix" ]; then
            echo "$prefix"
        fi
    fi
    ;;
Linux)
    if command -v llvm-config-21 >/dev/null 2>&1; then
        llvm-config-21 --prefix
    elif command -v llvm-config >/dev/null 2>&1; then
        case "$(llvm-config --version)" in
        21*) llvm-config --prefix ;;
        esac
    fi
    ;;
esac
