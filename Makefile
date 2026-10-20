# Requires GNU make (macOS ships 3.81; Linux distributions ship 4.x).
CARGO_FLAGS ?=
OUT ?=

# llvm-sys treats a set-but-empty prefix as a path, so export only a found one.
LLVM_SYS_211_PREFIX ?= $(shell sh tools/llvm-prefix.sh)
ifneq ($(LLVM_SYS_211_PREFIX),)
export LLVM_SYS_211_PREFIX
else
unexport LLVM_SYS_211_PREFIX
endif

.PHONY: all build test check-x86 bench inspect

all: build test check-x86

build:
	cargo build $(CARGO_FLAGS) --release --examples

test:
	cargo test $(CARGO_FLAGS)
	cargo test $(CARGO_FLAGS) --all-targets
	cargo test $(CARGO_FLAGS) --release --all-targets
	cargo clippy $(CARGO_FLAGS) --all-targets -- -D warnings

check-x86:
	cargo build $(CARGO_FLAGS) --release --example x86_check
	for check in special 'special --avx2' 100x100 '100x100 --avx2' 'depths --avx2' depths2; do \
		set -- $$check; \
		dir="$$(mktemp -d "$${TMPDIR:-/tmp}/tigerboost-x86.XXXXXX")" || exit 1; \
		echo "x86_check $$check"; \
		cargo run $(CARGO_FLAGS) --release --quiet --example x86_check -- "$$1" "$$dir/out" $$2 || { echo "x86_check $$check failed; outputs in $$dir"; exit 1; }; \
		rm -rf "$$dir"; \
	done

bench:
	cargo run $(CARGO_FLAGS) --release --example scalar_bench -- 100x100
	cargo run $(CARGO_FLAGS) --release --example scalar_bench -- 300x100
	cargo run $(CARGO_FLAGS) --release --example cold_bench -- 100x100 --samples 5000

inspect:
	dir="$(OUT)"; \
	if [ -z "$$dir" ]; then \
		dir="$$(mktemp -d "$${TMPDIR:-/tmp}/tigerboost-inspect.XXXXXX")/code" || exit 1; \
	fi; \
	cargo run $(CARGO_FLAGS) --release --example inspect_code -- "$$dir" 100x100 && \
	echo "inspection output: $$dir"
