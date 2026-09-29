status: pass

Implemented the requested slice in four tracked files. No commits, delegation, cargo fmt, rustfmt, or saturating arithmetic. All required gates passed.

| Item | Result |
|---|---|
| A | Private `Target` carries `cpu`, `packing`, `shared_lane`, `move_chunk_origin`, and `clear_upper_halves`. `Target::resolve` is the only triple-string classifier. `prepare_module` resolves once and passes that value through emission; `ordered_predicates` resolves once for inspection. Machine/function CPU attributes, lane ordering, `raw_of`, and leaf permutation use the descriptor. Tests emit both alias pairs with both option values; `x86_64h` selects AVX2 when requested. |
| B1 | `constants_block_layout_matches_plan` emits explicit `aarch64-apple-darwin` and `x86_64-unknown-linux-gnu` modules. Exact depth-three leaf orders are respectively `24,25,20,21,26,27,22,23` and `27,26,23,22,25,24,21,20`. Both retain exact thresholds, padding, remaining leaves, linkage, and alignment assertions. |
| B2 | AVX2 leaf slots subtract `chunk_origin`. A production-emitter test deliberately supplies moving origins with Movemask packing, checks the second chunk's relative leaf offset, and verifies remaining-byte bounds across three chunks. |
| B3 | Every continuation takes row, chunk base, then sums. Transfers advance by `origins[next] - origins[this]`; x86 origins are zero. Removed the pointer-count split and duplicate parameter attributes. The remaining base has alignment 8 and exact `dereferenceable` bytes. Existing ABI, threshold-address, and transfer assertions now require two pointers on both targets. |
| B4 | `vsli` declaration and merge selection both use `target.packing`. Removed the redundant `pair == 0` Insert arm. Exhaustive depth-1–12 permutation tests still cover all three packings and swapped pairs. |
| B5 | Explicit final `vzeroupper` requires a wide comparison in an earlier chunk and no wide comparison in the final chunk. Existing wide-to-shallow tests survive unchanged. New absence tests cover depth 0/1/2 forests with 2, 9, and 17 trees, for Linux and Darwin x86 emission. |
| B6 | Removed `entry_count.max(1)` and the identical `operand` tuple element. Empty forests return before constants allocation; every validated nonempty tree has at least one leaf. |
| B7 | Added a 10-tree, two-chunk Rosetta fixture with separate leaf bit fields, nonzero constant leaf, repeated features, wide/narrow compares, and two explicitly asserted swapped pairs. Both SSE2 and AVX2 execute all 256 feature-bit patterns. Six empty-forest biases run on both paths: +0, -0, 7, smallest positive subnormal, payload quiet NaN, negative signaling NaN. Native checking now compares exact bits, matching the C driver. |
| B8 | Added “Add-based mask merge” to RESULTS.md with all requested historical SSE2 counts and final static counts. |
| B9 | Covered by A. No `TreePath` or `arm64` decision variable remains. |

Assembly was captured before edits with the release `inspect_code` example for 100x100 and 300x100. A second capture after only the Target refactor and merge cleanup matched all six baseline `.s` files byte-for-byte. Final x86 SSE2 and AVX2 `.s` files also match baseline byte-for-byte for both workloads. The remaining ARM64 emission changes come from the single-base ABI; the AVX2 address/clear changes do not change its emission.

Counts exclude directives and alignment padding. Stack operations include push/pop. These are separate target emissions, not MCJIT disassembly.

| Target | Workload | Continuation instructions before → after | Wrapper instructions before → after | Continuation stack operations before → after | Wrapper stack operations before → after |
|---|---|---:|---:|---:|---:|
| ARM64 | 100x100 | 1974 → 1974 | 26 → 25 | 0 → 0 | 20 → 20 |
| ARM64 | 300x100 | 5876 → 5877 | 26 → 25 | 0 → 0 | 20 → 20 |
| SSE2 | 100x100 | 2134 → 2134 | 15 → 15 | 2 → 2 | 10 → 10 |
| SSE2 | 300x100 | 6454 → 6454 | 15 → 15 | 10 → 10 | 10 → 10 |
| AVX2 | 100x100 | 1334 → 1334 | 15 → 15 | 0 → 0 | 10 → 10 |
| AVX2 | 300x100 | 3978 → 3978 | 15 → 15 | 0 → 0 | 10 → 10 |

ARM64's 300-tree continuations gain one instruction; the wrapper loses one, so total instructions remain 5902. At 100 trees total instructions fall 2000 → 1999. The wrapper's `mov x22, x21` disappears. Constants accesses use x21, and transfers update that base by the chunk delta. Continuations no longer use x22: distinct general registers fall 9 → 8 at 100 trees and 10 → 9 at 300 trees. Distinct vector-register counts remain 28 and 29 respectively. Register allocation and instruction scheduling also change; wrapper saves/restores remain unchanged. Both SSE2 kernels retain zero constant-pool references.

A standalone public-API probe captured AVX2 for two trees at each shallow depth before and after. Each diff deletes exactly the explicit clear and its spill/reload; wrappers remain identical.

| Depth of both trees | Continuation instructions before → after | Continuation stack operations before → after | `vzeroupper` before → after |
|---|---:|---:|---:|
| 0 | 8 → 5 | 2 → 0 | 1 → 0 |
| 1 | 13 → 10 | 2 → 0 | 1 → 0 |
| 2 | 13 → 10 | 2 → 0 | 1 → 0 |

The gate environment used `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'`.

| Gate | Result |
|---|---|
| `cargo build --offline --release --examples` | Passed before and after edits. |
| `cargo test --offline --all-targets` | 72 passed, 0 failed. |
| `cargo test --offline --release --all-targets` | 72 passed, 0 failed. |
| `cargo test --offline --doc` | 1 passed, 0 failed. |
| `cargo clippy --offline --all-targets -- -D warnings` | Passed. |
| Final `inspect_code` captures and comparisons | All four x86 comparisons passed; ARM64 and shallow AVX2 differences reported above. |
| `git diff --check` | Passed. |

Each all-targets profile passed 31 library tests, 1 differential test, 11 JIT integration tests, 11 model-reference tests, 6 cold_bench tests, 3 scalar_bench tests, and 9 x86_check tests. inspect_code has no tests. Five new tests were added. Intentional mismatch and invalid-assembly diagnostics remain in the existing negative gate test; both rejection assertions passed.

Every command below used `cargo run --offline --release --example x86_check --` followed by these arguments. Each reported `ok 256 rows`.

| Arguments | PASS line |
|---|---|
| `special /tmp/tb-tg-x1` | `PASS kernel` |
| `special /tmp/tb-tg-x2 --avx2` | `PASS kernel-avx2` |
| `100x100 /tmp/tb-tg-x3 --avx2` | `PASS kernel-avx2` |
| `300x100 /tmp/tb-tg-x4` | `PASS kernel` |
| `depths2 /tmp/tb-tg-x5` | `PASS kernel` |

Ran the requested `grep -rn 'arm64\|aarch64\|x86_64\|TreePath' src/` with the `fn resolve` line filter. It reports 46 lines: two architecture literals inside `Target::resolve`, two host CPU-feature helper references, one LLVM intrinsic name, and 41 unit-test fixture/name/assertion lines. None are another triple classifier. A second search for triple conditionals, `architecture`, `if.*arm64`, `TreePath`, `pointer_count`, and `saturating` reports only:

```text
src/jit.rs:65:        let architecture = triple.split('-').next().unwrap();
src/jit.rs:66:        let (packing, shared_lane, move_chunk_origin) = match architecture {
```

| File | Lines before | Lines after |
|---|---:|---:|
| src/jit.rs | 2104 | 2210 |
| tests/jit.rs | 436 | 460 |
| examples/x86_check.rs | 450 | 536 |
| plans/perf-study-2026-09-28/RESULTS.md | 302 | 327 |
| Total | 3292 | 3533 |

Within src/jit.rs, production lines fall 862 → 849; its unit-test section grows 1242 → 1361. VERDICT.md is the requested untracked report and is excluded from this table.

Self-review found no unresolved failure or scope expansion. The diff retains the existing comparison, load-alignment, tail-call, leaf-permutation, and prediction assertions, with ABI assertions updated to the requested signature. The moving-origin Movemask test exercises the real emitter without changing resolver behavior. No public API was added. Native and Rosetta checks exercise production compilation/emission; the final x86 comparisons cover the entire assembly files, including constants. No timing claims are made.

Evidence is in `/tmp/tb-target-review/`: `before-{100,300}`, `target-{100,300}`, `after-{100,300}`, per-kernel `.diff` files, `narrow.rs`, `narrow-{before,after}-asm`, `count_assembly.py`, `assembly-counts.{txt,json}`, `target-grep.txt`, and build/test/clippy/x86 logs. The five requested Rosetta output directories remain at `/tmp/tb-tg-x1` through `/tmp/tb-tg-x5`.
