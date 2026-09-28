status: pass

Performance is mixed. With both flags, the 300-tree hot median rose from 240.14 to 245.38 ns (+2.18%). After the other model, dependent p50 rose from 125 to 208 ns for all three flag combinations. Batched-only hot medians improved by 1.09% at 100 trees and 0.51% at 300 trees. Timings are indicative; all four cold runs used separate processes. Both fields remain false by default.

`reuse_features` replaces 132 of the 600 lane fills on `100x100` with existing normalized SSA values: 22%, leaving 468 row loads in emitted IR. The count sums distinct features within each eight-tree chunk from the production plan dump. The existing compilation pipeline includes `function(early-cse)`. Final assembly is byte-identical when toggling reuse for both batching settings on ARM64, generic x86, and AVX2. The timing differences between those identical assembly outputs do not demonstrate reduced machine loads.

`batched_leaves` emits comparisons and leaf slots in pass one, then leaf loads and sum updates in pass two. Tree order, sum assignments, constant-tree order, final reduction, returns, chunks, and transfers are preserved. It reduces ARM64 assembly by two instructions and generic x86 by four on this workload; AVX2 instruction count is unchanged.

The changed files are listed below. `tools/x86_check.sh` already reads variants from the manifest and needed no edit.

| File | Change |
|---|---|
| `src/jit.rs` | Two default-false fields; chunk-local normalized feature map; optional two-pass leaf emission; IR order, load-count, and NaN tests. |
| `tests/differential.rs` | Cycle all four flag combinations across 96 seeded forests and three chunk sizes; retain reference checks and dropped-model coverage. |
| `tests/scalar_jit.rs` | Explicit all-false options versus `Default` assembly equality; `from_host` leaves both new flags false. |
| `examples/scalar_bench.rs` | Three requested variants; verify their configuration and predictions. |
| `examples/cold_bench.rs` | Independent `--batched` / `--reuse` flags, printed option values, parser tests, and production benchmark smoke tests for every combination. |
| `examples/inspect_code.rs` | Three new variants on both targets and their x86 AVX2 twins. |
| `examples/x86_check.rs` | Six total variants, native validation with each variant's options, emitted x86 checks, and updated harness tests. |
| `VERDICT.md` | Measurements and self-review. |

Validation ran on native ARM64, macOS 15.6, LLVM 21.1.8, rustc 1.93.0-nightly (c90bcb957, 2025-11-06). Every Cargo command was offline. `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and `RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'` were set for the gates.

| Gate | Result |
|---|---|
| Release examples build | Passed. |
| Debug / release all-target tests | 61 passed in each profile. |
| Doctests | 1 passed. |
| Clippy, all targets, `-D warnings` | Passed. |
| Before/after `inspect_code`, `100x100` | All six pre-existing assemblies and six plan dumps passed `cmp`. Baseline was captured before edits from commit `04475f83d997e4167e13ec7c1e893d3e14601823`. |
| Rosetta, `100x100`, generic and `--avx2` | Six variants × 256 rows passed in each mode, bitwise versus reference. |
| Rosetta, `special`, generic and `--avx2` | Six variants × 256 rows passed in each mode; covers NaNs, infinities, signed zero, repeated features, odd depths, and constant trees. |
| Hot benchmarks, `100x100` and `300x100` | Both completed; all kernels verified before timing. |
| Cold benchmark, `100x100 --samples 3000` | All four flag combinations completed. |
| Scope and whitespace | Only allowed source files and this report changed; `git diff --check` passed. No formatter or commits. |

Assembly counts below cover `predict` and every continuation in the separately emitted `100x100` assembly, excluding directives, data, and alignment padding. They are not disassembly of MCJIT memory. Load columns count instructions with memory sources. `ldp` counts once regardless of width and includes wrapper restores. Stack columns count explicit memory-access instructions, including push/pop; implicit x86 call/return-address traffic is excluded. ARM64 has no continuation stack accesses. Generic x86 at eight trees/chunk saves/restores `rbp` twice, for four continuation stack instructions; AVX2 has none. These stack counts are unchanged by either flag.

ARM64 uses `aarch64-apple-darwin`. Each variant also has one pointer `ldr x` in the wrapper.

| Variant | Instructions | `ldr d` | `ldr q` | `ld1.d` | `ld1r.2d` | `ldp` | Stack wrapper | Stack continuations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tree-major-4-2sums | 2253 | 345 | 88 | 235 | 3 | 141 | 20 | 0 |
| tree-major-8-2sums | 2143 | 316 | 26 | 172 | 2 | 186 | 20 | 0 |
| tree-major-8-2sums-batched | 2141 | 316 | 22 | 172 | 2 | 188 | 20 | 0 |
| tree-major-8-2sums-reuse | 2143 | 316 | 26 | 172 | 2 | 186 | 20 | 0 |
| tree-major-8-2sums-batched-reuse | 2141 | 316 | 22 | 172 | 2 | 188 | 20 | 0 |

x86 counts use `x86_64-unknown-linux-gnu`; the Rosetta correctness gate separately emits `x86_64-apple-darwin`. Folded leaf loads are shown as memory-source `addsd` / `vaddsd` counts. AVX2 also folds all 200 threshold loads into `vcmpgt_oqpd`.

| Variant | Instructions | `vmovsd` | `vmovhpd` | `movsd` | `movhpd` | Leaf `addsd` / `vaddsd` | Stack wrapper | Stack continuations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tree-major-4-2sums | 2332 | 0 | 0 | 294 | 235 | 99 | 10 | 0 |
| tree-major-8-2sums | 2300 | 0 | 0 | 292 | 174 | 99 | 10 | 4 |
| tree-major-8-2sums-batched | 2296 | 0 | 0 | 292 | 174 | 99 | 10 | 4 |
| tree-major-8-2sums-reuse | 2300 | 0 | 0 | 292 | 174 | 99 | 10 | 4 |
| tree-major-8-2sums-batched-reuse | 2296 | 0 | 0 | 292 | 174 | 99 | 10 | 4 |
| tree-major-4-2sums-avx2 | 1486 | 284 | 226 | 0 | 0 | 100 | 10 | 0 |
| tree-major-8-2sums-avx2 | 1451 | 274 | 161 | 0 | 0 | 100 | 10 | 0 |
| tree-major-8-2sums-batched-avx2 | 1451 | 274 | 161 | 0 | 0 | 100 | 10 | 0 |
| tree-major-8-2sums-reuse-avx2 | 1451 | 274 | 161 | 0 | 0 | 100 | 10 | 0 |
| tree-major-8-2sums-batched-reuse-avx2 | 1451 | 274 | 161 | 0 | 0 | 100 | 10 | 0 |

Hot medians below are native ARM64 ns/prediction, generic target, five rounds of 50,000 dependent calls. Driver cost is included. The dependent chain visits 35 rows at 100 trees and 12 rows at 300 trees.

| Variant | `100x100` median ns | `300x100` median ns |
|---|---:|---:|
| Reference | 287.04 | 857.37 |
| Tree-major, 4 trees/chunk, sums 1 | 93.88 | 256.18 |
| Tree-major, 4 trees/chunk, sums 2 | 94.44 | 254.67 |
| Tree-major, 4 trees/chunk, sums 4 | 94.58 | 256.00 |
| Tree-major, 8 trees/chunk, sums 1 | 91.72 | 242.73 |
| Tree-major, 8 trees/chunk, sums 2 | 90.67 | 240.14 |
| Tree-major, 8 trees/chunk, sums 4 | 91.23 | 241.53 |
| Tree-major, 16 trees/chunk, sums 1 | 93.64 | 241.32 |
| Tree-major, 16 trees/chunk, sums 2 | 88.22 | 232.55 |
| Tree-major, 16 trees/chunk, sums 4 | 89.52 | 246.07 |
| Tree-major, 8 trees/chunk, sums 2, batched | 89.68 | 238.92 |
| Tree-major, 8 trees/chunk, sums 2, reuse | 91.57 | 242.37 |
| Tree-major, 8 trees/chunk, sums 2, batched+reuse | 88.96 | 245.38 |
| Driver floor | 5.67 | 5.66 |

Cold results below are native ARM64 ns/call, eight trees/chunk, two sums, 3,000 retained samples per condition and row sequence. Timer resolution was about 41 ns. The 64 MiB condition disturbs data before copying the row and timing prediction. The other-model condition runs the unchanged 2,000-tree disturbance model. All cold runs visit 35 distinct rows in the dependent chain.

| Flags | Row sequence | Condition | p50 ns | p90 ns | p99 ns |
|---|---|---|---:|---:|---:|
| none | shuffled | hot | 83 | 84 | 125 |
| none | shuffled | data 64MiB | 791 | 1084 | 2375 |
| none | shuffled | other model | 125 | 167 | 208 |
| none | dependent | hot | 83 | 84 | 84 |
| none | dependent | data 64MiB | 792 | 1500 | 2958 |
| none | dependent | other model | 125 | 167 | 209 |
| `--batched` | shuffled | hot | 83 | 84 | 125 |
| `--batched` | shuffled | data 64MiB | 750 | 917 | 2167 |
| `--batched` | shuffled | other model | 208 | 209 | 291 |
| `--batched` | dependent | hot | 83 | 84 | 125 |
| `--batched` | dependent | data 64MiB | 792 | 1125 | 2416 |
| `--batched` | dependent | other model | 208 | 209 | 250 |
| `--reuse` | shuffled | hot | 83 | 84 | 125 |
| `--reuse` | shuffled | data 64MiB | 875 | 1042 | 2167 |
| `--reuse` | shuffled | other model | 208 | 209 | 250 |
| `--reuse` | dependent | hot | 83 | 84 | 125 |
| `--reuse` | dependent | data 64MiB | 917 | 1792 | 2666 |
| `--reuse` | dependent | other model | 208 | 250 | 333 |
| `--batched --reuse` | shuffled | hot | 83 | 84 | 125 |
| `--batched --reuse` | shuffled | data 64MiB | 833 | 1041 | 2250 |
| `--batched --reuse` | shuffled | other model | 167 | 209 | 250 |
| `--batched --reuse` | dependent | hot | 83 | 84 | 166 |
| `--batched --reuse` | dependent | data 64MiB | 792 | 1042 | 2458 |
| `--batched --reuse` | dependent | other model | 208 | 209 | 292 |

The remaining cold conditions, timer floors, means, and maxima are retained in the full logs.

| Evidence | Location |
|---|---|
| Before assembly and plans | `/tmp/tb-exp-before-26773` |
| After assembly and plans | `/tmp/tb-exp-after-29567` |
| Baseline SHA-256 values | `/tmp/tb-exp-baseline-sha256.txt` |
| Instruction counts and opcode breakdowns | `/tmp/tb-exp-counts.json` |
| Hot runs | `/tmp/tb-exp-hot-100.log`, `/tmp/tb-exp-hot-300.log` |
| Cold runs | `/tmp/tb-exp-cold-off.log`, `/tmp/tb-exp-cold-batched.log`, `/tmp/tb-exp-cold-reuse.log`, `/tmp/tb-exp-cold-batched-reuse.log` |
| Test and Clippy logs | `/tmp/tb-exp-test-debug.log`, `/tmp/tb-exp-test-release.log`, `/tmp/tb-exp-test-doc.log`, `/tmp/tb-exp-clippy.log` |
| Rosetta logs | `/tmp/tb-exp-x86.log`, `/tmp/tb-exp-x86-avx2.log`, `/tmp/tb-exp-x86-special.log`, `/tmp/tb-exp-x86-special-avx2.log` |

Self-review checked both `TreePath` arms, the shared leaf-load emission, constant trees, sum order, chunk transfers, and map lifetime. The map stores the value returned by `load_feature` after NaN normalization and is recreated at every chunk. The four-tree load-count test repeats the same features in a second chunk to catch accidental reuse across boundaries. IR tests inspect production emission before optimization; predictions exercise `compile_with`, and Rosetta executes emitted x86 code. Differential tests retain exact leaf-index-sensitive sums. No planner, JIT ABI, dependency, README, or plan changes were needed. The initial untracked `CODEX_TASK.md` and `codex_run.log` were left alone.
