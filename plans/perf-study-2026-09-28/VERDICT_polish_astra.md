status: pass

All eight requested items are implemented. All gates pass. Work started at `7b79d32`;
both baseline captures completed before source edits. No benchmark timing runs, commits,
formatters, delegation, stash, checkout, or saturating arithmetic were used.

| Item | Result |
|---|---|
| 1 | The macOS Rosetta test now executes `100x100` with AVX2 before its existing SSE2 success, mismatch, and invalid-assembly checks. It runs in both Cargo test profiles. README includes the corresponding command. Inspection confirms 54 swapped x86 pairs. |
| 2 | Generic two-lane emission adds shifted masks. `MaskPacking` gives `raw_of` distinct `Insert`, `Add`, and `Movemask` modes. The permutation's bounds, uniqueness, and complete-coverage assertions remain. Tests cover all three modes with unswapped, fully swapped, and alternating swapped pairs through depth 12; the depth-six emission test requires vector adds and rejects mask `and/or` and constant-pool references. |
| 3 | Both threshold paths use an f64 GEP with one index, `slot - chunk_origin`, and `chunk_base.unwrap_or(constants)`. The old branch, second GEP form, and unreachable threshold-origin error are gone. The pointer test still checks exact parameter bounds and offsets. Captures after this change match the isolated add-packing capture for all three targets at both sizes. |
| 4 | PLAN and README describe thresholds in target lane order, per-chunk padding and leaves, the ARM64 chunk base serving both, and AVX2 leaf permutation. README identifies the historical study revision and the layout revision. RESULTS adds the layout counts and the maintainer-supplied historical timing comparison. Both final Open lists remove the completed layout study. Historical experiment sections remain. |
| 5 | Removed only the obsolete large-offset premise from `dropped_model_and_last_tree_leaf`; lifetime and prediction checks remain. Renamed the empty/constant integration test to describe its reference comparison. `inspect_code` writes one `plan.txt`, with ARM64 and x86 thresholds from the emitter's lane-order logic through `ordered_predicates`. Both benchmark headers use the sum count from the actual compiled plan. |
| 6 | The AVX2 test requires exactly one leaf `vaddsd` with a memory operand; compare counts, movemask counts, and forbidden-opcode assertions remain. `constant_bits` reads LLVM aggregate elements without modifying the module. The two extraction tricks and the example's `.quad` parser are removed. The 300-tree layout test moved into the library with its original workload and assertions; a test-only module reuses the example fixture. The partition test now uses `plan_with`, exact `0..2` / `2..4` chunks, and reference replay. |
| 7 | Unswapped AVX2 leaves return the identity block before `raw_of`. Other permutations compute and retain each raw slot once. Swap masks use `usize`; `Trees::new` restricts depth below `usize::BITS`, so every valid model's complete-pair mask fits, including models later rejected by LLVM's array-length limit. Existing identity-order checks remain. |
| 8 | An empty forest emits only the C wrapper returning the bias, without constants or continuations. Tests inspect the single return for ARM64, SSE2, and AVX2 emission, and execute native predictions preserving positive zero, negative zero, a quiet NaN payload, and a negative signaling NaN payload. A declared two-feature model still rejects a one-feature row. The existing zero-feature empty-forest check remains. |

Baseline assembly is in `/tmp/tb-pol-before{,300}`. The isolated item-2 capture is in
`/tmp/tb-pol-add{,300}`. Final assembly is in `/tmp/tb-pol-after{,300}`.
The intermediate capture contains only the mask-packing change and its test additions.
Every final file matches that intermediate capture byte for byte. ARM64 and AVX2 also
match the original baseline, including constants. SSE2 therefore changes only for item 2.

| Trees | File | Final bytes | Byte comparison |
|---:|---|---:|---|
| 100 | `aarch64-apple-darwin-kernel.s` | 226111 | PASS, original baseline |
| 100 | `x86_64-unknown-linux-gnu-kernel-avx2.s` | 224568 | PASS, original baseline |
| 100 | `x86_64-unknown-linux-gnu-kernel.s` | 234936 | PASS, isolated item 2 |
| 300 | `aarch64-apple-darwin-kernel.s` | 674628 | PASS, original baseline |
| 300 | `x86_64-unknown-linux-gnu-kernel-avx2.s` | 671922 | PASS, original baseline |
| 300 | `x86_64-unknown-linux-gnu-kernel.s` | 704617 | PASS, isolated item 2 |

SSE2 counts cover continuation functions, excluding the C wrapper, directives, and alignment
padding. Constant-pool loads count the `.LCPI` address/load sequences, once per loaded constant.
Stack operations include explicit stack memory operands and implicit `pushq` / `popq` accesses.
Including push/pop explains the difference from the older layout report's 2 / 30 stack counts.

| Trees | Instructions before → after | Constant-pool loads before → after | Stack operations before → after |
|---:|---:|---:|---:|
| 100 | 2267 → 2134 | 13 → 0 | 6 → 2 |
| 300 | 6859 → 6454 | 38 → 0 | 42 → 10 |

Byte sizes, SHA-256 hashes, and counts are in [the comparison record](/tmp/tb-pol-comparison.json).
The final inspection check compares all 600 / 1800 printed thresholds with the emitted constants
for ARM64, SSE2, and AVX2. It also confirms one plan file per directory and 66 / 161 ARM64
swaps and 54 / 172 x86 swaps. See [inspection checks](/tmp/tb-pol-inspection-check.log).

All Cargo commands used `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21` and
`RUSTDOCFLAGS='-C link-arg=/opt/homebrew/lib/libzstd.dylib'`.

| Gate | Result |
|---|---|
| Before `inspect_code -- /tmp/tb-pol-before 100x100` | PASS, before edits |
| Before `inspect_code -- /tmp/tb-pol-before300 300x100` | PASS, before edits |
| `cargo build --offline --release --examples` | PASS, [log](/tmp/tb-pol-build.log) |
| `cargo test --offline --all-targets` | PASS, 67 tests, [log](/tmp/tb-pol-test-debug.log) |
| `cargo test --offline --release --all-targets` | PASS, 67 tests, [log](/tmp/tb-pol-test-release.log) |
| `cargo test --offline --doc` | PASS, 1 doctest, [log](/tmp/tb-pol-test-doc.log) |
| `cargo clippy --offline --all-targets -- -D warnings` | PASS, [log](/tmp/tb-pol-clippy.log) |
| After `inspect_code -- /tmp/tb-pol-after 100x100` | PASS, [log](/tmp/tb-pol-inspect-100.log) |
| After `inspect_code -- /tmp/tb-pol-after300 300x100` | PASS, [log](/tmp/tb-pol-inspect-300.log) |
| Byte comparisons and SSE2 counts | PASS, both workloads |
| `git diff --check` | PASS |

| Test binary | Passed in each profile |
|---|---:|
| Library | 29 |
| `differential` | 1 |
| `jit` | 10 |
| `model_reference` | 11 |
| `cold_bench` | 6 |
| `inspect_code` | 0 |
| `scalar_bench` | 3 |
| `x86_check` | 7 |
| Total | 67 |

Each command below used `cargo run --offline --release --example x86_check --` and printed
`ok 256 rows` followed by the recorded PASS line.

| Arguments | PASS line | Log |
|---|---|---|
| `special /tmp/tb-pol-x1` | `PASS kernel` | [x1](/tmp/tb-pol-x1.log) |
| `special /tmp/tb-pol-x2 --avx2` | `PASS kernel-avx2` | [x2](/tmp/tb-pol-x2.log) |
| `100x100 /tmp/tb-pol-x3` | `PASS kernel` | [x3](/tmp/tb-pol-x3.log) |
| `100x100 /tmp/tb-pol-x4 --avx2` | `PASS kernel-avx2` | [x4](/tmp/tb-pol-x4.log) |
| `300x100 /tmp/tb-pol-x5` | `PASS kernel` | [x5](/tmp/tb-pol-x5.log) |
| `depths2 /tmp/tb-pol-x6` | `PASS kernel` | [x6](/tmp/tb-pol-x6.log) |

Self-review checked the add simulation against emitted vector operations, odd-depth false
padding, swap widths, identity leaves, chunk-relative addresses, exact bias bits, and row-length
validation. The assembly comparisons isolate the only nonempty-kernel change. Constants tests
still exercise production emission, and the moved layout test retains the same fixture and
checks. `ordered_predicates` has a current caller in `inspect_code`; test fixture exposure is
under `cfg(test)` only. No unrelated public types or runtime conditions were added.
Final review restored the benchmark plans' original lifetime: save the sum count, compile, then
drop the temporary plan before measurement. Build, both test profiles, doctest, and Clippy passed
again after that correction. No emitter code changed after the final assembly captures.

Tracked changes stay within the allowed source, example, test, and documentation paths.
The CSV writer and all reviewer prototypes remain untouched. The x86 linker emits its existing
`-no_pie` and unaligned-pointer warnings; every linked gate succeeds. Expected mismatch and
invalid-assembly diagnostics in the Cargo logs belong to the retained negative tests. No gate
failure or unresolved blocker remains.
