Minor at `examples/cold_bench.rs:483`. CSV output can fail while the benchmark reports success. `write_csv` returns without flushing its `BufWriter`; an error writing the final buffer is discarded during destruction. Reproduced through the actual example with `--samples 1 --out probes/limited.csv`, `RLIMIT_FSIZE=0`, and `SIGXFSZ` ignored: exit status 0, empty stderr, zero-byte CSV. Larger exports can lose their final buffered portion the same way. Add `writer.flush()?` before returning. Reproduction: [probes/csv_error.py](probes/csv_error.py); captured result: [probes/csv-error.log](probes/csv-error.log).

Minor at `PLAN.md:38`, `PLAN.md:45`, and `PLAN.md:74`. The current-design description still places all thresholds before all leaves, says AVX2 leaves retain original order, and lists the leaf-offset layout study as open. The implementation places thresholds, padding, and leaves together for each chunk (`src/jit.rs:201`), swaps predicates according to the target (`src/jit.rs:750`), and permutes AVX2 leaves too (`src/jit.rs:797`). Following the documented AVX2 layout would select the wrong leaves after a swap. `plans/perf-study-2026-09-28/RESULTS.md:282` repeats the completed open item. `README.md:146` calls the older 85.3/228.2 ns measurements the current kernel's results, while `plans/perf-study-2026-09-28/VERDICT_layout_astra.md:3` explicitly records no timings for the layout implementation. Identify the measured revision and link the layout report; retain historical experiment sections as history. The brief's claim that RESULTS records every measurement and experiment is therefore incomplete.

Minor at `examples/scalar_bench.rs:70` and `examples/cold_bench.rs:416`. Both benchmark headers always report two sums. The accepted workload `1x1` uses one sum because `src/sort_policy.rs:71` clamps the count to the number of trees. The CSV reproduction also captures this inaccurate header. Report the actual plan count, or omit the hardcoded count.

No blocker or major finding. No selected-leaf disagreement was found. I read the production Rust, tests, examples, C driver, current design documents, and recorded experiments, including the layout report. Tracked files remain unchanged.

The following changes are ranked by value for effort. Line counts are estimates; instruction counts refer to emitted code, not measured latency.

| Rank | Location and change | Estimated saving or effort | Emitted code and coverage |
|---:|---|---|---|
| 1 | `examples/cold_bench.rs:483`: flush explicitly and retain a failure regression test. | Add 1 production line and a small subprocess test. | No kernel change. Detects failed final writes. |
| 2 | `tests/jit.rs:23`, `tests/differential.rs:98`, `examples/x86_check.rs:428`: make forced lane swaps and encoded mixed forests permanent SSE2/AVX2 execution cases. The exhaustive swap test enables AVX2 only on a supported native host; the randomized differential test compiles the default path twice; the automated Rosetta execution test runs only the default path. | Roughly 50–100 test/helper lines using the existing driver. | No kernel change. Adds direct AVX2 leaf-selection coverage on this ARM64 development host. The public-API probe below supplies working fixtures. |
| 3 | `PLAN.md:38`, `README.md:82`, `README.md:146`, `plans/perf-study-2026-09-28/RESULTS.md:282`, and the two benchmark headers: describe the chunk base as serving thresholds and leaves, document both target lane orders and AVX2 permutation, close the layout item, and label measurements by revision. | Roughly 15–25 lines edited; 0 kernel instructions. | Documentation and reporting only. |
| 4 | `src/jit.rs:1243`: replace the AVX2 test's hand-written displacement/register grammar and preceding-instruction check with a direct assertion that the single `vaddsd` has a memory operand. Keep compare counts, movemask counts, and forbidden-opcode checks. | Remove about 25–35 test lines. | No kernel change. Preserves the folded-leaf-load assertion; stops testing assembler spelling and unrelated preceding loads. |
| 5 | `src/jit.rs:816`: for `TreePath::Avx2` with `swaps == 0`, return `(0, tree.leafs.clone())`. `raw_of` is the identity in this case, including odd depths. | Add about 3 lines. Avoid 7,424 / 19,712 calls to `raw_of` for the inspected 100 / 300 tree AVX2 emissions, plus their `seen` allocations. | Identical constants and kernel instructions. Compile-time work only; no compile-time speedup measured. Existing identity-layout tests should remain. Counts: [probes/identity-counts.log](probes/identity-counts.log). |
| 6 | `src/sort_policy/tests.rs:272`: replace the manually assembled `Plan` with a `plan_with` fixture and exact chunk-range assertions. Partitioning now counts one action per whole tree; it does not inspect the hand-written predicate ranges. | Remove about 25–30 test lines. | No kernel change. Keeps partition-boundary coverage; arbitrary visit-order coverage also exists at `src/sort_policy/tests.rs:226` and `:318`. |
| 7 | `src/jit.rs:180`: optionally return the bias directly from the C wrapper for an empty forest. The general continuation machinery currently survives even when there is no tree to evaluate. | Add roughly 10–20 lines. For the inspected negative-zero bias, an estimated 29 → 3 ARM64 instructions, eliminating the 160-byte frame and 20 stack accesses. | Changes only empty-forest emission. Preserve row-length checking and exact bias bits, including NaN and negative zero. Low priority because it adds a special case outside the main workload. Current assembly: [probes/empty-arm64.s](probes/empty-arm64.s). |

I found no demonstrated cheap latency improvement for the ordinary depth-six kernel. The inspected assembly confirms paired ARM64 threshold loads, indexed leaf loads, folded AVX2 threshold comparisons, and one final AVX2 `vzeroupper`. The remaining SSE2 stack traffic is real. These counts reproduce the layout report; none establishes a timing improvement. No timing comparison was attempted with the other reviewer running concurrently.

| Target | Trees | Continuation instructions | C wrapper instructions | Continuation stack accesses | `vzeroupper` |
|---|---:|---:|---:|---:|---:|
| ARM64 | 100 | 1,974 | 26 | 0 | 0 |
| ARM64 | 300 | 5,876 | 26 | 0 | 0 |
| SSE2 | 100 | 2,267 | 15 | 2 | 0 |
| SSE2 | 300 | 6,859 | 15 | 30 | 0 |
| AVX2 | 100 | 1,334 | 15 | 0 | 1 |
| AVX2 | 300 | 3,978 | 15 | 0 | 1 |

Assembly and plan dumps are under [probes/asm](probes/asm) and [probes/asm300](probes/asm300). [probes/count_assembly.py](probes/count_assembly.py) counts assembly instructions excluding directives and alignment padding; stack accesses count instructions referencing the stack, not transferred words.

Validation at `166a9d69fe59a5eb46a00653d861eda13ca9badc` used offline Cargo and the supplied LLVM installation.

| Check | Result |
|---|---|
| Debug and release `cargo test --offline --all-targets` | 65 tests passed in each profile. |
| Doctest | 1 passed. |
| Clippy, all targets, warnings denied | Passed. |
| Existing Rosetta examples | Passed: `special` SSE2 and AVX2, `depths` AVX2, `depths2` SSE2, `300x100` AVX2. |
| Added native public-API probes | 96 model cases; 582,280 predictions through `compile` and `compile_plan`, after dropping the source model. A separate zero-feature empty forest preserves negative-zero bias. |
| Added Rosetta public-emission probes | 84 models per path; 155,672 predictions across SSE2 and AVX2, bit-exact against the reference. |

[probes/public_api.rs](probes/public_api.rs) forces either original lane to be shared at every complete pair. Native runs cover depths 1–20, exhaustively through depth 16; Rosetta runs exhaustively through depth 14. Mixed forests cover 0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 24, and 33 trees, with interleaved nonzero constants, repeated features and predicates, all three NaN modes, signed quiet/signaling NaNs, NaN thresholds, infinities, signed zeros, and subnormal boundaries. Separate constant forests cross chunk boundaries. Leaves encode independent tree fields in at most 53 bits, so reassociation and cancellation cannot hide a wrong selected leaf. Results are in [probes/public-api.log](probes/public-api.log). Rerun from the worktree root with the supplied LLVM environment and `CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml`.

The runtime depth coverage is bounded. `src/jit.rs:229` limits the complete constants array to `u32::MAX` entries, making 31 the largest possible individual compiled-tree depth on this host. Depths 21–31 were inspected for bit-width and permutation bounds but not allocated or executed; a depth-31 tree alone needs 16 GiB of input leaves before LLVM allocations. In the two-lane path at even depths, each lane's highest bit supplies a negative weight. At odd depths, false padding removes that weight from the second lane. Swapping complete real pairs preserves the mapping, and the bias makes all reachable slots contiguous. The AVX2 path directly permutes level bits. These source checks support the deeper cases but are not runtime coverage of them.

The prediction implementation passes this review within the stated validation limits. The new layout and lane ordering preserve selected leaves in every executed case, including AVX2's permuted tables and odd depths. Fix the silent CSV failure and update the stale descriptions. Spend the next small cleanup on permanent cross-target tests and shorter assembly assertions; the evidence does not justify another kernel redesign or a new performance claim.
