Minor at `src/jit.rs:1553` and `src/jit.rs:1570`. `constants_block_layout_matches_plan` emits for the host but still expects the ARM64 leaf permutation. After the add-based SSE2 change, this assertion is wrong on x86-64. The depth-three tree in `mixed_depth_model` produces these leaf blocks:

| Emission | Leaf block |
|---|---|
| ARM64; hardcoded test expectation | 24, 25, 20, 21, 26, 27, 22, 23 |
| SSE2 | 27, 26, 23, 22, 25, 24, 21, 20 |

Reproduced by copying the crate under `probes/x86-fixture` and changing only this test's emission triple to `x86_64-unknown-linux-gnu`. The unchanged assertion fails before native JIT execution. This was an explicit x86 emission test on ARM64, not a native x86 Cargo run. Run the fixture for both explicit targets with their own exact expected arrays. Do not change the correct SSE2 permutation to satisfy the stale fixture. Reproduction: `probes/reproduce_layout.py`; failure: `probes/x86-layout-failure.log`.

Minor at `src/jit.rs:699`. AVX2 emission inserts an explicit `vzeroupper` whenever the last chunk has no wide comparison, even when the entire forest has depth at most two and never uses a YMM register. For two depth-one trees, the final continuation unnecessarily stores xmm0 to `-8(%rsp)`, clears the upper halves, and reloads xmm0. Constant-only forests do this too. Require an earlier wide comparison before inserting this explicit clear; retain it for shallow tails following wide chunks. Removing just these three assembly instructions preserved bit-exact Rosetta results for depth-one, depth-two, and constant-only fixtures. Evidence: `probes/cases/forced-1-0/avx2.s`, `probes/cases/forced-2-0/avx2.s`, `probes/cases/constant-1/avx2.s`, and their `no-clear.s` / `no-clear.log` siblings. No native x86 latency claim follows from that assembly experiment.

No blocker or major finding. No selected-leaf disagreement was found. Review covered production Rust, tests, examples and support code, the C driver, README, PLAN, RESULTS, both previous review reports, and the polish changes at `de6693e`. HEAD was `e466d0eb33e08e6dafe0851c3a83f33aeb093f03`. Tracked files remain unchanged; no commits, formatting, or delegation were used.

The add-based emitter at `src/jit.rs:624` and `raw_of` at `src/jit.rs:808` both accumulate negative shifted comparison bits. Thresholds and leaves use the same chunk origin, with x86's origin zero. Swaps affect only complete real pairs; padding compares against positive infinity and remains false. The wider swap mask fits every validated model depth. The constants-array length check precedes permutation, limiting an individually compilable tree to depth 31 on this host. Depths 21–31 were checked in source for shift widths and address bounds, not allocated or executed.

`probes/public_api.rs` exercises only public model, planning, compilation, and assembly-emission APIs. Each tree's selected leaf occupies a separate bit field in an exactly representable sum. This prevents two wrong selections from cancelling. Native predictions run through both `compile` and `compile_plan` after dropping the source model, with short-row rejection and offset slices. Rosetta executes separately emitted SSE2 and AVX2 code using the existing C driver.

| Probe | Models | Predictions | Result |
|---|---:|---:|---|
| Native ARM64, both compilation entry points | 170 | 606,800 | Exact agreement |
| SSE2 under Rosetta | 162 | 287,016 | Exact agreement |
| AVX2 under Rosetta | 162 | 287,016 | Exact agreement |

Forced-sharing fixtures cover both lane choices at depths 1–20 natively, exhaustively through depth 16; Rosetta runs the exhaustive depth-1–16 cases. Mixed forests cover 1, 2, 3, 7, 8, 9, 15, 16, 17, 24, and 33 trees, repeated features, odd depths, nonzero constants, all NaN modes, signed signaling and quiet NaNs, infinities, signed zeros, subnormals, arbitrary floating-point bit patterns, and exact/adjacent thresholds. Separate constant forests cross chunk boundaries. Empty forests with zero or two declared features preserve positive zero, negative zero, and signed NaN bias payloads with a NaN scale. Native probes include 61 models with ARM64 swaps; each Rosetta path includes 52 models with x86 swaps. Results: `probes/public-api.log`. Rerun with `LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21 CARGO_TARGET_DIR=target cargo run --offline --release --manifest-path probes/Cargo.toml --bin public_api`.

| Check against unchanged tracked source | Result |
|---|---|
| Offline debug and release tests, all targets | 67 passed in each profile |
| Doctest | 1 passed |
| Clippy, all targets, warnings denied | Passed |
| Release example build | Passed |
| Existing `x86_check`, `special` and `300x100`, SSE2 and AVX2 | All four runs passed, 256 rows each |

Logs are `probes/test-debug.log`, `probes/test-release.log`, `probes/test-doc.log`, `probes/clippy.log`, `probes/build-examples.log`, and `probes/rosetta-*.log`. The Cargo logs contain intentional mismatch and invalid-assembly diagnostics from the gate's negative tests. LLVM was 21.1.8. Tests used the supplied LLVM prefix and rustdoc zstd link flag.

The previous reviews' requested changes are present and work within this coverage: the AVX2 swap-bearing Rosetta gate now runs; add packing removes the SSE2 mask constants; threshold addressing is unified; `ordered_predicates` supplies one inspection file with both target orders; benchmark headers use the actual sum count; obsolete test premises and extraction machinery are gone; unswapped AVX2 leaves use the identity path; and empty forests return bias without a continuation. README, PLAN, and RESULTS now describe the current layout and identify historical measurements. The remaining stale expectation is the host-dependent layout test above. The excluded CSV flush issue was not revisited.

Fresh assembly counts reproduce the polish report. Counts exclude directives and alignment padding; stack operations include push/pop. ARM64 and AVX2 retain zero continuation stack operations on these workloads. SSE2 has no `.LCPI` mask references.

| Target | Trees | Continuation instructions | Wrapper instructions | Continuation stack operations |
|---|---:|---:|---:|---:|
| ARM64 | 100 | 1,974 | 26 | 0 |
| ARM64 | 300 | 5,876 | 26 | 0 |
| SSE2 | 100 | 2,134 | 15 | 2 |
| SSE2 | 300 | 6,454 | 15 | 10 |
| AVX2 | 100 | 1,334 | 15 | 0 |
| AVX2 | 300 | 3,978 | 15 | 0 |

Assembly is under `probes/assembly-100` and `probes/assembly-300`; counter and results are `probes/count_assembly.py` and `probes/assembly-counts.log`. These are separate target emissions, not MCJIT disassembly.

Inlining `CompiledModel::predict` regressed the 300-tree timings, so it is not a recommended win. The one-line prototype removes a Rust call frame and the `Result` store/reload, but all three paired 300-tree hot runs were slower. The small cold run was slower too. Runs alternated order for hot measurements; another reviewer was active, and the cold sample count was only 32. These numbers reject a speedup claim; they do not establish a stable regression magnitude.

| Workload | Baseline hot medians across three runs, ns | Inline hot medians, ns | Baseline → inline shuffled cold p50 after 64 MiB, ns |
|---|---|---|---|
| 100x100 | 89.41, 90.70, 88.56 | 90.60, 89.97, 89.34 | 1,292 → 1,208 |
| 300x100 | 232.19, 233.85, 233.73 | 236.58, 237.63, 235.55 | 3,500 → 3,834 |

The prototype is isolated under `probes/inline-variant`; its ten JIT integration tests passed. All three 100-tree emitted kernels are byte-identical to baseline. Reproduction and numbers are `probes/make_inline_variant.py`, `probes/inline_measure.py`, and `probes/inline-numbers.json`; caller assembly is in `probes/scalar-driver-excerpt.asm` and `probes/inline-driver.asm`.

Remaining work ranked by value per effort:

| Rank | Location and change | Value and effort |
|---:|---|---|
| 1 | `src/jit.rs:1546`: run the layout fixture for explicit ARM64 and SSE2 targets with exact target-specific expected leaves. | Fixes the reproduced x86 test failure and exposes it on ARM64. Small test change; no kernel change. |
| 2 | `examples/x86_check.rs:428`, `tests/differential.rs:98`: retain a small encoded mixed-forest Rosetta fixture for both paths, plus empty-bias cases. | Existing 100-tree AVX2 gate covers swaps, but its 1–16 leaf values cannot identify every tree's selection independently. The randomized encoded test currently compiles only the default path. Reuse a small subset of the public probes; no new harness required. |
| 3 | `src/jit.rs:699`: omit the explicit final clear when the forest never performs a wide comparison. | Removes three instructions and two stack accesses in inspected shallow/constant AVX2 kernels. Add narrow-only absence tests while preserving the existing wide-to-shallow tail tests. Native x86 benefit remains unmeasured. |
| 4 | `src/jit.rs:252`: remove `entry_count = entry_count.max(1)`. `src/jit.rs:1738`: remove the now-identical `operand` tuple element and use `get_operand(1)` directly. | Small remnants of the removed empty-forest path and two GEP forms. Every nonempty validated forest contributes at least one leaf, and both threshold GEPs now have one index. No behavior change or new abstraction. |

The prediction implementation passes this review within the stated execution limits. Fix the stale x86 layout assertion; keep the current kernel and the recent packing/addressing changes. Stronger permanent cross-target fixtures offer more value than another kernel rewrite. The demonstrated remaining saving inside emitted kernels is confined to shallow AVX2 forests, and the public-API inlining change did not improve measured performance. No tracked implementation change is included in this verdict.
