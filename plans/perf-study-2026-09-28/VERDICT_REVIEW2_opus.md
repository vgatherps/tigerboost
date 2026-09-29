## Review of tigerboost at 166a9d6

I found no correctness bugs. The main findings are a test gap around the new AVX2 swapped-leaf path, one easy SSE2 win (about 6% fewer instructions), one simplification that leaves the emitted assembly unchanged, and several stale doc statements.

Worktree `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-a57a89688eb80e932`, HEAD 166a9d6. `git status --short` shows only `?? probes/`. No tracked file was touched.

### Correctness: no disagreement found

I checked `compile(...).predict(row)` against `Trees::predict` with a random forest generator, `probes/fuzz`. Leaves are random integers below 2^36, so the sum is exact however it is reassociated, and any wrong leaf shows up as a mismatch. The generator covers:
- 0 to 100 trees and 1 to 16 features, with all three NaN modes mixed.
- Depth 0 to 14 (0 to 20 in a separate run), with repeated features and exact duplicate predicates.
- NaN, ±inf, ±0, subnormal, MIN/MAX and random thresholds.
- Rows drawn from the specials plus each threshold and its `next_up` / `next_down`.
- Scale in {1, −1, 2, 0.5} and an integer bias.

About 20% of models trigger the x86 lane swap and a similar share the ARM64 swap. I counted this by mirroring `order_lanes` through the public plan.

| Run | Cases | Mismatches |
|---|---:|---:|
| Native ARM64, debug build (overflow checks on) | 5,300 | 0 |
| Native ARM64, release build | 22,000 | 0 |
| Native ARM64, depth up to 18–20 | 212 | 0 |
| x86 SSE2 under Rosetta, via `emit_assembly_with` + `tools/x86_check.c` | 610 | 0 |
| x86 AVX2 under Rosetta | 1,010 (228 with swapped pairs) | 0 |

A flipped bit in `expected.bin` makes the AVX2 harness fail, so the Rosetta check does catch mismatches. I also read the code paths by hand and they hold: the odd-depth padding pair is never swapped (`chunks_exact_mut` over `depth` only), padding reuses lane 0's value against +inf so it is always false in every NaN mode, and `raw_of` matches `vsli` / shift-and-or / movemask for swapped pairs. Both debug and release `cargo test` pass, and clippy is clean.

### Findings

1. **Minor, test gap for the newest code** (`examples/x86_check.rs:425-444`, `tests/jit.rs:23-82`, `README.md:108-112`). The AVX2 kernel with permuted leaves never runs in `cargo test` on the ARM64 dev machine:
   - The lane-swap test builds an AVX2 kernel only on an AVX2 host.
   - The macOS Rosetta gate test runs only the SSE2 kernel on `Distinct(5, 8)`, which has one x86 swap.
   - The README gate list (`special`, `depths`, `depths2`, with and without `--avx2`) has zero swaps for either target.

   | Workload | Full pairs | x86 swaps | ARM64 swaps |
   |---|---:|---:|---:|
   | special | 16 | 0 | 0 |
   | depths | 12 | 0 | 0 |
   | depths2 | 13 | 0 | 0 |
   | 5x8 | 15 | 1 | 0 |
   | 100x100 | 300 | 54 | 66 |
   | 300x100 | 900 | 172 | 161 |

   Only the unit test's static leaf-slot check (one tree) plus a manual `300x100 --avx2` run cover it. Rosetta does run AVX2 correctly, so the fix is cheap: run the gate test with `--avx2` on a workload that has swaps (for example 100x100), and add that command to the README list.

2. **Minor, the generic two-lane path loads a constant from memory in every chunk** (`src/jit.rs:595-608`, `raw_of` at `src/jit.rs:790-814`).
   - The mask merge `(m << p) | (acc & (2^p − 1))` needs a vector constant for `p ≥ 2`. At depth ≥ 5 on x86 SSE2 this costs `movabsq $.LCPI; movdqa` in every chunk plus a `pand` per tree: 13 constant-pool loads at 100 trees, 38 at 300.
   - This goes against what `pointer_parameters_have_bounds` (`src/jit.rs:1693-1698`) asserts. That test passes only because its trees are depth 3.
   - The fix is easy win #1 below.

3. **Minor, stale docs:**
   - `PLAN.md:38` says the block holds "thresholds ... in first-use order, then each tree's leaf block". It is now laid out per chunk: that chunk's thresholds in swapped lane order, padding to 8 entries, then its leaf blocks.
   - `PLAN.md:45` says the AVX2 "leaf block keeps its original order". That is now false: AVX2 leaves are permuted whenever a pair is swapped (54 pairs at 100x100).
   - `PLAN.md:35` and `README.md:81-82` say "per-chunk threshold base ... so paired loads stay within `ldp` range". That base now addresses the leaves too.
   - `PLAN.md` "Open" and `RESULTS.md:282` still list "ARM64 leaf-offset layout study", which 91951d5 did.
   - The layout slice's numbers are not in `RESULTS.md` (they are only in the commit message and `VERDICT_layout_astra.md`). The README timing tables (85 / 228 ns) predate that slice.

4. **Minor, stale test premises:**
   - `dropped_model_and_last_tree_leaf` (`src/jit.rs:1741`) asserts `preceding_leaves > 4095` to force large ARM64 leaf offsets. With per-chunk layout the last tree's offset is chunk-relative and small, so the assertion no longer guards anything. Deep trees still reach the large-offset path, and my depth-14 runs covered it.
   - `empty_and_constant_forests_need_no_row_loads` (`tests/jit.rs:270`) never checks row loads.

5. **Minor, misleading inspection output** (`examples/inspect_code.rs:45-58`). The plan dump is written three times with identical content, and it lists `plan.predicates()` in the order before lane swaps. For swapped pairs it no longer matches the threshold constants in the target's `.s` file.

6. **Nit, very deep trees.** `order_lanes` does `swaps |= 1 << number` on a `u16`, so depth ≥ 33 would overflow (a panic in debug, wraps in release) before any error is returned. Depth 32 gets a clean LLVM array-length error. This cannot happen in practice because depth 33 needs a 64 GiB leaves vector.

### Easy wins and simplifications, ranked by value per effort

| # | Change | Effect (verified) | Cost |
|---|---|---|---|
| 1 | In the generic two-lane path (non-ARM64), merge masks by adding instead: `acc = acc + (m << p)`. Lane values become minus the pattern bits, which is still a bijection. `raw_of` needs a third mode (ARM64 `sli` insert / generic add / movemask). | SSE2 instructions 2267 → 2134 at 100 trees and 6859 → 6454 at 300 (−5.9%). Constant-pool loads 13 → 0 and 38 → 0. Stack ops in continuations 6 → 2 and 42 → 10. ARM64 and AVX2 assembly byte-identical. Full test suite and 320 Rosetta SSE2 fuzz cases pass. Untimed on native x86, but it has fewer instructions and a shorter dependency chain. Prototype: `probes/addvariant`. | about −5 emitter lines, +4 in `raw_of` |
| 2 | Address thresholds through one origin-relative path: `chunk_base.unwrap_or(constants)` plus `slot - leaf_origin`, like leaves already do. Use a one-index f64 GEP in the AVX2 path too. | Assembly byte-identical on 100x100, 300x100, depths, depths2 and special for all three targets. Removes the if/else (`src/jit.rs:524-543`), a second GEP form (`:477-482`) and the unreachable "threshold slot precedes chunk base" error. One test line changes (`pointer_parameters_have_bounds` GEP operand index). Prototype: `probes/simplify`. | about −12 lines |
| 3 | Add a swap-bearing `--avx2` Rosetta run to the macOS gate test and the README list (finding 1). | Covers the newest emitted path in `cargo test`. | about 3 lines |
| 4 | Fix the PLAN/README/RESULTS text in finding 3, and rename `leaf_origin` to a chunk-origin name once it serves thresholds too. | Docs only. | about 10 lines |
| 5 | Add one test helper to read the constants block. Two unit tests use a builder `build_extract_value` trick (`src/jit.rs:1012-1060`, `1322-1360`) and the example test parses `.quad`. | Coverage unchanged. | about −30 test lines |
| 6 | Compute `raw_of` once per pattern in `permuted_leaves` (it runs twice now). | Compile time only. | about 3 lines |

One tempting simplification does not work: always emitting the explicit `vzeroupper` in the last AVX2 chunk (dropping `has_wide_compare`, `src/jit.rs:694-697`). It adds a spill and reload of xmm0 (`vmovsd %xmm0, -8(%rsp)`) on 4 of 5 workloads, so the condition must stay.

I found nothing cheap left on ARM64. Each tree costs 3 `fcmgt` + 2 `sli` + 2 vector-to-GPR moves + 2 `add` + `ldr` + `fadd`. Fusing the two moves into one would take a per-lane shift and cost more vector ops, and the leaf immediate cannot be zero because of the bias. The 61 `mov.16b` copies come from pairs where both features are shared, which the study already measured as neutral.

### Verdict

The one-kernel crate with per-chunk layout and per-target lane order is correct as far as I could probe. About 28k random forests matched the reference exactly on native ARM64 and under Rosetta for SSE2 and AVX2, with swaps, odd depths, deep trees and all NaN modes. The new code is sound; its weak spot is test coverage. The AVX2 swapped-leaf path, the newest emitted code, never runs in the default `cargo test` or the README's Rosetta list, and that should be fixed first. The best easy win is add-based mask merging on the generic path: about 6% fewer SSE2 instructions, no per-chunk constant-pool loads, fewer stack ops, and no change on ARM64 or AVX2. Threshold addressing can be unified with identical assembly. `PLAN.md` still describes the old layout and says the AVX2 leaves keep their original order, which is now wrong.

Everything is in `probes/`:
- `probes/fuzz` — the fuzzer (`tb-probe-fuzz <start> <count> [sse2|avx2]`, env `MAXDEPTH`, `TREES`, `KEEP`) and `examples/swaps.rs` for swap counts.
- `probes/addvariant` — the add-merge prototype.
- `probes/simplify` — the unified-addressing prototype.
- `probes/compare_asm.sh` — diffs a variant's assembly against master.
- `probes/asm*` and `probes/out-*` — the emitted assembly.