status: pass

Both emitter changes are unconditional. All required gates passed. No timings collected.

At 300 trees, SSE2 still has 30 stack operations: down from 46 before this change, but above the study's 12. At 100 trees, ARM64 has 1974 instructions, one above the study's combined result. These results are retained without workarounds.

`emit_plan` now lays out each chunk's thresholds, pads that threshold region to eight entries, then stores its trees' leaf blocks in visit order. Constant trees retain one entry. ARM64 leaf addresses use the existing third pointer argument; x86 uses constant displacements from the constants pointer. Checked allocation arithmetic and exact remaining-byte `dereferenceable` bounds remain in place. Placement alone reproduces the study's ARM64 counts exactly and leaves x86 instruction counts unchanged. Leaf-offset materializations fall to zero.

`order_lanes` counts real feature uses within each chunk. Exactly one shared feature goes in lane 1 on ARM64 and lane 0 on x86. Both-shared and neither-shared pairs retain level order; odd-depth padding does not count as reuse and is not swapped. The emitter reorders thresholds with their features. `raw_of` simulates the swapped source bits for both paths, and `permuted_leaves` constructs each leaf table from that simulation. The shared plan is unchanged.

Counts cover continuation functions, excluding the wrapper, matching the study's instruction-count boundary. Feature doubles count the full width of each row memory operand, including repeated and unused loaded elements. Leaf-offset materializations count extra immediate moves, `movk`, and shifted address adds feeding leaf loads; chunk-base updates are excluded. Stack operations count instructions referencing the stack. ARM64 `ld1` below means `ld1.d`; separate `ld1r.2d` counts stay at 1/3 for 100/300 trees in every variant.

| ARM64 trees | Variant | Instructions | Feature doubles | Leaf-offset materializations | `mov.16b` | `mov.d` | `ld1` | Stack operations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | Before | 2060 | 358 | 45 | 97 | 296 | 98 | 0 |
| 100 | Placement only | 2026 | 358 | 0 | 95 | 296 | 98 | 0 |
| 100 | Lane order only | 2009 | 358 | 45 | 61 | 362 | 32 | 0 |
| 100 | Both | 1974 | 358 | 0 | 61 | 362 | 32 | 0 |
| 300 | Before | 6247 | 974 | 267 | 326 | 973 | 213 | 0 |
| 300 | Placement only | 6009 | 974 | 0 | 328 | 973 | 213 | 0 |
| 300 | Lane order only | 6122 | 974 | 266 | 240 | 1132 | 52 | 0 |
| 300 | Both | 5876 | 974 | 0 | 234 | 1132 | 52 | 0 |

| Target | Trees | Variant | Instructions | Feature doubles | Leaf-offset materializations | Stack operations |
|---|---:|---|---:|---:|---:|---:|
| SSE2 | 100 | Before | 2295 | 365 | 0 | 4 |
| SSE2 | 100 | Placement only | 2295 | 365 | 0 | 4 |
| SSE2 | 100 | Lane order only | 2267 | 364 | 0 | 2 |
| SSE2 | 100 | Both | 2267 | 364 | 0 | 2 |
| SSE2 | 300 | Before | 6925 | 986 | 0 | 46 |
| SSE2 | 300 | Placement only | 6925 | 986 | 0 | 46 |
| SSE2 | 300 | Lane order only | 6859 | 981 | 0 | 30 |
| SSE2 | 300 | Both | 6859 | 981 | 0 | 30 |
| AVX2 | 100 | Before | 1378 | 367 | 0 | 0 |
| AVX2 | 100 | Placement only | 1378 | 367 | 0 | 0 |
| AVX2 | 100 | Lane order only | 1334 | 366 | 0 | 0 |
| AVX2 | 100 | Both | 1334 | 366 | 0 | 0 |
| AVX2 | 300 | Before | 4114 | 996 | 0 | 0 |
| AVX2 | 300 | Placement only | 4114 | 996 | 0 | 0 |
| AVX2 | 300 | Lane order only | 3978 | 985 | 0 | 0 |
| AVX2 | 300 | Both | 3978 | 985 | 0 | 0 |

The four variants give both marginal comparisons: Before → Placement only and Lane order only → Both isolate placement; Before → Lane order only and Placement only → Both isolate lane order.

| Target | Trees | Study before → both | This run before → both |
|---|---:|---:|---:|
| ARM64 | 100 | 2060 → 1973 | 2060 → 1974 |
| ARM64 | 300 | 6247 → 5882 | 6247 → 5876 |
| SSE2 | 100 | 2295 → 2284 | 2295 → 2267 |
| SSE2 | 300 | 6925 → 6892 | 6925 → 6859 |
| AVX2 | 100 | 1390 → 1361 | 1378 → 1334 |
| AVX2 | 300 | 4151 → 4026 | 4114 → 3978 |

Two verified differences prevent exact agreement with the study. Its `choose_perm` probe swaps according to relative use counts, including pairs where both features are shared. This task explicitly keeps those pairs in level order. Also, this worktree starts at `131c251`, whose `target-features` suppresses intermediate `vzeroupper`; the study source sets that attribute to an empty string. The 12/37 intermediate continuations account for the AVX2 baseline difference.

The study's ARM64 counter also loses row-pointer provenance at `mov x8, x20`, followed by `ld1.d { v4 }[1], [x8], #8` and `ldr d15, [x8]`. Correcting that count changes the 100-tree baseline from 356 to 358 feature doubles and from 97 to 98 `ld1.d` instructions. The new counter follows copied row pointers. x86 counts include two doubles for `movupd` and four for 256-bit memory operands, rather than counting every memory instruction as one double.

| Gate | Result |
|---|---|
| Both baseline `inspect_code` commands | Pass |
| `cargo build --offline --release --examples` | Pass |
| `cargo test --offline --all-targets` | Pass |
| `cargo test --offline --release --all-targets` | Pass |
| `cargo test --offline --doc` | Pass |
| `cargo clippy --offline --all-targets -- -D warnings` | Pass |
| Both final `inspect_code` commands | Pass |
| `x86_check special /tmp/tb-lay-x1` | Pass |
| `x86_check special /tmp/tb-lay-x2 --avx2` | Pass |
| `x86_check depths /tmp/tb-lay-x3 --avx2` | Pass |
| `x86_check depths2 /tmp/tb-lay-x4` | Pass |
| `x86_check 300x100 /tmp/tb-lay-x5 --avx2` | Pass |

New tests inspect emitted feature lanes and threshold constants for all three targets, check AVX2 leaf slots independently, and exhaustively execute `compile` for depths 1–10 with either original lane shared. Existing layout assertions now follow chunk visit order. The 300×100 example test checks every emitted threshold pair and leaf block and proves all 38 chunk regions fit within 32 KiB; full chunks occupy 4480 bytes. The existing pointer test checks exact bounds and chunk-relative threshold addresses. The public randomized test and x86 execution gates pass unchanged.

The placement-only snapshot passed the emitter tests. The lane-only measurement uses an isolated copy under `probes/lane-only`, applying only the lane change to the saved baseline; its public JIT and randomized differential tests also pass. The final production code has no variant switches. A final test-name clarification added an explicit no-swaps assertion; that test was rerun in debug/release, and Clippy was rerun.

Self-review checked chunk-relative leaf offsets, one-entry constants, empty forests, padding, exact pointer bounds, threshold/feature agreement, both-shared tie behavior, chunk isolation, and the shared permutation simulation. Surviving assertions were retained or updated to the required layout. Tracked changes are confined to `src/jit.rs`, `tests/jit.rs`, and `examples/inspect_code.rs`. No formatting command, commits, delegation, saturating arithmetic, or unrelated fixes.

Artifacts: [counts](probes/counts.json), [count script](probes/py/count_layout.py), [gate commands and results](probes/gates/results.json), [gate logs](probes/gates/), and [source snapshots](probes/snapshots/). Assembly directories are `/tmp/tb-lay-before{,300}`, `/tmp/tb-lay-layout{,300}`, `/tmp/tb-lay-lanes{,300}`, and `/tmp/tb-lay-after{,300}`. Final assembly matches the first combined capture byte for byte. The original study counters are retained under `probes/py` for comparison.
