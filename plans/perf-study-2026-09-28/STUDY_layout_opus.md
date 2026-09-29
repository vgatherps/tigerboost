## Layout and ordering study, tigerboost at 881825f

Worktree `/Users/samschetterer/src/tigerboost/.claude/worktrees/agent-a7555f8422ff47e3b`. `git status --short` shows only `?? probes/`. No tracked files were touched.

**Bottom line.** Two emitter changes cut instructions with no downside I could find: per-target lane order within each pair, and putting each chunk's leaves next to its thresholds. A better tree order would cut 7–8% of feature loads, but it costs seconds of planning and makes x86 spill. Neither timed win clears the noise. Every variant is within about 1.5% hot on ARM64, and the machine had a load average of 20–27 during the runs (the other agent was building). All claims below rest on static counts from the emitted assembly.

### Probe setup
- `probes/tb/` is a copy of the crate with env hooks:
  - `TB_ORDER=<file>` sets an explicit visit order.
  - `TB_LEAF=chunk` puts each chunk's leaves next to its thresholds.
  - `TB_THR=block` loads thresholds as `<4 x double>` blocks.
  - `TB_VLOAD=1|single` loads an adjacent feature pair as one vector.
- `probes/tb/examples/probe.rs` has `stats`, `order`, `emit` and `bench` subcommands. `TB_LANES=plain|swap|rswap|adj` permutes each tree's levels and remaps its leaves.
- `probes/py/arm_count.py` and `probes/py/x86_count.py` classify the emitted instructions.
- The probe's baseline assembly is byte-identical to the real crate's.
- Every variant was compiled natively and all 256 rows match `Trees::predict` of the original model.
- ARM64 baseline: 2060 instructions at 100 trees, 6247 at 300. Of the doubles loaded, 356 / 974 are features, 300 / 900 threshold vectors and 100 / 300 leaves. No spills.

### Ranked by value per effort

**1. Leaves next to their chunk's thresholds, addressed from the chunk base (ARM64).**
- Today the leaves sit in model order after all thresholds. Any leaf past 32 KiB needs extra address instructions: 45 at 100 trees (a `mov w` plus register offset); at 300 trees 198 `add …, lsl #12`, 66 `mov` and 3 `movk`, about 267 in all.
- The change: lay each chunk out as [its thresholds, padded to 8 slots][its trees' leaf blocks], and address leaves from the existing threshold-base argument `x22`. No new register or argument is needed.
- Result: all leaf extras go away. Each chunk hop now costs 2 instructions instead of 1, because block bases grow past 4 KiB.

| | 100 trees | 300 trees |
|---|---:|---:|
| ARM64 instructions | 2060 → 2026 (−34, −1.7%) | 6247 → 6009 (−238, −3.8%) |
| unpaired threshold `ldr q` (vectors loaded unchanged) | 24 → 30 | 46 → 80 |

- x86 SSE2 and AVX2 are unchanged (6925 / 4151 at 300), because they use a 32-bit displacement.
- These are integer instructions, not loads. Hot time did not move measurably.
- Code cost: about 40 lines in the constants-block layout in `emit_plan`, plus changing the `constants_block_layout_matches_plan` test.

**2. Lane order within a pair, chosen per target (swap the two lanes of a pair, and its leaf mapping).**
- When a feature is used by other lanes in the chunk, where it should go depends on the target.
- On ARM64 it belongs in lane 1. An insert is destructive, so a shared value in lane 0 costs a `mov.16b` copy, and the single-use value then goes in through `add` plus `ld1` lane load.
- On x86 it belongs in lane 0. There lane 1 folds into `movhpd`/`vmovhpd` from memory.

| variant | ARM64 100 / 300 | SSE2 100 / 300 | AVX2 100 / 300 |
|---|---:|---:|---:|
| baseline | 2060 / 6247 | 2295 / 6925 (stack ops 4 / 46) | 1390 / 4151 |
| shared feature in lane 1 (`swap`) | **2006 / 6110** | 2330 / 6985 | 1453 / 4305 |
| shared feature in lane 0 (`rswap`) | 2110 / 6420 | **2284 / 6892 (stack ops 2 / 12)** | **1361 / 4026** |

- On ARM64, `swap` changes these counts: `mov.16b` 97 → 57 (100) and 326 → 228 (300); `ld1` lane loads 97 → 31 / 213 → 50; row address adds 96 → 32 / 215 → 53; `mov.d` goes up 296 → 361 / 973 → 1129.
- Feature doubles loaded are the same (356 → 358, 974 → 974), so load slots do not change.
- Combined with change 1 on ARM64: 1973 / 5882, which is −4.2% / −5.8%.
- The lane order is target-specific, so it belongs in the emitter, not the shared plan. On ARM64 and SSE2 this means composing a per-tree lane swap into `permuted_leaves`. The AVX2 path would need a leaf permutation for the first time; today its leaf table is in original order.
- Code cost: about 60–80 lines plus tests.

**3. Threshold `ldp q` pairing: nothing robust to gain.**
- Unpaired `ldr q` is 24 of 162 threshold load instructions at 100 trees and 46 of 473 at 300. Perfect pairing would save at most 12 / 23 instructions, and no load slots if an `ldp q` counts as two.
- The count moves with incidental scheduling changes (22 to 80 across my variants).
- Forcing pairs with `<4 x double>` block loads plus shuffles made it worse: unpaired 24 → 64 and 46 → 166, instructions +16 / +60.
- x86: AVX2 already folds the threshold into `vcmpgt_oqpd`. SSE2 cannot fold it, because `cmpltpd` needs the threshold in the destination register, and the other operand order is unordered-greater, which is wrong for NaN.
- No change recommended.

**4. Tree order (question 1): the heuristic is 7–8% above what local search finds, and closing that gap is not worth it now.**
Sum over chunks of distinct features:

| method | 100×100 | 300×100 | 100 corr | 300 corr |
|---|---:|---:|---:|---:|
| model order | 468 | 1419 | 358 | 1090 |
| `shared_feature_order` | 358 | 974 | 173 | 419 |
| greedy extending by fewest new features | 353 | 958 | – | – |
| current order plus pairwise-swap hill climb | 351 | 958 | – | – |
| simulated-annealing swap search, 8×20M iterations | 333 | 897 | 154 | 331 |
| same, 4×200M iterations | – | 892 | – | – |
| trivial lower bound, Σ⌈uses_f/8⌉ | 113 | 263 | 105 | 204 |

("corr" is a synthetic variant where each tree draws its features from a window of 8 consecutive indices.)

- The cheap planner tweaks get only 1.5–2%. Annealing saves −25 / −77 feature doubles, about −3.3% / −3.5% of all load slots.
- In the assembly: ARM64 row load instructions 330 → 297 and 886 → 832, but total instructions go −21 / +10 (more copies, unpaired thresholds +12 / +20).
- x86 gets worse: SSE2 +61 / +179 instructions with stack ops 4 → 32 and 46 → 164; AVX2 +21 / +48 with stack ops 0 → 4 and 0 → 32.
- Planning time goes from microseconds to seconds.
- By the earlier shared-order result (445 fewer doubles bought about 11.6 ns at 300 trees), 77 doubles is worth about 2 ns, which is below the noise.
- Not worth a planner change. If revisited, it needs an x86 register-pressure term.

**5. Pairing levels whose features are adjacent in the row, or the same feature (question 2): no load reduction.**

| | pairs that are the same feature or adjacent ascending, level order | best pairing |
|---|---:|---:|
| random 100 trees | 8 / 300 | 33 / 300 |
| random 300 trees | 18 / 900 | 123 / 900 |
| correlated 100 trees | 66 / 300 | 228 / 300 |
| correlated 300 trees | 193 / 900 | 692 / 900 |

- LLVM never makes one `ldr q` out of an adjacent pair, because those features are almost always reused elsewhere in the chunk.
- Emitting an explicit `<2 x double>` row load for adjacent pairs cuts instructions (correlated 1817 → 1730 and 5467 → 5217) but adds loads, because the other scalar uses are no longer merged with it: row load instructions 137 → 202 and 329 → 503, feature doubles 173 → 282 and 419 → 729. That loses on the binding resource.
- Restricting it to pairs where both features are used once in the chunk leaves only 2–3 pairs.
- Pairing to maximise same or adjacent features does help on correlated models: ARM64 1905 → 1817 (100) and 5770 → 5467 (300); AVX2 1202 → 1117 and 3535 → 3274. The saving is `dup` and fewer copies, not loads. On random models it adds nothing beyond item 2.
- Not recommended until a real model shows correlated feature indices.

### Other observations
- The current SSE2 kernel spills at 300 trees: 46 stack operations, and 4 at 100 (`movapd …, -24(%rsp)`). The shared-feature order raised register pressure there. Lane order `rswap` cuts it to 12.
- Hot timings (15 rounds × 50k calls; load average 20–27):
  - At 100 trees: base 89.5, swap 88.5, adj 88.0, leaf 88.5, leaf+swap 88.0, annealed order 88.1 ns.
  - At 300 trees, two runs: base 234.9 / 235.9, leaf 237.3 / 234.7, others 234.6–237.0 ns.
  - None of this separates from noise. It fits load pipes being the binding resource: the wins above are integer and vector-move instructions.

### Recommendation
- If anything is done, do changes 1 and 2. They are about 120 lines, emitter-only, and give −4 to −6% ARM64 instructions and −2 to −3% on x86 with fewer SSE2 spills.
- Time them on a quiet machine before merging. Expect little hot gain on the M4. The x86 effect is untimed.
