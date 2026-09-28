# Review reconciliation (2026-09-28)

Two independent reviews of `SYNTHESIS.md` (opus 5.5: `VERDICT_REVIEW_opus.md`; gpt-6-astra:
`VERDICT_REVIEW_astra.md`). No blockers. Both request the same revisions. This note records what
changes in the proposal and the agreed experiment order. Where this note and `SYNTHESIS.md`
disagree, this note wins.

## Maintainer decisions taken during review

- Real models do not share thresholds. The binarize formulation (SYNTHESIS 5.6 and the shared-border
  rows in section 3) is dropped entirely. Nothing is optimized for shared borders.
- No AVX-512 dependence. NEON and AVX2 decide; SSE2 is the x86 fallback.
- Thresholds and leaves come off a base register through a pointer argument. Decided.

## Corrections to SYNTHESIS

1. NEON tree-major is not shown faster than the scalar path. Same-machine medians at 100/300 trees:
   constants pointer + per-chunk pointer alone 94.1/259 ns, generated tree-major 93.5/263, scalar
   carry 86.8/237. The tree-major numbers come from a different harness (generated kernels with
   arbitrary constants, plain-call chunks, different row cycle), so they are motivation for a
   controlled comparison, not a ranking.
2. The residency forecast ("22 → 16 instructions per tree, ~40% fewer loads") was about 3x too
   optimistic. From the real plan on 100x100, 8-tree windows touch 34-42 distinct features; only 22%
   (tree order) to 29-35% (GP4 order) of lane fills can come from a register. Eight registers already
   capture that reuse. Expect ~20 instructions per tree and 13-20% fewer load slots, not 40%.
3. `ldp q,q` costs two load slots on M4 (measured), `ldp d,d` and `ldr q` one. Threshold vector loads
   are therefore not cheaper than paired scalar loads. Tree-major's load floor is ~275-290 cycles per
   100 trees against the scalar 300-cycle compare floor. Headroom exists but is under 12%.
4. `llc` emits `mov.d` plus a copy, not `zip1`, when building a pair from two scalar-inserted values.
5. The x86 carry fallback (`llvm.x86.addcarry.64` with `next_up(t)` thresholds) is wrong when MXCSR
   DAZ is set: feature 0 or a denormal against a 0.0 border selects the wrong leaf (verified under
   Rosetta). The x86 scalar control is astra A's `select(bit, index + mask, index)` form
   (`lea; ucomisd; cmovbe`), 2458 instructions with the shared block and ADD offsets.
6. Leaf-table permutation for the NEON `sli` reduction depends on depth: bias 10 / 36 / 136 for depth
   4 / 6 / 8 (formula `2^(p-1) * (1 + 2^p)` for depth `2p`), and the lane order must be the plan's
   actual level order or the table permuted to match. ADD-ed block offsets work with mixed depths and
   need no `2^d` alignment. Odd depths need a defined tail.
7. `AsTrue` lanes cannot share a vector compare with `AsIs`/`AsFalse` lanes; normalize `AsTrue`
   inputs before the common ordered compare, or group lanes. `AsIs` and `AsFalse` need no load-time
   normalization (1089 special-value cases checked natively against `Trees::leaf_indices`).
8. Chunk signatures: `Slots` vectors never shrink, and `emit_plan` sizes parameters from vector
   length, so an expanded within-chunk resident set can push live features into stack arguments even
   when only six are live at the boundary. Whole-tree actions need explicit compact boundary slots;
   the budgets can leave the public API only after that redesign.
9. Cold calls: independent leaf misses do overlap (about 13 in flight); linear growth with tree count
   means a fixed concurrency, not zero overlap. The constants block adds ~75 (100 trees) / 225 (300)
   threshold lines per call on ARM64, so a hot win can regress cold. Every change is now judged on a
   hot and a cold column measured on identical rows.
10. x86 300-tree tree-major code is 33 KB, still over a 32 KB L1I. "3x" is the Alder Lake llvm-mca
    figure; Zen 3 is 1.7x. MCJIT ignores the target machine's CPU string; pin `x86-64-v3` through the
    function attributes.
11. GP4 admission order costs ~26-50 ms of planning at 300 trees, not sub-millisecond; only literal
    tree order is that cheap.
12. Sums: 1/2/4/8 do not time identically (117.8 / 115.5 / 125.3 / 126.4 ns at 100 trees), because
    the sweep also changes residency and argument pressure. Keep two; nothing favors more today.

## Agreed experiment order

1. Comparable hot and cold harness (`examples/cold_bench.rs`): shuffled all-256-row and dependent
   traces, timer-only control, data pollution 256 KiB / 4 / 16 / 64 MiB, other-model call, checked
   `predict` path, raw per-call p50/p90/p99.
2. Land the constants block in production (`src/jit.rs`): one private constant block, pointer
   argument through wrapper and every continuation, ARM64 per-chunk pointer for `ldp` pairing,
   ADD-ed leaf offsets, wrapper + continuation even for one chunk. Controls: GP12/FP9 and GP4/FP9.
   Expected hot: 121 → ~95 ns and 345 → ~261 at 100/300 trees; cold has no verified number yet.
3. Opt-in whole-tree action with tree-boundary chunks (2/4/8-tree sweep), NEON `fcmgt.2d` + `sli`
   and AVX2 `vcmpgt_oqpd mem` + `vmovmskpd` (SSE2 fallback), original level order first, two scalar
   sums, no speculative cross-boundary residency; compared on the same harness against the scalar
   constants path and the scalar controls (ARM64 inline-asm carry, x86 select form). A target keeps
   tree-major only if it beats its scalar control on hot and does not regress cold.
4. Within-chunk feature retention for lane fills, then feature-overlap admission, then optional
   cross-boundary retention capped at six, each measured separately.
5. Four sums (two `<2 x double>` on NEON, four scalar on x86) only when the integrated kernel is fast
   enough for the chain to show; grouped early leaf issue as a separate cold experiment.
