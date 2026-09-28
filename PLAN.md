# CatBoost JIT plan

Status: the tree-major kernel is the only evaluation order. Constants come from one block through
a pointer argument, each tree is evaluated with two-lane (NEON) or four-plus-two-lane (AVX2,
opt-in) vector comparisons whose masks address a compile-time-permuted leaf table, and chunks of
whole trees are chained with `musttail` continuations that carry only pointers and sums. Measured
on the ARM64 host: 121 → 87 ns hot at 100 trees, ~345 → 237 at 300, cold neutral. x86-64 kernels
run and match the reference under Rosetta 2; no x86 speed has been measured. The [README](README.md)
documents the API and the numbers; `plans/perf-study-2026-09-28/` holds the study, the reviews and
the results that led here.

## Goal and agreed constraints

Single-row prediction latency for forests of fewer than 300 trees, depth about 6, one feature
row per call.

- Numeric, symmetric trees; `f64` features, thresholds, leaves, and arithmetic.
- Arithmetic may reorder. Split decisions and selected leaves must remain exact.
- LLVM 21 through Inkwell/MCJIT. One plan shape for ARM64 and x86-64; AVX2 is an explicit
  emission option, never a dependency; AVX-512 is out.
- Real models do not share thresholds across trees; nothing is optimized for shared borders.
- Immutable feature/model reads during prediction; no writes to caller/model memory.
- Every performance change is judged on a hot column (`scalar_bench`) and a cold column
  (`cold_bench`) on identical rows.

## Design as built

Validated model → tree-major plan → chunked LLVM functions.

The plan is one `EvaluateTree` action per tree in tree order (constant trees keep an
`Admit`/`Retire` pair), predicates padded with a constant-false `+inf` level on odd depths, leaves
accumulated round-robin into a few sums, chunks of `chunk_trees` whole trees. Nothing is live across
a chunk boundary except the row pointer, the constants pointer (plus a per-chunk threshold base on
ARM64) and the sums; `preserve_nonecc` passes those in registers.

The emitter builds one private constant block: thresholds one slot per predicate in first-use
order, then each tree's leaf block. A tree's thresholds are consecutive, so each lane pair or
lane quad is one vector load. Lane masks are packed with shift-left-insert on ARM64 (generic
shift/or elsewhere) into two signed lane values; the leaf block for that tree is permuted at
compile time by simulating the same arithmetic, and the permutation is asserted to be a bijection.
Under `EmitOptions { avx2 }` on x86-64, lanes map to level bits directly through `vmovmskpd`, the
threshold vector folds into `vcmpgt_oqpd`, and the leaf block keeps its original order.

## What was tried and rejected

- Rolling scheduler with logical GP/FP register budgets and feature residency across trees:
  worth 2 to 5% at best, because `preserve_nonecc` passes only eight FP arguments and every
  carried value beyond that spills; deleted.
- Two-lane pairs of one feature against two thresholds: 98% of pairs hit two different trees, so
  both lanes were extracted and merged separately; slower than scalar once constants were cheap.
- Scalar carry chains (`fcmp; adc` inline asm on ARM64, `addcarry` with `next_up` thresholds on
  x86): fast but capped by the flag-compare port, and the x86 form is wrong under DAZ.
- Binarize-then-byte-compare: only wins with shared thresholds, which real models lack.
- Small code model for MCJIT, backend `Default`, extra IR passes, host CPU strings: no gain or
  unsafe with the default memory manager.
- Chunks of 16 or more trees: faster by a few percent but spills reappear.

## Open

- Emission-order experiments: issue a chunk's leaf loads after all its compares; reuse features
  already loaded within a chunk (bounded at 10 to 20% fewer loads on NEON, and it relieves load
  ports on x86).
- Four sums, only once a kernel is fast enough for the addition chain to show.
- Native x86-64 timing of the SSE2 and AVX2 kernels; Rosetta cannot provide it.
- A representative user model: real depth range, NaN modes, feature counts, and whether inputs
  arrive hot or cold decide what the deployment number is.
