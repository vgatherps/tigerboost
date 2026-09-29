# CatBoost JIT plan

Status: one kernel, shared-feature tree order, eight trees per chunk, two sums. No planning
options remain. Chunks use `musttail` continuations carrying only pointers and sums. The default
emission compares two lanes; the x86 AVX2 twin compares four then two lanes. Recorded ARM64 hot
medians are 85 / 228 ns at 100 / 300 trees. x86-64 kernels match the reference under Rosetta 2;
no native x86 timing is recorded. The [README](README.md) documents the API;
[RESULTS.md](plans/perf-study-2026-09-28/RESULTS.md) records the measurements and experiments.

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

Validated model → shared-feature tree order → eight-tree LLVM continuations.

`compile(&context, &trees)` plans internally; `plan` remains public for inspection.
The plan has one `EvaluateTree` action per tree with splits and one `Constant` action per
constant tree. Each chunk starts with the highest feature-frequency score, then takes trees
sharing the most features with the chunk; ties use model order. Constant trees come last.
Odd depths get a constant-false `+inf` padding level. Leaves accumulate round-robin into two
sums, clamped to the tree count (one sum for an empty forest). Eight whole trees form a chunk.
The emitter performs the final reduction and scale/bias.
Only the row pointer, constants pointer, per-chunk threshold base on ARM64, and sums cross
chunk boundaries; `preserve_nonecc` passes them in registers.

The emitter builds one private constant block: thresholds one slot per predicate in first-use
order, then each tree's leaf block. A tree's thresholds are consecutive, so each lane pair or
lane quad is one vector load. Lane masks are packed with shift-left-insert on ARM64 (generic
shift/or elsewhere) into two signed lane values; the leaf block for that tree is permuted at
compile time over reachable patterns with the padding bit false. Each block has the original
leaf count, and the permutation is asserted to be a bijection.
Under `EmitOptions { avx2 }` on x86-64, lanes map to level bits directly through `vmovmskpd`, the
threshold vector folds into `vcmpgt_oqpd`, and the leaf block keeps its original order. Only the
last continuation clears the upper halves. Host execution requires all seven LLVM feature flags
listed in the README.

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
- Emission-order changes inside a chunk (leaf loads issued after all compares; reuse of a feature
  already loaded in the chunk): the emitted assembly is identical or within four instructions,
  because `early-cse` and the machine scheduler already do both. The remaining row loads are
  distinct features.
- Chunks of 16 trees: a few percent faster than 8 but the first stack spills appear; 32 and 64
  spill heavily and lose the gain. Eight is the largest spill-free size.
- Cross-boundary feature reuse through fences, volatile residency, scheduler changes, carried
  vectors, and scalar feature arguments: fewer row loads paid back in spills, register moves,
  lost threshold pairing, or packing. Scalar carry was neutral in model order and slower in
  shared-feature order. Rejected; see [RESULTS.md](plans/perf-study-2026-09-28/RESULTS.md).

## Open

- ARM64 leaf-offset layout study.
- Native x86-64 timing of the SSE2 and AVX2 kernels; Rosetta cannot provide it.
- A representative user model: real depth range, NaN modes, feature counts, and whether inputs
  arrive hot or cold decide what the deployment number is.
