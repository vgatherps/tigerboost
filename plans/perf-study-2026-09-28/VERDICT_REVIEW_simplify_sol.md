# Findings

1. minor — `README.md:61`: The plan is described as having “one action per tree,” but every depth-zero tree has an `Admit`/`Retire` pair. A reader using the public `Action` and `Plan::actions` API will expect the wrong action count and shape for constant forests.

2. minor — `README.md:79`: The continuation signature is said to carry only the row pointer, constants pointer, and sums. ARM64 also carries `chunk_base` as a third pointer (`src/jit.rs:265-268`, `src/jit.rs:330-346`, `src/jit.rs:402-405`). The text hides a target-specific parameter that exists to keep threshold offsets reachable.

3. minor — `README.md:125-136`: The cited tracked study does not support the labels as written. Its 792 ns cold result is for four trees per chunk (`plans/perf-study-2026-09-28/RESULTS.md:49-54`), while README assigns it to eight. Its 2253 ARM64, 2332 SSE2, and 1486 AVX2 instruction counts are also four-tree results (`RESULTS.md:63-65`, `RESULTS.md:86-90`), but README does not name that chunk size after presenting the eight-tree default. Readers cannot reproduce the claimed configurations from the cited record.

4. minor — `PLAN.md:56`: “Chunks of 16 or more trees: faster by a few percent” overstates the recorded sweep. Sixteen was faster than eight, but 32 was slower at both 100 and 300 trees; 64 was only slightly faster (`plans/perf-study-2026-09-28/RESULTS.md:42-47`). This gives the wrong reason for rejecting every size above eight.

# Verdict

Accept with fixes. No blocker or major finding exists in the deletion, plan semantics, tests, or emitter. `plan` clamps sums correctly for empty and small forests; chunks keep whole trees; constant trees retain `Admit`/`Retire`; replay computes decisions from row values and original thresholds; differential leaves use disjoint exact fields through bit 47. Surviving bounds, alignment, `dereferenceable`, `nuw`, ARM64 `chunk_base`, SSE2/NEON, AVX2, and `EmitOptions` paths remain. Fresh `inspect_code` output matched all three SHA-256 values in `VERDICT.md`, and `tools/x86_check.sh special --avx2` passed all three variants on 256 rows. Debug and release all-target tests each passed 56 tests, the README doctest passed, and clippy passed with warnings denied. Fix the four documentation statements before treating README and PLAN as the durable design record.
