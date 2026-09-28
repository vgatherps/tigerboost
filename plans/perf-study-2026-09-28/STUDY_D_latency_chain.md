# Study D: latency chain, front-end and cache bounds, and whether the benchmark measures the use case

Read `STUDY_CONTEXT.md` first. Then this.

## Question in one sentence

For one call of a 100- to 300-tree kernel, which bound wins on each target (instruction
throughput, dependency-chain latency, instruction fetch, or data cache), what changes once the
instruction count drops two to three times, and does the current benchmark measure the
maintainer's real use case (one row at a time, interleaved with other work)?

## Read list

Primary anchors (read these; reading anything else in the repo to understand code is fine):

- `examples/scalar_bench.rs` whole file (dependent-chain driver `next_row`, `measure`,
  variant rotation), `examples/support/mod.rs` and `examples/support/distinct.rs`
  (workload shapes; 256 rows of 100 f64).
- `src/jit.rs` lines 22-56 (`CompiledModel::predict`, the row length check, the `JitFunction`
  call), 356-415 (leaf lookup, round-robin accumulation into `sums`, the final `AddSums` and
  scale/bias), 421-442 (continuation transfer).
- `src/sort_policy.rs` lines 143-174 (`sum_count` and the `AddSums` tail) and 478-484
  (`retire`: round-robin sum assignment).
- `plans/implementation.md` lines 72-82 (accumulator sweep under the OLD single-function
  backend: extra accumulators regressed; that backend spilled, so this evidence is stale),
  143-159 and 161-194 (current measurements, trace sizes 35/13/3 distinct rows).
- `captures/100x100/aarch64-apple-darwin-scheduled-12gp-9fp-2sums.s` and the x86-64 file, for
  code bytes (`.section`/size directives or `llvm-mc` + `llvm-objdump` to get exact bytes)
  and the dependency shape of one tree (compare chain, selector OR chain, leaf load, fadd).

## Questions

D1. Bound model for the current kernel. For ARM64 (Apple M4 P-core) and a representative
x86-64 core (pick one you can cite numbers for, e.g. Zen 4/5 or Golden Cove; state the
source), estimate for 100 and 300 depth-6 trees:
  - throughput bound: instructions / sustainable IPC;
  - latency bound: the longest dependency chain. Per tree: feature load -> fcmp -> cset ->
    or chain (6 deep as emitted; check whether LLVM reassociates it) -> leaf address ->
    leaf load -> fadd into a round-robin sum. Across trees: the sum chain of length
    trees / sum_count, plus the final `AddSums`, times fadd latency;
  - front-end bound: code bytes / fetch bandwidth, given the kernel is straight-line and
    loop-free so a call streams the whole kernel through decode (ARM64: 4 bytes per
    instruction, about 18 KB at 100 trees; x86-64: measure the bytes; `movabsq` is 10 bytes).
    Say whether the x86 uop cache can hold it (it cannot at these sizes; quantify) and what
    the legacy decode path sustains;
  - data cache: leaf tables are trees x 64 x 8 bytes (51 KB at 100 trees, 154 KB at 300);
    one line per tree is touched per call, which lines depends on the row. Row is 800 bytes.
    Give the warm and cold pictures against L1D/L2 sizes and latencies per target, and
    whether independent leaf loads across trees get enough memory-level parallelism to hide
    misses.
  State which bound is active today per target (ARM64 has a measurement: about 510 cycles
  for 4453 instructions; x86 has none) and reconcile with the measurement where one exists.

D2. After the instruction cut. Pieces A and C aim at roughly 1500 to 2500 instructions for
100 trees. Recompute the bounds. Which bound becomes active, per target? In particular
(H4): with two accumulators the sum chain is 50 fadds at 100 trees and 150 at 300; give the
cycles at each target's fadd latency and say at what accumulator count the chain stops
mattering. Recommend an accumulator count and reduction shape (round-robin scalar sums,
pairwise tree reduction, or two-lane vector sums built with one `ins`/`vunpcklpd` per pair)
with its instruction cost. Note that more sums cost registers only, since `preserve_nonecc`
frees them; the old regression at 4 sums (`plans/implementation.md:74-80`) was measured
under the spilling backend.

D3. Continuation and wrapper latency. Each `musttail` transfer is a direct branch with
argument shuffles; the wrapper does a full save frame (160 bytes on ARM64) and a call. Estimate
the cycles per call for 38 boundaries plus the wrapper, per target (branch cost, any
store-forwarding, the return). Is this in the noise at 500 cycles, and at 200 cycles?

D4. Does the benchmark measure the use case? The maintainer's scenario is one row at a time,
each call issued when a new feature set arrives, with other work between calls (so code and
leaf tables may be partly evicted from L1 and the branch/fetch state is cold). The current
harness runs 50,000 dependent calls over 256 rows, visiting 35/13/3 distinct rows, with
everything hot. Say concretely what that hides (code and leaf-table warmth; row warmth;
dependent chaining makes it latency-like, which is right) and propose a measurement that
approximates the real scenario: e.g. evict L1/L2 between calls with a pollution loop of known
size, or interleave calls with an unrelated working set, and report per-call latency
distribution (median, p99) rather than a mean over a hot loop. Give the exact harness change
(as a new example under `examples/`, described, not necessarily written) and what to expect.
Also check `CompiledModel::predict` overhead (length check, indirect call through
`JitFunction`) and say whether it matters at 100-cycle kernels.

D5. Optional, only if time remains: run `scalar_bench 100x100` once (it sweeps 1/2/4 sums at
several GP budgets) and report min and median per variant, labelled as taken under load, as a
sanity check of D2's chain estimate.

## Deliverable

Report at `probes/REPORT_D.md` (or the path your launch instructions name), per the report
contract in `STUDY_CONTEXT.md`. Section 3 must contain the bound table (rows: throughput,
chain latency, front-end, data cache; columns: ARM64 100/300 trees, x86-64 100/300 trees;
cells: cycles, for the current kernel and for the projected reduced kernel), the accumulator
recommendation with its instruction cost, and the measurement plan from D4.
