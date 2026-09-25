# Validation, performance, and completion

Apply this plan throughout implementation. It is not a final cleanup phase.

## Correctness evidence

- Small constructed forests: reference → scalar JIT → rolling scalar JIT → SIMD JIT.
- Compare per-tree leaf indices exactly; compare predictions with a documented reassociation-aware error bound. Include cancellation cases and all supported special-value behavior.
- Exercise malformed models and short input vectors before unsafe codegen entry. Verify executable/model-data ownership through the public API.
- Each scheduling/codegen change adds focused tests for its changed behavior; tests must not simply repeat the implementation's algorithm.

## Benchmark matrix

Keep these implementations available to the development harness:

1. Generic tree-order `f64` reference/interpreter.
2. Specialized scalar tree-order JIT.
3. Rolling scheduled scalar JIT.
4. Rolling scheduled SIMD JIT.

Use the same model and row corpus for paired comparisons. Include a representative user model when available. Generated forests cover low/high feature overlap, repeated exact predicates, multiple thresholds per feature, varying depths, many features, and large model/code footprints. Label synthetic results as synthetic.

Measure compilation separately from prediction. Use optimized builds; warm executable code and data for steady-state measurements, rotate through nonconstant input rows, consume outputs, and amortize harness overhead across repeated single-row calls. This remains single-row inference, not a batch-vectorized kernel.

Independent calls can overlap in hardware: their amortized cost is not single-row dependency latency. Also measure a chain where each prediction selects the next input row. Verify equivalent variants follow the same row trace; report the selection/driver cost separately without subtracting it. Keep independent-call cost as a separately labeled measurement, not the latency gate.

Randomize/interleave variant order and repeat runs to expose drift in the noisy environment. Report central latency and spread; collect individual-call tail estimates only when timing overhead permits. Distinguish hot-model and model-switching/cold-cache experiments rather than mixing them.

Record target, toolchain, LLVM/Inkwell versions and passes, model/row hashes or fixture identities, tree/depth/feature counts, live cap, accumulation count, mode, code bytes, compile time, and observed spill traffic. Require the before/after comparison to use the same workload and settings except the intended change.

## Keep complexity earned

Start with small experiments over active-tree cap, accumulator count, and scalar/SIMD mode. No runtime auto-tuner or per-CPU policy framework.

If latency regresses, determine whether the cause is longer live ranges, extraction overhead, more comparisons/reloads, leaf-table access, or instruction footprint before adding a fix. Inspect a minimal reproducer and relevant emitted code. Do not hide a regression inside aggregate speedup numbers.

Only introduce compile-time indexes, persistent caches, model partitioning, or more elaborate scheduling when a measured bottleneck justifies them. Preserve the smallest working implementation while testing the next change.

## Slice: final handoff

- Document constructing symmetric-tree models, compiling once, supplying immutable `f64` rows, and receiving predictions. Include ownership and supported-model boundaries.
- Provide reproducible test/benchmark commands using constructed forests; representative models may be supplied externally.
- Publish the variant comparison with any regressions and missing architecture coverage prominent.
- Check the implementation against every requirement in [the overall plan](../PLAN.md); do not equate a green scalar test with the whole goal.

Completion requires a usable model-to-JIT path, correct rolling reuse, explicit SIMD exploration, measured useful speed over generic evaluation, and evidence for the scheduling contribution beyond basic specialization. The unknown theoretical optimization ceiling prevents claiming an exact 80–90% fraction.
