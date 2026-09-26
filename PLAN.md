# CatBoost JIT plan

Status: resource-aware sorting, bounded continuation emission, and full-width SIMD masks are reviewed and validated. [Current measurements](plans/implementation.md#bounded-continuation-kernels) show faster scalar prediction and stack-free ARM64 continuation bodies on the tested cases; SIMD remains opt-in and slower than scalar. Narrow traces, custom-budget spills, and untested x86 execution remain limitations. The [README](README.md) documents the API.

## Goal and agreed constraints

Single-row prediction latency. Capture most useful model-specific optimization with a small implementation. The stated 80–90% is a design aim, not a measurable fraction of an unknown optimum.

- Numeric, symmetric trees; `f64` features, thresholds, leaves, and arithmetic.
- Arithmetic may reorder. Split decisions and leaf-index bits must remain correct.
- LLVM 21 backend through Inkwell/MCJIT, following breadboard's integration. This replaces the earlier Cranelift choice. Our planner selects reuse, SIMD packs, admission, retirement, and accumulation order.
- One scheduling strategy for x86-64 and ARM64; no per-CPU instruction-cost model.
- Immutable feature/model reads during prediction; no writes to caller/model memory.
- Independent GP/FP budgets constrain live values and primitive peaks. Admit selectors only with feasible work; retire completed trees and release features under pressure, without a whole-group barrier.

## Implemented design

Validated model → compile-time schedule → model-specialized LLVM function.

The pure plan supplies chunks of at most 16 comparison tests and stable incoming value slots. Multi-chunk kernels transfer real live selectors/features/sums through matching `preserve_nonecc` signatures with `musttail`; one-chunk kernels keep a direct C entry. Default budgets are GP12/FP9 with two sums.

Feature offsets, borders, destination bits, and leaf-table bases are known while compiling. Runtime executes selected loads, comparisons, bit operations, leaf lookups, and additions. It does not walk split metadata or run the scheduling policy.

Separate model predicates from their uses at tree levels. The same predicate may feed several bits. Distinct thresholds on one feature share its load. SIMD packs share one feature and contain explicit thresholds and consumers. Finish a tree as soon as its bits are complete; immediately accumulate its leaf and release its partial index.

Use a few independent sums to shorten the addition dependency chain. Account for sums, resident feature values, live selectors and primitive scratch before admitting work; vary GP/FP budgets and scalar/SIMD choice through measurements. Do not add a general optimizer, register allocator, runtime work queues, or a global table of all comparison results.

## Subplan history

| Subplan | Concrete result | Evidence needed before proceeding |
|---|---|---|
| [LLVM integration](plans/llvm-backend.md) | Breadboard-style LLVM integration with checked ownership and annotations | Executed kernel tests, verified IR, target assembly |
| [Model and reference](plans/model-and-reference.md) | Validated typed model; straightforward `f64` evaluator | A few hand-derived leaf-index and prediction tests |
| [Scalar JIT](plans/scalar-jit.md) | Working model-specialized scalar kernel | Differential correctness, memory ownership checks, baseline latency |
| [Rolling scheduler](plans/rolling-scheduler.md) | Explicit shared-load/comparison schedule with immediate refill | Schedule invariants, targeted reuse tests, measured improvement |
| [Portable SIMD](plans/portable-simd.md) | Explicit two-lane `f64` comparison packs | Scalar/SIMD agreement, both-target lowering, end-to-end measurements |
| [Performance and completion](plans/performance-and-completion.md) | Reproducible evidence and usable API | Representative models, all preceding gates, honest portability report |

These subplans record the earlier implementation sequence, including the superseded active-tree cap. Cranelift experiments and cap-based measurements are historical evidence, not current scheduler validation.

## Observed compiler limitation

Read-only load elimination can merge distant reads into one long-lived value. Earlier cap-based LLVM output delayed some consumers and spilled. Logical lifetime counts are not physical-register guarantees; historical assembly findings and rejected backend experiments are in the implementation notes.

Keep valid read-only annotations. Do not manufacture side effects, volatile accesses, fake dependencies, or empty calls used only as barriers. The implemented continuations carry real state and bound cross-chunk reordering; LLVM still schedules within each chunk, and logical budgets do not guarantee spill-free custom configurations.

## Input and output scope

Build the symmetric-tree evaluator and JIT from the in-memory model first. Output is a scalar raw prediction including scale/bias. Empty forests return bias; depth-zero trees are constant leaves. JSON loading is deferred, not a prerequisite for codegen. Binary CBM loading and probability transforms are outside this scope.

A representative user model and input rows are needed for deployment-level performance claims. Small constructed forests suffice for evaluator/JIT correctness and initial measurements. Do not install Python/CatBoost or build a native compatibility test pipeline.

## Completion evidence and remaining boundaries

- In-memory models compile and predict through the documented, checked API; ownership tests cover dropped source models and retained kernels.
- Hand-derived cases and seeded forests compare reference, tree-order, scheduled scalar, and SIMD leaf decisions. Cancellation tests accommodate reordered sums; overflow/bitwise equivalence is not promised.
- Four historical synthetic workloads retain cap-based results, including an artificial repeated-threshold reuse case. Three reproducible distinct-threshold workloads now validate continuations and SIMD masks; all 256 rows agree before timing.
- Historical latency, compilation cost, code size, spills, disjoint slowdown and short traces remain recorded. Actual ARM64 captures show zero continuation-body stack traffic for the measured GP12/FP9 cases, not a universal physical-register bound.
- No representative user model or x86-64 execution evidence was supplied. Deployment-level performance and broader platform validation remain unestablished.
