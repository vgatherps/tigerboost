# CatBoost JIT plan

Status: implementation approved. Subagents implement one reviewable chunk at a time; the parent reviews code, dedicated tests, and measurements before allowing the next chunk. Progress and review decisions are recorded in [implementation notes](plans/implementation.md).

## Goal and agreed constraints

Single-row prediction latency. Capture most useful model-specific optimization with a small implementation. The stated 80–90% is a design aim, not a measurable fraction of an unknown optimum.

- Numeric, symmetric trees; `f64` features, thresholds, leaves, and arithmetic.
- Arithmetic may reorder. Split decisions and leaf-index bits must remain correct.
- LLVM 21 backend through Inkwell/MCJIT, following breadboard's integration. This replaces the earlier Cranelift choice. Our JIT selects reuse, SIMD packs, admission, retirement, and accumulation order.
- One scheduling strategy for x86-64 and ARM64; no per-CPU instruction-cost model.
- Immutable feature/model reads during prediction; no writes to caller/model memory.
- Rolling bounded active-tree set. Admit replacements when slots become free; no whole-group completion barrier.

## Result to build

Validated model → compile-time schedule → model-specialized LLVM function.

Feature offsets, borders, destination bits, and leaf-table bases are known while compiling. Runtime executes selected loads, comparisons, bit operations, leaf lookups, and additions. It does not walk split metadata or run the scheduling policy.

Separate model predicates from their uses at tree levels. The same predicate may feed several bits. Distinct thresholds on one feature share its load. SIMD packs share one feature and contain explicit thresholds and consumers. Finish a tree as soon as its bits are complete; immediately accumulate its leaf and release its partial index.

Use a few independent sums to shorten the addition dependency chain. Treat their count, active-tree cap, and scalar/SIMD choice as a small measured policy space. Do not add a general optimizer, register allocator, runtime work queues, or a global table of all comparison results.

## Ordered subplans

| Subplan | Concrete result | Evidence needed before proceeding |
|---|---|---|
| [LLVM integration](plans/llvm-backend.md) | Breadboard-style LLVM integration with checked ownership and annotations | Executed kernel tests, verified IR, target assembly |
| [Model and reference](plans/model-and-reference.md) | Validated typed model; straightforward `f64` evaluator | A few hand-derived leaf-index and prediction tests |
| [Scalar JIT](plans/scalar-jit.md) | Working model-specialized scalar kernel | Differential correctness, memory ownership checks, baseline latency |
| [Rolling scheduler](plans/rolling-scheduler.md) | Explicit shared-load/comparison schedule with immediate refill | Schedule invariants, targeted reuse tests, measured improvement |
| [Portable SIMD](plans/portable-simd.md) | Explicit two-lane `f64` comparison packs | Scalar/SIMD agreement, both-target lowering, end-to-end measurements |
| [Performance and completion](plans/performance-and-completion.md) | Reproducible evidence and usable API | Representative models, all preceding gates, honest portability report |

LLVM integration and the model contract are independent pieces. Production codegen depends on both. Each subplan specifies smaller implementation slices; land and validate each before extending it. Tests and benchmark recording start with the first runnable slice, not at the end. The completed Cranelift experiments remain historical evidence, not an active implementation gate.

## Compiler uncertainty to resolve first

Read-only load elimination can merge distant reads into one long-lived value. The Cranelift probe demonstrated spilling and consumer-driven reordering; those results do not establish LLVM's behavior. Inspect LLVM output on the real model kernels before changing the schedule or pass setup. Logical schedule lifetime counts are not physical-register guarantees.

Keep valid read-only annotations. Do not manufacture side effects, volatile accesses, fake dependencies, or helper-call barriers to enforce source order. If measured code defeats the intended schedule, document the reproducer and choose the smallest justified change before continuing.

## Input and output scope

Build the symmetric-tree evaluator and JIT from the in-memory model first. Output is a scalar raw prediction including scale/bias. Empty forests return bias; depth-zero trees are constant leaves. JSON loading is deferred, not a prerequisite for codegen. Binary CBM loading and probability transforms are outside this scope.

A representative user model and input rows are needed for deployment-level performance claims. Small constructed forests suffice for evaluator/JIT correctness and initial measurements. Do not install Python/CatBoost or build a native compatibility test pipeline.

## Completion

- In-memory symmetric-tree models compile and predict through the documented API.
- Reference, scalar JIT, scheduled JIT, and SIMD agree on split decisions; finite sums agree within a documented error bound accommodating reassociation.
- Feature reuse and rolling retirement are observed in generated code and measured on realistic models.
- Compare generic evaluation, direct tree-order JIT, scheduled scalar JIT, and scheduled SIMD JIT with equivalent timing boundaries.
- Report latency, compilation cost, code size, spill behavior, and regressions. Distinguish successful cross-compilation from execution/performance on another CPU.
- No outstanding correctness failures or unexplained performance regressions on the agreed workload. A slower SIMD experiment is evidence, not a speedup claim.
