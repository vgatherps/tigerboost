# Model-specialized scalar JIT

Depends on the model/reference contract and [LLVM integration](llvm-backend.md). A real-model adapter may follow separately; validated in-memory models are sufficient for this slice.

## Slice: executable baseline

- Compile one specialized prediction function per model. Encode feature offsets, thresholds, level bit positions, and leaf-table bases while compiling.
- Start in original tree order with scalar comparisons, index construction, leaf lookup, and accumulation. This establishes what specialization alone buys over generic metadata-driven evaluation.
- Keep model/leaf storage stable and owned for as long as compiled code can access it. The compiled model owns executable-memory lifetime. No pointer may outlive its allocation.
- Expose a small checked prediction entry point accepting an immutable feature slice. Validate input length before entering the kernel; isolate the unsafe call behind that boundary.
- Apply LLVM read-only memory annotations with proven input ranges and alignment. Use constant globals for owned immutable model tables; caller features are immutable only during each invocation. Prove bounded leaf indices before inbounds addressing.
- Keep the prediction kernel free of allocations, runtime scheduling, and writes to input/model data. Compiler-generated stack spills or ABI saves are a distinct implementation detail to measure.
- Apply scale/bias as required by the settled model contract. Omit probability transforms unless separately requested.

Tests: differential leaf indices and predictions against the independent reference; threshold edge cases; malformed input length; model ownership across moves; repeated calls with changing inputs; independent compiled models. No test relies on a dangling pointer being detected by chance.

Gate: runnable scalar JIT, all semantic tests passing, saved generated code, and initial latency/compile-size measurements.

## Slice: controlled accumulation

Implement a small fixed number of independent accumulators and combine them at return. Assign retired trees deterministically. Arithmetic ordering is explicitly ours; LLVM need not discover reassociation. Inspect optimized IR and assembly to check whether the compiler regroups independent chains and extends shared-value lifetimes.

Measure 1/2/4 accumulators on the same kernels before choosing a default. Check cancellation-heavy inputs with an absolute error bound scaled by the sum of absolute leaf contributions, rather than relative error alone near zero. Preserve exact leaf decisions.

Gate: a measured choice with code-size and spill effects recorded. Keep this baseline callable by the benchmark harness after adding the scheduler so optimization claims can isolate its contribution.
