# Cranelift behavior: establish the facts

Historical subplan. User switched to LLVM; no further Cranelift work is required. Completed evidence is in [the experiment report](cranelift-results.md).

## Question

Can repeated immutable loads become long-lived values that spill instead of being cheaply reread? If so, does that matter for this JIT's intended schedules?

## Source evidence, not measurements

Reviewed upstream sources on 2026-09-24; pin a released Cranelift version before implementing and recheck its behavior.

- [Pure-operation classification](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/codegen/src/inst_predicates.rs): ordinary loads with `readonly`, `notrap`, and `can_move` participate as pure operations in the current optimizer.
- [Optimization](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/codegen/src/egraph/mod.rs) and [placement](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/codegen/src/egraph/elaborate.rs): equivalent computations can be merged; pure operations are placed as consumers demand them and reused thereafter. Emission order alone is not a lifetime contract.
- [Rematerialization rules](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/codegen/src/opts/remat.isle): listed rules cover constants and some cheap arithmetic. They do not establish a general facility for repeating immutable memory loads instead of spilling.
- [Allocator interface](https://github.com/bytecodealliance/regalloc2/blob/main/src/lib.rs): operands, register constraints, and allocation moves are exposed; there is no general original-memory-load recipe in this interface.
- [Allocator design](https://github.com/bytecodealliance/regalloc2/blob/main/doc/ION.md): live ranges can split, and redundant spill/reload moves can be removed. These mechanisms mitigate pressure but do not prove that source-memory rereads replace necessary spills.

Inference: the proposed failure mechanism is plausible. Neither its occurrence nor its cost in Tigerboost has been measured. Read-only does not mean loading everything at entry or pinning it in registers.

## Slice: isolated experiment

Add a small development executable once implementation begins; no changes to model semantics required.

1. Pin compatible Cranelift crates and record compiler version, target triple, target features, optimization settings, and allocator settings.
2. Generate straight-line kernels from explicit load/use sequences. Each returns a scalar dependent on every tested input. No calls, user-memory stores, or constant-only fake work in the measured kernel.
3. Cover immediate reuse, distant reuse with intervening independent work, many features revisited in the same order, and many features revisited in reverse order. Sweep modest through high pressure, e.g. 4/16/64/128 feature values.
4. Compare identical programs under ordinary valid loads versus `readonly` loads; independently vary `can_move` where the pinned release supports it. Keep access validity and alignment facts identical. Compare `speed` against `none` as a diagnostic, not a predetermined production choice.
5. Save pre-optimization IR, post-optimization IR, disassembly, compilation time, code size, and latency. Inspect whether repeated reads merged and where loads moved.
6. Identify feature reads versus stack spill/reload accesses. Separate callee-save traffic and frame bookkeeping from pressure spills; stack-frame size alone is insufficient evidence.

Correctness: compare against a plain reference over multiple inputs; independently vary every feature to demonstrate it contributes. Enable Cranelift verification and the register-allocation checker in development runs. Avoid unstable assembly snapshots as the sole regression test.

## Gate and decision

Deliver a short result table plus an annotated small example if a problem appears. Native execution is required for timing; cross-target compilation supplies lowering evidence only.

- If annotations produce good code, keep normal speed optimization and proceed.
- If only extreme pressure spills, establish practical scheduling parameters using the model kernels next.
- If a small realistic schedule regresses, isolate the cause before changing production design. Reorder our dataflow/uses or adjust a documented compiler setting only when the experiment supports it.
- Do not preemptively disable useful optimization or add instruction-order barriers. A compiler patch is a separate, evidence-backed scope decision.

An inconclusive experiment must be reported as inconclusive. The later real-model measurements remain necessary even if this experiment passes.
