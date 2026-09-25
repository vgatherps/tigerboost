# LLVM integration

User switched from Cranelift to LLVM, following `/Users/samschetterer/src/breadboard`. The model, rolling scheduler, explicit SIMD, and single-row latency goals are unchanged.

## Reference and initial settings

Verified breadboard's `rs/breadboard-jit/Cargo.toml`, lockfile, and `src/llvm_compiler.rs`: Inkwell 0.9.0, llvm-sys 211.0.1, MCJIT, `OptimizationLevel::Less`, and `function(early-cse)` for its no-inline-kernel path. Installed LLVM is 21.1.8, selected by `LLVM_SYS_211_PREFIX`.

Reuse that integration and initial pass setup. Do not import breadboard's graph ABI, external-call machinery, chunk splitting, or its preserve_none/FastISel workaround. Tigerboost emits its numeric operations directly. No Rust-bitcode harvesting is needed.

Use a generic target configuration; do not add per-CPU tuning. Breadboard stays read-only. Its performance claims and graph-specific policies are not evidence about CatBoost kernels.

## First chunk

A small development kernel establishes compilation, invocation, annotations, and ownership before production codegen. Dedicated tests cover varying every feature, changing rows between calls, short-input rejection, and independent compiled kernels. Verify LLVM IR before and after passes; capture both forms and target assembly. A separate target-machine assembly emission is not a byte-for-byte dump of MCJIT memory.

Implemented in `examples/llvm_smoke.rs`; parent-reviewed, with five dedicated tests passing in debug and release. Run `cargo test --offline --example llvm_smoke` and `cargo run --offline --release --example llvm_smoke`. Requires LLVM 21; select its installation through `LLVM_SYS_211_PREFIX` when it is not discoverable by llvm-sys.

The execution engine owns code and must die before its LLVM context. A scoped context is sufficient for the development example. The production compiled model must also own stable leaf storage; no borrowed function pointer may escape that owner's lifetime. No leaked contexts or speculative Send/Sync implementations.

## Memory and floating point

Prove row length and alignment in the checked Rust entry point before invoking generated code. Use valid parameter `readonly`, alignment, and dereferenceability annotations; use a read-only function memory effect where supported. Generated prediction code writes neither caller nor model memory.

Do not put `!invariant.load` on caller features: LLVM requires the value to stay unchanged wherever that address is dereferenceable, beyond one invocation. Rows may change between calls. JIT-owned immutable model tables can instead be constant globals. See the [LLVM 21 load contract](https://releases.llvm.org/21.1.0/docs/LangRef.html#load-instruction).

Explicitly choose accumulation order in our generated dataflow. Allow arithmetic reassociation where the user permits it, without `nnan` or `ninf` promises. Do not relax comparisons: equality and model-defined missing-value decisions must stay exact. Avoid unrelated fast-math flags for operations the kernel does not perform.

## Gate

Working LLVM integration and dedicated tests, reviewed unsafe boundaries, verified annotations, and inspected generated code. Then build the typed model/reference and scalar predictor. Measure actual model compilation and dependent-call latency as those become available; do not replace this with another broad synthetic compiler study.
