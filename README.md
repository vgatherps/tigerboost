# tigerboost

A model-specialized LLVM JIT for single-row prediction over numeric symmetric
trees. Features, thresholds, leaves, and arithmetic are `f64`; output is one raw
scalar score. Models are constructed in memory. JSON/CBM loading, categorical
features, and probability transforms are not implemented.

## Use

```rust
use inkwell::context::Context;
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};
use tigerboost::sort_policy::plan;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Trees::new(
        vec![SingleTree {
            selections: vec![Discriminator {
                feature_idx: FeatureIndex(0),
                threshold: 1.0,
            }],
            leafs: vec![2.0, 5.0],
        }],
        vec![NanMode::AsIs],
        2.0,  // scale
        -1.0, // bias
    )?;
    let context = Context::create();
    let plan = plan(&model);
    let predictor = compile(&context, &plan)?;
    drop(plan);
    drop(model);
    assert_eq!(predictor.predict(&[1.0])?, 3.0);
    assert_eq!(predictor.predict(&[2.0])?, 9.0);
    Ok(())
}
```

`Trees::new` validates leaf counts and feature indices. `Trees::predict` is the
independent tree-order reference. Compiled models own their code and copied
constants, so the source `Trees` may be dropped; the LLVM `Context` must outlive
the compiled model. `predict` checks that the row covers every declared feature.
Rows stay read-only during each call and may change between calls. Prediction
writes neither caller memory nor model data.

At original tree level `k`, a strict `feature > threshold` decision sets bit `k`
of the leaf index. Each feature's `NanMode` applies before comparison: `AsIs`
leaves NaN unchanged (ordered comparisons are false), `AsFalse` replaces it with
negative infinity, and `AsTrue` replaces it with positive infinity. `AsTrue` is
not an unconditional true bit: positive infinity is not greater than itself.
A NaN threshold also compares false.

The score is the sum of selected leaves, scaled and then biased. Depth-zero
trees are constants; an empty forest returns bias without applying scale.
Arithmetic may reassociate: bitwise agreement and matching overflow behavior
with the reference are not promised. Comparisons retain NaN/infinity semantics.

## How a prediction runs

`sort_policy::plan(&model)` produces a plan without LLVM. Trees use one fixed
shared-feature order: each chunk starts with the tree whose features occur most
often in the forest, then adds trees sharing the most features with that chunk.
Ties use original tree order; constant trees follow the trees with splits.
Each tree has one `EvaluateTree` or `Constant` action. Chunks hold eight whole
trees, with selected leaves accumulated round-robin into two sums (clamped to
the tree count, with one sum for an empty forest). The sums are added at the end.
There are no planning options.

`jit::compile(&context, &plan)` emits that plan. All thresholds and all leaf
tables live in one private constant block that every function reaches through a
pointer argument, so inside a function each threshold is one load and each leaf
lookup is one indexed load (on x86-64 the leaf load folds into the add). A tree of
depth `d` is evaluated as `ceil(d/2)` two-lane vector comparisons: its feature
values are loaded into lane pairs, compared against the tree's thresholds (stored
consecutively), and the lane masks are packed with shift-left-insert into two
signed lane values whose sum addresses a leaf table that was permuted at compile
time to match. Odd depths get a constant-false padding level. The permutation is
defined by simulating that lane arithmetic for every bit pattern, and checked to
be a bijection when the model is compiled.

Each chunk is a separate `noinline preserve_nonecc` function that transfers only
the row pointer, the constants pointer (on ARM64 also a per-chunk threshold
base so paired loads stay within `ldp` range), and the running sums to the next chunk
with a `musttail` call; a C `predict` wrapper calls the first one. The split
exists to stop LLVM from hoisting loads across many trees and spilling; eight
trees per chunk is the largest spill-free size measured on ARM64.

`jit::compile_with(&context, &plan, EmitOptions { avx2: true })` selects an
x86-64 emission that compares four lanes then two per tree with the threshold
vector as a memory operand and reads the level bits back with `vmovmskpd`. It
sets `x86-64-v3` on the emitted functions. `compile_with` returns an error if
AVX2 is requested on a host without it. `EmitOptions::from_host()` enables it
when LLVM reports `+avx2`; `compile` uses the default two-lane emission.

## Build and checks

Requires LLVM 21 and Inkwell 0.9.0 (pinned in Cargo.toml). With the existing
Homebrew LLVM installation used here on ARM64:

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo build --offline --release --examples
cargo test --offline --all-targets
cargo test --offline --release --all-targets
RUSTDOCFLAGS="-C link-arg=/opt/homebrew/lib/libzstd.dylib" cargo test --offline --doc
cargo clippy --offline --all-targets -- -D warnings
tools/x86_check.sh special
tools/x86_check.sh special --avx2
tools/x86_check.sh depths --avx2
tools/x86_check.sh depths2
```

The macOS ARM64 Cargo configuration links `/opt/homebrew/lib` for zstd;
rustdoc does not inherit that setting.

`tools/x86_check.sh <workload> [--avx2]` emits the kernel (or its AVX2 twin) for
`x86_64-apple-darwin`, links it with a C driver, runs it under Rosetta 2
(`arch -x86_64`) and compares every prediction bit for bit with the reference.
Rosetta executes the code but its timing says nothing about an x86 CPU; no x86
performance has been measured.

## Measurements

Apple M4 Pro, LLVM 21.1.8, generic target, serial runs. Hot numbers are
`examples/scalar_bench.rs` medians of five rounds of 50,000 dependent calls, ns
per call including about 5 ns of driver. Cold numbers are
`examples/cold_bench.rs` per-call p50 over shuffled rows after 64 MiB of unrelated
read-modify-write traffic. Both workloads are synthetic depth-6 forests with
random feature reuse and globally distinct thresholds.

| 100 trees, 100 features | hot median ns | cold p50 ns |
|---|---:|---:|
| scheduled scalar kernel before this work | 121 | 791 |
| previous model-order kernel | 87 | 792 (measured at 4 per chunk) |
| current shared-feature kernel | 85 | — |

| 300 trees, 100 features | hot median ns |
|---|---:|
| scheduled scalar kernel before this work | ~345 |
| previous model-order kernel | 237 |
| current shared-feature kernel | 228 |

The current kernel's recorded hot medians are 85.3 / 228.2 ns at 100 / 300
trees. Measurements and rejected experiments are in
[RESULTS.md](plans/perf-study-2026-09-28/RESULTS.md).

Commands:

```sh
cargo run --offline --release --example scalar_bench -- 100x100
cargo run --offline --release --example cold_bench -- 100x100 --samples 5000
cargo run --offline --release --example inspect_code -- /tmp/tigerboost-code 100x100
```

Every example accepts `<trees>x<features>` (for example `100x100`, `300x100`,
`500x100`, `500x1000`), `special`, `depths`, or `depths2`. Generic workloads
have depth 6 and allow 1–10,922 trees with a positive feature count. `special`
covers NaN modes and special values; `depths` and `depths2` cover depths 1–10.
`scalar_bench` times the reference, the kernel, and its AVX2 twin on x86 hosts
with AVX2 support, plus the driver floor. `cold_bench` accepts `--samples` and
`--out`; its CSV columns are `condition,ns`.

Inspection requires a fresh output directory. It uses `emit_assembly` for the
default ARM64/x86-64 assembly and `emit_assembly_with` for the x86 AVX2 twin,
with plan dumps alongside each file. These are separate target emissions,
not MCJIT's executable bytes.
