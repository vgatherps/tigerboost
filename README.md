# tigerboost

A model-specialized LLVM JIT for single-row prediction over numeric symmetric
trees. Features, thresholds, leaves, and arithmetic are `f64`; output is one raw
scalar score. Models are constructed in memory. JSON/CBM loading, categorical
features, and probability transforms are not implemented.

## Use

```rust
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

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
    let predictor = compile(&model)?;
    drop(model);
    assert_eq!(predictor.predict(&[1.0])?, 3.0);
    assert_eq!(predictor.predict(&[2.0])?, 9.0);
    Ok(())
}
```

`Trees::new` validates leaf counts and feature indices. `Trees::predict` is the
independent tree-order reference. `predict` checks that the row covers every
declared feature. Rows stay read-only during each call and may change between
calls; prediction writes neither caller memory nor model data.

A compiled model owns its LLVM context, code and copied constants, so the source
`Trees` and `Plan` may be dropped. `CompiledModel` is `Send` and `Sync`: one model
can serve several threads calling `predict` at once, or move to another thread.
Compiling and emitting assembly take turns behind one process-wide lock, because
LLVM's target registry is not safe to use from two threads at once; `predict`
never waits on it.

At original tree level `k`, a strict `feature > threshold` decision sets bit `k`
of the leaf index. Each feature's `NanMode` applies before comparison: `AsIs`
leaves NaN unchanged (ordered comparisons are false), `AsFalse` replaces it with
negative infinity, and `AsTrue` replaces it with positive infinity. `AsTrue` is
not an unconditional true bit: positive infinity is not greater than itself.
A NaN threshold also compares false.
`AsIs` and `AsFalse` compile to the same code because an ordered `>` is false
for NaN and negative infinity alike; only `AsTrue` adds a NaN check per feature
load.

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

`jit::compile(&model)` plans internally and emits the kernel.
`jit::compile_plan(&plan)` compiles an inspected plan. One private constant
block holds, per chunk, the thresholds in target lane order, zero-padded to a
multiple of eight entries, then that chunk's leaf blocks in tree visit order.
Functions reach this block through a pointer argument, so each threshold is one load and each leaf
lookup is one indexed load (on x86-64 the leaf load folds into the add). A tree of
depth `d` is evaluated as `ceil(d/2)` two-lane vector comparisons: its feature
values are loaded into lane pairs, compared against the tree's thresholds (stored
consecutively), and the lane masks are packed with shift-left-insert on ARM64
or addition of shifted masks on generic targets into two
signed lane values whose sum addresses a leaf table that was permuted at compile
time to match. Odd depths get a constant-false padding level. The permutation is
defined by simulating that lane arithmetic for reachable bit patterns, with the
padding bit always false, and checked to be a bijection onto the original leaf
count when the model is compiled. Within each complete pair, exactly one shared
feature goes to lane 1 on ARM64 and lane 0 on x86. Both-shared and neither-shared
pairs keep their order; padding is never swapped. Thresholds follow their features.

Each chunk is a separate `noinline preserve_nonecc` function that transfers only
the row pointer, the chunk base (a pointer into the constant block that each hand-off
advances to the next chunk's thresholds), and the running sums to the next chunk
with a `musttail` call; a C `predict` wrapper calls the first one. The split
exists to stop LLVM from hoisting loads across many trees and spilling; eight
trees per chunk is the largest spill-free size measured on ARM64. For an empty
forest, the C wrapper returns the bias directly without a continuation.

`jit::compile_with(&model, EmitOptions { avx2: true })` selects an x86-64
emission that compares a tree's predicates four lanes at a time (a final
two-lane group covers a padded depth that is not a multiple of four), takes the
threshold vector as a memory operand, and reads the lane bits back with `vmovmskpd`. Its leaf
blocks are permuted when pairs are swapped; without swaps they retain original order. It
sets `x86-64-v3` on the emitted functions. `compile_with` returns an error if
AVX2 is requested and the host lacks any required x86-64-v3 feature. `EmitOptions::from_host()` enables
it when LLVM reports all of `+avx2`, `+bmi1`, `+bmi2`, `+fma`, `+lzcnt`, `+movbe`,
and `+f16c`; `compile` uses the default two-lane emission. Internal continuations
skip `vzeroupper`; the final continuation clears the upper halves before returning
when any chunk used a four-lane compare, and a forest with no four-lane compare emits
no clear.

## Build and checks

Requires Rust, GNU make, LLVM 21 (found through `llvm-config` or
`LLVM_SYS_211_PREFIX`), and a C compiler. The x86 check also needs a way to run
x86-64 code:

| Host | LLVM | x86 check |
|---|---|---|
| macOS | Homebrew `llvm@21` | Rosetta 2 (`arch -x86_64`), `clang` |
| Linux x86-64 | `llvm-config-21` or `llvm-config` 21.x | `cc` only |
| Linux aarch64 | `llvm-config-21` or `llvm-config` 21.x | `gcc-x86-64-linux-gnu` and `qemu-user`; the check binary is linked `-static`, so qemu needs no sysroot |

```sh
make                                # build, test, check-x86
make test                           # cargo test (with doctests), --all-targets, release, clippy
make check-x86                      # x86_check on six workloads; fails without a runner
make bench                          # scalar_bench 100x100 and 300x100, cold_bench 100x100
make inspect OUT=/tmp/tb-code       # inspect_code 100x100 into a fresh directory
make CARGO_FLAGS=--offline test     # extra cargo flags
```

`LLVM_SYS_211_PREFIX` from the environment or the make command line wins.
Otherwise `tools/llvm-prefix.sh` asks `brew --prefix llvm@21` on macOS (the
Homebrew for this CPU first) or `llvm-config-21`/`llvm-config` on Linux, and
leaves the variable unset when neither answers, so llvm-sys searches `PATH`
itself.

Without make (one of the six x86 workloads shown):

```sh
export LLVM_SYS_211_PREFIX=/opt/homebrew/opt/llvm@21
cargo build --offline --release --examples
cargo test --offline
cargo test --offline --all-targets
cargo test --offline --release --all-targets
cargo clippy --offline --all-targets -- -D warnings
cargo run --offline --release --example x86_check -- special /tmp/tb-x86-special --avx2
```

Plain `cargo test` runs the doctests but not the examples' tests; `--all-targets`
runs those but not the doctests, so `make test` runs both. On macOS ARM64,
`.cargo/config.toml` adds `/opt/homebrew/lib` to the link path for Homebrew
LLVM's zstd. rustdoc does not take rustflags, and an Intel Homebrew zstd in
`/usr/local/lib` would come first on llvm-sys's search path, so the same file
passes rustdoc the arm64 `libzstd.dylib` by path.

`x86_check <workload> <fresh-output-dir> [--avx2]` refuses an existing directory,
emits the kernel (or its AVX2 twin) for `x86_64-apple-darwin` on macOS or
`x86_64-unknown-linux-gnu` on Linux, links it with a C driver, runs it (Rosetta 2,
natively, or under `qemu-x86_64`) and compares every prediction bit for bit with
the reference. The x86_check tests in `cargo test --all-targets` execute
`special`, `100x100`, `depths`, and `depths2` kernels this way. Without an
x86-64 runner they pass and write `skipped: <reason>` straight to stderr, past
the test harness's capture, while `make check-x86` fails. On Linux x86-64 the
AVX2 variants skip when the CPU lacks AVX2. Emulated timing says nothing about
an x86 CPU; no x86 performance has been measured.

## Measurements

Apple M4 Pro, LLVM 21.1.8, generic target, serial runs. The table has hot numbers:
`examples/scalar_bench.rs` medians of five rounds of 50,000 dependent calls, ns per call
including about 5 ns of driver. Cold numbers, recorded in
[RESULTS.md](plans/perf-study-2026-09-28/RESULTS.md), are `examples/cold_bench.rs` per-call
p50 over shuffled rows after 64 MiB of unrelated read-modify-write traffic; the row is copied
into a scratch buffer just before the timed call, so the traffic evicts code and constants,
not the row. Both workloads are synthetic depth-6 forests with random feature reuse,
globally distinct thresholds, and every feature `AsIs`.

| Shared-feature kernel revision / run | 100 trees, hot ns | 300 trees, hot ns |
|---|---:|---:|
| scalar-carry study based on `6cae781`, carry disabled | 85.3 | 228.2 |
| before layout, `131c251`, back-to-back layout run | 89.6–91.5 | 229.6–234.0 |
| layout and lane order, `91951d5`, same run | 89.3 | 232.4 |

The first row is historical; the layout rows were measured back to back on the M4.
Static layout counts are in [the layout report](plans/perf-study-2026-09-28/VERDICT_layout_astra.md);
the timing comparison, earlier cold measurements, and rejected experiments are in
[RESULTS.md](plans/perf-study-2026-09-28/RESULTS.md).

Commands (`make bench`, `make inspect` run the same examples):

```sh
cargo run --offline --release --example scalar_bench -- 100x100
cargo run --offline --release --example scalar_bench -- 300x100
cargo run --offline --release --example cold_bench -- 100x100 --samples 5000
cargo run --offline --release --example inspect_code -- /tmp/tigerboost-code 100x100
```

Every example accepts `<trees>x<features>` (for example `100x100`, `300x100`,
`500x100`, `500x1000`), `special`, `depths`, or `depths2`. Generic workloads
have depth 6 and allow 1–10,922 trees with a positive feature count. `special`
covers NaN modes and special values; `depths` and `depths2` cover depths 1–10.
`scalar_bench` times the reference, the kernel, and its AVX2 twin on x86 hosts
with the required x86-64-v3 features, plus the driver floor. `cold_bench` accepts
`--samples` and `--out`; its CSV columns are `condition,ns`.

Inspection requires a fresh output directory. It uses `emit_assembly` for the
default ARM64/x86-64 assembly and `emit_assembly_with` for the x86 AVX2 twin,
with one `plan.txt` containing both targets' thresholds in emitted lane order,
plus the shared chunks and actions. These are separate target emissions,
not MCJIT's executable bytes.
