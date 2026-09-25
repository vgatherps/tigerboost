# Explicit portable SIMD

Depends on a correct scheduled scalar JIT. Keep the same admission policy and model semantics so the effect of vectorization is measurable.

## Slice: two-lane comparison packs

- Emit LLVM `<2 x double>` operations: broadcast one feature, compare against two distinct threshold constants, extract the two result bits, and update the declared tree/level consumers.
- Deduplicate identical predicates before packing. Each lane may fan out to several consumers.
- Construct threshold vectors while compiling and store constants with valid lifetime/alignment. Do not build threshold arrays during prediction.
- Use scalar evaluation for a remaining single predicate. Do not fill an unused lane by reading past any array.
- Consume the result before scheduling another pack. Retire completed trees and refill after the pack's consumers are processed, using the same rolling rules.
- Retain the scalar mode as a correctness/performance control. No CPU-specific pack selector or hand-written ISA sequence in the initial design.

Inspect the pinned LLVM x86-64 and ARM64 output for comparison and mask extraction. Do not assume equivalent instruction costs or a speedup from using only two `f64` lanes. Keep comparison semantics exact; arithmetic reassociation permission does not apply to split decisions.

## Dedicated checks

- Exercise all four two-lane comparison outcomes; verify lane-to-consumer mapping explicitly.
- Equality, signed zero, infinities, missing-value behavior, and odd predicate counts must match scalar execution.
- Cover repeated predicates, consumers at different tree levels, and two tree completions from one pack.
- Execute randomized differential tests against the reference and scheduled scalar kernel.
- Cross-compile representative kernels for both backends and inspect their comparison/extraction/leaf-update sequences. Execute on each architecture when hardware is available; report any execution gap.

## Performance gate

Measure complete prediction calls, including broadcast, constant loading, mask extraction, bit distribution, leaf access, and accumulation. Compare scalar and SIMD with the same active-tree cap and accumulator count. Record added temporaries and spills.

Adopt SIMD where the supported workload shows a repeatable benefit. If it loses, preserve the evidence and examine pack formation or extraction before adding complexity. Do not report vectorized comparison count as proof of faster prediction. A SIMD regression is not permission to silently drop the requested exploration.
