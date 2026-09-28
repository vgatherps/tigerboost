use inkwell::context::Context;
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};
use tigerboost::sort_policy::plan;

fn next(state: &mut u64, limit: usize) -> usize {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545_f491_4f6c_dd1d) % limit as u64) as usize
}

#[test]
fn seeded_forests_preserve_every_trees_leaf_index() {
    const SEED: u64 = 0x4d59_5df4_d0f3_3173;
    const VALUES: [f64; 10] = [
        f64::NEG_INFINITY,
        -2.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        f64::INFINITY,
        f64::NAN,
    ];
    for case in 0..96 {
        let seed = SEED ^ case as u64;
        let mut random = seed;
        let feature_count = 1 + case % 8;
        let modes = (0..feature_count)
            .map(|feature| [NanMode::AsIs, NanMode::AsFalse, NanMode::AsTrue][(case + feature) % 3])
            .collect();
        let mut trees = Vec::new();
        let mut prior_depth = 0;
        for _ in 0..case % 9 {
            let depth = next(&mut random, 7);
            let selections = (0..depth)
                .map(|_| Discriminator {
                    feature_idx: FeatureIndex(next(&mut random, feature_count)),
                    threshold: VALUES[next(&mut random, VALUES.len())],
                })
                .collect();
            // Disjoint fields, at most 8 * 6 bits: every reordered sum is exact in f64.
            let leafs = (0..1u64 << depth)
                .map(|index| (index << prior_depth) as f64)
                .collect();
            trees.push(SingleTree { leafs, selections });
            prior_depth += depth;
        }
        assert!(prior_depth <= 48);
        let model = Trees::new(trees, modes, 1.0, 0.0).unwrap();
        let rows: Vec<Vec<f64>> = (0..20)
            .map(|row| {
                (0..feature_count)
                    .map(|_| {
                        VALUES[if row < VALUES.len() {
                            row
                        } else {
                            next(&mut random, VALUES.len())
                        }]
                    })
                    .collect()
            })
            .collect();
        let context = Context::create();
        let expected: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
        let compiled = compile(&context, &plan(&model)).unwrap();
        drop(model);
        for (row_index, row) in rows.iter().enumerate() {
            assert_eq!(
                compiled.predict(row),
                Ok(expected[row_index]),
                "seed={seed:#x} case={case} row_index={row_index} row={row:?}"
            );
        }
    }
}
