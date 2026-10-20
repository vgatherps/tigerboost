use tigerboost::jit::{compile, compile_plan};
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
    const VALUES: [f64; 13] = [
        f64::NEG_INFINITY,
        -2.0,
        -1.0,
        -0.0,
        0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(1.0f64.to_bits() - 1),
        1.0,
        f64::from_bits(1.0f64.to_bits() + 1),
        2.0,
        f64::INFINITY,
        f64::NAN,
    ];
    let mut seen_depths = [[false; 10]; 4];
    for case in 0..96 {
        let seed = SEED ^ case as u64;
        let mut random = seed;
        let count_index = case % 4;
        let tree_count = [9, 17, 25, 40][count_index];
        let feature_count = 3 + next(&mut random, 6);
        let modes = (0..feature_count)
            .map(|feature| [NanMode::AsIs, NanMode::AsFalse, NanMode::AsTrue][(case + feature) % 3])
            .collect();
        let mut trees = Vec::new();
        let mut fields = Vec::new();
        let mut prior_bits = 0;
        for tree in 0..tree_count {
            // Reserve one bit for every later tree, including nonzero constants.
            let max_depth = 9.min(53 - prior_bits - (tree_count - tree - 1));
            let depth = if tree % 4 == 0 {
                0
            } else if tree == 1 {
                1 + case / 4 % 9
            } else {
                1 + next(&mut random, max_depth)
            };
            assert!(depth <= max_depth);
            seen_depths[count_index][depth] = true;
            let repeated_feature = next(&mut random, feature_count);
            let selections = (0..depth)
                .map(|level| Discriminator {
                    feature_idx: FeatureIndex(if level % 2 == 0 {
                        repeated_feature
                    } else {
                        next(&mut random, feature_count)
                    }),
                    threshold: VALUES[next(&mut random, VALUES.len())],
                })
                .collect();
            let leafs = (0..1u64 << depth)
                .map(|index| ((if depth == 0 { 1 } else { index }) << prior_bits) as f64)
                .collect();
            trees.push(SingleTree { leafs, selections });
            fields.push((prior_bits, depth.max(1), depth == 0));
            prior_bits += depth.max(1);
        }
        assert!(prior_bits <= 53);
        let model = Trees::new(trees, modes, 1.0, 0.0).unwrap();
        let rows: Vec<Vec<f64>> = (0..64)
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
        let expected: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
        let indices: Vec<_> = rows
            .iter()
            .map(|row| model.leaf_indices(row).unwrap())
            .collect();
        let inspected = plan(&model);
        assert_eq!(inspected.actions().len(), tree_count);
        assert_eq!(inspected.chunks().len(), tree_count.div_ceil(8));
        let kernels = [
            compile_plan(&inspected).unwrap(),
            compile(&model).unwrap(),
        ];
        drop(inspected);
        drop(model);
        for compiled in kernels {
            for (row_index, row) in rows.iter().enumerate() {
                let actual = compiled.predict(row).unwrap();
                assert_eq!(
                    actual, expected[row_index],
                    "seed={seed:#x} case={case} row_index={row_index} row={row:?}"
                );
                for (tree, &(shift, width, constant)) in fields.iter().enumerate() {
                    let selected = ((actual as u64) >> shift) & ((1 << width) - 1);
                    assert_eq!(
                        selected,
                        if constant {
                            1
                        } else {
                            indices[row_index][tree] as u64
                        }
                    );
                }
            }
        }
    }
    assert!(seen_depths.iter().flatten().all(|seen| *seen));
}
