use inkwell::context::Context;
use std::num::NonZeroUsize;
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};
use tigerboost::sort_policy::{Action, ComparisonMode, Order, RegisterBudget, SortOptions, plan};

fn next(state: &mut u64, limit: usize) -> usize {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545_f491_4f6c_dd1d) % limit as u64) as usize
}

#[test]
fn seeded_forests_preserve_every_trees_leaf_index() {
    let mut carried_indices = false;
    let mut carried_features = false;
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
        let budget = RegisterBudget {
            general: NonZeroUsize::new([4, 6, 12, 20][case / 8 % 4]).unwrap(),
            floating: NonZeroUsize::new(8).unwrap(),
        };
        let accumulators = NonZeroUsize::new([1, 2, 4][case / 32]).unwrap();
        let context = Context::create();
        let expected: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
        let mut kernels = Vec::new();
        for options in [
            SortOptions {
                order: Order::TreeOrder,
                accumulators: NonZeroUsize::new(1).unwrap(),
                comparisons: ComparisonMode::Scalar,
            },
            SortOptions {
                order: Order::Scheduled(budget),
                accumulators,
                comparisons: ComparisonMode::Scalar,
            },
            SortOptions {
                order: Order::Scheduled(budget),
                accumulators,
                comparisons: ComparisonMode::Simd2,
            },
        ]
        .into_iter()
        .chain([1, 3, 8].map(|count| SortOptions {
            order: Order::TreeMajor {
                chunk_trees: NonZeroUsize::new(count).unwrap(),
            },
            accumulators: NonZeroUsize::new(2).unwrap(),
            comparisons: ComparisonMode::Scalar,
        })) {
            let plan = plan(&model, options).unwrap();
            for chunk in plan.chunks() {
                carried_indices |= chunk.incoming.indices.iter().any(Option::is_some);
                carried_features |= chunk.incoming.features.iter().any(Option::is_some);
            }
            kernels.push((options, compile(&context, &plan).unwrap()));
        }
        drop(model);
        for (options, compiled) in kernels {
            for (row_index, row) in rows.iter().enumerate() {
                assert_eq!(
                    compiled.predict(row),
                    Ok(expected[row_index]),
                    "seed={seed:#x} case={case} options={options:?} row_index={row_index} row={row:?}"
                );
            }
        }
    }
    assert!(carried_indices && carried_features);
}

#[test]
fn continuations_reuse_index_and_feature_slots_with_fresh_values() {
    let context = Context::create();
    let model = Trees::new(
        (0..8)
            .map(|tree| SingleTree {
                leafs: (0..64u64)
                    .map(|index| (index << (tree * 6)) as f64)
                    .collect(),
                selections: (0..6)
                    .map(|level| Discriminator {
                        feature_idx: FeatureIndex(level % 2),
                        threshold: (tree * 6 + level) as f64 / 8.0 - 3.0,
                    })
                    .collect(),
            })
            .collect(),
        vec![NanMode::AsFalse, NanMode::AsTrue],
        1.0,
        0.0,
    )
    .unwrap();
    let rows = [
        [f64::NAN; 2],
        [f64::NEG_INFINITY; 2],
        [f64::INFINITY; 2],
        [-2.0, 0.0],
        [0.5, -0.0],
    ];
    let expected: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
    let mut kernels = Vec::new();
    for comparisons in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
        let plan = plan(
            &model,
            SortOptions {
                order: Order::Scheduled(RegisterBudget {
                    general: NonZeroUsize::new(4).unwrap(),
                    floating: NonZeroUsize::new(4).unwrap(),
                }),
                comparisons,
                ..SortOptions::default()
            },
        )
        .unwrap();
        assert_eq!(plan.chunks().len(), 3);
        let first = &plan.chunks()[1].incoming;
        let second = &plan.chunks()[2].incoming;
        assert_ne!(first.indices[0].unwrap(), second.indices[0].unwrap());
        assert_ne!(first.features[0].unwrap(), second.features[0].unwrap());
        let loaded: Vec<_> = plan
            .actions()
            .iter()
            .filter_map(|a| match a {
                Action::Load { feature, .. } => Some(*feature),
                _ => None,
            })
            .collect();
        for slots in [first, second] {
            assert_eq!(slots.features.len(), 1);
            let id = slots.features[0].unwrap().0;
            assert!(
                loaded[..id].contains(&loaded[id]),
                "carried value must be a reload"
            );
        }
        kernels.push(compile(&context, &plan).unwrap());
    }
    drop(model);
    for compiled in kernels {
        for (row, expected) in rows.iter().zip(&expected) {
            assert_eq!(compiled.predict(row), Ok(*expected));
        }
    }
}
