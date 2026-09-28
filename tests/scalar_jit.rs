use inkwell::context::Context;
use std::num::NonZeroUsize;
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{
    Discriminator, FeatureIndex, NanMode, RowLengthError, SingleTree, Trees,
};
use tigerboost::sort_policy::{Action, ComparisonMode, Order, RegisterBudget, SortOptions, plan};

fn split(feature: usize, threshold: f64) -> Discriminator {
    Discriminator {
        feature_idx: FeatureIndex(feature),
        threshold,
    }
}

fn indexed_tree(selections: Vec<Discriminator>) -> SingleTree {
    SingleTree {
        leafs: (0..1usize << selections.len()).map(|i| i as f64).collect(),
        selections,
    }
}

#[test]
fn tree_major_matches_reference_on_special_values() {
    let context = Context::create();
    let values = [
        f64::NAN,
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::INFINITY,
    ];
    let thresholds = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.5, 0.5];
    let options = SortOptions {
        order: Order::TreeMajor {
            chunk_trees: NonZeroUsize::new(4).unwrap(),
        },
        ..SortOptions::default()
    };
    for mode in [NanMode::AsIs, NanMode::AsFalse, NanMode::AsTrue] {
        for depth in 1..=8 {
            for offset in 0..thresholds.len() {
                let model = Trees::new(
                    vec![indexed_tree(
                        (0..depth)
                            .map(|level| {
                                split(level % 3, thresholds[(level + offset) % thresholds.len()])
                            })
                            .collect(),
                    )],
                    vec![mode; 3],
                    2.5,
                    -3.0,
                )
                .unwrap();
                let compiled = compile(&context, &plan(&model, options).unwrap()).unwrap();
                for first in values {
                    for second in values {
                        for third in values {
                            let row = [first, second, third];
                            assert_eq!(
                                compiled.predict(&row),
                                model.predict(&row),
                                "depth={depth} mode={mode:?} offset={offset} row={row:?}"
                            );
                        }
                    }
                }
            }
        }
        let model = Trees::new(
            vec![indexed_tree(vec![
                split(0, 0.0),
                split(1, -0.5),
                split(0, 0.0),
            ])],
            vec![mode; 2],
            1.0,
            0.0,
        )
        .unwrap();
        let compiled = compile(&context, &plan(&model, options).unwrap()).unwrap();
        for first in values {
            for second in values {
                let row = [first, second];
                assert_eq!(compiled.predict(&row), model.predict(&row));
            }
        }
    }
}

#[test]
fn tree_major_dropped_model_and_odd_depth_last_tree() {
    let context = Context::create();
    let model = Trees::new(
        (0..70)
            .map(|tree| {
                let depth = tree % 7 + 1;
                SingleTree {
                    selections: (0..depth)
                        .map(|level| split((tree + level) % 9, 0.0))
                        .collect(),
                    leafs: (0..1usize << depth)
                        .map(|leaf| (tree * 128 + leaf) as f64)
                        .collect(),
                }
            })
            .collect(),
        vec![NanMode::AsIs; 9],
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(model.trees()[69].selections.len(), 7);
    let rows = [[-1.0; 9], [1.0; 9]];
    assert!(
        model
            .leaf_indices(&rows[0])
            .unwrap()
            .iter()
            .all(|&index| index == 0)
    );
    assert!(
        model
            .leaf_indices(&rows[1])
            .unwrap()
            .iter()
            .zip(model.trees())
            .all(|(&index, tree)| index == tree.leafs.len() - 1)
    );
    let expected = rows.map(|row| model.predict(&row).unwrap());
    let plan = plan(
        &model,
        SortOptions {
            order: Order::TreeMajor {
                chunk_trees: NonZeroUsize::new(8).unwrap(),
            },
            ..SortOptions::default()
        },
    )
    .unwrap();
    assert_eq!(plan.chunks().len(), 9);
    let compiled = compile(&context, &plan).unwrap();
    drop(plan);
    drop(model);
    for (row, expected) in rows.iter().zip(expected) {
        assert_eq!(compiled.predict(row), Ok(expected));
    }
}

fn configurations() -> impl Iterator<Item = SortOptions> {
    std::iter::once(Order::TreeOrder)
        .chain([4, 6, 12, 20].map(|general| {
            Order::Scheduled(RegisterBudget {
                general: NonZeroUsize::new(general).unwrap(),
                floating: NonZeroUsize::new(12).unwrap(),
            })
        }))
        .flat_map(|order| {
            [1, 2, 4].into_iter().flat_map(move |count| {
                [ComparisonMode::Scalar, ComparisonMode::Simd2].map(move |comparisons| {
                    SortOptions {
                        order,
                        accumulators: NonZeroUsize::new(count).unwrap(),
                        comparisons,
                    }
                })
            })
        })
}

#[test]
fn paired_thresholds_preserve_both_lane_orders_and_odd_tail() {
    let context = Context::create();
    let options = SortOptions {
        order: Order::Scheduled(RegisterBudget {
            general: NonZeroUsize::new(6).unwrap(),
            floating: NonZeroUsize::new(12).unwrap(),
        }),
        comparisons: ComparisonMode::Simd2,
        ..SortOptions::default()
    };
    let mut outcomes = [false; 4];
    for thresholds in [[0.0, 1.0], [1.0, 0.0]] {
        for mode in [NanMode::AsIs, NanMode::AsFalse, NanMode::AsTrue] {
            let first = indexed_tree(vec![
                split(0, thresholds[0]),
                split(0, thresholds[1]),
                split(0, 2.0),
                split(0, f64::INFINITY),
                split(0, f64::NEG_INFINITY),
            ]);
            let mut second = first.clone();
            second.selections.rotate_left(1);
            for leaf in &mut second.leafs {
                *leaf *= 64.0;
            }
            let model = Trees::new(vec![first, second], vec![mode], 1.0, 0.0).unwrap();
            let plan = plan(&model, options).unwrap();
            let packs: Vec<_> = plan
                .actions()
                .iter()
                .filter_map(|action| {
                    if let Action::Compare { comparisons, .. } = action {
                        Some(comparisons)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(
                packs.iter().map(|pack| pack.len()).collect::<Vec<_>>(),
                [2, 2, 1]
            );
            assert_eq!(
                packs[0]
                    .iter()
                    .map(|comparison| plan.predicates()[comparison.predicate.0].threshold)
                    .collect::<Vec<_>>(),
                thresholds
            );
            for comparison in packs.iter().flat_map(|pack| pack.iter()) {
                assert_eq!(comparison.consumers.len(), 2);
                assert_ne!(comparison.consumers[0].level, comparison.consumers[1].level);
            }
            let compiled = compile(&context, &plan).unwrap();
            for value in [
                f64::NEG_INFINITY,
                -0.0,
                0.0,
                0.5,
                1.0,
                2.0,
                f64::INFINITY,
                f64::NAN,
            ] {
                let prediction = compiled.predict(&[value]).unwrap();
                assert_eq!(prediction, model.predict(&[value]).unwrap());
                outcomes[prediction as usize & 3] = true;
            }
        }
    }
    assert!(outcomes.into_iter().all(|seen| seen));
}

#[test]
fn indexed_leaves_expose_original_level_bits_and_strict_comparisons() {
    let context = Context::create();
    let model = Trees::new(
        vec![indexed_tree(vec![
            split(2, 0.0),
            split(0, 1.0),
            split(1, -1.0),
        ])],
        vec![NanMode::AsIs; 3],
        1.0,
        0.0,
    )
    .unwrap();
    for options in configurations() {
        let compiled = compile(&context, &plan(&model, options).unwrap()).unwrap();
        for index in 0..8 {
            let row = [
                if index & 2 != 0 { 2.0 } else { 1.0 },
                if index & 4 != 0 { 0.0 } else { -1.0 },
                if index & 1 != 0 { 1.0 } else { -0.0 },
            ];
            assert_eq!(compiled.predict(&row).unwrap(), index as f64);
        }
        for row in [
            [f64::NAN; 3],
            [f64::NEG_INFINITY; 3],
            [f64::INFINITY; 3],
            [
                f64::from_bits(1.0f64.to_bits() + 1),
                -1.0,
                f64::from_bits(1),
            ],
        ] {
            assert_eq!(compiled.predict(&row), model.predict(&row));
        }
    }
}

#[test]
fn repeated_features_nan_modes_and_scale_bias_match_reference() {
    let context = Context::create();
    let model = Trees::new(
        vec![
            indexed_tree(vec![
                split(0, f64::NEG_INFINITY),
                split(1, 0.0),
                split(0, f64::INFINITY),
            ]),
            indexed_tree(vec![split(0, -1.0), split(2, 0.0), split(0, -1.0)]),
            SingleTree {
                leafs: vec![-4.0],
                selections: vec![],
            },
        ],
        vec![NanMode::AsTrue, NanMode::AsFalse, NanMode::AsIs],
        2.5,
        -3.0,
    )
    .unwrap();
    for options in configurations() {
        let compiled = compile(&context, &plan(&model, options).unwrap()).unwrap();
        for row in [
            [f64::NAN; 3],
            [-f64::NAN, 1.0, 2.0],
            [f64::NEG_INFINITY, f64::INFINITY, 0.0],
            [f64::INFINITY, -0.0, 1.0],
            [1.0, -1.0, 3.0],
        ] {
            assert_eq!(compiled.predict(&row), model.predict(&row));
        }
    }
}

#[test]
fn empty_and_constant_forests_need_no_row_loads() {
    let context = Context::create();
    let empty = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    for options in configurations() {
        assert_eq!(
            compile(&context, &plan(&empty, options).unwrap())
                .unwrap()
                .predict(&[])
                .unwrap()
                .to_bits(),
            (-0.0f64).to_bits()
        );
        for leaves in [
            vec![1.5, -2.0],
            vec![f64::INFINITY],
            vec![f64::INFINITY, f64::NEG_INFINITY],
            vec![f64::NAN],
        ] {
            let model = Trees::new(
                leaves
                    .into_iter()
                    .map(|leaf| SingleTree {
                        leafs: vec![leaf],
                        selections: vec![],
                    })
                    .collect(),
                vec![],
                2.0,
                3.0,
            )
            .unwrap();
            let result = compile(&context, &plan(&model, options).unwrap())
                .unwrap()
                .predict(&[])
                .unwrap();
            let expected = model.predict(&[]).unwrap();
            assert!(result == expected || result.is_nan() && expected.is_nan());
        }
    }
}

#[test]
fn reordered_sums_allow_cancellation_error_scaled_by_absolute_leaves() {
    let context = Context::create();
    let values = [1e16, 1.0, -1e16, 3.0];
    let model = Trees::new(
        values
            .map(|value| SingleTree {
                leafs: vec![value, -value],
                selections: vec![split(0, 0.0)],
            })
            .into(),
        vec![NanMode::AsIs],
        1.0,
        0.0,
    )
    .unwrap();
    let tolerance =
        values.len() as f64 * f64::EPSILON * values.iter().map(|value| value.abs()).sum::<f64>();
    for options in configurations() {
        let compiled = compile(&context, &plan(&model, options).unwrap()).unwrap();
        for row in [[-1.0], [1.0]] {
            assert!(
                (compiled.predict(&row).unwrap() - model.predict(&row).unwrap()).abs() <= tolerance
            );
        }
    }
}

#[test]
fn row_check_and_reused_row_allocation_are_respected() {
    let context = Context::create();
    let model = Trees::new(
        vec![indexed_tree(vec![split(1, 0.0)])],
        vec![NanMode::AsIs; 2],
        1.0,
        0.0,
    )
    .unwrap();
    let compiled = compile(&context, &plan(&model, SortOptions::default()).unwrap()).unwrap();
    assert_eq!(
        compiled.predict(&[1.0]),
        Err(RowLengthError {
            expected: 2,
            actual: 1
        })
    );
    let mut row = [123.0, -1.0, 999.0];
    assert_eq!(compiled.predict(&row), Ok(0.0));
    row[1] = 1.0;
    assert_eq!(compiled.predict(&row), Ok(1.0));
    assert_eq!(row, [123.0, 1.0, 999.0]);
}

#[test]
fn moved_model_owns_leaf_tables_after_source_and_other_kernel_drop() {
    let context = Context::create();
    let compiled = {
        let model = Trees::new(
            vec![SingleTree {
                leafs: vec![10.0, 20.0],
                selections: vec![split(0, 0.0)],
            }],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap();
        compile(&context, &plan(&model, SortOptions::default()).unwrap()).unwrap()
    };
    let mut moved = vec![compiled];
    {
        let other_context = Context::create();
        let other_model = Trees::new(
            vec![indexed_tree(vec![split(0, 0.0)])],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap();
        let other = compile(
            &other_context,
            &plan(&other_model, SortOptions::default()).unwrap(),
        )
        .unwrap();
        assert_eq!(other.predict(&[1.0]), Ok(1.0));
    }
    let compiled = moved.pop().unwrap();
    assert_eq!(compiled.predict(&[-1.0]), Ok(10.0));
    assert_eq!(compiled.predict(&[1.0]), Ok(20.0));
}
