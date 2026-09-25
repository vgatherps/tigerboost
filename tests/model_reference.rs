use tigerboost::parsed_tree::{
    Discriminator, FeatureIndex, ModelError, NanMode, RowLengthError, SingleTree, Trees,
};

fn split(feature: usize, threshold: f64) -> Discriminator {
    Discriminator {
        threshold,
        feature_idx: FeatureIndex(feature),
    }
}

fn indexed_tree(selections: Vec<Discriminator>) -> SingleTree {
    SingleTree {
        leafs: (0..1usize << selections.len())
            .map(|index| index as f64)
            .collect(),
        selections,
    }
}

fn model(tree: SingleTree, modes: Vec<NanMode>) -> Trees {
    Trees::new(vec![tree], modes, 1.0, 0.0).unwrap()
}

#[test]
fn depths_zero_through_four_keep_original_level_bits() {
    for depth in 0..=4 {
        let trees = model(
            indexed_tree((0..depth).map(|feature| split(feature, 0.0)).collect()),
            vec![NanMode::AsIs; depth],
        );
        for expected in 0..1usize << depth {
            let row: Vec<_> = (0..depth)
                .map(|feature| {
                    if expected & (1 << feature) == 0 {
                        -1.0
                    } else {
                        1.0
                    }
                })
                .collect();
            assert_eq!(trees.leaf_indices(&row).unwrap(), [expected]);
            assert_eq!(trees.predict(&row).unwrap(), expected as f64);
        }
    }
}

#[test]
fn leaf_count_is_power_of_two_not_depth_squared() {
    for (depth, actual, expected) in [(0, 0, 1), (1, 1, 2), (2, 3, 4), (3, 9, 8), (4, 15, 16)] {
        let error = Trees::new(
            vec![SingleTree {
                leafs: vec![0.0; actual],
                selections: vec![split(0, 0.0); depth],
            }],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ModelError::LeafCount {
                tree: 0,
                expected,
                actual
            }
        );
    }
}

#[test]
fn depth_limit_checks_leaf_bytes_before_shifting_or_indexing() {
    let first_unaddressable_depth = usize::BITS as usize - 4;
    for depth in [
        first_unaddressable_depth,
        usize::BITS as usize,
        usize::BITS as usize + 1,
    ] {
        let error = Trees::new(
            vec![SingleTree {
                leafs: vec![],
                selections: vec![split(0, 0.0); depth],
            }],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap_err();
        assert_eq!(error, ModelError::DepthTooLarge { tree: 0, depth });
    }
    let depth = first_unaddressable_depth - 1;
    let error = Trees::new(
        vec![SingleTree {
            leafs: vec![],
            selections: vec![split(0, 0.0); depth],
        }],
        vec![NanMode::AsIs],
        1.0,
        0.0,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ModelError::LeafCount {
            tree: 0,
            expected: 1 << depth,
            actual: 0
        }
    );
}

#[test]
fn invalid_feature_reports_original_tree_and_level() {
    for feature in [2, usize::MAX] {
        let error = Trees::new(
            vec![
                indexed_tree(vec![]),
                indexed_tree(vec![split(0, 0.0), split(feature, 1.0)]),
            ],
            vec![NanMode::AsIs; 2],
            1.0,
            0.0,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ModelError::FeatureOutOfRange {
                tree: 1,
                level: 1,
                feature: FeatureIndex(feature),
                feature_count: 2,
            }
        );
    }
}

#[test]
fn rows_are_checked_before_either_evaluation_path() {
    let trees = model(indexed_tree(vec![split(1, 0.0)]), vec![NanMode::AsIs; 2]);
    assert_eq!(
        trees.predict(&[1.0]),
        Err(RowLengthError {
            expected: 2,
            actual: 1
        })
    );
    assert_eq!(
        trees.leaf_indices(&[]),
        Err(RowLengthError {
            expected: 2,
            actual: 0
        })
    );
    assert_eq!(trees.predict(&[-1.0, 1.0, f64::NAN]).unwrap(), 1.0);
}

#[test]
fn duplicate_feature_and_predicate_keep_each_consumers_bit() {
    let trees = model(
        indexed_tree(vec![split(0, 1.0), split(0, -1.0), split(0, 1.0)]),
        vec![NanMode::AsIs],
    );
    for (value, index) in [(-2.0, 0), (-1.0, 0), (0.0, 2), (1.0, 2), (2.0, 7)] {
        assert_eq!(trees.leaf_indices(&[value]).unwrap(), [index]);
        assert_eq!(trees.predict(&[value]).unwrap(), index as f64);
    }
}

#[test]
fn strict_comparison_handles_adjacent_floats_zeros_and_infinities() {
    let one = model(indexed_tree(vec![split(0, 1.0)]), vec![NanMode::AsIs]);
    for (value, index) in [
        (f64::from_bits(1.0f64.to_bits() - 1), 0),
        (1.0, 0),
        (f64::from_bits(1.0f64.to_bits() + 1), 1),
    ] {
        assert_eq!(one.leaf_indices(&[value]).unwrap(), [index]);
    }
    for threshold in [-0.0, 0.0] {
        let zero = model(indexed_tree(vec![split(0, threshold)]), vec![NanMode::AsIs]);
        for (value, index) in [
            (-f64::from_bits(1), 0),
            (-0.0, 0),
            (0.0, 0),
            (f64::from_bits(1), 1),
        ] {
            assert_eq!(zero.leaf_indices(&[value]).unwrap(), [index]);
        }
    }
    let extremes = model(
        indexed_tree(vec![split(0, f64::NEG_INFINITY), split(0, f64::INFINITY)]),
        vec![NanMode::AsIs],
    );
    for (value, index) in [
        (f64::NEG_INFINITY, 0),
        (-f64::MAX, 1),
        (f64::MAX, 1),
        (f64::INFINITY, 1),
    ] {
        assert_eq!(extremes.leaf_indices(&[value]).unwrap(), [index]);
    }
}

#[test]
fn nan_modes_replace_values_before_comparison_including_infinite_borders() {
    for (mode, expected) in [
        (NanMode::AsIs, 0),
        (NanMode::AsFalse, 0),
        (NanMode::AsTrue, 3),
    ] {
        let trees = model(
            indexed_tree(vec![
                split(0, f64::NEG_INFINITY),
                split(0, 0.0),
                split(0, f64::INFINITY),
            ]),
            vec![mode],
        );
        for nan in [f64::NAN, -f64::NAN, f64::from_bits(0x7ff0_0000_0000_0001)] {
            assert_eq!(trees.leaf_indices(&[nan]).unwrap(), [expected]);
        }
        assert_eq!(trees.leaf_indices(&[1.0]).unwrap(), [3]);
        let nan_border = model(indexed_tree(vec![split(0, f64::NAN)]), vec![mode]);
        for value in [f64::NAN, f64::NEG_INFINITY, 0.0, f64::INFINITY] {
            assert_eq!(nan_border.leaf_indices(&[value]).unwrap(), [0]);
        }
    }
}

#[test]
fn modes_are_per_feature_and_scale_bias_apply_after_tree_order_sum() {
    let trees = Trees::new(
        vec![
            SingleTree {
                leafs: vec![10.0, 20.0, 30.0, 40.0],
                selections: vec![split(0, 0.0), split(1, 0.0)],
            },
            SingleTree {
                leafs: vec![-4.0],
                selections: vec![],
            },
            SingleTree {
                leafs: vec![2.0, 8.0],
                selections: vec![split(0, 0.0)],
            },
        ],
        vec![NanMode::AsTrue, NanMode::AsFalse],
        2.5,
        -3.0,
    )
    .unwrap();
    assert_eq!(
        trees.leaf_indices(&[f64::NAN, f64::NAN]).unwrap(),
        [1, 0, 1]
    );
    assert_eq!(trees.predict(&[f64::NAN, f64::NAN]).unwrap(), 57.0);
    assert_eq!(trees.trees().len(), 3);
    assert_eq!(trees.feature_modes(), [NanMode::AsTrue, NanMode::AsFalse]);
    assert_eq!(trees.scale(), 2.5);
    assert_eq!(trees.bias(), -3.0);
}

#[test]
fn empty_forest_returns_bias_even_with_nonfinite_scale() {
    for scale in [1.0, f64::INFINITY, f64::NAN] {
        for bias in [7.0, -0.0, f64::NEG_INFINITY] {
            let trees = Trees::new(vec![], vec![], scale, bias).unwrap();
            assert_eq!(trees.predict(&[]).unwrap().to_bits(), bias.to_bits());
            assert!(trees.leaf_indices(&[]).unwrap().is_empty());
        }
    }
}

#[test]
fn constant_tree_and_nonfinite_model_values_remain_f64() {
    let trees = Trees::new(
        vec![SingleTree {
            leafs: vec![4.0],
            selections: vec![],
        }],
        vec![],
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(trees.predict(&[]).unwrap(), 11.0);
    assert_eq!(trees.leaf_indices(&[]).unwrap(), [0]);
    let nonfinite = model(
        SingleTree {
            leafs: vec![f64::NAN, f64::INFINITY],
            selections: vec![split(0, 0.0)],
        },
        vec![NanMode::AsIs],
    );
    assert!(nonfinite.predict(&[-1.0]).unwrap().is_nan());
    assert_eq!(nonfinite.predict(&[1.0]).unwrap(), f64::INFINITY);
}
