use tigerboost::jit::compile;
use tigerboost::parsed_tree::{
    Discriminator, FeatureIndex, NanMode, RowLengthError, SingleTree, Trees,
};
use tigerboost::sort_policy::plan;

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
fn every_leaf_pattern_survives_shared_feature_lane_swaps() {
    for depth in 1..=10 {
        for shared_lane in 0..2 {
            let selections: Vec<_> = (0..depth)
                .map(|level| split(level, (2 * level) as f64 - 5.0))
                .collect();
            let shared: Vec<_> = selections
                .iter()
                .skip(shared_lane)
                .step_by(2)
                .cloned()
                .collect();
            let mut companion = indexed_tree(shared);
            for leaf in &mut companion.leafs {
                *leaf *= (1usize << depth) as f64;
            }
            let model = Trees::new(
                vec![indexed_tree(selections.clone()), companion],
                vec![NanMode::AsIs; depth],
                1.0,
                0.0,
            )
            .unwrap();
            let mut kernels = vec![compile(&model).unwrap()];
            if tigerboost::jit::EmitOptions::from_host().avx2 {
                kernels.push(
                    tigerboost::jit::compile_with(
                        &model,
                        tigerboost::jit::EmitOptions { avx2: true },
                    )
                    .unwrap(),
                );
            }
            for pattern in 0..1usize << depth {
                let row: Vec<_> = selections
                    .iter()
                    .enumerate()
                    .map(|(level, split)| split.threshold + f64::from(pattern & (1 << level) != 0))
                    .collect();
                let shared_pattern = (shared_lane..depth)
                    .step_by(2)
                    .enumerate()
                    .fold(0, |bits, (lane, level)| {
                        bits | (((pattern >> level) & 1) << lane)
                    });
                let expected = (pattern | (shared_pattern << depth)) as f64;
                assert_eq!(model.predict(&row), Ok(expected));
                for compiled in &kernels {
                    assert_eq!(
                        compiled.predict(&row),
                        Ok(expected),
                        "depth={depth} shared_lane={shared_lane} pattern={pattern}"
                    );
                }
            }
        }
    }
}

#[test]
fn matches_reference_on_special_values() {
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
                let compiled = compile(&model).unwrap();
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
        let compiled = compile(&model).unwrap();
        for first in values {
            for second in values {
                let row = [first, second];
                assert_eq!(compiled.predict(&row), model.predict(&row));
            }
        }
    }
}

#[test]
fn dropped_model_and_odd_depth_last_tree() {
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
    let plan = plan(&model);
    assert_eq!(plan.chunks().len(), 9);
    let compiled = compile(&model).unwrap();
    drop(plan);
    drop(model);
    for (row, expected) in rows.iter().zip(expected) {
        assert_eq!(compiled.predict(row), Ok(expected));
    }
}

#[test]
fn indexed_leaves_expose_original_level_bits_and_strict_comparisons() {
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
    let compiled = compile(&model).unwrap();
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

#[test]
fn repeated_features_nan_modes_and_scale_bias_match_reference() {
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
    let compiled = compile(&model).unwrap();
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

#[test]
fn empty_and_constant_forests_match_reference() {
    let empty = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    assert_eq!(
        compile(&empty)
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
        let result = compile(&model).unwrap().predict(&[]).unwrap();
        let expected = model.predict(&[]).unwrap();
        assert!(result == expected || result.is_nan() && expected.is_nan());
    }
}

#[test]
fn reordered_sums_allow_cancellation_error_scaled_by_absolute_leaves() {
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
    let compiled = compile(&model).unwrap();
    for row in [[-1.0], [1.0]] {
        assert!(
            (compiled.predict(&row).unwrap() - model.predict(&row).unwrap()).abs() <= tolerance
        );
    }
}

#[test]
fn row_check_and_reused_row_allocation_are_respected() {
    let model = Trees::new(
        vec![indexed_tree(vec![split(1, 0.0)])],
        vec![NanMode::AsIs; 2],
        1.0,
        0.0,
    )
    .unwrap();
    let compiled = compile(&model).unwrap();
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
        compile(&model).unwrap()
    };
    let mut moved = vec![compiled];
    {
        let other_model = Trees::new(
            vec![indexed_tree(vec![split(0, 0.0)])],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap();
        let other = compile(&other_model).unwrap();
        assert_eq!(other.predict(&[1.0]), Ok(1.0));
    }
    let compiled = moved.pop().unwrap();
    assert_eq!(compiled.predict(&[-1.0]), Ok(10.0));
    assert_eq!(compiled.predict(&[1.0]), Ok(20.0));
}

#[path = "../examples/support/mod.rs"]
mod support;

#[test]
fn avx2_clears_upper_halves_once_before_the_final_return() {
    use tigerboost::jit::{EmitOptions, emit_assembly_with};
    use tigerboost::sort_policy::Action;

    let mut models = vec![support::workload(support::Workload::parse("100x100").unwrap()).0];
    for tail in [vec![0], vec![1], vec![2], vec![0, 1, 2]] {
        let trees = [6; 8]
            .into_iter()
            .chain(tail)
            .map(|depth| indexed_tree((0..depth).map(|feature| split(feature, 0.0)).collect()))
            .collect();
        let model = Trees::new(trees, vec![NanMode::AsIs; 6], 1.0, 0.0).unwrap();
        let plan = plan(&model);
        assert_eq!(plan.chunks().len(), 2);
        for action in &plan.actions()[plan.chunks()[1].actions.clone()] {
            let (Action::Constant { tree, .. } | Action::EvaluateTree { tree, .. }) = action;
            assert!(model.trees()[tree.0].selections.len() <= 2);
        }
        models.push(model);
    }
    for model in models {
        for triple in ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"] {
            let assembly =
                emit_assembly_with(&plan(&model), triple, EmitOptions { avx2: true })
                    .unwrap();
            let lines: Vec<_> = assembly.lines().map(str::trim).collect();
            let clears: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(_, line)| **line == "vzeroupper")
                .map(|(index, _)| index)
                .collect();
            assert_eq!(clears.len(), 1, "{triple}, {} trees", model.trees().len());
            let final_ret = lines.iter().rposition(|line| *line == "retq").unwrap();
            assert!(clears[0] < final_ret);
            assert!(lines[clears[0] + 1..final_ret].iter().all(|line| {
                !line.contains("%ymm") && !line.starts_with("call") && !line.starts_with("jmp")
            }));
        }
    }
}

#[test]
fn avx2_narrow_and_constant_forests_leave_upper_halves_untouched() {
    use tigerboost::jit::{EmitOptions, emit_assembly_with};

    for depth in 0..=2 {
        for count in [2, 9, 17] {
            let model = Trees::new(
                (0..count)
                    .map(|_| indexed_tree((0..depth).map(|feature| split(feature, 0.0)).collect()))
                    .collect(),
                vec![NanMode::AsIs; 2],
                1.0,
                0.0,
            )
            .unwrap();
            for triple in ["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"] {
                let assembly =
                    emit_assembly_with(&plan(&model), triple, EmitOptions { avx2: true })
                        .unwrap();
                assert!(
                    !assembly.contains("%ymm"),
                    "{triple} depth={depth} count={count}"
                );
                assert!(
                    !assembly.contains("vzeroupper"),
                    "{triple} depth={depth} count={count}"
                );
            }
        }
    }
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn compiled_model_is_send_and_sync() {
    assert_send_sync::<tigerboost::jit::CompiledModel>();
}

// Each tree's leaf index lands in its own bit field, so any wrong leaf changes the sum.
fn bit_field_model() -> Trees {
    let feature_count = 4;
    let mut trees = Vec::new();
    let mut prior_bits = 0;
    for tree in 0..20usize {
        let depth = if tree == 0 { 6 } else { 1 + tree % 3 };
        let selections = (0..depth)
            .map(|level| split((tree + level) % feature_count, (level as f64) - 1.5))
            .collect();
        let leafs = (0..1u64 << depth).map(|index| (index << prior_bits) as f64).collect();
        trees.push(SingleTree { leafs, selections });
        prior_bits += depth;
    }
    assert!(prior_bits <= 53);
    let modes = (0..feature_count)
        .map(|feature| [NanMode::AsIs, NanMode::AsFalse, NanMode::AsTrue][feature % 3])
        .collect();
    Trees::new(trees, modes, 1.0, 0.0).unwrap()
}

fn bit_field_rows() -> Vec<Vec<f64>> {
    const VALUES: [f64; 6] = [-2.0, -1.5, -0.5, 0.5, 3.0, f64::NAN];
    (0..256usize)
        .map(|row| (0..4).map(|feature| VALUES[(row >> (2 * feature)) % VALUES.len()]).collect())
        .collect()
}

#[test]
fn one_model_shared_by_threads_predicting_at_once() {
    let model = bit_field_model();
    let rows = bit_field_rows();
    let expected: Vec<u64> = rows.iter().map(|row| model.predict(row).unwrap().to_bits()).collect();
    let compiled = compile(&model).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for (row, expected) in rows.iter().zip(&expected) {
                    assert_eq!(compiled.predict(row).unwrap().to_bits(), *expected);
                }
            });
        }
    });
}

#[test]
fn model_moved_to_another_thread_predicts_and_drops_there() {
    let model = bit_field_model();
    let rows = bit_field_rows();
    let expected: Vec<u64> = rows.iter().map(|row| model.predict(row).unwrap().to_bits()).collect();
    let compiled = compile(&model).unwrap();
    drop(model);
    std::thread::spawn(move || {
        for (row, expected) in rows.iter().zip(&expected) {
            assert_eq!(compiled.predict(row).unwrap().to_bits(), *expected);
        }
        drop(compiled);
    })
    .join()
    .unwrap();
}

fn one_split_model(leafs: [f64; 2]) -> Trees {
    Trees::new(
        vec![SingleTree {
            selections: vec![split(0, 0.0)],
            leafs: leafs.to_vec(),
        }],
        vec![NanMode::AsIs],
        1.0,
        0.0,
    )
    .unwrap()
}

#[test]
fn models_compiled_back_to_back_stay_independent_in_either_drop_order() {
    {
        let (source_a, source_b) = (one_split_model([10.0, 20.0]), one_split_model([0.0, 1.0]));
        let a = compile(&source_a).unwrap();
        let b = compile(&source_b).unwrap();
        drop((source_a, source_b));
        assert_eq!(a.predict(&[1.0]), Ok(20.0));
        assert_eq!(b.predict(&[1.0]), Ok(1.0));
        drop(a);
        assert_eq!(b.predict(&[1.0]), Ok(1.0));
    }
    {
        let (source_a, source_b) = (one_split_model([10.0, 20.0]), one_split_model([0.0, 1.0]));
        let a = compile(&source_a).unwrap();
        let b = compile(&source_b).unwrap();
        drop((source_a, source_b));
        assert_eq!(a.predict(&[1.0]), Ok(20.0));
        assert_eq!(b.predict(&[1.0]), Ok(1.0));
        drop(b);
        assert_eq!(a.predict(&[1.0]), Ok(20.0));
    }
}

#[test]
fn threads_compile_their_own_models_concurrently() {
    let handles: Vec<_> = (0..4)
        .map(|thread| {
            std::thread::spawn(move || {
                let model = one_split_model([thread as f64, 100.0 + thread as f64]);
                let compiled = compile(&model).unwrap();
                (thread, compiled.predict(&[1.0]).unwrap())
            })
        })
        .collect();
    for handle in handles {
        let (thread, prediction) = handle.join().unwrap();
        assert_eq!(prediction, 100.0 + thread as f64);
    }
}
