use inkwell::context::Context;
use tigerboost::jit::compile;
use tigerboost::parsed_tree::{
    Discriminator, FeatureIndex, NanMode, RowLengthError, SingleTree, Trees,
};

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
    let compiled = compile(&context, &model).unwrap();
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
    let compiled = compile(&context, &model).unwrap();
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
fn empty_and_constant_forests_need_no_row_loads() {
    let context = Context::create();
    let empty = Trees::new(vec![], vec![], f64::NAN, -0.0).unwrap();
    assert_eq!(
        compile(&context, &empty)
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
        let result = compile(&context, &model).unwrap().predict(&[]).unwrap();
        let expected = model.predict(&[]).unwrap();
        assert!(result == expected || result.is_nan() && expected.is_nan());
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
    let compiled = compile(&context, &model).unwrap();
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
        compile(&context, &model).unwrap()
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
        let other = compile(&other_context, &other_model).unwrap();
        assert_eq!(other.predict(&[1.0]), Ok(1.0));
    }
    let compiled = moved.pop().unwrap();
    assert_eq!(compiled.predict(&[-1.0]), Ok(10.0));
    assert_eq!(compiled.predict(&[1.0]), Ok(20.0));
}
