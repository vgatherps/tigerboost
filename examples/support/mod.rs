use std::collections::HashSet;
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

mod distinct;

pub const ROWS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    Distinct(usize, usize),
    Special,
    Depths,
    Depths2,
}

impl Workload {
    pub fn parse(name: &str) -> Result<Self, &'static str> {
        const ERROR: &str = "workload must be <trees>x<features>, special, depths, or depths2";
        match name {
            "special" => Ok(Self::Special),
            "depths" => Ok(Self::Depths),
            "depths2" => Ok(Self::Depths2),
            _ => {
                let (trees, features) = name.split_once('x').ok_or(ERROR)?;
                let trees = trees.parse::<usize>().map_err(|_| ERROR)?;
                let features = features.parse::<usize>().map_err(|_| ERROR)?;
                if trees == 0 || trees > distinct::MAX_TREES || features == 0 {
                    return Err(ERROR);
                }
                Ok(Self::Distinct(trees, features))
            }
        }
    }
}

pub fn workload(kind: Workload) -> (Trees, Vec<Vec<f64>>) {
    match kind {
        Workload::Distinct(trees, features) => distinct::workload(trees, features),
        Workload::Special => special_workload(),
        Workload::Depths => depths_workload(&[1, 2, 3, 4, 5, 6, 7], 0),
        Workload::Depths2 => depths_workload(&[8, 9, 10], 28),
    }
}

pub fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn next_random(state: &mut u64, limit: usize) -> usize {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545_f491_4f6c_dd1d) % limit as u64) as usize
}

fn encoded_leafs(depth: usize, prior_depth: &mut usize) -> Vec<f64> {
    let next_depth = prior_depth.checked_add(depth).unwrap();
    assert!(next_depth <= 53, "leaf fields must fit in 53 bits");
    let leafs = (0..1usize << depth)
        .map(|index| ((index as u64) << *prior_depth) as f64)
        .collect();
    *prior_depth = next_depth;
    leafs
}

pub fn assert_distinct_rows(rows: &[Vec<f64>]) {
    let distinct: HashSet<Vec<u64>> = rows
        .iter()
        .map(|row| row.iter().map(|value| value.to_bits()).collect())
        .collect();
    assert_eq!(distinct.len(), rows.len(), "workload rows must be distinct");
}

pub fn distinct_leaf_counts(model: &Trees, rows: &[Vec<f64>]) -> Vec<usize> {
    let mut leaves = vec![HashSet::new(); model.trees().len()];
    for row in rows {
        for (tree, leaf) in model.leaf_indices(row).unwrap().into_iter().enumerate() {
            leaves[tree].insert(leaf);
        }
    }
    leaves.into_iter().map(|indices| indices.len()).collect()
}

pub fn special_workload() -> (Trees, Vec<Vec<f64>>) {
    const DEPTHS: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    const DEEP_THRESHOLDS: [f64; 8] = [-1.0, -0.0, 0.0, 1.0, -0.42, f64::from_bits(1), 1.37, -1.37];
    const VALUES: [f64; 11] = [
        f64::NAN,
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::INFINITY,
        1.37,
        -0.42,
        f64::from_bits(1),
        f64::from_bits((1u64 << 63) | 1),
    ];

    let mut prior_depth = 0;
    let trees = DEPTHS
        .into_iter()
        .enumerate()
        .map(|(tree, depth)| {
            let selections = (0..depth)
                .map(|level| {
                    let feature = if tree == 2 { 0 } else { (tree + level) % 6 };
                    let threshold = match tree {
                        1 => f64::NAN,
                        2 => [f64::INFINITY, f64::NEG_INFINITY][level],
                        3 => [-0.0, 0.0, 1.37][level],
                        _ => DEEP_THRESHOLDS[level],
                    };
                    Discriminator {
                        feature_idx: FeatureIndex(feature),
                        threshold,
                    }
                })
                .collect();
            let leafs = encoded_leafs(depth, &mut prior_depth);
            SingleTree { leafs, selections }
        })
        .collect();
    let feature_modes = vec![
        NanMode::AsIs,
        NanMode::AsFalse,
        NanMode::AsTrue,
        NanMode::AsIs,
        NanMode::AsFalse,
        NanMode::AsTrue,
    ];
    let model = Trees::new(trees, feature_modes, 1.0, 0.0)
        .expect("the special x86-check model must be valid");
    let mut random = 0x5350_4543_4941_4c31;
    let rows: Vec<Vec<f64>> = (0..ROWS)
        .map(|_| {
            (0..6)
                .map(|_| VALUES[next_random(&mut random, VALUES.len())])
                .collect()
        })
        .collect();
    assert_distinct_rows(&rows);
    for (tree, (&depth, count)) in DEPTHS
        .iter()
        .zip(distinct_leaf_counts(&model, &rows))
        .enumerate()
    {
        let minimum = (1usize << depth).div_ceil(4);
        assert!(
            count >= minimum,
            "special tree {tree} depth {depth} reached {count} leaves, expected at least {minimum}"
        );
    }
    (model, rows)
}

fn depth_threshold(index: usize) -> f64 {
    index as f64 / 4.0 - 7.0
}

pub fn depths_workload(depths: &[usize], first_threshold: usize) -> (Trees, Vec<Vec<f64>>) {
    const FEATURES: usize = 12;

    let mut prior_depth = 0;
    let mut threshold_index = first_threshold;
    let trees: Vec<_> = depths
        .iter()
        .copied()
        .map(|depth| {
            let selections = (0..depth)
                .map(|level| {
                    let feature = if depth == 9 && level == 8 {
                        0
                    } else {
                        level % FEATURES
                    };
                    let split = Discriminator {
                        feature_idx: FeatureIndex(feature),
                        threshold: depth_threshold(threshold_index),
                    };
                    threshold_index += 1;
                    split
                })
                .collect();
            let leafs = encoded_leafs(depth, &mut prior_depth);
            SingleTree { leafs, selections }
        })
        .collect();
    let model = Trees::new(trees, vec![NanMode::AsIs; FEATURES], 1.0, 0.0)
        .expect("the depths x86-check model must be valid");

    let thresholds_by_feature: Vec<Vec<f64>> = (0..FEATURES)
        .map(|feature| {
            model
                .trees()
                .iter()
                .flat_map(|tree| &tree.selections)
                .filter(|split| split.feature_idx == FeatureIndex(feature))
                .map(|split| split.threshold)
                .collect()
        })
        .collect();
    let mut random = 0x4445_5054_4853_3031;
    let mut rows = vec![vec![0.0; FEATURES]; ROWS];
    for feature in 0..FEATURES {
        let mut values = thresholds_by_feature[feature].clone();
        values.extend([-16.0, -1.0, -0.0, 0.0, 1.0, 16.0]);
        for row in &mut rows {
            row[feature] = values[next_random(&mut random, values.len())];
        }
        for (row, threshold) in thresholds_by_feature[feature].iter().enumerate() {
            rows[row][feature] = *threshold;
        }
    }
    assert_distinct_rows(&rows);
    (model, rows)
}
