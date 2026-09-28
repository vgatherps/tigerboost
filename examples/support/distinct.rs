use super::{ROWS, random};
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

const DEPTH: usize = 6;
const GRID: usize = 65536;
pub(super) const MAX_TREES: usize = GRID / DEPTH;

fn dyadic(index: usize) -> f64 {
    (index as f64) / ((GRID / 8) as f64) - 4.0
}

pub(super) fn workload(tree_count: usize, feature_count: usize) -> (Trees, Vec<Vec<f64>>) {
    let splits = tree_count.checked_mul(DEPTH).unwrap();
    assert!((1..=GRID).contains(&splits));
    assert!(feature_count > 0);
    let mut state = 0x4e41544956454636;
    let mut thresholds: Vec<_> = (0..splits).map(|i| dyadic(i * GRID / splits)).collect();
    for i in (1..splits).rev() {
        let j = (random(&mut state) % (i as u64 + 1)) as usize;
        thresholds.swap(i, j);
    }
    let trees = (0..tree_count)
        .map(|tree| SingleTree {
            selections: (0..DEPTH)
                .map(|level| Discriminator {
                    feature_idx: FeatureIndex((random(&mut state) % feature_count as u64) as usize),
                    threshold: thresholds[tree * DEPTH + level],
                })
                .collect(),
            leafs: (0..1usize << DEPTH)
                .map(|_| (1 + random(&mut state) % 16) as f64)
                .collect(),
        })
        .collect();
    let rows = (0..ROWS)
        .map(|_| {
            (0..feature_count)
                .map(|_| dyadic((random(&mut state) % GRID as u64) as usize))
                .collect()
        })
        .collect();
    (
        Trees::new(trees, vec![NanMode::AsIs; feature_count], 1.0, 0.0).unwrap(),
        rows,
    )
}
