use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

pub const FEATURES: usize = 16;
pub const ROWS: usize = 256;

pub fn workload() -> (Trees, Vec<[f64; FEATURES]>) {
    let mut state = 42u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as usize
    };
    let trees = (0..64)
        .map(|_| SingleTree {
            selections: (0..6)
                .map(|_| Discriminator {
                    feature_idx: FeatureIndex(random() % FEATURES),
                    threshold: (random() % 2048) as f64 / 256.0 - 4.0,
                })
                .collect(),
            leafs: (0..64).map(|_| (1 + random() % 16) as f64).collect(),
        })
        .collect();
    let rows = (0..ROWS)
        .map(|_| std::array::from_fn(|_| (random() % 2048) as f64 / 256.0 - 4.0))
        .collect();
    (
        Trees::new(trees, vec![NanMode::AsIs; FEATURES], 1.0, 0.0).unwrap(),
        rows,
    )
}
