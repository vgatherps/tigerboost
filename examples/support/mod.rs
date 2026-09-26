use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

mod distinct;

pub const ROWS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    Mixed,
    ReuseHeavy,
    Disjoint,
    Wide,
    Distinct(usize, usize),
}

impl Workload {
    pub fn parse(name: &str) -> Result<Self, &'static str> {
        match name {
            "mixed" => Ok(Self::Mixed),
            "reuse-heavy" => Ok(Self::ReuseHeavy),
            "disjoint" => Ok(Self::Disjoint),
            "wide" => Ok(Self::Wide),
            "100x100" => Ok(Self::Distinct(100, 100)),
            "500x100" => Ok(Self::Distinct(500, 100)),
            "500x1000" => Ok(Self::Distinct(500, 1000)),
            _ => Err(
                "workload must be mixed, reuse-heavy, disjoint, wide, 100x100, 500x100, or 500x1000",
            ),
        }
    }
}

pub fn workload(kind: Workload) -> (Trees, Vec<Vec<f64>>) {
    let (tree_count, features) = match kind {
        Workload::Mixed => (64, 16),
        Workload::ReuseHeavy => (64, 4),
        Workload::Disjoint => (64, 384),
        Workload::Wide => (128, 128),
        Workload::Distinct(trees, features) => return distinct::workload(trees, features),
    };
    let mut state = 42u64;
    let mut random = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 32) as usize
    };
    let trees = (0..tree_count)
        .map(|tree| SingleTree {
            selections: (0..6)
                .map(|level| {
                    let feature = random() % features;
                    Discriminator {
                        feature_idx: FeatureIndex(if kind == Workload::Disjoint {
                            tree * 6 + level
                        } else {
                            feature
                        }),
                        threshold: if kind == Workload::ReuseHeavy {
                            (random() % 8) as f64 - 4.0
                        } else {
                            (random() % 2048) as f64 / 256.0 - 4.0
                        },
                    }
                })
                .collect(),
            leafs: (0..64).map(|_| (1 + random() % 16) as f64).collect(),
        })
        .collect();
    let rows = (0..ROWS)
        .map(|_| {
            (0..features)
                .map(|_| (random() % 2048) as f64 / 256.0 - 4.0)
                .collect()
        })
        .collect();
    (
        Trees::new(trees, vec![NanMode::AsIs; features], 1.0, 0.0).unwrap(),
        rows,
    )
}
