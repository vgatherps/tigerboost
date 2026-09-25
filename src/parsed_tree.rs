use std::{error::Error, fmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FeatureIndex(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanMode {
    /// Leave NaN unchanged; every ordered comparison with it is false.
    AsIs,
    /// Replace NaN with negative infinity before comparing.
    AsFalse,
    /// Replace NaN with positive infinity before comparing, not a constant true bit.
    AsTrue,
}

#[derive(Clone, Debug)]
pub struct Discriminator {
    pub threshold: f64,
    pub feature_idx: FeatureIndex,
}

#[derive(Clone, Debug)]
pub struct SingleTree {
    pub leafs: Vec<f64>,
    pub selections: Vec<Discriminator>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ModelError {
    FeatureStorageTooLarge {
        feature_count: usize,
    },
    DepthTooLarge {
        tree: usize,
        depth: usize,
    },
    LeafCount {
        tree: usize,
        expected: usize,
        actual: usize,
    },
    FeatureOutOfRange {
        tree: usize,
        level: usize,
        feature: FeatureIndex,
        feature_count: usize,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureStorageTooLarge { feature_count } => {
                write!(f, "{feature_count} f64 features exceed addressable storage")
            }
            Self::DepthTooLarge { tree, depth } => write!(
                f,
                "tree {tree}: depth {depth} exceeds addressable f64 leaf storage"
            ),
            Self::LeafCount {
                tree,
                expected,
                actual,
            } => write!(f, "tree {tree}: expected {expected} leaves, got {actual}"),
            Self::FeatureOutOfRange {
                tree,
                level,
                feature,
                feature_count,
            } => write!(
                f,
                "tree {tree}, level {level}: feature {} is outside {feature_count} features",
                feature.0
            ),
        }
    }
}

impl Error for ModelError {}

#[derive(Debug, PartialEq, Eq)]
pub struct RowLengthError {
    pub expected: usize,
    pub actual: usize,
}

impl fmt::Display for RowLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expected at least {} features, got {}",
            self.expected, self.actual
        )
    }
}

impl Error for RowLengthError {}

/// Validated, immutable numeric symmetric trees over flat f64 feature rows.
#[derive(Debug)]
pub struct Trees {
    trees: Vec<SingleTree>,
    feature_modes: Vec<NanMode>,
    scale: f64,
    bias: f64,
}

impl Trees {
    /// Depth is limited by addressable leaf storage: 2^depth f64 values must fit isize::MAX bytes.
    pub fn new(
        trees: Vec<SingleTree>,
        feature_modes: Vec<NanMode>,
        scale: f64,
        bias: f64,
    ) -> Result<Self, ModelError> {
        let max_values = isize::MAX as usize / size_of::<f64>();
        if feature_modes.len() > max_values {
            return Err(ModelError::FeatureStorageTooLarge {
                feature_count: feature_modes.len(),
            });
        }
        for (tree, description) in trees.iter().enumerate() {
            let depth = description.selections.len();
            if depth >= usize::BITS as usize || (1usize << depth) > max_values {
                return Err(ModelError::DepthTooLarge { tree, depth });
            }
            let expected = 1usize << depth;
            if description.leafs.len() != expected {
                return Err(ModelError::LeafCount {
                    tree,
                    expected,
                    actual: description.leafs.len(),
                });
            }
            for (level, split) in description.selections.iter().enumerate() {
                if split.feature_idx.0 >= feature_modes.len() {
                    return Err(ModelError::FeatureOutOfRange {
                        tree,
                        level,
                        feature: split.feature_idx,
                        feature_count: feature_modes.len(),
                    });
                }
            }
        }
        Ok(Self {
            trees,
            feature_modes,
            scale,
            bias,
        })
    }

    pub fn trees(&self) -> &[SingleTree] {
        &self.trees
    }

    pub fn feature_modes(&self) -> &[NanMode] {
        &self.feature_modes
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Tree-order reference, independent of scheduling and code generation.
    pub fn predict(&self, row: &[f64]) -> Result<f64, RowLengthError> {
        self.check_row(row)?;
        if self.trees.is_empty() {
            return Ok(self.bias);
        }
        let mut sum = 0.0;
        for tree in &self.trees {
            sum += tree.leafs[self.leaf_index(tree, row)];
        }
        Ok(sum * self.scale + self.bias)
    }

    pub fn leaf_indices(&self, row: &[f64]) -> Result<Vec<usize>, RowLengthError> {
        self.check_row(row)?;
        Ok(self
            .trees
            .iter()
            .map(|tree| self.leaf_index(tree, row))
            .collect())
    }

    fn check_row(&self, row: &[f64]) -> Result<(), RowLengthError> {
        if row.len() < self.feature_modes.len() {
            return Err(RowLengthError {
                expected: self.feature_modes.len(),
                actual: row.len(),
            });
        }
        Ok(())
    }

    fn leaf_index(&self, tree: &SingleTree, row: &[f64]) -> usize {
        let mut index = 0;
        for (level, split) in tree.selections.iter().enumerate() {
            let mut value = row[split.feature_idx.0];
            if value.is_nan() {
                value = match self.feature_modes[split.feature_idx.0] {
                    NanMode::AsIs => value,
                    NanMode::AsFalse => f64::NEG_INFINITY,
                    NanMode::AsTrue => f64::INFINITY,
                };
            }
            index |= usize::from(value > split.threshold) << level;
        }
        index
    }
}
