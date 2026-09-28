mod support;

use inkwell::context::Context;
use std::{
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};
use support::{ROWS, Workload};
use tigerboost::jit::{EmitOptions, compile, emit_assembly_with};
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};
use tigerboost::sort_policy::{ComparisonMode, Order, RegisterBudget, SortOptions, plan};

const USAGE: &str = "usage: x86_check <workload> <fresh-output-dir> [--avx2]";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkloadChoice {
    Standard(Workload),
    Special,
}

fn parse_workload(name: &str) -> Result<WorkloadChoice, &'static str> {
    if name == "special" {
        return Ok(WorkloadChoice::Special);
    }
    if let Ok(workload) = Workload::parse(name) {
        return Ok(WorkloadChoice::Standard(workload));
    }
    let (trees, features) = name.split_once('x').ok_or(USAGE)?;
    let trees = trees.parse::<usize>().map_err(|_| USAGE)?;
    let features = features.parse::<usize>().map_err(|_| USAGE)?;
    if trees == 0 || features == 0 || trees > 10_922 {
        return Err(USAGE);
    }
    Ok(WorkloadChoice::Standard(Workload::Distinct(
        trees, features,
    )))
}

fn special_workload() -> (Trees, Vec<Vec<f64>>) {
    const DEPTHS: [usize; 7] = [0, 1, 2, 3, 5, 6, 8];
    const THRESHOLDS: [f64; 7] = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        1.37,
        -0.42,
    ];
    const VALUES: [f64; 9] = [
        f64::NAN,
        f64::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::INFINITY,
        1.37,
        -0.42,
    ];

    let mut threshold = 0;
    let mut next_leaf = 1;
    let trees = DEPTHS
        .into_iter()
        .enumerate()
        .map(|(tree, depth)| {
            let selections = (0..depth)
                .map(|level| {
                    let feature = if tree == 2 { 0 } else { (tree + level) % 6 };
                    let selection = Discriminator {
                        feature_idx: FeatureIndex(feature),
                        threshold: THRESHOLDS[threshold % THRESHOLDS.len()],
                    };
                    threshold += 1;
                    selection
                })
                .collect();
            let leaf_count = 1usize << depth;
            let leafs = (next_leaf..next_leaf + leaf_count)
                .map(|leaf| leaf as f64)
                .collect();
            next_leaf += leaf_count;
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
    let rows = (0..ROWS)
        .map(|row| {
            (0..6)
                .map(|feature| VALUES[(row + feature) % VALUES.len()])
                .collect()
        })
        .collect();
    (model, rows)
}

fn workload(choice: WorkloadChoice) -> (Trees, Vec<Vec<f64>>) {
    match choice {
        WorkloadChoice::Standard(choice) => support::workload(choice),
        WorkloadChoice::Special => special_workload(),
    }
}

fn variants() -> [(&'static str, SortOptions); 5] {
    [
        ("scalar-gp12-fp9-2sums", SortOptions::default()),
        (
            "scalar-gp4-fp9-2sums",
            SortOptions {
                order: Order::Scheduled(RegisterBudget {
                    general: NonZeroUsize::new(4).unwrap(),
                    floating: NonZeroUsize::new(9).unwrap(),
                }),
                accumulators: NonZeroUsize::new(2).unwrap(),
                comparisons: ComparisonMode::Scalar,
            },
        ),
        (
            "tree-order-2sums",
            SortOptions {
                order: Order::TreeOrder,
                accumulators: NonZeroUsize::new(2).unwrap(),
                comparisons: ComparisonMode::Scalar,
            },
        ),
        (
            "tree-major-4-2sums",
            SortOptions {
                order: Order::TreeMajor {
                    chunk_trees: NonZeroUsize::new(4).unwrap(),
                },
                accumulators: NonZeroUsize::new(2).unwrap(),
                comparisons: ComparisonMode::Scalar,
            },
        ),
        (
            "tree-major-8-2sums",
            SortOptions {
                order: Order::TreeMajor {
                    chunk_trees: NonZeroUsize::new(8).unwrap(),
                },
                accumulators: NonZeroUsize::new(2).unwrap(),
                comparisons: ComparisonMode::Scalar,
            },
        ),
    ]
}

fn reference_scores(model: &Trees, rows: &[Vec<f64>]) -> Result<Vec<f64>, Box<dyn Error>> {
    rows.iter()
        .map(|row| model.predict(row).map_err(Into::into))
        .collect()
}

#[cfg(test)]
fn check_variants_natively(model: &Trees, rows: &[Vec<f64>]) -> Result<(), Box<dyn Error>> {
    let expected = reference_scores(model, rows)?;
    let context = Context::create();
    for (name, options) in variants() {
        let plan = plan(model, options)?;
        let compiled = compile(&context, &plan)?;
        for (row, (&expected, features)) in expected.iter().zip(rows).enumerate() {
            assert_eq!(
                compiled.predict(features)?,
                expected,
                "native {name} mismatch at row {row}"
            );
        }
    }
    Ok(())
}

fn write_f64s(path: &Path, values: impl IntoIterator<Item = f64>) -> Result<(), Box<dyn Error>> {
    let mut output = BufWriter::new(File::create_new(path)?);
    for value in values {
        output.write_all(&value.to_le_bytes())?;
    }
    output.flush()?;
    Ok(())
}

fn write_outputs(
    output: &Path,
    model: &Trees,
    rows: &[Vec<f64>],
    emit_options: EmitOptions,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(rows.len(), ROWS);
    let expected = reference_scores(model, rows)?;
    let feature_count = model.feature_modes().len();
    let context = Context::create();

    fs::create_dir(output)?;
    for (name, options) in variants() {
        let plan = plan(model, options)?;
        let compiled = compile(&context, &plan)?;
        for (row, (&expected, features)) in expected.iter().zip(rows).enumerate() {
            assert_eq!(
                compiled.predict(features)?,
                expected,
                "native {name} mismatch at row {row}"
            );
        }
        let assembly = emit_assembly_with(&context, &plan, "x86_64-apple-darwin", emit_options)?;
        let assembly = assembly.trim_end_matches('\0');
        assert!(
            !assembly.as_bytes().contains(&0),
            "{name} assembly contains an interior NUL byte"
        );
        let path = output.join(format!("{name}.s"));
        File::create_new(&path)?.write_all(assembly.as_bytes())?;
        println!("{}", path.display());
    }

    let rows_path = output.join("rows.bin");
    write_f64s(&rows_path, rows.iter().flatten().copied())?;
    println!("{}", rows_path.display());
    let expected_path = output.join("expected.bin");
    write_f64s(&expected_path, expected)?;
    println!("{}", expected_path.display());
    let manifest_path = output.join("manifest.txt");
    let mut manifest = File::create_new(&manifest_path)?;
    writeln!(manifest, "features {feature_count}")?;
    writeln!(manifest, "rows {ROWS}")?;
    writeln!(manifest, "avx2 {}", u8::from(emit_options.avx2))?;
    for (name, _) in variants() {
        writeln!(manifest, "variant {name}")?;
    }
    println!("{}", manifest_path.display());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let choice = parse_workload(&arguments.next().ok_or(USAGE)?)?;
    let output = PathBuf::from(arguments.next().ok_or(USAGE)?);
    let avx2 = match arguments.next().as_deref() {
        None => false,
        Some("--avx2") => true,
        Some(_) => return Err(USAGE.into()),
    };
    if arguments.next().is_some() {
        return Err(USAGE.into());
    }
    let (model, rows) = workload(choice);
    write_outputs(&output, &model, &rows, EmitOptions { avx2 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!(
                "tigerboost-x86-check-{}-{sequence}",
                std::process::id()
            )))
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    panic!("failed to remove {}: {error}", self.0.display());
                }
            }
        }
    }

    #[test]
    fn special_model_is_valid_and_covers_every_mode_and_threshold_kind() {
        let (model, rows) = special_workload();
        assert_eq!(rows.len(), ROWS);
        for mode in [NanMode::AsIs, NanMode::AsFalse, NanMode::AsTrue] {
            assert!(model.feature_modes().contains(&mode));
        }
        let thresholds: Vec<_> = model
            .trees()
            .iter()
            .flat_map(|tree| tree.selections.iter().map(|split| split.threshold))
            .collect();
        assert!(thresholds.iter().any(|threshold| threshold.is_nan()));
        assert!(thresholds.contains(&f64::INFINITY));
        assert!(thresholds.contains(&f64::NEG_INFINITY));
        assert!(
            thresholds
                .iter()
                .any(|threshold| threshold.to_bits() == (-0.0f64).to_bits())
        );
        assert!(
            thresholds
                .iter()
                .any(|threshold| threshold.to_bits() == 0.0f64.to_bits())
        );
        assert!(model.trees().iter().any(|tree| {
            tree.selections.iter().enumerate().any(|(index, split)| {
                tree.selections[index + 1..]
                    .iter()
                    .any(|other| other.feature_idx == split.feature_idx)
            })
        }));
    }

    #[test]
    fn outputs_have_the_documented_sizes_and_symbols() {
        let directory = TempDirectory::new();
        let (model, rows) = support::workload(Workload::Distinct(5, 8));
        write_outputs(&directory.0, &model, &rows, EmitOptions::default()).unwrap();
        assert_eq!(
            fs::metadata(directory.0.join("rows.bin")).unwrap().len(),
            256 * 8 * 8
        );
        assert_eq!(
            fs::metadata(directory.0.join("expected.bin"))
                .unwrap()
                .len(),
            256 * 8
        );
        let manifest = fs::read_to_string(directory.0.join("manifest.txt")).unwrap();
        assert_eq!(manifest.lines().filter(|line| *line == "avx2 0").count(), 1);
        assert_eq!(
            manifest
                .lines()
                .filter(|line| line.starts_with("variant "))
                .count(),
            5
        );
        for (name, _) in variants() {
            let assembly = fs::read(directory.0.join(format!("{name}.s"))).unwrap();
            assert!(!assembly.is_empty());
            assert!(!assembly.contains(&0));
            assert!(
                assembly
                    .windows("_predict".len())
                    .any(|window| window == b"_predict")
            );
        }
    }

    #[test]
    fn avx2_outputs_record_options_and_emit_vector_compares() {
        let directory = TempDirectory::new();
        let (model, rows) = support::workload(Workload::Distinct(5, 8));
        write_outputs(&directory.0, &model, &rows, EmitOptions { avx2: true }).unwrap();
        let manifest = fs::read_to_string(directory.0.join("manifest.txt")).unwrap();
        assert_eq!(manifest.lines().filter(|line| *line == "avx2 1").count(), 1);
        let context = Context::create();
        for (name, options) in variants() {
            let assembly = fs::read_to_string(directory.0.join(format!("{name}.s"))).unwrap();
            let plan = plan(&model, options).unwrap();
            let expected = emit_assembly_with(
                &context,
                &plan,
                "x86_64-apple-darwin",
                EmitOptions { avx2: true },
            )
            .unwrap();
            assert_eq!(assembly, expected.trim_end_matches('\0'));
            if matches!(options.order, Order::TreeMajor { .. }) {
                assert_eq!(assembly.matches("vcmpgt_oqpd").count(), 10);
                assert_eq!(assembly.matches("vmovmskpd").count(), 10);
            }
        }
    }

    #[test]
    fn every_variant_matches_the_reference_natively() {
        let (special_model, special_rows) = special_workload();
        check_variants_natively(&special_model, &special_rows).unwrap();
        let (distinct_model, distinct_rows) = support::workload(Workload::Distinct(20, 30));
        check_variants_natively(&distinct_model, &distinct_rows).unwrap();
    }
}
