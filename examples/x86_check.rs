mod support;

#[cfg(test)]
use std::collections::HashSet;
use std::{
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::Command,
};
use support::{ROWS, Workload, workload};
#[cfg(test)]
use support::{assert_distinct_rows, depths_workload, distinct_leaf_counts, special_workload};
use tigerboost::jit::{EmitOptions, compile, emit_assembly_with};
#[cfg(test)]
use tigerboost::parsed_tree::NanMode;
use tigerboost::parsed_tree::Trees;
use tigerboost::sort_policy::plan;

const USAGE: &str =
    "usage: x86_check <special|depths|depths2|<trees>x<features>> <fresh-output-dir> [--avx2]";

fn reference_scores(model: &Trees, rows: &[Vec<f64>]) -> Result<Vec<f64>, Box<dyn Error>> {
    rows.iter()
        .map(|row| model.predict(row).map_err(Into::into))
        .collect()
}

fn check_natively(
    model: &Trees,
    rows: &[Vec<f64>],
    expected: &[f64],
) -> Result<(), Box<dyn Error>> {
    let compiled = compile(model)?;
    for (row, (&expected, features)) in expected.iter().zip(rows).enumerate() {
        assert_eq!(
            compiled.predict(features)?.to_bits(),
            expected.to_bits(),
            "native mismatch at row {row}"
        );
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
) -> Result<&'static str, Box<dyn Error>> {
    assert_eq!(rows.len(), ROWS);
    let expected = reference_scores(model, rows)?;
    let feature_count = model.feature_modes().len();

    fs::create_dir(output)?;
    check_natively(model, rows, &expected)?;
    let plan = plan(model);
    let name = if emit_options.avx2 {
        "kernel-avx2"
    } else {
        "kernel"
    };
    let assembly = emit_assembly_with(&plan, "x86_64-apple-darwin", emit_options)?;
    let path = output.join(format!("{name}.s"));
    File::create_new(&path)?.write_all(assembly.as_bytes())?;
    println!("{}", path.display());

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
    writeln!(manifest, "variant {name}")?;
    println!("{}", manifest_path.display());
    Ok(name)
}

fn run(command: &mut Command) -> Result<(), Box<dyn Error>> {
    let status = command.status()?;
    if !status.success() {
        return Err(format!("{command:?} exited with {status}").into());
    }
    Ok(())
}

fn check_kernel(output: &Path, name: &str) -> Result<(), Box<dyn Error>> {
    let result = (|| {
        let object = output.join(format!("{name}.o"));
        let binary = output.join(format!("{name}.check"));
        run(Command::new("clang")
            .args(["-arch", "x86_64", "-c"])
            .arg(output.join(format!("{name}.s")))
            .arg("-o")
            .arg(&object))?;
        run(Command::new("clang")
            .args(["-arch", "x86_64", "-O1", "-Wl,-no_pie"])
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/x86_check.c"))
            .arg(&object)
            .arg("-o")
            .arg(&binary))?;
        run(Command::new("arch").arg("-x86_64").arg(&binary).arg(output))
    })();
    println!("{} {name}", if result.is_ok() { "PASS" } else { "FAIL" });
    result
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let choice = Workload::parse(&arguments.next().ok_or(USAGE)?)?;
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
    let name = write_outputs(&output, &model, &rows, EmitOptions { avx2 })?;
    check_kernel(&output, name)
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

    fn assert_leaf_fields_decode(model: &Trees, rows: &[Vec<f64>]) {
        for row in rows {
            let indices = model.leaf_indices(row).unwrap();
            let mut prior_depth = 0;
            let mut encoded = 0u64;
            for (tree, index) in model.trees().iter().zip(indices) {
                encoded |= (index as u64) << prior_depth;
                prior_depth += tree.selections.len();
            }
            assert!(prior_depth <= 53);
            assert_eq!(model.predict(row), Ok(encoded as f64));
        }
    }

    fn mutated_leaf_model(model: &Trees, tree: usize, leaf: usize) -> Trees {
        let mut trees = model.trees().to_vec();
        trees[tree].leafs[leaf] += 1000.0;
        Trees::new(
            trees,
            model.feature_modes().to_vec(),
            model.scale(),
            model.bias(),
        )
        .unwrap()
    }

    #[test]
    fn special_model_is_valid_and_covers_every_mode_and_threshold_kind() {
        let (model, rows) = special_workload();
        assert_eq!(rows.len(), ROWS);
        assert_eq!(
            model
                .trees()
                .iter()
                .map(|tree| tree.selections.len())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_distinct_rows(&rows);
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
        assert!(
            rows.iter()
                .flatten()
                .any(|value| value.to_bits() == f64::from_bits(1).to_bits())
        );
        assert!(
            rows.iter()
                .flatten()
                .any(|value| { value.to_bits() == f64::from_bits((1u64 << 63) | 1).to_bits() })
        );
        assert_eq!(model.trees()[2].selections.len(), 2);
        assert_eq!(
            model.trees()[2].selections[0].feature_idx,
            model.trees()[2].selections[1].feature_idx
        );
        let counts = distinct_leaf_counts(&model, &rows);
        assert_eq!(counts.len(), model.trees().len());
        for (&count, tree) in counts.iter().zip(model.trees()) {
            assert!(count >= (1usize << tree.selections.len()).div_ceil(4));
        }
        assert_leaf_fields_decode(&model, &rows);
        println!(
            "special distinct rows: {}; leaf counts: {counts:?}",
            rows.len()
        );
    }

    #[test]
    fn depths_models_cover_depths_threshold_hits_and_leaf_fields() {
        assert_eq!(Workload::parse("depths"), Ok(Workload::Depths));
        assert_eq!(Workload::parse("depths2"), Ok(Workload::Depths2));
        let (first, first_rows) = depths_workload(&[1, 2, 3, 4, 5, 6, 7], 0);
        let (second, second_rows) = depths_workload(&[8, 9, 10], 28);
        let depths: Vec<_> = first
            .trees()
            .iter()
            .chain(second.trees())
            .map(|tree| tree.selections.len())
            .collect();
        assert_eq!(depths, (1..=10).collect::<Vec<_>>());
        assert_eq!(first.feature_modes().len(), 12);
        assert_eq!(second.feature_modes().len(), 12);

        let splits: Vec<_> = first
            .trees()
            .iter()
            .chain(second.trees())
            .flat_map(|tree| &tree.selections)
            .collect();
        assert!(splits.iter().all(|split| split.threshold.is_finite()));
        assert_eq!(
            splits
                .iter()
                .map(|split| split.threshold.to_bits())
                .collect::<HashSet<_>>()
                .len(),
            splits.len()
        );
        let depth_nine = second
            .trees()
            .iter()
            .find(|tree| tree.selections.len() == 9)
            .unwrap();
        assert!(
            depth_nine
                .selections
                .iter()
                .enumerate()
                .any(|(index, split)| {
                    depth_nine.selections[index + 1..]
                        .iter()
                        .any(|other| other.feature_idx == split.feature_idx)
                })
        );

        for (model, rows) in [(&first, &first_rows), (&second, &second_rows)] {
            assert_distinct_rows(rows);
            for split in model.trees().iter().flat_map(|tree| &tree.selections) {
                assert!(rows.iter().any(|row| {
                    row[split.feature_idx.0].to_bits() == split.threshold.to_bits()
                }));
            }
            assert_leaf_fields_decode(model, rows);
        }
        println!(
            "depths distinct rows: {}, leaf counts: {:?}; depths2 distinct rows: {}, leaf counts: {:?}",
            first_rows.len(),
            distinct_leaf_counts(&first, &first_rows),
            second_rows.len(),
            distinct_leaf_counts(&second, &second_rows)
        );
    }

    #[test]
    fn exported_rows_detect_deep_leaf_mutants() {
        for (model, rows, depth) in [
            {
                let (model, rows) = special_workload();
                (model, rows, 8)
            },
            {
                let (model, rows) = depths_workload(&[1, 2, 3, 4, 5, 6, 7], 0);
                (model, rows, 7)
            },
        ] {
            let tree = model
                .trees()
                .iter()
                .position(|tree| tree.selections.len() == depth)
                .unwrap();
            let leaf = model.leaf_indices(&rows[0]).unwrap()[tree];
            let mutant = mutated_leaf_model(&model, tree, leaf);
            let expected = reference_scores(&model, &rows).unwrap();
            let actual = reference_scores(&mutant, &rows).unwrap();
            assert!(
                expected
                    .iter()
                    .zip(actual)
                    .any(|(expected, actual)| expected != &actual)
            );
        }
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
            1
        );
        let assembly = fs::read(directory.0.join("kernel.s")).unwrap();
        assert!(write_outputs(&directory.0, &model, &rows, EmitOptions::default()).is_err());
        assert_eq!(fs::read(directory.0.join("kernel.s")).unwrap(), assembly);
        assert!(!assembly.is_empty());
        assert!(!assembly.contains(&0));
        assert!(
            assembly
                .windows("_predict".len())
                .any(|window| window == b"_predict")
        );
    }

    #[test]
    fn avx2_outputs_record_options_and_emit_vector_compares() {
        let directory = TempDirectory::new();
        let (model, rows) = support::workload(Workload::Distinct(5, 8));
        write_outputs(&directory.0, &model, &rows, EmitOptions { avx2: true }).unwrap();
        let manifest = fs::read_to_string(directory.0.join("manifest.txt")).unwrap();
        assert_eq!(manifest.lines().filter(|line| *line == "avx2 1").count(), 1);
        let assembly = fs::read_to_string(directory.0.join("kernel-avx2.s")).unwrap();
        let plan = plan(&model);
        let expected = emit_assembly_with(
            &plan,
            "x86_64-apple-darwin",
            EmitOptions { avx2: true },
        )
        .unwrap();
        assert_eq!(assembly, expected);
        assert_eq!(assembly.matches("vcmpgt_oqpd").count(), 10);
        assert_eq!(assembly.matches("vmovmskpd").count(), 10);
    }

    #[test]
    fn kernel_matches_the_reference_natively() {
        for kind in [
            Workload::Special,
            Workload::Depths,
            Workload::Depths2,
            Workload::Distinct(20, 30),
        ] {
            let (model, rows) = workload(kind);
            check_natively(&model, &rows, &reference_scores(&model, &rows).unwrap()).unwrap();
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn encoded_mixed_forest_keeps_swapped_leaf_fields_on_both_paths() {
        use tigerboost::jit::ordered_predicates;
        use tigerboost::parsed_tree::{Discriminator, FeatureIndex, SingleTree};
        use tigerboost::sort_policy::Action;

        let mut fields = Vec::new();
        let mut shift = 0;
        let trees = [
            vec![0, 1, 2, 3, 4],
            vec![1, 3],
            vec![5],
            vec![6, 7],
            vec![5, 6, 5],
            vec![7],
            vec![1, 3],
            vec![5],
            vec![],
            vec![6],
        ]
        .into_iter()
        .map(|features| {
            let depth = features.len();
            let width = depth.max(1);
            fields.push((shift, width));
            let leafs = (0..1usize << depth)
                .map(|leaf| ((if depth == 0 { 1 } else { leaf }) << shift) as f64)
                .collect();
            shift += width;
            let selections = features
                .into_iter()
                .map(|feature| Discriminator {
                    feature_idx: FeatureIndex(feature),
                    threshold: 0.0,
                })
                .collect();
            SingleTree { leafs, selections }
        })
        .collect();
        assert!(shift <= 53);
        let model = Trees::new(trees, vec![NanMode::AsIs; 8], 1.0, 0.0).unwrap();
        let plan = plan(&model);
        assert_eq!(plan.chunks().len(), 2);
        let range = plan
            .actions()
            .iter()
            .find_map(|action| match action {
                Action::EvaluateTree {
                    tree, predicates, ..
                } if tree.0 == 0 => Some(predicates.clone()),
                _ => None,
            })
            .unwrap();
        let ordered = ordered_predicates(&plan, "x86_64-apple-darwin");
        assert_eq!(
            ordered[range]
                .iter()
                .map(|split| split.feature_idx.0)
                .collect::<Vec<_>>(),
            [1, 0, 3, 2, 4, 0],
        );
        let rows: Vec<Vec<f64>> = (0..ROWS)
            .map(|pattern| {
                (0..8)
                    .map(|feature| f64::from(pattern & (1 << feature) != 0))
                    .collect()
            })
            .collect();
        for row in &rows {
            let score = model.predict(row).unwrap() as u64;
            for ((tree, leaf), &(shift, width)) in model
                .trees()
                .iter()
                .zip(model.leaf_indices(row).unwrap())
                .zip(&fields)
            {
                assert_eq!(
                    (score >> shift) & ((1 << width) - 1),
                    if tree.selections.is_empty() {
                        1
                    } else {
                        leaf as u64
                    },
                );
            }
        }
        for avx2 in [false, true] {
            let directory = TempDirectory::new();
            let name = write_outputs(&directory.0, &model, &rows, EmitOptions { avx2 }).unwrap();
            check_kernel(&directory.0, name).unwrap();
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn empty_forest_preserves_bias_bits_on_both_paths() {
        let rows = vec![vec![0.0; 2]; ROWS];
        for bits in [
            0,
            (-0.0f64).to_bits(),
            7.0f64.to_bits(),
            1,
            0x7ff8_1234_5678_9abc,
            0xfff0_0000_0000_0001,
        ] {
            let model = Trees::new(
                vec![],
                vec![NanMode::AsIs; 2],
                f64::NAN,
                f64::from_bits(bits),
            )
            .unwrap();
            assert!(
                reference_scores(&model, &rows)
                    .unwrap()
                    .iter()
                    .all(|value| value.to_bits() == bits)
            );
            for avx2 in [false, true] {
                let directory = TempDirectory::new();
                let name =
                    write_outputs(&directory.0, &model, &rows, EmitOptions { avx2 }).unwrap();
                check_kernel(&directory.0, name).unwrap();
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn gate_executes_kernel_and_rejects_mismatches_and_invalid_assembly() {
        let avx2_directory = TempDirectory::new();
        let (model, rows) = workload(Workload::Distinct(100, 100));
        let name =
            write_outputs(&avx2_directory.0, &model, &rows, EmitOptions { avx2: true }).unwrap();
        check_kernel(&avx2_directory.0, name).unwrap();
        let directory = TempDirectory::new();
        let (model, rows) = workload(Workload::Distinct(5, 8));
        let name = write_outputs(&directory.0, &model, &rows, EmitOptions::default()).unwrap();
        check_kernel(&directory.0, name).unwrap();
        let expected = directory.0.join("expected.bin");
        let mut bytes = fs::read(&expected).unwrap();
        bytes[..8].copy_from_slice(&(-1.0f64).to_le_bytes());
        fs::write(expected, bytes).unwrap();
        assert!(check_kernel(&directory.0, name).is_err());
        fs::write(
            directory.0.join(format!("{name}.s")),
            "invalid instruction\n",
        )
        .unwrap();
        assert!(check_kernel(&directory.0, name).is_err());
    }
}
