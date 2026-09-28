//! Synthetic dependent-call timings, including the row-selection driver.
#[path = "support/bench.rs"]
mod bench;
mod support;

use bench::{next_row, verify};
use inkwell::context::Context;
use std::{error::Error, hint::black_box, num::NonZeroUsize, time::Instant};
use support::{ROWS, Workload, workload};
use tigerboost::jit::compile;
use tigerboost::sort_policy::{SortOptions, plan};

const CALLS: usize = 50_000;
const ROUNDS: usize = 5;

fn tree_major_variants() -> impl Iterator<Item = (String, SortOptions)> {
    [4, 8, 16].into_iter().flat_map(|chunk_trees| {
        [1, 2, 4].map(move |count| {
            (
                format!("Tree-major, {chunk_trees} trees/chunk, sums {count}"),
                SortOptions {
                    chunk_trees: NonZeroUsize::new(chunk_trees).unwrap(),
                    accumulators: NonZeroUsize::new(count).unwrap(),
                },
            )
        })
    })
}

fn measure(predict: impl Fn(usize) -> f64, calls: usize) -> f64 {
    let mut index = 0;
    let start = Instant::now();
    for _ in 0..calls {
        index = next_row(index, black_box(predict(black_box(index))));
    }
    black_box(index);
    start.elapsed().as_secs_f64() * 1e9 / calls as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let choice = Workload::parse(arguments.next().as_deref().unwrap_or("mixed"))?;
    if arguments.next().is_some() {
        return Err("usage: scalar_bench [workload]".into());
    }
    let (model, rows) = workload(choice);
    let context = Context::create();
    let mut kernels = Vec::new();
    for (label, options) in tree_major_variants() {
        let start = Instant::now();
        let compiled = compile(&context, &plan(&model, options))?;
        let compile_ms = start.elapsed().as_secs_f64() * 1e3;
        kernels.push((label, compiled, compile_ms));
    }
    let (scores, distinct) = verify(&model, &kernels[0].1, &rows, CALLS);
    for (_, compiled, _) in &kernels[1..] {
        verify(&model, compiled, &rows, CALLS);
    }
    let reference = |index: usize| model.predict(black_box(rows[index].as_slice())).unwrap();
    let floor = |index| scores[index];
    let measure_variant = |variant, calls| match variant {
        0 => measure(reference, calls),
        last if last == kernels.len() + 1 => measure(floor, calls),
        index => {
            let compiled = &kernels[index - 1].1;
            measure(
                |row| compiled.predict(black_box(rows[row].as_slice())).unwrap(),
                calls,
            )
        }
    };
    let mut order: Vec<_> = (0..kernels.len() + 2).collect();
    for &variant in &order {
        measure_variant(variant, 10_000);
    }
    let mut samples = vec![vec![0.0; order.len()]; ROUNDS];
    for sample in &mut samples {
        for &variant in &order {
            sample[variant] = measure_variant(variant, CALLS);
        }
        order.reverse();
    }
    println!(
        "Synthetic {choice:?}: {} trees, depth 6, {} features, {ROWS} rows; generic target",
        model.trees().len(),
        model.feature_modes().len()
    );
    println!("Dependent chain visits {distinct} distinct rows");
    println!(
        "{ROUNDS} rounds x {CALLS} calls; ns/prediction includes driver (floor not subtracted)"
    );
    println!("Variant                    median      min      max   compile ms");
    for variant in 0..order.len() {
        let (label, compile_ms) = match variant {
            0 => ("Reference", "-".to_owned()),
            last if last == kernels.len() + 1 => ("Driver floor", "-".to_owned()),
            index => (
                kernels[index - 1].0.as_str(),
                format!("{:.3}", kernels[index - 1].2),
            ),
        };
        let mut values: Vec<_> = samples.iter().map(|sample| sample[variant]).collect();
        values.sort_by(f64::total_cmp);
        println!(
            "{label:<25} {:>7.2}  {:>7.2}  {:>7.2}  {compile_ms:>11}",
            values[ROUNDS / 2],
            values[0],
            values[ROUNDS - 1]
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependent_driver_stays_in_range_and_cannot_choose_the_same_row() {
        for index in [0, 1, 127, ROWS - 1] {
            for prediction in 0..ROWS {
                let next = next_row(index, prediction as f64);
                assert!(next < ROWS);
                assert_ne!(next, index);
            }
        }
    }

    #[test]
    fn workload_agrees_and_dependent_trace_visits_multiple_rows() {
        let (model, rows) = workload(Workload::Mixed);
        let context = Context::create();
        let compiled = compile(&context, &plan(&model, SortOptions::default())).unwrap();
        let (scores, distinct) = verify(&model, &compiled, &rows, CALLS);
        assert_eq!(scores.len(), ROWS);
        assert!(distinct > 1);
    }

    #[test]
    fn fixed_workloads_have_expected_shapes_reuse_and_matching_traces() {
        let context = Context::create();
        for (kind, trees, features) in [
            (Workload::ReuseHeavy, 64, 4),
            (Workload::Disjoint, 64, 384),
            (Workload::Wide, 128, 128),
        ] {
            let (model, rows) = workload(kind);
            assert_eq!(model.trees().len(), trees);
            assert_eq!(model.feature_modes().len(), features);
            assert_eq!(rows.len(), ROWS);
            assert!(rows.iter().all(|row| row.len() == features));
            assert!(
                model
                    .trees()
                    .iter()
                    .all(|tree| tree.selections.len() == 6 && tree.leafs.len() == 64)
            );
            for (index, split) in model
                .trees()
                .iter()
                .flat_map(|tree| &tree.selections)
                .enumerate()
            {
                if kind == Workload::Disjoint {
                    assert_eq!(split.feature_idx.0, index);
                } else if kind == Workload::ReuseHeavy {
                    assert!(split.feature_idx.0 < 4);
                    assert!(
                        split.threshold >= -4.0
                            && split.threshold < 4.0
                            && split.threshold.fract() == 0.0
                    );
                }
            }
            let expected = [4, 8, 16]
                .into_iter()
                .flat_map(|chunk_trees| [1, 2, 4].map(move |sums| (chunk_trees, sums)));
            assert_eq!(tree_major_variants().count(), 9);
            for ((label, options), (chunk_trees, sums)) in tree_major_variants().zip(expected) {
                assert_eq!(
                    label,
                    format!("Tree-major, {chunk_trees} trees/chunk, sums {sums}")
                );
                assert_eq!(options.chunk_trees.get(), chunk_trees);
                assert_eq!(options.accumulators.get(), sums);
                let compiled = compile(&context, &plan(&model, options)).unwrap();
                let (_, distinct) = verify(&model, &compiled, &rows, CALLS);
                assert!(distinct > 1, "{kind:?} {label}");
            }
        }
    }

    #[test]
    fn distinct_workloads_are_deterministic_and_match_reference() {
        let context = Context::create();
        for (name, trees, features) in [
            ("100x100", 100, 100),
            ("300x100", 300, 100),
            ("500x100", 500, 100),
            ("500x1000", 500, 1000),
        ] {
            let kind = Workload::parse(name).unwrap();
            let (model, rows) = workload(kind);
            let (again, repeated_rows) = workload(kind);
            assert_eq!(model.trees().len(), trees);
            assert_eq!(model.feature_modes().len(), features);
            assert_eq!(rows.len(), ROWS);
            assert!(rows.iter().all(|row| row.len() == features));
            assert_eq!(rows, repeated_rows);
            let mut borders = std::collections::BTreeSet::new();
            for (tree, repeated) in model.trees().iter().zip(again.trees()) {
                assert_eq!(tree.selections.len(), 6);
                assert_eq!(tree.leafs.len(), 64);
                assert_eq!(tree.leafs, repeated.leafs);
                for (split, repeated) in tree.selections.iter().zip(&repeated.selections) {
                    assert_eq!(split.feature_idx, repeated.feature_idx);
                    assert_eq!(split.threshold.to_bits(), repeated.threshold.to_bits());
                    assert!(borders.insert(split.threshold.to_bits()));
                }
            }
            assert_eq!(borders.len(), trees * 6);
            for (_, options) in tree_major_variants() {
                let compiled = compile(&context, &plan(&model, options)).unwrap();
                for row in &rows {
                    assert_eq!(compiled.predict(row), model.predict(row));
                }
            }
        }
        assert!(Workload::parse("not-a-workload").is_err());
    }
}
