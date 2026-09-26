//! Synthetic dependent-call timings, including the row-selection driver.
mod support;

use inkwell::context::Context;
use std::{error::Error, hint::black_box, num::NonZeroUsize, time::Instant};
use support::{ROWS, Workload, workload};
use tigerboost::jit::{CompiledModel, compile};
use tigerboost::parsed_tree::Trees;
use tigerboost::sort_policy::{ComparisonMode, Order, SortOptions, plan};

const CALLS: usize = 50_000;
const ROUNDS: usize = 5;

fn next_row(index: usize, prediction: f64) -> usize {
    (index + (prediction as usize | 1)) & (ROWS - 1)
}

fn verify(model: &Trees, compiled: &CompiledModel<'_>, rows: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let scores: Vec<_> = rows.iter().map(|row| model.predict(row).unwrap()).collect();
    for (row, &score) in rows.iter().zip(&scores) {
        assert_eq!(compiled.predict(row).unwrap(), score);
    }
    let (mut reference, mut jit, mut floor) = (0, 0, 0);
    let mut visited = [false; ROWS];
    for _ in 0..CALLS {
        assert_eq!((reference, reference), (jit, floor));
        visited[reference] = true;
        reference = next_row(reference, model.predict(&rows[reference]).unwrap());
        jit = next_row(jit, compiled.predict(&rows[jit]).unwrap());
        floor = next_row(floor, scores[floor]);
    }
    assert_eq!((reference, reference), (jit, floor));
    let distinct = visited.into_iter().filter(|&seen| seen).count();
    assert!(distinct > 1, "dependent chain must visit varied rows");
    assert_ne!(
        reference,
        next_row(reference, scores[reference]),
        "dependent chain settled on one row"
    );
    (scores, distinct)
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
    let orders = std::iter::once(Order::TreeOrder)
        .chain([1, 2, 4, 8].map(|cap| Order::Rolling(NonZeroUsize::new(cap).unwrap())));
    for order in orders {
        for count in [1, 2, 4] {
            let options = SortOptions {
                order,
                accumulators: NonZeroUsize::new(count).unwrap(),
                comparisons: ComparisonMode::Scalar,
            };
            let start = Instant::now();
            let compiled = compile(&context, &plan(&model, options))?;
            let compile_ms = start.elapsed().as_secs_f64() * 1e3;
            let label = match order {
                Order::TreeOrder => format!("Tree order, sums {count}"),
                Order::Rolling(cap) => format!("Rolling {cap}, sums {count}"),
            };
            kernels.push((label, compiled, compile_ms));
        }
    }
    for cap in [1, 2, 4, 8] {
        let options = SortOptions {
            order: Order::Rolling(NonZeroUsize::new(cap).unwrap()),
            accumulators: NonZeroUsize::new(2).unwrap(),
            comparisons: ComparisonMode::Simd2,
        };
        let start = Instant::now();
        let compiled = compile(&context, &plan(&model, options))?;
        let compile_ms = start.elapsed().as_secs_f64() * 1e3;
        kernels.push((format!("SIMD2 rolling {cap}, sums 2"), compiled, compile_ms));
    }
    let (scores, distinct) = verify(&model, &kernels[0].1, &rows);
    for (_, compiled, _) in &kernels[1..] {
        verify(&model, compiled, &rows);
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
        let (scores, distinct) = verify(&model, &compiled, &rows);
        assert_eq!(scores.len(), ROWS);
        assert!(distinct > 1);
    }

    #[test]
    fn fixed_workloads_have_expected_shapes_reuse_and_matching_simd_traces() {
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
            let compiled = compile(
                &context,
                &plan(
                    &model,
                    SortOptions {
                        order: Order::Rolling(NonZeroUsize::new(2).unwrap()),
                        accumulators: NonZeroUsize::new(2).unwrap(),
                        comparisons: ComparisonMode::Simd2,
                    },
                ),
            )
            .unwrap();
            let (_, distinct) = verify(&model, &compiled, &rows);
            assert!(distinct > 1, "{kind:?}");
        }
    }
}
