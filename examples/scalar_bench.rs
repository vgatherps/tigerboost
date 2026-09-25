//! Synthetic dependent-call timings, including the row-selection driver.
use inkwell::context::Context;
use std::{error::Error, hint::black_box, num::NonZeroUsize, time::Instant};
use tigerboost::jit::{CompileOptions, CompiledModel, Order, compile_with_options};
use tigerboost::parsed_tree::{Discriminator, FeatureIndex, NanMode, SingleTree, Trees};

const FEATURES: usize = 16;
const ROWS: usize = 256;
const CALLS: usize = 50_000;
const ROUNDS: usize = 5;

fn workload() -> (Trees, Vec<[f64; FEATURES]>) {
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

fn next_row(index: usize, prediction: f64) -> usize {
    (index + prediction as usize + 1) & (ROWS - 1)
}

fn verify(
    model: &Trees,
    compiled: &CompiledModel<'_>,
    rows: &[[f64; FEATURES]],
) -> (Vec<f64>, usize) {
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
    let (model, rows) = workload();
    let context = Context::create();
    let mut kernels = Vec::new();
    let orders = std::iter::once(Order::TreeOrder)
        .chain([1, 2, 4, 8].map(|cap| Order::Rolling(NonZeroUsize::new(cap).unwrap())));
    for order in orders {
        for count in [1, 2, 4] {
            let options = CompileOptions {
                order,
                accumulators: NonZeroUsize::new(count).unwrap(),
            };
            let start = Instant::now();
            let compiled = compile_with_options(&context, &model, options)?;
            let compile_ms = start.elapsed().as_secs_f64() * 1e3;
            let label = match order {
                Order::TreeOrder => format!("Tree order, sums {count}"),
                Order::Rolling(cap) => format!("Rolling {cap}, sums {count}"),
            };
            kernels.push((label, compiled, compile_ms));
        }
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
    println!("Synthetic: 64 trees, depth 6, {FEATURES} features, {ROWS} rows; generic target");
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
    fn workload_agrees_and_dependent_trace_visits_multiple_rows() {
        let (model, rows) = workload();
        let context = Context::create();
        let compiled = compile_with_options(&context, &model, CompileOptions::default()).unwrap();
        let (scores, distinct) = verify(&model, &compiled, &rows);
        assert_eq!(scores.len(), ROWS);
        assert!(distinct > 1);
    }
}
