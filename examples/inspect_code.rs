//! Writes separately emitted target assembly, not MCJIT byte disassembly.
mod support;

use inkwell::context::Context;
use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    num::NonZeroUsize,
    path::PathBuf,
};
use support::Workload;
use tigerboost::jit::emit_assembly;
use tigerboost::sort_policy::{ComparisonMode, Order, RegisterBudget, SortOptions, plan};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: inspect_code <fresh-output-directory> [workload]")?,
    );
    let choice = match arguments.next() {
        Some(name) => Workload::parse(name.to_str().ok_or("workload name must be UTF-8")?)?,
        None => Workload::Mixed,
    };
    if arguments.next().is_some() {
        return Err("usage: inspect_code <fresh-output-directory> [workload]".into());
    }
    fs::create_dir(&output)?;
    let (model, _) = support::workload(choice);
    let context = Context::create();
    for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
        for (name, options) in [
            (
                "tree-order-1",
                SortOptions {
                    order: Order::TreeOrder,
                    accumulators: NonZeroUsize::new(1).unwrap(),
                    comparisons: ComparisonMode::Scalar,
                },
            ),
            (
                "scheduled-12gp-9fp-2sums",
                SortOptions {
                    order: Order::Scheduled(RegisterBudget {
                        general: NonZeroUsize::new(12).unwrap(),
                        floating: NonZeroUsize::new(9).unwrap(),
                    }),
                    accumulators: NonZeroUsize::new(2).unwrap(),
                    comparisons: ComparisonMode::Scalar,
                },
            ),
            (
                "simd2-scheduled-6gp-9fp-2sums",
                SortOptions {
                    order: Order::Scheduled(RegisterBudget {
                        general: NonZeroUsize::new(6).unwrap(),
                        floating: NonZeroUsize::new(9).unwrap(),
                    }),
                    accumulators: NonZeroUsize::new(2).unwrap(),
                    comparisons: ComparisonMode::Simd2,
                },
            ),
            (
                "simd2-scheduled-12gp-9fp-2sums",
                SortOptions {
                    order: Order::Scheduled(RegisterBudget {
                        general: NonZeroUsize::new(12).unwrap(),
                        floating: NonZeroUsize::new(9).unwrap(),
                    }),
                    accumulators: NonZeroUsize::new(2).unwrap(),
                    comparisons: ComparisonMode::Simd2,
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
        ] {
            let plan = plan(&model, options)?;
            let assembly = emit_assembly(&context, &plan, triple)?;
            let path = output.join(format!("{triple}-{name}.s"));
            File::create_new(&path)?.write_all(assembly.as_bytes())?;
            println!("{}", path.display());
            let path = path.with_extension("plan.txt");
            let mut dump = File::create_new(&path)?;
            writeln!(dump, "{options:?}\nsums={}\nchunks:", plan.sum_count())?;
            for chunk in plan.chunks() {
                writeln!(dump, "{chunk:?}")?;
            }
            writeln!(dump, "predicates:")?;
            for (id, predicate) in plan.predicates().iter().enumerate() {
                writeln!(dump, "{id}: {predicate:?}")?;
            }
            writeln!(dump, "commands:")?;
            for (id, action) in plan.actions().iter().enumerate() {
                writeln!(dump, "{id}: {action:?}")?;
            }
            println!("{}", path.display());
        }
    }
    Ok(())
}
