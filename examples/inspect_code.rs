//! Writes separately emitted target assembly, not MCJIT byte disassembly.
mod support;

use inkwell::context::Context;
use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};
use support::Workload;
use tigerboost::jit::{EmitOptions, emit_assembly, emit_assembly_with};
use tigerboost::sort_policy::plan;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: inspect_code <fresh-output-directory> [workload]")?,
    );
    let choice = match arguments.next() {
        Some(name) => Workload::parse(name.to_str().ok_or("workload name must be UTF-8")?)?,
        None => Workload::Distinct(100, 100),
    };
    if arguments.next().is_some() {
        return Err("usage: inspect_code <fresh-output-directory> [workload]".into());
    }
    fs::create_dir(&output)?;
    let (model, _) = support::workload(choice);
    let context = Context::create();
    let plan = plan(&model);
    for (triple, name, avx2) in [
        ("aarch64-apple-darwin", "kernel", false),
        ("x86_64-unknown-linux-gnu", "kernel", false),
        ("x86_64-unknown-linux-gnu", "kernel-avx2", true),
    ] {
        let assembly = if avx2 {
            emit_assembly_with(&context, &plan, triple, EmitOptions { avx2: true })?
        } else {
            emit_assembly(&context, &plan, triple)?
        };
        let path = output.join(format!("{triple}-{name}.s"));
        File::create_new(&path)?.write_all(assembly.as_bytes())?;
        println!("{}", path.display());
        let path = path.with_extension("plan.txt");
        let mut dump = File::create_new(&path)?;
        writeln!(dump, "sums={}\nchunks:", plan.sum_count())?;
        for chunk in plan.chunks() {
            writeln!(dump, "{chunk:?}")?;
        }
        writeln!(dump, "predicates:")?;
        for (id, predicate) in plan.predicates().iter().enumerate() {
            writeln!(dump, "{id}: {predicate:?}")?;
        }
        writeln!(dump, "actions:")?;
        for (id, action) in plan.actions().iter().enumerate() {
            writeln!(dump, "{id}: {action:?}")?;
        }
        println!("{}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tigerboost::sort_policy::Action;

    #[test]
    fn every_300_tree_chunk_keeps_thresholds_and_leaves_within_32_kib() {
        let context = Context::create();
        let (model, _) = support::workload(Workload::Distinct(300, 100));
        let plan = plan(&model);
        let assembly = emit_assembly(&context, &plan, "aarch64-apple-darwin").unwrap();
        let constants: Vec<_> = assembly
            .split_once("l_constants:\n")
            .unwrap()
            .1
            .lines()
            .map(str::trim)
            .take_while(|line| line.starts_with(".quad"))
            .map(|line| {
                u64::from_str_radix(
                    line.split_whitespace()
                        .nth(1)
                        .unwrap()
                        .strip_prefix("0x")
                        .unwrap(),
                    16,
                )
                .unwrap()
            })
            .collect();
        let mut slot = 0usize;
        assert_eq!(plan.chunks().len(), 38);
        for chunk in plan.chunks() {
            let first = slot;
            let actions = &plan.actions()[chunk.actions.clone()];
            for action in actions {
                let Action::EvaluateTree { predicates, .. } = action else {
                    unreachable!()
                };
                for pair in plan.predicates()[predicates.clone()].chunks_exact(2) {
                    let mut expected = [pair[0].threshold.to_bits(), pair[1].threshold.to_bits()];
                    let mut actual = [constants[slot], constants[slot + 1]];
                    expected.sort_unstable();
                    actual.sort_unstable();
                    assert_eq!(actual, expected);
                    slot += 2;
                }
            }
            let padded = first + (slot - first).next_multiple_of(8);
            assert!(constants[slot..padded].iter().all(|&bits| bits == 0));
            slot = padded;
            for action in actions {
                let Action::EvaluateTree { tree, .. } = action else {
                    unreachable!()
                };
                let tree = &model.trees()[tree.0];
                let mut expected: Vec<_> = tree.leafs.iter().map(|leaf| leaf.to_bits()).collect();
                let mut actual = constants[slot..slot + tree.leafs.len()].to_vec();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected);
                slot += tree.leafs.len();
            }
            assert_eq!(
                (slot - first) * size_of::<f64>(),
                actions.len() * (6 + 64) * 8
            );
            assert!((slot - first) * size_of::<f64>() <= 32 * 1024);
        }
        assert_eq!(slot, constants.len());
    }
}
