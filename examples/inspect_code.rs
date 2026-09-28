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
use tigerboost::jit::{EmitOptions, emit_assembly_with};
use tigerboost::sort_policy::{SortOptions, plan};

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
        for (name, chunk_trees) in [("tree-major-8-2sums", 8), ("tree-major-4-2sums", 4)] {
            let options = SortOptions {
                chunk_trees: NonZeroUsize::new(chunk_trees).unwrap(),
                accumulators: NonZeroUsize::new(2).unwrap(),
            };
            let plan = plan(&model, options);
            let avx2_variant = (triple == "x86_64-unknown-linux-gnu")
                .then_some((format!("{name}-avx2"), EmitOptions { avx2: true }));
            for (name, emit_options) in
                std::iter::once((name.to_owned(), EmitOptions::default())).chain(avx2_variant)
            {
                let assembly = emit_assembly_with(&context, &plan, triple, emit_options)?;
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
    }
    Ok(())
}
