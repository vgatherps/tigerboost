//! Writes separately emitted target assembly, not MCJIT byte disassembly.
mod support;

use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::PathBuf,
};
use support::Workload;
use tigerboost::jit::{EmitOptions, emit_assembly, emit_assembly_with, ordered_predicates};
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
    let plan = plan(&model);
    for (triple, name, avx2) in [
        ("aarch64-apple-darwin", "kernel", false),
        ("x86_64-unknown-linux-gnu", "kernel", false),
        ("x86_64-unknown-linux-gnu", "kernel-avx2", true),
    ] {
        let assembly = if avx2 {
            emit_assembly_with(&plan, triple, EmitOptions { avx2: true })?
        } else {
            emit_assembly(&plan, triple)?
        };
        let path = output.join(format!("{triple}-{name}.s"));
        File::create_new(&path)?.write_all(assembly.as_bytes())?;
        println!("{}", path.display());
    }
    let path = output.join("plan.txt");
    let mut dump = File::create_new(&path)?;
    writeln!(dump, "sums={}\nchunks:", plan.sum_count())?;
    for chunk in plan.chunks() {
        writeln!(dump, "{chunk:?}")?;
    }
    for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
        writeln!(dump, "thresholds in emitted lane order ({triple}):")?;
        for (id, predicate) in ordered_predicates(&plan, triple).iter().enumerate() {
            writeln!(dump, "{id}: {predicate:?}")?;
        }
    }
    writeln!(dump, "actions:")?;
    for (id, action) in plan.actions().iter().enumerate() {
        writeln!(dump, "{id}: {action:?}")?;
    }
    println!("{}", path.display());
    Ok(())
}
