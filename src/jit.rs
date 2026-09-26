use crate::parsed_tree::{NanMode, RowLengthError};
use crate::sort_policy::{Action, Comparison, Plan};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::VectorType;
use inkwell::values::{BasicValue, FastMathFlags, FloatValue};
use inkwell::{AddressSpace, FloatPredicate, OptimizationLevel};
use std::error::Error;

type PredictFn = unsafe extern "C" fn(*const f64) -> f64;
type CompileResult<T> = Result<T, Box<dyn Error>>;
const BACKEND_OPTIMIZATION: OptimizationLevel = OptimizationLevel::Less;

pub struct CompiledModel<'ctx> {
    // Retains the engine, including executable code and private constant globals.
    function: JitFunction<'ctx, PredictFn>,
    feature_count: usize,
}

impl CompiledModel<'_> {
    pub fn predict(&self, row: &[f64]) -> Result<f64, RowLengthError> {
        if row.len() < self.feature_count {
            return Err(RowLengthError {
                expected: self.feature_count,
                actual: row.len(),
            });
        }
        // SAFETY: row covers all validated feature indices, is aligned and stays
        // immutable during the call. The retained function owns code and tables.
        Ok(unsafe { self.function.call(row.as_ptr()) })
    }
}

pub fn compile<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
) -> CompileResult<CompiledModel<'ctx>> {
    Target::initialize_native(&InitializationConfig::default())?;
    let triple = TargetMachine::get_default_triple();
    let (module, _) = prepare_module(context, plan, &triple)?;
    let engine = module.create_jit_execution_engine(BACKEND_OPTIMIZATION)?;
    // SAFETY: emit_plan builds exactly C ABI double(const double*).
    let function = unsafe { engine.get_function::<PredictFn>("predict")? };
    Ok(CompiledModel {
        function,
        feature_count: plan.model().feature_modes().len(),
    })
}

/// Separate target emission for inspection, not disassembly of MCJIT's executable bytes.
pub fn emit_assembly(
    context: &Context,
    plan: &Plan<'_>,
    target_triple: &str,
) -> CompileResult<String> {
    Target::initialize_all(&InitializationConfig::default());
    let triple = TargetTriple::create(target_triple);
    let (module, machine) = prepare_module(context, plan, &triple)?;
    let buffer = machine.write_to_memory_buffer(&module, FileType::Assembly)?;
    Ok(String::from_utf8(buffer.as_slice().to_vec())?)
}

fn prepare_module<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    triple: &TargetTriple,
) -> CompileResult<(Module<'ctx>, TargetMachine)> {
    let target = Target::from_triple(triple)?;
    let machine = target
        .create_target_machine(
            triple,
            "generic",
            "",
            BACKEND_OPTIMIZATION,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or("could not create target machine")?;
    let module = emit_plan(context, plan)?;
    module.set_triple(triple);
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.verify()?;
    module.run_passes(
        "function(early-cse)",
        &machine,
        PassBuilderOptions::create(),
    )?;
    module.verify()?;
    Ok((module, machine))
}

fn emit_plan<'ctx>(context: &'ctx Context, plan: &Plan<'_>) -> CompileResult<Module<'ctx>> {
    let model = plan.model();
    let module = context.create_module("tigerboost");
    let builder = context.create_builder();
    let f64_type = context.f64_type();
    let index_type = context.i64_type();
    let zero = index_type.const_zero();
    let function = module.add_function(
        "predict",
        f64_type.fn_type(&[context.ptr_type(AddressSpace::default()).into()], false),
        None,
    );
    function.set_call_conventions(0);
    let attribute = |location, name, value| {
        let kind = Attribute::get_named_enum_kind_id(name);
        assert_ne!(kind, 0, "missing LLVM attribute {name}");
        function.add_attribute(location, context.create_enum_attribute(kind, value));
    };
    attribute(AttributeLoc::Param(0), "readonly", 0);
    attribute(AttributeLoc::Param(0), "align", 8);
    let row_bytes = model.feature_modes().len() * size_of::<f64>();
    if row_bytes != 0 {
        attribute(AttributeLoc::Param(0), "dereferenceable", row_bytes as u64);
    }
    // LLVM 21: Ref=1 in each of four two-bit memory locations, including globals.
    attribute(AttributeLoc::Function, "memory", 0b01_01_01_01);
    for (name, value) in [("target-cpu", "generic"), ("target-features", "")] {
        function.add_attribute(
            AttributeLoc::Function,
            context.create_string_attribute(name, value),
        );
    }
    builder.position_at_end(context.append_basic_block(function, "entry"));
    let row = function.get_first_param().unwrap().into_pointer_value();
    let mut indices = vec![None; model.trees().len()];
    let mut loaded_features = Vec::new();
    let mut sums = vec![f64_type.const_zero(); plan.sum_count()];
    for action in plan.actions() {
        match action {
            Action::Admit(tree) => indices[tree.0] = Some(zero),
            Action::Load { value: id, feature } => {
                // SAFETY: Trees validates the feature index and row byte span;
                // CompiledModel::predict checks that the caller supplies that span.
                let address = unsafe {
                    builder.build_in_bounds_gep(
                        f64_type,
                        row,
                        &[index_type.const_int(feature.0 as u64, false)],
                        "feature_ptr",
                    )?
                };
                let loaded = builder.build_load(f64_type, address, "feature")?;
                loaded.as_instruction_value().unwrap().set_alignment(8)?;
                let mut value = loaded.into_float_value();
                let mode = model.feature_modes()[feature.0];
                if mode != NanMode::AsIs {
                    let is_nan =
                        builder.build_float_compare(FloatPredicate::UNO, value, value, "is_nan")?;
                    let replacement = if mode == NanMode::AsFalse {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    };
                    value = builder
                        .build_select(
                            is_nan,
                            f64_type.const_float(replacement),
                            value,
                            "nan_value",
                        )?
                        .into_float_value();
                }
                assert_eq!(id.0, loaded_features.len(), "feature value loaded twice");
                loaded_features.push(Some(value));
            }
            Action::Compare {
                feature,
                comparisons,
            } => {
                let feature =
                    loaded_features[feature.0].expect("comparison requires a live feature value");
                let threshold = |comparison: &Comparison| {
                    f64_type.const_float(plan.predicates()[comparison.predicate.0].threshold)
                };
                let bits = match comparisons.as_slice() {
                    [single] => vec![builder.build_float_compare(
                        FloatPredicate::OGT,
                        feature,
                        threshold(single),
                        "split",
                    )?],
                    [first, second] => {
                        let mut features = f64_type.vec_type(2).get_undef();
                        for lane in 0..2 {
                            features = builder.build_insert_element(
                                features,
                                feature,
                                context.i32_type().const_int(lane, false),
                                "splat",
                            )?;
                        }
                        let thresholds =
                            VectorType::const_vector(&[threshold(first), threshold(second)]);
                        let result = builder.build_float_compare(
                            FloatPredicate::OGT,
                            features,
                            thresholds,
                            "splits",
                        )?;
                        let mut bits = Vec::new();
                        for lane in 0..2 {
                            bits.push(
                                builder
                                    .build_extract_element(
                                        result,
                                        context.i32_type().const_int(lane, false),
                                        "lane",
                                    )?
                                    .into_int_value(),
                            );
                        }
                        bits
                    }
                    _ => unreachable!("comparison pack must contain one or two predicates"),
                };
                for (comparison, bit) in comparisons.iter().zip(bits) {
                    let bit = builder.build_int_z_extend(bit, index_type, "bit")?;
                    for consumer in &comparison.consumers {
                        let level_bit = builder.build_left_shift(
                            bit,
                            index_type.const_int(consumer.level as u64, false),
                            "level_bit",
                        )?;
                        let index =
                            indices[consumer.tree.0].expect("comparison requires an active tree");
                        indices[consumer.tree.0] =
                            Some(builder.build_or(index, level_bit, "leaf_index")?);
                    }
                }
            }
            Action::Release(feature) => {
                loaded_features[feature.0]
                    .take()
                    .expect("release requires a live feature value");
            }
            Action::Retire { tree: tree_id, sum } => {
                let index = indices[tree_id.0]
                    .take()
                    .expect("retirement requires an active tree");
                let tree = &model.trees()[tree_id.0];
                let leaf_count = u32::try_from(tree.leafs.len())
                    .map_err(|_| "leaf table exceeds LLVM array length limit")?;
                let leaf_type = f64_type.array_type(leaf_count);
                let leaves = module.add_global(leaf_type, None, &format!("leaves_{}", tree_id.0));
                leaves.set_linkage(Linkage::Private);
                leaves.set_constant(true);
                leaves.set_alignment(8);
                let values: Vec<_> = tree
                    .leafs
                    .iter()
                    .map(|&leaf| f64_type.const_float(leaf))
                    .collect();
                leaves.set_initializer(&f64_type.const_array(&values));
                // SAFETY: exactly depth bits form index, and Trees requires 2^depth leaves.
                let address = unsafe {
                    builder.build_in_bounds_gep(
                        leaf_type,
                        leaves.as_pointer_value(),
                        &[zero, index],
                        "leaf_ptr",
                    )?
                };
                let leaf = builder.build_load(f64_type, address, "leaf")?;
                leaf.as_instruction_value().unwrap().set_alignment(8)?;
                let accumulator = &mut sums[sum.0];
                *accumulator = reassociate(builder.build_float_add(
                    *accumulator,
                    leaf.into_float_value(),
                    "sum",
                )?)?;
            }
            Action::AddSums {
                destination,
                source,
            } => {
                sums[destination.0] = reassociate(builder.build_float_add(
                    sums[destination.0],
                    sums[source.0],
                    "reduced",
                )?)?;
            }
            Action::ReturnScaled(sum) => {
                let scaled = reassociate(builder.build_float_mul(
                    sums[sum.0],
                    f64_type.const_float(model.scale()),
                    "scaled",
                )?)?;
                let result = reassociate(builder.build_float_add(
                    scaled,
                    f64_type.const_float(model.bias()),
                    "biased",
                )?)?;
                builder.build_return(Some(&result))?;
            }
            Action::ReturnBias => {
                builder.build_return(Some(&f64_type.const_float(model.bias())))?;
            }
        }
    }
    Ok(module)
}

fn reassociate(value: FloatValue<'_>) -> CompileResult<FloatValue<'_>> {
    if let Some(instruction) = value.as_instruction_value() {
        instruction.set_fast_math_flags(FastMathFlags::AllowReassoc)?;
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsed_tree::{Discriminator, FeatureIndex, SingleTree, Trees};
    use crate::sort_policy::{ComparisonMode, Order, SortOptions, plan};
    use std::num::NonZeroUsize;

    #[test]
    fn emits_predict_for_aarch64_and_x86_64() {
        let context = Context::create();
        let model = Trees::new(
            vec![SingleTree {
                leafs: vec![1.0, 2.0],
                selections: vec![Discriminator {
                    feature_idx: FeatureIndex(0),
                    threshold: 0.0,
                }],
            }],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap();
        for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            let assembly =
                emit_assembly(&context, &plan(&model, SortOptions::default()), triple).unwrap();
            assert!(assembly.contains("predict"), "{triple}: {assembly}");
        }
    }

    #[test]
    fn rolling_shares_loads_and_comparisons_before_llvm_passes() {
        let context = Context::create();
        let split = Discriminator {
            feature_idx: FeatureIndex(0),
            threshold: 0.0,
        };
        let model = Trees::new(
            vec![
                SingleTree {
                    leafs: vec![0.0, 1.0, 2.0, 3.0],
                    selections: vec![split.clone(), split.clone()],
                },
                SingleTree {
                    leafs: vec![0.0, 1.0, 2.0, 3.0],
                    selections: vec![
                        split,
                        Discriminator {
                            feature_idx: FeatureIndex(0),
                            threshold: 1.0,
                        },
                    ],
                },
            ],
            vec![NanMode::AsIs],
            1.0,
            0.0,
        )
        .unwrap();
        for (order, comparisons, loads, scalars, pairs) in [
            (Order::TreeOrder, ComparisonMode::Scalar, 6, 4, 0),
            (Order::TreeOrder, ComparisonMode::Simd2, 6, 4, 0),
            (
                Order::Rolling(NonZeroUsize::new(2).unwrap()),
                ComparisonMode::Scalar,
                3,
                2,
                0,
            ),
            (
                Order::Rolling(NonZeroUsize::new(2).unwrap()),
                ComparisonMode::Simd2,
                3,
                0,
                1,
            ),
        ] {
            let module = emit_plan(
                &context,
                &plan(
                    &model,
                    SortOptions {
                        order,
                        comparisons,
                        ..SortOptions::default()
                    },
                ),
            )
            .unwrap();
            module.verify().unwrap();
            let ir = module.print_to_string().to_string();
            assert_eq!(ir.matches("load double").count(), loads);
            assert_eq!(ir.matches("fcmp ogt double").count(), scalars);
            assert_eq!(ir.matches("fcmp ogt <2 x double>").count(), pairs);
            assert_eq!(ir.matches("extractelement <2 x i1>").count(), pairs * 2);
        }
    }

    #[test]
    fn ir_has_readonly_rows_constant_leaves_and_only_reassociation() {
        let context = Context::create();
        let model = Trees::new(
            vec![SingleTree {
                leafs: vec![1.0, 2.0],
                selections: vec![Discriminator {
                    feature_idx: FeatureIndex(0),
                    threshold: 0.0,
                }],
            }],
            vec![NanMode::AsIs],
            2.0,
            3.0,
        )
        .unwrap();
        let module = emit_plan(&context, &plan(&model, SortOptions::default())).unwrap();
        module.verify().unwrap();
        let ir = module.print_to_string().to_string();
        for expected in [
            "private constant [2 x double]",
            "readonly align 8 dereferenceable(8)",
            "memory(read)",
            "fcmp ogt double",
            "fadd reassoc double",
            "fmul reassoc double",
        ] {
            assert!(ir.contains(expected), "{expected}: {ir}");
        }
        for forbidden in [
            "invariant.load",
            "store ",
            "nnan",
            "ninf",
            "nsz",
            "fast double",
        ] {
            assert!(!ir.contains(forbidden), "{forbidden}: {ir}");
        }
    }
}
