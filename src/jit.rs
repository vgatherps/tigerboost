use crate::parsed_tree::{NanMode, RowLengthError, SingleTree};
use crate::sort_policy::{Action, Comparison, Plan};
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::context::Context;
use inkwell::execution_engine::JitFunction;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, FastMathFlags, FloatValue, PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, OptimizationLevel};
use std::error::Error;

type PredictFn = unsafe extern "C" fn(*const f64) -> f64;
type CompileResult<T> = Result<T, Box<dyn Error>>;
const BACKEND_OPTIMIZATION: OptimizationLevel = OptimizationLevel::Less;
// LLVM 21 preserve_nonecc; only used between matching continuation signatures.
const PRESERVE_NONE_CALL_CONVENTION: u32 = 21;

#[derive(Clone, Copy, Debug, Default)]
pub struct EmitOptions {
    pub avx2: bool,
}

impl EmitOptions {
    pub fn from_host() -> Self {
        Self {
            avx2: TargetMachine::get_host_cpu_features()
                .to_string()
                .split(',')
                .any(|feature| feature == "+avx2"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TreePath {
    Lanes2,
    Avx2,
}

impl TreePath {
    fn target_cpu(self) -> &'static str {
        match self {
            Self::Lanes2 => "generic",
            Self::Avx2 => "x86-64-v3",
        }
    }
}

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
    compile_with(context, plan, EmitOptions::default())
}

pub fn compile_with<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    options: EmitOptions,
) -> CompileResult<CompiledModel<'ctx>> {
    Target::initialize_native(&InitializationConfig::default())?;
    let triple = TargetMachine::get_default_triple();
    let (module, _) = prepare_module(context, plan, &triple, options)?;
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
    emit_assembly_with(context, plan, target_triple, EmitOptions::default())
}

/// Separate target emission with explicit target options.
pub fn emit_assembly_with(
    context: &Context,
    plan: &Plan<'_>,
    target_triple: &str,
    options: EmitOptions,
) -> CompileResult<String> {
    Target::initialize_all(&InitializationConfig::default());
    let triple = TargetTriple::create(target_triple);
    let (module, machine) = prepare_module(context, plan, &triple, options)?;
    let buffer = machine.write_to_memory_buffer(&module, FileType::Assembly)?;
    Ok(String::from_utf8(buffer.as_slice().to_vec())?)
}

fn prepare_module<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    triple: &TargetTriple,
    options: EmitOptions,
) -> CompileResult<(Module<'ctx>, TargetMachine)> {
    let tree_path = if options.avx2 && triple.as_str().to_bytes().starts_with(b"x86_64-") {
        TreePath::Avx2
    } else {
        TreePath::Lanes2
    };
    let target = Target::from_triple(triple)?;
    let machine = target
        .create_target_machine(
            triple,
            tree_path.target_cpu(),
            "",
            BACKEND_OPTIMIZATION,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or("could not create target machine")?;
    let module = emit_plan(context, plan, triple, tree_path)?;
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

fn emit_plan<'ctx>(
    context: &'ctx Context,
    plan: &Plan<'_>,
    triple: &TargetTriple,
    tree_path: TreePath,
) -> CompileResult<Module<'ctx>> {
    let model = plan.model();
    let module = context.create_module("tigerboost");
    module.set_triple(triple);
    let architecture = triple.as_str().to_bytes();
    let arm64 = architecture.starts_with(b"aarch64") || architecture.starts_with(b"arm64");
    let builder = context.create_builder();
    let f64_type = context.f64_type();
    let index_type = context.i64_type();
    let zero = index_type.const_zero();
    let mut predicate_slots = vec![None; plan.predicates().len()];
    let mut evaluated_trees = vec![false; model.trees().len()];
    let mut next_predicate_slot = 0;
    for action in plan.actions() {
        if let Action::Compare { comparisons, .. } = action {
            for comparison in comparisons {
                let slot = &mut predicate_slots[comparison.predicate.0];
                if slot.is_none() {
                    *slot = Some(next_predicate_slot);
                    next_predicate_slot += 1;
                }
            }
        }
        if let Action::EvaluateTree {
            tree, predicates, ..
        } = action
        {
            evaluated_trees[tree.0] = true;
            for predicate in predicates {
                let slot = &mut predicate_slots[predicate.0];
                assert!(slot.is_none(), "tree-major predicate used twice");
                *slot = Some(next_predicate_slot);
                next_predicate_slot += 1;
            }
        }
    }
    assert_eq!(
        next_predicate_slot,
        plan.predicates().len(),
        "predicate is never compared"
    );
    let predicate_slots: Vec<_> = predicate_slots.into_iter().map(Option::unwrap).collect();
    let leaf_start = plan
        .predicates()
        .len()
        .checked_next_multiple_of(8)
        .ok_or("constant block exceeds addressable entry count")?;
    let mut entry_count = leaf_start;
    let mut leaf_base = Vec::with_capacity(model.trees().len());
    for (tree, &evaluated) in model.trees().iter().zip(&evaluated_trees) {
        leaf_base.push(entry_count);
        let length = tree
            .leafs
            .len()
            .checked_mul(
                if tree_path == TreePath::Lanes2 && evaluated && tree.selections.len() % 2 != 0 {
                    2
                } else {
                    1
                },
            )
            .ok_or("tree-major leaf block exceeds addressable entry count")?;
        entry_count = entry_count
            .checked_add(length)
            .ok_or("constant block exceeds addressable entry count")?;
    }
    entry_count = entry_count.max(1);
    let array_length =
        u32::try_from(entry_count).map_err(|_| "constant block exceeds LLVM array length limit")?;
    let constant_bytes = entry_count
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| isize::try_from(bytes).ok())
        .ok_or("constant block exceeds addressable byte count")? as u64;
    let constant_type = f64_type.array_type(array_length);
    let constants = module.add_global(constant_type, None, "constants");
    constants.set_linkage(Linkage::Private);
    constants.set_constant(true);
    constants.set_alignment(64);
    let mut values = vec![f64_type.const_zero(); plan.predicates().len()];
    for (predicate, &slot) in plan.predicates().iter().zip(&predicate_slots) {
        values[slot] = f64_type.const_float(predicate.threshold);
    }
    values.resize(leaf_start, f64_type.const_zero());
    let mut leaf_bias = vec![0; model.trees().len()];
    for (id, tree) in model.trees().iter().enumerate() {
        if tree_path == TreePath::Lanes2 && evaluated_trees[id] {
            let (bias, leaves) = permuted_leaves(tree);
            leaf_bias[id] = bias;
            values.extend(leaves.into_iter().map(|leaf| f64_type.const_float(leaf)));
        } else {
            values.extend(tree.leafs.iter().map(|&leaf| f64_type.const_float(leaf)));
        }
    }
    values.resize(entry_count, f64_type.const_zero());
    constants.set_initializer(&f64_type.const_array(&values));
    let chunks = plan.chunks();
    let chunk_first_slot: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            plan.actions()[chunk.actions.clone()]
                .iter()
                .filter_map(|action| match action {
                    Action::Compare { comparisons, .. } => comparisons
                        .iter()
                        .map(|comparison| predicate_slots[comparison.predicate.0])
                        .min(),
                    Action::EvaluateTree { predicates, .. } => predicates
                        .iter()
                        .map(|predicate| predicate_slots[predicate.0])
                        .min(),
                    _ => None,
                })
                .min()
                .unwrap_or(0)
        })
        .collect();
    let index_slots = chunks
        .iter()
        .map(|c| c.incoming.indices.len())
        .max()
        .unwrap();
    let feature_slots = chunks
        .iter()
        .map(|c| c.incoming.features.len())
        .max()
        .unwrap();
    let pointer_count = 2 + usize::from(arm64);
    let sum_offset = pointer_count + index_slots;
    let feature_offset = sum_offset + plan.sum_count();
    let row_type = context.ptr_type(AddressSpace::default());
    let mut parameter_types = vec![row_type.into(); pointer_count];
    parameter_types.resize(sum_offset, index_type.into());
    parameter_types.resize(feature_offset + feature_slots, f64_type.into());
    let continuation_type = f64_type.fn_type(&parameter_types, false);
    let wrapper = module.add_function("predict", f64_type.fn_type(&[row_type.into()], false), None);
    let functions: Vec<_> = (0..chunks.len())
        .map(|n| {
            module.add_function(
                &format!("continuation{n}"),
                continuation_type,
                Some(Linkage::Internal),
            )
        })
        .collect();
    for (number, function) in functions.iter().copied().chain(Some(wrapper)).enumerate() {
        function.set_call_conventions(if function == wrapper {
            0
        } else {
            PRESERVE_NONE_CALL_CONVENTION
        });
        let attribute = |location, name, value| {
            let kind = Attribute::get_named_enum_kind_id(name);
            assert_ne!(kind, 0, "missing LLVM attribute {name}");
            function.add_attribute(location, context.create_enum_attribute(kind, value));
        };
        attribute(AttributeLoc::Function, "noinline", 0);
        attribute(AttributeLoc::Param(0), "noalias", 0);
        attribute(AttributeLoc::Param(0), "readonly", 0);
        attribute(AttributeLoc::Param(0), "align", 8);
        let row_bytes = model.feature_modes().len() * size_of::<f64>();
        if row_bytes != 0 {
            attribute(AttributeLoc::Param(0), "dereferenceable", row_bytes as u64);
        }
        if function != wrapper {
            attribute(AttributeLoc::Param(1), "noalias", 0);
            attribute(AttributeLoc::Param(1), "readonly", 0);
            attribute(AttributeLoc::Param(1), "align", 64);
            attribute(AttributeLoc::Param(1), "dereferenceable", constant_bytes);
            if arm64 {
                let remaining_bytes = constant_bytes
                    .checked_sub(
                        (chunk_first_slot[number] as u64)
                            .checked_mul(size_of::<f64>() as u64)
                            .ok_or("chunk base exceeds addressable byte count")?,
                    )
                    .ok_or("chunk base exceeds constant block")?;
                attribute(AttributeLoc::Param(2), "noalias", 0);
                attribute(AttributeLoc::Param(2), "readonly", 0);
                attribute(AttributeLoc::Param(2), "align", 8);
                attribute(AttributeLoc::Param(2), "dereferenceable", remaining_bytes);
            }
        }
        // LLVM 21: Ref=1 in each of four two-bit memory locations, including globals.
        attribute(AttributeLoc::Function, "memory", 0b01_01_01_01);
        for (name, value) in [
            ("target-cpu", tree_path.target_cpu()),
            ("target-features", ""),
        ] {
            function.add_attribute(
                AttributeLoc::Function,
                context.create_string_attribute(name, value),
            );
        }
    }
    let arguments = |row: PointerValue<'ctx>,
                     constants: PointerValue<'ctx>,
                     chunk: usize|
     -> CompileResult<Vec<BasicMetadataValueEnum<'ctx>>> {
        let mut args = vec![row.into(), constants.into()];
        if arm64 {
            // SAFETY: the base is a threshold entry, or entry zero for a chunk without tests.
            let base = unsafe {
                builder.build_in_bounds_gep(
                    f64_type,
                    constants,
                    &[index_type.const_int(chunk_first_slot[chunk] as u64, false)],
                    "chunk_base",
                )?
            };
            args.push(base.into());
        }
        args.resize(sum_offset, index_type.get_undef().into());
        args.resize(feature_offset + feature_slots, f64_type.get_undef().into());
        Ok(args)
    };
    builder.position_at_end(context.append_basic_block(wrapper, "entry"));
    // The first chunk has no carried values and initializes its sums to zero.
    let call = builder.build_call(
        functions[0],
        &arguments(
            wrapper.get_first_param().unwrap().into_pointer_value(),
            constants.as_pointer_value(),
            0,
        )?,
        "prediction",
    )?;
    call.set_call_convention(PRESERVE_NONE_CALL_CONVENTION);
    builder.build_return(Some(&call.try_as_basic_value().unwrap_basic()))?;
    let mut indices: Vec<Option<inkwell::values::IntValue<'ctx>>> = vec![None; model.trees().len()];
    let mut loaded_features: Vec<Option<FloatValue<'ctx>>> = Vec::new();
    let vsli = if arm64
        && plan.actions().iter().any(|action| {
            matches!(action,
                Action::EvaluateTree { predicates, .. } if predicates.len() > 2
            )
        }) {
        Some(
            Intrinsic::find("llvm.aarch64.neon.vsli")
                .unwrap()
                .get_declaration(&module, &[index_type.vec_type(2).into()])
                .unwrap(),
        )
    } else {
        None
    };
    let avx2_intrinsics = (tree_path == TreePath::Avx2).then(|| {
        let declaration = |name| {
            Intrinsic::find(name)
                .unwrap()
                .get_declaration(&module, &[])
                .unwrap()
        };
        (
            (
                declaration("llvm.x86.avx.cmp.pd.256"),
                declaration("llvm.x86.avx.movmsk.pd.256"),
            ),
            (
                declaration("llvm.x86.sse2.cmp.pd"),
                declaration("llvm.x86.sse2.movmsk.pd"),
            ),
        )
    });
    for (chunk_number, chunk) in chunks.iter().enumerate() {
        let function = functions[chunk_number];
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let row = function.get_first_param().unwrap().into_pointer_value();
        let constants = function.get_nth_param(1).unwrap().into_pointer_value();
        let chunk_base = arm64.then(|| function.get_nth_param(2).unwrap().into_pointer_value());
        indices.fill(None);
        loaded_features.fill(None);
        for (slot, tree) in chunk.incoming.indices.iter().enumerate() {
            if let Some(tree) = tree {
                indices[tree.0] = Some(
                    function
                        .get_nth_param((pointer_count + slot) as u32)
                        .unwrap()
                        .into_int_value(),
                );
            }
        }
        for (slot, value) in chunk.incoming.features.iter().enumerate() {
            if let Some(value) = value {
                loaded_features[value.0] = Some(
                    function
                        .get_nth_param((feature_offset + slot) as u32)
                        .unwrap()
                        .into_float_value(),
                );
            }
        }
        let mut sums: Vec<_> = (0..plan.sum_count())
            .map(|sum| {
                if chunk_number == 0 {
                    f64_type.const_zero()
                } else {
                    function
                        .get_nth_param((sum_offset + sum) as u32)
                        .unwrap()
                        .into_float_value()
                }
            })
            .collect();
        for action in &plan.actions()[chunk.actions.clone()] {
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
                        let is_nan = builder.build_float_compare(
                            FloatPredicate::UNO,
                            value,
                            value,
                            "is_nan",
                        )?;
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
                    let feature = loaded_features[feature.0]
                        .expect("comparison requires a live feature value");
                    let threshold = |comparison: &Comparison| -> CompileResult<FloatValue<'ctx>> {
                        let slot = predicate_slots[comparison.predicate.0];
                        // SAFETY: every slot indexes the threshold prefix of constants.
                        let address = unsafe {
                            if let Some(base) = chunk_base {
                                let offset = slot
                                    .checked_sub(chunk_first_slot[chunk_number])
                                    .ok_or("threshold slot precedes chunk base")?;
                                builder.build_in_bounds_gep(
                                    f64_type,
                                    base,
                                    &[index_type.const_int(offset as u64, false)],
                                    "threshold_ptr",
                                )?
                            } else {
                                builder.build_in_bounds_gep(
                                    constant_type,
                                    constants,
                                    &[zero, index_type.const_int(slot as u64, false)],
                                    "threshold_ptr",
                                )?
                            }
                        };
                        let loaded = builder.build_load(f64_type, address, "threshold")?;
                        loaded.as_instruction_value().unwrap().set_alignment(8)?;
                        Ok(loaded.into_float_value())
                    };
                    let bits = match comparisons.as_slice() {
                        [single] => vec![builder.build_float_compare(
                            FloatPredicate::OGT,
                            feature,
                            threshold(single)?,
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
                            let mut thresholds = f64_type.vec_type(2).get_undef();
                            for (lane, comparison) in [first, second].into_iter().enumerate() {
                                thresholds = builder.build_insert_element(
                                    thresholds,
                                    threshold(comparison)?,
                                    context.i32_type().const_int(lane as u64, false),
                                    "thresholds",
                                )?;
                            }
                            let result = builder.build_float_compare(
                                FloatPredicate::OGT,
                                features,
                                thresholds,
                                "splits",
                            )?;
                            let result = builder.build_int_s_extend(
                                result,
                                index_type.vec_type(2),
                                "full_masks",
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
                        let scalar = comparisons.len() == 1;
                        let bit = if scalar {
                            builder.build_int_z_extend(bit, index_type, "bit")?
                        } else {
                            bit
                        };
                        for consumer in &comparison.consumers {
                            let level_bit = if scalar {
                                builder.build_left_shift(
                                    bit,
                                    index_type.const_int(consumer.level as u64, false),
                                    "level_bit",
                                )?
                            } else {
                                builder.build_and(
                                    bit,
                                    index_type.const_int(1u64 << consumer.level, false),
                                    "level_bit",
                                )?
                            };
                            let index = indices[consumer.tree.0]
                                .expect("comparison requires an active tree");
                            indices[consumer.tree.0] =
                                Some(builder.build_or(index, level_bit, "leaf_index")?);
                        }
                    }
                }
                Action::EvaluateTree {
                    tree,
                    predicates,
                    sum,
                } if tree_path == TreePath::Avx2 => {
                    let (wide, narrow) = avx2_intrinsics.unwrap();
                    let mut index = zero;
                    for (group_number, group) in predicates.chunks(4).enumerate() {
                        let lanes = group.len();
                        assert!(lanes == 2 || lanes == 4);
                        let first_level = group_number * 4;
                        let slot = predicate_slots[group[0].0];
                        for (lane, predicate) in group.iter().enumerate() {
                            assert_eq!(
                                predicate_slots[predicate.0],
                                slot.checked_add(lane).unwrap()
                            );
                        }
                        // SAFETY: consecutive slots cover this group's thresholds in constants.
                        let address = unsafe {
                            builder.build_in_bounds_gep(
                                constant_type,
                                constants,
                                &[zero, index_type.const_int(slot as u64, false)],
                                "threshold_ptr",
                            )?
                        };
                        let vector_type = f64_type.vec_type(lanes as u32);
                        let thresholds = builder.build_load(vector_type, address, "thresholds")?;
                        thresholds
                            .as_instruction_value()
                            .unwrap()
                            .set_alignment(8)?;
                        let mut features = vector_type.get_undef();
                        let mut first_value = None;
                        for (lane, predicate) in group.iter().enumerate() {
                            let value = if first_level + lane
                                == model.trees()[tree.0].selections.len()
                            {
                                first_value.expect("padding follows a real level")
                            } else {
                                let feature = plan.predicates()[predicate.0].feature_idx;
                                // SAFETY: Trees validates features; predict checks the row span.
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
                                    let is_nan = builder.build_float_compare(
                                        FloatPredicate::UNO,
                                        value,
                                        value,
                                        "is_nan",
                                    )?;
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
                                value
                            };
                            if lane == 0 {
                                first_value = Some(value);
                            }
                            features = builder.build_insert_element(
                                features,
                                value,
                                context.i32_type().const_int(lane as u64, false),
                                "features",
                            )?;
                        }
                        let (compare, movemask) = if lanes == 4 { wide } else { narrow };
                        let masks = builder
                            .build_call(
                                compare,
                                &[
                                    features.into(),
                                    thresholds.into(),
                                    context.i8_type().const_int(30, false).into(),
                                ],
                                "splits",
                            )?
                            .try_as_basic_value()
                            .unwrap_basic();
                        let mask = builder
                            .build_call(movemask, &[masks.into()], "mask")?
                            .try_as_basic_value()
                            .unwrap_basic()
                            .into_int_value();
                        let mask = builder.build_int_z_extend(mask, index_type, "group_index")?;
                        let shifted = if first_level == 0 {
                            mask
                        } else {
                            builder.build_left_shift(
                                mask,
                                index_type.const_int(first_level as u64, false),
                                "upper",
                            )?
                        };
                        index = builder.build_or(index, shifted, "leaf_index")?;
                    }
                    let slot = builder.build_int_nuw_add(
                        index,
                        index_type.const_int(leaf_base[tree.0] as u64, false),
                        "leaf_slot",
                    )?;
                    // SAFETY: depth bits bound index to this tree's original leaf block.
                    let address = unsafe {
                        builder.build_in_bounds_gep(
                            constant_type,
                            constants,
                            &[zero, slot],
                            "leaf_ptr",
                        )?
                    };
                    let leaf = builder.build_load(f64_type, address, "leaf")?;
                    leaf.as_instruction_value().unwrap().set_alignment(8)?;
                    sums[sum.0] = reassociate(builder.build_float_add(
                        sums[sum.0],
                        leaf.into_float_value(),
                        "sum",
                    )?)?;
                }
                Action::EvaluateTree {
                    tree,
                    predicates,
                    sum,
                } => {
                    let pairs = predicates.len() / 2;
                    let mut accumulator: Option<inkwell::values::VectorValue<'ctx>> = None;
                    for (pair_number, pair) in predicates.chunks_exact(2).enumerate() {
                        let slot = predicate_slots[pair[0].0];
                        assert_eq!(predicate_slots[pair[1].0], slot.checked_add(1).unwrap());
                        // SAFETY: adjacent slots cover both thresholds inside constants.
                        let address = unsafe {
                            if let Some(base) = chunk_base {
                                let offset = slot
                                    .checked_sub(chunk_first_slot[chunk_number])
                                    .ok_or("threshold slot precedes chunk base")?;
                                builder.build_in_bounds_gep(
                                    f64_type,
                                    base,
                                    &[index_type.const_int(offset as u64, false)],
                                    "threshold_ptr",
                                )?
                            } else {
                                builder.build_in_bounds_gep(
                                    constant_type,
                                    constants,
                                    &[zero, index_type.const_int(slot as u64, false)],
                                    "threshold_ptr",
                                )?
                            }
                        };
                        let thresholds =
                            builder.build_load(f64_type.vec_type(2), address, "thresholds")?;
                        thresholds
                            .as_instruction_value()
                            .unwrap()
                            .set_alignment(8)?;
                        let mut features = f64_type.vec_type(2).get_undef();
                        let mut first_value = None;
                        for (lane, predicate) in pair.iter().enumerate() {
                            let value = if pair_number * 2 + lane
                                == model.trees()[tree.0].selections.len()
                            {
                                first_value.expect("padding follows a real level")
                            } else {
                                let feature = plan.predicates()[predicate.0].feature_idx;
                                // SAFETY: Trees validates features; predict checks the row span.
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
                                    let is_nan = builder.build_float_compare(
                                        FloatPredicate::UNO,
                                        value,
                                        value,
                                        "is_nan",
                                    )?;
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
                                value
                            };
                            if lane == 0 {
                                first_value = Some(value);
                            }
                            features = builder.build_insert_element(
                                features,
                                value,
                                context.i32_type().const_int(lane as u64, false),
                                "features",
                            )?;
                        }
                        let masks = builder.build_float_compare(
                            FloatPredicate::OGT,
                            features,
                            thresholds.into_vector_value(),
                            "splits",
                        )?;
                        let masks = builder.build_int_s_extend(
                            masks,
                            index_type.vec_type(2),
                            "full_masks",
                        )?;
                        accumulator = Some(if let Some(previous) = accumulator {
                            if let Some(vsli) = vsli {
                                builder
                                    .build_call(
                                        vsli,
                                        &[
                                            previous.into(),
                                            masks.into(),
                                            context
                                                .i32_type()
                                                .const_int(pair_number as u64, false)
                                                .into(),
                                        ],
                                        "acc",
                                    )?
                                    .try_as_basic_value()
                                    .unwrap_basic()
                                    .into_vector_value()
                            } else {
                                let shift = index_type.const_int(pair_number as u64, false);
                                let low = index_type.const_int((1u64 << pair_number) - 1, false);
                                let shifted = builder.build_left_shift(
                                    masks,
                                    inkwell::types::VectorType::const_vector(&[shift, shift]),
                                    "shifted",
                                )?;
                                let kept = builder.build_and(
                                    previous,
                                    inkwell::types::VectorType::const_vector(&[low, low]),
                                    "kept",
                                )?;
                                builder.build_or(shifted, kept, "acc")?
                            }
                        } else {
                            masks
                        });
                    }
                    let accumulator = accumulator.expect("evaluated tree must have splits");
                    let v0 = builder
                        .build_extract_element(accumulator, context.i32_type().const_zero(), "v0")?
                        .into_int_value();
                    let v1 = builder
                        .build_extract_element(
                            accumulator,
                            context.i32_type().const_int(1, false),
                            "v1",
                        )?
                        .into_int_value();
                    let shifted = builder.build_left_shift(
                        v1,
                        index_type.const_int(pairs as u64, false),
                        "upper",
                    )?;
                    let raw = builder.build_int_add(v0, shifted, "raw")?;
                    let base = i64::try_from(leaf_base[tree.0])?
                        .checked_add(leaf_bias[tree.0])
                        .ok_or("tree-major leaf base overflow")?;
                    let slot = builder.build_int_add(
                        raw,
                        index_type.const_int(base as u64, false),
                        "leaf_slot",
                    )?;
                    // SAFETY: the checked permutation bounds the slot to this tree's block.
                    let address = unsafe {
                        builder.build_in_bounds_gep(
                            constant_type,
                            constants,
                            &[zero, slot],
                            "leaf_ptr",
                        )?
                    };
                    let leaf = builder.build_load(f64_type, address, "leaf")?;
                    leaf.as_instruction_value().unwrap().set_alignment(8)?;
                    sums[sum.0] = reassociate(builder.build_float_add(
                        sums[sum.0],
                        leaf.into_float_value(),
                        "sum",
                    )?)?;
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
                    let slot = builder.build_int_nuw_add(
                        index,
                        index_type.const_int(leaf_base[tree_id.0] as u64, false),
                        "leaf_slot",
                    )?;
                    // SAFETY: depth bits bound index to this tree's leaf block in constants.
                    let address = unsafe {
                        builder.build_in_bounds_gep(
                            constant_type,
                            constants,
                            &[zero, slot],
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
        if let Some(next) = chunks.get(chunk_number + 1) {
            let mut args = arguments(row, constants, chunk_number + 1)?;
            for (slot, tree) in next.incoming.indices.iter().enumerate() {
                if let Some(tree) = tree {
                    args[pointer_count + slot] =
                        indices[tree.0].expect("carried index must be live").into();
                }
            }
            for (sum, value) in sums.iter().enumerate() {
                args[sum_offset + sum] = (*value).into();
            }
            for (slot, value) in next.incoming.features.iter().enumerate() {
                if let Some(value) = value {
                    args[feature_offset + slot] = loaded_features[value.0]
                        .expect("carried feature must be live")
                        .into();
                }
            }
            let call = builder.build_call(functions[chunk_number + 1], &args, "continued")?;
            call.set_call_convention(PRESERVE_NONE_CALL_CONVENTION);
            call.set_tail_call_kind(inkwell::values::LLVMTailCallKind::LLVMTailCallKindMustTail);
            builder.build_return(Some(&call.try_as_basic_value().unwrap_basic()))?;
        }
    }
    Ok(module)
}

fn raw_of(pattern: u32, pairs: usize) -> i64 {
    assert!((1..=16).contains(&pairs));
    let mut lanes = [0i64; 2];
    for pair in 0..pairs {
        for (lane, accumulator) in lanes.iter_mut().enumerate() {
            let mask = -i64::from(pattern & (1u32 << (pair * 2 + lane)) != 0);
            *accumulator = if pair == 0 {
                mask
            } else {
                (mask.checked_shl(pair as u32).unwrap()) | (*accumulator & ((1i64 << pair) - 1))
            };
        }
    }
    lanes[0]
        .checked_add(lanes[1].checked_shl(pairs as u32).unwrap())
        .unwrap()
}

fn permuted_leaves(tree: &SingleTree) -> (i64, Vec<f64>) {
    let depth = tree.selections.len();
    let pairs = depth.div_ceil(2);
    let count = 1u32
        .checked_shl(u32::try_from(pairs.checked_mul(2).unwrap()).unwrap())
        .unwrap();
    let bias = (0..count)
        .map(|pattern| raw_of(pattern, pairs))
        .min()
        .unwrap()
        .checked_neg()
        .unwrap();
    let mut leaves = vec![0.0; count as usize];
    let mut seen = vec![false; count as usize];
    for pattern in 0..count {
        let slot = usize::try_from(bias.checked_add(raw_of(pattern, pairs)).unwrap()).unwrap();
        assert!(
            slot < leaves.len() && !seen[slot],
            "tree-major permutation is not a bijection"
        );
        seen[slot] = true;
        if (pattern as usize) < tree.leafs.len() {
            leaves[slot] = tree.leafs[pattern as usize];
        }
    }
    assert!(seen.into_iter().all(|present| present));
    (bias, leaves)
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
    use crate::sort_policy::{ComparisonMode, Order, RegisterBudget, SortOptions, plan};
    use inkwell::values::{AnyValue, FunctionValue, InstructionOpcode};
    use std::num::NonZeroUsize;

    fn tree_major_options(chunk_trees: usize) -> SortOptions {
        SortOptions {
            order: Order::TreeMajor {
                chunk_trees: NonZeroUsize::new(chunk_trees).unwrap(),
            },
            ..SortOptions::default()
        }
    }

    fn tree_major_model(depths: &[usize]) -> Trees {
        Trees::new(
            depths
                .iter()
                .map(|&depth| SingleTree {
                    selections: (0..depth)
                        .map(|level| Discriminator {
                            feature_idx: FeatureIndex(level),
                            threshold: 0.0,
                        })
                        .collect(),
                    leafs: (0..1usize << depth).map(|leaf| leaf as f64).collect(),
                })
                .collect(),
            vec![NanMode::AsIs; 8],
            1.0,
            0.0,
        )
        .unwrap()
    }

    #[test]
    fn raw_of_permutation_is_a_bijection_for_depths_1_to_8() {
        let context = Context::create();
        for depth in 1..=8 {
            let model = tree_major_model(&[depth]);
            let tree = &model.trees()[0];
            let (bias, leaves) = permuted_leaves(tree);
            let pairs = depth.div_ceil(2);
            let mut slots: Vec<_> = (0..1u32 << (pairs * 2))
                .map(|pattern| bias + raw_of(pattern, pairs))
                .collect();
            slots.sort_unstable();
            assert_eq!(slots, (0..leaves.len() as i64).collect::<Vec<_>>());
            let plan = plan(&model, tree_major_options(2)).unwrap();
            let compiled = compile(&context, &plan).unwrap();
            for pattern in 0..1u32 << (pairs * 2) {
                let slot = (bias + raw_of(pattern, pairs)) as usize;
                if (pattern as usize) < tree.leafs.len() {
                    let row: Vec<_> = (0..8)
                        .map(|level| f64::from(pattern & (1 << level) != 0))
                        .collect();
                    assert_eq!(leaves[slot], pattern as f64);
                    assert_eq!(compiled.predict(&row), Ok(leaves[slot]));
                    assert_eq!(compiled.predict(&row), model.predict(&row));
                } else {
                    assert_eq!(leaves[slot], 0.0);
                }
            }
        }
    }

    #[test]
    fn tree_major_plan_shape() {
        use crate::sort_policy::{PredicateId, Slots, SumId, TreeId};
        let depths = [0, 1, 2, 3, 6];
        let model = tree_major_model(&depths);
        let plan = plan(&model, tree_major_options(2)).unwrap();
        assert!(!plan.actions().iter().any(|action| matches!(
            action,
            Action::Load { .. } | Action::Release(_) | Action::Compare { .. }
        )));
        assert_eq!(plan.actions()[0], Action::Admit(TreeId(0)));
        assert_eq!(
            plan.actions()[1],
            Action::Retire {
                tree: TreeId(0),
                sum: SumId(0)
            }
        );
        let mut next_predicate = 0;
        for (tree, &depth) in depths.iter().enumerate().skip(1) {
            let Action::EvaluateTree {
                tree: id,
                predicates,
                sum,
            } = &plan.actions()[tree + 1]
            else {
                panic!("nonconstant tree must have one EvaluateTree action");
            };
            assert_eq!(*id, TreeId(tree));
            assert_eq!(*sum, SumId(tree % 2));
            assert_eq!(predicates.len(), depth.div_ceil(2) * 2);
            for (level, predicate) in predicates.iter().enumerate() {
                assert_eq!(*predicate, PredicateId(next_predicate));
                next_predicate += 1;
                let actual = &plan.predicates()[predicate.0];
                let expected = if level == depth {
                    Discriminator {
                        feature_idx: model.trees()[tree].selections[0].feature_idx,
                        threshold: f64::INFINITY,
                    }
                } else {
                    model.trees()[tree].selections[level].clone()
                };
                assert_eq!(actual.feature_idx, expected.feature_idx);
                assert_eq!(actual.threshold.to_bits(), expected.threshold.to_bits());
            }
        }
        assert_eq!(plan.predicates().len(), next_predicate);
        assert_eq!(plan.actions().len(), 8);
        assert_eq!(
            plan.actions()[6],
            Action::AddSums {
                destination: SumId(0),
                source: SumId(1)
            }
        );
        assert_eq!(plan.actions()[7], Action::ReturnScaled(SumId(0)));
        assert_eq!(plan.chunks().len(), depths.len().div_ceil(2));
        for (chunk, (range, tests)) in plan.chunks().iter().zip([(0..3, 2), (3..5, 6), (5..8, 6)]) {
            assert_eq!(chunk.actions, range);
            assert_eq!(chunk.incoming, Slots::default());
            assert_eq!(chunk.tests, tests);
            let actual: usize = plan.actions()[chunk.actions.clone()]
                .iter()
                .map(|action| match action {
                    Action::EvaluateTree { predicates, .. } => predicates.len(),
                    _ => 0,
                })
                .sum();
            assert_eq!(chunk.tests, actual);
        }
        let simd = crate::sort_policy::plan(
            &model,
            SortOptions {
                comparisons: ComparisonMode::Simd2,
                ..tree_major_options(2)
            },
        )
        .unwrap();
        assert_eq!(plan.actions(), simd.actions());
        assert_eq!(plan.chunks(), simd.chunks());
    }

    #[test]
    fn tree_major_emits_vector_compares_only() {
        let context = Context::create();
        let model = tree_major_model(&[6, 6]);
        let plan = plan(&model, tree_major_options(1)).unwrap();
        assert_eq!(plan.sum_count(), 2);
        for (triple, pointer_count) in
            [("aarch64-apple-darwin", 3), ("x86_64-unknown-linux-gnu", 2)]
        {
            let arm64 = pointer_count == 3;
            let module = emit_plan(
                &context,
                &plan,
                &TargetTriple::create(triple),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            for number in 0..2 {
                let function = module
                    .get_function(&format!("continuation{number}"))
                    .unwrap();
                assert_eq!(function.count_params(), pointer_count + 2);
                let ir = function.print_to_string().to_string();
                assert_eq!(ir.matches("fcmp ogt <2 x double>").count(), 3);
                assert_eq!(
                    ir.matches("call <2 x i64> @llvm.aarch64.neon.vsli.v2i64")
                        .count(),
                    if arm64 { 2 } else { 0 }
                );
                assert!(!ir.contains("fcmp ogt double"));
                assert_eq!(ir.matches("load <2 x double>").count(), 3);
                for instruction in function.get_first_basic_block().unwrap().get_instructions() {
                    if instruction.get_opcode() == InstructionOpcode::Load {
                        assert_eq!(instruction.get_alignment().unwrap(), 8);
                    }
                }
            }
            let assembly = emit_assembly(&context, &plan, triple).unwrap();
            if arm64 {
                assert!(assembly.contains("fcmgt.2d"));
                assert!(assembly.contains("sli"));
            } else {
                assert!(!assembly.contains("llvm.aarch64.neon.vsli"));
                assert!(!assembly.contains("ucomisd"));
            }
        }
    }

    #[test]
    fn avx2_off_is_byte_identical() {
        let context = Context::create();
        for model in [mixed_depth_model(), tree_major_model(&[0, 1, 3, 5, 6, 8])] {
            for options in [SortOptions::default(), tree_major_options(1)] {
                let plan = plan(&model, options).unwrap();
                for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
                    let original = emit_assembly(&context, &plan, triple).unwrap();
                    assert_eq!(
                        original,
                        emit_assembly_with(&context, &plan, triple, EmitOptions::default())
                            .unwrap(),
                    );
                    if triple == "aarch64-apple-darwin" {
                        assert_eq!(
                            original,
                            emit_assembly_with(&context, &plan, triple, EmitOptions { avx2: true })
                                .unwrap(),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn avx2_tree_major_emits_folded_compares() {
        let context = Context::create();
        Target::initialize_all(&InitializationConfig::default());
        let triple = "x86_64-unknown-linux-gnu";
        for depth in [1usize, 3, 5, 6, 8] {
            let model = tree_major_model(&[depth]);
            let plan = plan(&model, tree_major_options(1)).unwrap();
            let (module, _) = prepare_module(
                &context,
                &plan,
                &TargetTriple::create(triple),
                EmitOptions { avx2: true },
            )
            .unwrap();
            let ir = module.print_to_string().to_string();
            let padded = depth.next_multiple_of(2);
            assert_eq!(
                ir.matches("call <4 x double> @llvm.x86.avx.cmp.pd.256")
                    .count(),
                padded / 4
            );
            assert_eq!(
                ir.matches("call <2 x double> @llvm.x86.sse2.cmp.pd")
                    .count(),
                padded % 4 / 2
            );
            assert!(!ir.contains("fcmp"));
            assert!(!ir.contains("vsli"));
            for function in module
                .get_functions()
                .filter(|function| function.count_basic_blocks() != 0)
            {
                for (name, value) in [("target-cpu", "x86-64-v3"), ("target-features", "")] {
                    assert_eq!(
                        function
                            .get_string_attribute(AttributeLoc::Function, name)
                            .unwrap()
                            .get_string_value()
                            .to_str()
                            .unwrap(),
                        value,
                    );
                }
                for block in function.get_basic_blocks() {
                    for instruction in block.get_instructions() {
                        if instruction.get_opcode() == InstructionOpcode::Load {
                            assert_eq!(instruction.get_alignment().unwrap(), 8);
                        }
                    }
                }
            }
            let assembly =
                emit_assembly_with(&context, &plan, triple, EmitOptions { avx2: true }).unwrap();
            let instructions: Vec<_> = assembly
                .lines()
                .filter_map(|line| {
                    let line = line.trim();
                    line.split_once(char::is_whitespace)
                })
                .collect();
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, _)| *op == "vcmpgt_oqpd")
                    .count(),
                padded.div_ceil(4)
            );
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, operands)| *op == "vcmpgt_oqpd" && operands.contains('('))
                    .count(),
                padded.div_ceil(4),
                "depth={depth}\n{assembly}"
            );
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, _)| *op == "vmovmskpd")
                    .count(),
                padded.div_ceil(4)
            );
            let leaf_adds: Vec<_> = instructions
                .iter()
                .enumerate()
                .filter(|(_, (op, _))| *op == "vaddsd")
                .collect();
            assert_eq!(leaf_adds.len(), 1, "depth={depth}\n{assembly}");
            for (position, (_, operands)) in leaf_adds {
                let (displacement, address) = operands.split_once('(').unwrap();
                assert!(
                    displacement
                        .strip_prefix('-')
                        .unwrap_or(displacement)
                        .bytes()
                        .all(|byte| byte.is_ascii_digit())
                );
                let (address, registers) = address.split_once(')').unwrap();
                let parts: Vec<_> = address.split(',').collect();
                assert!(parts.len() == 2 || parts.len() == 3);
                for register in &parts[..2] {
                    let register = register.strip_prefix('%').unwrap();
                    assert!(!register.is_empty());
                    assert!(
                        register
                            .bytes()
                            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
                    );
                }
                if parts.len() == 3 {
                    assert!(["1", "2", "4", "8"].contains(&parts[2]));
                }
                assert!(registers.starts_with(", %xmm"), "depth={depth}\n{assembly}");
                if position != 0 {
                    let (op, operands) = instructions[position - 1];
                    assert!(
                        !(op == "vmovsd"
                            && operands
                                .split_once(')')
                                .is_some_and(|(_, registers)| registers.starts_with(", %xmm"))),
                        "separate leaf load: depth={depth}\n{assembly}"
                    );
                }
            }
            for forbidden in [
                "ucomisd", "vucomisd", "cmpltpd", "vcmpltpd", "psllq", "pshufd",
            ] {
                assert!(
                    !instructions.iter().any(|(op, _)| *op == forbidden),
                    "{forbidden}: depth={depth}"
                );
            }
        }
    }

    #[test]
    fn avx2_leaf_blocks_are_unpermuted() {
        let context = Context::create();
        Target::initialize_all(&InitializationConfig::default());
        let model = tree_major_model(&[0, 1, 3, 5, 6, 8]);
        let plan = plan(&model, tree_major_options(2)).unwrap();
        for triple in ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"] {
            let (module, _) = prepare_module(
                &context,
                &plan,
                &TargetTriple::create(triple),
                EmitOptions { avx2: true },
            )
            .unwrap();
            let mut expected: Vec<_> = plan.predicates().iter().map(|p| p.threshold).collect();
            expected.resize(expected.len().next_multiple_of(8), 0.0);
            for tree in model.trees() {
                if triple == "aarch64-apple-darwin" && !tree.selections.is_empty() {
                    expected.extend(permuted_leaves(tree).1);
                } else {
                    expected.extend(&tree.leafs);
                }
            }
            let expected: Vec<_> = expected
                .into_iter()
                .map(|value| context.f64_type().const_float(value))
                .collect();
            let actual = module
                .get_global("constants")
                .unwrap()
                .get_initializer()
                .unwrap()
                .into_array_value();
            assert_eq!(actual.get_type().len() as usize, expected.len());
            assert_eq!(actual, context.f64_type().const_array(&expected));
        }
    }

    #[test]
    fn emit_options_from_host_matches_llvm() {
        assert_eq!(
            EmitOptions::from_host().avx2,
            TargetMachine::get_host_cpu_features()
                .to_string()
                .split(',')
                .any(|f| f == "+avx2"),
        );
        assert!(!EmitOptions::default().avx2);
    }

    fn mixed_depth_model() -> Trees {
        let shared = Discriminator {
            feature_idx: FeatureIndex(2),
            threshold: 1.37,
        };
        Trees::new(
            vec![
                SingleTree {
                    leafs: vec![9.0],
                    selections: vec![],
                },
                SingleTree {
                    leafs: vec![10.0, 11.0],
                    selections: vec![shared.clone()],
                },
                SingleTree {
                    leafs: (20..28).map(|leaf| leaf as f64).collect(),
                    selections: vec![
                        shared,
                        Discriminator {
                            feature_idx: FeatureIndex(0),
                            threshold: 2.61,
                        },
                        Discriminator {
                            feature_idx: FeatureIndex(0),
                            threshold: -0.42,
                        },
                    ],
                },
            ],
            vec![NanMode::AsIs; 3],
            2.0,
            3.0,
        )
        .unwrap()
    }

    fn assert_opaque_constants(
        function: FunctionValue<'_>,
        comparisons: usize,
        retirements: usize,
    ) {
        let mut thresholds = 0;
        let mut leaves = 0;
        for instruction in function.get_first_basic_block().unwrap().get_instructions() {
            let is_threshold = instruction.get_opcode() == InstructionOpcode::FCmp;
            let is_leaf = instruction.get_opcode() == InstructionOpcode::FAdd
                && instruction
                    .get_name()
                    .unwrap()
                    .to_bytes()
                    .starts_with(b"sum");
            if is_threshold || is_leaf {
                let value = instruction.get_operand(1).unwrap().unwrap_value();
                let load = value
                    .as_instruction_value()
                    .expect("constant must come from a load");
                assert_eq!(load.get_opcode(), InstructionOpcode::Load);
                assert_eq!(load.get_alignment().unwrap(), 8);
                thresholds += usize::from(is_threshold);
                leaves += usize::from(is_leaf);
            }
        }
        assert_eq!((thresholds, leaves), (comparisons, retirements));
    }

    #[test]
    fn constants_block_layout_matches_plan() {
        let context = Context::create();
        let model = mixed_depth_model();
        let plan = plan(&model, SortOptions::default()).unwrap();
        assert_eq!(plan.predicates().len(), 3);
        assert_eq!(
            plan.predicates()
                .iter()
                .map(|p| p.threshold)
                .collect::<Vec<_>>(),
            [1.37, 2.61, -0.42],
        );
        assert!(plan.actions().iter().any(|action| matches!(action,
            Action::Compare { comparisons, .. }
                if comparisons.iter().any(|comparison| comparison.consumers.len() == 2)
        )));
        let mut first_use = Vec::new();
        for action in plan.actions() {
            if let Action::Compare { comparisons, .. } = action {
                for comparison in comparisons {
                    if !first_use.contains(&comparison.predicate.0) {
                        first_use.push(comparison.predicate.0);
                    }
                }
            }
        }
        assert_eq!(first_use.len(), plan.predicates().len());
        assert_ne!(first_use, (0..plan.predicates().len()).collect::<Vec<_>>());
        let module = emit_plan(
            &context,
            &plan,
            &TargetMachine::get_default_triple(),
            TreePath::Lanes2,
        )
        .unwrap();
        module.verify().unwrap();
        assert_eq!(module.get_globals().count(), 1);
        let constants = module.get_global("constants").unwrap();
        assert_eq!(constants.get_linkage(), Linkage::Private);
        assert!(constants.is_constant());
        assert_eq!(constants.get_alignment(), 64);
        let mut expected: Vec<_> = first_use
            .iter()
            .map(|&predicate| plan.predicates()[predicate].threshold)
            .collect();
        expected.resize(8, 0.0);
        expected.extend([
            9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        ]);
        let expected: Vec<_> = expected
            .into_iter()
            .map(|value| context.f64_type().const_float(value))
            .collect();
        assert_eq!(
            constants.get_initializer().unwrap().into_array_value(),
            context.f64_type().const_array(&expected),
        );
        let compiled = compile(&context, &plan).unwrap();
        for row in [[0.0, -1.0, 0.0], [2.0, 0.0, 2.0], [3.0, 0.0, 3.0]] {
            assert_eq!(compiled.predict(&row), model.predict(&row));
        }
    }

    #[test]
    fn one_chunk_model_keeps_wrapper_and_opaque_loads() {
        let context = Context::create();
        let model = mixed_depth_model();
        let plan = plan(&model, SortOptions::default()).unwrap();
        assert_eq!(plan.chunks().len(), 1);
        Target::initialize_all(&InitializationConfig::default());
        for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            let (module, _) = prepare_module(
                &context,
                &plan,
                &TargetTriple::create(triple),
                EmitOptions::default(),
            )
            .unwrap();
            assert_eq!(module.get_functions().count(), 2);
            let wrapper = module.get_function("predict").unwrap();
            assert_eq!(wrapper.count_params(), 1);
            let ir = wrapper.print_to_string().to_string();
            assert_eq!(
                ir.matches("call preserve_nonecc double @continuation0")
                    .count(),
                1
            );
            let continuation = module.get_function("continuation0").unwrap();
            assert_opaque_constants(continuation, 3, 3);
            let ir = continuation.print_to_string().to_string();
            assert_eq!(ir.matches("load double").count(), 8);
            assert!(!ir.contains("@constants"));
        }
    }

    #[test]
    fn pointer_parameters_have_bounds() {
        let context = Context::create();
        let model = Trees::new(
            (0..9)
                .map(|tree| SingleTree {
                    leafs: (0..8).map(|leaf| (tree * 8 + leaf) as f64).collect(),
                    selections: [1.37, 2.61, -0.42]
                        .map(|threshold| Discriminator {
                            feature_idx: FeatureIndex(tree),
                            threshold,
                        })
                        .into(),
                })
                .collect(),
            vec![NanMode::AsIs; 9],
            1.0,
            0.0,
        )
        .unwrap();
        let plan = plan(
            &model,
            SortOptions {
                order: Order::TreeOrder,
                ..SortOptions::default()
            },
        )
        .unwrap();
        assert_eq!(plan.chunks().len(), 2);
        for (triple, pointer_count) in
            [("x86_64-unknown-linux-gnu", 2), ("aarch64-apple-darwin", 3)]
        {
            let module = emit_plan(
                &context,
                &plan,
                &TargetTriple::create(triple),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            let constants = module.get_global("constants").unwrap();
            let entries = constants
                .get_initializer()
                .unwrap()
                .into_array_value()
                .get_type()
                .len();
            assert_eq!(entries, 104);
            for (number, first_predicate) in [0, 16].into_iter().enumerate() {
                let function = module
                    .get_function(&format!("continuation{number}"))
                    .unwrap();
                assert_eq!(
                    function
                        .get_param_iter()
                        .filter(|value| value.is_pointer_value())
                        .count(),
                    pointer_count,
                );
                for parameter in 0..pointer_count {
                    assert!(
                        function
                            .get_nth_param(parameter as u32)
                            .unwrap()
                            .is_pointer_value()
                    );
                    for (name, value) in [
                        ("readonly", 0),
                        ("noalias", 0),
                        ("align", if parameter == 1 { 64 } else { 8 }),
                        (
                            "dereferenceable",
                            match parameter {
                                0 => 9 * 8,
                                1 => u64::from(entries) * 8,
                                2 => (u64::from(entries) - first_predicate) * 8,
                                _ => unreachable!(),
                            },
                        ),
                    ] {
                        assert_eq!(
                            function
                                .get_enum_attribute(
                                    AttributeLoc::Param(parameter as u32),
                                    Attribute::get_named_enum_kind_id(name),
                                )
                                .unwrap()
                                .get_enum_value(),
                            value,
                            "{triple} continuation{number} parameter {parameter} {name}",
                        );
                    }
                }
                let instructions: Vec<_> = function
                    .get_first_basic_block()
                    .unwrap()
                    .get_instructions()
                    .collect();
                let thresholds: Vec<_> = instructions
                    .iter()
                    .filter(|instruction| {
                        instruction.get_opcode() == InstructionOpcode::Load
                            && instruction
                                .get_name()
                                .unwrap()
                                .to_bytes()
                                .starts_with(b"threshold")
                    })
                    .collect();
                assert_eq!(thresholds.len(), plan.chunks()[number].tests);
                for (offset, load) in thresholds.iter().enumerate() {
                    let gep = load
                        .get_operand(0)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    assert_eq!(gep.get_opcode(), InstructionOpcode::GetElementPtr);
                    let (parameter, operand, index) = if pointer_count == 3 {
                        (2, 1, offset as u64)
                    } else {
                        (1, 2, first_predicate + offset as u64)
                    };
                    assert_eq!(
                        gep.get_operand(0).unwrap().unwrap_value(),
                        function.get_nth_param(parameter).unwrap(),
                    );
                    assert_eq!(
                        gep.get_operand(operand)
                            .unwrap()
                            .unwrap_value()
                            .into_int_value()
                            .get_zero_extended_constant(),
                        Some(index),
                    );
                }
                if pointer_count == 3 && number == 0 {
                    let call = instructions
                        .iter()
                        .find(|instruction| instruction.get_opcode() == InstructionOpcode::Call)
                        .unwrap();
                    let base = call
                        .get_operand(2)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    assert_eq!(base.get_opcode(), InstructionOpcode::GetElementPtr);
                    assert_eq!(
                        base.get_operand(0).unwrap().unwrap_value(),
                        function.get_nth_param(1).unwrap(),
                    );
                    assert_eq!(
                        base.get_operand(1)
                            .unwrap()
                            .unwrap_value()
                            .into_int_value()
                            .get_zero_extended_constant(),
                        Some(16),
                    );
                }
            }
            let assembly = emit_assembly(&context, &plan, triple).unwrap();
            if pointer_count == 2 {
                assert!(
                    assembly
                        .lines()
                        .any(|line| { line.contains("ucomisd") && line.contains("(%") }),
                    "{assembly}"
                );
                assert!(
                    !assembly
                        .lines()
                        .any(|line| { line.contains("movabsq") && line.contains("$.LCPI") }),
                    "{assembly}"
                );
            } else {
                assert!(
                    assembly.lines().any(|line| {
                        let mut words = line.split_whitespace();
                        matches!(words.next(), Some("ldr" | "ldp"))
                            && words
                                .next()
                                .is_some_and(|register| register.starts_with('d'))
                    }),
                    "{assembly}"
                );
                assert!(!assembly.contains("movk"), "{assembly}");
            }
        }
    }

    #[test]
    fn dropped_model_and_last_tree_leaf() {
        let context = Context::create();
        let model = Trees::new(
            (0..70)
                .map(|tree| SingleTree {
                    leafs: (0..64).map(|leaf| (tree * 64 + leaf) as f64).collect(),
                    selections: (0..6)
                        .map(|feature| Discriminator {
                            feature_idx: FeatureIndex(feature),
                            threshold: 1.37,
                        })
                        .collect(),
                })
                .collect(),
            vec![NanMode::AsIs; 6],
            2.0,
            3.0,
        )
        .unwrap();
        let rows = [[0.0; 6], [2.0; 6]];
        assert_eq!(model.leaf_indices(&rows[0]).unwrap()[0], 0);
        assert_eq!(model.leaf_indices(&rows[1]).unwrap()[69], 63);
        let preceding_leaves: usize = model.trees()[..69]
            .iter()
            .map(|tree| tree.leafs.len())
            .sum();
        assert!(preceding_leaves > 4095);
        let expected = rows.map(|row| model.predict(&row).unwrap());
        let mut kernels = Vec::new();
        for order in [Order::TreeOrder, SortOptions::default().order] {
            let plan = plan(
                &model,
                SortOptions {
                    order,
                    ..SortOptions::default()
                },
            )
            .unwrap();
            kernels.push(compile(&context, &plan).unwrap());
            drop(plan);
        }
        drop(model);
        for compiled in kernels {
            for (row, expected) in rows.iter().zip(expected) {
                assert_eq!(compiled.predict(row), Ok(expected));
            }
        }
    }

    #[test]
    fn empty_and_constant_forests_still_predict() {
        let context = Context::create();
        for leaves in [vec![], vec![1.5, -2.0, 4.0]] {
            let empty = leaves.is_empty();
            let model = Trees::new(
                leaves
                    .into_iter()
                    .map(|leaf| SingleTree {
                        leafs: vec![leaf],
                        selections: vec![],
                    })
                    .collect(),
                vec![],
                if empty { f64::NAN } else { 2.0 },
                3.0,
            )
            .unwrap();
            let plan = plan(&model, SortOptions::default()).unwrap();
            let module = emit_plan(
                &context,
                &plan,
                &TargetMachine::get_default_triple(),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            let constants = module.get_global("constants").unwrap();
            let expected = if empty {
                vec![0.0]
            } else {
                vec![1.5, -2.0, 4.0]
            };
            let values: Vec<_> = expected
                .into_iter()
                .map(|value| context.f64_type().const_float(value))
                .collect();
            assert_eq!(
                constants.get_initializer().unwrap().into_array_value(),
                context.f64_type().const_array(&values),
            );
            let compiled = compile(&context, &plan).unwrap();
            assert_eq!(compiled.predict(&[]), Ok(if empty { 3.0 } else { 10.0 }));
            assert_eq!(compiled.predict(&[]), model.predict(&[]));
        }
    }

    #[test]
    fn simd_pair_loads_both_thresholds() {
        let context = Context::create();
        let mut outcomes = [false; 4];
        for thresholds in [[1.37, 2.61], [2.61, 1.37]] {
            let model = Trees::new(
                vec![SingleTree {
                    leafs: vec![0.0, 1.0, 2.0, 3.0],
                    selections: thresholds
                        .map(|threshold| Discriminator {
                            feature_idx: FeatureIndex(0),
                            threshold,
                        })
                        .into(),
                }],
                vec![NanMode::AsIs],
                1.0,
                0.0,
            )
            .unwrap();
            let plan = plan(
                &model,
                SortOptions {
                    comparisons: ComparisonMode::Simd2,
                    ..SortOptions::default()
                },
            )
            .unwrap();
            assert!(plan.actions().iter().any(|action| matches!(action,
                Action::Compare { comparisons, .. } if comparisons.len() == 2
            )));
            let pair = plan
                .actions()
                .iter()
                .find_map(|action| match action {
                    Action::Compare { comparisons, .. } if comparisons.len() == 2 => {
                        Some(comparisons)
                    }
                    _ => None,
                })
                .unwrap();
            let mut predicate_slots = vec![None; plan.predicates().len()];
            let mut next_slot = 0;
            for action in plan.actions() {
                if let Action::Compare { comparisons, .. } = action {
                    for comparison in comparisons {
                        if predicate_slots[comparison.predicate.0].is_none() {
                            predicate_slots[comparison.predicate.0] = Some(next_slot);
                            next_slot += 1;
                        }
                    }
                }
            }
            assert_eq!(next_slot, plan.predicates().len());
            let predicate_slots: Vec<_> = predicate_slots.into_iter().map(Option::unwrap).collect();
            let chunk_min_slot = plan.actions()[plan.chunks()[0].actions.clone()]
                .iter()
                .flat_map(|action| match action {
                    Action::Compare { comparisons, .. } => comparisons.as_slice(),
                    _ => &[],
                })
                .map(|comparison| predicate_slots[comparison.predicate.0])
                .min()
                .unwrap();
            for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
                let module = emit_plan(
                    &context,
                    &plan,
                    &TargetTriple::create(triple),
                    TreePath::Lanes2,
                )
                .unwrap();
                module.verify().unwrap();
                let function = module.get_function("continuation0").unwrap();
                let compare = function
                    .get_first_basic_block()
                    .unwrap()
                    .get_instructions()
                    .find(|instruction| instruction.get_opcode() == InstructionOpcode::FCmp)
                    .unwrap();
                assert!(
                    compare
                        .print_to_string()
                        .to_string()
                        .contains("fcmp ogt <2 x double>")
                );
                let mut vector = compare.get_operand(1).unwrap().unwrap_value();
                let mut loads = Vec::new();
                for lane in [1usize, 0] {
                    let insert = vector.as_instruction_value().unwrap();
                    assert_eq!(insert.get_opcode(), InstructionOpcode::InsertElement);
                    assert_eq!(
                        insert
                            .get_operand(2)
                            .unwrap()
                            .unwrap_value()
                            .into_int_value()
                            .get_zero_extended_constant(),
                        Some(lane as u64),
                    );
                    let load = insert
                        .get_operand(1)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    assert_eq!(load.get_opcode(), InstructionOpcode::Load);
                    assert_eq!(load.get_alignment().unwrap(), 8);
                    let gep = load
                        .get_operand(0)
                        .unwrap()
                        .unwrap_value()
                        .as_instruction_value()
                        .unwrap();
                    assert_eq!(gep.get_opcode(), InstructionOpcode::GetElementPtr);
                    let slot = predicate_slots[pair[lane].predicate.0];
                    let (parameter, operand, index) = if triple.starts_with("aarch64") {
                        (2, 1, slot - chunk_min_slot)
                    } else {
                        (1, 2, slot)
                    };
                    assert_eq!(
                        gep.get_operand(0).unwrap().unwrap_value(),
                        function.get_nth_param(parameter).unwrap(),
                    );
                    assert_eq!(
                        gep.get_operand(operand)
                            .unwrap()
                            .unwrap_value()
                            .into_int_value()
                            .get_zero_extended_constant(),
                        Some(index as u64),
                    );
                    loads.push(load);
                    vector = insert.get_operand(0).unwrap().unwrap_value();
                }
                assert_ne!(loads[0], loads[1]);
            }
            let compiled = compile(&context, &plan).unwrap();
            for value in [0.0, 1.37, 2.0, 2.61, 3.0, f64::NAN] {
                let expected = model.predict(&[value]).unwrap();
                assert_eq!(compiled.predict(&[value]), Ok(expected));
                outcomes[expected as usize] = true;
            }
        }
        assert!(outcomes.into_iter().all(|outcome| outcome));
    }

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
            let assembly = emit_assembly(
                &context,
                &plan(&model, SortOptions::default()).unwrap(),
                triple,
            )
            .unwrap();
            assert!(assembly.contains("predict"), "{triple}: {assembly}");
        }
    }

    #[test]
    fn scheduled_order_shares_loads_and_comparisons_before_llvm_passes() {
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
            (Order::TreeOrder, ComparisonMode::Scalar, 10, 4, 0),
            (Order::TreeOrder, ComparisonMode::Simd2, 10, 4, 0),
            (
                Order::Scheduled(RegisterBudget {
                    general: NonZeroUsize::new(6).unwrap(),
                    floating: NonZeroUsize::new(12).unwrap(),
                }),
                ComparisonMode::Scalar,
                5,
                2,
                0,
            ),
            (
                Order::Scheduled(RegisterBudget {
                    general: NonZeroUsize::new(6).unwrap(),
                    floating: NonZeroUsize::new(12).unwrap(),
                }),
                ComparisonMode::Simd2,
                5,
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
                )
                .unwrap(),
                &TargetMachine::get_default_triple(),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            let ir = module.print_to_string().to_string();
            assert_eq!(ir.matches("load double").count(), loads);
            assert_eq!(ir.matches("fcmp ogt double").count(), scalars);
            assert_eq!(ir.matches("fcmp ogt <2 x double>").count(), pairs);
            assert_eq!(ir.matches("sext <2 x i1>").count(), pairs);
            assert_eq!(ir.matches("to <2 x i64>").count(), pairs);
            assert_eq!(ir.matches("extractelement <2 x i64>").count(), pairs * 2);
            assert!(!ir.contains("extractelement <2 x i1>"));
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
        let module = emit_plan(
            &context,
            &plan(&model, SortOptions::default()).unwrap(),
            &TargetMachine::get_default_triple(),
            TreePath::Lanes2,
        )
        .unwrap();
        module.verify().unwrap();
        let ir = module.print_to_string().to_string();
        assert_eq!(module.get_functions().count(), 2);
        assert_eq!(module.get_function("predict").unwrap().count_params(), 1);
        assert_eq!(
            ir.matches("call preserve_nonecc double @continuation0")
                .count(),
            1
        );
        let continuation = module.get_function("continuation0").unwrap();
        assert_opaque_constants(continuation, 1, 1);
        for expected in [
            "@constants = private constant [10 x double]",
            "align 64",
            "noalias readonly",
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

    #[test]
    fn continuations_follow_plan_ranges_with_identical_musttail_signatures() {
        let context = Context::create();
        let model = Trees::new(
            (0..12)
                .map(|tree| SingleTree {
                    leafs: (0..16).map(|leaf| leaf as f64).collect(),
                    selections: (0..4)
                        .map(|level| Discriminator {
                            feature_idx: FeatureIndex(level % 2),
                            threshold: (tree * 4 + level) as f64,
                        })
                        .collect(),
                })
                .collect(),
            vec![NanMode::AsIs; 2],
            1.0,
            0.0,
        )
        .unwrap();
        for comparisons in [ComparisonMode::Scalar, ComparisonMode::Simd2] {
            let plan = plan(
                &model,
                SortOptions {
                    comparisons,
                    ..SortOptions::default()
                },
            )
            .unwrap();
            assert!(plan.chunks().len() > 1);
            let module = emit_plan(
                &context,
                &plan,
                &TargetMachine::get_default_triple(),
                TreePath::Lanes2,
            )
            .unwrap();
            module.verify().unwrap();
            assert_eq!(module.get_functions().count(), plan.chunks().len() + 1);
            let first = module.get_function("continuation0").unwrap();
            for (number, chunk) in plan.chunks().iter().enumerate() {
                let function = module
                    .get_function(&format!("continuation{number}"))
                    .unwrap();
                assert_eq!(function.get_type(), first.get_type());
                assert_eq!(
                    function.get_call_conventions(),
                    PRESERVE_NONE_CALL_CONVENTION
                );
                assert!(
                    function
                        .get_enum_attribute(
                            AttributeLoc::Function,
                            Attribute::get_named_enum_kind_id("noinline")
                        )
                        .is_some()
                );
                let ir = function.print_to_string().to_string();
                assert_eq!(
                    ir.matches("fcmp ogt double").count()
                        + 2 * ir.matches("fcmp ogt <2 x double>").count(),
                    chunk.tests
                );
                let loads = plan.actions()[chunk.actions.clone()]
                    .iter()
                    .filter(|a| matches!(a, Action::Load { .. } | Action::Retire { .. }))
                    .count();
                assert_eq!(ir.matches("load double").count(), loads + chunk.tests);
                let mut lines = ir.lines();
                if number + 1 < plan.chunks().len() {
                    let call = lines.find(|line| line.contains("musttail call")).unwrap();
                    assert!(call.contains(&format!("@continuation{}(", number + 1)));
                    assert!(call.contains("preserve_nonecc"));
                    assert!(lines.next().unwrap().trim().starts_with("ret double"));
                    assert_eq!(ir.matches("call ").count(), 1);
                } else {
                    assert!(!ir.contains("call "));
                }
            }
            let wrapper = module.get_function("predict").unwrap();
            assert_eq!(wrapper.get_call_conventions(), 0);
            assert_eq!(wrapper.count_params(), 1);
            assert_eq!(
                wrapper
                    .print_to_string()
                    .to_string()
                    .matches("call ")
                    .count(),
                1
            );
            for triple in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
                assert!(
                    emit_assembly(&context, &plan, triple)
                        .unwrap()
                        .contains("continuation")
                );
            }
        }
    }
}
