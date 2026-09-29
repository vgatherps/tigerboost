#![doc = include_str!("../README.md")]

pub mod jit;
pub mod parsed_tree;
pub mod sort_policy;

#[cfg(test)]
extern crate self as tigerboost;
#[cfg(test)]
#[path = "../examples/support/mod.rs"]
pub mod test_workloads;
