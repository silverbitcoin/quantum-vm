//! # Quantum VM
//!
//! Bytecode interpreter for Quantum smart contracts.
//!
//! This crate provides:
//! - Bytecode instruction set (100+ operations)
//! - Stack-based interpreter
//! - Fuel metering
//! - Resource safety enforcement
//! - Native function implementations

#![warn(missing_docs, rust_2018_idioms)]
#![forbid(unsafe_code)]

/// Bytecode format and instruction definitions
pub mod bytecode;

/// Bytecode verification and validation
pub mod verifier;

/// Bytecode interpreter and execution engine
pub mod interpreter;

/// Runtime environment and state management
pub mod runtime;

/// Native function implementations
pub mod native;

pub use bytecode::{Instruction, Bytecode, Module};
pub use verifier::{BytecodeVerifier, VerifierError, VerifierResult};
pub use interpreter::Interpreter;
pub use runtime::Runtime;
pub use native::NativeFunctions;
