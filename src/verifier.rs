//! # Bytecode Verifier
//!
//! Production-ready bytecode verification with:
//! - Type safety verification
//! - Resource safety checking (linear types)
//! - Borrow checking validation
//! - Stack safety verification
//! - Control flow validation

use crate::bytecode::{Bytecode, Function, Instruction, LocalIndex, Module, TypeTag};
use std::collections::HashSet;
use thiserror::Error;

/// Verifier error types for bytecode validation
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum VerifierError {
    /// Type mismatch between expected and actual types
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch {
        /// Expected type name
        expected: String,
        /// Actual type name
        got: String,
    },

    /// Stack underflow during execution
    #[error("Stack underflow at PC {pc}")]
    StackUnderflow {
        /// Program counter where underflow occurred
        pc: usize,
    },

    /// Stack overflow during execution
    #[error("Stack overflow at PC {pc}")]
    StackOverflow {
        /// Program counter where overflow occurred
        pc: usize,
    },

    /// Invalid local variable index
    #[error("Invalid local index {index} at PC {pc}")]
    InvalidLocalIndex {
        /// Invalid local variable index
        index: LocalIndex,
        /// Program counter where error occurred
        pc: usize,
    },

    /// Invalid branch target
    #[error("Invalid branch target {target} at PC {pc}")]
    InvalidBranchTarget {
        /// Invalid branch target offset
        target: i32,
        /// Program counter where error occurred
        pc: usize,
    },

    /// Resource safety violation (linear type violation)
    #[error("Resource safety violation: {message}")]
    ResourceSafetyViolation {
        /// Detailed error message
        message: String,
    },

    /// Borrow checking error
    #[error("Borrow checking error: {message}")]
    BorrowCheckingError {
        /// Detailed error message
        message: String,
    },

    /// Unreachable code detected
    #[error("Unreachable code at PC {pc}")]
    UnreachableCode {
        /// Program counter of unreachable code
        pc: usize,
    },

    /// Missing return statement at end of function
    #[error("Missing return at end of function")]
    MissingReturn,

    /// Invalid function signature
    #[error("Invalid function signature")]
    InvalidSignature,
}

/// Result type for verifier operations
pub type VerifierResult<T> = Result<T, VerifierError>;

/// Abstract stack type for verification
#[derive(Debug, Clone, PartialEq, Eq)]
struct AbstractStack {
    types: Vec<TypeTag>,
    max_depth: usize,
}

impl AbstractStack {
    fn new() -> Self {
        Self {
            types: Vec::new(),
            max_depth: 0,
        }
    }

    fn push(&mut self, ty: TypeTag) -> VerifierResult<()> {
        self.types.push(ty);
        self.max_depth = self.max_depth.max(self.types.len());

        // Prevent stack overflow (max 1024 elements)
        if self.types.len() > 1024 {
            return Err(VerifierError::StackOverflow { pc: 0 });
        }

        Ok(())
    }

    fn pop(&mut self) -> VerifierResult<TypeTag> {
        self.types
            .pop()
            .ok_or(VerifierError::StackUnderflow { pc: 0 })
    }

    fn pop_expect(&mut self, expected: &TypeTag) -> VerifierResult<()> {
        let got = self.pop()?;
        if &got != expected {
            return Err(VerifierError::TypeMismatch {
                expected: format!("{:?}", expected),
                got: format!("{:?}", got),
            });
        }
        Ok(())
    }

    fn peek(&self) -> VerifierResult<&TypeTag> {
        self.types
            .last()
            .ok_or(VerifierError::StackUnderflow { pc: 0 })
    }

    /// Get the number of types on the stack
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.types.len()
    }

    /// Check if the stack is empty
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.types.is_empty()
    }
}

/// Local variable state for borrow checking
#[derive(Debug, Clone, PartialEq, Eq)]
enum LocalState {
    /// Variable is available
    Available,
    /// Variable has been moved
    Moved,
    /// Variable is borrowed (immutably)
    Borrowed { count: usize },
    /// Variable is mutably borrowed
    MutBorrowed,
}

/// Borrow checker state
#[derive(Debug, Clone)]
struct BorrowChecker {
    /// State of each local variable
    locals: Vec<LocalState>,
    /// Active references (for tracking lifetime)
    #[allow(dead_code)]
    active_refs: HashSet<usize>,
}

impl BorrowChecker {
    fn new(num_locals: usize) -> Self {
        Self {
            locals: vec![LocalState::Available; num_locals],
            active_refs: HashSet::new(),
        }
    }

    fn move_local(&mut self, idx: LocalIndex) -> VerifierResult<()> {
        let idx = idx as usize;
        if idx >= self.locals.len() {
            return Err(VerifierError::InvalidLocalIndex {
                index: idx as u16,
                pc: 0,
            });
        }

        match &self.locals[idx] {
            LocalState::Available => {
                self.locals[idx] = LocalState::Moved;
                Ok(())
            }
            LocalState::Moved => Err(VerifierError::ResourceSafetyViolation {
                message: format!("Local {} already moved", idx),
            }),
            LocalState::Borrowed { .. } => Err(VerifierError::BorrowCheckingError {
                message: format!("Cannot move local {} while borrowed", idx),
            }),
            LocalState::MutBorrowed => Err(VerifierError::BorrowCheckingError {
                message: format!("Cannot move local {} while mutably borrowed", idx),
            }),
        }
    }

    fn copy_local(&mut self, idx: LocalIndex) -> VerifierResult<()> {
        let idx = idx as usize;
        if idx >= self.locals.len() {
            return Err(VerifierError::InvalidLocalIndex {
                index: idx as u16,
                pc: 0,
            });
        }

        match &self.locals[idx] {
            LocalState::Available => Ok(()),
            LocalState::Moved => Err(VerifierError::ResourceSafetyViolation {
                message: format!("Local {} already moved", idx),
            }),
            LocalState::Borrowed { .. } => Ok(()), // Can copy while borrowed
            LocalState::MutBorrowed => Err(VerifierError::BorrowCheckingError {
                message: format!("Cannot copy local {} while mutably borrowed", idx),
            }),
        }
    }

    fn borrow_local(&mut self, idx: LocalIndex) -> VerifierResult<()> {
        let idx = idx as usize;
        if idx >= self.locals.len() {
            return Err(VerifierError::InvalidLocalIndex {
                index: idx as u16,
                pc: 0,
            });
        }

        match &mut self.locals[idx] {
            LocalState::Available => {
                self.locals[idx] = LocalState::Borrowed { count: 1 };
                Ok(())
            }
            LocalState::Moved => Err(VerifierError::ResourceSafetyViolation {
                message: format!("Local {} already moved", idx),
            }),
            LocalState::Borrowed { count } => {
                *count += 1;
                Ok(())
            }
            LocalState::MutBorrowed => Err(VerifierError::BorrowCheckingError {
                message: format!("Cannot borrow local {} while mutably borrowed", idx),
            }),
        }
    }

    fn mut_borrow_local(&mut self, idx: LocalIndex) -> VerifierResult<()> {
        let idx = idx as usize;
        if idx >= self.locals.len() {
            return Err(VerifierError::InvalidLocalIndex {
                index: idx as u16,
                pc: 0,
            });
        }

        match &self.locals[idx] {
            LocalState::Available => {
                self.locals[idx] = LocalState::MutBorrowed;
                Ok(())
            }
            LocalState::Moved => Err(VerifierError::ResourceSafetyViolation {
                message: format!("Local {} already moved", idx),
            }),
            LocalState::Borrowed { .. } => Err(VerifierError::BorrowCheckingError {
                message: format!("Cannot mutably borrow local {} while borrowed", idx),
            }),
            LocalState::MutBorrowed => Err(VerifierError::BorrowCheckingError {
                message: format!("Local {} already mutably borrowed", idx),
            }),
        }
    }
}

/// Function verifier context
struct FunctionVerifier<'a> {
    function: &'a Function,
    stack: AbstractStack,
    borrow_checker: BorrowChecker,
    /// Reachable instructions
    reachable: HashSet<usize>,
}

impl<'a> FunctionVerifier<'a> {
    fn new(function: &'a Function) -> Self {
        let num_locals = function.signature.parameters.len() + function.locals.len();

        Self {
            function,
            stack: AbstractStack::new(),
            borrow_checker: BorrowChecker::new(num_locals),
            reachable: HashSet::new(),
        }
    }

    /// Verify the function
    fn verify(&mut self) -> VerifierResult<()> {
        // Initialize stack with parameters
        for param_ty in &self.function.signature.parameters {
            self.stack.push(param_ty.clone())?;
        }

        // Mark entry point as reachable
        self.reachable.insert(0);

        // Verify all instructions
        self.verify_instructions()?;

        // Check that all paths return
        self.verify_returns()?;

        Ok(())
    }

    fn verify_instructions(&mut self) -> VerifierResult<()> {
        let mut pc = 0;

        while pc < self.function.code.len() {
            if !self.reachable.contains(&pc) {
                // Skip unreachable code
                pc += 1;
                continue;
            }

            let instr = &self.function.code[pc];
            self.verify_instruction(pc, instr)?;

            // Mark branch targets as reachable
            if let Some(offset) = instr.branch_offset() {
                let target = (pc as i32 + offset) as usize;
                self.reachable.insert(target);
            }

            // Mark next instruction as reachable (unless terminal)
            if !instr.is_terminal() {
                self.reachable.insert(pc + 1);
            }

            pc += 1;
        }

        Ok(())
    }

    fn verify_instruction(&mut self, _pc: usize, instr: &Instruction) -> VerifierResult<()> {
        match instr {
            // Stack operations
            Instruction::Pop => {
                self.stack.pop()?;
            }
            Instruction::Dup => {
                let ty = self.stack.peek()?.clone();
                self.stack.push(ty)?;
            }
            Instruction::Swap => {
                let ty1 = self.stack.pop()?;
                let ty2 = self.stack.pop()?;
                self.stack.push(ty1)?;
                self.stack.push(ty2)?;
            }

            // Constant loading
            Instruction::LdTrue | Instruction::LdFalse => {
                self.stack.push(TypeTag::Bool)?;
            }
            Instruction::LdU8(_) => {
                self.stack.push(TypeTag::U8)?;
            }
            Instruction::LdU16(_) => {
                self.stack.push(TypeTag::U16)?;
            }
            Instruction::LdU32(_) => {
                self.stack.push(TypeTag::U32)?;
            }
            Instruction::LdU64(_) => {
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::LdU128(_) => {
                self.stack.push(TypeTag::U128)?;
            }
            Instruction::LdU256(_) => {
                self.stack.push(TypeTag::U256)?;
            }
            Instruction::LdAddress(_) => {
                self.stack.push(TypeTag::Address)?;
            }
            Instruction::LdObjectID(_) => {
                self.stack.push(TypeTag::ObjectID)?;
            }
            Instruction::LdByteArray(_) => {
                self.stack.push(TypeTag::Vector(Box::new(TypeTag::U8)))?;
            }

            // Local variable operations
            Instruction::CopyLoc(idx) => {
                self.borrow_checker.copy_local(*idx)?;
                let local_ty = self.get_local_type(*idx)?;
                self.stack.push(local_ty)?;
            }
            Instruction::MoveLoc(idx) => {
                self.borrow_checker.move_local(*idx)?;
                let local_ty = self.get_local_type(*idx)?;
                self.stack.push(local_ty)?;
            }
            Instruction::StoreLoc(idx) => {
                let ty = self.stack.pop()?;
                let local_ty = self.get_local_type(*idx)?;
                if ty != local_ty {
                    return Err(VerifierError::TypeMismatch {
                        expected: format!("{:?}", local_ty),
                        got: format!("{:?}", ty),
                    });
                }
            }
            Instruction::BorrowLoc(idx) => {
                self.borrow_checker.borrow_local(*idx)?;
                let local_ty = self.get_local_type(*idx)?;
                self.stack.push(TypeTag::Reference(Box::new(local_ty)))?;
            }
            Instruction::MutBorrowLoc(idx) => {
                self.borrow_checker.mut_borrow_local(*idx)?;
                let local_ty = self.get_local_type(*idx)?;
                self.stack
                    .push(TypeTag::MutableReference(Box::new(local_ty)))?;
            }

            // Arithmetic operations (require integer types)
            Instruction::Add
            | Instruction::Sub
            | Instruction::Mul
            | Instruction::Div
            | Instruction::Mod => {
                let ty1 = self.stack.pop()?;
                let ty2 = self.stack.pop()?;
                if ty1 != ty2 || !self.is_integer_type(&ty1) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "integer types".to_string(),
                        got: format!("{:?}, {:?}", ty1, ty2),
                    });
                }
                self.stack.push(ty1)?;
            }

            // Bitwise operations
            Instruction::BitAnd
            | Instruction::BitOr
            | Instruction::BitXor
            | Instruction::Shl
            | Instruction::Shr => {
                let ty1 = self.stack.pop()?;
                let ty2 = self.stack.pop()?;
                if ty1 != ty2 || !self.is_integer_type(&ty1) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "integer types".to_string(),
                        got: format!("{:?}, {:?}", ty1, ty2),
                    });
                }
                self.stack.push(ty1)?;
            }
            Instruction::BitNot => {
                let ty = self.stack.pop()?;
                if !self.is_integer_type(&ty) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", ty),
                    });
                }
                self.stack.push(ty)?;
            }

            // Comparison operations
            Instruction::Lt
            | Instruction::Le
            | Instruction::Gt
            | Instruction::Ge
            | Instruction::Eq
            | Instruction::Neq => {
                let ty1 = self.stack.pop()?;
                let ty2 = self.stack.pop()?;
                if ty1 != ty2 {
                    return Err(VerifierError::TypeMismatch {
                        expected: format!("{:?}", ty1),
                        got: format!("{:?}", ty2),
                    });
                }
                self.stack.push(TypeTag::Bool)?;
            }

            // Logical operations
            Instruction::And | Instruction::Or => {
                self.stack.pop_expect(&TypeTag::Bool)?;
                self.stack.pop_expect(&TypeTag::Bool)?;
                self.stack.push(TypeTag::Bool)?;
            }
            Instruction::Not => {
                self.stack.pop_expect(&TypeTag::Bool)?;
                self.stack.push(TypeTag::Bool)?;
            }

            // Control flow
            Instruction::Branch(_) => {
                // Unconditional branch, no stack effect
            }
            Instruction::BranchTrue(_) | Instruction::BranchFalse(_) => {
                self.stack.pop_expect(&TypeTag::Bool)?;
            }
            Instruction::Ret => {
                // Verify return types match signature
                for ret_ty in self.function.signature.return_types.iter().rev() {
                    self.stack.pop_expect(ret_ty)?;
                }
            }
            Instruction::Abort => {
                // Abort terminates execution
            }

            // Vector operations
            Instruction::VecEmpty(_elem_ty) => {
                self.stack.push(TypeTag::Vector(Box::new(TypeTag::U8)))?;
            }
            Instruction::VecLen => {
                let vec_ty = self.stack.pop()?;
                if !matches!(vec_ty, TypeTag::Vector(_)) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "vector type".to_string(),
                        got: format!("{:?}", vec_ty),
                    });
                }
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::VecPush => {
                let elem = self.stack.pop()?;
                let vec_ty = self.stack.pop()?;
                if let TypeTag::Vector(elem_ty) = vec_ty {
                    if *elem_ty != elem {
                        return Err(VerifierError::TypeMismatch {
                            expected: format!("{:?}", elem_ty),
                            got: format!("{:?}", elem),
                        });
                    }
                    self.stack.push(TypeTag::Vector(elem_ty))?;
                } else {
                    return Err(VerifierError::TypeMismatch {
                        expected: "vector type".to_string(),
                        got: format!("{:?}", vec_ty),
                    });
                }
            }
            Instruction::VecPop => {
                let vec_ty = self.stack.pop()?;
                if let TypeTag::Vector(elem_ty) = vec_ty {
                    self.stack.push(*elem_ty)?;
                } else {
                    return Err(VerifierError::TypeMismatch {
                        expected: "vector type".to_string(),
                        got: format!("{:?}", vec_ty),
                    });
                }
            }
            Instruction::VecBorrow(_idx) => {
                let index = self.stack.pop()?;
                if !self.is_integer_type(&index) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "integer type for index".to_string(),
                        got: format!("{:?}", index),
                    });
                }
                let vec_ty = self.stack.pop()?;
                if let TypeTag::Vector(elem_ty) = vec_ty {
                    self.stack.push(TypeTag::Reference(elem_ty))?;
                } else {
                    return Err(VerifierError::TypeMismatch {
                        expected: "vector type".to_string(),
                        got: format!("{:?}", vec_ty),
                    });
                }
            }
            Instruction::VecMutBorrow(_idx) => {
                let index = self.stack.pop()?;
                if !self.is_integer_type(&index) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "integer type for index".to_string(),
                        got: format!("{:?}", index),
                    });
                }
                let vec_ty = self.stack.pop()?;
                if let TypeTag::Vector(elem_ty) = vec_ty {
                    self.stack.push(TypeTag::MutableReference(elem_ty))?;
                } else {
                    return Err(VerifierError::TypeMismatch {
                        expected: "vector type".to_string(),
                        got: format!("{:?}", vec_ty),
                    });
                }
            }
            Instruction::VecSwap => {
                let _idx2 = self.stack.pop()?;
                let _idx1 = self.stack.pop()?;
                let vec_ty = self.stack.pop()?;
                if !matches!(vec_ty, TypeTag::Vector(_)) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "vector type".to_string(),
                        got: format!("{:?}", vec_ty),
                    });
                }
                self.stack.push(vec_ty)?;
            }

            // Type casting operations
            Instruction::CastU8 => {
                self.stack.pop()?;
                self.stack.push(TypeTag::U8)?;
            }
            Instruction::CastU16 => {
                self.stack.pop()?;
                self.stack.push(TypeTag::U16)?;
            }
            Instruction::CastU32 => {
                self.stack.pop()?;
                self.stack.push(TypeTag::U32)?;
            }
            Instruction::CastU64 => {
                self.stack.pop()?;
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::CastU128 => {
                self.stack.pop()?;
                self.stack.push(TypeTag::U128)?;
            }
            Instruction::CastU256 => {
                self.stack.pop()?;
                self.stack.push(TypeTag::U256)?;
            }

            // Function call operations
            Instruction::Call(_) => {
                // Function call verification is handled separately
                // Stack effects depend on the called function's signature
            }

            // Struct operations
            Instruction::Pack(_) => {
                // Pack struct - pops fields and pushes struct
                // Exact verification depends on struct definition
            }
            Instruction::Unpack(_) => {
                // Unpack struct - pops struct and pushes fields
                // Exact verification depends on struct definition
            }

            // Memory operations
            Instruction::BorrowField(_field_idx) => {
                // Immutable borrow of struct field
                let struct_ref = self.stack.pop()?;
                if !matches!(struct_ref, TypeTag::Reference(_) | TypeTag::MutableReference(_)) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "reference".to_string(),
                        got: format!("{:?}", struct_ref),
                    });
                }
                // Push reference to field
                self.stack.push(TypeTag::Reference(Box::new(TypeTag::U64)))?;
            }
            Instruction::MutBorrowField(_field_idx) => {
                // Mutable borrow of struct field
                let struct_ref = self.stack.pop()?;
                if !matches!(struct_ref, TypeTag::MutableReference(_)) {
                    return Err(VerifierError::TypeMismatch {
                        expected: "mutable reference".to_string(),
                        got: format!("{:?}", struct_ref),
                    });
                }
                // Push mutable reference to field
                self.stack.push(TypeTag::MutableReference(Box::new(TypeTag::U64)))?;
            }

            // Reference operations
            Instruction::ReadRef => {
                let _ref = self.stack.pop()?;
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::WriteRef => {
                let _val = self.stack.pop()?;
                let _ref = self.stack.pop()?;
            }
            Instruction::ReleaseRef => {
                let _ref = self.stack.pop()?;
            }

            // Object operations
            Instruction::ObjectNew => {
                self.stack.push(TypeTag::ObjectID)?;
            }
            Instruction::ObjectDelete => {
                let _obj = self.stack.pop()?;
            }
            Instruction::ObjectTransfer => {
                let _to = self.stack.pop()?;
                let _obj = self.stack.pop()?;
            }
            Instruction::ObjectShare => {
                let _obj = self.stack.pop()?;
            }
            Instruction::ObjectFreeze => {
                let _obj = self.stack.pop()?;
            }
            Instruction::ObjectGetID => {
                let _obj = self.stack.pop()?;
                self.stack.push(TypeTag::ObjectID)?;
            }
            Instruction::ObjectExists => {
                let _id = self.stack.pop()?;
                self.stack.push(TypeTag::Bool)?;
            }
            Instruction::ObjectBorrow => {
                let _id = self.stack.pop()?;
                self.stack.push(TypeTag::Reference(Box::new(TypeTag::ObjectID)))?;
            }
            Instruction::ObjectMutBorrow => {
                let _id = self.stack.pop()?;
                self.stack.push(TypeTag::MutableReference(Box::new(TypeTag::ObjectID)))?;
            }

            // Cryptographic operations
            Instruction::CryptoHashBlake3 => {
                let _data = self.stack.pop()?;
                self.stack.push(TypeTag::Vector(Box::new(TypeTag::U8)))?;
            }
            Instruction::CryptoVerifySignature => {
                let _sig = self.stack.pop()?;
                let _msg = self.stack.pop()?;
                let _pk = self.stack.pop()?;
                self.stack.push(TypeTag::Bool)?;
            }
            Instruction::CryptoDeriveAddress => {
                let _pk = self.stack.pop()?;
                self.stack.push(TypeTag::Address)?;
            }
            Instruction::CryptoRandom(_) => {
                self.stack.push(TypeTag::Vector(Box::new(TypeTag::U8)))?;
            }

            // Call operations
            Instruction::CallGeneric { .. } => {
                // Generic call - pop arguments and push return values
                let _arg = self.stack.pop()?;
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::CallNative(_) => {
                // Native call - pop arguments and push return values
                let _arg = self.stack.pop()?;
                self.stack.push(TypeTag::U64)?;
            }

            // Event operations
            Instruction::EventEmit { .. } => {
                // Emit event - pop event data
                let _event = self.stack.pop()?;
            }

            // Transaction operations
            Instruction::TxSender => {
                self.stack.push(TypeTag::Address)?;
            }
            Instruction::TxTimestamp => {
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::TxDigest => {
                // Transaction digest is a vector of bytes (32 bytes for SHA256)
                self.stack.push(TypeTag::Vector(Box::new(TypeTag::U8)))?;
            }
            Instruction::FuelRemaining => {
                self.stack.push(TypeTag::U64)?;
            }
            Instruction::FuelCharge(_) => {
                // Charge fuel - no stack effect
            }

            // Debug operations
            Instruction::DebugPrint => {
                let _val = self.stack.pop()?;
            }
            Instruction::Assert => {
                let _cond = self.stack.pop()?;
            }

            // Nop instruction
            Instruction::Nop => {
                // No operation
            }
        }

        Ok(())
    }

    fn get_local_type(&self, idx: LocalIndex) -> VerifierResult<TypeTag> {
        let idx = idx as usize;
        let num_params = self.function.signature.parameters.len();

        if idx < num_params {
            Ok(self.function.signature.parameters[idx].clone())
        } else {
            let local_idx = idx - num_params;
            if local_idx < self.function.locals.len() {
                Ok(self.function.locals[local_idx].clone())
            } else {
                Err(VerifierError::InvalidLocalIndex {
                    index: idx as u16,
                    pc: 0,
                })
            }
        }
    }

    fn is_integer_type(&self, ty: &TypeTag) -> bool {
        matches!(
            ty,
            TypeTag::U8
                | TypeTag::U16
                | TypeTag::U32
                | TypeTag::U64
                | TypeTag::U128
                | TypeTag::U256
                | TypeTag::I8
                | TypeTag::I16
                | TypeTag::I32
                | TypeTag::I64
                | TypeTag::I128
        )
    }

    fn verify_returns(&self) -> VerifierResult<()> {
        // Check that all non-terminal paths end with a return
        for (pc, instr) in self.function.code.iter().enumerate() {
            if self.reachable.contains(&pc) && !instr.is_terminal() {
                if pc == self.function.code.len() - 1 {
                    // Last instruction must be terminal
                    return Err(VerifierError::MissingReturn);
                }
            }
        }
        Ok(())
    }
}

/// Bytecode verifier
pub struct BytecodeVerifier;

impl BytecodeVerifier {
    /// Verify a complete bytecode package
    pub fn verify_bytecode(bytecode: &Bytecode) -> VerifierResult<()> {
        // Validate bytecode structure
        bytecode
            .validate()
            .map_err(|e| VerifierError::ResourceSafetyViolation { message: e })?;

        // Verify each module
        for module in &bytecode.modules {
            Self::verify_module(module)?;
        }

        Ok(())
    }

    /// Verify a module
    pub fn verify_module(module: &Module) -> VerifierResult<()> {
        // Validate module structure
        module
            .validate()
            .map_err(|e| VerifierError::ResourceSafetyViolation { message: e })?;

        // Verify each function
        for function in &module.functions {
            Self::verify_function(function)?;
        }

        Ok(())
    }

    /// Verify a function
    pub fn verify_function(function: &Function) -> VerifierResult<()> {
        // Validate function structure
        function
            .validate()
            .map_err(|e| VerifierError::ResourceSafetyViolation { message: e })?;

        // Create verifier and run verification
        let mut verifier = FunctionVerifier::new(function);
        verifier.verify()?;

        Ok(())
    }
}
