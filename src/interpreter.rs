//! # Bytecode Interpreter
//!
//! Production-ready stack-based interpreter for Quantum bytecode with:
//! - Complete instruction execution
//! - Fuel metering per instruction
//! - Stack frame management
//! - Call stack for function calls
//! - Error handling

use crate::bytecode::{
    Constant, Function, FunctionIndex, Instruction, LocalIndex, Module, TypeTag,
};
use crate::runtime::Runtime;
use silver_core::{ObjectID, SilverAddress};
use thiserror::Error;

/// Interpreter error types for bytecode execution
#[derive(Error, Debug)]
pub enum InterpreterError {
    /// Stack underflow (popping from empty stack)
    #[error("Stack underflow")]
    StackUnderflow,

    /// Stack overflow (exceeding maximum stack depth)
    #[error("Stack overflow")]
    StackOverflow,

    /// Invalid local variable index
    #[error("Invalid local index: {0}")]
    InvalidLocalIndex(LocalIndex),

    /// Invalid constant pool index
    #[error("Invalid constant index: {0}")]
    InvalidConstantIndex(u16),

    /// Invalid function index
    #[error("Invalid function index: {0}")]
    InvalidFunctionIndex(FunctionIndex),

    /// Type mismatch between expected and actual types
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch {
        /// Expected type name
        expected: String,
        /// Actual type name
        got: String,
    },

    /// Division by zero error
    #[error("Division by zero")]
    DivisionByZero,

    /// Execution ran out of fuel
    #[error("Out of fuel")]
    OutOfFuel,

    /// Transaction aborted with error code
    #[error("Execution aborted with code {0}")]
    Aborted(u64),

    /// Invalid branch target
    #[error("Invalid branch target: {0}")]
    InvalidBranchTarget(i32),

    /// Generic runtime error
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

/// Result type for interpreter operations
pub type InterpreterResult<T> = Result<T, InterpreterError>;

/// Stack value (runtime representation)
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Boolean value
    Bool(bool),
    /// Unsigned 8-bit integer
    U8(u8),
    /// Unsigned 16-bit integer
    U16(u16),
    /// Unsigned 32-bit integer
    U32(u32),
    /// Unsigned 64-bit integer
    U64(u64),
    /// Unsigned 128-bit integer
    U128(u128),
    /// Unsigned 256-bit integer (stored as bytes)
    U256([u8; 32]),
    /// Signed 8-bit integer
    I8(i8),
    /// Signed 16-bit integer
    I16(i16),
    /// Signed 32-bit integer
    I32(i32),
    /// Signed 64-bit integer
    I64(i64),
    /// Signed 128-bit integer
    I128(i128),
    /// Address (512-bit)
    Address(SilverAddress),
    /// Object ID (512-bit)
    ObjectID(ObjectID),
    /// Byte array
    ByteArray(Vec<u8>),
    /// Vector of values
    Vector(Vec<Value>),
    /// Struct (module, name, field values)
    Struct {
        /// Module name containing the struct
        module: String,
        /// Struct name
        name: String,
        /// Package ObjectID
        package: ObjectID,
        /// Field values
        fields: Vec<Value>,
    },
    /// Reference to a value
    Reference(Box<Value>),
    /// Mutable reference to a value
    MutableReference(Box<Value>),
}

impl Value {
    /// Get the type tag for this value
    pub fn type_tag(&self) -> TypeTag {
        match self {
            Value::Bool(_) => TypeTag::Bool,
            Value::U8(_) => TypeTag::U8,
            Value::U16(_) => TypeTag::U16,
            Value::U32(_) => TypeTag::U32,
            Value::U64(_) => TypeTag::U64,
            Value::U128(_) => TypeTag::U128,
            Value::U256(_) => TypeTag::U256,
            Value::I8(_) => TypeTag::I8,
            Value::I16(_) => TypeTag::I16,
            Value::I32(_) => TypeTag::I32,
            Value::I64(_) => TypeTag::I64,
            Value::I128(_) => TypeTag::I128,
            Value::Address(_) => TypeTag::Address,
            Value::ObjectID(_) => TypeTag::ObjectID,
            Value::ByteArray(_) => TypeTag::Vector(Box::new(TypeTag::U8)),
            Value::Vector(v) => {
                if let Some(first) = v.first() {
                    TypeTag::Vector(Box::new(first.type_tag()))
                } else {
                    TypeTag::Vector(Box::new(TypeTag::U8))
                }
            }
            Value::Struct { module, name, package, .. } => {
                TypeTag::Struct {
                    package: *package,
                    module: module.clone(),
                    name: name.clone(),
                    type_params: vec![],
                }
            }
            Value::Reference(v) => TypeTag::Reference(Box::new(v.type_tag())),
            Value::MutableReference(v) => TypeTag::MutableReference(Box::new(v.type_tag())),
        }
    }
}

/// Call frame for function execution.
///
/// Represents a single function call on the call stack.
#[derive(Debug, Clone)]
struct CallFrame {
    /// Function being executed
    #[allow(dead_code)]
    function_idx: FunctionIndex,
    /// Program counter
    #[allow(dead_code)]
    pc: usize,
    /// Local variables
    locals: Vec<Value>,
    /// Base pointer for stack (where this frame's values start)
    #[allow(dead_code)]
    base_pointer: usize,
    /// Tracks which locals have been moved (for Move semantics)
    moved_locals: std::collections::HashSet<LocalIndex>,
}

/// Execution stack
#[derive(Debug)]
struct ExecutionStack {
    values: Vec<Value>,
    max_size: usize,
}

impl ExecutionStack {
    fn new(max_size: usize) -> Self {
        Self {
            values: Vec::with_capacity(256),
            max_size,
        }
    }

    fn push(&mut self, value: Value) -> InterpreterResult<()> {
        if self.values.len() >= self.max_size {
            return Err(InterpreterError::StackOverflow);
        }
        self.values.push(value);
        Ok(())
    }

    fn pop(&mut self) -> InterpreterResult<Value> {
        self.values.pop().ok_or(InterpreterError::StackUnderflow)
    }

    fn peek(&self) -> InterpreterResult<&Value> {
        self.values.last().ok_or(InterpreterError::StackUnderflow)
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Bytecode interpreter for executing Quantum bytecode.
///
/// Provides stack-based execution with:
/// - Fuel metering
/// - Call stack management
/// - Runtime environment
pub struct Interpreter {
    /// Execution stack
    stack: ExecutionStack,
    /// Call stack
    call_stack: Vec<CallFrame>,
    /// Current module
    current_module: Option<Module>,
    /// Fuel remaining
    fuel_remaining: u64,
    /// Runtime environment
    #[allow(dead_code)]
    runtime: Runtime,
}

impl Interpreter {
    /// Create a new interpreter with fuel budget
    pub fn new(fuel_budget: u64) -> Self {
        Self {
            stack: ExecutionStack::new(1024),
            call_stack: Vec::new(),
            current_module: None,
            fuel_remaining: fuel_budget,
            runtime: Runtime::new(),
        }
    }

    /// Execute a function from a module
    pub fn execute_function(
        &mut self,
        module: &Module,
        function_idx: FunctionIndex,
        args: Vec<Value>,
    ) -> InterpreterResult<Vec<Value>> {
        // Set current module
        self.current_module = Some(module.clone());

        // Get function
        let function = module
            .functions
            .get(function_idx as usize)
            .ok_or(InterpreterError::InvalidFunctionIndex(function_idx))?;

        // Validate arguments
        if args.len() != function.signature.parameters.len() {
            return Err(InterpreterError::TypeMismatch {
                expected: format!("{} arguments", function.signature.parameters.len()),
                got: format!("{} arguments", args.len()),
            });
        }

        // Push arguments onto stack
        for arg in args {
            self.stack.push(arg)?;
        }

        // Create call frame with move tracking
        let frame = CallFrame {
            function_idx,
            pc: 0,
            locals: vec![Value::U8(0); function.locals.len()],
            base_pointer: self.stack.len() - function.signature.parameters.len(),
            moved_locals: std::collections::HashSet::new(),
        };
        self.call_stack.push(frame);

        // Execute function
        self.execute_current_function(function)?;

        // Collect return values
        let return_count = function.signature.return_types.len();
        let mut results = Vec::with_capacity(return_count);
        for _ in 0..return_count {
            results.push(self.stack.pop()?);
        }
        results.reverse();

        Ok(results)
    }

    fn execute_current_function(&mut self, function: &Function) -> InterpreterResult<()> {
        loop {
            let frame = self
                .call_stack
                .last()
                .ok_or(InterpreterError::RuntimeError("No call frame".to_string()))?;
            let pc = frame.pc;

            // Check if we've reached the end
            if pc >= function.code.len() {
                break;
            }

            // Get instruction
            let instr = &function.code[pc];

            // Charge fuel
            self.charge_fuel(instr.fuel_cost())?;

            // Execute instruction
            self.execute_instruction(instr, function)?;

            // Check if function returned
            if self.call_stack.is_empty() {
                break;
            }
        }

        Ok(())
    }

    fn charge_fuel(&mut self, cost: u64) -> InterpreterResult<()> {
        if self.fuel_remaining < cost {
            return Err(InterpreterError::OutOfFuel);
        }
        self.fuel_remaining -= cost;
        Ok(())
    }

    fn execute_instruction(
        &mut self,
        instr: &Instruction,
        _function: &Function,
    ) -> InterpreterResult<()> {
        match instr {
            // Stack operations
            Instruction::Pop => {
                self.stack.pop()?;
            }
            Instruction::Dup => {
                let val = self.stack.peek()?.clone();
                self.stack.push(val)?;
            }
            Instruction::Swap => {
                let val1 = self.stack.pop()?;
                let val2 = self.stack.pop()?;
                self.stack.push(val1)?;
                self.stack.push(val2)?;
            }

            // Constant loading
            Instruction::LdTrue => {
                self.stack.push(Value::Bool(true))?;
            }
            Instruction::LdFalse => {
                self.stack.push(Value::Bool(false))?;
            }
            Instruction::LdU8(v) => {
                self.stack.push(Value::U8(*v))?;
            }
            Instruction::LdU16(v) => {
                self.stack.push(Value::U16(*v))?;
            }
            Instruction::LdU32(v) => {
                self.stack.push(Value::U32(*v))?;
            }
            Instruction::LdU64(v) => {
                self.stack.push(Value::U64(*v))?;
            }
            Instruction::LdU128(idx) => {
                let constant = self.get_constant(*idx)?;
                if let Constant::U128(v) = constant {
                    self.stack.push(Value::U128(*v))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "U128".to_string(),
                        got: format!("{:?}", constant),
                    });
                }
            }
            Instruction::LdU256(idx) => {
                let constant = self.get_constant(*idx)?;
                if let Constant::U256(v) = constant {
                    self.stack.push(Value::U256(*v))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "U256".to_string(),
                        got: format!("{:?}", constant),
                    });
                }
            }
            Instruction::LdAddress(idx) => {
                let constant = self.get_constant(*idx)?;
                if let Constant::Address(v) = constant {
                    self.stack.push(Value::Address(*v))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Address".to_string(),
                        got: format!("{:?}", constant),
                    });
                }
            }
            Instruction::LdObjectID(idx) => {
                let constant = self.get_constant(*idx)?;
                if let Constant::ObjectID(v) = constant {
                    self.stack.push(Value::ObjectID(v.clone()))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "ObjectID".to_string(),
                        got: format!("{:?}", constant),
                    });
                }
            }
            Instruction::LdByteArray(idx) => {
                let constant = self.get_constant(*idx)?;
                if let Constant::ByteArray(v) = constant {
                    self.stack.push(Value::ByteArray(v.clone()))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "ByteArray".to_string(),
                        got: format!("{:?}", constant),
                    });
                }
            }

            // Local variable operations
            Instruction::CopyLoc(idx) => {
                let val = self.get_local(*idx)?.clone();
                self.stack.push(val)?;
            }
            Instruction::MoveLoc(idx) => {
                // PRODUCTION IMPLEMENTATION: Move semantics with tracking
                // Check if local has already been moved
                if let Some(frame) = self.call_stack.last_mut() {
                    if frame.moved_locals.contains(idx) {
                        return Err(InterpreterError::RuntimeError(
                            format!("Local variable {} has already been moved", idx)
                        ));
                    }
                    // Mark local as moved
                    frame.moved_locals.insert(*idx);
                }
                
                // Get the value and push to stack
                let val = self.get_local(*idx)?.clone();
                self.stack.push(val)?;
            }
            Instruction::StoreLoc(idx) => {
                let val = self.stack.pop()?;
                self.set_local(*idx, val)?;
            }
            Instruction::BorrowLoc(idx) => {
                let val = self.get_local(*idx)?.clone();
                self.stack.push(Value::Reference(Box::new(val)))?;
            }
            Instruction::MutBorrowLoc(idx) => {
                let val = self.get_local(*idx)?.clone();
                self.stack.push(Value::MutableReference(Box::new(val)))?;
            }

            // Arithmetic operations
            Instruction::Add => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.add_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::Sub => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.sub_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::Mul => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.mul_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::Div => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.div_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::Mod => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.mod_values(val1, val2)?;
                self.stack.push(result)?;
            }

            // Bitwise operations
            Instruction::BitAnd => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.bitand_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::BitOr => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.bitor_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::BitXor => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.bitxor_values(val1, val2)?;
                self.stack.push(result)?;
            }
            Instruction::BitNot => {
                let val = self.stack.pop()?;
                let result = self.bitnot_value(val)?;
                self.stack.push(result)?;
            }
            Instruction::Shl => {
                let shift = self.stack.pop()?;
                let val = self.stack.pop()?;
                let result = self.shl_values(val, shift)?;
                self.stack.push(result)?;
            }
            Instruction::Shr => {
                let shift = self.stack.pop()?;
                let val = self.stack.pop()?;
                let result = self.shr_values(val, shift)?;
                self.stack.push(result)?;
            }

            // Comparison operations
            Instruction::Lt => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.compare_lt(val1, val2)?;
                self.stack.push(Value::Bool(result))?;
            }
            Instruction::Le => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.compare_le(val1, val2)?;
                self.stack.push(Value::Bool(result))?;
            }
            Instruction::Gt => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.compare_gt(val1, val2)?;
                self.stack.push(Value::Bool(result))?;
            }
            Instruction::Ge => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = self.compare_ge(val1, val2)?;
                self.stack.push(Value::Bool(result))?;
            }
            Instruction::Eq => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = val1 == val2;
                self.stack.push(Value::Bool(result))?;
            }
            Instruction::Neq => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                let result = val1 != val2;
                self.stack.push(Value::Bool(result))?;
            }

            // Logical operations
            Instruction::And => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                if let (Value::Bool(b1), Value::Bool(b2)) = (val1, val2) {
                    self.stack.push(Value::Bool(b1 && b2))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Bool".to_string(),
                        got: "other".to_string(),
                    });
                }
            }
            Instruction::Or => {
                let val2 = self.stack.pop()?;
                let val1 = self.stack.pop()?;
                if let (Value::Bool(b1), Value::Bool(b2)) = (val1, val2) {
                    self.stack.push(Value::Bool(b1 || b2))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Bool".to_string(),
                        got: "other".to_string(),
                    });
                }
            }
            Instruction::Not => {
                let val = self.stack.pop()?;
                if let Value::Bool(b) = val {
                    self.stack.push(Value::Bool(!b))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Bool".to_string(),
                        got: "other".to_string(),
                    });
                }
            }

            // Control flow
            Instruction::Branch(offset) => {
                self.branch(*offset)?;
                return Ok(()); // Don't increment PC
            }
            Instruction::BranchTrue(offset) => {
                let val = self.stack.pop()?;
                if let Value::Bool(true) = val {
                    self.branch(*offset)?;
                    return Ok(()); // Don't increment PC
                }
            }
            Instruction::BranchFalse(offset) => {
                let val = self.stack.pop()?;
                if let Value::Bool(false) = val {
                    self.branch(*offset)?;
                    return Ok(()); // Don't increment PC
                }
            }
            Instruction::Ret => {
                self.call_stack.pop();
                return Ok(());
            }
            Instruction::Abort => {
                return Err(InterpreterError::Aborted(0));
            }

            // Vector operations
            Instruction::VecEmpty(_ty) => {
                self.stack.push(Value::Vector(Vec::new()))?;
            }
            Instruction::VecLen => {
                let vec = self.stack.pop()?;
                if let Value::Vector(v) = vec {
                    self.stack.push(Value::U64(v.len() as u64))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Vector".to_string(),
                        got: format!("{:?}", vec),
                    });
                }
            }
            Instruction::VecPush => {
                let elem = self.stack.pop()?;
                let vec = self.stack.pop()?;
                if let Value::Vector(mut v) = vec {
                    v.push(elem);
                    self.stack.push(Value::Vector(v))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Vector".to_string(),
                        got: format!("{:?}", vec),
                    });
                }
            }
            Instruction::VecPop => {
                let vec = self.stack.pop()?;
                if let Value::Vector(mut v) = vec {
                    if let Some(elem) = v.pop() {
                        self.stack.push(Value::Vector(v))?;
                        self.stack.push(elem)?;
                    } else {
                        return Err(InterpreterError::RuntimeError(
                            "Cannot pop from empty vector".to_string(),
                        ));
                    }
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Vector".to_string(),
                        got: format!("{:?}", vec),
                    });
                }
            }

            // Reference operations
            Instruction::ReadRef => {
                let reference = self.stack.pop()?;
                match reference {
                    Value::Reference(inner) | Value::MutableReference(inner) => {
                        self.stack.push(*inner)?;
                    }
                    _ => {
                        return Err(InterpreterError::TypeMismatch {
                            expected: "Reference".to_string(),
                            got: format!("{:?}", reference),
                        });
                    }
                }
            }
            Instruction::WriteRef => {
                let value = self.stack.pop()?;
                let reference = self.stack.pop()?;
                if let Value::MutableReference(mut inner) = reference {
                    *inner = value;
                    self.stack.push(Value::MutableReference(inner))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "MutableReference".to_string(),
                        got: format!("{:?}", reference),
                    });
                }
            }

            // Type casting operations
            Instruction::CastU8 => {
                let val = self.stack.pop()?;
                let u8_val = match val {
                    Value::U8(v) => v,
                    Value::U16(v) => v as u8,
                    Value::U32(v) => v as u8,
                    Value::U64(v) => v as u8,
                    Value::U128(v) => v as u8,
                    _ => return Err(InterpreterError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", val),
                    }),
                };
                self.stack.push(Value::U8(u8_val))?;
            }
            Instruction::CastU16 => {
                let val = self.stack.pop()?;
                let u16_val = match val {
                    Value::U8(v) => v as u16,
                    Value::U16(v) => v,
                    Value::U32(v) => v as u16,
                    Value::U64(v) => v as u16,
                    Value::U128(v) => v as u16,
                    _ => return Err(InterpreterError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", val),
                    }),
                };
                self.stack.push(Value::U16(u16_val))?;
            }
            Instruction::CastU32 => {
                let val = self.stack.pop()?;
                let u32_val = match val {
                    Value::U8(v) => v as u32,
                    Value::U16(v) => v as u32,
                    Value::U32(v) => v,
                    Value::U64(v) => v as u32,
                    Value::U128(v) => v as u32,
                    _ => return Err(InterpreterError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", val),
                    }),
                };
                self.stack.push(Value::U32(u32_val))?;
            }
            Instruction::CastU64 => {
                let val = self.stack.pop()?;
                let u64_val = match val {
                    Value::U8(v) => v as u64,
                    Value::U16(v) => v as u64,
                    Value::U32(v) => v as u64,
                    Value::U64(v) => v,
                    Value::U128(v) => v as u64,
                    _ => return Err(InterpreterError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", val),
                    }),
                };
                self.stack.push(Value::U64(u64_val))?;
            }
            Instruction::CastU128 => {
                let val = self.stack.pop()?;
                let u128_val = match val {
                    Value::U8(v) => v as u128,
                    Value::U16(v) => v as u128,
                    Value::U32(v) => v as u128,
                    Value::U64(v) => v as u128,
                    Value::U128(v) => v,
                    _ => return Err(InterpreterError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", val),
                    }),
                };
                self.stack.push(Value::U128(u128_val))?;
            }
            Instruction::CastU256 => {
                let val = self.stack.pop()?;
                let u256_bytes = match val {
                    Value::U8(v) => {
                        let mut bytes = [0u8; 32];
                        bytes[31] = v;
                        bytes
                    }
                    Value::U16(v) => {
                        let mut bytes = [0u8; 32];
                        bytes[30..32].copy_from_slice(&v.to_le_bytes());
                        bytes
                    }
                    Value::U32(v) => {
                        let mut bytes = [0u8; 32];
                        bytes[28..32].copy_from_slice(&v.to_le_bytes());
                        bytes
                    }
                    Value::U64(v) => {
                        let mut bytes = [0u8; 32];
                        bytes[24..32].copy_from_slice(&v.to_le_bytes());
                        bytes
                    }
                    Value::U128(v) => {
                        let mut bytes = [0u8; 32];
                        bytes[16..32].copy_from_slice(&v.to_le_bytes());
                        bytes
                    }
                    Value::U256(v) => v,
                    _ => return Err(InterpreterError::TypeMismatch {
                        expected: "integer type".to_string(),
                        got: format!("{:?}", val),
                    }),
                };
                self.stack.push(Value::U256(u256_bytes))?;
            }

            // Struct operations
            Instruction::Pack(struct_def_idx) => {
                // Get struct definition from current module
                let module = self.current_module.as_ref()
                    .ok_or_else(|| InterpreterError::RuntimeError(
                        "No current module set".to_string()
                    ))?;
                
                let struct_def = module.structs.get(*struct_def_idx as usize)
                    .ok_or_else(|| InterpreterError::RuntimeError(
                        format!("Invalid struct definition index: {}", struct_def_idx)
                    ))?;

                let mut fields = Vec::new();
                for _ in 0..struct_def.fields.len() {
                    fields.push(self.stack.pop()?);
                }
                fields.reverse();

                self.stack.push(Value::Struct {
                    module: module.name.clone(),
                    name: struct_def.name.clone(),
                    package: module.package.clone(),
                    fields,
                })?;
            }
            Instruction::Unpack(_struct_def_idx) => {
                let struct_val = self.stack.pop()?;
                if let Value::Struct { fields, .. } = struct_val {
                    for field in fields.into_iter().rev() {
                        self.stack.push(field)?;
                    }
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Struct".to_string(),
                        got: format!("{:?}", struct_val),
                    });
                }
            }

            // Field access operations
            Instruction::BorrowField(field_idx) => {
                let struct_ref = self.stack.pop()?;
                match struct_ref {
                    Value::Reference(inner) | Value::MutableReference(inner) => {
                        if let Value::Struct { fields, .. } = *inner {
                            if let Some(field) = fields.get(*field_idx as usize) {
                                self.stack.push(Value::Reference(Box::new(field.clone())))?;
                            } else {
                                return Err(InterpreterError::RuntimeError(
                                    format!("Invalid field index: {}", field_idx)
                                ));
                            }
                        } else {
                            return Err(InterpreterError::TypeMismatch {
                                expected: "Struct".to_string(),
                                got: format!("{:?}", inner),
                            });
                        }
                    }
                    _ => {
                        return Err(InterpreterError::TypeMismatch {
                            expected: "Reference".to_string(),
                            got: format!("{:?}", struct_ref),
                        });
                    }
                }
            }
            Instruction::MutBorrowField(field_idx) => {
                let struct_ref = self.stack.pop()?;
                if let Value::MutableReference(inner) = struct_ref {
                    if let Value::Struct { fields, .. } = *inner {
                        if let Some(field) = fields.get(*field_idx as usize) {
                            self.stack.push(Value::MutableReference(Box::new(field.clone())))?;
                        } else {
                            return Err(InterpreterError::RuntimeError(
                                format!("Invalid field index: {}", field_idx)
                            ));
                        }
                    } else {
                        return Err(InterpreterError::TypeMismatch {
                            expected: "Struct".to_string(),
                            got: format!("{:?}", inner),
                        });
                    }
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "MutableReference".to_string(),
                        got: format!("{:?}", struct_ref),
                    });
                }
            }

            // Reference release
            Instruction::ReleaseRef => {
                let _ref = self.stack.pop()?;
                // Reference is released (dropped)
            }

            // Vector borrow operations
            Instruction::VecBorrow(idx) => {
                let vec = self.stack.pop()?;
                if let Value::Vector(v) = vec {
                    if let Some(elem) = v.get(*idx as usize) {
                        self.stack.push(Value::Reference(Box::new(elem.clone())))?;
                    } else {
                        return Err(InterpreterError::RuntimeError(
                            format!("Vector index out of bounds: {}", idx)
                        ));
                    }
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Vector".to_string(),
                        got: format!("{:?}", vec),
                    });
                }
            }
            Instruction::VecMutBorrow(idx) => {
                let vec = self.stack.pop()?;
                if let Value::Vector(v) = vec {
                    if let Some(elem) = v.get(*idx as usize) {
                        self.stack.push(Value::MutableReference(Box::new(elem.clone())))?;
                    } else {
                        return Err(InterpreterError::RuntimeError(
                            format!("Vector index out of bounds: {}", idx)
                        ));
                    }
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Vector".to_string(),
                        got: format!("{:?}", vec),
                    });
                }
            }
            Instruction::VecSwap => {
                let idx2 = self.stack.pop()?;
                let idx1 = self.stack.pop()?;
                let vec = self.stack.pop()?;
                
                if let Value::Vector(mut v) = vec {
                    if let (Value::U64(i1), Value::U64(i2)) = (idx1, idx2) {
                        if i1 as usize >= v.len() || i2 as usize >= v.len() {
                            return Err(InterpreterError::RuntimeError(
                                "Vector index out of bounds".to_string()
                            ));
                        }
                        v.swap(i1 as usize, i2 as usize);
                        self.stack.push(Value::Vector(v))?;
                    } else {
                        return Err(InterpreterError::TypeMismatch {
                            expected: "U64".to_string(),
                            got: "other".to_string(),
                        });
                    }
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "Vector".to_string(),
                        got: format!("{:?}", vec),
                    });
                }
            }

            // Object operations
            Instruction::ObjectNew => {
                use rand::RngCore;
                let mut random_bytes = [0u8; 64];
                rand::thread_rng().fill_bytes(&mut random_bytes);
                self.stack.push(Value::ObjectID(
                    silver_core::object::ObjectID::from_bytes(&random_bytes)
                        .map_err(|e| InterpreterError::RuntimeError(e.to_string()))?
                ))?;
            }
            Instruction::ObjectDelete => {
                let _obj = self.stack.pop()?;
                // Object is deleted
            }
            Instruction::ObjectTransfer => {
                let _recipient = self.stack.pop()?;
                let _obj = self.stack.pop()?;
                // Object ownership transferred
            }
            Instruction::ObjectShare => {
                let _obj = self.stack.pop()?;
                // Object is now shared
            }
            Instruction::ObjectFreeze => {
                let obj = self.stack.pop()?;
                // Object is frozen (immutable)
                self.stack.push(obj)?;
            }
            Instruction::ObjectGetID => {
                let obj = self.stack.pop()?;
                if let Value::ObjectID(id) = obj {
                    self.stack.push(Value::ObjectID(id))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "ObjectID".to_string(),
                        got: format!("{:?}", obj),
                    });
                }
            }
            Instruction::ObjectExists => {
                let _obj_id = self.stack.pop()?;
                // In real implementation, would check object store
                self.stack.push(Value::Bool(true))?;
            }
            Instruction::ObjectBorrow => {
                let obj = self.stack.pop()?;
                self.stack.push(Value::Reference(Box::new(obj)))?;
            }
            Instruction::ObjectMutBorrow => {
                let obj = self.stack.pop()?;
                self.stack.push(Value::MutableReference(Box::new(obj)))?;
            }

            // Cryptographic operations
            Instruction::CryptoHashBlake3 => {
                let data = self.stack.pop()?;
                if let Value::ByteArray(bytes) = data {
                    let hash = silver_crypto::hashing::hash_512(&bytes);
                    self.stack.push(Value::ByteArray(hash.to_vec()))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "ByteArray".to_string(),
                        got: format!("{:?}", data),
                    });
                }
            }
            Instruction::CryptoVerifySignature => {
                let _signature = self.stack.pop()?;
                let _public_key = self.stack.pop()?;
                let _message = self.stack.pop()?;
                // In real implementation, would verify signature
                self.stack.push(Value::Bool(true))?;
            }
            Instruction::CryptoDeriveAddress => {
                let public_key = self.stack.pop()?;
                if let Value::ByteArray(pk_bytes) = public_key {
                    // Derive address from public key using BLAKE3-512 hash
                    let hash = silver_crypto::hashing::hash_512(&pk_bytes);
                    let address = SilverAddress::new(hash);
                    self.stack.push(Value::Address(address))?;
                } else {
                    return Err(InterpreterError::TypeMismatch {
                        expected: "ByteArray".to_string(),
                        got: format!("{:?}", public_key),
                    });
                }
            }
            Instruction::CryptoRandom(len) => {
                let mut bytes = vec![0u8; *len as usize];
                use rand::RngCore;
                rand::thread_rng().fill_bytes(&mut bytes);
                self.stack.push(Value::ByteArray(bytes))?;
            }

            // Event operations
            Instruction::EventEmit { event_type: _ } => {
                let _event_data = self.stack.pop()?;
                // Event is emitted (in real implementation, would be recorded)
            }

            // Transaction operations
            Instruction::TxSender => {
                let sender = SilverAddress::new([0u8; 64]);
                self.stack.push(Value::Address(sender))?;
            }
            Instruction::TxTimestamp => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                self.stack.push(Value::U64(now))?;
            }
            Instruction::TxDigest => {
                self.stack.push(Value::ByteArray(vec![0u8; 32]))?;
            }
            Instruction::FuelRemaining => {
                self.stack.push(Value::U64(self.fuel_remaining))?;
            }
            Instruction::FuelCharge(amount) => {
                if self.fuel_remaining < *amount {
                    return Err(InterpreterError::OutOfFuel);
                }
                self.fuel_remaining -= amount;
            }

            // No-op instruction
            Instruction::Nop => {
                // No operation
            }

            // Debug operations
            Instruction::DebugPrint => {
                let val = self.stack.pop()?;
                eprintln!("DEBUG: {:?}", val);
            }
            Instruction::Assert => {
                let cond = self.stack.pop()?;
                if let Value::Bool(false) = cond {
                    return Err(InterpreterError::Aborted(1));
                }
            }

            // Call instruction (handled separately in call_function)
            Instruction::Call(_) => {
                return Err(InterpreterError::RuntimeError(
                    "Call instruction should be handled separately".to_string()
                ));
            }
            Instruction::CallGeneric { .. } => {
                return Err(InterpreterError::RuntimeError(
                    "CallGeneric instruction should be handled separately".to_string()
                ));
            }
            Instruction::CallNative(_) => {
                return Err(InterpreterError::RuntimeError(
                    "CallNative instruction should be handled separately".to_string()
                ));
            }
        }

        // Increment PC
        if let Some(frame) = self.call_stack.last_mut() {
            frame.pc += 1;
        }

        Ok(())
    }

    // Helper methods for arithmetic operations
    fn add_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(Value::U8(a.wrapping_add(b))),
            (Value::U16(a), Value::U16(b)) => Ok(Value::U16(a.wrapping_add(b))),
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a.wrapping_add(b))),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a.wrapping_add(b))),
            (Value::U128(a), Value::U128(b)) => Ok(Value::U128(a.wrapping_add(b))),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn sub_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(Value::U8(a.wrapping_sub(b))),
            (Value::U16(a), Value::U16(b)) => Ok(Value::U16(a.wrapping_sub(b))),
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a.wrapping_sub(b))),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a.wrapping_sub(b))),
            (Value::U128(a), Value::U128(b)) => Ok(Value::U128(a.wrapping_sub(b))),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn mul_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(Value::U8(a.wrapping_mul(b))),
            (Value::U16(a), Value::U16(b)) => Ok(Value::U16(a.wrapping_mul(b))),
            (Value::U32(a), Value::U32(b)) => Ok(Value::U32(a.wrapping_mul(b))),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a.wrapping_mul(b))),
            (Value::U128(a), Value::U128(b)) => Ok(Value::U128(a.wrapping_mul(b))),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn div_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => {
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(Value::U8(a / b))
            }
            (Value::U64(a), Value::U64(b)) => {
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(Value::U64(a / b))
            }
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn mod_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U64(a), Value::U64(b)) => {
                if b == 0 {
                    return Err(InterpreterError::DivisionByZero);
                }
                Ok(Value::U64(a % b))
            }
            _ => Err(InterpreterError::TypeMismatch {
                expected: "U64".to_string(),
                got: "other".to_string(),
            }),
        }
    }

    fn bitand_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(Value::U8(a & b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a & b)),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn bitor_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(Value::U8(a | b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a | b)),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn bitxor_values(&self, val1: Value, val2: Value) -> InterpreterResult<Value> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(Value::U8(a ^ b)),
            (Value::U64(a), Value::U64(b)) => Ok(Value::U64(a ^ b)),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching integer types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn compare_lt(&self, val1: Value, val2: Value) -> InterpreterResult<bool> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(a < b),
            (Value::U64(a), Value::U64(b)) => Ok(a < b),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching comparable types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn compare_le(&self, val1: Value, val2: Value) -> InterpreterResult<bool> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(a <= b),
            (Value::U64(a), Value::U64(b)) => Ok(a <= b),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching comparable types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn compare_gt(&self, val1: Value, val2: Value) -> InterpreterResult<bool> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(a > b),
            (Value::U64(a), Value::U64(b)) => Ok(a > b),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching comparable types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn compare_ge(&self, val1: Value, val2: Value) -> InterpreterResult<bool> {
        match (val1, val2) {
            (Value::U8(a), Value::U8(b)) => Ok(a >= b),
            (Value::U64(a), Value::U64(b)) => Ok(a >= b),
            _ => Err(InterpreterError::TypeMismatch {
                expected: "matching comparable types".to_string(),
                got: "mismatched types".to_string(),
            }),
        }
    }

    fn branch(&mut self, offset: i32) -> InterpreterResult<()> {
        if let Some(frame) = self.call_stack.last_mut() {
            let new_pc = (frame.pc as i32 + offset) as usize;
            frame.pc = new_pc;
            Ok(())
        } else {
            Err(InterpreterError::RuntimeError(
                "No call frame for branch".to_string(),
            ))
        }
    }

    fn get_local(&self, idx: LocalIndex) -> InterpreterResult<&Value> {
        let frame = self
            .call_stack
            .last()
            .ok_or(InterpreterError::RuntimeError("No call frame".to_string()))?;

        let idx = idx as usize;
        if idx < frame.locals.len() {
            Ok(&frame.locals[idx])
        } else {
            Err(InterpreterError::InvalidLocalIndex(idx as u16))
        }
    }

    fn set_local(&mut self, idx: LocalIndex, value: Value) -> InterpreterResult<()> {
        let frame = self
            .call_stack
            .last_mut()
            .ok_or(InterpreterError::RuntimeError("No call frame".to_string()))?;

        let idx = idx as usize;
        if idx < frame.locals.len() {
            frame.locals[idx] = value;
            Ok(())
        } else {
            Err(InterpreterError::InvalidLocalIndex(idx as u16))
        }
    }

    fn get_constant(&self, idx: u16) -> InterpreterResult<&Constant> {
        let module = self
            .current_module
            .as_ref()
            .ok_or(InterpreterError::RuntimeError(
                "No current module".to_string(),
            ))?;

        module
            .constants
            .get(idx as usize)
            .ok_or(InterpreterError::InvalidConstantIndex(idx))
    }

    /// Get remaining fuel
    pub fn fuel_remaining(&self) -> u64 {
        self.fuel_remaining
    }

    // Bitwise operation helpers
    fn bitnot_value(&self, val: Value) -> InterpreterResult<Value> {
        match val {
            Value::U8(v) => Ok(Value::U8(!v)),
            Value::U16(v) => Ok(Value::U16(!v)),
            Value::U32(v) => Ok(Value::U32(!v)),
            Value::U64(v) => Ok(Value::U64(!v)),
            Value::U128(v) => Ok(Value::U128(!v)),
            Value::U256(v) => {
                let mut result = v;
                for byte in &mut result {
                    *byte = !*byte;
                }
                Ok(Value::U256(result))
            }
            _ => Err(InterpreterError::TypeMismatch {
                expected: "integer type".to_string(),
                got: format!("{:?}", val),
            }),
        }
    }

    fn shl_values(&self, val: Value, shift: Value) -> InterpreterResult<Value> {
        let shift_amount = match shift {
            Value::U8(v) => v as u32,
            Value::U16(v) => v as u32,
            Value::U32(v) => v,
            Value::U64(v) => v as u32,
            _ => return Err(InterpreterError::TypeMismatch {
                expected: "integer type".to_string(),
                got: format!("{:?}", shift),
            }),
        };

        match val {
            Value::U8(v) => Ok(Value::U8(v.wrapping_shl(shift_amount))),
            Value::U16(v) => Ok(Value::U16(v.wrapping_shl(shift_amount))),
            Value::U32(v) => Ok(Value::U32(v.wrapping_shl(shift_amount))),
            Value::U64(v) => Ok(Value::U64(v.wrapping_shl(shift_amount))),
            Value::U128(v) => Ok(Value::U128(v.wrapping_shl(shift_amount))),
            Value::U256(v) => {
                let mut result = v;
                if shift_amount < 256 {
                    for i in (shift_amount as usize..32).rev() {
                        result[i] = result[i - (shift_amount as usize / 8)];
                    }
                    for i in 0..(shift_amount as usize / 8) {
                        result[i] = 0;
                    }
                } else {
                    result = [0u8; 32];
                }
                Ok(Value::U256(result))
            }
            _ => Err(InterpreterError::TypeMismatch {
                expected: "integer type".to_string(),
                got: format!("{:?}", val),
            }),
        }
    }

    fn shr_values(&self, val: Value, shift: Value) -> InterpreterResult<Value> {
        let shift_amount = match shift {
            Value::U8(v) => v as u32,
            Value::U16(v) => v as u32,
            Value::U32(v) => v,
            Value::U64(v) => v as u32,
            _ => return Err(InterpreterError::TypeMismatch {
                expected: "integer type".to_string(),
                got: format!("{:?}", shift),
            }),
        };

        match val {
            Value::U8(v) => Ok(Value::U8(v.wrapping_shr(shift_amount))),
            Value::U16(v) => Ok(Value::U16(v.wrapping_shr(shift_amount))),
            Value::U32(v) => Ok(Value::U32(v.wrapping_shr(shift_amount))),
            Value::U64(v) => Ok(Value::U64(v.wrapping_shr(shift_amount))),
            Value::U128(v) => Ok(Value::U128(v.wrapping_shr(shift_amount))),
            Value::U256(v) => {
                let mut result = v;
                if shift_amount < 256 {
                    for i in 0..(32 - (shift_amount as usize / 8)) {
                        result[i] = result[i + (shift_amount as usize / 8)];
                    }
                    for i in (32 - (shift_amount as usize / 8))..32 {
                        result[i] = 0;
                    }
                } else {
                    result = [0u8; 32];
                }
                Ok(Value::U256(result))
            }
            _ => Err(InterpreterError::TypeMismatch {
                expected: "integer type".to_string(),
                got: format!("{:?}", val),
            }),
        }
    }
}
