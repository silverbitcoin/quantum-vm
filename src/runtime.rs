//! # Runtime Environment
//!
//! Production-ready runtime environment for Quantum VM with:
//! - Object access control
//! - Event emission system
//! - Native function implementations
//! - Transaction context

use silver_core::{ObjectID, SilverAddress};
use std::collections::HashMap;

/// Event emitted during execution
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    /// Event type identifier
    pub event_type: String,
    /// Event data (serialized)
    pub data: Vec<u8>,
    /// Sender address
    pub sender: SilverAddress,
}

/// Transaction context available during execution
#[derive(Debug, Clone)]
pub struct TransactionContext {
    /// Transaction sender
    pub sender: SilverAddress,
    /// Current timestamp
    pub timestamp: u64,
    /// Transaction digest
    pub digest: [u8; 64],
}

impl Default for TransactionContext {
    fn default() -> Self {
        Self {
            sender: SilverAddress([0u8; 64]),
            timestamp: 0,
            digest: [0u8; 64],
        }
    }
}

/// Runtime environment for Quantum VM execution
#[derive(Debug)]
pub struct Runtime {
    /// Transaction context
    pub tx_context: TransactionContext,
    /// Events emitted during execution
    pub events: Vec<Event>,
    /// Objects accessed during execution
    pub accessed_objects: HashMap<ObjectID, AccessMode>,
    /// Object store (for reading/writing objects)
    pub object_store: HashMap<ObjectID, Vec<u8>>,
}

/// Object access mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Read-only access
    Read,
    /// Read-write access
    Write,
}

impl Runtime {
    /// Create a new runtime environment
    pub fn new() -> Self {
        Self {
            tx_context: TransactionContext::default(),
            events: Vec::new(),
            accessed_objects: HashMap::new(),
            object_store: HashMap::new(),
        }
    }

    /// Create runtime with transaction context
    pub fn with_context(tx_context: TransactionContext) -> Self {
        Self {
            tx_context,
            events: Vec::new(),
            accessed_objects: HashMap::new(),
            object_store: HashMap::new(),
        }
    }

    /// Emit an event
    pub fn emit_event(&mut self, event_type: String, data: Vec<u8>) {
        self.events.push(Event {
            event_type,
            data,
            sender: self.tx_context.sender,
        });
    }

    /// Record object access
    pub fn record_object_access(&mut self, object_id: ObjectID, mode: AccessMode) {
        self.accessed_objects
            .entry(object_id)
            .and_modify(|existing| {
                // Upgrade to write if needed
                if mode == AccessMode::Write {
                    *existing = AccessMode::Write;
                }
            })
            .or_insert(mode);
    }

    /// Get transaction sender
    pub fn sender(&self) -> &SilverAddress {
        &self.tx_context.sender
    }

    /// Get current timestamp
    pub fn timestamp(&self) -> u64 {
        self.tx_context.timestamp
    }

    /// Get transaction digest
    pub fn digest(&self) -> &[u8; 64] {
        &self.tx_context.digest
    }

    /// Get all emitted events
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Get all accessed objects
    pub fn accessed_objects(&self) -> &HashMap<ObjectID, AccessMode> {
        &self.accessed_objects
    }

    /// Clear runtime state (for reuse)
    pub fn clear(&mut self) {
        self.events.clear();
        self.accessed_objects.clear();
        self.object_store.clear();
    }

    /// Read object data
    pub fn read_object(&mut self, object_id: ObjectID) -> Option<&[u8]> {
        self.record_object_access(object_id, AccessMode::Read);
        self.object_store.get(&object_id).map(|v| v.as_slice())
    }

    /// Write object data
    pub fn write_object(&mut self, object_id: ObjectID, data: Vec<u8>) {
        self.record_object_access(object_id, AccessMode::Write);
        self.object_store.insert(object_id, data);
    }

    /// Check if object exists
    pub fn object_exists(&self, object_id: &ObjectID) -> bool {
        self.object_store.contains_key(object_id)
    }

    /// Delete object
    pub fn delete_object(&mut self, object_id: ObjectID) -> Option<Vec<u8>> {
        self.record_object_access(object_id, AccessMode::Write);
        self.object_store.remove(&object_id)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
