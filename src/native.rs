//! # Native Functions
//!
//! Native function implementations for cryptographic operations
//! and system functions.

use silver_core::SilverAddress;
use silver_crypto::{derive_address as crypto_derive_address, hash_512};

/// Native function registry
pub struct NativeFunctions;

impl NativeFunctions {
    /// Hash data with Blake3-512
    pub fn hash_blake3(data: &[u8]) -> [u8; 64] {
        hash_512(data)
    }

    /// Derive address from public key
    pub fn derive_address(public_key: &[u8]) -> SilverAddress {
        crypto_derive_address(public_key)
    }

    /// Generate random bytes (from transaction context)
    pub fn random_bytes(seed: &[u8], length: usize) -> Vec<u8> {
        // In production, would use proper randomness from transaction context
        // For now, use deterministic generation from seed
        let mut result = Vec::with_capacity(length);
        for i in 0..length {
            let hash = hash_512(&[seed, &[i as u8]].concat());
            result.push(hash[0]);
        }
        result
    }
}
