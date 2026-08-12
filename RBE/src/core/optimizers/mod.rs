pub mod adam;
pub mod riemannian_adam;
pub mod analyzer;
pub mod config;
pub mod simd_utils;
// 주요 타입들 재수출
pub use adam::{AdamState, AdamBuffer};
pub use riemannian_adam::{RiemannianAdamState};
pub use analyzer::{TransformAnalyzer};
pub use config::{OptimizerConfig, AdamConfig, RiemannianAdamConfig};
pub use simd_utils::*;
/// 최적화 기법 종류
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    Adam,
    RiemannianAdam,
    SGD,
    RMSprop,
} 

#[cfg(test)]
pub mod __tests__;

#[cfg(test)]
mod tests {
    use super::*;
} 