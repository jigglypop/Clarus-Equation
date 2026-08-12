//! # RBE 핵심 라이브러리 모듈
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩의 핵심 구성 요소들

pub mod encoder;
pub mod decoder;
pub mod math;
pub mod matrix;
pub mod optimizers;
pub mod packed_params;
pub mod systems;
pub mod generator;
pub mod differential; // 새로운 통합 미분 시스템

// 주요 타입들 재수출
pub use encoder::*;
pub use decoder::*;
pub use packed_params::*;
pub use systems::*;
pub use generator::*;
pub use differential::*; // 새로운 미분 시스템 타입들

// 각 모듈이 자체 테스트를 포함함
