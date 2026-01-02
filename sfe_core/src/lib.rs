#![allow(non_local_definitions)]
// pyo3의 `#[pymethods]` 매크로가 최신 rustc에서 `non_local_definitions` 경고를 유발할 수 있습니다.
// 동작 문제는 아니며 매크로 확장에 기인한 경고이므로, 크레이트 수준에서만 억제합니다.

use pyo3::prelude::*;

pub mod controller;
pub mod engine;

pub use controller::{ControlStrategy, HardwareSpec, SfeController};
pub use engine::benchmark::run_sweep_benchmark;
pub use engine::core::QSFEngine;
pub use engine::ibm_api::IbmClient;
pub use engine::noise::generate_pink_noise;
pub use engine::optimizer::run_pulse_optimizer;
pub use engine::qec::simulate_repetition_code;
pub use engine::suppresson::{run_suppresson_evidence_analysis, SuppressonMode, SuppressonScanner};

/// Python에서 호출 가능한 SFE 엔진 래퍼
#[pyclass]
struct PyQSFEngine {
    engine: QSFEngine,
}

#[pymethods]
impl PyQSFEngine {
    #[new]
    fn new(size: usize) -> Self {
        PyQSFEngine {
            engine: QSFEngine::new(size),
        }
    }

    fn step(&mut self) {
        self.engine.step();
    }

    fn get_field(&self) -> Vec<f64> {
        self.engine.phi.to_vec()
    }

    fn set_curvature_suppression(&mut self, alpha2: f64) {
        self.engine.alpha2 = alpha2;
    }
}

#[pymodule]
fn sfe_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQSFEngine>()?;
    Ok(())
}
