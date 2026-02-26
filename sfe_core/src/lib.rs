#![allow(non_local_definitions)]

pub mod controller;
pub mod engine;

pub use controller::{ControlStrategy, HardwareSpec, CeController};
pub use engine::benchmark::run_sweep_benchmark;
pub use engine::core::QCEngine;
pub use engine::ibm_api::IbmClient;
pub use engine::noise::generate_pink_noise;
pub use engine::optimizer::run_pulse_optimizer;
pub use engine::qec::simulate_repetition_code;
pub use engine::suppresson::{run_suppresson_evidence_analysis, SuppressonMode, SuppressonScanner};

#[cfg(feature = "python")]
mod python_binding {
    use pyo3::prelude::*;
    use crate::engine::core::QCEngine;

    #[pyclass]
    struct PyQCEngine {
        engine: QCEngine,
    }

    #[pymethods]
    impl PyQCEngine {
        #[new]
        fn new(size: usize) -> Self {
            PyQCEngine {
                engine: QCEngine::new(size),
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
        m.add_class::<PyQCEngine>()?;
        Ok(())
    }
}
