#![allow(non_local_definitions)]

pub mod controller;
pub mod engine;

pub use controller::{ControlStrategy, HardwareSpec, CeController};
pub use engine::core::QCEngine;
pub use engine::ibm_api::IbmClient;
pub use engine::noise::generate_pink_noise;
pub use engine::optimizer::run_pulse_optimizer;
pub use engine::qec::simulate_repetition_code;
pub use engine::suppresson::{run_suppresson_evidence_analysis, SuppressonMode, SuppressonScanner};

#[cfg(feature = "python")]
mod python_binding {
    use pyo3::prelude::*;
    use pyo3::exceptions::PyValueError;
    use crate::engine::core::QCEngine;
    use crate::engine::constants::{CeConstants, CE};

    #[pyclass(name = "QCEngine")]
    struct PyQCEngine {
        engine: QCEngine,
    }

    #[pymethods]
    impl PyQCEngine {
        #[new]
        fn new(size: usize) -> Self {
            PyQCEngine { engine: QCEngine::new(size) }
        }

        fn step(&mut self) { self.engine.step(); }

        fn get_field(&self) -> Vec<f64> { self.engine.phi.to_vec() }

        fn get_velocity(&self) -> Vec<f64> { self.engine.dphi.to_vec() }

        fn set_curvature_suppression(&mut self, alpha2: f64) {
            self.engine.alpha2 = alpha2;
        }

        fn set_source(&mut self, source: Vec<f64>) -> PyResult<()> {
            if source.len() != self.engine.source_j.len() {
                return Err(PyValueError::new_err("source length mismatch"));
            }
            for (i, v) in source.iter().enumerate() {
                self.engine.source_j[i] = *v;
            }
            Ok(())
        }
    }

    #[pyclass(name = "CeConstants")]
    struct PyCeConstants {
        inner: CeConstants,
    }

    #[pymethods]
    impl PyCeConstants {
        #[new]
        fn new() -> Self {
            PyCeConstants { inner: CE.clone() }
        }

        #[getter] fn alpha_s(&self) -> f64 { self.inner.alpha_s }
        #[getter] fn alpha_w(&self) -> f64 { self.inner.alpha_w }
        #[getter] fn alpha_em_mz(&self) -> f64 { self.inner.alpha_em_mz }
        #[getter] fn sin2_theta_w(&self) -> f64 { self.inner.sin2_theta_w }
        #[getter] fn alpha_inv_0(&self) -> f64 { self.inner.alpha_inv_0 }
        #[getter] fn delta(&self) -> f64 { self.inner.delta }
        #[getter] fn d_eff(&self) -> f64 { self.inner.d_eff }
        #[getter] fn epsilon2(&self) -> f64 { self.inner.epsilon2 }
        #[getter] fn omega_b(&self) -> f64 { self.inner.omega_b }
        #[getter] fn omega_lambda(&self) -> f64 { self.inner.omega_lambda }
        #[getter] fn omega_dm(&self) -> f64 { self.inner.omega_dm }
        #[getter] fn f_factor(&self) -> f64 { self.inner.f_factor }
        #[getter] fn m_h_gev(&self) -> f64 { self.inner.m_h_gev }

        fn verify(&self) -> Vec<(String, f64, f64, f64)> {
            self.inner.verify().iter().map(|d| {
                (d.name.to_string(), d.predicted, d.observed, d.error_pct)
            }).collect()
        }
    }

    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyQCEngine>()?;
        m.add_class::<PyCeConstants>()?;
        Ok(())
    }
}
