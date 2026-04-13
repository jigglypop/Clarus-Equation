#![allow(non_local_definitions)]
//! Minimal legacy CE reference crate.
//!
//! The canonical Rust compute surface now lives in `clarus/core`.
//! This crate now keeps only the old field integrator as a compatibility shim.

pub mod engine;

pub use engine::core::{QCEngine, QCEngineConfig};

#[cfg(feature = "python")]
mod python_binding {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use crate::engine::core::QCEngine;

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

    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyQCEngine>()?;
        Ok(())
    }
}
