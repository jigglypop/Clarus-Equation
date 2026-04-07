#![allow(non_local_definitions)]

pub mod controller;
pub mod engine;

pub use controller::{ControlStrategy, HardwareSpec, CeController};
pub use engine::benchmark::run_sweep_benchmark;
pub use engine::brain::BrainEngine;
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
    use crate::engine::brain::{BrainEngine, BrainState};
    use crate::engine::constants::{CeConstants, CE};

    // ---------------------------------------------------------------
    // QCEngine
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // BrainEngine
    // ---------------------------------------------------------------
    #[pyclass(name = "BrainEngine")]
    struct PyBrainEngine {
        engine: BrainEngine,
    }

    #[pyclass(name = "BrainState")]
    #[derive(Clone)]
    struct PyBrainState {
        #[pyo3(get)] r: f64,
        #[pyo3(get)] k: f64,
        #[pyo3(get)] phi_global: f64,
        #[pyo3(get)] pi_global: f64,
        #[pyo3(get)] field_energy: f64,
        #[pyo3(get)] suppression_factor: f64,
        #[pyo3(get)] memory_norm: f64,
    }

    impl From<BrainState> for PyBrainState {
        fn from(s: BrainState) -> Self {
            PyBrainState {
                r: s.r, k: s.k,
                phi_global: s.phi_global, pi_global: s.pi_global,
                field_energy: s.field_energy,
                suppression_factor: s.suppression_factor,
                memory_norm: s.memory_norm,
            }
        }
    }

    #[pymethods]
    impl PyBrainEngine {
        #[new]
        fn new(field_size: usize) -> Self {
            PyBrainEngine { engine: BrainEngine::new(field_size) }
        }

        fn step(&mut self, external_input: Option<Vec<f64>>) -> PyBrainState {
            let state = self.engine.step(external_input.as_deref());
            state.into()
        }

        fn run(&mut self, steps: usize) -> Vec<PyBrainState> {
            self.engine.run(steps).into_iter().map(Into::into).collect()
        }

        fn set_goal(&mut self, goal: Vec<f64>) {
            self.engine.set_goal(&goal);
        }

        fn get_field(&self) -> Vec<f64> { self.engine.field.phi.to_vec() }

        fn get_memory(&self) -> Vec<f64> { self.engine.memory.to_vec() }

        #[getter] fn rho_mem(&self) -> f64 { self.engine.rho_mem }
        #[getter] fn w_mem(&self) -> f64 { self.engine.w_mem }
    }

    // ---------------------------------------------------------------
    // CeConstants -- 45 constants from {e, pi, i, 1, 0}
    // ---------------------------------------------------------------
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

    // ---------------------------------------------------------------
    // TopK sparse activation -- Rust kernel (legacy, list-based)
    // ---------------------------------------------------------------
    #[pyfunction]
    fn topk_sparse(data: Vec<f64>, ratio: f64) -> (Vec<f64>, usize) {
        let n = data.len();
        let k = std::cmp::max(1, (ratio * n as f64).ceil() as usize).min(n);
        if k >= n {
            return (data, n);
        }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            data[b].abs().partial_cmp(&data[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = vec![0.0; n];
        for &i in &indices[..k] {
            out[i] = data[i];
        }
        (out, k)
    }

    #[pyfunction]
    fn topk_sparse_batch(data: Vec<f64>, row_len: usize, ratio: f64) -> Vec<f64> {
        use rayon::prelude::*;
        let k = std::cmp::max(1, (ratio * row_len as f64).ceil() as usize).min(row_len);
        if k >= row_len {
            return data;
        }
        let mut out = vec![0.0; data.len()];
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(row, out_row)| {
                let src = &data[row * row_len..(row + 1) * row_len];
                let mut indices: Vec<usize> = (0..row_len).collect();
                indices.sort_unstable_by(|&a, &b| {
                    src[b].abs().partial_cmp(&src[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
                });
                for &i in &indices[..k] {
                    out_row[i] = src[i];
                }
            });
        out
    }

    // ---------------------------------------------------------------
    // NN ops -- numpy-backed fused kernels (zero-copy)
    // ---------------------------------------------------------------
    use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
    use crate::engine::nn_ops;

    /// Fused SiLU + TopK sparse masking.
    /// input: flat f32 array [n_rows * dim].
    /// Returns (output, mask) as numpy arrays.
    #[pyfunction]
    fn nn_topk_silu_fwd<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
        dim: usize,
        ratio: f32,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<u8>) {
        let data = input.as_slice().expect("contiguous input");
        let (out, mask) = nn_ops::topk_silu_fwd(data, dim, ratio);
        (out.into_pyarray(py), mask.into_pyarray(py))
    }

    /// TopK SiLU backward.
    #[pyfunction]
    fn nn_topk_silu_bwd<'py>(
        py: Python<'py>,
        grad: PyReadonlyArray1<'py, f32>,
        input: PyReadonlyArray1<'py, f32>,
        mask: PyReadonlyArray1<'py, u8>,
        dim: usize,
    ) -> &'py PyArray1<f32> {
        let g = grad.as_slice().expect("contiguous grad");
        let x = input.as_slice().expect("contiguous input");
        let m = mask.as_slice().expect("contiguous mask");
        nn_ops::topk_silu_bwd(g, x, m, dim).into_pyarray(py)
    }

    /// Fused LBO norm forward (post-LayerNorm).
    /// normed: flat [n_rows*dim], v: flat [rank*dim], ...
    /// Returns (output, curvature).
    #[pyfunction]
    fn nn_lbo_fused_fwd<'py>(
        py: Python<'py>,
        normed: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        h: f32,
        scale: PyReadonlyArray1<'py, f32>,
        bias: PyReadonlyArray1<'py, f32>,
        alpha_conf: f32,
        dim: usize,
        rank: usize,
    ) -> (&'py PyArray1<f32>, f32) {
        let (out, curv) = nn_ops::lbo_fused_fwd(
            normed.as_slice().expect("contiguous"),
            v.as_slice().expect("contiguous"),
            h,
            scale.as_slice().expect("contiguous"),
            bias.as_slice().expect("contiguous"),
            alpha_conf,
            dim,
            rank,
        );
        (out.into_pyarray(py), curv)
    }

    /// Power iteration step for sigma_max(V).
    /// Returns (new_spectral_v, sigma_max).
    #[pyfunction]
    fn nn_power_iter<'py>(
        py: Python<'py>,
        v_mat: PyReadonlyArray1<'py, f32>,
        spectral_v: PyReadonlyArray1<'py, f32>,
        dim: usize,
        rank: usize,
    ) -> (&'py PyArray1<f32>, f32) {
        let (new_v, sigma) = nn_ops::power_iter_step(
            v_mat.as_slice().expect("contiguous"),
            spectral_v.as_slice().expect("contiguous"),
            dim,
            rank,
        );
        (new_v.into_pyarray(py), sigma)
    }

    /// Gauge lattice 3-channel fused forward.
    /// Returns output flat [n_rows*dim].
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_gauge_lattice_fwd<'py>(
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
        su3_up: PyReadonlyArray1<'py, f32>,
        su3_down: PyReadonlyArray1<'py, f32>,
        su2_up: PyReadonlyArray1<'py, f32>,
        su2_down: PyReadonlyArray1<'py, f32>,
        u1_up: PyReadonlyArray1<'py, f32>,
        u1_down: PyReadonlyArray1<'py, f32>,
        mix_down: PyReadonlyArray1<'py, f32>,
        mix_up: PyReadonlyArray1<'py, f32>,
        d3: usize, d2: usize, d1: usize,
        h3: usize, h2: usize, h1: usize,
        mix_rank: usize,
        ratio: f32,
        dim: usize,
    ) -> &'py PyArray1<f32> {
        nn_ops::gauge_lattice_fwd(
            input.as_slice().expect("contiguous"),
            su3_up.as_slice().expect("contiguous"),
            su3_down.as_slice().expect("contiguous"),
            su2_up.as_slice().expect("contiguous"),
            su2_down.as_slice().expect("contiguous"),
            u1_up.as_slice().expect("contiguous"),
            u1_down.as_slice().expect("contiguous"),
            mix_down.as_slice().expect("contiguous"),
            mix_up.as_slice().expect("contiguous"),
            d3, d2, d1, h3, h2, h1, mix_rank, ratio, dim,
        ).into_pyarray(py)
    }

    // ---------------------------------------------------------------
    // Module
    // ---------------------------------------------------------------
    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyQCEngine>()?;
        m.add_class::<PyBrainEngine>()?;
        m.add_class::<PyBrainState>()?;
        m.add_class::<PyCeConstants>()?;
        m.add_function(wrap_pyfunction!(topk_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(topk_sparse_batch, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_bwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_lbo_fused_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_power_iter, m)?)?;
        m.add_function(wrap_pyfunction!(nn_gauge_lattice_fwd, m)?)?;
        Ok(())
    }
}
