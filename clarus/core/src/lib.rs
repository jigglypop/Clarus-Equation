#![allow(non_local_definitions)]
//! Canonical Rust compute surface for the Clarus runtime.

pub mod engine;

pub use engine::field::{BoundaryMode, FieldConfig, FieldEngine, FieldState, FieldStepOutput};
pub use engine::kernel::{ModeParams, StepConfig, StepOutput, StpParams, apply_dale_sign, brain_step};
pub use engine::runtime_types::{CellState, Mode, RelaxInput, RelaxOutput, SnapshotMeta};

#[cfg(feature = "python")]
mod python_binding {
    use pyo3::prelude::*;
    use numpy::{PyReadonlyArray1, PyArray1, IntoPyArray};
    use crate::engine::nn_ops;
    use crate::engine::ce_riemann;
    use crate::engine::kernel;
    use crate::engine::runtime_types;

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

    #[pyfunction]
    fn nn_ce_pack_sparse<'py>(
        py: Python<'py>,
        w: PyReadonlyArray1<'py, f32>,
        dim: usize,
        zero_tol: f32,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<i32>, &'py PyArray1<i32>) {
        let data = w.as_slice().expect("contiguous");
        let (vals, cols, rows) = ce_riemann::pack_sparse_csr(data, dim, zero_tol);
        (vals.into_pyarray(py), cols.into_pyarray(py), rows.into_pyarray(py))
    }

    #[pyfunction]
    fn nn_ce_metric_basis_fwd<'py>(
        py: Python<'py>,
        codebook: PyReadonlyArray1<'py, f32>,
        m_ref: PyReadonlyArray1<'py, f32>,
        n_code: usize,
        dim: usize,
        rank: usize,
    ) -> &'py PyArray1<f32> {
        let cb = codebook.as_slice().expect("contiguous");
        let mr = m_ref.as_slice().expect("contiguous");
        ce_riemann::metric_basis_from_codebook(cb, mr, n_code, dim, rank).into_pyarray(py)
    }

    #[pyfunction]
    fn nn_ce_codebook_pull<'py>(
        py: Python<'py>,
        m: PyReadonlyArray1<'py, f32>,
        codebook: PyReadonlyArray1<'py, f32>,
        n_code: usize,
        dim: usize,
        beta: f32,
        cb_w: f32,
    ) -> (&'py PyArray1<f32>, f32) {
        let m_s = m.as_slice().expect("contiguous");
        let cb = codebook.as_slice().expect("contiguous");
        let (grad, energy) = ce_riemann::codebook_pull(m_s, cb, n_code, dim, beta, cb_w);
        (grad.into_pyarray(py), energy)
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_relax_fwd<'py>(
        py: Python<'py>,
        values: PyReadonlyArray1<'py, f32>,
        col_idx: PyReadonlyArray1<'py, i32>,
        row_ptr: PyReadonlyArray1<'py, i32>,
        b: PyReadonlyArray1<'py, f32>,
        phi: PyReadonlyArray1<'py, f32>,
        m0: PyReadonlyArray1<'py, f32>,
        codebook: PyReadonlyArray1<'py, f32>,
        metric_basis: PyReadonlyArray1<'py, f32>,
        dim: usize,
        n_code: usize,
        rank: usize,
        portal: f32,
        bypass: f32,
        t_wake: f32,
        beta: f32,
        cb_w: f32,
        lambda0: f32,
        lambda_phi: f32,
        lambda_var: f32,
        tau: f32,
        dt: f32,
        max_steps: usize,
        tol: f32,
        anneal_ratio: f32,
        noise_scale: f32,
        seed: u64,
    ) -> (
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        usize,
    ) {
        let out = ce_riemann::relax_forward(
            values.as_slice().expect("contiguous"),
            col_idx.as_slice().expect("contiguous"),
            row_ptr.as_slice().expect("contiguous"),
            b.as_slice().expect("contiguous"),
            phi.as_slice().expect("contiguous"),
            m0.as_slice().expect("contiguous"),
            codebook.as_slice().expect("contiguous"),
            metric_basis.as_slice().expect("contiguous"),
            dim, n_code, rank,
            portal, bypass, t_wake, beta, cb_w,
            lambda0, lambda_phi, lambda_var,
            tau, dt, max_steps, tol, anneal_ratio, noise_scale, seed,
        );
        (
            out.best_m.into_pyarray(py),
            out.energy.into_pyarray(py),
            out.delta.into_pyarray(py),
            out.e_hop.into_pyarray(py),
            out.e_bias.into_pyarray(py),
            out.e_portal.into_pyarray(py),
            out.e_cb.into_pyarray(py),
            out.bypass_hist.into_pyarray(py),
            out.steps,
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_brain_step<'py>(
        py: Python<'py>,
        w_values: PyReadonlyArray1<'py, f32>,
        w_col_idx: PyReadonlyArray1<'py, i32>,
        w_row_ptr: PyReadonlyArray1<'py, i32>,
        activation: PyReadonlyArray1<'py, f32>,
        refractory: PyReadonlyArray1<'py, f32>,
        memory_trace: PyReadonlyArray1<'py, f32>,
        adaptation: PyReadonlyArray1<'py, f32>,
        stp_u: PyReadonlyArray1<'py, f32>,
        stp_x: PyReadonlyArray1<'py, f32>,
        bitfield: PyReadonlyArray1<'py, u8>,
        external: PyReadonlyArray1<'py, f32>,
        goal: PyReadonlyArray1<'py, f32>,
        replay: PyReadonlyArray1<'py, f32>,
        mode: u8,
        energy_budget: usize,
    ) -> (
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<u8>,
        usize,
        f32,
    ) {
        let mode_enum = match mode {
            1 => runtime_types::Mode::Nrem,
            2 => runtime_types::Mode::Rem,
            _ => runtime_types::Mode::Wake,
        };
        let mp = kernel::ModeParams::from_mode(mode_enum);
        let cfg = kernel::StepConfig {
            energy_budget,
            ..Default::default()
        };
        let mut act = activation.as_slice().expect("contiguous").to_vec();
        let mut refr = refractory.as_slice().expect("contiguous").to_vec();
        let mut mem = memory_trace.as_slice().expect("contiguous").to_vec();
        let mut adapt = adaptation.as_slice().expect("contiguous").to_vec();
        let mut su = stp_u.as_slice().expect("contiguous").to_vec();
        let mut sx = stp_x.as_slice().expect("contiguous").to_vec();
        let mut bit = bitfield.as_slice().expect("contiguous").to_vec();
        let out = kernel::brain_step(
            w_values.as_slice().expect("contiguous"),
            w_col_idx.as_slice().expect("contiguous"),
            w_row_ptr.as_slice().expect("contiguous"),
            &mut act,
            &mut refr,
            &mut mem,
            &mut adapt,
            &mut su,
            &mut sx,
            &mut bit,
            external.as_slice().expect("contiguous"),
            goal.as_slice().expect("contiguous"),
            replay.as_slice().expect("contiguous"),
            &mp,
            &cfg,
        );
        (
            act.into_pyarray(py),
            refr.into_pyarray(py),
            mem.into_pyarray(py),
            adapt.into_pyarray(py),
            su.into_pyarray(py),
            sx.into_pyarray(py),
            bit.into_pyarray(py),
            out.active_count,
            out.energy,
        )
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn nn_ce_mfa_fwd<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f32>,
        k: PyReadonlyArray1<'py, f32>,
        v: PyReadonlyArray1<'py, f32>,
        n: usize,
        d: usize,
        sigma_grav: f32,
        w_lang: f32,
        w_grav: f32,
        causal: bool,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<f32>) {
        let (out, attn) = nn_ops::ce_mfa_fwd(
            q.as_slice().expect("contiguous q"),
            k.as_slice().expect("contiguous k"),
            v.as_slice().expect("contiguous v"),
            n, d, sigma_grav, w_lang, w_grav, causal,
        );
        (out.into_pyarray(py), attn.into_pyarray(py))
    }

    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(topk_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(topk_sparse_batch, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_topk_silu_bwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_lbo_fused_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_power_iter, m)?)?;
        m.add_function(wrap_pyfunction!(nn_gauge_lattice_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_pack_sparse, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_metric_basis_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_codebook_pull, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_relax_fwd, m)?)?;
        m.add_function(wrap_pyfunction!(nn_brain_step, m)?)?;
        m.add_function(wrap_pyfunction!(nn_ce_mfa_fwd, m)?)?;
        Ok(())
    }
}
