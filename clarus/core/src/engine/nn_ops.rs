//! Fused neural-network ops for ClarusLM.
//!
//! All functions operate on flat f32 slices laid out row-major.
//! Matrix multiplications use ndarray (matrixmultiply SIMD backend).
//! Row-parallel via rayon where beneficial.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::cmp;

// ---- helpers ---------------------------------------------------------------

#[inline(always)]
fn silu_f32(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---- TopK SiLU -------------------------------------------------------------

/// Fused SiLU + TopK sparse masking (forward).
///
/// `input`: flat `[n_rows * dim]`, `dim`: row width, `ratio`: keep fraction.
/// Returns `(output, mask)`.
pub fn topk_silu_fwd(input: &[f32], dim: usize, ratio: f32) -> (Vec<f32>, Vec<u8>) {
    let k = cmp::max(1, (ratio * dim as f32).ceil() as usize).min(dim);
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let mut mask = vec![0u8; n];

    if k >= dim {
        output
            .par_chunks_mut(dim)
            .zip(mask.par_chunks_mut(dim))
            .enumerate()
            .for_each(|(r, (out, msk))| {
                let src = &input[r * dim..(r + 1) * dim];
                for j in 0..dim {
                    out[j] = silu_f32(src[j]);
                    msk[j] = 1;
                }
            });
        return (output, mask);
    }

    output
        .par_chunks_mut(dim)
        .zip(mask.par_chunks_mut(dim))
        .enumerate()
        .for_each(|(r, (out, msk))| {
            let src = &input[r * dim..(r + 1) * dim];
            for j in 0..dim {
                out[j] = silu_f32(src[j]);
            }
            let mut abs_vals: Vec<f32> = out.iter().map(|x| x.abs()).collect();
            abs_vals.select_nth_unstable_by(dim - k, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let thr = abs_vals[dim - k];
            for j in 0..dim {
                if out[j].abs() >= thr {
                    msk[j] = 1;
                } else {
                    out[j] = 0.0;
                }
            }
        });
    (output, mask)
}

/// TopK SiLU backward.
pub fn topk_silu_bwd(grad: &[f32], input: &[f32], mask: &[u8], dim: usize) -> Vec<f32> {
    let n = grad.len();
    let mut grad_in = vec![0.0f32; n];

    grad_in
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(r, gi)| {
            let base = r * dim;
            for j in 0..dim {
                if mask[base + j] == 1 {
                    let x = input[base + j];
                    let s = sigmoid_f32(x);
                    gi[j] = grad[base + j] * s * (1.0 + x * (1.0 - s));
                }
            }
        });
    grad_in
}

// ---- LBO Norm (ndarray-backed matmul) ---------------------------------------

/// Fused LBO normalization forward (post-LayerNorm).
///
/// Uses ndarray `dot()` (matrixmultiply SIMD) for projections.
pub fn lbo_fused_fwd(
    normed: &[f32],
    v: &[f32],
    h: f32,
    scale: &[f32],
    bias: &[f32],
    alpha_conf: f32,
    dim: usize,
    rank: usize,
) -> (Vec<f32>, f32) {
    let n_rows = normed.len() / dim;

    // conformal factor
    let phi_sq: f32 = normed.iter().map(|&x| x * x).sum::<f32>() / normed.len() as f32;
    let conformal = (-alpha_conf.abs() * phi_sq).exp();

    // V_eff = V * conformal  [rank, dim]
    let v_scaled: Vec<f32> = v.iter().map(|&x| x * conformal).collect();
    let x_mat = ArrayView2::from_shape((n_rows, dim), normed).unwrap();
    let v_mat = ArrayView2::from_shape((rank, dim), &v_scaled).unwrap();

    // proj = X @ V_eff^T  -> [n_rows, rank]   (ndarray SIMD dot)
    let proj = x_mat.dot(&v_mat.t());
    // xW = proj @ V_eff   -> [n_rows, dim]    (ndarray SIMD dot)
    let xw = proj.dot(&v_mat);

    // output + curvature
    let scale_v = ArrayView1::from(scale);
    let bias_v = ArrayView1::from(bias);
    let one_minus_h = 1.0 - h;
    let mut output = vec![0.0f32; normed.len()];
    let mut curv_sum = 0.0f64;

    for r in 0..n_rows {
        let base = r * dim;
        for j in 0..dim {
            let lx = x_mat[[r, j]] - xw[[r, j]];
            curv_sum += (lx as f64) * (lx as f64);
            output[base + j] = (one_minus_h * x_mat[[r, j]] + h * xw[[r, j]])
                * scale_v[j]
                + bias_v[j];
        }
    }
    (output, (curv_sum / (n_rows * dim) as f64) as f32)
}

/// Power iteration: 1 step for sigma_max(V).
pub fn power_iter_step(
    v_mat: &[f32],
    spectral_v: &[f32],
    dim: usize,
    rank: usize,
) -> (Vec<f32>, f32) {
    let v_nd = ArrayView2::from_shape((rank, dim), v_mat).unwrap();
    let sv = ArrayView1::from_shape(dim, spectral_v).unwrap();

    // u = V @ sv  [rank]
    let u_raw = v_nd.dot(&sv);
    let u_norm = u_raw.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let u = u_raw.mapv(|x| x / u_norm);

    // new_v = V^T @ u  [dim]
    let vt = v_nd.t();
    let nv_raw = vt.dot(&u);
    let nv_norm = nv_raw.mapv(|x| x * x).sum().sqrt().max(1e-12);
    let new_v = nv_raw.mapv(|x| x / nv_norm);

    // sigma = ||V @ new_v||
    let sigma = v_nd.dot(&new_v).mapv(|x| x * x).sum().sqrt();

    (new_v.to_vec(), sigma)
}

// ---- Gauge lattice (ndarray matmul per channel) ----------------------------

/// Single gauge channel: up -> SiLU -> TopK -> down.
fn channel_fwd(
    x: &ArrayView1<f32>,
    up_w: &ArrayView2<f32>,   // [hid, d_in]
    down_w: &ArrayView2<f32>, // [d_in, hid]
    k: usize,
) -> Array1<f32> {
    // hidden = x @ up^T  -> [hid]
    let hidden_raw = up_w.dot(x);
    let hid = hidden_raw.len();

    // SiLU + TopK
    let mut hidden: Vec<f32> = hidden_raw.iter().map(|&v| silu_f32(v)).collect();
    if k < hid {
        let mut abs_h: Vec<f32> = hidden.iter().map(|v| v.abs()).collect();
        abs_h.select_nth_unstable_by(hid - k, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let thr = abs_h[hid - k];
        for v in hidden.iter_mut() {
            if v.abs() < thr {
                *v = 0.0;
            }
        }
    }

    // output = down @ hidden  -> [d_in]  (down is [d_in, hid])
    let h_arr = ArrayView1::from(&hidden);
    down_w.dot(&h_arr)
}

/// Gauge lattice 3-channel forward.
#[allow(clippy::too_many_arguments)]
pub fn gauge_lattice_fwd(
    input: &[f32],
    su3_up: &[f32],
    su3_down: &[f32],
    su2_up: &[f32],
    su2_down: &[f32],
    u1_up: &[f32],
    u1_down: &[f32],
    mix_down: &[f32],
    mix_up: &[f32],
    d3: usize, d2: usize, d1: usize,
    h3: usize, h2: usize, h1: usize,
    mix_rank: usize,
    ratio: f32,
    dim: usize,
) -> Vec<f32> {
    let _n_rows = input.len() / dim;
    let k3 = cmp::max(1, (ratio * h3 as f32).ceil() as usize).min(h3);
    let k2 = cmp::max(1, (ratio * h2 as f32).ceil() as usize).min(h2);
    let k1 = cmp::max(1, (ratio * h1 as f32).ceil() as usize).min(h1);
    let has_mix = mix_rank > 0 && !mix_down.is_empty() && !mix_up.is_empty();

    let su3_up_nd = ArrayView2::from_shape((h3, d3), su3_up).unwrap();
    let su3_dn_nd = ArrayView2::from_shape((d3, h3), su3_down).unwrap();
    let su2_up_nd = ArrayView2::from_shape((h2, d2), su2_up).unwrap();
    let su2_dn_nd = ArrayView2::from_shape((d2, h2), su2_down).unwrap();
    let u1_up_nd = ArrayView2::from_shape((h1, d1), u1_up).unwrap();
    let u1_dn_nd = ArrayView2::from_shape((d1, h1), u1_down).unwrap();

    let x_mat = ArrayView2::from_shape((_n_rows, dim), input).unwrap();
    let mut output = vec![0.0f32; input.len()];

    let s3 = d3;
    let s32 = d3 + d2;

    output
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(r, out)| {
            let x_row = x_mat.row(r);
            let x3 = x_row.slice(ndarray::s![..s3]);
            let x2 = x_row.slice(ndarray::s![s3..s32]);
            let x1 = x_row.slice(ndarray::s![s32..]);

            let y3 = channel_fwd(&x3, &su3_up_nd, &su3_dn_nd, k3);
            let y2 = channel_fwd(&x2, &su2_up_nd, &su2_dn_nd, k2);
            let y1 = channel_fwd(&x1, &u1_up_nd, &u1_dn_nd, k1);

            out[..s3].copy_from_slice(y3.as_slice().unwrap());
            out[s3..s32].copy_from_slice(y2.as_slice().unwrap());
            out[s32..].copy_from_slice(y1.as_slice().unwrap());

            if has_mix {
                let md = ArrayView2::from_shape((mix_rank, dim), mix_down).unwrap();
                let mu = ArrayView2::from_shape((dim, mix_rank), mix_up).unwrap();
                let out_view = ArrayView1::from(&*out);
                let proj = md.dot(&out_view);
                let mix_result = mu.dot(&proj);
                for j in 0..dim {
                    out[j] += mix_result[j];
                }
            }
        });
    output
}
