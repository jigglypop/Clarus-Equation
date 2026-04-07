//! Fused neural-network ops for ClarusLM.
//!
//! All functions operate on flat f32 slices laid out row-major.
//! Row-parallel via rayon where beneficial.

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
/// Returns `(output, mask)` where mask is `Vec<u8>` (1=kept, 0=zeroed).
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
            // quickselect for kth-largest absolute value
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
                    msk[j] = 0;
                }
            }
        });
    (output, mask)
}

/// TopK SiLU backward.
///
/// `grad`:  flat `[n_rows * dim]` upstream gradient.
/// `input`: flat `[n_rows * dim]` original input to SiLU.
/// `mask`:  flat `[n_rows * dim]` kept-mask (1/0).
/// Returns `grad_input`.
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

// ---- LBO Norm ---------------------------------------------------------------

/// Fused LBO normalization forward (post-LayerNorm).
///
/// Inputs (all flat row-major):
///   `normed`: `[n_rows * dim]` -- already layer-normalised.
///   `v`:      `[rank * dim]`   -- projection matrix V.
///   `h`:      step size (clamped by caller).
///   `scale`:  `[dim]`
///   `bias`:   `[dim]`
///   `alpha_conf`: conformal self-reference coefficient.
///
/// Returns `(output [n_rows*dim], curvature)`.
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

    // V_eff = V * conformal
    let v_eff: Vec<f32> = v.iter().map(|&x| x * conformal).collect();

    let mut output = vec![0.0f32; normed.len()];

    let curvature_sum: f64 = output
        .par_chunks_mut(dim)
        .enumerate()
        .map(|(r, out)| {
            let x = &normed[r * dim..(r + 1) * dim];
            let mut row_curv = 0.0f64;

            // proj = x @ V_eff^T  [rank]
            let mut proj = vec![0.0f32; rank];
            for i in 0..rank {
                let mut s = 0.0f32;
                let v_row = &v_eff[i * dim..(i + 1) * dim];
                for j in 0..dim {
                    s += x[j] * v_row[j];
                }
                proj[i] = s;
            }

            // out = (x - h*(x - xW)) * scale + bias
            //     = (x*(1-h) + h*xW) * scale + bias
            let one_minus_h = 1.0 - h;
            for j in 0..dim {
                let mut xw_j = 0.0f32;
                for i in 0..rank {
                    xw_j += proj[i] * v_eff[i * dim + j];
                }
                let lx = x[j] - xw_j;
                row_curv += (lx as f64) * (lx as f64);
                out[j] = (one_minus_h * x[j] + h * xw_j) * scale[j] + bias[j];
            }
            row_curv
        })
        .sum();

    let curvature = (curvature_sum / (n_rows * dim) as f64) as f32;
    (output, curvature)
}

/// Power iteration: 1 step for sigma_max(V).
/// `v_mat`: `[rank * dim]`, `spectral_v`: `[dim]` (unit vector).
/// Returns `(new_spectral_v, sigma_max)`.
pub fn power_iter_step(v_mat: &[f32], spectral_v: &[f32], dim: usize, rank: usize) -> (Vec<f32>, f32) {
    // u = normalize(V @ spectral_v)  [rank]
    let mut u = vec![0.0f32; rank];
    for i in 0..rank {
        let mut s = 0.0f32;
        for j in 0..dim {
            s += v_mat[i * dim + j] * spectral_v[j];
        }
        u[i] = s;
    }
    let u_norm = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in u.iter_mut() {
        *x /= u_norm;
    }

    // new_v = normalize(V^T @ u)  [dim]
    let mut new_v = vec![0.0f32; dim];
    for j in 0..dim {
        let mut s = 0.0f32;
        for i in 0..rank {
            s += v_mat[i * dim + j] * u[i];
        }
        new_v[j] = s;
    }
    let v_norm = new_v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in new_v.iter_mut() {
        *x /= v_norm;
    }

    // sigma_max = ||V @ new_v||
    let mut vv = vec![0.0f32; rank];
    for i in 0..rank {
        let mut s = 0.0f32;
        for j in 0..dim {
            s += v_mat[i * dim + j] * new_v[j];
        }
        vv[i] = s;
    }
    let sigma = vv.iter().map(|x| x * x).sum::<f32>().sqrt();

    (new_v, sigma)
}

/// Gauge lattice 3-channel forward (fused up+silu+topk+down+mix).
///
/// `input`:    `[n_rows * dim]`
/// `su3_up`:   `[h3 * d3]` (row-major)  ... etc.
/// `mix_down`: `[mix_rank * dim]`, `mix_up`: `[dim * mix_rank]` (optional, empty=skip)
///
/// Returns `output [n_rows * dim]`.
#[allow(clippy::too_many_arguments)]
pub fn gauge_lattice_fwd(
    input: &[f32],
    su3_up: &[f32], su3_down: &[f32],
    su2_up: &[f32], su2_down: &[f32],
    u1_up: &[f32],  u1_down: &[f32],
    mix_down: &[f32], mix_up: &[f32],
    d3: usize, d2: usize, d1: usize,
    h3: usize, h2: usize, h1: usize,
    mix_rank: usize,
    ratio: f32,
    dim: usize,
) -> Vec<f32> {
    let n_rows = input.len() / dim;
    let k3 = cmp::max(1, (ratio * h3 as f32).ceil() as usize).min(h3);
    let k2 = cmp::max(1, (ratio * h2 as f32).ceil() as usize).min(h2);
    let k1 = cmp::max(1, (ratio * h1 as f32).ceil() as usize).min(h1);
    let has_mix = mix_rank > 0 && !mix_down.is_empty() && !mix_up.is_empty();

    let mut output = vec![0.0f32; input.len()];

    output
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(r, out)| {
            let x = &input[r * dim..(r + 1) * dim];

            // ---- channel helper (inline) ----
            macro_rules! channel_fwd {
                ($x_slice:expr, $up:expr, $down:expr, $din:expr, $hid:expr, $k:expr, $out_slice:expr) => {{
                    let mut hidden = vec![0.0f32; $hid];
                    for i in 0..$hid {
                        let mut s = 0.0f32;
                        for j in 0..$din {
                            s += $x_slice[j] * $up[i * $din + j];
                        }
                        hidden[i] = silu_f32(s);
                    }
                    if $k < $hid {
                        let mut abs_h: Vec<f32> = hidden.iter().map(|v| v.abs()).collect();
                        abs_h.select_nth_unstable_by($hid - $k, |a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        });
                        let thr = abs_h[$hid - $k];
                        for v in hidden.iter_mut() {
                            if v.abs() < thr { *v = 0.0; }
                        }
                    }
                    for j in 0..$din {
                        let mut s = 0.0f32;
                        for i in 0..$hid {
                            s += hidden[i] * $down[j * $hid + i];
                        }
                        $out_slice[j] = s;
                    }
                }};
            }

            let s3 = d3;
            let s32 = d3 + d2;
            channel_fwd!(&x[..s3], su3_up, su3_down, d3, h3, k3, &mut out[..s3]);
            channel_fwd!(&x[s3..s32], su2_up, su2_down, d2, h2, k2, &mut out[s3..s32]);
            channel_fwd!(&x[s32..], u1_up, u1_down, d1, h1, k1, &mut out[s32..]);

            if has_mix {
                let mut proj = vec![0.0f32; mix_rank];
                for i in 0..mix_rank {
                    let mut s = 0.0f32;
                    for j in 0..dim {
                        s += out[j] * mix_down[i * dim + j];
                    }
                    proj[i] = s;
                }
                for j in 0..dim {
                    let mut s = 0.0f32;
                    for i in 0..mix_rank {
                        s += proj[i] * mix_up[j * mix_rank + i];
                    }
                    out[j] += s;
                }
            }
        });
    output
}
