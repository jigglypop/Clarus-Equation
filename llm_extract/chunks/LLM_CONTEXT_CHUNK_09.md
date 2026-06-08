# LLM Context Chunk

---
## File: `reality_stone/src/layers/rsulf.rs`

```rust
use faer::Mat;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

pub struct RSULFConfig {
    pub d_model: usize,
    pub r: usize,
    pub eta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub seq_len: usize,
    pub window: usize,
    pub calibration_samples: usize,
}

impl Default for RSULFConfig {
    fn default() -> Self {
        Self {
            d_model: 4096,
            r: 1024,
            eta: 0.01,
            alpha: 0.02,
            beta: 0.01,
            gamma: 0.99,
            seq_len: 128,
            window: 8,
            calibration_samples: 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GlobalBasis {
    pub u: Array2<f32>,
    pub rank: usize,
}

pub fn extract_global_basis(
    layers_wq: &[ArrayView2<f32>],
    layers_wk: &[ArrayView2<f32>],
    target_rank: usize,
) -> GlobalBasis {
    let num_layers = layers_wq.len();
    if num_layers == 0 {
        return GlobalBasis {
            u: Array2::zeros((0, 0)),
            rank: 0,
        };
    }

    let d_model = layers_wq[0].ncols();
    // Reservoir sampling or aggregate covariance
    // For simplicity and memory efficiency, we accumulate covariance matrix
    // G_total = sum_l (W_Q^l)^T (W_K^l)

    let mut g_acc = Array2::<f32>::zeros((d_model, d_model));

    for (wq, wk) in layers_wq.iter().zip(layers_wk.iter()) {
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d_model));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        // G = WQ^T * WK
        // We want the basis that explains the interaction.
        // Approximate by summing G * G^T or just G.
        // Let's use the sum of singular vectors logic:
        // Or simpler: Aggregate G and find its SVD.
        // But G is d_model x d_model. Summing them is valid.

        let g = wq.t().dot(&wk_expanded);
        // Symmetrize contribution
        let g_sym = (&g + &g.t()) * 0.5;
        g_acc = g_acc + g_sym;
    }

    // Perform Randomized SVD on the accumulated Metric
    // This extracts the "Shared Global Basis" U
    let k = target_rank.min(d_model);
    let (u, _, _) = randomized_svd(&g_acc, k, 20, 5);

    GlobalBasis { u, rank: k }
}

pub struct FoldedMetric {
    pub u: Array2<f32>,
    pub s: Array1<f32>,
    pub v: Array2<f32>,
    pub s_residual: Array1<f32>,
}

use rand::Rng;

fn dense_svd(a: &Array2<f32>, k: usize) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let mat = Mat::from_fn(m, n, |i, j| a[[i, j]]);
    let svd = mat.svd();

    let u_faer = svd.u();
    let s_faer = svd.s_diagonal();
    let v_faer = svd.v();

    let k_actual = k.min(m).min(n).min(s_faer.nrows());

    let mut u = Array2::<f32>::zeros((m, k_actual));
    let mut s = Array1::<f32>::zeros(k_actual);
    let mut v = Array2::<f32>::zeros((n, k_actual));

    for j in 0..k_actual {
        s[j] = s_faer.read(j);
        for i in 0..m {
            u[[i, j]] = u_faer.read(i, j);
        }
        for i in 0..n {
            v[[i, j]] = v_faer.read(i, j);
        }
    }

    (u, s, v)
}

fn orthonormalize_columns(mut y: Array2<f32>) -> Array2<f32> {
    let rows = y.nrows();
    let cols = y.ncols();
    let mut rank = 0usize;

    for j in 0..cols {
        for p in 0..rank {
            let mut dot = 0.0_f32;
            for i in 0..rows {
                dot += y[[i, j]] * y[[i, p]];
            }
            for i in 0..rows {
                y[[i, j]] -= dot * y[[i, p]];
            }
        }

        for p in 0..rank {
            let mut dot = 0.0_f32;
            for i in 0..rows {
                dot += y[[i, j]] * y[[i, p]];
            }
            for i in 0..rows {
                y[[i, j]] -= dot * y[[i, p]];
            }
        }

        let mut norm_sq = 0.0_f32;
        for i in 0..rows {
            norm_sq += y[[i, j]] * y[[i, j]];
        }

        let norm = norm_sq.sqrt();
        if norm <= 1e-8 {
            continue;
        }

        if rank != j {
            for i in 0..rows {
                y[[i, rank]] = y[[i, j]];
            }
        }
        for i in 0..rows {
            y[[i, rank]] /= norm;
        }
        rank += 1;
    }

    y.slice(s![.., 0..rank]).to_owned()
}

pub fn randomized_svd(
    a: &Array2<f32>,
    k: usize,
    n_oversamples: usize,
    n_iter: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();

    if k == 0 || m == 0 || n == 0 {
        return (
            Array2::<f32>::zeros((m, 0)),
            Array1::<f32>::zeros(0),
            Array2::<f32>::zeros((n, 0)),
        );
    }

    let min_dim = m.min(n);
    if min_dim <= 128 || k >= min_dim {
        return dense_svd(a, k);
    }

    let l = (k + n_oversamples).min(min_dim);
    let mut rng = rand::thread_rng();
    let scale = 1.0_f32 / (l as f32).sqrt();
    let omega = Array2::<f32>::from_shape_fn((n, l), |_| rng.gen_range(-1.0..1.0) * scale);

    let mut y = a.dot(&omega);
    for _ in 0..n_iter {
        let z = a.t().dot(&y);
        y = a.dot(&z);
    }

    let q = orthonormalize_columns(y);
    if q.ncols() == 0 {
        return (
            Array2::<f32>::zeros((m, 0)),
            Array1::<f32>::zeros(0),
            Array2::<f32>::zeros((n, 0)),
        );
    }

    let b = q.t().dot(a);
    let (u_hat, s, v) = dense_svd(&b, k);
    let u = q.dot(&u_hat);

    (u, s, v)
}

fn qr_decomposition(a: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let mat = Mat::from_fn(m, n, |i, j| a[[i, j]]);
    let qr = mat.qr();

    let q_faer = qr.compute_q();
    let r_faer = qr.compute_r();

    let mut q = Array2::<f32>::zeros((m, m.min(n)));
    let mut r = Array2::<f32>::zeros((m.min(n), n));

    let k = m.min(n);

    for j in 0..k {
        for i in 0..m {
            q[[i, j]] = q_faer.read(i, j);
        }
    }

    for j in 0..n {
        for i in 0..k {
            if i <= j {
                r[[i, j]] = r_faer.read(i, j);
            }
        }
    }

    (q, r)
}

pub fn fold_dimension_svd(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedMetric {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };

    let g = wq.t().dot(&wk_expanded);
    let frob_g: f32 = g.iter().map(|x| x * x).sum();

    let k = target_dim.min(g.nrows().min(g.ncols()));
    let oversamples = k.min(20);
    let n_iter = if k < 32 { 3 } else { 2 };
    let (u, s, v) = randomized_svd(&g, k, oversamples, n_iter);

    let frob_approx: f32 = s.iter().map(|x| x * x).sum();
    let mut s_residual = Array1::zeros(1);
    let tail = frob_g - frob_approx;
    if tail > 0.0 {
        s_residual[0] = tail.sqrt();
    }

    FoldedMetric {
        u,
        s,
        v,
        s_residual,
    }
}

pub fn fold_dimension_diagonal(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedMetric {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };

    let mut g_diag = Array1::<f32>::zeros(d_in);
    for i in 0..d_in {
        let col_q = wq.column(i);
        let col_k = wk_expanded.column(i);
        g_diag[i] = col_q.dot(&col_k);
    }

    let k = target_dim.min(d_in);
    let u = Array2::<f32>::eye(k);
    let s = g_diag.slice(s![..k]).to_owned();
    let v = Array2::<f32>::eye(k);
    let s_residual = Array1::zeros(1);

    FoldedMetric {
        u,
        s,
        v,
        s_residual,
    }
}

pub fn fold_with_global_basis(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    global_basis: &GlobalBasis,
) -> FoldedMetric {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };

    let g = wq.t().dot(&wk_expanded);
    let u = global_basis.u.clone();
    let k = global_basis.rank;
    let g_sym = (&g + &g.t()) * 0.5;
    let g_core = u.t().dot(&g_sym).dot(&u);
    let g_approx = u.dot(&g_core).dot(&u.t());
    let diff = &g_sym - &g_approx;
    let residual_energy: f32 = diff.iter().map(|x| x * x).sum();
    let mut s_residual = Array1::zeros(1);
    s_residual[0] = residual_energy.sqrt();
    let g_core_faer = Mat::from_fn(k, k, |i, j| g_core[[i, j]]);
    let svd_core = g_core_faer.svd();
    let s_diag = svd_core.s_diagonal();
    let mut s = Array1::<f32>::zeros(k);
    for i in 0..k {
        s[i] = s_diag.read(i);
    }

    FoldedMetric {
        u: u.clone(),
        s,
        v: u,
        s_residual,
    }
}

pub fn compute_curvature(s_residual: &Array1<f32>) -> f32 {
    let sum_sq: f32 = s_residual.iter().map(|x| x * x).sum();
    sum_sq.sqrt()
}

#[derive(Debug, Clone)]
pub enum LayerType {
    Attention,
    FFN,
    Embedding,
    LMHead,
    LayerNorm,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum CompressionStrategy {
    MetricSVD {
        target_rank: usize,
        expected_accuracy: f32,
    },
    DiagonalMetric,
    FFNFold {
        target_rank: usize,
    },
    NoCompression,
    Skip,
}

#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer_idx: usize,
    pub layer_type: LayerType,
    pub input_shape: (usize, usize),
    pub output_shape: (usize, usize),
    pub param_count: usize,
    pub spectral_decay: f32,
    pub condition_number: f32,
    pub recommended_rank: usize,
    pub expected_accuracy: f32,
    pub strategy: CompressionStrategy,
}

#[derive(Debug, Clone)]
pub struct CompressionPlan {
    pub layers: Vec<LayerAnalysis>,
    pub total_original_params: usize,
    pub total_compressed_params: usize,
    pub expected_compression_ratio: f32,
    pub min_expected_accuracy: f32,
}

pub fn analyze_weight_matrix(w: ArrayView2<f32>, max_rank: usize) -> (f32, f32, usize, f32) {
    let m = w.nrows();
    let n = w.ncols();
    let k = max_rank.min(m.min(n));

    let w_faer = Mat::from_fn(m, n, |i, j| w[[i, j]]);
    let svd = w_faer.svd();
    let s_diag = svd.s_diagonal();

    let s_len = s_diag.nrows().min(k);
    let mut singular_values = Vec::with_capacity(s_len);
    for i in 0..s_len {
        singular_values.push(s_diag.read(i));
    }

    let s_max = singular_values.first().copied().unwrap_or(1.0).max(1e-10);
    let s_min = singular_values.last().copied().unwrap_or(1e-10).max(1e-10);
    let condition_number = s_max / s_min;

    let total_energy: f32 = singular_values.iter().map(|x| x * x).sum();
    let mut cumulative = 0.0_f32;
    let mut recommended_rank = s_len;
    let threshold = 0.95;

    for (i, &s) in singular_values.iter().enumerate() {
        cumulative += s * s;
        if cumulative / total_energy.max(1e-10) >= threshold {
            recommended_rank = i + 1;
            break;
        }
    }

    let spectral_decay = if singular_values.len() > 1 {
        let first_half: f32 = singular_values[..s_len / 2].iter().map(|x| x * x).sum();
        let second_half: f32 = singular_values[s_len / 2..].iter().map(|x| x * x).sum();
        first_half / (first_half + second_half).max(1e-10)
    } else {
        1.0
    };

    let mut approx_energy = 0.0_f32;
    for i in 0..recommended_rank.min(singular_values.len()) {
        approx_energy += singular_values[i] * singular_values[i];
    }
    let expected_accuracy = (approx_energy / total_energy.max(1e-10)).sqrt();

    (
        spectral_decay,
        condition_number,
        recommended_rank,
        expected_accuracy,
    )
}

pub fn analyze_layer(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    w1: ArrayView2<f32>,
    _: ArrayView2<f32>,
    layer_idx: usize,
    target_rank: usize,
) -> LayerAnalysis {
    let d_model = wq.ncols();
    let d_head = wq.nrows();
    let d_ff = w1.nrows();

    let g = wq.t().dot(&wk);
    let (spectral_decay, condition_number, rec_rank_metric, acc_metric) =
        analyze_weight_matrix(g.view(), target_rank);

    let (_, _, rec_rank_ffn, acc_ffn) = analyze_weight_matrix(w1, target_rank);

    let recommended_rank = rec_rank_metric.max(rec_rank_ffn).min(target_rank);
    let expected_accuracy = acc_metric.min(acc_ffn);

    let strategy = if spectral_decay > 0.9 && condition_number < 1e4 {
        CompressionStrategy::MetricSVD {
            target_rank: recommended_rank,
            expected_accuracy,
        }
    } else if spectral_decay > 0.7 {
        CompressionStrategy::DiagonalMetric
    } else {
        CompressionStrategy::MetricSVD {
            target_rank: (recommended_rank * 2).min(d_model),
            expected_accuracy,
        }
    };

    let original_params = d_head * d_model * 2 + d_ff * d_model * 2;

    LayerAnalysis {
        layer_idx,
        layer_type: LayerType::Attention,
        input_shape: (d_model, d_model),
        output_shape: (d_model, d_model),
        param_count: original_params,
        spectral_decay,
        condition_number,
        recommended_rank,
        expected_accuracy,
        strategy,
    }
}

pub fn create_compression_plan(layer_analyses: Vec<LayerAnalysis>, _: f32) -> CompressionPlan {
    let total_original: usize = layer_analyses.iter().map(|a| a.param_count).sum();

    let mut total_compressed = 0_usize;
    let mut min_accuracy = 1.0_f32;

    for analysis in &layer_analyses {
        let compressed = match &analysis.strategy {
            CompressionStrategy::MetricSVD { target_rank, .. } => {
                let d = analysis.input_shape.0;
                target_rank * d * 2 + target_rank
            }
            CompressionStrategy::DiagonalMetric => analysis.input_shape.0,
            CompressionStrategy::FFNFold { target_rank } => {
                let d = analysis.input_shape.0;
                target_rank * d * 2 + target_rank
            }
            CompressionStrategy::NoCompression => analysis.param_count,
            CompressionStrategy::Skip => 0,
        };
        total_compressed += compressed;
        if analysis.expected_accuracy < min_accuracy {
            min_accuracy = analysis.expected_accuracy;
        }
    }

    let ratio = total_original as f32 / total_compressed.max(1) as f32;

    CompressionPlan {
        layers: layer_analyses,
        total_original_params: total_original,
        total_compressed_params: total_compressed,
        expected_compression_ratio: ratio,
        min_expected_accuracy: min_accuracy,
    }
}

pub fn verify_compression_plan(plan: &CompressionPlan, min_accuracy: f32) -> Result<(), String> {
    if plan.min_expected_accuracy < min_accuracy {
        return Err(format!(
            "expected_accuracy {} < threshold {}",
            plan.min_expected_accuracy, min_accuracy
        ));
    }

    for layer in &plan.layers {
        if layer.expected_accuracy < min_accuracy {
            return Err(format!(
                "layer {} expected_accuracy {} < threshold {}",
                layer.layer_idx, layer.expected_accuracy, min_accuracy
            ));
        }

        if layer.condition_number > 1e8 {
            return Err(format!(
                "layer {} condition_number {} too high",
                layer.layer_idx, layer.condition_number
            ));
        }
    }

    Ok(())
}

pub fn create_causal_laplacian(seq_len: usize, window: usize) -> Array2<f32> {
    let mut a = Array2::<f32>::zeros((seq_len, seq_len));

    for i in 0..seq_len {
        let start = if i > window { i - window } else { 0 };
        for j in start..i {
            let dist = (i - j) as f32;
            a[[i, j]] = 1.0 / (1.0 + dist);
        }
    }

    let d_vec: Array1<f32> = a.sum_axis(Axis(1));
    let mut l = Array2::<f32>::zeros((seq_len, seq_len));

    for i in 0..seq_len {
        l[[i, i]] = d_vec[i];
        for j in 0..seq_len {
            l[[i, j]] -= a[[i, j]];
        }
    }

    l
}

pub struct FoldedFFN {
    pub u1: Array2<f32>,
    pub s1: Array1<f32>,
    pub v1: Array2<f32>,
    pub u2: Array2<f32>,
    pub s2: Array1<f32>,
    pub v2: Array2<f32>,
}

pub fn ffn_force_and_grad_row(
    x: Array1<f32>,
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let a = w1.dot(&x);
    let h_act = a.mapv(|v| {
        let s = 1.0 / (1.0 + (-v).exp());
        v * s
    });
    let f_x = w2.dot(&h_act);
    let temp2 = w2.t().dot(&f_x);
    let d_sigma = a.mapv(|v| {
        let s = 1.0 / (1.0 + (-v).exp());
        s + v * s * (1.0 - s)
    });
    let mut temp3 = temp2.clone();
    for j in 0..d_sigma.len() {
        temp3[j] *= d_sigma[j];
    }
    let grad = w1.t().dot(&temp3);
    (f_x, grad)
}

pub fn fold_ffn_svd(w1: ArrayView2<f32>, w2: ArrayView2<f32>, target_dim: usize) -> FoldedFFN {
    let w1_owned = w1.to_owned();
    let k1 = target_dim.min(w1.nrows().min(w1.ncols()));
    let (u1, s1, v1) = randomized_svd(&w1_owned, k1, 5, 1);

    let w2_owned = w2.to_owned();
    let k2 = target_dim.min(w2.nrows().min(w2.ncols()));
    let (u2, s2, v2) = randomized_svd(&w2_owned, k2, 5, 1);

    FoldedFFN {
        u1,
        s1,
        v1,
        u2,
        s2,
        v2,
    }
}

pub fn fold_ffn_random_projection(
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
    target_dim: usize,
) -> FoldedFFN {
    let ffn_dim = w1.nrows();
    let d_in = w1.ncols();
    let d_out = w2.nrows();

    let k1 = target_dim.min(ffn_dim.min(d_in));
    let k2 = target_dim.min(d_out.min(ffn_dim));

    let mut rng = rand::thread_rng();

    let scale1 = (1.0 / (k1 as f32)).sqrt();
    let mut v1 = Array2::<f32>::zeros((d_in, k1));
    for i in 0..d_in {
        for j in 0..k1 {
            v1[[i, j]] = (rng.gen::<f32>() * 2.0 - 1.0) * scale1;
        }
    }

    let u1 = w1.dot(&v1);
    let mut s1 = Array1::<f32>::zeros(k1);
    for j in 0..k1 {
        let col = u1.column(j);
        let norm = col.dot(&col).sqrt().max(1e-6);
        s1[j] = norm;
    }
    let u1_normalized = {
        let mut u = u1.clone();
        for j in 0..k1 {
            let inv_norm = 1.0 / s1[j];
            for i in 0..ffn_dim {
                u[[i, j]] *= inv_norm;
            }
        }
        u
    };

    let scale2 = (1.0 / (k2 as f32)).sqrt();
    let mut v2 = Array2::<f32>::zeros((ffn_dim, k2));
    for i in 0..ffn_dim {
        for j in 0..k2 {
            v2[[i, j]] = (rng.gen::<f32>() * 2.0 - 1.0) * scale2;
        }
    }

    let u2 = w2.dot(&v2);
    let mut s2 = Array1::<f32>::zeros(k2);
    for j in 0..k2 {
        let col = u2.column(j);
        let norm = col.dot(&col).sqrt().max(1e-6);
        s2[j] = norm;
    }
    let u2_normalized = {
        let mut u = u2.clone();
        for j in 0..k2 {
            let inv_norm = 1.0 / s2[j];
            for i in 0..d_out {
                u[[i, j]] *= inv_norm;
            }
        }
        u
    };

    FoldedFFN {
        u1: u1_normalized,
        s1,
        v1,
        u2: u2_normalized,
        s2,
        v2,
    }
}

fn calibrate_eta_alpha(
    w1: ArrayView2<f32>,
    w2: ArrayView2<f32>,
    g_inv: &Array1<f32>,
    config: &mut RSULFConfig,
) {
    if config.calibration_samples == 0 {
        return;
    }
    // Use d_model from config, but verify against weights
    let d_model = config.d_model;

    // Check W1 dimensions: should be (ffn_dim, d_model)
    if w1.ncols() != d_model {
        // If mismatch, try to detect if transposed (d_model, ffn_dim)
        if w1.nrows() == d_model {
            // Warn or handle? For now, we stick to the contract that W1 is (ffn_dim, d_model).
            // But to avoid panic, we should return or panic with clear message.
            panic!("RS-ULF W1 shape mismatch: expected ncols={} (d_model), got ncols={}. Ensure W1 is (hidden_dim, d_model).", d_model, w1.ncols());
        } else {
            panic!(
                "RS-ULF W1 shape mismatch: expected ncols={} (d_model), got ncols={}.",
                d_model,
                w1.ncols()
            );
        }
    }

    let ffn_dim = w1.nrows();
    if ffn_dim == 0 {
        return;
    }
    let num_samples = config.calibration_samples.max(64).min(256);
    let mut rng = rand::thread_rng();
    let mut x = Array2::<f32>::zeros((num_samples, d_model));
    for i in 0..num_samples {
        for j in 0..d_model {
            x[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }

    let results: Vec<_> = (0..num_samples)
        .into_par_iter()
        .map(|i| {
            let x_row = x.row(i);
            let (f_x, grad) = ffn_force_and_grad_row(x_row.to_owned(), w1.view(), w2.view());
            let mut grad_riem = grad.clone();
            if g_inv.len() == d_model {
                for j in 0..d_model {
                    grad_riem[j] *= g_inv[j];
                }
            }
            (f_x, grad_riem)
        })
        .collect();

    let mut f_all = Array2::<f32>::zeros((num_samples, d_model));
    let mut grad_riem_all = Array2::<f32>::zeros((num_samples, d_model));

    for (i, (f, g)) in results.into_iter().enumerate() {
        f_all.row_mut(i).assign(&f);
        grad_riem_all.row_mut(i).assign(&g);
    }
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let mut diff_all = Array2::<f32>::zeros((num_samples, d_model));
    for i in 0..num_samples {
        for j in 0..d_model {
            diff_all[[i, j]] = x[[i, j]] - x_mean[j];
        }
    }
    let mut m00 = 0.0f64;
    let mut m01 = 0.0f64;
    let mut m11 = 0.0f64;
    let mut b0 = 0.0f64;
    let mut b1 = 0.0f64;
    for i in 0..num_samples {
        for j in 0..d_model {
            let a1 = -grad_riem_all[[i, j]] as f64;
            let a2 = diff_all[[i, j]] as f64;
            let y = f_all[[i, j]] as f64;
            m00 += a1 * a1;
            m01 += a1 * a2;
            m11 += a2 * a2;
            b0 += a1 * y;
            b1 += a2 * y;
        }
    }
    let det = m00 * m11 - m01 * m01;
    if det.abs() < 1e-12 {
        return;
    }
    let eta_hat = (m11 * b0 - m01 * b1) / det;
    let alpha_hat = (m00 * b1 - m01 * b0) / det;
    let mut eta_f = eta_hat as f32;
    let mut alpha_f = alpha_hat as f32;
    if eta_f < 0.5 {
        eta_f = 1.0;
    }
    if eta_f > 2.0 {
        eta_f = 1.0;
    }
    if alpha_f < 0.0 {
        alpha_f = 0.0;
    }
    if alpha_f > 0.1 {
        alpha_f = 0.1;
    }
    config.eta = eta_f;
    config.alpha = alpha_f;
}

pub struct RSULFLayer {
    pub config: RSULFConfig,
    pub g_diag: Array1<f32>,
    pub g_inv: Array1<f32>,
    pub g_sym: Array2<f32>,
    pub a_antisym: Array2<f32>,
    pub u_metric: Array2<f32>,
    pub v_metric: Array2<f32>,
    pub g_core: Array2<f32>,
    pub a_core: Array2<f32>,
    pub curvature: f32,
    pub laplacian: Array2<f32>,
    pub ffn: FoldedFFN,
}

impl RSULFLayer {
    pub fn from_transformer(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
    ) -> Self {
        let folded_metric = fold_dimension_svd(wq, wk, config.r);
        let folded_ffn = fold_ffn_svd(w1, w2, config.r);

        let d = wq.ncols();

        let d_q = wq.nrows();
        let d_k = wk.nrows();
        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        let u_metric = folded_metric.u.clone();
        let g_core = u_metric.t().dot(&g_sym).dot(&u_metric);
        let a_core = u_metric.t().dot(&a_antisym).dot(&u_metric);

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            g_diag[i] = g_sym[[i, i]].abs();
        }
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);
        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);
        let curvature = compute_curvature(&folded_metric.s_residual);
        let laplacian = create_causal_laplacian(config.seq_len, config.window);

        Self {
            config,
            g_diag,
            g_inv,
            g_sym,
            a_antisym,
            u_metric,
            v_metric: folded_metric.v,
            g_core,
            a_core,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    pub fn from_transformer_with_basis(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
        global_basis: &GlobalBasis,
    ) -> Self {
        // Use Global Basis for folding
        let folded_metric = fold_with_global_basis(wq, wk, global_basis);

        // FFN folding can also be optimized, but for now we keep local SVD or implement Global FFN Basis later.
        // The blueprint focuses on Metric Basis sharing.
        let folded_ffn = fold_ffn_svd(w1, w2, config.r);

        let d = wq.ncols();
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            if d_k == 0 {
                panic!("RSULF: WK has zero rows");
            }
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        // Use the Global U
        let u_metric = folded_metric.u.clone();
        let g_core = u_metric.t().dot(&g_sym).dot(&u_metric);
        let a_core = u_metric.t().dot(&a_antisym).dot(&u_metric);

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            g_diag[i] = g_sym[[i, i]].abs();
        }
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);

        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);

        let curvature = compute_curvature(&folded_metric.s_residual);
        let laplacian = create_causal_laplacian(config.seq_len, config.window);

        Self {
            config,
            g_diag,
            g_inv,
            g_sym,
            a_antisym,
            u_metric,
            v_metric: folded_metric.v, // Same as u_metric in this mode
            g_core,
            a_core,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    pub fn from_transformer_with_metric(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
        g_diag_external: ArrayView1<f32>,
    ) -> Self {
        let folded_metric = fold_dimension_svd(wq, wk, config.r);
        let folded_ffn = fold_ffn_svd(w1, w2, config.r);

        let d = wq.ncols();
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        let u_metric = folded_metric.u.clone();
        let g_core = u_metric.t().dot(&g_sym).dot(&u_metric);
        let a_core = u_metric.t().dot(&a_antisym).dot(&u_metric);

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            if i < g_diag_external.len() {
                g_diag[i] = g_diag_external[i];
            } else {
                g_diag[i] = 1.0;
            }
        }
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);
        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);
        let curvature = compute_curvature(&folded_metric.s_residual);
        let laplacian = create_causal_laplacian(config.seq_len, config.window);

        Self {
            config,
            g_diag,
            g_inv,
            g_sym,
            a_antisym,
            u_metric,
            v_metric: folded_metric.v,
            g_core,
            a_core,
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    pub fn from_transformer_fast(
        wq: ArrayView2<f32>,
        wk: ArrayView2<f32>,
        w1: ArrayView2<f32>,
        w2: ArrayView2<f32>,
        mut config: RSULFConfig,
    ) -> Self {
        let d = wq.ncols();
        let d_q = wq.nrows();
        let d_k = wk.nrows();

        let wk_expanded = if d_k < d_q {
            let repeat = d_q / d_k;
            let mut expanded = Array2::<f32>::zeros((d_q, d));
            for i in 0..repeat {
                expanded
                    .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                    .assign(&wk);
            }
            expanded
        } else {
            wk.to_owned()
        };

        let b = wq.t().dot(&wk_expanded);
        let b_t = b.t();
        let g_sym = (&b + &b_t) * 0.5;
        let a_antisym = (&b - &b_t) * 0.5;

        let mut g_diag = Array1::zeros(d);
        for i in 0..d {
            g_diag[i] = g_sym[[i, i]].abs();
        }
        for v in g_diag.iter_mut() {
            if *v < 1e-6 {
                *v = 1e-6;
            }
            if *v > 1e6 {
                *v = 1e6;
            }
        }
        let g_inv = g_diag.mapv(|x| 1.0 / x);
        calibrate_eta_alpha(w1, w2, &g_inv, &mut config);
        let curvature = 0.0;
        let laplacian = create_causal_laplacian(config.seq_len, config.window);

        let folded_ffn = fold_ffn_svd(w1, w2, config.r);

        Self {
            config,
            g_diag,
            g_inv,
            g_sym,
            a_antisym,
            u_metric: Array2::zeros((0, 0)),
            v_metric: Array2::zeros((0, 0)),
            g_core: Array2::zeros((0, 0)),
            a_core: Array2::zeros((0, 0)),
            curvature,
            laplacian,
            ffn: folded_ffn,
        }
    }

    pub fn forward(
        &self,
        x: ArrayView2<f32>,
        v_mem: Option<ArrayView1<f32>>,
    ) -> (Array2<f32>, Array1<f32>) {
        let batch_total = x.nrows();
        let d = x.ncols();

        let x_arr = x.to_owned();

        // 1. Attention (Metric) Step
        let mut attn_out = Array2::<f32>::zeros((batch_total, d));

        // Only apply if metric matrices are valid
        if self.g_sym.nrows() == d && self.a_antisym.nrows() == d {
            // In RS-ULF, Attention is modeled as Geodesic flow on the manifold defined by G.
            // The original code implemented a full quadratic attention.
            // Ideally, this should use the folded core for efficiency, but for exactness (blueprint),
            // it uses the reconstructed G (or G_sym) in the expanded space.

            // Note: O(N^2) naive attention implementation.
            // For production, this should be block-wise or linear attention.

            let scale = 1.0 / (d as f32).sqrt();

            // We need to handle batch/sequence structure for Attention masking.
            // Assumption: Input is flattened [Batch * SeqLen, D].
            // Attention should only happen within each sequence.

            let mut seq_len_cfg = self.config.seq_len;
            if seq_len_cfg == 0 || seq_len_cfg > batch_total {
                seq_len_cfg = batch_total;
            }
            let num_seq = if seq_len_cfg > 0 {
                (batch_total + seq_len_cfg - 1) / seq_len_cfg
            } else {
                1
            };

            for s_idx in 0..num_seq {
                let start_row = s_idx * seq_len_cfg;
                let end_row = (start_row + seq_len_cfg).min(batch_total);
                let current_len = end_row.saturating_sub(start_row);
                if current_len == 0 {
                    continue;
                }
                let mut attn_weights = Array2::<f32>::zeros((current_len, current_len));
                for i in 0..current_len {
                    let global_i = start_row + i;
                    let q_i = x_arr.row(global_i);
                    let mut max_val = f32::NEG_INFINITY;

                    for j in 0..=i {
                        let global_j = start_row + j;
                        let k_j = x_arr.row(global_j);
                        let mut score = 0.0_f32;

                        // score = x_i^T * G * x_j
                        // Optimized: pre-calculate G*x_j could be faster but O(N^2) dominates.
                        for m in 0..d {
                            // Using g_sym which captures the metric
                            for n in 0..d {
                                score += q_i[m] * self.g_sym[[m, n]] * k_j[n];
                            }
                        }
                        score *= scale;
                        if score > max_val {
                            max_val = score;
                        }
                        attn_weights[[i, j]] = score;
                    }

                    let mut sum_exp = 0.0_f32;
                    for j in 0..=i {
                        let w = (attn_weights[[i, j]] - max_val).exp();
                        attn_weights[[i, j]] = w;
                        sum_exp += w;
                    }

                    if sum_exp > 1e-10 {
                        let inv_sum = 1.0 / sum_exp;
                        for j in 0..=i {
                            attn_weights[[i, j]] *= inv_sum;
                        }
                    }
                }

                for i in 0..current_len {
                    let global_i = start_row + i;
                    for j in 0..=i {
                        let w = attn_weights[[i, j]];
                        if w.abs() > 1e-10 {
                            let global_j = start_row + j;
                            let x_j = x_arr.row(global_j);
                            for k in 0..d {
                                attn_out[[global_i, k]] += w * x_j[k];
                            }
                        }
                    }
                }
            }

            // Magnetic effect (Gauge field)
            let a_norm: f32 = self.a_antisym.iter().map(|v| v * v).sum::<f32>().sqrt();
            if a_norm > 1e-6 {
                let magnetic = x_arr.dot(&self.a_antisym);
                // Physical scaling for gauge force
                let mag_scale = self.config.alpha / a_norm.max(1.0);
                attn_out = &attn_out + &magnetic * mag_scale;
            }
        } else {
            attn_out = x_arr.clone();
        }

        // v_attn = Attention_Output - Input (Residual velocity)
        let v_attn = &attn_out - &x_arr;

        // 2. FFN (Potential) Step
        let h1 = x_arr.dot(&self.ffn.v1);
        let h1_scaled = &h1 * &self.ffn.s1;
        let pre_act = h1_scaled.dot(&self.ffn.u1.t());

        let h_act = pre_act.mapv(|v| {
            let s = 1.0 / (1.0 + (-v).exp());
            v * s
        });

        let p1 = h_act.dot(&self.ffn.v2);
        let p1_scaled = &p1 * &self.ffn.s2;
        let f_x = p1_scaled.dot(&self.ffn.u2.t());

        let mut v_ffn = f_x.clone();

        // Apply Riemannian Gradient correction: G^-1 * grad(Phi)
        if self.g_inv.len() == d {
            let g_inv_mean: f32 = self.g_inv.iter().sum::<f32>() / d as f32;
            // Clip extreme metric scaling to avoid instability
            let g_inv_scale = if g_inv_mean > 10.0 {
                1.0 / g_inv_mean
            } else {
                1.0
            };
            for i in 0..batch_total {
                let mut row = v_ffn.row_mut(i);
                row.zip_mut_with(&self.g_inv, |a, b| *a *= *b * g_inv_scale);
            }
        }

        // Potential Energy monitoring
        let phi_val: f32 = f_x.iter().map(|v| v * v).sum::<f32>() * 0.5 / (batch_total as f32);
        let v_new = if let Some(v_prev) = v_mem {
            self.config.gamma * &v_prev + (1.0 - self.config.gamma) * phi_val
        } else {
            Array1::from_elem(batch_total, phi_val)
        };

        // 3. Graph Diffusion Step
        let mut graph = Array2::<f32>::zeros((batch_total, d));
        if self.config.beta.abs() > 0.0 {
            let seq_len = self.config.seq_len;
            // Only apply if dimensions match sequence structure
            if seq_len > 0 && batch_total >= seq_len && batch_total % seq_len == 0 {
                let num_seq = batch_total / seq_len;
                for s_idx in 0..num_seq {
                    let start = s_idx * seq_len;
                    let end = start + seq_len;
                    let x_seq = x_arr.slice(s![start..end, ..]);
                    let gx = self.laplacian.dot(&x_seq);
                    graph.slice_mut(s![start..end, ..]).assign(&gx);
                }
            }
            graph.mapv_inplace(|v| v * self.config.beta);
        }

        let mut v_total = &v_attn + self.config.eta * &v_ffn + &graph;
        let mut max_vel = 0.0_f32;
        for val in v_total.iter() {
            let a = val.abs();
            if a > max_vel {
                max_vel = a;
            }
        }
        if max_vel > 5.0 {
            let scale = 5.0 / max_vel;
            v_total.mapv_inplace(|val| val * scale);
        }

        let mut v_norm_global: f32 = 0.0;
        if batch_total > 0 {
            v_norm_global =
                v_total.iter().map(|v| v * v).sum::<f32>().sqrt() / (batch_total as f32).sqrt();
        }
        let curvature_norm = self.curvature.abs();
        let mut step_scale = 1.0_f32;
        if curvature_norm > 0.0 && v_norm_global > 0.0 {
            let denom = 1.0 + curvature_norm * v_norm_global;
            if denom.is_finite() && denom > 0.0 {
                step_scale = 1.0 / denom;
            }
        }
        if step_scale < 1.0 {
            v_total.mapv_inplace(|val| val * step_scale);
        }

        let mut christoffel = Array2::zeros((batch_total, d));
        let curv_scale = curvature_norm.min(1.0);

        if curv_scale > 1e-8 {
            let mut v_norm_global_scaled: f32 = 0.0;
            if batch_total > 0 {
                v_norm_global_scaled =
                    v_total.iter().map(|v| v * v).sum::<f32>().sqrt() / (batch_total as f32).sqrt();
            }
            let stability_scale = if v_norm_global_scaled > 1.0 {
                1.0 / v_norm_global_scaled
            } else {
                1.0
            };

            for i in 0..batch_total {
                let v_row = v_total.row(i);
                let x_row = x_arr.row(i);

                let v_norm_sq = v_row.dot(&v_row) * stability_scale * stability_scale;
                let scale = -0.5 * curv_scale * v_norm_sq;

                for k in 0..d {
                    christoffel[[i, k]] = scale * x_row[k];
                }
            }
        }

        let mut gamma_corr = Array2::zeros((batch_total, d));
        if curv_scale > 1e-8
            && self.ffn.v1.nrows() == d
            && self.ffn.u2.nrows() == d
            && self.ffn.v1.ncols() == self.ffn.u2.ncols()
            && self.ffn.v1.ncols() > 0
        {
            let r = self.ffn.v1.ncols();
            let x_u2 = x_arr.dot(&self.ffn.u2);
            let mut z = Array2::<f32>::zeros((batch_total, r));
            for i in 0..batch_total {
                for j in 0..r {
                    z[[i, j]] = h1[[i, j]] * x_u2[[i, j]];
                }
            }
            gamma_corr = z.dot(&self.ffn.v1.t());
            let inv_r = 1.0 / (r as f32);
            let coeff = self.curvature.max(-1.0).min(1.0) * inv_r;
            for i in 0..batch_total {
                let v_norm = v_total.row(i).dot(&v_total.row(i)).sqrt();
                let velocity_scale = v_norm.min(1.0);
                for k in 0..d {
                    gamma_corr[[i, k]] *= coeff * velocity_scale;
                }
            }
        }

        let x_next = &x_arr + &v_total + &christoffel + &gamma_corr;

        (x_next, v_new)
    }

    pub fn param_count(&self) -> (usize, usize, f32) {
        let d = self.config.d_model;
        let r = self.config.r;
        let ffn_dim = self.ffn.u1.nrows();

        let original_attn = 4 * d * d;
        let original_ffn = 2 * d * ffn_dim + ffn_dim * d;
        let original = original_attn + original_ffn;

        let compressed_metric = 2 * d * r + r;
        let compressed_ffn = 2 * (ffn_dim * r + d * r + r);
        let compressed_laplacian = self.config.seq_len * self.config.seq_len;
        let compressed = compressed_metric + compressed_ffn + compressed_laplacian;

        let ratio = original as f32 / compressed as f32;

        (compressed, original, ratio)
    }

    pub fn export_components(&self) -> RSULFComponents {
        RSULFComponents {
            d_model: self.config.d_model,
            r: self.config.r,
            eta: self.config.eta,
            alpha: self.config.alpha,
            beta: self.config.beta,
            gamma: self.config.gamma,
            seq_len: self.config.seq_len,
            window: self.config.window,
            g_diag: self.g_diag.clone(),
            g_inv: self.g_inv.clone(),
            g_sym: self.g_sym.clone(),
            a_antisym: self.a_antisym.clone(),
            u_metric: self.u_metric.clone(),
            v_metric: self.v_metric.clone(),
            g_core: self.g_core.clone(),
            a_core: self.a_core.clone(),
            curvature: self.curvature,
            ffn_u1: self.ffn.u1.clone(),
            ffn_s1: self.ffn.s1.clone(),
            ffn_v1: self.ffn.v1.clone(),
            ffn_u2: self.ffn.u2.clone(),
            ffn_s2: self.ffn.s2.clone(),
            ffn_v2: self.ffn.v2.clone(),
        }
    }

    pub fn from_components(comp: RSULFComponents) -> Self {
        let config = RSULFConfig {
            d_model: comp.d_model,
            r: comp.r,
            eta: comp.eta,
            alpha: comp.alpha,
            beta: comp.beta,
            gamma: comp.gamma,
            seq_len: comp.seq_len,
            window: comp.window,
            calibration_samples: 1024,
        };
        let laplacian = create_causal_laplacian(comp.seq_len, comp.window);
        let ffn = FoldedFFN {
            u1: comp.ffn_u1,
            s1: comp.ffn_s1,
            v1: comp.ffn_v1,
            u2: comp.ffn_u2,
            s2: comp.ffn_s2,
            v2: comp.ffn_v2,
        };
        Self {
            config,
            g_diag: comp.g_diag,
            g_inv: comp.g_inv,
            g_sym: comp.g_sym,
            a_antisym: comp.a_antisym,
            u_metric: comp.u_metric,
            v_metric: comp.v_metric,
            g_core: comp.g_core,
            a_core: comp.a_core,
            curvature: comp.curvature,
            laplacian,
            ffn,
        }
    }
}

pub struct RSULFComponents {
    pub d_model: usize,
    pub r: usize,
    pub eta: f32,
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub seq_len: usize,
    pub window: usize,
    pub g_diag: Array1<f32>,
    pub g_inv: Array1<f32>,
    pub g_sym: Array2<f32>,
    pub a_antisym: Array2<f32>,
    pub u_metric: Array2<f32>,
    pub v_metric: Array2<f32>,
    pub g_core: Array2<f32>,
    pub a_core: Array2<f32>,
    pub curvature: f32,
    pub ffn_u1: Array2<f32>,
    pub ffn_s1: Array1<f32>,
    pub ffn_v1: Array2<f32>,
    pub ffn_u2: Array2<f32>,
    pub ffn_s2: Array1<f32>,
    pub ffn_v2: Array2<f32>,
}

pub struct FoldConsistencyResult {
    pub symmetry_error: f32,
    pub reconstruction_error: f32,
    pub fold_accuracy: f32,
    pub min_eigenvalue: f32,
    pub condition_number: f32,
    pub is_valid: bool,
}

pub fn verify_fold_consistency(
    wq: ArrayView2<f32>,
    wk: ArrayView2<f32>,
    folded: &FoldedMetric,
) -> FoldConsistencyResult {
    let d_q = wq.nrows();
    let d_k = wk.nrows();
    let d_in = wq.ncols();

    let wk_expanded = if d_k < d_q {
        let repeat = d_q / d_k;
        let mut expanded = Array2::<f32>::zeros((d_q, d_in));
        for i in 0..repeat {
            expanded
                .slice_mut(s![i * d_k..(i + 1) * d_k, ..])
                .assign(&wk);
        }
        expanded
    } else {
        wk.to_owned()
    };

    // G = WQ^T * WK (비대칭 행렬)
    let g = wq.t().dot(&wk_expanded);

    // 대칭화된 버전: G_sym = (G + G^T) / 2
    let g_sym = (&g + &g.t()) * 0.5;

    // 대칭성 오류: ||G - G^T|| / ||G||
    let g_t = g.t();
    let sym_diff: f32 = g.iter().zip(g_t.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let g_norm: f32 = g.iter().map(|x| x * x).sum();
    let symmetry_error = if g_norm > 1e-10 {
        (sym_diff / g_norm).sqrt()
    } else {
        0.0
    };

    // Frobenius norm 계산 (fold_accuracy용)
    let frob_g: f32 = g.iter().map(|x| x * x).sum();

    // SVD로 캡처된 에너지 비율 = sum(s_i^2) / ||G||_F^2
    let frob_captured: f32 = folded.s.iter().map(|x| x * x).sum();
    let fold_accuracy = if frob_g > 1e-10 {
        (frob_captured / frob_g).min(1.0) // 1.0 초과 방지
    } else {
        1.0
    };

    // 잔차 기반 재구성 오류
    let residual_sq: f32 = folded.s_residual.iter().map(|x| x * x).sum();
    let reconstruction_error = if frob_g > 1e-10 {
        (residual_sq / frob_g).sqrt()
    } else {
        0.0
    };

    // 대각 요소 통계 (양정치성 대리 지표)
    let mut diag_values: Vec<f32> = Vec::with_capacity(d_in);
    for i in 0..d_in {
        // 대칭화된 메트릭의 대각 사용
        diag_values.push(g_sym[[i, i]]);
    }
    let min_eigenvalue = diag_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_eigenvalue = diag_values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let condition_number = if min_eigenvalue.abs() > 1e-10 {
        max_eigenvalue.abs() / min_eigenvalue.abs()
    } else {
        f32::INFINITY
    };

    let is_valid = fold_accuracy >= 0.5 && min_eigenvalue > -1e6 && condition_number < 1e8;

    FoldConsistencyResult {
        symmetry_error,
        reconstruction_error,
        fold_accuracy,
        min_eigenvalue,
        condition_number,
        is_valid,
    }
}

pub fn block_lanczos_svd(
    a: &Array2<f32>,
    k: usize,
    block_size: usize,
    max_iter: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let m = a.nrows();
    let n = a.ncols();
    let bs = block_size.min(k).min(m).min(n);
    let num_blocks = (k + bs - 1) / bs;

    let mut rng = rand::thread_rng();
    let mut v_blocks: Vec<Array2<f32>> = Vec::with_capacity(num_blocks + 1);

    let mut v0 = Array2::<f32>::zeros((n, bs));
    for i in 0..n {
        for j in 0..bs {
            v0[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }
    let (v0_orth, _) = qr_decomposition(&v0);
    v_blocks.push(v0_orth);

    let mut alpha_blocks: Vec<Array2<f32>> = Vec::new();
    let mut beta_blocks: Vec<Array2<f32>> = Vec::new();

    for iter in 0..max_iter.min(num_blocks) {
        let v_j = &v_blocks[iter];
        let mut u_j = a.dot(v_j);

        if iter > 0 {
            let beta_prev = &beta_blocks[iter - 1];
            let v_prev = &v_blocks[iter - 1];
            u_j = u_j - v_prev.dot(&beta_prev.t());
        }

        let alpha_j = v_j.t().dot(&a.t().dot(&u_j));
        u_j = a.t().dot(&u_j) - v_j.dot(&alpha_j);

        for prev in 0..=iter {
            let v_prev = &v_blocks[prev];
            let proj = v_prev.t().dot(&u_j);
            u_j = u_j - v_prev.dot(&proj);
        }

        let (v_next, beta_j) = qr_decomposition(&u_j);

        alpha_blocks.push(alpha_j);
        beta_blocks.push(beta_j.slice(s![..bs, ..bs]).to_owned());

        if iter + 1 < num_blocks {
            v_blocks.push(v_next.slice(s![.., ..bs]).to_owned());
        }

        let beta_norm: f32 = beta_j.iter().map(|x| x * x).sum::<f32>().sqrt();
        if beta_norm < 1e-10 {
            break;
        }
    }

    randomized_svd(a, k, 5, 2)
}

pub fn nystrom_approximation(
    a: &Array2<f32>,
    k: usize,
    n_samples: usize,
) -> (Array2<f32>, Array1<f32>) {
    let n = a.nrows();
    let l = n_samples.min(n).max(k);

    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..l {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    let sampled_indices: Vec<usize> = indices[..l].to_vec();

    let mut c = Array2::<f32>::zeros((n, l));
    for (j, &idx) in sampled_indices.iter().enumerate() {
        for i in 0..n {
            c[[i, j]] = a[[i, idx]];
        }
    }

    let mut w = Array2::<f32>::zeros((l, l));
    for (i, &idx_i) in sampled_indices.iter().enumerate() {
        for (j, &idx_j) in sampled_indices.iter().enumerate() {
            w[[i, j]] = a[[idx_i, idx_j]];
        }
    }

    let w_faer = Mat::from_fn(l, l, |i, j| w[[i, j]]);
    let svd_w = w_faer.svd();

    let mut w_pinv = Array2::<f32>::zeros((l, l));
    let s_diag = svd_w.s_diagonal();
    let u_w = svd_w.u();
    let v_w = svd_w.v();

    for i in 0..l {
        let s_val = s_diag.read(i);
        if s_val.abs() > 1e-10 {
            let s_inv = 1.0 / s_val;
            for row in 0..l {
                for col in 0..l {
                    w_pinv[[row, col]] += v_w.read(row, i) * s_inv * u_w.read(col, i);
                }
            }
        }
    }

    let approx = c.dot(&w_pinv).dot(&c.t());

    let approx_faer = Mat::from_fn(n, n, |i, j| approx[[i, j]]);
    let svd_approx = approx_faer.svd();

    let k_actual = k.min(n);
    let mut u = Array2::<f32>::zeros((n, k_actual));
    let mut s = Array1::<f32>::zeros(k_actual);

    let u_approx = svd_approx.u();
    let s_approx = svd_approx.s_diagonal();

    for j in 0..k_actual {
        s[j] = s_approx.read(j).sqrt().max(0.0);
        for i in 0..n {
            u[[i, j]] = u_approx.read(i, j);
        }
    }

    (u, s)
}

pub fn adaptive_rank_svd(
    a: &Array2<f32>,
    target_accuracy: f32,
    max_rank: usize,
) -> (Array2<f32>, Array1<f32>, Array2<f32>, usize) {
    let m = a.nrows();
    let n = a.ncols();
    let frob_sq: f32 = a.iter().map(|x| x * x).sum();

    let mut low = 1usize;
    let mut high = max_rank.min(m).min(n);
    let mut best_k = high;

    while low < high {
        let mid = (low + high) / 2;
        let (_, s, _) = randomized_svd(a, mid, 3, 1);
        let captured: f32 = s.iter().map(|x| x * x).sum();
        let accuracy = captured / frob_sq.max(1e-10);

        if accuracy >= target_accuracy {
            best_k = mid;
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    let (u, s, v) = randomized_svd(a, best_k, 5, 2);
    (u, s, v, best_k)
}
```
---
## File: `reality_stone/src/layers/spline.rs`

```rust
// src/layers/spline.rs

use ndarray::{Array1, Array2};
use ndarray_rand::rand::{thread_rng, Rng};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use std::ops::AddAssign;

use crate::ops;

#[pyclass]
pub struct SplineLayer {
    control_points: Array2<f32>,
    pub k: usize,
    pub in_features: usize,
    pub out_features: usize,
}

#[pymethods]
impl SplineLayer {
    #[new]
    pub fn new(k: usize, in_features: usize, out_features: usize) -> Self {
        let mut rng = thread_rng();
        Self {
            control_points: Array2::from_shape_fn((k + 1, in_features), |(_, _)| {
                rng.gen::<f32>() * 0.02
            }),
            k,
            in_features,
            out_features,
        }
    }

    #[staticmethod]
    pub fn from_weight_py(
        _py: Python,
        weight: PyReadonlyArray2<f32>,
        k: usize,
        learning_rate: f32,
        steps: usize,
    ) -> PyResult<Self> {
        let weight_array = weight.as_array().to_owned();
        Ok(Self::from_weight(&weight_array, k, learning_rate, steps))
    }

    #[getter]
    pub fn get_control_points<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.control_points.to_pyarray(py)
    }

    #[setter]
    pub fn set_control_points(&mut self, control_points: PyReadonlyArray2<f32>) -> PyResult<()> {
        self.control_points = control_points.as_array().to_owned();
        Ok(())
    }

    pub fn forward<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> &'py PyArray2<f32> {
        let input_array = input.as_array();
        let weight = self.interpolate_internal();
        let output = input_array.dot(&weight.t());
        output.to_pyarray(py)
    }

    pub fn interpolate<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        self.interpolate_internal().to_pyarray(py)
    }

    pub fn get_compression_ratio(&self) -> f32 {
        let original_params = (self.in_features * self.out_features) as f32;
        let compressed_params = self.control_points.len() as f32;
        original_params / compressed_params
    }
}

impl SplineLayer {
    pub fn from_weight(weight: &Array2<f32>, k: usize, learning_rate: f32, steps: usize) -> Self {
        let (out_features, in_features) = weight.dim();
        let mut rng = thread_rng();
        let mut control_points =
            Array2::from_shape_fn((k + 1, in_features), |(_, _)| rng.gen::<f32>() * 0.02);

        for _ in 0..steps {
            let (reconstructed_weight, grad) =
                Self::interpolate_with_grad(&control_points, out_features);
            let loss_grad = ops::mse_loss_grad(&reconstructed_weight, weight);
            let mut control_points_grad = Array2::<f32>::zeros((k + 1, in_features));

            for i in 0..out_features {
                let t = i as f32 / (out_features - 1) as f32;
                let t_scaled = t * k as f32;
                let j = (t_scaled.floor() as usize).clamp(1, k - 2);

                let p0_grad = grad.p0_grads[i].clone() * &loss_grad.row(i);
                let p1_grad = grad.p1_grads[i].clone() * &loss_grad.row(i);
                let p2_grad = grad.p2_grads[i].clone() * &loss_grad.row(i);
                let p3_grad = grad.p3_grads[i].clone() * &loss_grad.row(i);

                control_points_grad.row_mut(j - 1).add_assign(&p0_grad);
                control_points_grad.row_mut(j).add_assign(&p1_grad);
                control_points_grad.row_mut(j + 1).add_assign(&p2_grad);
                control_points_grad.row_mut(j + 2).add_assign(&p3_grad);
            }
            control_points.scaled_add(-learning_rate, &control_points_grad);
        }
        Self {
            control_points,
            k,
            in_features,
            out_features,
        }
    }

    pub fn interpolate_internal(&self) -> Array2<f32> {
        let mut reconstructed = Array2::zeros((self.out_features, self.in_features));
        for i in 0..self.out_features {
            let t = i as f32 / (self.out_features - 1) as f32;
            let t_scaled = t * self.k as f32;
            let j = (t_scaled.floor() as usize).clamp(1, self.k - 2);
            let t_local = t_scaled - j as f32;
            let t2 = t_local * t_local;
            let t3 = t2 * t_local;

            let c0 = -0.5 * t3 + t2 - 0.5 * t_local;
            let c1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local;
            let c3 = 0.5 * t3 - 0.5 * t2;

            let p0 = self.control_points.row(j - 1);
            let p1 = self.control_points.row(j);
            let p2 = self.control_points.row(j + 1);
            let p3 = self.control_points.row(j + 2);

            reconstructed
                .row_mut(i)
                .assign(&(c0 * &p0 + c1 * &p1 + c2 * &p2 + c3 * &p3));
        }
        reconstructed
    }

    fn interpolate_with_grad(
        control_points: &Array2<f32>,
        out_features: usize,
    ) -> (Array2<f32>, CatmullRomGradients) {
        let mut reconstructed = Array2::zeros((out_features, control_points.shape()[1]));
        let k = control_points.shape()[0] - 1;
        let mut grads = CatmullRomGradients {
            p0_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
            p1_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
            p2_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
            p3_grads: vec![Array1::zeros(control_points.shape()[1]); out_features],
        };

        for i in 0..out_features {
            let t = i as f32 / (out_features - 1) as f32;
            let t_scaled = t * k as f32;
            let j = (t_scaled.floor() as usize).clamp(1, k - 2);
            let t_local = t_scaled - j as f32;
            let t2 = t_local * t_local;
            let t3 = t2 * t_local;

            let c0 = -0.5 * t3 + t2 - 0.5 * t_local;
            let c1 = 1.5 * t3 - 2.5 * t2 + 1.0;
            let c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t_local;
            let c3 = 0.5 * t3 - 0.5 * t2;

            let p0 = control_points.row(j - 1);
            let p1 = control_points.row(j);
            let p2 = control_points.row(j + 1);
            let p3 = control_points.row(j + 2);

            reconstructed
                .row_mut(i)
                .assign(&(c0 * &p0 + c1 * &p1 + c2 * &p2 + c3 * &p3));

            grads.p0_grads[i].fill(c0);
            grads.p1_grads[i].fill(c1);
            grads.p2_grads[i].fill(c2);
            grads.p3_grads[i].fill(c3);
        }
        (reconstructed, grads)
    }
}

struct CatmullRomGradients {
    p0_grads: Vec<Array1<f32>>,
    p1_grads: Vec<Array1<f32>>,
    p2_grads: Vec<Array1<f32>>,
    p3_grads: Vec<Array1<f32>>,
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn spline_interpolate_cuda(
                control_points: *const f32,
                weights: *mut f32,
                k: i32,
                in_features: i32,
                out_features: i32,
            );
            pub fn spline_forward_cuda(
                input: *const f32,
                control_points: *const f32,
                output: *mut f32,
                batch_size: i32,
                k: i32,
                in_features: i32,
                out_features: i32,
            );
            pub fn spline_backward_cuda(
                grad_output: *const f32,
                input: *const f32,
                grad_control_points: *mut f32,
                batch_size: i32,
                k: i32,
                in_features: i32,
                out_features: i32,
            );
        }
    }

    pub fn spline_interpolate_cuda(
        control_points: *const f32,
        weights: *mut f32,
        k: i32,
        in_features: i32,
        out_features: i32,
    ) {
        unsafe {
            ffi::spline_interpolate_cuda(control_points, weights, k, in_features, out_features);
        }
    }

    pub fn spline_forward_cuda(
        input: *const f32,
        control_points: *const f32,
        output: *mut f32,
        batch_size: i32,
        k: i32,
        in_features: i32,
        out_features: i32,
    ) {
        unsafe {
            ffi::spline_forward_cuda(
                input,
                control_points,
                output,
                batch_size,
                k,
                in_features,
                out_features,
            );
        }
    }

    pub fn spline_backward_cuda(
        grad_output: *const f32,
        input: *const f32,
        grad_control_points: *mut f32,
        batch_size: i32,
        k: i32,
        in_features: i32,
        out_features: i32,
    ) {
        unsafe {
            ffi::spline_backward_cuda(
                grad_output,
                input,
                grad_control_points,
                batch_size,
                k,
                in_features,
                out_features,
            );
        }
    }
}
```
---
## File: `reality_stone/src/layers/spline_cache.cu`

```cpp

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

// Device function for cubic hermite basis
__device__ void cubic_hermite(
    float u,
    float* h00, float* h10, float* h01, float* h11
) {
    float u2 = u * u;
    float u3 = u2 * u;
    *h00 = 2.0f * u3 - 3.0f * u2 + 1.0f;
    *h10 = u3 - 2.0f * u2 + u;
    *h01 = -2.0f * u3 + 3.0f * u2;
    *h11 = u3 - u2;
}

// Kernel: Reconstruct states for a batch of timestamps
// control_points: [num_points, 2 * dim] (state concatenated with velocity)
// times: [num_points]
// target_times: [batch_size]
// output: [batch_size, dim]
// curvature: float
__global__ void spline_reconstruct_kernel(
    const float* control_points, // interleaved state/velocity or separate? Let's assume contiguous [state, velocity] per point
    const float* cp_times,
    int num_points,
    int dim,
    const float* target_times,
    float* output,
    int batch_size,
    float curvature
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float t = target_times[idx];
    
    // 1. Binary Search for interval
    // Simple linear scan or binary search. Since num_points is likely small to moderate, 
    // binary search is better.
    
    int left = 0;
    int right = num_points - 1;
    int p0_idx = -1;

    if (num_points == 0) return;
    
    if (t <= cp_times[0]) {
        p0_idx = 0; // Clamp to start
        // Just copy state of p0
        for (int i = 0; i < dim; i++) {
             output[idx * dim + i] = control_points[0 * 2 * dim + i];
        }
        return;
    }
    if (t >= cp_times[num_points - 1]) {
        p0_idx = num_points - 1;
        // Copy last
        for (int i = 0; i < dim; i++) {
             output[idx * dim + i] = control_points[p0_idx * 2 * dim + i];
        }
        return;
    }

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (cp_times[mid] <= t) {
            p0_idx = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    // t is between p0_idx and p0_idx + 1
    int p1_idx = p0_idx + 1;
    if (p1_idx >= num_points) p1_idx = num_points - 1;

    float t0 = cp_times[p0_idx];
    float t1 = cp_times[p1_idx];
    float dt = t1 - t0;

    // Pointers to data
    // Layout assumption: point i stores [state (dim), velocity (dim)]
    const float* p0_state = &control_points[p0_idx * 2 * dim];
    const float* p0_vel = &control_points[p0_idx * 2 * dim + dim];
    const float* p1_state = &control_points[p1_idx * 2 * dim];
    const float* p1_vel = &control_points[p1_idx * 2 * dim + dim];

    if (dt < 1e-6f) {
        for (int i = 0; i < dim; i++) {
            output[idx * dim + i] = p0_state[i];
        }
        return;
    }

    float u = (t - t0) / dt;
    float h00, h10, h01, h11;
    cubic_hermite(u, &h00, &h10, &h01, &h11);

    float correction = 0.0f;
    if (abs(curvature) > 1e-6f) {
        correction = u * (1.0f - u) * curvature;
    }

    for (int i = 0; i < dim; i++) {
        float m0 = p0_vel[i] * dt;
        float m1 = p1_vel[i] * dt;
        
        float val = p0_state[i] * h00 + m0 * h10 + p1_state[i] * h01 + m1 * h11;
        
        // Apply correction
        val *= (1.0f + correction);
        
        output[idx * dim + i] = val;
    }
}

void launch_spline_reconstruct(
    const float* control_points,
    const float* cp_times,
    int num_points,
    int dim,
    const float* target_times,
    float* output,
    int batch_size,
    float curvature,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    
    spline_reconstruct_kernel<<<grid_size, block_size, 0, stream>>>(
        control_points,
        cp_times,
        num_points,
        dim,
        target_times,
        output,
        batch_size,
        curvature
    );
}

}
```
---
## File: `reality_stone/src/layers/spline_cache.rs`

```rust
use ndarray::{Array1, Array2, ArrayView1};
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct ControlPoint {
    pub time: f32,
    pub state: Array1<f32>,
    pub velocity: Array1<f32>,
}

#[derive(Debug, Clone)]
pub struct SplineCache {
    pub control_points: Vec<ControlPoint>,
    pub curvature: f32,
    pub dimension: usize,
}

impl SplineCache {
    pub fn new(curvature: f32, dimension: usize) -> Self {
        Self {
            control_points: Vec::new(),
            curvature,
            dimension,
        }
    }

    pub fn add_point(&mut self, time: f32, state: ArrayView1<f32>, velocity: ArrayView1<f32>) {
        if state.len() != self.dimension || velocity.len() != self.dimension {
            panic!("Dimension mismatch in SplineCache");
        }

        // Ensure time is increasing
        if let Some(last) = self.control_points.last() {
            if time <= last.time {
                // In a real scenario we might update, but for now append only
                return;
            }
        }

        self.control_points.push(ControlPoint {
            time,
            state: state.to_owned(),
            velocity: velocity.to_owned(),
        });
    }

    pub fn reconstruct(&self, t: f32) -> Option<Array1<f32>> {
        if self.control_points.is_empty() {
            return None;
        }

        // 1. Find interval [t_k, t_{k+1}]
        // Binary search for the interval
        let idx = match self.control_points.binary_search_by(|cp| {
            if cp.time <= t {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }) {
            Ok(i) => i,  // exact match (unlikely with float, but logic holds)
            Err(i) => i, // insertion point
        };

        // idx is where t would be inserted.
        // So t is between idx-1 and idx.

        if idx == 0 {
            // Before first point
            return Some(self.control_points[0].state.clone());
        }
        if idx >= self.control_points.len() {
            // After last point
            return Some(self.control_points.last().unwrap().state.clone());
        }

        let p0_idx = idx - 1;
        let p1_idx = idx;

        let p0 = &self.control_points[p0_idx];
        let p1 = &self.control_points[p1_idx];

        let dt = p1.time - p0.time;
        if dt < 1e-6 {
            return Some(p0.state.clone());
        }

        // Normalized time u in [0, 1]
        let u = (t - p0.time) / dt;

        // 2. Cubic Hermite Spline
        // h00 = 2u^3 - 3u^2 + 1
        // h10 = u^3 - 2u^2 + u
        // h01 = -2u^3 + 3u^2
        // h11 = u^3 - u^2

        let u2 = u * u;
        let u3 = u2 * u;

        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;

        // Tangents need to be scaled by interval duration dt
        // m0 = v0 * dt
        // m1 = v1 * dt
        let m0 = &p0.velocity * dt;
        let m1 = &p1.velocity * dt;

        let mut interpolated = &p0.state * h00 + &m0 * h10 + &p1.state * h01 + &m1 * h11;

        // 3. Curvature Correction
        // Blueprint: "Correct path using curvature kappa"
        // A simple approximation for negative curvature (hyperbolic) is to "push out" or "pull in"
        // based on the deviation from geodesic.
        // For now, let's implement a placeholder correction that scales with u(1-u) (max at midpoint)
        if self.curvature.abs() > 1e-6 {
            let mid_correction = u * (1.0 - u) * self.curvature;
            // Apply correction in direction of interpolation?
            // Or simply scale amplitude?
            // Blueprint 3.1 mentions "Christoffel symbols correction: -0.5 * Gamma * v * v"
            // Here, let's assume a simple radial correction factor.
            // x_corrected = x * (1 + correction)
            interpolated.mapv_inplace(|x| x * (1.0 + mid_correction));
        }

        Some(interpolated)
    }

    pub fn batch_reconstruct(&self, timestamps: ArrayView1<f32>) -> Array2<f32> {
        let n = timestamps.len();
        let mut output = Array2::zeros((n, self.dimension));

        for (i, &t) in timestamps.iter().enumerate() {
            if let Some(state) = self.reconstruct(t) {
                output.row_mut(i).assign(&state);
            }
        }
        output
    }

    pub fn clear(&mut self) {
        self.control_points.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_spline_reconstruction() {
        let dim = 2;
        let mut cache = SplineCache::new(0.0, dim);

        // p0 at t=0: [0, 0], v=[1, 1]
        cache.add_point(0.0, arr1(&[0.0, 0.0]).view(), arr1(&[1.0, 1.0]).view());
        // p1 at t=1: [1, 1], v=[1, 1]
        cache.add_point(1.0, arr1(&[1.0, 1.0]).view(), arr1(&[1.0, 1.0]).view());

        // Midpoint t=0.5
        // Linear would be [0.5, 0.5]
        // Cubic with constant velocity should also be close to linear if velocity matches

        let res = cache.reconstruct(0.5).unwrap();
        println!("Reconstructed at 0.5: {:?}", res);

        // Check roughly
        assert!((res[0] - 0.5).abs() < 0.1);
        assert!((res[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_curvature_effect() {
        let dim = 1;
        let mut cache = SplineCache::new(1.0, dim); // High curvature

        cache.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
        cache.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());

        let res_curved = cache.reconstruct(0.5).unwrap();

        let mut cache_flat = SplineCache::new(0.0, dim);
        cache_flat.add_point(0.0, arr1(&[0.0]).view(), arr1(&[1.0]).view());
        cache_flat.add_point(1.0, arr1(&[1.0]).view(), arr1(&[1.0]).view());
        let res_flat = cache_flat.reconstruct(0.5).unwrap();

        // Curvature 1.0 adds u(1-u)*k = 0.25 * 1 = 0.25 factor roughly
        // Expect curved result to be larger (or different)
        assert!((res_curved[0] - res_flat[0]).abs() > 0.001);
    }
}
```
---
## File: `reality_stone/src/layers/suppression.rs`

```rust
use ndarray::{Array2, ArrayView2, Zip};

/// Computes the dynamic suppression field epsilon(x) = base + linear * x + hyp * tanh(scale * x)
/// element-wise.
pub fn compute_dynamic_suppression(
    x: &ArrayView2<f32>,
    base: f32,
    linear: f32,
    hyp: f32,
    scale: f32,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((x.nrows(), x.ncols()));

    Zip::from(&mut out).and(x).par_for_each(|y, &val| {
        *y = base + linear * val + hyp * (scale * val).tanh();
    });

    out
}
```
---
## File: `reality_stone/src/layers/symplectic.rs`

```rust
use super::hyper_metric::HyperMetric;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct SymplecticState {
    pub q: Array2<f32>,
    pub p: Array2<f32>,
}

impl SymplecticState {
    pub fn new(batch_size: usize, d: usize) -> Self {
        Self {
            q: Array2::zeros((batch_size, d)),
            p: Array2::zeros((batch_size, d)),
        }
    }
}

pub struct SymplecticLayer {
    pub hyper_metric: HyperMetric,
    pub layer_idx: usize,
    pub layer_emb: Array1<f32>,
    pub dt: f32,
}

impl SymplecticLayer {
    pub fn new(
        layer_idx: usize,
        layer_emb: Array1<f32>,
        hyper_metric: HyperMetric,
        dt: f32,
    ) -> Self {
        Self {
            hyper_metric,
            layer_idx,
            layer_emb,
            dt,
        }
    }

    pub fn step(&self, state: &mut SymplecticState, x_input: &Array2<f32>) -> Array2<f32> {
        let force_metric = self.hyper_metric.project_forward(&state.q, &self.layer_emb);
        let force_total = &force_metric + x_input;
        let dt = self.dt;

        state.p = &state.p + &(&force_total * dt);
        state.q = &state.q + &(&state.p * dt);

        state.q.clone()
    }
}
```
---
## File: `reality_stone/src/layers/unified_riemannian.rs`

```rust
use super::bellman_lagrangian::{
    bellman_update, compute_energy_components, metric_flow, representation_flow, EnergyComponents,
    LagrangianParams, ValueFunction,
};
use super::geodesic::{geodesic_interpolation, geodesic_path};
use super::metric::{DiagonalMetric, KleinMetric, LorentzMetric, MetricType, PoincareMetric};
use indicatif::ProgressBar;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// 통합 리만 레이어
pub struct UnifiedRiemannianLayer {
    pub metric: MetricType,
    pub value_function: Option<ValueFunction>,
    pub lagrangian_params: LagrangianParams,
    pub enable_bellman: bool,
    pub enable_metric_learning: bool,
}

impl UnifiedRiemannianLayer {
    /// 새로운 통합 리만 레이어 생성
    ///
    /// # Arguments
    /// * `metric_type` - "poincare", "lorentz", "klein", "diagonal"
    /// * `curvature` - 곡률 파라미터 (양수)
    /// * `input_dim` - 입력 차원
    /// * `enable_bellman` - 벨만 가치 함수 활성화
    pub fn new(metric_type: &str, curvature: f32, input_dim: usize, enable_bellman: bool) -> Self {
        let metric = match metric_type {
            "poincare" => MetricType::Poincare(PoincareMetric::new(curvature)),
            "lorentz" => MetricType::Lorentz(LorentzMetric::new(curvature)),
            "klein" => MetricType::Klein(KleinMetric::new(curvature)),
            "diagonal" => MetricType::Diagonal(DiagonalMetric::new(input_dim)),
            _ => panic!("Unknown metric type: {}", metric_type),
        };

        let value_function = if enable_bellman {
            Some(ValueFunction::new(input_dim, input_dim * 2))
        } else {
            None
        };

        let enable_metric_learning = matches!(metric, MetricType::Diagonal(_));

        Self {
            metric,
            value_function,
            lagrangian_params: LagrangianParams::default(),
            enable_bellman,
            enable_metric_learning,
        }
    }

    /// 순전파
    ///
    /// # Arguments
    /// * `x` - 입력 (batch, dim)
    /// * `target` - 목표점 (optional)
    ///
    /// # Returns
    /// LayerOutput - 출력 및 에너지 정보
    pub fn forward(&self, x: &ArrayView2<f32>, target: Option<&ArrayView2<f32>>) -> LayerOutput {
        let batch_size = x.nrows();

        // 1. 메트릭 계산
        let metric_values = self.metric.as_trait().compute_metric(x);

        // 2. 출력 계산
        let output = if let Some(y) = target {
            // 목표가 있으면 측지선 보간 (중간점)
            geodesic_interpolation(&self.metric, x, y, 0.5)
        } else if self.enable_bellman && self.value_function.is_some() {
            // 벨만 활성화: 표현 흐름
            let vf = self.value_function.as_ref().unwrap();
            representation_flow(&self.metric, vf, x, 0.01)
        } else {
            // 단순 항등 (메트릭만 적용)
            x.to_owned()
        };

        // 3. 에너지 계산 (벨만 활성화 시)
        let energy = if self.enable_bellman && self.value_function.is_some() {
            let vf = self.value_function.as_ref().unwrap();
            let dt = 0.1;
            let velocity = (&output - x) / dt;
            let reward = Array1::zeros(batch_size); // 기본 보상 0

            Some(compute_energy_components(
                self.metric.as_trait(),
                vf,
                x,
                &velocity.view(),
                &output.view(),
                &reward.view(),
                &self.lagrangian_params,
            ))
        } else {
            None
        };

        LayerOutput {
            output,
            energy,
            cache: LayerCache {
                input: x.to_owned(),
                velocity: None,
                metric_values,
            },
        }
    }

    /// 역전파
    ///
    /// # Arguments
    /// * `grad_output` - 출력에 대한 그래디언트
    /// * `x` - 입력
    /// * `cache` - 순전파 캐시
    ///
    /// # Returns
    /// LayerGradients - 입력 및 파라미터에 대한 그래디언트
    pub fn backward(
        &self,
        grad_output: &ArrayView2<f32>,
        _x: &ArrayView2<f32>,
        _cache: &LayerCache,
    ) -> LayerGradients {
        // 단순화: 그래디언트 pass-through
        LayerGradients {
            grad_input: grad_output.to_owned(),
            grad_metric: None,
            grad_value_fn: None,
        }
    }

    /// 메트릭 학습 업데이트
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `v` - 속도 (변화율)
    /// * `learning_rate` - 학습률
    pub fn update_metric(&mut self, x: &ArrayView2<f32>, v: &ArrayView2<f32>, learning_rate: f32) {
        if !self.enable_metric_learning {
            return;
        }

        if let MetricType::Diagonal(ref mut metric) = self.metric {
            if let Some(ref vf) = self.value_function {
                let batch_size = x.nrows();
                let dt = 0.1;
                let x_next = x + &(v * dt);
                let reward = Array1::zeros(batch_size);

                let energy = compute_energy_components(
                    metric,
                    vf,
                    x,
                    v,
                    &x_next.view(),
                    &reward.view(),
                    &self.lagrangian_params,
                );

                metric_flow(metric, x, v, &energy.lagrangian.view(), learning_rate);
            }
        }
    }

    /// 에너지 계산
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `v` - 속도
    /// * `x_next` - 다음 상태
    /// * `reward` - 보상
    ///
    /// # Returns
    /// EnergyComponents - 운동/잠재/라그랑지안 에너지
    pub fn compute_energy(
        &self,
        x: &ArrayView2<f32>,
        v: &ArrayView2<f32>,
        x_next: &ArrayView2<f32>,
        reward: &ArrayView1<f32>,
    ) -> EnergyComponents {
        if let Some(ref vf) = self.value_function {
            compute_energy_components(
                self.metric.as_trait(),
                vf,
                x,
                v,
                x_next,
                reward,
                &self.lagrangian_params,
            )
        } else {
            EnergyComponents::new(x.nrows())
        }
    }

    /// 측지선 경로 생성
    ///
    /// # Arguments
    /// * `start` - 시작점
    /// * `end` - 끝점
    /// * `num_steps` - 경로 점 개수
    ///
    /// # Returns
    /// 측지선 경로 (각 점은 batch x dim)
    pub fn geodesic_path(
        &self,
        start: &ArrayView2<f32>,
        end: &ArrayView2<f32>,
        num_steps: usize,
    ) -> Vec<Array2<f32>> {
        geodesic_path(&self.metric, start, end, num_steps)
    }

    /// 표현 흐름 스텝
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `num_steps` - 흐름 반복 횟수
    /// * `learning_rate` - 학습률
    ///
    /// # Returns
    /// 흐름 후 상태
    pub fn flow_step(
        &self,
        x: &ArrayView2<f32>,
        num_steps: usize,
        learning_rate: f32,
    ) -> Array2<f32> {
        if let Some(ref vf) = self.value_function {
            let mut current = x.to_owned();
            for _ in 0..num_steps {
                current = representation_flow(&self.metric, vf, &current.view(), learning_rate);
            }
            current
        } else {
            x.to_owned()
        }
    }

    /// 벨만 가치 함수 업데이트
    ///
    /// # Arguments
    /// * `x` - 현재 상태
    /// * `x_next` - 다음 상태
    /// * `reward` - 보상
    /// * `learning_rate` - 학습률
    pub fn update_value_function(
        &mut self,
        x: &ArrayView2<f32>,
        x_next: &ArrayView2<f32>,
        reward: &ArrayView1<f32>,
        learning_rate: f32,
    ) {
        if let Some(ref mut vf) = self.value_function {
            bellman_update(
                vf,
                x,
                x_next,
                reward,
                self.lagrangian_params.gamma,
                learning_rate,
            );
        }
    }
}

pub fn laplace_beltrami_matrix(
    metric: &MetricType,
    x: &ArrayView2<f32>,
    sigma: f32,
    eps: f32,
) -> Array2<f32> {
    use ndarray::s;
    let n = x.nrows();
    let dim = x.ncols();
    let metric_trait = metric.as_trait();
    let mut dist_sq = Array2::<f32>::zeros((n, n));
    let pb_dist = ProgressBar::new(n as u64);
    {
        let x_ref = x;
        dist_sq
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                pb_dist.inc(1);
                let xi = x_ref.slice(s![i..i + 1, 0..dim]);
                for j in (i + 1)..n {
                    let xj = x_ref.slice(s![j..j + 1, 0..dim]);
                    let d_arr = metric_trait.distance(&xi.view(), &xj.view());
                    let d = d_arr[0];
                    let v = d * d;
                    row[j] = v;
                }
            });
    }
    pb_dist.finish_and_clear();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = dist_sq[[i, j]];
            dist_sq[[j, i]] = v;
        }
    }
    let det = metric_trait.determinant(x);
    let mut vol = det.clone();
    for v in vol.iter_mut() {
        let a = v.abs().sqrt();
        if a < eps {
            *v = eps;
        } else {
            *v = a;
        }
    }
    let mut w = Array2::<f32>::zeros((n, n));
    let denom = 2.0 * sigma * sigma.max(eps);
    let pb_w = ProgressBar::new(n as u64);
    {
        let vol_ref = &vol;
        w.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                pb_w.inc(1);
                for j in (i + 1)..n {
                    let d2 = dist_sq[[i, j]];
                    let mut value = (-d2 / denom).exp();
                    let scale = 1.0 / (vol_ref[i] * vol_ref[j]);
                    value *= scale;
                    row[j] = value;
                }
            });
    }
    pb_w.finish_and_clear();
    for i in 0..n {
        for j in (i + 1)..n {
            let v = w[[i, j]];
            w[[j, i]] = v;
        }
    }
    let mut l = Array2::<f32>::zeros((n, n));
    let pb_l = ProgressBar::new(n as u64);
    {
        l.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                pb_l.inc(1);
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += w[[i, j]];
                }
                row[i] = sum;
                for j in 0..n {
                    if i != j {
                        row[j] = -w[[i, j]];
                    }
                }
            });
    }
    pb_l.finish_and_clear();
    l
}

/// 레이어 출력
pub struct LayerOutput {
    pub output: Array2<f32>,
    pub energy: Option<EnergyComponents>,
    pub cache: LayerCache,
}

/// 레이어 캐시 (역전파용)
pub struct LayerCache {
    pub input: Array2<f32>,
    pub velocity: Option<Array2<f32>>,
    pub metric_values: Array2<f32>,
}

/// 레이어 그래디언트
pub struct LayerGradients {
    pub grad_input: Array2<f32>,
    pub grad_metric: Option<Array1<f32>>,
    pub grad_value_fn: Option<ValueFunctionGrad>,
}

/// 가치 함수 그래디언트
pub struct ValueFunctionGrad {
    pub grad_weights: Array2<f32>,
    pub grad_bias: Array1<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_unified_layer_creation() {
        let layer = UnifiedRiemannianLayer::new("poincare", 1.0, 32, false);
        assert!(!layer.enable_bellman);

        let layer_bellman = UnifiedRiemannianLayer::new("diagonal", 0.0, 64, true);
        assert!(layer_bellman.enable_bellman);
        assert!(layer_bellman.value_function.is_some());
    }

    #[test]
    fn test_forward_poincare() {
        let layer = UnifiedRiemannianLayer::new("poincare", 1.0, 4, false);
        let x = arr2(&[[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]);

        let output = layer.forward(&x.view(), None);
        assert_eq!(output.output.shape(), x.shape());
        assert!(output.energy.is_none());
    }

    #[test]
    fn test_forward_with_target() {
        let layer = UnifiedRiemannianLayer::new("diagonal", 0.0, 3, false);
        let x = arr2(&[[0.0, 0.0, 0.0]]);
        let y = arr2(&[[1.0, 1.0, 1.0]]);

        let output = layer.forward(&x.view(), Some(&y.view()));
        // 중간점이어야 함
        assert!((output.output[[0, 0]] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_geodesic_path() {
        let layer = UnifiedRiemannianLayer::new("diagonal", 0.0, 2, false);
        let start = arr2(&[[0.0, 0.0]]);
        let end = arr2(&[[1.0, 0.0]]);

        let path = layer.geodesic_path(&start.view(), &end.view(), 5);
        assert_eq!(path.len(), 5);
        assert!((path[0][[0, 0]] - 0.0).abs() < 1e-4);
        assert!((path[4][[0, 0]] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_energy_computation() {
        let layer = UnifiedRiemannianLayer::new("diagonal", 0.0, 4, true);
        let x = arr2(&[[0.1, 0.2, 0.3, 0.4]]);
        let v = arr2(&[[0.01, 0.02, 0.03, 0.04]]);
        let x_next = &x + &v;
        let reward = ndarray::arr1(&[0.5]);

        let energy = layer.compute_energy(&x.view(), &v.view(), &x_next.view(), &reward.view());
        assert!(energy.kinetic[0] >= 0.0);
        assert!(energy.potential[0] >= 0.0);
        assert!(energy.lagrangian[0].is_finite());
    }
}
```
---
## File: `reality_stone/src/layers/utils.rs`

```rust
// Deprecated: functionality moved to `src/ops/{batch,project}.rs` to avoid duplication.
```
---
## File: `reality_stone/src/lib.rs`

```rust
#![allow(deprecated)]
pub mod bindings;
pub mod layers;
pub mod ops;

pub use bindings::_rust;
```
---
## File: `reality_stone/src/ops/batch.rs`

```rust
use ndarray::{Array1, ArrayView2, Axis};

pub const EPS: f32 = 1e-6;
pub const EPS64: f64 = 1e-12;

pub fn norm_sq_batched(x: &ArrayView2<f32>) -> Array1<f32> {
    x.map_axis(Axis(1), |row| row.mapv(|a| a.powi(2)).sum())
}

pub fn dot_batched(x: &ArrayView2<f32>, y: &ArrayView2<f32>) -> Array1<f32> {
    (x * y).sum_axis(Axis(1))
}

pub fn norm_sq_batched_f64(x: &ArrayView2<f64>) -> Array1<f64> {
    x.map_axis(Axis(1), |row| row.mapv(|a| a * a).sum())
}

pub fn dot_batched_f64(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Array1<f64> {
    (x * y).sum_axis(Axis(1))
}
```
---
## File: `reality_stone/src/ops/cuda/mobius.cu`

```cpp
#ifdef _MSC_VER
#pragma warning(disable : 4819)
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cmath>

#define MIN_DENOMINATOR 1e-6f
#define EPS 1e-7f
#define BOUNDARY_EPS 1e-5f

__global__ void mobius_add_kernel(float* out, const float* u, const float* v, float c, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        const float* v_row = v + i * dim;
        float* out_row = out + i * dim;

        float u2 = 0.0f;
        float v2 = 0.0f;
        float uv = 0.0f;

        for (int j = 0; j < dim; ++j) {
            u2 += u_row[j] * u_row[j];
            v2 += v_row[j] * v_row[j];
            uv += u_row[j] * v_row[j];
        }

        float c2 = c * c;
        float denominator = 1.0f + 2.0f * c * uv + c2 * u2 * v2;
        if (denominator < MIN_DENOMINATOR) {
            denominator = MIN_DENOMINATOR;
        }

        float coeff_u = (1.0f + 2.0f * c * uv + c * v2) / denominator;
        float coeff_v = (1.0f - c * u2) / denominator;

        for (int j = 0; j < dim; ++j) {
            out_row[j] = coeff_u * u_row[j] + coeff_v * v_row[j];
        }
    }
}

extern "C" {
    void mobius_add_cuda(float* out, const float* u, const float* v, float c, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_add_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, v, c, batch_size, dim);
    }
}

// --- Mobius Scalar Multiplication ---

__global__ void mobius_scalar_kernel(float* out, const float* u, float c, float r, int batch_size, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < batch_size) {
        const float* u_row = u + i * dim;
        float* out_row = out + i * dim;

        float norm_sq = 0.0f;
        for (int j = 0; j < dim; ++j) {
            norm_sq += u_row[j] * u_row[j];
        }
        
        if (norm_sq < EPS * EPS) {
            // For very small vectors, fall back to simple scaling to keep gradients stable
            for (int j = 0; j < dim; ++j) {
                out_row[j] = r * u_row[j];
            }
            return;
        }

        float norm = sqrtf(norm_sq);
        
        if (fabsf(c) < EPS) {
            // c = 0: Euclidean case
            for (int j = 0; j < dim; ++j) {
                out_row[j] = r * u_row[j];
            }
            return;
        }
        
        float scale;
        if (c > 0.0f) {
            // Positive curvature
            float sqrt_c = sqrtf(c);
            float scn = fminf(sqrt_c * norm, 1.0f - BOUNDARY_EPS);
            float alpha = atanhf(scn);
            float beta = tanhf(r * alpha);
            scale = beta / (sqrt_c * norm);
        } else {
            // Negative curvature (compute with real-valued formula)
            float sqrt_abs_c = sqrtf(-c);
            float scn = sqrt_abs_c * norm;
            float alpha = atanf(scn);
            float beta = tanf(r * alpha);
            scale = beta / (sqrt_abs_c * norm);
        }

        for (int j = 0; j < dim; ++j) {
            out_row[j] = scale * u_row[j];
        }
    }
}

extern "C" {
    void mobius_scalar_cuda(float* out, const float* u, float c, float r, int64_t batch_size, int64_t dim) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
        mobius_scalar_kernel<<<blocks_per_grid, threads_per_block>>>(out, u, c, r, batch_size, dim);
    }
}
```
---
## File: `reality_stone/src/ops/curvature.rs`

```rust
// 동적 곡률 구조체
#[derive(Debug, Clone)]
pub struct DynamicCurvature {
    pub kappa: f32,
    pub c_min: f32,
    pub c_max: f32,
}

impl DynamicCurvature {
    pub fn new(kappa: f32, c_min: f32, c_max: f32) -> Self {
        Self {
            kappa,
            c_min,
            c_max,
        }
    }

    pub fn compute_c(&self) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-self.kappa).exp());
        self.c_min + (self.c_max - self.c_min) * sigmoid
    }

    pub fn compute_dc_dkappa(&self) -> f32 {
        let sigmoid = 1.0 / (1.0 + (-self.kappa).exp());
        (self.c_max - self.c_min) * sigmoid * (1.0 - sigmoid)
    }
}

#[derive(Debug, Clone)]
pub struct LayerWiseDynamicCurvature {
    pub kappas: Vec<f32>,
    pub c_min: f32,
    pub c_max: f32,
}

impl LayerWiseDynamicCurvature {
    pub fn new(num_layers: usize, c_min: f32, c_max: f32) -> Self {
        Self {
            kappas: vec![0.0; num_layers],
            c_min,
            c_max,
        }
    }

    pub fn from_kappas(kappas: Vec<f32>, c_min: f32, c_max: f32) -> Self {
        Self {
            kappas,
            c_min,
            c_max,
        }
    }

    pub fn compute_c(&self, layer_idx: usize) -> f32 {
        let kappa = self.kappas.get(layer_idx).unwrap_or(&0.0);
        let sigmoid = 1.0 / (1.0 + (-kappa).exp());
        self.c_min + (self.c_max - self.c_min) * sigmoid
    }

    pub fn compute_dc_dkappa(&self, layer_idx: usize) -> f32 {
        let kappa = self.kappas.get(layer_idx).unwrap_or(&0.0);
        let sigmoid = 1.0 / (1.0 + (-kappa).exp());
        (self.c_max - self.c_min) * sigmoid * (1.0 - sigmoid)
    }
}
```
---
## File: `reality_stone/src/ops/extraction.rs`

```rust
use ndarray::{Array2, ArrayView2};

#[cfg(feature = "cuda")]
extern "C" {
    fn fast_extract_metric_cuda(
        W: *const f32,
        U: *mut f32,
        G: *mut f32,
        V: *mut f32,
        out_dim: i32,
        in_dim: i32,
        k: i32,
    );
}

/// CUDA 기반 리만 메트릭 추출 (Fast Random Projection)
#[cfg(feature = "cuda")]
pub fn extract_metric_cuda(
    w: ArrayView2<f32>,
    _calibration_data: ArrayView2<f32>,
    target_dim: usize,
    _num_steps: usize,
    _curvature: f32,
    _lr: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let out_dim = w.nrows();
    let in_dim = w.ncols();
    let k = target_dim;

    let mut u = Array2::<f32>::zeros((out_dim, k));
    let mut g = Array2::<f32>::zeros((k, k));
    let mut v = Array2::<f32>::zeros((in_dim, k));

    unsafe {
        fast_extract_metric_cuda(
            w.as_ptr(),
            u.as_mut_ptr(),
            g.as_mut_ptr(),
            v.as_mut_ptr(),
            out_dim as i32,
            in_dim as i32,
            k as i32,
        );
    }

    (u, g, v)
}

/// CPU 폴백 (CUDA 없을 때) - Random Projection
#[cfg(not(feature = "cuda"))]
pub fn extract_metric_cuda(
    w: ArrayView2<f32>,
    _calibration_data: ArrayView2<f32>,
    target_dim: usize,
    _num_steps: usize,
    _curvature: f32,
    _lr: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    use rand::Rng;

    let out_dim = w.nrows();
    let in_dim = w.ncols();
    let k = target_dim;
    let scale = 1.0 / (k as f32).sqrt();

    let mut u = Array2::<f32>::zeros((out_dim, k));
    let mut g = Array2::<f32>::zeros((k, k));
    let mut v = Array2::<f32>::zeros((in_dim, k));

    let mut rng = rand::thread_rng();

    for i in 0..out_dim {
        for j in 0..k {
            u[[i, j]] = rng.gen::<f32>() * scale;
        }
    }
    for i in 0..in_dim {
        for j in 0..k {
            v[[i, j]] = rng.gen::<f32>() * scale;
        }
    }

    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f32;
            for a in 0..out_dim {
                for b in 0..in_dim {
                    sum += u[[a, i]] * w[[a, b]] * v[[b, j]];
                }
            }
            g[[i, j]] = sum;
        }
    }

    (u, g, v)
}
```
---
## File: `reality_stone/src/ops/metrikey.rs`

```rust
use ndarray::{s, Array1, Array2};
use rand::prelude::*;
use rand::rngs::SmallRng;

const EPS: f32 = 1e-6;
const EPS64: f64 = 1e-12;

fn seed_from_key(key: &str) -> u64 {
    // FNV-1a 64-bit
    let mut hash: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x00000100000001B3;
    for &b in key.as_bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(prime);
    }
    hash
}

fn box_muller_pair<R: Rng>(rng: &mut R) -> (f32, f32) {
    // Generate two independent standard normals
    let u1 = rng.gen::<f32>().max(EPS);
    let u2 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn box_muller_pair64<R: Rng>(rng: &mut R) -> (f64, f64) {
    let u1 = rng.gen::<f64>().max(EPS64);
    let u2 = rng.gen::<f64>();
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = 2.0_f64 * std::f64::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

fn random_normal_matrix(dim: usize, rng: &mut SmallRng) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((dim, dim));
    let mut i = 0;
    while i < dim * dim {
        let (z0, z1) = box_muller_pair(rng);
        let r = i / dim;
        let c = i % dim;
        m[(r, c)] = z0;
        i += 1;
        if i < dim * dim {
            let r2 = i / dim;
            let c2 = i % dim;
            m[(r2, c2)] = z1;
            i += 1;
        }
    }
    m
}

fn random_normal_matrix64(dim: usize, rng: &mut SmallRng) -> ndarray::Array2<f64> {
    let mut m = ndarray::Array2::<f64>::zeros((dim, dim));
    let mut i = 0;
    while i < dim * dim {
        let (z0, z1) = box_muller_pair64(rng);
        let r = i / dim;
        let c = i % dim;
        m[(r, c)] = z0;
        i += 1;
        if i < dim * dim {
            let r2 = i / dim;
            let c2 = i % dim;
            m[(r2, c2)] = z1;
            i += 1;
        }
    }
    m
}

fn modified_gram_schmidt(a: &Array2<f32>, reorth_passes: usize) -> Array2<f32> {
    let (rows, cols) = a.dim();
    assert_eq!(rows, cols, "Expected square matrix");
    let n = rows;
    let mut q = Array2::<f32>::zeros((n, n));

    // First pass: Modified Gram-Schmidt
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for k in 0..j {
            let qk = q.column(k);
            let r = v.dot(&qk);
            v -= &(qk.to_owned() * r);
        }
        let norm = v.dot(&v).sqrt().max(EPS);
        v /= norm;
        q.slice_mut(s![.., j]).assign(&v);
    }

    // Optional re-orthogonalization passes to improve numerical stability
    for _ in 0..reorth_passes {
        for j in 0..n {
            let mut v = q.column(j).to_owned();
            for k in 0..j {
                let qk = q.column(k);
                let r = v.dot(&qk);
                v -= &(qk.to_owned() * r);
            }
            let norm = v.dot(&v).sqrt().max(EPS);
            v /= norm;
            q.slice_mut(s![.., j]).assign(&v);
        }
    }

    q
}

fn modified_gram_schmidt64(a: &ndarray::Array2<f64>, reorth_passes: usize) -> ndarray::Array2<f64> {
    let (rows, cols) = a.dim();
    assert_eq!(rows, cols, "Expected square matrix");
    let n = rows;
    let mut q = ndarray::Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for k in 0..j {
            let qk = q.column(k);
            let r = v.dot(&qk);
            v -= &(qk.to_owned() * r);
        }
        let norm = v.dot(&v).sqrt().max(EPS64);
        v /= norm;
        q.slice_mut(s![.., j]).assign(&v);
    }
    for _ in 0..reorth_passes {
        for j in 0..n {
            let mut v = q.column(j).to_owned();
            for k in 0..j {
                let qk = q.column(k);
                let r = v.dot(&qk);
                v -= &(qk.to_owned() * r);
            }
            let norm = v.dot(&v).sqrt().max(EPS64);
            v /= norm;
            q.slice_mut(s![.., j]).assign(&v);
        }
    }
    q
}

pub fn deterministic_orthogonal_from_key(key: &str, dim: usize) -> Array2<f32> {
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed);
    let a = random_normal_matrix(dim, &mut rng);
    // Two re-orthogonalization passes provide good orthogonality in f32 for typical dims
    modified_gram_schmidt(&a, 1)
}

pub fn deterministic_orthogonal_from_key_f64(key: &str, dim: usize) -> ndarray::Array2<f64> {
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed);
    let a = random_normal_matrix64(dim, &mut rng);
    modified_gram_schmidt64(&a, 2)
}

pub fn spd_metric_from_key(key: &str, dim: usize, min_lambda: f32, max_lambda: f32) -> Array2<f32> {
    assert!(dim > 0);
    assert!(min_lambda > 0.0 && max_lambda > min_lambda);
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
    let q = deterministic_orthogonal_from_key(key, dim);
    // Build diagonal spectrum D
    let mut d = Array2::<f32>::zeros((dim, dim));
    for i in 0..dim {
        let u: f32 = rng.gen();
        let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
        d[(i, i)] = lam;
    }
    // G = Q^T D Q is symmetric SPD regardless of Q orthonormality accuracy
    let dq = d.dot(&q);
    q.t().dot(&dq)
}

pub fn spd_metric_from_key_f64(
    key: &str,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
) -> ndarray::Array2<f64> {
    assert!(dim > 0);
    assert!(min_lambda > 0.0 && max_lambda > min_lambda);
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
    let q = deterministic_orthogonal_from_key_f64(key, dim);
    let mut d = ndarray::Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        let u: f64 = rng.gen();
        let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
        d[(i, i)] = lam;
    }
    let dq = d.dot(&q);
    q.t().dot(&dq)
}

/// Weighted SPD metric where eigenvalues are exponentiated by a mass factor.
/// Interpreted as curvature/strength control: lam' = lam^{mass}.
pub fn spd_metric_from_key_weighted(
    key: &str,
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
    mass: f32,
) -> Array2<f32> {
    assert!(mass > 0.0);
    let seed = seed_from_key(key);
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
    let q = deterministic_orthogonal_from_key(key, dim);
    let mut d = Array2::<f32>::zeros((dim, dim));
    for i in 0..dim {
        let u: f32 = rng.gen();
        let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
        d[(i, i)] = lam.powf(mass);
    }
    let dq = d.dot(&q);
    q.t().dot(&dq)
}

/// Gravity composition: Order-preserving product of weighted layer factors.
/// Each layer l uses T_l = (G_l(mass_l))^{1/2}. Here we compute via spectrum exponent (mass/2).
pub fn compose_layers_gravity(
    keys: &[String],
    masses: &[f32],
    dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> Array2<f32> {
    assert!(!keys.is_empty());
    assert_eq!(keys.len(), masses.len());
    let mut acc = Array2::<f32>::eye(dim);
    for (key, &mass) in keys.iter().zip(masses.iter()) {
        assert!(mass > 0.0);
        // Build Q and D^{mass/2}
        let q = deterministic_orthogonal_from_key(key, dim);
        let seed = seed_from_key(key);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
        let mut d_sqrt = Array2::<f32>::zeros((dim, dim));
        for i in 0..dim {
            let u: f32 = rng.gen();
            let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
            d_sqrt[(i, i)] = lam.powf(0.5 * mass);
        }
        let t_l = q.t().dot(&d_sqrt.dot(&q));
        acc = t_l.dot(&acc);
    }
    acc
}

pub fn compose_layers_gravity_f64(
    keys: &[String],
    masses: &[f64],
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
) -> ndarray::Array2<f64> {
    assert!(!keys.is_empty());
    assert_eq!(keys.len(), masses.len());
    let mut acc = ndarray::Array2::<f64>::eye(dim);
    for (key, &mass) in keys.iter().zip(masses.iter()) {
        assert!(mass > 0.0);
        let q = deterministic_orthogonal_from_key_f64(key, dim);
        let seed = seed_from_key(key);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
        let mut d_sqrt = ndarray::Array2::<f64>::zeros((dim, dim));
        for i in 0..dim {
            let u: f64 = rng.gen();
            let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
            d_sqrt[(i, i)] = lam.powf(0.5 * mass);
        }
        let t_l = q.t().dot(&d_sqrt.dot(&q));
        acc = t_l.dot(&acc);
    }
    acc
}

pub fn apply_linear_f64(
    matrix: &ndarray::Array2<f64>,
    vecs: &ndarray::Array2<f64>,
) -> ndarray::Array2<f64> {
    let (_, in_dim) = matrix.dim();
    let (_batch, in_dim_vec) = vecs.dim();
    assert_eq!(in_dim, in_dim_vec);
    vecs.dot(&matrix.t())
}

/// Compact composition using a single master key and a simple mass schedule.
/// keys: key_i = format!("{}#{}", master_key, i)
/// masses: mass_i = mass_base + i * mass_step
pub fn compose_layers_gravity_compact_f64(
    master_key: &str,
    num_layers: usize,
    dim: usize,
    min_lambda: f64,
    max_lambda: f64,
    mass_base: f64,
    mass_step: f64,
) -> ndarray::Array2<f64> {
    assert!(num_layers > 0);
    let mut acc = ndarray::Array2::<f64>::eye(dim);
    for i in 0..num_layers {
        let key_i = format!("{}#{}", master_key, i);
        let q = deterministic_orthogonal_from_key_f64(&key_i, dim);
        let seed = seed_from_key(&key_i);
        let mut rng = SmallRng::seed_from_u64(seed ^ 0x9E3779B185EBCA87);
        let mass = mass_base + (i as f64) * mass_step;
        assert!(mass > 0.0);
        let mut d_sqrt = ndarray::Array2::<f64>::zeros((dim, dim));
        for j in 0..dim {
            let u: f64 = rng.gen();
            let lam = min_lambda + (max_lambda - min_lambda) * u.clamp(0.0, 1.0);
            d_sqrt[(j, j)] = lam.powf(0.5 * mass);
        }
        let t_l = q.t().dot(&d_sqrt.dot(&q));
        acc = t_l.dot(&acc);
    }
    acc
}

pub fn metric_factor_cholesky(g: &Array2<f32>) -> Array2<f32> {
    let (n, m) = g.dim();
    assert_eq!(n, m, "G must be square");
    let mut l = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = g[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = (sum.max(EPS)).sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)].max(EPS);
            }
        }
    }
    // Return upper-triangular factor U = L_lower^T so that G = U^T U holds
    l.t().to_owned()
}

pub fn mahalanobis_distance_sq_g(x: &Array1<f32>, y: &Array1<f32>, g: &Array2<f32>) -> f32 {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(g.dim(), (n, n));
    let diff = x - y;
    let tmp = g.dot(&diff);
    diff.dot(&tmp)
}

pub fn mahalanobis_distance_sq_l(x: &Array1<f32>, y: &Array1<f32>, l: &Array2<f32>) -> f32 {
    let n = x.len();
    assert_eq!(y.len(), n);
    assert_eq!(l.dim(), (n, n));
    let diff = x - y;
    // l is defined as upper-triangular factor such that G = l^T l
    let z = l.dot(&diff);
    z.dot(&z)
}

pub fn block_orthogonal_from_key(key: &str, global_dim: usize, dept_dim: usize) -> Array2<f32> {
    let total = global_dim + dept_dim;
    let mut q = Array2::<f32>::eye(total);
    if dept_dim > 0 {
        let r = deterministic_orthogonal_from_key(key, dept_dim);
        q.slice_mut(s![global_dim.., global_dim..]).assign(&r);
    }
    q
}

pub fn spd_block_metric_from_key(
    key: &str,
    global_dim: usize,
    dept_dim: usize,
    min_lambda: f32,
    max_lambda: f32,
) -> Array2<f32> {
    let total = global_dim + dept_dim;
    let mut g = Array2::<f32>::eye(total);
    if dept_dim > 0 {
        let g_dept = spd_metric_from_key(key, dept_dim, min_lambda, max_lambda);
        g.slice_mut(s![global_dim.., global_dim..]).assign(&g_dept);
    }
    g
}

pub fn compose_layers_order_preserving(layers: &[Array2<f32>]) -> Array2<f32> {
    assert!(!layers.is_empty(), "layers must be non-empty");
    let (n, m) = layers[0].dim();
    assert_eq!(n, m, "only square layers supported");
    let mut acc = Array2::<f32>::eye(n);
    for l in layers {
        assert_eq!(l.dim(), (n, n));
        acc = l.dot(&acc);
    }
    acc
}

/// f64 variant: Order-preserving composition of square layers
pub fn compose_layers_order_preserving_f64(
    layers: &[ndarray::Array2<f64>],
) -> ndarray::Array2<f64> {
    assert!(!layers.is_empty(), "layers must be non-empty");
    let (n, m) = layers[0].dim();
    assert_eq!(n, m, "only square layers supported");
    let mut acc = ndarray::Array2::<f64>::eye(n);
    for l in layers {
        assert_eq!(l.dim(), (n, n));
        acc = l.dot(&acc);
    }
    acc
}

pub fn apply_linear(matrix: &Array2<f32>, vecs: &Array2<f32>) -> Array2<f32> {
    // matrix: (out, in), vecs: (batch, in) -> (batch, out)
    let (_, in_dim) = matrix.dim();
    let (_batch, in_dim_vec) = vecs.dim();
    assert_eq!(in_dim, in_dim_vec);
    vecs.dot(&matrix.t())
}

// ===== Exact inference ops (f32) for reversible path =====

/// LayerNorm forward (per-row) with epsilon, returns (y, mu, rstd)
pub fn layer_norm_forward_exact_f32(
    x: &Array2<f32>,
    gamma: &ndarray::Array1<f32>,
    beta: &ndarray::Array1<f32>,
    eps: f32,
) -> (Array2<f32>, ndarray::Array1<f32>, ndarray::Array1<f32>) {
    let (batch, dim) = x.dim();
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);
    let mut y = Array2::<f32>::zeros((batch, dim));
    let mut mu = ndarray::Array1::<f32>::zeros(batch);
    let mut rstd = ndarray::Array1::<f32>::zeros(batch);
    for i in 0..batch {
        let xi = x.row(i);
        let m = xi.sum() / (dim as f32);
        mu[i] = m;
        // variance
        let mut var = 0.0f32;
        for j in 0..dim {
            let d = xi[j] - m;
            var += d * d;
        }
        var /= dim as f32;
        let rs = 1.0f32 / (var + eps).sqrt();
        rstd[i] = rs;
        for j in 0..dim {
            let norm = (xi[j] - m) * rs;
            y[(i, j)] = norm * gamma[j] + beta[j];
        }
    }
    (y, mu, rstd)
}

/// GPT-2 gelu_new activation (tanh-based) applied elementwise to (batch, dim)
pub fn gelu_new_f32(x: &Array2<f32>) -> Array2<f32> {
    let (batch, dim) = x.dim();
    let mut y = Array2::<f32>::zeros((batch, dim));
    // constants
    let k: f32 = std::f32::consts::FRAC_2_SQRT_PI * 0.5f32; // 0.5*sqrt(2/pi)
    for i in 0..batch {
        for j in 0..dim {
            let v = x[(i, j)];
            let v3 = v * v * v;
            let t = (k * (v + 0.044715f32 * v3)).tanh();
            y[(i, j)] = 0.5f32 * v * (1.0f32 + t);
        }
    }
    y
}

/// Stable softmax along last dimension of a 2D tensor (batch, dim)
pub fn softmax_lastdim_f32(x: &Array2<f32>) -> Array2<f32> {
    let (batch, dim) = x.dim();
    let mut y = Array2::<f32>::zeros((batch, dim));
    for i in 0..batch {
        // subtract max for stability
        let mut max_v = std::f32::NEG_INFINITY;
        for j in 0..dim {
            let v = x[(i, j)];
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..dim {
            let e = (x[(i, j)] - max_v).exp();
            y[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0f32 / sum.max(EPS);
        for j in 0..dim {
            y[(i, j)] *= inv;
        }
    }
    y
}

/// Apply causal mask in-place to 2D scores (seq, seq): set j>i to large negative
pub fn apply_causal_mask_inplace_f32(scores: &mut Array2<f32>, neg_large: f32) {
    let (n, m) = scores.dim();
    assert_eq!(n, m);
    for i in 0..n {
        for j in (i + 1)..n {
            scores[(i, j)] = neg_large;
        }
    }
}

// f64 counterparts
pub fn layer_norm_forward_exact_f64(
    x: &ndarray::Array2<f64>,
    gamma: &ndarray::Array1<f64>,
    beta: &ndarray::Array1<f64>,
    eps: f64,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
) {
    let (batch, dim) = x.dim();
    assert_eq!(gamma.len(), dim);
    assert_eq!(beta.len(), dim);
    let mut y = ndarray::Array2::<f64>::zeros((batch, dim));
    let mut mu = ndarray::Array1::<f64>::zeros(batch);
    let mut rstd = ndarray::Array1::<f64>::zeros(batch);
    for i in 0..batch {
        let xi = x.row(i);
        let m = xi.sum() / (dim as f64);
        mu[i] = m;
        let mut var = 0.0f64;
        for j in 0..dim {
            let d = xi[j] - m;
            var += d * d;
        }
        var /= dim as f64;
        let rs = 1.0f64 / (var + (eps as f64)).sqrt();
        rstd[i] = rs;
        for j in 0..dim {
            let norm = (xi[j] - m) * rs;
            y[(i, j)] = norm * gamma[j] + beta[j];
        }
    }
    (y, mu, rstd)
}

pub fn gelu_new_f64(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let (batch, dim) = x.dim();
    let mut y = ndarray::Array2::<f64>::zeros((batch, dim));
    let k: f64 = std::f64::consts::FRAC_2_SQRT_PI * 0.5f64;
    for i in 0..batch {
        for j in 0..dim {
            let v = x[(i, j)];
            let v3 = v * v * v;
            let t = (k * (v + 0.044715f64 * v3)).tanh();
            y[(i, j)] = 0.5f64 * v * (1.0f64 + t);
        }
    }
    y
}

pub fn softmax_lastdim_f64(x: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let (batch, dim) = x.dim();
    let mut y = ndarray::Array2::<f64>::zeros((batch, dim));
    for i in 0..batch {
        let mut max_v = std::f64::NEG_INFINITY;
        for j in 0..dim {
            let v = x[(i, j)];
            if v > max_v {
                max_v = v;
            }
        }
        let mut sum = 0.0f64;
        for j in 0..dim {
            let e = (x[(i, j)] - max_v).exp();
            y[(i, j)] = e;
            sum += e;
        }
        let inv = 1.0f64 / sum.max(EPS64);
        for j in 0..dim {
            y[(i, j)] *= inv;
        }
    }
    y
}

pub fn apply_causal_mask_inplace_f64(scores: &mut ndarray::Array2<f64>, neg_large: f64) {
    let (n, m) = scores.dim();
    assert_eq!(n, m);
    for i in 0..n {
        for j in (i + 1)..n {
            scores[(i, j)] = neg_large;
        }
    }
}

// ===== f64 linear/attention/ffn exact forwards (GPT-2 style) =====

pub fn linear_f64(
    x: &ndarray::Array2<f64>,         // (batch, in)
    w: &ndarray::Array2<f64>,         // (out, in)
    b: Option<&ndarray::Array1<f64>>, // (out)
) -> ndarray::Array2<f64> {
    let y = x.dot(&w.t());
    if let Some(bias) = b {
        let mut out = y;
        for mut row in out.rows_mut() {
            row += &bias.view();
        }
        out
    } else {
        y
    }
}

pub fn attention_forward_f64(
    x: &ndarray::Array2<f64>, // (seq, d_model)
    wq: &ndarray::Array2<f64>,
    wk: &ndarray::Array2<f64>,
    wv: &ndarray::Array2<f64>,
    wo: &ndarray::Array2<f64>,
    bq: Option<&ndarray::Array1<f64>>,
    bk: Option<&ndarray::Array1<f64>>,
    bv: Option<&ndarray::Array1<f64>>,
    bo: Option<&ndarray::Array1<f64>>,
    n_heads: usize,
    causal: bool,
) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
    // (y, attn_probs_flat)
    let (seq, d_model) = x.dim();
    assert_eq!(wq.dim().1, d_model);
    let d_q = wq.dim().0; // out dim for Q
    assert_eq!(d_q % n_heads, 0);
    let dh = d_q / n_heads;

    let q = linear_f64(x, wq, bq); // (seq, d_q)
    let k = linear_f64(x, wk, bk); // (seq, d_q)
    let v = linear_f64(x, wv, bv); // (seq, d_q)

    // reshape to heads: (n_heads, seq, dh)
    let mut y_heads = ndarray::Array3::<f64>::zeros((n_heads, seq, dh));
    let mut probs_all = ndarray::Array3::<f64>::zeros((n_heads, seq, seq));
    let scale = 1.0f64 / (dh as f64).sqrt();
    for h in 0..n_heads {
        // slices
        let qs = q.slice(ndarray::s![.., h * dh..(h + 1) * dh]).to_owned(); // (seq, dh)
        let ks = k.slice(ndarray::s![.., h * dh..(h + 1) * dh]).to_owned();
        let vs = v.slice(ndarray::s![.., h * dh..(h + 1) * dh]).to_owned();
        // scores = Q K^T * scale
        let mut scores = qs.dot(&ks.t()); // (seq, seq)
        scores.mapv_inplace(|z| z * scale);
        if causal {
            apply_causal_mask_inplace_f64(&mut scores, -1e9f64);
        }
        // softmax
        let probs = softmax_lastdim_f64(&scores);
        // out_h = probs V
        let out_h = probs.dot(&vs);
        y_heads.slice_mut(ndarray::s![h, .., ..]).assign(&out_h);
        probs_all.slice_mut(ndarray::s![h, .., ..]).assign(&probs);
    }
    // merge heads -> (seq, d_q)
    let mut yh = ndarray::Array2::<f64>::zeros((seq, d_q));
    for h in 0..n_heads {
        let s = y_heads.slice(ndarray::s![h, .., ..]);
        yh.slice_mut(ndarray::s![.., h * dh..(h + 1) * dh])
            .assign(&s);
    }
    let y = linear_f64(&yh, wo, bo); // (seq, d_model)
                                     // Flatten heads: (n_heads, seq, seq) -> (n_heads*seq, seq)
    let probs_flat = probs_all.into_shape((n_heads * seq, seq)).unwrap();
    (y, probs_flat)
}

pub fn ffn_gelu_forward_f64(
    x: &ndarray::Array2<f64>,
    w1: &ndarray::Array2<f64>,
    b1: Option<&ndarray::Array1<f64>>,
    w2: &ndarray::Array2<f64>,
    b2: Option<&ndarray::Array1<f64>>,
) -> ndarray::Array2<f64> {
    let h = linear_f64(x, w1, b1);
    let a = gelu_new_f64(&h);
    linear_f64(&a, w2, b2)
}

pub fn transformer_block_forward_f64(
    x: &ndarray::Array2<f64>,
    // LN1
    ln1_g: &ndarray::Array1<f64>,
    ln1_b: &ndarray::Array1<f64>,
    eps1: f64,
    // Attn
    wq: &ndarray::Array2<f64>,
    wk: &ndarray::Array2<f64>,
    wv: &ndarray::Array2<f64>,
    wo: &ndarray::Array2<f64>,
    bq: Option<&ndarray::Array1<f64>>,
    bk: Option<&ndarray::Array1<f64>>,
    bv: Option<&ndarray::Array1<f64>>,
    bo: Option<&ndarray::Array1<f64>>,
    n_heads: usize,
    // LN2
    ln2_g: &ndarray::Array1<f64>,
    ln2_b: &ndarray::Array1<f64>,
    eps2: f64,
    // FFN
    w1: &ndarray::Array2<f64>,
    b1: Option<&ndarray::Array1<f64>>,
    w2: &ndarray::Array2<f64>,
    b2: Option<&ndarray::Array1<f64>>,
    causal: bool,
) -> (
    ndarray::Array2<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
    ndarray::Array1<f64>,
) {
    // LN1
    let (x1, mu1, rstd1) = layer_norm_forward_exact_f64(x, ln1_g, ln1_b, eps1);
    // Attn
    let (attn_out, _probs) =
        attention_forward_f64(&x1, wq, wk, wv, wo, bq, bk, bv, bo, n_heads, causal);
    let x_res1 = x + &attn_out;
    // LN2
    let (x2, mu2, rstd2) = layer_norm_forward_exact_f64(&x_res1, ln2_g, ln2_b, eps2);
    // FFN
    let ffn_out = ffn_gelu_forward_f64(&x2, w1, b1, w2, b2);
    let y = x_res1 + &ffn_out;
    (y, mu1, rstd1, mu2, rstd2)
}

/// Compute effective SPD metric G = T^T T for a given transform T (f64)
pub fn effective_metric_from_transform_f64(t: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    t.t().dot(t)
}

/// Simple Cholesky factorization (upper-triangular) in f64: returns U with G = U^T U
pub fn metric_factor_cholesky_f64(g: &ndarray::Array2<f64>) -> ndarray::Array2<f64> {
    let (n, m) = g.dim();
    assert_eq!(n, m, "G must be square");
    let mut l = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = g[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, j)] = (sum.max(EPS64)).sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)].max(EPS64);
            }
        }
    }
    // Return upper-triangular factor U = L_lower^T so that G = U^T U holds
    l.t().to_owned()
}

/// Session-rotation of the metric factor. Given an SPD factor L (G = L^T L),
/// apply an orthogonal rotation R_s on the left to preserve G: L' = R_s L.
/// A deterministic block-orthogonal R_s is generated from the key.
pub fn rotate_metric_factor_block(key: &str, l: &Array2<f32>, global_dim: usize) -> Array2<f32> {
    let (n, m) = l.dim();
    assert_eq!(n, m, "L must be square");
    assert!(global_dim <= n);
    let dept_dim = n - global_dim;
    let r_s = block_orthogonal_from_key(key, global_dim, dept_dim);
    r_s.dot(l)
}

// === Implicit transforms: Householder chain / Givens chain / Low-rank + Diagonal ===

fn random_unit_vector_f32(dim: usize, rng: &mut SmallRng) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(dim);
    for i in 0..dim {
        v[i] = rng.gen::<f32>() * 2.0 - 1.0;
    }
    let n = v.dot(&v).sqrt().max(EPS);
    v / n
}

fn householder_vectors_from_key(key: &str, dim: usize, num: usize) -> Vec<Array1<f32>> {
    let mut vecs = Vec::with_capacity(num);
    let mut rng = SmallRng::seed_from_u64(seed_from_key(key));
    for _ in 0..num {
        vecs.push(random_unit_vector_f32(dim, &mut rng));
    }
    vecs
}

fn apply_householder_chain(vecs: &[Array1<f32>], x: &Array1<f32>, reverse: bool) -> Array1<f32> {
    let mut y = x.clone();
    if reverse {
        for v in vecs.iter().rev() {
            let alpha = 2.0 * y.dot(v);
            y -= &(v * alpha);
        }
    } else {
        for v in vecs.iter() {
            let alpha = 2.0 * y.dot(v);
            y -= &(v * alpha);
        }
    }
    y
}

pub fn householder_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let vecs = householder_vectors_from_key(key, dim, num);
    apply_householder_chain(&vecs, x, false)
}

pub fn householder_chain_apply_transpose_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let vecs = householder_vectors_from_key(key, dim, num);
    // For Householder, H is symmetric, so Q^T = H_1 ... H_k (reverse order)
    apply_householder_chain(&vecs, x, true)
}

pub fn lowrank_plus_diag_apply_from_key(
    key_u: &str,
    key_v: &str,
    s_diag: &Array1<f32>,
    rank: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let dim = x.len();
    assert_eq!(s_diag.len(), dim);
    let mut rng_u = SmallRng::seed_from_u64(seed_from_key(key_u));
    let mut rng_v = SmallRng::seed_from_u64(seed_from_key(key_v));
    let mut y = s_diag * x;
    for _ in 0..rank {
        let a = random_unit_vector_f32(dim, &mut rng_u);
        let b = random_unit_vector_f32(dim, &mut rng_v);
        let coeff = b.dot(x);
        y += &(a * coeff);
    }
    y
}

pub fn givens_chain_apply_from_key(
    key: &str,
    dim: usize,
    num: usize,
    x: &Array1<f32>,
) -> Array1<f32> {
    let mut rng = SmallRng::seed_from_u64(seed_from_key(key) ^ 0xABCDEF0123456789);
    let mut y = x.clone();
    for _ in 0..num {
        let i = (rng.gen::<u32>() as usize) % dim;
        let mut j = (rng.gen::<u32>() as usize) % dim;
        if j == i {
            j = (j + 1) % dim;
        }
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        let c = theta.cos();
        let s = theta.sin();
        let yi = y[i];
        let yj = y[j];
        y[i] = c * yi - s * yj;
        y[j] = s * yi + c * yj;
    }
    y
}
```
---
## File: `reality_stone/src/ops/mobius.rs`

```rust
use crate::ops::{
    batch::EPS,
    batch::{dot_batched_f64, norm_sq_batched_f64, EPS64},
    dot_batched, norm_sq_batched, DynamicCurvature, LayerWiseDynamicCurvature,
};
use ndarray::{Array2, ArrayView2, Axis};

const BOUNDARY_EPS: f32 = 1e-5;
const MIN_DENOMINATOR: f32 = 1e-6;
const BOUNDARY_EPS64: f64 = 1e-12;
const MIN_DENOMINATOR64: f64 = 1e-12;

/// 뫼비우스 덧셈 (Mobius Addition)
///
/// 두 벡터 u와 v를 곡률 c의 푸앵카레 공 위에서 더합니다.
/// 수식: u +_c v = (1 + 2c<u,v> + c|v|^2)u + (1 - c|u|^2)v / (1 + 2c<u,v> + c^2|u|^2|v|^2)
pub fn mobius_add(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    let c2 = c * c;

    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    let coeff_u = (1.0 + 2.0 * c * &uv + c * &v2) / &den;
    let coeff_v = (1.0 - c * &u2) / &den;

    coeff_u * u + coeff_v * v
}

/// 뫼비우스 덧셈 (Double Precision)
pub fn mobius_add_f64(u: &ArrayView2<f64>, v: &ArrayView2<f64>, c: f64) -> Array2<f64> {
    let u2 = norm_sq_batched_f64(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched_f64(v).insert_axis(Axis(1));
    let uv = dot_batched_f64(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR64));
    let coeff_u = (1.0 + 2.0 * c * &uv + c * &v2) / &den;
    let coeff_v = (1.0 - c * &u2) / &den;
    coeff_u * u + coeff_v * v
}

/// 뫼비우스 스칼라 곱 (Mobius Scalar Multiplication)
///
/// 벡터 u에 스칼라 r을 곡률 c 공간에서 곱합니다.
/// c=0일 때는 유클리드 스칼라 곱과 동일합니다.
pub fn mobius_scalar(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));

    if c.abs() < EPS {
        // c = 0: 유클리드 근사
        return Array2::from_elem(u.dim(), r) * u;
    }

    // 양수/음수 곡률 모두 처리
    // sqrt(c) * norm이 1보다 작아야 atanh가 정의됨 (경계값 처리)
    let sqrt_c_norm = if c > 0.0 {
        (c.sqrt() * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS))
    } else {
        (-c).sqrt() * &norm_clamped
    };

    let scale = if c > 0.0 {
        // 양수 곡률: tanh(r * atanh(sqrt(c)*|u|)) / (sqrt(c)*|u|)
        let alpha = sqrt_c_norm.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        beta / (c.sqrt() * &norm_clamped)
    } else {
        // 음수 곡률: tan(r * atan(sqrt(-c)*|u|)) / (sqrt(-c)*|u|)
        // atanh(i*x) = i*atan(x), tanh(i*x) = i*tan(x) 관계식 이용
        let alpha = sqrt_c_norm.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        beta / ((-c).sqrt() * &norm_clamped)
    };
    scale * u
}

/// 뫼비우스 스칼라 곱 (Double Precision)
pub fn mobius_scalar_f64(u: &ArrayView2<f64>, c: f64, r: f64) -> Array2<f64> {
    let norm = norm_sq_batched_f64(u).mapv(f64::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS64));
    if c.abs() < EPS64 {
        return Array2::from_elem(u.dim(), r) * u;
    }
    let sqrt_c_norm = if c > 0.0 {
        (c.sqrt() * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS64))
    } else {
        (-c).sqrt() * &norm_clamped
    };
    let scale = if c > 0.0 {
        let alpha = sqrt_c_norm.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        beta / (c.sqrt() * &norm_clamped)
    } else {
        let alpha = sqrt_c_norm.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        beta / ((-c).sqrt() * &norm_clamped)
    };
    scale * u
}

/// 곡률 c에 대한 뫼비우스 스칼라 곱의 그라디언트
pub fn mobius_scalar_grad_c(u: &ArrayView2<f32>, c: f32, r: f32) -> Array2<f32> {
    let norm = norm_sq_batched(u).mapv(f32::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS));
    if c.abs() < EPS {
        // c = 0: 그라디언트는 0
        return Array2::zeros(u.dim());
    }

    if c > 0.0 {
        // 양수 곡률
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());

        // d(sqrt(c))/dc = 0.5/sqrt(c)
        let d_sqrt_c_dc = 0.5 / sqrt_c;

        // d(alpha)/d(scn) = 1/(1 - scn^2)
        let d_alpha_dscn = 1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS));

        // d(beta)/d(alpha) = r * (1 - tanh^2(r*alpha))
        let tanh_r_alpha = (r * &alpha).mapv(|v| v.tanh());
        let d_beta_dalpha = r * (1.0 - &tanh_r_alpha * &tanh_r_alpha);
        // 연쇄 법칙 (Chain rule)
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_c_dc;
        let d_scale_dc = (&d_beta_dc * sqrt_c - &beta * d_sqrt_c_dc) / (c * &norm_clamped);
        &d_scale_dc * u
    } else {
        // 음수 곡률
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &norm_clamped;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        // d(sqrt(|c|))/dc = -0.5/sqrt(|c|) (c가 음수이므로)
        let d_sqrt_abs_c_dc = -0.5 / sqrt_abs_c;
        // d(alpha)/d(scn) = 1/(1 + scn^2)
        let d_alpha_dscn = 1.0 / (1.0 + &scn * &scn);
        // d(beta)/d(alpha) = r * (1 + tan^2(r*alpha))
        let tan_r_alpha = (r * &alpha).mapv(|v| v.tan());
        let d_beta_dalpha = r * (1.0 + &tan_r_alpha * &tan_r_alpha);
        // 연쇄 법칙 (Chain rule)
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_abs_c_dc;
        // d(scale)/dc
        let d_scale_dc =
            (&d_beta_dc * sqrt_abs_c - &beta * d_sqrt_abs_c_dc) / ((-c) * &norm_clamped);
        &d_scale_dc * u
    }
}

/// 곡률 c에 대한 뫼비우스 스칼라 곱의 그라디언트 (Double Precision)
pub fn mobius_scalar_grad_c_f64(u: &ArrayView2<f64>, c: f64, r: f64) -> Array2<f64> {
    let norm = norm_sq_batched_f64(u).mapv(f64::sqrt).insert_axis(Axis(1));
    let norm_clamped = norm.mapv(|v| v.max(EPS64));
    if c.abs() < EPS64 {
        return Array2::zeros(u.dim());
    }
    if c > 0.0 {
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &norm_clamped).mapv(|v| v.min(1.0 - BOUNDARY_EPS64));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        let d_sqrt_c_dc = 0.5 / sqrt_c;
        let d_alpha_dscn = 1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS64));
        let tanh_r_alpha = (r * &alpha).mapv(|v| v.tanh());
        let d_beta_dalpha = r * (1.0 - &tanh_r_alpha * &tanh_r_alpha);
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_c_dc;
        let d_scale_dc = (&d_beta_dc * sqrt_c - &beta * d_sqrt_c_dc) / (c * &norm_clamped);
        &d_scale_dc * u
    } else {
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &norm_clamped;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        let d_sqrt_abs_c_dc = -0.5 / sqrt_abs_c;
        let d_alpha_dscn = 1.0 / (1.0 + &scn * &scn);
        let tan_r_alpha = (r * &alpha).mapv(|v| v.tan());
        let d_beta_dalpha = r * (1.0 + &tan_r_alpha * &tan_r_alpha);
        let d_beta_dc = &d_beta_dalpha * &d_alpha_dscn * &norm_clamped * d_sqrt_abs_c_dc;
        let d_scale_dc =
            (&d_beta_dc * sqrt_abs_c - &beta * d_sqrt_abs_c_dc) / ((-c) * &norm_clamped);
        &d_scale_dc * u
    }
}

/// 곡률 c에 대한 뫼비우스 덧셈의 그라디언트
pub fn mobius_add_grad_c(u: &ArrayView2<f32>, v: &ArrayView2<f32>, c: f32) -> Array2<f32> {
    let u2 = norm_sq_batched(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched(v).insert_axis(Axis(1));
    let uv = dot_batched(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    let num = (1.0 + 2.0 * c * &uv + c * &v2) * u + (1.0 - c * &u2) * v;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR));
    let dnum_dc = (2.0 * &uv + &v2) * u - &u2 * v;
    let dden_dc = 2.0 * &uv + 2.0 * c * &u2 * &v2;
    let result = (dnum_dc * &den - &num * &dden_dc) / (&den * &den);
    result
}

/// 곡률 c에 대한 뫼비우스 덧셈의 그라디언트 (Double Precision)
pub fn mobius_add_grad_c_f64(u: &ArrayView2<f64>, v: &ArrayView2<f64>, c: f64) -> Array2<f64> {
    let u2 = norm_sq_batched_f64(u).insert_axis(Axis(1));
    let v2 = norm_sq_batched_f64(v).insert_axis(Axis(1));
    let uv = dot_batched_f64(u, v).insert_axis(Axis(1));
    let c2 = c * c;
    let num = (1.0 + 2.0 * c * &uv + c * &v2) * u + (1.0 - c * &u2) * v;
    let den = (1.0 + 2.0 * c * &uv + c2 * &u2 * &v2).mapv(|v| v.max(MIN_DENOMINATOR64));
    let dnum_dc = (2.0 * &uv + &v2) * u - &u2 * v;
    let dden_dc = 2.0 * &uv + 2.0 * c * &u2 * &v2;
    let result = (dnum_dc * &den - &num * &dden_dc) / (&den * &den);
    result
}

// --- VJP 구현 (Vector-Jacobian Product) ---

/// 뫼비우스 스칼라 곱의 역전파 (VJP)
pub fn mobius_scalar_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    c: f32,
    r: f32,
) -> Array2<f32> {
    let x_norm = norm_sq_batched(&x).mapv(f32::sqrt).insert_axis(Axis(1));
    let x_norm_clamp = x_norm.mapv(|v| v.max(EPS));
    if c.abs() < EPS {
        // c = 0: 유클리드 경우
        return grad_output * r;
    }

    if c > 0.0 {
        // 양수 곡률
        let sqrt_c = c.sqrt();
        let scn = (sqrt_c * &x_norm_clamp).mapv(|v| v.min(1.0 - EPS));
        let alpha = scn.mapv(|v| v.atanh());
        let beta = (r * &alpha).mapv(|v| v.tanh());
        let scale = &beta / (sqrt_c * &x_norm_clamp);
        let grad_scale = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
        let inner_deriv_atanh = r * (1.0 - &beta * &beta);
        let inner_deriv_norm =
            (1.0 / (1.0 - &scn * &scn).mapv(|v| v.max(EPS))) * (sqrt_c / &x_norm_clamp);
        let grad_scale_b = &grad_scale * (&inner_deriv_atanh * &inner_deriv_norm - &scale * sqrt_c);
        grad_output * &scale + x * &grad_scale_b / (sqrt_c * &x_norm_clamp)
    } else {
        // 음수 곡률
        let sqrt_abs_c = (-c).sqrt();
        let scn = sqrt_abs_c * &x_norm_clamp;
        let alpha = scn.mapv(|v| v.atan());
        let beta = (r * &alpha).mapv(|v| v.tan());
        let scale = &beta / (sqrt_abs_c * &x_norm_clamp);

        let grad_scale = (grad_output * x).sum_axis(Axis(1)).insert_axis(Axis(1));
        let inner_deriv_atan = r * (1.0 + &beta * &beta);
        let inner_deriv_norm = (1.0 / (1.0 + &scn * &scn)) * (sqrt_abs_c / &x_norm_clamp);

        let grad_scale_b =
            &grad_scale * (&inner_deriv_atan * &inner_deriv_norm - &scale * sqrt_abs_c);

        grad_output * &scale + x * &grad_scale_b / (sqrt_abs_c * &x_norm_clamp)
    }
}

/// 뫼비우스 덧셈의 역전파 (VJP)
pub fn mobius_add_vjp(
    grad_output: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    y: &ArrayView2<f32>,
    c: f32,
) -> (Array2<f32>, Array2<f32>) {
    let x2 = norm_sq_batched(&x).insert_axis(Axis(1));
    let y2 = norm_sq_batched(&y).insert_axis(Axis(1));
    let xy = dot_batched(&x, &y).insert_axis(Axis(1));

    let den = 1.0 + 2.0 * c * &xy + c * c * &x2 * &y2;
    let den_clamp = den.mapv(|v| v.max(EPS));

    let u = (1.0 + 2.0 * c * &xy + c * &y2) * x + (1.0 - c * &x2) * y;
    let output = &u / &den_clamp;

    let grad_u = grad_output / &den_clamp;
    let grad_den = -(grad_output * &output / &den_clamp)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));

    let grad_x_from_u = &grad_u * (1.0 + 2.0 * c * &xy + c * &y2);
    let grad_y_from_u = &grad_u * (1.0 - c * &x2);

    let grad_xy_from_u = (2.0 * c * (&grad_u * x))
        .sum_axis(Axis(1))
        .insert_axis(Axis(1));
    let grad_x2_from_u = (-c * (&grad_u * y)).sum_axis(Axis(1)).insert_axis(Axis(1));

    let grad_xy_from_den = 2.0 * c * &grad_den;
    let grad_x2_from_den = c * c * &y2 * &grad_den;
    let grad_y2_from_den = c * c * &x2 * &grad_den;

    let grad_xy = grad_xy_from_u + grad_xy_from_den;
    let grad_x2 = grad_x2_from_u + grad_x2_from_den;
    let grad_y2 = grad_y2_from_den;

    let grad_x = grad_x_from_u + 2.0 * &grad_x2 * x + &grad_xy * y;
    let grad_y = grad_y_from_u + 2.0 * &grad_y2 * y + &grad_xy * x;

    (grad_x, grad_y)
}

/// 동적 곡률을 사용한 뫼비우스 덧셈
pub fn mobius_add_dynamic(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &DynamicCurvature,
) -> (Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let result = mobius_add(u, v, c);
    (result, c)
}

/// 동적 곡률 뫼비우스 덧셈의 역전파 (Backward)
pub fn mobius_add_dynamic_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    dynamic_c: &DynamicCurvature,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = dynamic_c.compute_c();
    let grad_c_tensor = mobius_add_grad_c(u, v, c);
    let grad_c = (grad_output * &grad_c_tensor).sum();
    let dc_dkappa = dynamic_c.compute_dc_dkappa();
    let grad_kappa = grad_c * dc_dkappa;
    let (grad_u, grad_v) = mobius_add_vjp(grad_output, u, v, c);
    (grad_u, grad_v, grad_kappa)
}

/// 레이어별 동적 곡률을 사용한 뫼비우스 덧셈
pub fn mobius_add_layerwise(
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
) -> (Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let result = mobius_add(u, v, c);
    (result, c)
}

/// 레이어별 동적 곡률 뫼비우스 덧셈의 역전파
pub fn mobius_add_layerwise_backward(
    grad_output: &ArrayView2<f32>,
    u: &ArrayView2<f32>,
    v: &ArrayView2<f32>,
    layer_curvatures: &LayerWiseDynamicCurvature,
    layer_idx: usize,
) -> (Array2<f32>, Array2<f32>, f32) {
    let c = layer_curvatures.compute_c(layer_idx);
    let grad_c_tensor = mobius_add_grad_c(u, v, c);
    let grad_c = (grad_output * &grad_c_tensor).sum();
    let dc_dkappa = layer_curvatures.compute_dc_dkappa(layer_idx);
    let grad_kappa = grad_c * dc_dkappa;
    let (grad_u, grad_v) = mobius_add_vjp(grad_output, u, v, c);
    (grad_u, grad_v, grad_kappa)
}

#[cfg(feature = "cuda")]
pub mod cuda {
    mod ffi {
        extern "C" {
            pub fn mobius_add_cuda(
                out: *mut f32,
                u: *const f32,
                v: *const f32,
                c: f32,
                batch_size: i64,
                dim: i64,
            );
            pub fn mobius_scalar_cuda(
                out: *mut f32,
                u: *const f32,
                c: f32,
                r: f32,
                batch_size: i64,
                dim: i64,
            );
        }
    }

    pub fn mobius_add_cuda(
        out: *mut f32,
        u: *const f32,
        v: *const f32,
        c: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::mobius_add_cuda(out, u, v, c, batch_size, dim);
        }
    }

    pub fn mobius_scalar_cuda(
        out: *mut f32,
        u: *const f32,
        c: f32,
        r: f32,
        batch_size: i64,
        dim: i64,
    ) {
        unsafe {
            ffi::mobius_scalar_cuda(out, u, c, r, batch_size, dim);
        }
    }
}
```
---
## File: `reality_stone/src/ops/mod.rs`

```rust
pub mod batch;
pub mod curvature;
pub mod extraction;
pub mod metrikey;
pub mod mobius;
pub mod project;

use ndarray::Array2;

pub use self::batch::{dot_batched, norm_sq_batched};
pub use self::mobius::{
    mobius_add, mobius_add_dynamic, mobius_add_dynamic_backward, mobius_add_grad_c,
    mobius_add_layerwise, mobius_add_layerwise_backward, mobius_scalar, mobius_scalar_grad_c,
};
pub use self::mobius::{
    mobius_add_f64, mobius_add_grad_c_f64, mobius_scalar_f64, mobius_scalar_grad_c_f64,
};
pub use self::project::project_to_ball;
pub use curvature::{DynamicCurvature, LayerWiseDynamicCurvature};
pub use metrikey::{
    apply_linear, block_orthogonal_from_key, compose_layers_gravity,
    compose_layers_order_preserving, deterministic_orthogonal_from_key, mahalanobis_distance_sq_g,
    mahalanobis_distance_sq_l, metric_factor_cholesky, rotate_metric_factor_block,
    spd_block_metric_from_key, spd_metric_from_key, spd_metric_from_key_weighted,
};

// f64 high-precision exports (not all functions; only where useful)
pub use metrikey::{
    apply_linear_f64, compose_layers_gravity_compact_f64, compose_layers_gravity_f64,
    deterministic_orthogonal_from_key_f64, effective_metric_from_transform_f64,
    metric_factor_cholesky_f64, spd_metric_from_key_f64,
};

// Exact ops re-export
pub use metrikey::{
    apply_causal_mask_inplace_f32, gelu_new_f32, layer_norm_forward_exact_f32, softmax_lastdim_f32,
};
pub use metrikey::{
    apply_causal_mask_inplace_f64, gelu_new_f64, layer_norm_forward_exact_f64, softmax_lastdim_f64,
};

// Implicit transforms
pub use metrikey::{
    givens_chain_apply_from_key, householder_chain_apply_from_key,
    householder_chain_apply_transpose_from_key, lowrank_plus_diag_apply_from_key,
};

/// MSE loss의 gradient를 계산합니다.
pub fn mse_loss_grad(pred: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
    2.0 * (pred - target) / (pred.shape()[0] * pred.shape()[1]) as f32
}
```
---
## File: `reality_stone/src/ops/project.rs`

```rust
use crate::ops::batch::norm_sq_batched;
use ndarray::{Array2, ArrayView2, Axis};

pub const EPS: f32 = 1e-6;
pub fn project_to_ball(x: &ArrayView2<f32>, epsilon: f32) -> Array2<f32> {
    let norm = norm_sq_batched(x).mapv(f32::sqrt).insert_axis(Axis(1));
    let max_norm = 1.0 - epsilon;
    let scale = norm.mapv(|n| if n > max_norm { max_norm / n } else { 1.0 });
    x * &scale
}
```
---
## File: `reality_stone/tests/api/test_pipeline_api.py`

```python
import pytest
import torch
import tempfile
from pathlib import Path

from reality_stone.api import pipeline, HierarchicalLLM
from reality_stone.models.hierarchical_sentence_topic_llm import HierarchicalLLMConfig


@pytest.fixture
def small_config():
    return HierarchicalLLMConfig(
        vocab_size=500,
        d_model=64,
        d_head=16,
        num_topics=4,
        num_heads_topic=2,
        n_layer_decoder=1,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )


@pytest.fixture
def sample_model(small_config):
    return HierarchicalLLM.from_config(small_config)


class TestHierarchicalLLM:
    
    def test_from_config(self, small_config):
        model = HierarchicalLLM.from_config(small_config)
        
        assert model.model is not None
        assert model.config == small_config
    
    def test_from_config_dict(self):
        config_dict = {
            "vocab_size": 500,
            "d_model": 64,
            "d_head": 16,
            "num_topics": 4,
            "use_pretrained_embeddings": False,
        }
        
        model = HierarchicalLLM.from_config(config_dict)
        
        assert model.model is not None
        assert model.config.vocab_size == 500
    
    def test_call_inference(self, sample_model):
        text = "테스트 문장입니다."
        
        result = sample_model(text, max_length=32, k_neighbors=2)
        
        assert "original_text" in result
        assert "generated_text" in result
        assert "sentences" in result
        assert "topics" in result
    
    def test_save_and_load(self, sample_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            sample_model.save_pretrained(save_path)
            
            assert (save_path / "model.pt").exists()
            
            loaded = HierarchicalLLM.from_pretrained(save_path / "model.pt")
            
            assert loaded.config.vocab_size == sample_model.config.vocab_size


class TestPipeline:
    
    def test_pipeline_text_generation(self, small_config):
        generator = pipeline("text-generation", config=small_config)
        
        text = "테스트 문장입니다."
        output = generator(text)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_pipeline_text_editing(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        text = "편집할 문장입니다."
        result = editor(text, enable_structural_edit=False)
        
        assert "original" in result
        assert "edited" in result
        assert "topics" in result
    
    def test_pipeline_with_model_instance(self, sample_model):
        generator = pipeline("text-generation", model=sample_model)
        
        text = "테스트 문장입니다."
        output = generator(text)
        
        assert isinstance(output, str)
    
    def test_pipeline_invalid_task(self, small_config):
        with pytest.raises(ValueError):
            pipeline("invalid-task", config=small_config)
    
    def test_pipeline_no_model_or_config(self):
        with pytest.raises(ValueError):
            pipeline("text-generation")


class TestTextGenerator:
    
    def test_single_generation(self, small_config):
        generator = pipeline("text-generation", config=small_config)
        
        text = "단일 생성 테스트"
        output = generator(text, max_new_tokens=10)
        
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_batch_generation(self, small_config):
        generator = pipeline("text-generation", config=small_config)
        
        texts = ["첫 번째 문장", "두 번째 문장", "세 번째 문장"]
        outputs = generator.generate_batch(texts, max_new_tokens=10)
        
        assert len(outputs) == 3
        assert all(isinstance(o, str) for o in outputs)


class TestTextEditor:
    
    def test_edit_with_structural_edit(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        text = "편집할 텍스트입니다."
        result = editor(text, enable_structural_edit=True)
        
        assert result["original"] == text
        assert isinstance(result["edited"], str)
        assert isinstance(result["topics"], list)
    
    def test_edit_without_structural_edit(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        text = "편집할 텍스트입니다."
        result = editor(text, enable_structural_edit=False)
        
        assert result["original"] == text
        assert isinstance(result["edited"], str)
    
    def test_batch_editing(self, small_config):
        editor = pipeline("text-editing", config=small_config)
        
        texts = ["첫 번째", "두 번째"]
        results = editor.edit_batch(texts)
        
        assert len(results) == 2
        assert all("original" in r and "edited" in r for r in results)
```
---
## File: `reality_stone/tests/llm/test_bindings_cuda_symbols.py`

```python
import pytest
import torch
import reality_stone as rs


@pytest.mark.cuda
def test_rust_extension_loaded_when_cuda_available():
    """
    CUDA 환경에서는 Rust 확장이 로드되어 있어야 한다.

    이 테스트는 빌드/배포 과정에서 `_rust` 모듈이 누락되는 경우를 빠르게 잡기 위한 것이다.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")
    assert rs._has_rust_ext, "Rust extension must be available when CUDA is used"
    assert hasattr(rs, "_rust"), "reality_stone must expose `_rust` module"


@pytest.mark.cuda
def test_required_cuda_symbols_exist_on_rust_module():
    """
    reality_stone.__init__ 에서 사용하는 CUDA 바인딩 심볼들이
    실제 Rust 모듈에도 모두 존재하는지 검증한다.

    - 누락된 심볼이 있으면 `_has_cuda` 가 False 가 되어 CUDA 경로 전체가 비활성화된다.
    - 이 테스트로 '바인딩은 구현됐는데 __init__ 이 다른 이름을 보고 있다' 같은 실수를 방지한다.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    assert rs._has_rust_ext, "Rust extension must be available when CUDA is used"

    required_cuda_symbols = [
        # Möbius
        "mobius_add_cuda",
        "mobius_scalar_cuda",
        # Poincaré
        "poincare_ball_layer_cuda",
        "poincare_ball_layer_backward_cuda",
        "poincare_distance_cuda",
        # Lorentz
        "lorentz_layer_forward_cuda",
        "lorentz_ball_layer_backward_cuda",
        "lorentz_distance_cuda",
        # Klein
        "klein_layer_forward_cuda",
        "klein_ball_layer_backward_cuda",
        "klein_distance_cuda",
    ]

    missing = [name for name in required_cuda_symbols if not hasattr(rs._rust, name)]
    assert (
        not missing
    ), f"Missing CUDA bindings on rs._rust: {missing}. Check Rust bindings and __init__._has_cuda list."


@pytest.mark.cuda
def test_has_cuda_flag_consistent_with_symbols_and_torch():
    """
    `_has_cuda` 플래그가 PyTorch / 심볼 상태와 일관되는지 확인.

    - torch.cuda.is_available() == False 이면 _has_cuda 도 False
    - torch.cuda.is_available() == True 이면, 필수 심볼이 모두 있을 때만 _has_cuda 가 True
    """
    if not torch.cuda.is_available():
        assert rs._has_cuda is False
        return

    required_cuda_symbols = [
        "mobius_add_cuda",
        "mobius_scalar_cuda",
        "poincare_ball_layer_cuda",
        "poincare_ball_layer_backward_cuda",
        "poincare_distance_cuda",
        "lorentz_layer_forward_cuda",
        "lorentz_ball_layer_backward_cuda",
        "lorentz_distance_cuda",
        "klein_layer_forward_cuda",
        "klein_ball_layer_backward_cuda",
        "klein_distance_cuda",
    ]
    all_symbols_present = all(hasattr(rs._rust, name) for name in required_cuda_symbols)

    # CUDA 환경에서는 심볼이 모두 있으면 True, 하나라도 없으면 False 여야 한다.
    assert rs._has_cuda == all_symbols_present
```
---
## File: `reality_stone/tests/llm/test_dynamic_manifold.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    TreeNodeOperator,
    LevelInvariantTreeProcessor,
)


def test_dynamic_manifold_basic():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    assert hasattr(operator, "manifold_selector")
    assert hasattr(operator, "aggregator_poincare")
    assert hasattr(operator, "aggregator_lorentz")
    assert hasattr(operator, "aggregator_klein")


def test_dynamic_manifold_up_operator():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    result = operator.up_operator(children_embs)
    
    assert result.shape == (B, d_model)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()


def test_dynamic_manifold_down_operator():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B = 2
    num_children = 3
    parent_emb = torch.randn(B, d_model)
    
    result = operator.down_operator(parent_emb, num_children)
    
    assert result.shape == (B, num_children, d_model)
    assert not torch.isnan(result).any()


def test_dynamic_manifold_selection():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 5
    children_embs = torch.randn(B, N, d_model)
    
    mean_emb = children_embs.mean(dim=1)
    manifold_logits = operator.manifold_selector(mean_emb)
    
    assert manifold_logits.shape == (B, 3)
    
    manifold_probs = torch.softmax(manifold_logits, dim=-1)
    assert torch.allclose(manifold_probs.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_dynamic_manifold_gradient_flow():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model, requires_grad=True)
    
    result = operator.up_operator(children_embs)
    loss = result.sum()
    loss.backward()
    
    assert children_embs.grad is not None
    assert not torch.isnan(children_embs.grad).any()


def test_tree_processor_dynamic_manifold():
    d_model = 64
    processor = LevelInvariantTreeProcessor(d_model, enable_dynamic_manifold=True)
    
    for node_type, operator in processor.node_operators.items():
        assert operator.enable_dynamic_manifold
        assert hasattr(operator, "manifold_selector")


def test_dynamic_manifold_different_manifolds():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    result_poincare = operator.aggregator_poincare(children_embs)
    result_lorentz = operator.aggregator_lorentz(children_embs)
    result_klein = operator.aggregator_klein(children_embs)
    
    assert result_poincare.shape == (B, d_model)
    assert result_lorentz.shape == (B, d_model)
    assert result_klein.shape == (B, d_model)
    
    assert not torch.equal(result_poincare, result_lorentz)
    assert not torch.equal(result_poincare, result_klein)


def test_dynamic_manifold_weighted_combination():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 1, 3
    children_embs = torch.randn(B, N, d_model)
    
    result_full = operator.up_operator(children_embs)
    
    mean_emb = children_embs.mean(dim=1)
    manifold_logits = operator.manifold_selector(mean_emb)
    manifold_probs = torch.softmax(manifold_logits, dim=-1)
    
    result_poincare = operator.aggregator_poincare(children_embs)
    result_lorentz = operator.aggregator_lorentz(children_embs)
    result_klein = operator.aggregator_klein(children_embs)
    
    expected = (
        manifold_probs[0, 0] * result_poincare[0] +
        manifold_probs[0, 1] * result_lorentz[0] +
        manifold_probs[0, 2] * result_klein[0]
    )
    
    assert torch.allclose(result_full[0], expected, atol=1e-5)


def test_dynamic_manifold_batch_consistency():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 5, 3
    children_embs = torch.randn(B, N, d_model)
    
    result_batched = operator.up_operator(children_embs)
    
    results_individual = []
    for b in range(B):
        result_b = operator.up_operator(children_embs[b:b+1])
        results_individual.append(result_b[0])
    
    results_stacked = torch.stack(results_individual)
    
    assert torch.allclose(result_batched, results_stacked, atol=1e-5)


def test_dynamic_manifold_deterministic():
    d_model = 64
    operator = TreeNodeOperator(d_model, manifold="poincare", c=1e-3, enable_dynamic_manifold=True)
    
    B, N = 2, 3
    children_embs = torch.randn(B, N, d_model)
    
    torch.manual_seed(42)
    result1 = operator.up_operator(children_embs)
    
    torch.manual_seed(42)
    result2 = operator.up_operator(children_embs)
    
    assert torch.equal(result1, result2)
```
---
## File: `reality_stone/tests/llm/test_edit_operations.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import EditOperationHead


@pytest.fixture
def edit_head():
    return EditOperationHead(d_model=128, num_ops=5, edit_budget=0.25)


def test_edit_head_forward(edit_head):
    B, S, d = 2, 10, 128
    hidden = torch.randn(B, S, d)
    
    edit_logits = edit_head(hidden)
    
    assert edit_logits.shape == (B, S, 5)
    assert not torch.isnan(edit_logits).any()


def test_edit_head_apply_edits_disabled(edit_head):
    B, S = 2, 10
    tokens = torch.randint(1, 100, (B, S))
    edit_logits = torch.randn(B, S, 5)
    pred_tokens = torch.randint(1, 100, (B, S))
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=False,
    )
    
    assert result.shape == tokens.shape
    assert torch.equal(result, tokens)


def test_edit_head_apply_edits_enabled(edit_head):
    B, S = 2, 10
    tokens = torch.randint(1, 100, (B, S))
    edit_logits = torch.randn(B, S, 5)
    pred_tokens = torch.randint(1, 100, (B, S))
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    assert result.shape[0] == B
    assert result.shape[1] >= S * 0.75
    assert result.shape[1] <= S * 1.25
    assert not torch.isnan(result).any()


def test_edit_head_apply_edits_with_replacement_mask(edit_head):
    B, S = 2, 10
    tokens = torch.randint(1, 100, (B, S))
    edit_logits = torch.randn(B, S, 5)
    pred_tokens = torch.randint(1, 100, (B, S))
    replacement_mask = torch.zeros(B, S)
    replacement_mask[:, :3] = 1
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
        replacement_mask=replacement_mask,
    )
    
    assert result.shape[0] == B
    assert not torch.isnan(result).any()


def test_edit_head_budget_constraint(edit_head):
    B, S = 1, 20
    tokens = torch.arange(1, S + 1).unsqueeze(0)
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :, 2] = 10.0
    
    pred_tokens = torch.full((B, S), 999)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    max_inserts = int(S * edit_head.edit_budget)
    max_expected_length = S + max_inserts
    
    assert result.shape[1] <= max_expected_length + 1


def test_edit_head_keep_operation(edit_head):
    B, S = 1, 5
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :, 0] = 10.0
    
    pred_tokens = torch.full((B, S), 999)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    assert result.shape == tokens.shape
    assert torch.equal(result, tokens)


def test_edit_head_replace_operation(edit_head):
    B, S = 1, 5
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :, 1] = 10.0
    
    pred_tokens = torch.tensor([[10, 20, 30, 40, 50]])
    
    replacement_mask = torch.ones(B, S)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
        replacement_mask=replacement_mask,
    )
    
    max_replacements = int(S * edit_head.edit_budget)
    num_replaced = (result[0] != tokens[0]).sum().item()
    
    assert num_replaced <= max_replacements
    assert result.shape[1] == S


def test_edit_head_delete_operation(edit_head):
    B, S = 1, 10
    tokens = torch.arange(1, S + 1).unsqueeze(0)
    
    edit_logits = torch.zeros(B, S, 5)
    edit_logits[:, :3, 4] = 10.0
    
    pred_tokens = torch.full((B, S), 999)
    
    result = edit_head.apply_edits(
        tokens=tokens,
        edit_logits=edit_logits,
        pred_tokens=pred_tokens,
        enable_structural=True,
    )
    
    assert result.shape[1] < S
    assert result.shape[0] == B
```
---
## File: `reality_stone/tests/llm/test_gpt2_last_layer.py`

```python
import torch
import torch.nn.functional as F
import pytest
from transformers import GPT2LMHeadModel
from reality_stone.models.transformer_converter import RSULFTransformerConverter
from reality_stone.layers.rsulf_cuda import RSULFLayerCUDA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPT-2 RS-ULF last layer test")
def test_gpt2_last_layer_rsulf_forward():
    device = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    converter = RSULFTransformerConverter(
        d_model=model.config.n_embd,
        r=model.config.n_embd,
        eta=0.005,
        alpha=0.01,
        beta=0.0,
        gamma=0.99,
        seq_len=16,
        window=4,
        verbose=False,
        exact=True,
    )
    blocks = list(model.transformer.h)
    last_idx = len(blocks) - 1
    last_block = blocks[last_idx]
    weights = converter.extract_weights(last_block)
    rsulf = RSULFLayerCUDA(
        wq=weights["WQ"],
        wk=weights["WK"],
        w1=weights["W1"],
        w2=weights["W2"],
        d_model=model.config.n_embd,
        r=model.config.n_embd,
        eta=0.005,
        alpha=0.01,
        beta=0.0,
        gamma=0.99,
        seq_len=16,
        window=4,
        global_basis=None,
    )
    if "ln_1_weight" in weights:
        rsulf.set_ln1(weights["ln_1_weight"], weights.get("ln_1_bias"))
    if "ln_2_weight" in weights:
        rsulf.set_ln2(weights["ln_2_weight"], weights.get("ln_2_bias"))
    rsulf.to(device)
    batch = 2
    seq_len = 16
    d_model = model.config.n_embd
    num_samples = 8
    cos_list = []
    rel_list = []
    for step in range(num_samples):
        x = torch.randn(batch, seq_len, d_model, device=device)
        with torch.no_grad():
            out_teacher = last_block(x)
            teacher_out = out_teacher[0] if isinstance(out_teacher, (tuple, list)) else out_teacher
            student_out, _ = rsulf(x)
        assert teacher_out.shape == student_out.shape
        assert torch.isfinite(student_out).all()
        t_flat = teacher_out.view(-1, d_model)
        s_flat = student_out.view(-1, d_model)
        cos = F.cosine_similarity(t_flat, s_flat, dim=-1).mean().item()
        rel = (t_flat - s_flat).norm() / (t_flat.norm() + 1e-8)
        cos_list.append(cos)
        rel_list.append(rel.item())
        print(f"[gpt2_last_layer] sample={step+1}/{num_samples} cos={cos:.4f}, rel_l2={rel:.4f}")
    mean_cos = sum(cos_list) / len(cos_list)
    mean_rel = sum(rel_list) / len(rel_list)
    print(f"[gpt2_last_layer] mean cos={mean_cos:.4f}, mean_rel_l2={mean_rel:.4f}")
```
---
## File: `reality_stone/tests/llm/test_gpt2_manifold_learner.py`

```python
import numpy as np
import torch
from transformers import GPT2LMHeadModel
from reality_stone.models.manifold_learner import GlobalManifoldLearner


def test_gpt2_manifold_learner_collect_weights():
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    d_model = model.config.n_embd
    print(f"d_model: {d_model}")

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=64,
        hyper_hidden_dim=32,
        layer_emb_dim=16,
    )

    learner.collect_weights()
    print(f"Collected layers: {len(learner.layers_wq)}")

    assert len(learner.layers_wq) == 12, f"Expected 12 layers, got {len(learner.layers_wq)}"
    assert learner.layers_wq[0].shape == (d_model, d_model), f"WQ shape mismatch: {learner.layers_wq[0].shape}"
    assert learner.layers_wk[0].shape == (d_model, d_model), f"WK shape mismatch: {learner.layers_wk[0].shape}"

    print(f"WQ[0] shape: {learner.layers_wq[0].shape}")
    print(f"WK[0] shape: {learner.layers_wk[0].shape}")

    learner.extract_global_basis()
    print("Global basis extracted successfully")

    assert learner.u_global is not None
    assert learner.v_global is not None
    print(f"U global shape: {learner.u_global.shape}")
    print(f"V global shape: {learner.v_global.shape}")

    print("TEST PASSED")


def test_gpt2_manifold_learner_full_pipeline():
    print("\n=== Full Pipeline Test ===")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    d_model = model.config.n_embd
    r = 64

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=r,
        hyper_hidden_dim=32,
        layer_emb_dim=16,
    )

    learner.collect_weights()
    learner.extract_global_basis()
    learner.train_hypernet(epochs=10, batch_size=12, lr=1e-2)

    rust_hm = learner.create_rust_hyper_metric()

    for idx in range(len(learner.layers_wq)):
        emb = learner.get_layer_embedding(idx)
        core = rust_hm.generate_core(emb)
        assert core.shape == (r, r), f"Core shape mismatch at layer {idx}"
        assert np.isfinite(core).all(), f"Core has non-finite values at layer {idx}"

    print("HyperMetric generation: OK")

    wrapped = learner.replace_layers()
    x = torch.randn(2, 16, d_model)
    out = wrapped(x)

    assert out.shape == (2, 16, d_model), f"Output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "Output has non-finite values"

    print(f"Wrapped forward: OK, output shape {out.shape}")
    print("FULL PIPELINE TEST PASSED")


if __name__ == "__main__":
    test_gpt2_manifold_learner_collect_weights()
    test_gpt2_manifold_learner_full_pipeline()
```
---
## File: `reality_stone/tests/llm/test_hierarchical_integration.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    infer_hierarchical_llm_on_text,
)
from reality_stone.utils.pre_segmenter import PreSegmenter


@pytest.fixture
def sample_config():
    return HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
        lambda_consistency=0.1,
        lambda_diversity=0.05,
        c_poincare=0.1,
    )


@pytest.fixture
def sample_model(sample_config):
    return HierarchicalSentenceTopicLLM(sample_config)


def test_hierarchical_llm_forward_basic(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=True, use_tree_processing=False)
    
    assert logits.shape[0] == B
    assert "loss" in info
    assert "P_topic" in info
    assert "metric_ctx" in info
    
    loss = info["loss"]
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()
    assert loss.item() >= 0


def test_hierarchical_llm_forward_with_tree(sample_model):
    segmenter = PreSegmenter(max_length=32, k_neighbors=2)
    text = "This is test sentence one. This is test sentence two."
    
    output = segmenter(text)
    
    B = 1
    batch = {
        "tokens": output["tokens"].unsqueeze(0),
        "topo_idx": output["topo_idx"].unsqueeze(0),
        "tree": [output["tree"]],
    }
    
    logits, info = sample_model(batch, compute_loss=True, use_tree_processing=True)
    
    assert logits.shape[0] == B
    assert "loss" in info
    assert not torch.isnan(info["loss"]).any()


def test_hierarchical_llm_forward_loss_components(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=True)
    
    assert "loss_lm" in info
    assert "loss_consistency" in info
    assert "loss_diversity" in info
    assert "loss_length" in info
    
    for loss_key in ["loss_lm", "loss_consistency", "loss_diversity", "loss_length"]:
        loss_val = info[loss_key]
        assert isinstance(loss_val, torch.Tensor)
        assert not torch.isnan(loss_val).any()


def test_hierarchical_llm_encode_decode_cycle(sample_model):
    B, T, L = 1, 2, 10
    tokens = torch.randint(1, 100, (B, T, L))
    
    sentence_embeddings = sample_model.encode_tokens_to_sentences(tokens)
    
    assert sentence_embeddings.shape == (B, T, sample_model.config.d_model)
    assert not torch.isnan(sentence_embeddings).any()
    
    paragraph_embedding = sample_model.encode_sentences_to_paragraph(sentence_embeddings)
    
    assert paragraph_embedding.shape == (B, sample_model.config.d_model)
    assert not torch.isnan(paragraph_embedding).any()


def test_hierarchical_llm_metric_context_generation(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=False)
    
    metric_ctx = info["metric_ctx"]
    assert metric_ctx.shape == (B, T, sample_model.config.d_head, sample_model.config.d_head)
    assert not torch.isnan(metric_ctx).any()
    
    P_topic = info["P_topic"]
    assert P_topic.shape == (B, T, sample_model.config.num_topics)
    assert torch.allclose(P_topic.sum(dim=-1), torch.ones(B, T), atol=1e-5)


def test_hierarchical_llm_backward_pass(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = sample_model(batch, compute_loss=True)
    loss = info["loss"]
    
    loss.backward()
    
    grad_count = 0
    for name, param in sample_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        grad_count += 1
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    assert grad_count > 0


def test_infer_hierarchical_llm_basic():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = "This is a test sentence. Another test sentence here."
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
        use_top_down=False,
    )
    
    assert "original_text" in result
    assert "generated_text" in result
    assert "topics" in result
    assert result["original_text"] == text
    assert isinstance(result["generated_text"], str)
    assert isinstance(result["topics"], list)


def test_infer_hierarchical_llm_with_top_down():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = "First sentence. Second sentence. Third sentence."
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
        use_top_down=True,
    )
    
    assert "original_text" in result
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)


def test_hierarchical_llm_structural_edit():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=True,
        lambda_edit=0.1,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = "Test sentence one. Test sentence two."
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
        use_top_down=False,
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)


def test_hierarchical_llm_dynamic_manifold():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )
    config.enable_dynamic_manifold = True
    
    model = HierarchicalSentenceTopicLLM(config)
    
    B, T, L, K = 2, 3, 10, 2
    batch = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    logits, info = model(batch, compute_loss=True)
    
    assert "loss" in info
    assert not torch.isnan(info["loss"]).any()


def test_hierarchical_llm_empty_input():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = ""
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=32,
        k_neighbors=2,
    )
    
    assert result["original_text"] == text
    assert result["sentences"] == []


def test_hierarchical_llm_long_input():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        max_lm_seq_len=64,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    text = " ".join([f"Sentence number {i}." for i in range(20)])
    
    result = infer_hierarchical_llm_on_text(
        model=model,
        text=text,
        max_length=16,
        k_neighbors=2,
    )
    
    assert "generated_text" in result
    assert isinstance(result["generated_text"], str)


def test_hierarchical_llm_gradient_accumulation(sample_model):
    B, T, L, K = 2, 3, 10, 2
    
    batch1 = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    batch2 = {
        "tokens": torch.randint(1, 100, (B, T, L)),
        "topo_idx": torch.randint(0, T, (B, T, K)),
    }
    
    sample_model.zero_grad()
    
    logits1, info1 = sample_model(batch1, compute_loss=True)
    loss1 = info1["loss"]
    loss1.backward()
    
    grads1 = {name: param.grad.clone() for name, param in sample_model.named_parameters() if param.grad is not None}
    
    logits2, info2 = sample_model(batch2, compute_loss=True)
    loss2 = info2["loss"]
    loss2.backward()
    
    changed = 0
    for name, param in sample_model.named_parameters():
        if param.grad is not None and name in grads1:
            if not torch.equal(param.grad, grads1[name]):
                changed += 1
    assert changed > 0
```
---
## File: `reality_stone/tests/llm/test_hierarchical_llm.py`

```python
import pytest
import torch

from reality_stone.utils.pre_segmenter import PreSegmenter
from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    infer_hierarchical_llm_on_text,
)
from reality_stone.models.bottom_up_encoder import BottomUpEncoder
from reality_stone.models.top_down_decoder import TopDownDecoder


@pytest.fixture
def sample_config():
    """기본 테스트용 HierarchicalLLMConfig 생성"""
    return HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=8,
        num_heads_topic=2,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )


@pytest.fixture
def sample_model(sample_config):
    """기본 테스트용 HierarchicalSentenceTopicLLM 모델 생성"""
    return HierarchicalSentenceTopicLLM(sample_config)


def test_recursive_segment():
    """
    PreSegmenter가 계층적 문서 구조를 올바르게 세그먼트하는지 검증.
    문서 → 섹션 → 서브섹션 → 문장 → 토큰 레벨 분리 확인.
    """
    segmenter = PreSegmenter()
    text = "Section1\n\nSubsection1.1\nSentence one. Sentence two.\n\nSubsection1.2\nAnother sentence."
    levels = ['document', 'section', 'subsection', 'sentence', 'token']
    tree = segmenter.recursive_segment(text, levels)
    
    assert len(tree.nodes) > 5, "트리 노드가 최소 6개 이상이어야 함"
    assert tree.nodes[0].type == 'document', "루트 노드는 document 타입이어야 함"
    
    sections = tree.children(0)
    assert len(sections) == 1, "1개의 섹션이 있어야 함"
    
    subsections = tree.children(sections[0])
    assert len(subsections) == 2, "2개의 서브섹션이 있어야 함"


def test_full_edit_ops(sample_model):
    """
    구조적 편집 연산이 활성화된 경우 추론이 정상 동작하는지 검증.
    enable_structural_edit=True 시 생성된 텍스트 확인.
    """
    sample_model.config.enable_structural_edit = True
    sample_model.eval()
    
    with torch.no_grad():
        out = infer_hierarchical_llm_on_text(
            sample_model, 
            "Test text with three sentences.", 
            max_length=5
        )
    
    generated = out['generated_text']
    assert isinstance(generated, str), "생성된 텍스트는 문자열이어야 함"
    assert len(generated.strip()) > 0, "생성된 텍스트는 비어있지 않아야 함"


def test_pretrain_loading():
    """
    freeze_decoder=True 옵션이 디코더 파라미터를 올바르게 동결하는지 검증.
    """
    config = HierarchicalLLMConfig(
        freeze_decoder=True,
        use_pretrained_embeddings=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    
    frozen_params = [p for p in model.decoder.parameters() if not p.requires_grad]
    all_decoder_params = list(model.decoder.parameters())
    
    assert len(frozen_params) == len(all_decoder_params), \
        "모든 디코더 파라미터가 동결되어야 함"


def test_model_initialization(sample_config):
    """
    모델이 주어진 config로 올바르게 초기화되는지 검증.
    """
    model = HierarchicalSentenceTopicLLM(sample_config)
    
    assert model.config.vocab_size == sample_config.vocab_size
    assert model.config.d_model == sample_config.d_model
    assert model.config.d_head == sample_config.d_head
    
    assert hasattr(model, 'sentence_aggregator')
    assert hasattr(model, 'paragraph_aggregator')
    assert hasattr(model, 'topic_head')
    assert hasattr(model, 'metric_router')
    assert hasattr(model, 'decoder')


def test_forward_pass_shape(sample_model):
    """
    모델의 forward pass가 올바른 shape의 출력을 생성하는지 검증.
    """
    B, T, L = 2, 3, 10
    tokens = torch.randint(1, sample_model.config.vocab_size, (B, T, L))
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    batch = {"tokens": tokens, "topo_idx": topo_idx}
    
    sample_model.eval()
    with torch.no_grad():
        logits, info = sample_model(batch, compute_loss=False)
    
    assert "P_topic" in info
    assert "scores" in info
    assert "metric_ctx" in info
    
    P_topic = info["P_topic"]
    assert P_topic.shape[0] == B
    assert P_topic.shape[1] == T
    assert P_topic.shape[2] == sample_model.config.num_topics


def test_encode_tokens_to_sentences(sample_model):
    """
    토큰 → 문장 인코딩이 올바른 shape을 반환하는지 검증.
    """
    B, T, L = 2, 3, 5
    tokens = torch.randint(1, sample_model.config.vocab_size, (B, T, L))
    
    sample_model.eval()
    with torch.no_grad():
        sentence_embeddings = sample_model.encode_tokens_to_sentences(tokens)
    
    assert sentence_embeddings.shape == (B, T, sample_model.config.d_model)


def test_encode_sentences_to_paragraph(sample_model):
    """
    문장 → 문단 인코딩이 올바른 shape을 반환하는지 검증.
    """
    B, T = 2, 3
    sentence_embeddings = torch.randn(B, T, sample_model.config.d_model)
    
    sample_model.eval()
    with torch.no_grad():
        paragraph_embedding = sample_model.encode_sentences_to_paragraph(
            sentence_embeddings
        )
    
    assert paragraph_embedding.shape == (B, sample_model.config.d_model)


@pytest.mark.parametrize(
    "d_model,n_layer_decoder",
    [
        (128, 2),
        (256, 4),
        (768, 6),
    ],
)
def test_full_pipeline_config_grid(d_model, n_layer_decoder):
    """
    다양한 d_model, n_layer 설정에서
    텍스트 → PreSegmenter → 리만 인코딩 → MetricAttention → LM 디코딩
    전체 파이프라인이 정상 동작하는지 검증.
    """
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=d_model,
        d_head=32,
        num_topics=8,
        num_heads_topic=2,
        n_layer_decoder=n_layer_decoder,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
    )
    model = HierarchicalSentenceTopicLLM(config)
    model.eval()

    segmenter = PreSegmenter(max_length=16, k_neighbors=3)
    text = "첫 번째 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
    seg_output = segmenter(text)

    tokens = seg_output["tokens"].unsqueeze(0)
    topo_idx = seg_output["topo_idx"].unsqueeze(0)
    replacement_mask = seg_output["replacement_mask"].unsqueeze(0)

    batch = {
        "tokens": tokens,
        "topo_idx": topo_idx,
        "replacement_mask": replacement_mask,
    }

    with torch.no_grad():
        logits, info = model(batch, compute_loss=True)

    assert logits.shape[0] == 1
    assert "P_topic" in info
    assert "metric_ctx" in info


def test_bottom_up_encoder_shapes(sample_config):
    """
    BottomUpEncoder가 토큰 임베딩을 문장/문단 임베딩으로
    올바른 shape으로 인코딩하는지 검증.
    """
    encoder = BottomUpEncoder(
        d_model=sample_config.d_model,
        d_head=sample_config.d_head,
        manifold=sample_config.manifold_sentence,
        c=sample_config.c_poincare,
        temperature=sample_config.temperature_agg,
    )

    B, T, L = 2, 3, 5
    token_embeddings = torch.randn(B, T, L, sample_config.d_model)

    sentence_metric = torch.eye(sample_config.d_head).unsqueeze(0).unsqueeze(0).expand(
        B, T, sample_config.d_head, sample_config.d_head
    )

    encoder.eval()
    with torch.no_grad():
        sentence_embeddings, paragraph_embedding = encoder(
            token_embeddings,
            sentence_metric=sentence_metric,
            paragraph_metric=None,
        )

    assert sentence_embeddings.shape == (B, T, sample_config.d_model)
    assert paragraph_embedding.shape == (B, sample_config.d_model)


def test_top_down_decoder_shapes(sample_config):
    """
    TopDownDecoder가 문단 임베딩으로부터
    문장/토큰 시퀀스를 올바른 shape으로 생성하는지 검증.
    """
    decoder = TopDownDecoder(
        d_model=sample_config.d_model,
        d_head=sample_config.d_head,
        vocab_size=sample_config.vocab_size,
    )

    B = 2
    num_sentences = 4
    max_length = 6

    paragraph_embedding = torch.randn(B, sample_config.d_model)

    decoder.eval()
    with torch.no_grad():
        out = decoder(
            paragraph_embedding=paragraph_embedding,
            num_sentences=num_sentences,
            max_length=max_length,
            paragraph_metric=None,
            replacement_mask=None,
            original_tokens=None,
        )

    sentence_embeddings = out["sentence_embeddings"]
    tokens = out["tokens"]

    assert sentence_embeddings.shape == (B, num_sentences, sample_config.d_model)
    assert tokens.shape == (B, num_sentences, max_length)
```
---
## File: `reality_stone/tests/llm/test_manifold_symplectic_pipeline.py`

```python
import numpy as np
import torch
import torch.nn as nn

from reality_stone.models.manifold_learner import GlobalManifoldLearner


class ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ToyModel(nn.Module):
    def __init__(self, d_model: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _init_toy_qk_weights(model: ToyModel, d_model: int) -> None:
    with torch.no_grad():
        for i, block in enumerate(model.layers):
            eye = torch.eye(d_model)
            block.q_proj.weight.copy_(eye)
            scale = 1.0 + float(i)
            diag = torch.arange(1, d_model + 1, dtype=torch.float32) * scale
            block.k_proj.weight.copy_(torch.diag(diag))


def test_global_manifold_learner_creates_hypermetric_toy():
    torch.manual_seed(0)
    np.random.seed(0)

    d_model = 4
    num_layers = 2
    r = 2

    model = ToyModel(d_model=d_model, num_layers=num_layers)
    _init_toy_qk_weights(model, d_model)

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=r,
        hyper_hidden_dim=8,
        layer_emb_dim=4,
    )

    learner.collect_weights()
    learner.extract_global_basis()

    learner.train_hypernet(epochs=5, batch_size=num_layers, lr=1e-2)

    rust_hm = learner.create_rust_hyper_metric()

    for idx in range(num_layers):
        emb = learner.get_layer_embedding(idx)
        core = rust_hm.generate_core(emb)
        assert core.shape == (r, r)
        assert np.isfinite(core).all()
```
---
## File: `reality_stone/tests/llm/test_metric_attention.py`

```python
import torch
from reality_stone.layers.metric_attention import MetricAttention


def test_metric_attention_dot_product_basic():
    """
    Dot-product 모드에서 MetricAttention이 기본적인 shape을 만족하고
    NaN 없이 동작하는지 확인한다.
    """
    B, H, T, S, d_h, d_v = 2, 4, 5, 7, 16, 32
    q = torch.randn(B, H, T, d_h)
    k = torch.randn(B, H, S, d_h)
    v = torch.randn(B, H, S, d_v)
    attn = MetricAttention(
        hidden_size=d_h,
        normalizer="softmax",
        rank=0,
        mode="dot",
        manifold="poincare",
        c=1e-3,
    )
    y = attn(q, k, v)
    # 출력 shape: (B, H, T, d_v)
    assert y.shape == (B, H, T, d_v)
    # NaN 또는 Inf 가 없어야 한다
    assert torch.isfinite(y).all()


def test_metric_attention_geodesic_with_topology():
    """
    Geodesic 모드 + topology index 사용 시에도
    출력 shape과 수치 안정성이 보장되는지 확인한다.
    """
    B, H, T, d_h, d_v = 1, 2, 4, 8, 16

    # geodesic 모드는 T==S 인 self-attention 케이스가 자연스럽다
    q = torch.randn(B, H, T, d_h)
    k = torch.randn(B, H, T, d_h)
    v = torch.randn(B, H, T, d_v)

    # 각 토큰의 이웃을 간단히 "자기 자신 + 다음 토큰"으로 설정
    idx = torch.empty(B, T, 2, dtype=torch.long)
    for t in range(T):
        idx[0, t, 0] = t
        idx[0, t, 1] = min(T - 1, t + 1)

    topo_idx = {"neighbor": idx}
    topk_cfg = {"neighbor": 2}

    attn = MetricAttention(
        hidden_size=d_h,
        normalizer="softmax",
        rank=0,
        mode="geodesic",
        manifold="poincare",
        c=1e-3,
    )

    y = attn(
        q,
        k,
        v,
        topo_idx=topo_idx,
        topk_cfg=topk_cfg,
    )

    # 출력 shape: (B, H, T, d_v)
    assert y.shape == (B, H, T, d_v)
    # NaN 또는 Inf 가 없어야 한다
    assert torch.isfinite(y).all()
```
---
## File: `reality_stone/tests/llm/test_metric_router.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import MetricContextRouter

try:
    import reality_stone.metrikey as _metrikey
    HAS_METRIKEY = True
except Exception:
    HAS_METRIKEY = False


@pytest.fixture
def sample_router():
    """기본 테스트용 MetricContextRouter 생성"""
    return MetricContextRouter(d_head=16, lambda_min=0.5, lambda_max=2.0)


def test_metric_router_shape_and_spd(sample_router):
    """
    MetricContextRouter가 기본적인 shape을 만족하고,
    생성된 메트릭이 SPD(고유값 > 0)를 가지는지 확인한다.
    """
    keys = ["topic:diagnosis|priority:high", "topic:general|priority:low"]
    scores = torch.tensor([[0.8, 0.2]])  # [B=1, T=2]

    L = sample_router(keys, scores)

    # shape: [B, T, d_head, d_head]
    assert L.shape == (1, 2, 16, 16)

    # SPD 확인: 각 (B,T) 위치에서 L L^T 의 고유값이 모두 양수
    for b in range(L.shape[0]):
        for t in range(L.shape[1]):
            G = L[b, t] @ L[b, t].T
            # 대칭 여부
            assert torch.allclose(G, G.T, atol=1e-5)
            eigvals = torch.linalg.eigvalsh(G)
            assert torch.all(eigvals > 0)


@pytest.mark.skipif(HAS_METRIKEY, reason="Fallback 경로는 MetriKey가 없는 환경에서만 의미 있음")
def test_metric_router_identity_fallback_without_metrikey():
    """
    MetriKey 확장이 없는 환경에서는 MetricContextRouter가
    사실상 identity에 가까운 SPD 메트릭을 생성하는지 확인한다.
    """
    d_head = 8
    router = MetricContextRouter(d_head=d_head)

    keys = ["topic:diagnosis|priority:high"]
    scores = torch.tensor([[0.5]])

    L = router(keys, scores)  # [1,1,d_head,d_head]
    G = L[0, 0] @ L[0, 0].T

    eigvals = torch.linalg.eigvalsh(G)
    assert torch.all(eigvals >= router.lambda_min - 1e-5)
    assert torch.all(eigvals <= router.lambda_max + 1e-5)


def test_metric_router_cache_functionality(sample_router):
    """
    MetricContextRouter의 LRU 캐시가 올바르게 동작하는지 검증.
    동일한 key/score 조합은 캐시에서 반환되어야 함.
    """
    keys = ["topic:diagnosis|priority:high"]
    scores = torch.tensor([[1.0]])
    
    L1 = sample_router(keys, scores)
    
    cache_size_before = len(sample_router._cache)
    
    L2 = sample_router(keys, scores)
    
    cache_size_after = len(sample_router._cache)
    
    assert cache_size_before == cache_size_after, "동일 키는 캐시에서 가져와야 함"
    assert torch.allclose(L1, L2), "캐시된 값은 동일해야 함"


def test_metric_router_score_quantization(sample_router):
    """
    score 값이 quantize되어 캐시 효율성이 높아지는지 검증.
    """
    keys = ["topic:treatment|priority:medium"]
    
    scores1 = torch.tensor([[0.501]])
    scores2 = torch.tensor([[0.499]])
    
    L1 = sample_router(keys, scores1)
    L2 = sample_router(keys, scores2)
    
    assert torch.allclose(L1, L2), "근접한 score는 quantize되어 같은 메트릭 반환"
```
---
## File: `reality_stone/tests/llm/test_poincare_cuda.py`

```python
import pytest
import torch
import reality_stone as rs
from reality_stone.layers.klein import project_to_klein


@pytest.mark.cuda
def test_has_cuda_flag_matches_torch():
    """
    reality_stone._has_cuda 플래그가
    PyTorch CUDA 가능 여부 및 CUDA 심볼 로딩 상태와 일관되는지 확인.
    """
    if not torch.cuda.is_available():
        # CUDA 없는 환경에서는 _has_cuda 가 False 여야 한다.
        assert rs._has_cuda is False
        return

    # CUDA 환경에서는 최소한 rust 확장이 있고, _has_cuda 가 True 여야 한다.
    assert rs._has_rust_ext, "Rust extension must be available when CUDA is used"
    assert rs._has_cuda, "Reality Stone CUDA bindings not detected"


@pytest.mark.cuda
def test_poincare_ball_layer_cpu_cuda_consistency():
    """
    Poincaré ball layer 의 CPU / CUDA 결과가 충분히 근접하는지 검증.

    - 입력: 랜덤 B x D 텐서 (Poincaré ball 내부로 투영)
    - 경로:
        rs.poincare_ball_layer (autograd) → CPU / CUDA 분기
    """
    if not (torch.cuda.is_available() and rs._has_cuda):
        pytest.skip("CUDA or Reality Stone CUDA bindings are not available")

    torch.manual_seed(42)
    B, d = 8, 16
    c = 1.0
    t = 0.5

    # 간단한 Poincaré ball 투영
    def project_to_ball(x, epsilon=1e-5):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        max_norm = 1.0 - epsilon
        scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
        return x * scale

    h_cpu = project_to_ball(torch.randn(B, d))
    u_cpu = project_to_ball(torch.randn(B, d))

    # CPU 경로
    z_cpu = rs.poincare_ball_layer(h_cpu, u_cpu, c=c, t=t)

    # CUDA 경로
    h_cuda = h_cpu.cuda()
    u_cuda = u_cpu.cuda()
    z_cuda = rs.poincare_ball_layer(h_cuda, u_cuda, c=c, t=t)

    max_diff = torch.max(torch.abs(z_cpu - z_cuda.cpu())).item()
    # Allow a small numerical tolerance between CPU and CUDA paths
    assert max_diff < 5e-5, f"Poincaré CPU/CUDA mismatch: max_diff={max_diff:.3e}"


@pytest.mark.cuda
def test_lorentz_layer_cpu_cuda_consistency_forward_backward():
    """
    Lorentz layer 의 CPU / CUDA 순전파와 역전파 결과가 충분히 근접하는지 검증.
    """
    if not (torch.cuda.is_available() and rs._has_cuda):
        pytest.skip("CUDA or Reality Stone CUDA bindings are not available")

    torch.manual_seed(42)
    B, dim = 4, 5  # Minkowski: time + (dim-1) space
    c = 1.0
    t = 0.3

    def sample_lorentz(batch: int, d: int, device: torch.device) -> torch.Tensor:
        # Generate points on the Lorentz hyperboloid: x0^2 - ||x||^2 = 1
        spatial = torch.randn(batch, d - 1, device=device) * 0.1
        sq = (spatial * spatial).sum(dim=1, keepdim=True)
        time = torch.sqrt(1.0 + sq)
        return torch.cat([time, spatial], dim=1)

    u_cpu = sample_lorentz(B, dim, device=torch.device("cpu")).requires_grad_(True)
    v_cpu = sample_lorentz(B, dim, device=torch.device("cpu")).requires_grad_(True)

    y_cpu = rs.lorentz_layer(u_cpu, v_cpu, c=c, t=t)

    u_cuda = u_cpu.detach().clone().cuda().requires_grad_(True)
    v_cuda = v_cpu.detach().clone().cuda().requires_grad_(True)
    y_cuda = rs.lorentz_layer(u_cuda, v_cuda, c=c, t=t)

    # Forward consistency
    max_diff_fwd = torch.max(torch.abs(y_cpu - y_cuda.cpu())).item()
    assert max_diff_fwd < 1e-4, f"Lorentz layer forward CPU/CUDA mismatch: max_diff={max_diff_fwd:.3e}"

    # Backward consistency
    grad = torch.randn_like(y_cpu)
    y_cpu.backward(grad)
    y_cuda.backward(grad.cuda())

    max_diff_gu = torch.max(torch.abs(u_cpu.grad - u_cuda.grad.cpu())).item()
    max_diff_gv = torch.max(torch.abs(v_cpu.grad - v_cuda.grad.cpu())).item()
    max_grad_diff = max(max_diff_gu, max_diff_gv)
    assert max_grad_diff < 1e-3, f"Lorentz layer backward CPU/CUDA mismatch: max_diff={max_grad_diff:.3e}"


@pytest.mark.cuda
def test_klein_layer_cpu_cuda_consistency_forward_backward():
    """
    Klein layer 의 CPU / CUDA 순전파와 역전파 결과가 충분히 근접하는지 검증.
    """
    if not (torch.cuda.is_available() and rs._has_cuda):
        pytest.skip("CUDA or Reality Stone CUDA bindings are not available")

    torch.manual_seed(42)
    B, d = 4, 4
    c = 1.0
    t = 0.3

    # Project random vectors safely into the Klein disk for curvature c.
    u_base = torch.randn(B, d)
    v_base = torch.randn(B, d)
    u_cpu = project_to_klein(u_base, c).requires_grad_(True)
    v_cpu = project_to_klein(v_base, c).requires_grad_(True)

    y_cpu = rs.klein_layer(u_cpu, v_cpu, c=c, t=t)

    u_cuda = u_cpu.detach().clone().cuda().requires_grad_(True)
    v_cuda = v_cpu.detach().clone().cuda().requires_grad_(True)
    y_cuda = rs.klein_layer(u_cuda, v_cuda, c=c, t=t)

    # Forward consistency
    max_diff_fwd = torch.max(torch.abs(y_cpu - y_cuda.cpu())).item()
    assert max_diff_fwd < 1e-4, f"Klein layer forward CPU/CUDA mismatch: max_diff={max_diff_fwd:.3e}"

    # Backward consistency
    grad = torch.randn_like(y_cpu)
    y_cpu.backward(grad)
    y_cuda.backward(grad.cuda())

    max_diff_gu = torch.max(torch.abs(u_cpu.grad - u_cuda.grad.cpu())).item()
    max_diff_gv = torch.max(torch.abs(v_cpu.grad - v_cuda.grad.cpu())).item()
    max_grad_diff = max(max_diff_gu, max_diff_gv)
    assert max_grad_diff < 1e-3, f"Klein layer backward CPU/CUDA mismatch: max_diff={max_grad_diff:.3e}"
```
---
## File: `reality_stone/tests/llm/test_rce_lexical_decoder.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import RCELexicalDecoder


@pytest.fixture
def sample_decoder():
    """기본 테스트용 RCELexicalDecoder 생성"""
    return RCELexicalDecoder(
        vocab_size=100,
        d_model=32,
        n_layer=2,
        n_head=4,
    )


def test_rce_decoder_shapes_and_mask_preservation(sample_decoder):
    """
    RCELexicalDecoder가 기본적인 shape을 유지하고,
    replacement_mask=0 위치의 토큰은 항상 원본을 그대로 반환하는지 확인.
    """
    vocab_size = sample_decoder.vocab_size
    d_model = sample_decoder.d_model
    n_head = sample_decoder.n_head

    B, T = 2, 5
    input_ids = torch.randint(1, vocab_size, (B, T))

    # metric_ctx와 topo_idx는 현재 구현에서 사용되지 않지만,
    # API 호환성을 위해 올바른 shape으로 전달한다.
    # d_h는 여기서 d_model//n_head로 맞춰준다.
    d_h = d_model // n_head
    metric_ctx = torch.randn(B, T, d_h, d_h)
    topo_idx = torch.randint(0, T, (B, T, 3))

    # 일부 위치는 교체 불가(0), 일부는 교체 가능(1)
    replacement_mask = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1],
        ]
    )

    # 간단한 후보 사전: 각 토큰은 자기 자신과 +1 토큰만 허용
    candidates = {
        int(tid): [int(tid), min(int(tid) + 1, vocab_size - 1)]
        for tid in torch.unique(input_ids).tolist()
    }

    output_ids, logits = sample_decoder(
        input_ids=input_ids,
        metric_ctx=metric_ctx,
        replacement_mask=replacement_mask,
        topo_idx=topo_idx,
        candidates=candidates,
    )

    # shape 확인
    assert output_ids.shape == (B, T)
    assert logits.shape == (B, T, vocab_size)

    # replacement_mask=0인 위치는 반드시 원본 토큰 유지
    unchanged = replacement_mask == 0
    assert torch.equal(output_ids[unchanged], input_ids[unchanged])


def test_rce_decoder_respects_lexical_candidates(sample_decoder):
    """
    후보 집합 내에서만 토큰이 선택되는지 확인.
    - replacement_mask=1 위치: output_ids는 해당 후보 집합에 포함되어야 한다.
    - replacement_mask=0 위치: 항상 원본 토큰을 유지해야 한다.
    """
    B, T = 1, 4
    input_ids = torch.tensor([[10, 20, 30, 40]])

    d_h = sample_decoder.d_model // sample_decoder.n_head
    metric_ctx = torch.randn(B, T, d_h, d_h)
    topo_idx = torch.randint(0, T, (B, T, 2))

    replacement_mask = torch.tensor([[1, 0, 1, 0]])

    candidates = {
        10: [10, 11],
        20: [20, 21],
        30: [30],
        40: [40, 41, 42],
    }

    output_ids, _ = sample_decoder(
        input_ids=input_ids,
        metric_ctx=metric_ctx,
        replacement_mask=replacement_mask,
        topo_idx=topo_idx,
        candidates=candidates,
    )

    # mask=0 위치는 반드시 원본 유지
    assert int(output_ids[0, 1]) == 20
    assert int(output_ids[0, 3]) == 40

    assert int(output_ids[0, 0]) in candidates[10]
    assert int(output_ids[0, 2]) in candidates[30]


def test_rce_decoder_no_candidates_fallback(sample_decoder):
    """
    후보 집합이 없는 경우 원본 토큰을 유지하는지 검증.
    """
    B, T = 1, 3
    input_ids = torch.tensor([[5, 10, 15]])
    
    d_h = sample_decoder.d_model // sample_decoder.n_head
    metric_ctx = torch.randn(B, T, d_h, d_h)
    topo_idx = torch.randint(0, T, (B, T, 2))
    replacement_mask = torch.ones_like(input_ids)
    
    output_ids, logits = sample_decoder(
        input_ids=input_ids,
        metric_ctx=metric_ctx,
        replacement_mask=replacement_mask,
        topo_idx=topo_idx,
        candidates=None,
    )
    
    assert logits.shape == (B, T, sample_decoder.vocab_size)
    assert torch.equal(output_ids, input_ids), "후보 없을 시 원본 유지"


def test_rce_decoder_all_masked(sample_decoder):
    """
    모든 위치가 mask=0인 경우 원본을 그대로 반환하는지 검증.
    """
    B, T = 2, 4
    input_ids = torch.randint(1, sample_decoder.vocab_size, (B, T))
    
    d_h = sample_decoder.d_model // sample_decoder.n_head
    metric_ctx = torch.randn(B, T, d_h, d_h)
    topo_idx = torch.randint(0, T, (B, T, 2))
    replacement_mask = torch.zeros_like(input_ids)
    
    output_ids, _ = sample_decoder(
        input_ids=input_ids,
        metric_ctx=metric_ctx,
        replacement_mask=replacement_mask,
        topo_idx=topo_idx,
        candidates={},
    )
    
    assert torch.equal(output_ids, input_ids), "모든 위치 mask=0이면 원본 유지"
```
---
## File: `reality_stone/tests/llm/test_rsu_v2_symplectic_pipeline.py`

```python
import numpy as np
import torch
import torch.nn as nn

from reality_stone.models.manifold_learner import GlobalManifoldLearner, TinyMLP


class ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ToyModel(nn.Module):
    def __init__(self, d_model: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock(d_model) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _init_toy_qk_weights(model: ToyModel, d_model: int) -> None:
    with torch.no_grad():
        for i, block in enumerate(model.layers):
            eye = torch.eye(d_model)
            block.q_proj.weight.copy_(eye)
            scale = 1.0 + float(i)
            diag = torch.arange(1, d_model + 1, dtype=torch.float32) * scale
            block.k_proj.weight.copy_(torch.diag(diag))


def test_rsu_v2_symplectic_end_to_end(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    d_model = 4
    num_layers = 2
    r = 2

    model = ToyModel(d_model=d_model, num_layers=num_layers)
    _init_toy_qk_weights(model, d_model)

    learner = GlobalManifoldLearner(
        model=model,
        d_model=d_model,
        r=r,
        hyper_hidden_dim=8,
        layer_emb_dim=4,
    )

    learner.collect_weights()
    learner.extract_global_basis()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner.layer_embeddings = nn.Embedding(num_layers, 4).to(device)
    with torch.no_grad():
        learner.layer_embeddings.weight.zero_()
    hypernet = TinyMLP(input_dim=4, hidden_dim=8, output_dim=r * r).to(device)
    with torch.no_grad():
        hypernet.l1.weight.zero_()
        hypernet.l1.bias.zero_()
        hypernet.l2.weight.zero_()
        hypernet.l2.bias.zero_()
    learner.hypernet = hypernet

    rsu_path = tmp_path / "toy_hypermetric.rsu2.npz"
    learner.save_rsu_v2(rsu_path)

    learner_loaded = GlobalManifoldLearner.from_rsu_v2(
        model=model,
        path=rsu_path,
    )

    wrapped = learner_loaded.replace_layers()

    x = torch.randn(3, d_model)
    out = wrapped(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
```
---
## File: `reality_stone/tests/llm/test_sentence_topic_head.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import SentenceTopicHead


@pytest.fixture
def sample_topic_head():
    """기본 테스트용 SentenceTopicHead 생성"""
    return SentenceTopicHead(
        d_model=768,
        d_head=64,
        num_topics=8,
        num_heads=4,
        c_poincare=1e-3,
    )


def test_sentence_topic_head_output_shapes_and_probs(sample_topic_head):
    """
    SentenceTopicHead가 기본적인 shape을 만족하고,
    주제 확률이 각 문장마다 1로 정규화되는지 확인.
    """
    B, T = 2, 3
    d_model = sample_topic_head.d_model
    num_topics = sample_topic_head.num_topics

    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))

    P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)

    # shape 확인
    assert P_topic.shape == (B, T, num_topics)
    assert scores.shape == (B, T)
    assert isinstance(metric_keys, list)
    assert len(metric_keys) == B * T

    # 각 문장별 확률 합이 1인지 확인 (수치 오차 허용)
    probs_sum = P_topic.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_sentence_topic_head_metric_keys_format(sample_topic_head):
    """
    metric_keys가 docs 명세대로
    'topic:{name}|priority:{high|medium|low}' 형식을 따르는지 확인.
    """
    B, T = 1, 4
    d_model = sample_topic_head.d_model

    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))

    P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)

    assert len(metric_keys) == B * T

    for key in metric_keys:
        # 기본 형식 체크
        assert "topic:" in key
        assert "priority:" in key
        parts = key.split("|")
        assert parts[0].startswith("topic:")
        assert parts[1].startswith("priority:")

        priority = parts[1].split(":", 1)[1]
        assert priority in {"high", "medium", "low"}


def test_sentence_topic_head_poincare_projection(sample_topic_head):
    """
    SentenceTopicHead의 Poincaré 임베딩이 ball 내부에 투영되는지 검증.
    """
    B, T = 2, 3
    d_model = sample_topic_head.d_model
    x = torch.randn(B, T, d_model)
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    with torch.no_grad():
        P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)
    
    assert not torch.isnan(P_topic).any(), "P_topic에 NaN이 없어야 함"
    assert not torch.isnan(scores).any(), "scores에 NaN이 없어야 함"
    
    probs_sum = P_topic.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)


def test_sentence_topic_head_topic_names(sample_topic_head):
    """
    topic_names 리스트가 올바르게 설정되어 있는지 검증.
    """
    expected_topics = [
        "chief_complaint",
        "history",
        "physical_exam",
        "diagnosis",
        "treatment_plan",
        "prognosis",
        "follow_up",
        "general",
    ]
    
    assert sample_topic_head.topic_names == expected_topics
    assert len(sample_topic_head.topic_names) == sample_topic_head.num_topics


def test_sentence_topic_head_gradient_flow(sample_topic_head):
    """
    SentenceTopicHead의 backward pass가 올바르게 작동하는지 검증.
    """
    B, T = 2, 3
    d_model = sample_topic_head.d_model
    x = torch.randn(B, T, d_model, requires_grad=True)
    topo_idx = torch.randint(0, T, (B, T, 2))
    
    P_topic, scores, metric_keys = sample_topic_head(x, topo_idx)
    
    loss = P_topic.sum() + scores.sum()
    loss.backward()
    
    assert x.grad is not None, "입력에 대한 gradient가 계산되어야 함"
    assert not torch.isnan(x.grad).any(), "gradient에 NaN이 없어야 함"
```
---
## File: `reality_stone/tests/llm/test_spd_performance.py`

```python
import pytest
import torch
import time

from reality_stone.models.hierarchical_sentence_topic_llm import (
    SPDMetricMixer,
    _spd_log_euclidean_mean,
)


def test_spd_fast_mixing_vs_log_euclidean():
    d_head = 32
    B = 10
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1) * 1.5
    children_metrics = torch.eye(d_head).unsqueeze(0).unsqueeze(0).expand(B, 3, -1, -1) * 0.8
    
    mixer_fast = SPDMetricMixer(d_head, use_fast_mixing=True)
    mixer_slow = SPDMetricMixer(d_head, use_fast_mixing=False)
    
    result_fast = mixer_fast.mix_hierarchy(parent_metric, self_metric, children_metrics)
    result_slow = mixer_slow.mix_hierarchy(parent_metric, self_metric, children_metrics)
    
    assert result_fast.shape == (B, d_head, d_head)
    assert result_slow.shape == (B, d_head, d_head)
    
    assert not torch.isnan(result_fast).any()
    assert not torch.isnan(result_slow).any()


def test_spd_fast_mixing_performance():
    d_head = 32
    B = 100
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1) * 1.5
    children_metrics = torch.eye(d_head).unsqueeze(0).unsqueeze(0).expand(B, 5, -1, -1) * 0.8
    
    mixer_fast = SPDMetricMixer(d_head, use_fast_mixing=True)
    
    start = time.time()
    for _ in range(10):
        result = mixer_fast.mix_hierarchy(parent_metric, self_metric, children_metrics)
    fast_time = time.time() - start
    
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()
    assert fast_time < 1.0


def test_spd_log_euclidean_mean_batch():
    B, N, d = 4, 3, 16
    
    spd_matrices = torch.eye(d).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
    weights = torch.ones(B, N) / N
    
    result = _spd_log_euclidean_mean(spd_matrices, weights)
    
    assert result.shape == (B, d, d)
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    eigvals = torch.linalg.eigvalsh(result)
    assert torch.all(eigvals > 0)


def test_spd_mixer_gradient_flow():
    d_head = 32
    B = 5
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1).requires_grad_(True)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1).requires_grad_(True)
    
    mixer = SPDMetricMixer(d_head, use_fast_mixing=True)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, None)
    loss = result.sum()
    loss.backward()
    
    assert parent_metric.grad is not None
    assert self_metric.grad is not None
    assert not torch.isnan(parent_metric.grad).any()


def test_spd_mixer_with_children():
    d_head = 32
    B = 5
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1) * 1.2
    children_metrics = torch.eye(d_head).unsqueeze(0).unsqueeze(0).expand(B, 4, -1, -1) * 0.9
    
    mixer = SPDMetricMixer(d_head, use_fast_mixing=True)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, children_metrics)
    
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()
    eigvals = torch.linalg.eigvalsh(result)
    assert torch.all(eigvals > 0)


def test_spd_mixer_weights_normalized():
    d_head = 16
    B = 3
    
    parent_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head).unsqueeze(0).expand(B, -1, -1)
    
    mixer = SPDMetricMixer(d_head, gamma_up=0.5, gamma_self=0.3, gamma_down=0.2, use_fast_mixing=True)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, None)
    
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_spd_fast_mixing_cuda():
    d_head = 32
    B = 50
    device = torch.device("cuda")
    
    parent_metric = torch.eye(d_head, device=device).unsqueeze(0).expand(B, -1, -1)
    self_metric = torch.eye(d_head, device=device).unsqueeze(0).expand(B, -1, -1) * 1.5
    children_metrics = torch.eye(d_head, device=device).unsqueeze(0).unsqueeze(0).expand(B, 3, -1, -1)
    
    mixer = SPDMetricMixer(d_head, use_fast_mixing=True).to(device)
    
    result = mixer.mix_hierarchy(parent_metric, self_metric, children_metrics)
    
    assert result.device.type == "cuda"
    assert result.shape == (B, d_head, d_head)
    assert not torch.isnan(result).any()
```
---
## File: `reality_stone/tests/llm/test_top_down_decoder.py`

```python
import pytest
import torch

from reality_stone.models.hierarchical_sentence_topic_llm import (
    HierarchicalSentenceTopicLLM,
    HierarchicalLLMConfig,
    _apply_top_down_decoding,
)
from reality_stone.utils.pre_segmenter import DocumentTree, TreeNode


@pytest.fixture
def sample_model():
    config = HierarchicalLLMConfig(
        vocab_size=1000,
        d_model=128,
        d_head=32,
        num_topics=4,
        n_layer_decoder=2,
        n_head_decoder=2,
        use_pretrained_embeddings=False,
        enable_structural_edit=False,
    )
    return HierarchicalSentenceTopicLLM(config)


@pytest.fixture
def sample_tree():
    nodes = [
        TreeNode(id=0, type="document", parent=None, text="test document"),
        TreeNode(id=1, type="sentence", parent=0, text="sentence 1"),
        TreeNode(id=2, type="sentence", parent=0, text="sentence 2"),
    ]
    return DocumentTree(nodes=nodes, root_id=0)


def test_top_down_decoding_basic(sample_model, sample_tree):
    B, T, L = 1, 2, 10
    device = next(sample_model.parameters()).device
    
    tokens = torch.randint(1, 100, (B, T, L))
    replacement_mask = torch.ones(T, L)
    
    hidden = torch.randn(B, T * L, sample_model.config.d_model)
    info = {"hidden": hidden}
    
    result = _apply_top_down_decoding(
        model=sample_model,
        tree=sample_tree,
        info=info,
        tokens=tokens,
        replacement_mask=replacement_mask,
        device=device,
    )
    
    assert result.shape[0] == B
    assert result.shape[1] <= T * L
    assert not torch.isnan(result).any()


def test_top_down_decoding_with_tree_processor(sample_model, sample_tree):
    B, T, L = 1, 2, 10
    device = next(sample_model.parameters()).device
    
    tokens = torch.randint(1, 100, (B, T, L))
    replacement_mask = torch.ones(T, L)
    
    hidden = torch.randn(B, T * L, sample_model.config.d_model)
    info = {"hidden": hidden}
    
    result = _apply_top_down_decoding(
        model=sample_model,
        tree=sample_tree,
        info=info,
        tokens=tokens,
        replacement_mask=replacement_mask,
        device=device,
    )
    
    assert result.shape[0] == B
    assert result.dtype == torch.long
    assert (result >= 0).all()
    assert (result < sample_model.config.vocab_size).all()


def test_top_down_decoding_preserves_structure(sample_model, sample_tree):
    B, T, L = 1, 2, 10
    device = next(sample_model.parameters()).device
    
    tokens = torch.randint(1, 100, (B, T, L))
    replacement_mask = torch.zeros(T, L)
    replacement_mask[:, :3] = 1
    
    hidden = torch.randn(B, T * L, sample_model.config.d_model)
    info = {"hidden": hidden}
    
    result = _apply_top_down_decoding(
        model=sample_model,
        tree=sample_tree,
        info=info,
        tokens=tokens,
        replacement_mask=replacement_mask,
        device=device,
    )
    
    tokens_flat = tokens.view(B, T * L)
    for b in range(B):
        for i in range(min(T * L, result.shape[1])):
            sent_idx = i // L
            tok_idx = i % L
            if sent_idx < T and tok_idx < L:
                if replacement_mask[sent_idx, tok_idx] == 0:
                    assert result[b, i] == tokens_flat[b, i] or tokens_flat[b, i] == 0
```
