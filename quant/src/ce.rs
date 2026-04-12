use std::f64::consts::PI;

fn lambert_w0(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        return x;
    }
    let mut w = if x < std::f64::consts::E {
        let l = (1.0 + x).ln();
        if l > 0.0 { l } else { x }
    } else {
        let lx = x.ln();
        lx - lx.ln()
    };
    for _ in 0..64 {
        let ew = w.exp();
        let wew = w * ew;
        let wp1 = w + 1.0;
        if wp1.abs() < 1e-30 { break; }
        let num = wew - x;
        let denom = ew * wp1 - (w + 2.0) * num / (2.0 * wp1);
        if denom.abs() < 1e-30 { break; }
        let delta = num / denom;
        w -= delta;
        if delta.abs() < 1e-15 * w.abs().max(1e-15) { break; }
    }
    w
}

fn leptonic_running() -> f64 {
    const M_Z_MEV: f64 = 91_188.0;
    let mz2 = M_Z_MEV * M_Z_MEV;
    let masses = [0.51100_f64, 105.658, 1776.86];
    masses.iter()
        .map(|&m| (mz2 / (m * m)).ln() - 5.0 / 3.0)
        .sum::<f64>() / (3.0 * PI)
}

fn solve_alpha_s() -> f64 {
    let alpha_inv_0 = 4.0 * PI.powi(3) + PI.powi(2) + PI;
    let delta_lep = leptonic_running();
    let delta_running = delta_lep + 3.750;
    let alpha_em_mz_target = 1.0 / (alpha_inv_0 - delta_running);

    let mut lo = 0.05_f64;
    let mut hi = 0.15_f64;
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        let s2tw = 4.0 * mid.powf(4.0 / 3.0);
        let alpha_total = 1.0 / (2.0 * PI);
        let alpha_w = (alpha_total - mid) / (1.0 + s2tw);
        let alpha_em = alpha_w * s2tw;
        if alpha_em < alpha_em_mz_target { hi = mid; } else { lo = mid; }
    }
    0.5 * (lo + hi)
}

#[derive(Clone, Debug)]
pub struct CeConst {
    pub alpha_total: f64,
    pub alpha_s: f64,
    pub alpha_w: f64,
    pub alpha_em_mz: f64,
    pub sin2_tw: f64,
    pub delta: f64,
    pub d_eff: f64,
    pub epsilon2: f64,
    pub omega_b: f64,
    pub omega_lambda: f64,
    pub omega_dm: f64,
    pub dark_ratio_r: f64,
    pub rho_contract: f64,
    pub epsilon_obs: f64,
    pub f_factor: f64,
}

impl CeConst {
    pub fn derive() -> Self {
        let alpha_total = 1.0 / (2.0 * PI);
        let alpha_s = solve_alpha_s();
        let sin2_tw = 4.0 * alpha_s.powf(4.0 / 3.0);
        let alpha_w = (alpha_total - alpha_s) / (1.0 + sin2_tw);
        let alpha_em_mz = alpha_w * sin2_tw;

        let delta = sin2_tw * (1.0 - sin2_tw);
        let d_eff = 3.0 + delta;

        let arg = -d_eff * (-d_eff).exp();
        let w0 = lambert_w0(arg);
        let epsilon2 = -w0 / d_eff;
        let omega_b = epsilon2;
        // Refined cosmology split from CE docs:
        // R = alpha_s * D_eff * (1 + epsilon^2 * delta)
        let dark_ratio_r = alpha_s * d_eff * (1.0 + epsilon2 * delta);
        let omega_lambda = (1.0 - epsilon2) / (1.0 + dark_ratio_r);
        let omega_dm = (1.0 - epsilon2) * dark_ratio_r / (1.0 + dark_ratio_r);
        let rho_contract = d_eff * epsilon2;
        let epsilon_obs = 2.0 * omega_lambda - 1.0;
        let f_factor = 1.0 + alpha_s * d_eff;

        Self {
            alpha_total,
            alpha_s,
            alpha_w,
            alpha_em_mz,
            sin2_tw,
            delta,
            d_eff,
            epsilon2,
            omega_b,
            omega_lambda,
            omega_dm,
            dark_ratio_r,
            rho_contract,
            epsilon_obs,
            f_factor,
        }
    }

    pub fn print_summary(&self) {
        println!("--- CE Quant Constants ---");
        println!("  alpha_total  = {:.6}", self.alpha_total);
        println!("  alpha_s      = {:.6}", self.alpha_s);
        println!("  alpha_w      = {:.6}", self.alpha_w);
        println!("  alpha_em(MZ) = {:.6}", self.alpha_em_mz);
        println!("  sin2_tw      = {:.6}", self.sin2_tw);
        println!("  D_eff        = {:.6}", self.d_eff);
        println!("  epsilon^2    = {:.6}", self.epsilon2);
        println!("  Omega_b      = {:.6}", self.omega_b);
        println!("  Omega_Lambda = {:.6}", self.omega_lambda);
        println!("  Omega_DM     = {:.6}", self.omega_dm);
        println!("  R_dark       = {:.6}", self.dark_ratio_r);
        println!("  rho_contract = {:.6}", self.rho_contract);
        println!("  epsilon_obs  = {:.6}", self.epsilon_obs);
        println!("  F_factor     = {:.6}", self.f_factor);
    }
}
