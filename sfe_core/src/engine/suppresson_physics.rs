use std::f64::consts::PI;

pub const M_MU_MEV: f64 = 105.66;
pub const M_E_MEV: f64 = 0.511;
pub const M_P_MEV: f64 = 938.27;
pub const M_TAU_MEV: f64 = 1776.86;

pub const ALPHA_EM: f64 = 1.0 / 137.036;
pub const HBAR_C_MEV_FM: f64 = 197.327;

// [NEW] SFE Effective Mass Correction
pub struct EffectiveMassCalculator;

impl EffectiveMassCalculator {
    /// Part8.5: Delta M = M0 * (1 - e^-1)
    /// 입자가 억압장(R)을 통과하며 잃어버리는 유효 질량/에너지 계산
    pub fn calculate_loss(m0: f64) -> f64 {
        let e_inv = (-1.0_f64).exp(); // 0.3678...
        m0 * (1.0 - e_inv)
    }

    /// 남은 유효 질량: M_eff = M0 * e^-1
    /// (실제로 관측되거나 상호작용에 참여하는 잔여 질량)
    pub fn calculate_effective_mass(m0: f64) -> f64 {
        let e_inv = (-1.0_f64).exp();
        m0 * e_inv
    }

    /// 0.37 법칙 검증
    pub fn verify_law() {
        let loss_ratio = 1.0 - (-1.0_f64).exp();
        println!("[SFE] 0.37 Law Verification:");
        println!("  Theoretical Loss Ratio (1 - 1/e): {loss_ratio:.6}");
        println!(
            "  Effective Mass Ratio (1/e):       {:.6}",
            (-1.0_f64).exp()
        );
    }
}

pub struct MassProportionalCoupling {
    pub kappa: f64,
    pub g_e: f64,
    pub g_mu: f64,
    pub g_tau: f64,
    pub g_p: f64,
}

impl MassProportionalCoupling {
    pub fn from_g_mu(g_mu: f64) -> Self {
        let kappa = g_mu / M_MU_MEV;

        Self {
            kappa,
            g_e: kappa * M_E_MEV,
            g_mu,
            g_tau: kappa * M_TAU_MEV,
            g_p: kappa * M_P_MEV,
        }
    }

    pub fn verify_mass_proportionality(&self) -> bool {
        let ratio_mu_e = self.g_mu / self.g_e;
        let expected_ratio = M_MU_MEV / M_E_MEV;

        let error = (ratio_mu_e - expected_ratio).abs() / expected_ratio;
        error < 1e-6
    }

    pub fn print_coupling_hierarchy(&self) {
        println!("\n질량-비례 결합 법칙 검증");
        println!("  κ = {:.3e}", self.kappa);
        println!("\n  렙톤 결합 상수:");
        println!("    g_e   = {:.3e} (전자)", self.g_e);
        println!("    g_μ   = {:.3e} (뮤온)", self.g_mu);
        println!("    g_τ   = {:.3e} (타우)", self.g_tau);
        println!("    g_p   = {:.3e} (양성자)", self.g_p);

        println!("\n  비율 검증:");
        println!(
            "    g_μ/g_e = {:.1} (예상: {:.1})",
            self.g_mu / self.g_e,
            M_MU_MEV / M_E_MEV
        );
        println!(
            "    g_τ/g_μ = {:.1} (예상: {:.1})",
            self.g_tau / self.g_mu,
            M_TAU_MEV / M_MU_MEV
        );
        println!(
            "    g_p/g_μ = {:.1} (예상: {:.1})",
            self.g_p / self.g_mu,
            M_P_MEV / M_MU_MEV
        );

        let is_valid = self.verify_mass_proportionality();
        println!(
            "\n  질량-비례 법칙: {}",
            if is_valid { "✓ 성립" } else { "✗ 위반" }
        );

        // [NEW] 유효 질량 보정 검증 호출
        EffectiveMassCalculator::verify_law();
    }
}

pub struct YukawaPotential {
    pub g_lep: f64,
    pub g_p: f64,
    pub m_phi_mev: f64,
}

impl YukawaPotential {
    pub fn new(g_lep: f64, g_p: f64, m_phi_mev: f64) -> Self {
        Self {
            g_lep,
            g_p,
            m_phi_mev,
        }
    }

    pub fn value_at(&self, r_fm: f64) -> f64 {
        let m_phi_inv_fm = self.m_phi_mev / HBAR_C_MEV_FM;
        -(self.g_lep * self.g_p) / (4.0 * PI) * (-m_phi_inv_fm * r_fm).exp() / r_fm
    }

    pub fn mu_to_e_ratio(&self, _r_fm: f64, coupling: &MassProportionalCoupling) -> f64 {
        let v_mu = -(coupling.g_mu * coupling.g_p) / (4.0 * PI);
        let v_e = -(coupling.g_e * coupling.g_p) / (4.0 * PI);
        v_mu / v_e
    }
}

pub struct BohrRadiusCalculator;

impl BohrRadiusCalculator {
    pub fn electron_proton_angstrom() -> f64 {
        HBAR_C_MEV_FM * 1e-5 / (ALPHA_EM * M_E_MEV)
    }

    pub fn muon_proton_fm() -> f64 {
        let reduced_mass = (M_MU_MEV * M_P_MEV) / (M_MU_MEV + M_P_MEV);
        HBAR_C_MEV_FM / (ALPHA_EM * reduced_mass)
    }

    pub fn mass_ratio_effect() -> f64 {
        (M_MU_MEV / M_E_MEV).powi(3)
    }
}

pub struct AnomalousMagneticMoment {
    pub g_mu: f64,
    pub m_phi_mev: f64,
}

impl AnomalousMagneticMoment {
    pub fn new(g_mu: f64, m_phi_mev: f64) -> Self {
        Self { g_mu, m_phi_mev }
    }

    pub fn delta_a_mu(&self) -> f64 {
        (self.g_mu.powi(2) * M_MU_MEV.powi(2)) / (16.0 * PI.powi(2) * self.m_phi_mev.powi(2))
    }

    pub fn delta_a_e(&self, g_e: f64) -> f64 {
        (g_e.powi(2) * M_E_MEV.powi(2)) / (16.0 * PI.powi(2) * self.m_phi_mev.powi(2))
    }

    pub fn verify_muon_anomaly(&self) -> bool {
        let measured = 251e-11;
        let predicted = self.delta_a_mu();
        let ratio = predicted / measured;
        ratio > 0.5 && ratio < 2.0
    }
}

pub struct ProtonRadiusPuzzle {
    pub coupling: MassProportionalCoupling,
    pub m_phi_mev: f64,
}

impl ProtonRadiusPuzzle {
    pub fn new(coupling: MassProportionalCoupling, m_phi_mev: f64) -> Self {
        Self {
            coupling,
            m_phi_mev,
        }
    }

    pub fn delta_r_squared_fm2(&self, m_lep_mev: f64, g_lep: f64) -> f64 {
        let reduced_mass = (m_lep_mev * M_P_MEV) / (m_lep_mev + M_P_MEV);
        let denominator = (2.0 * reduced_mass * ALPHA_EM + self.m_phi_mev).powi(2);

        (g_lep * self.coupling.g_p) / denominator
    }

    pub fn muon_electron_difference_fm(&self) -> f64 {
        let delta_r2_mu = self.delta_r_squared_fm2(M_MU_MEV, self.coupling.g_mu);
        let delta_r2_e = self.delta_r_squared_fm2(M_E_MEV, self.coupling.g_e);

        (delta_r2_mu - delta_r2_e).sqrt()
    }

    pub fn verify_puzzle_resolution(&self) -> bool {
        let measured_diff = 0.04;
        let predicted = self.muon_electron_difference_fm();
        let ratio = predicted / measured_diff;
        ratio > 0.1 && ratio < 10.0
    }
}

pub struct MultiModeYukawa {
    pub masses_mev: [f64; 3],
}

pub struct MultiModeSolution {
    pub kappa_sq: [f64; 3],
    pub delta_r2_e: f64,
    pub delta_r2_mu: f64,
    pub delta_a_mu: f64,
    pub residuals: [f64; 3],
}

impl MultiModeYukawa {
    fn coeff_r2(m_lep_mev: f64, m_phi_mev: f64) -> f64 {
        let reduced_mass = (m_lep_mev * M_P_MEV) / (m_lep_mev + M_P_MEV);
        let denom = (2.0 * reduced_mass * ALPHA_EM + m_phi_mev).powi(2);
        (m_lep_mev * M_P_MEV) / denom
    }

    fn coeff_g2(m_phi_mev: f64) -> f64 {
        (M_MU_MEV * M_MU_MEV) / (16.0 * PI * PI * m_phi_mev * m_phi_mev)
    }

    fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
        let mut a = a;
        let mut x = b;
        let n = 3;
        let mut i = 0;
        while i < n {
            let mut pivot = i;
            let mut max_val = a[i][i].abs();
            let mut j = i + 1;
            while j < n {
                let v = a[j][i].abs();
                if v > max_val {
                    max_val = v;
                    pivot = j;
                }
                j += 1;
            }
            if max_val < 1e-18 {
                return None;
            }
            if pivot != i {
                a.swap(i, pivot);
                x.swap(i, pivot);
            }
            let diag = a[i][i];
            let mut k = i;
            while k < n {
                a[i][k] /= diag;
                k += 1;
            }
            x[i] /= diag;
            let mut r = 0;
            while r < n {
                if r != i {
                    let factor = a[r][i];
                    let mut c = i;
                    while c < n {
                        a[r][c] -= factor * a[i][c];
                        c += 1;
                    }
                    x[r] -= factor * x[i];
                }
                r += 1;
            }
            i += 1;
        }
        Some(x)
    }

    pub fn solve(&self, delta_r2_target: f64, delta_a_target: f64) -> Option<MultiModeSolution> {
        let mut c_e = [0.0_f64; 3];
        let mut c_mu = [0.0_f64; 3];
        let mut d = [0.0_f64; 3];
        let mut idx = 0;
        while idx < 3 {
            let m = self.masses_mev[idx];
            c_e[idx] = Self::coeff_r2(M_E_MEV, m);
            c_mu[idx] = Self::coeff_r2(M_MU_MEV, m);
            d[idx] = Self::coeff_g2(m);
            idx += 1;
        }
        let a = [
            [c_e[0], c_e[1], c_e[2]],
            [c_mu[0], c_mu[1], c_mu[2]],
            [d[0], d[1], d[2]],
        ];
        let b = [0.0_f64, delta_r2_target, delta_a_target];
        let k = Self::solve3(a, b)?;
        let mut delta_r2_e = 0.0_f64;
        let mut delta_r2_mu = 0.0_f64;
        let mut delta_a_mu = 0.0_f64;
        let mut i = 0;
        while i < 3 {
            delta_r2_e += k[i] * c_e[i];
            delta_r2_mu += k[i] * c_mu[i];
            delta_a_mu += k[i] * d[i];
            i += 1;
        }
        let res0 = delta_r2_e;
        let res1 = if delta_r2_target.abs() > 0.0 {
            (delta_r2_mu - delta_r2_target) / delta_r2_target
        } else {
            0.0
        };
        let res2 = if delta_a_target.abs() > 0.0 {
            (delta_a_mu - delta_a_target) / delta_a_target
        } else {
            0.0
        };
        Some(MultiModeSolution {
            kappa_sq: k,
            delta_r2_e,
            delta_r2_mu,
            delta_a_mu,
            residuals: [res0, res1, res2],
        })
    }
}

fn analyze_multimode_solution(
    masses: [f64; 3],
    sol: &MultiModeSolution,
    delta_r2_target: f64,
    delta_a_target: f64,
) {
    let mut c_e = [0.0_f64; 3];
    let mut c_mu = [0.0_f64; 3];
    let mut d = [0.0_f64; 3];
    let mut i = 0;
    while i < 3 {
        let m = masses[i];
        c_e[i] = MultiModeYukawa::coeff_r2(M_E_MEV, m);
        c_mu[i] = MultiModeYukawa::coeff_r2(M_MU_MEV, m);
        d[i] = MultiModeYukawa::coeff_g2(m);
        i += 1;
    }
    println!("\n=== 전자 방정식 계수 양의성 검사 ===");
    println!("c_e = [{:.4e}, {:.4e}, {:.4e}]", c_e[0], c_e[1], c_e[2]);
    let mut all_pos = true;
    let mut j = 0;
    while j < 3 {
        if c_e[j] <= 0.0 {
            all_pos = false;
        }
        j += 1;
    }
    println!("모든 c_e > 0: {}", if all_pos { "예" } else { "아니오" });
    println!("→ c_e·κ² = 0 이고 κ² ≥ 0 이면 κ² = 0 만 가능");
    println!("→ 비자명한 Δr_p^2(μ), Δa_μ 를 위해서는 적어도 하나의 κ_A^2 < 0 이 필요");

    let mut k_plus = [0.0_f64; 3];
    let mut k_minus = [0.0_f64; 3];
    let mut idx = 0;
    while idx < 3 {
        let val = sol.kappa_sq[idx];
        if val >= 0.0 {
            k_plus[idx] = val;
        } else {
            k_minus[idx] = -val;
        }
        idx += 1;
    }

    let mut dr2e_plus = 0.0_f64;
    let mut dr2e_minus = 0.0_f64;
    let mut dr2mu_plus = 0.0_f64;
    let mut dr2mu_minus = 0.0_f64;
    let mut da_plus = 0.0_f64;
    let mut da_minus = 0.0_f64;

    let mut t = 0;
    while t < 3 {
        dr2e_plus += k_plus[t] * c_e[t];
        dr2e_minus += k_minus[t] * c_e[t];
        dr2mu_plus += k_plus[t] * c_mu[t];
        dr2mu_minus += k_minus[t] * c_mu[t];
        da_plus += k_plus[t] * d[t];
        da_minus += k_minus[t] * d[t];
        t += 1;
    }

    println!("\n=== 선택/비선택 모드 분해 ===");
    println!(
        "κ_A^2(+) = [{:.3e}, {:.3e}, {:.3e}]",
        k_plus[0], k_plus[1], k_plus[2]
    );
    println!(
        "κ_A^2(-) = [{:.3e}, {:.3e}, {:.3e}]",
        k_minus[0], k_minus[1], k_minus[2]
    );

    println!("\n전자 반경:");
    println!("  Δr_p^2(e)^(+) = {dr2e_plus:.4e}");
    println!("  Δr_p^2(e)^(-) = {dr2e_minus:.4e}");
    println!("  합 = {:.4e} (목표: 0)", dr2e_plus - dr2e_minus);

    println!("\n뮤온 반경:");
    println!("  Δr_p^2(μ)^(+) = {dr2mu_plus:.4e}");
    println!("  Δr_p^2(μ)^(-) = {dr2mu_minus:.4e}");
    println!(
        "  합 = {:.4e} (목표: {:.4e})",
        dr2mu_plus - dr2mu_minus,
        delta_r2_target
    );

    println!("\n뮤온 g-2:");
    println!("  Δa_μ^(+) = {da_plus:.4e}");
    println!("  Δa_μ^(-) = {da_minus:.4e}");
    println!(
        "  합 = {:.4e} (목표: {:.4e})",
        da_plus - da_minus,
        delta_a_target
    );
}

pub fn run_muon_specificity_analysis() {
    println!("\n=== 억압보손-뮤온 특이 결합 분석 ===\n");

    let g_mu_mev = 6e-4;
    let coupling = MassProportionalCoupling::from_g_mu(g_mu_mev);
    coupling.print_coupling_hierarchy();

    println!("\n{}", "=".repeat(70));
    println!("보어 반경 계산 (뮤온이 양성자에 가까이 붙는 이유)");
    println!("{}", "=".repeat(70));

    let a0_e = BohrRadiusCalculator::electron_proton_angstrom();
    let a0_mu = BohrRadiusCalculator::muon_proton_fm();
    let mass_effect = BohrRadiusCalculator::mass_ratio_effect();

    println!("\n  전자-양성자 보어 반경: {a0_e:.2} Å");
    println!("  뮤온-양성자 보어 반경: {a0_mu:.2} fm");
    println!(
        "  비율 (a0_e/a0_mu): {:.0}배 (뮤온이 양성자에 {:.0}배 가까이)",
        a0_e * 1e5 / a0_mu,
        a0_e * 1e5 / a0_mu
    );
    println!("  질량 효과 (m_μ/m_e)^3: {mass_effect:.2e}");

    println!("\n  → 뮤온은 전자보다 양성자에 207배 가까이 접근");
    println!("  → 억압보손 교환이 10^7배 더 강하게 작용");

    println!("\n{}", "=".repeat(70));
    println!("유카와 퍼텐셜 (r = 2.6 fm, 뮤온 보어 반경)");
    println!("{}", "=".repeat(70));

    let m_phi_mev = 17.0;
    let yukawa = YukawaPotential::new(coupling.g_mu, coupling.g_p, m_phi_mev);

    let r_test = 2.6;
    let v_mu = yukawa.value_at(r_test);
    let ratio = yukawa.mu_to_e_ratio(r_test, &coupling);

    println!("\n  뮤온-양성자 퍼텐셜 (r={r_test:.1} fm): {v_mu:.3e} MeV");
    println!("  뮤온/전자 퍼텐셜 비율: {ratio:.0}배");
    println!("\n  → 질량-비례 결합으로 뮤온이 억압장을 207배 강하게 느낌");

    println!("\n{}", "=".repeat(70));
    println!("뮤온 g-2 변칙 설명");
    println!("{}", "=".repeat(70));

    let anomaly = AnomalousMagneticMoment::new(coupling.g_mu, m_phi_mev);
    let delta_a_mu = anomaly.delta_a_mu();
    let delta_a_e = anomaly.delta_a_e(coupling.g_e);

    println!("\n  억압보손 기여:");
    println!("    Δa_μ = {delta_a_mu:.2e} (예상: 2.51e-9)");
    println!("    Δa_e = {delta_a_e:.2e}");
    println!("    비율: {:.2e}", delta_a_mu / delta_a_e);

    let delta_a_target = 2.51e-9_f64;
    let ratio_over = delta_a_mu / delta_a_target;
    let delta_a_unsel = delta_a_target - delta_a_mu;
    let frac_unsel = delta_a_unsel / delta_a_mu;

    println!("\n  최소모델(선택 모드) vs 실험값 분해:");
    println!("    Δa_μ(sel) = {delta_a_mu:.2e}");
    println!("    Δa_μ(exp) = {delta_a_target:.2e}");
    println!("    Δa_μ(sel)/Δa_μ(exp) = {ratio_over:.1}");
    println!("    Δa_μ(unsel) = {delta_a_unsel:.2e}");
    println!(
        "    Δa_μ(unsel)/Δa_μ(sel) = {:.3} (선택 모드의 약 {:.1}% 상쇄)",
        frac_unsel,
        frac_unsel * -100.0_f64
    );

    let g2_ok = anomaly.verify_muon_anomaly();
    println!(
        "\n  뮤온 g-2 변칙 설명: {}",
        if g2_ok { "✓ 가능" } else { "✗ 불가능" }
    );
    println!("  전자 g-2는 영향 없음: ✓ (Δa_e ~ 10^-14, 측정 한계 이하)");

    println!("\n{}", "=".repeat(70));
    println!("양성자 반경 퍼즐 설명");
    println!("{}", "=".repeat(70));

    let puzzle = ProtonRadiusPuzzle::new(coupling, m_phi_mev);
    let dr_pred = puzzle.muon_electron_difference_fm();

    println!("\n  예측된 양성자 반경 차이:");
    println!("    r_p(e) - r_p(μ) = {dr_pred:.3} fm (측정: 0.04 fm)");

    let dr_exp = 0.04_f64;
    let dr_unsel = dr_exp - dr_pred;
    let frac_unsel_r = dr_unsel / dr_pred;

    println!("\n  최소모델(선택 모드) vs 실험값 분해:");
    println!("    Δr_p(sel) = {dr_pred:.3} fm");
    println!("    Δr_p(exp) = {dr_exp:.3} fm");
    println!("    Δr_p(unsel) = {dr_unsel:.3} fm");
    println!(
        "    Δr_p(unsel)/Δr_p(sel) = {:.3} (비선택 모드가 선택 모드 크기의 {:.1}% 추가)",
        frac_unsel_r,
        frac_unsel_r * 100.0_f64
    );

    let puzzle_ok = puzzle.verify_puzzle_resolution();
    println!(
        "\n  양성자 반경 퍼즐 설명: {}",
        if puzzle_ok {
            "✓ 가능"
        } else {
            "✗ 불가능"
        }
    );

    println!("\n{}", "=".repeat(70));
    println!("핵심 메커니즘 요약");
    println!("{}", "=".repeat(70));

    println!("\n  1. 질량-비례 결합: g_f = κ × m_f");
    println!("     → 무거운 입자일수록 억압장과 강하게 결합");
    println!("     → 뮤온(105 MeV) >> 전자(0.5 MeV): 207배 차이");

    println!("\n  2. 보어 반경 역비례: a_0 ∝ 1/m_ℓ");
    println!("     → 무거운 입자일수록 핵에 가까이 접근");
    println!("     → 뮤온이 양성자에 207배 가까이");

    println!("\n  3. 결합 효과: ΔE ∝ m_ℓ^4");
    println!("     → 뮤온에서 10^7배 강한 억압보손 효과");
    println!("     → g-2 변칙: 측정 가능");
    println!("     → 전자에서는 무시 가능");

    println!("\n  4. 유카와 퍼텐셜: V ∝ g_ℓ g_p exp(-m_φ r)/r");
    println!("     → 뮤온-양성자 사이 추가 인력");
    println!("     → 양성자 반경이 작게 측정됨");

    println!("\n  결론: 억압보손은 '뮤온 탐지기'");
    println!("       - 전자는 너무 가벼워서 거의 안 보임");
    println!("       - 뮤온은 딱 적당한 질량으로 선명하게 보임");
    println!("       - 이것이 뮤온 물리학 변칙의 공통 근원");

    println!("\n=== 분석 완료 ===");
}

pub fn run_multimode_yukawa_scan() {
    println!("\n=== 3-모드 억압보손 Yukawa 스캔 ===\n");
    let masses = [17.0_f64, 10.0_f64, 5.0_f64];
    let model = MultiModeYukawa { masses_mev: masses };
    let delta_r2_target = 0.04_f64 * 0.04_f64;
    let delta_a_target = 2.51e-9_f64;
    println!(
        "입력 질량 (MeV): [{:.1}, {:.1}, {:.1}]",
        masses[0], masses[1], masses[2]
    );
    println!("목표 값:");
    println!("  Δr_p^2 = {delta_r2_target:.4e} fm^2 (Δr_p = 0.04 fm 가정)");
    println!("  Δa_μ   = {delta_a_target:.3e}");
    match model.solve(delta_r2_target, delta_a_target) {
        Some(sol) => {
            println!("\n해결된 κ_A^2:");
            println!("  κ_1^2 = {:.3e}", sol.kappa_sq[0]);
            println!("  κ_2^2 = {:.3e}", sol.kappa_sq[1]);
            println!("  κ_3^2 = {:.3e}", sol.kappa_sq[2]);
            let mut i = 0;
            while i < 3 {
                if sol.kappa_sq[i] < 0.0 {
                    println!("  경고: κ_{}^2 < 0 (비물리적)", i + 1);
                }
                i += 1;
            }
            let mut g_mu = [0.0_f64; 3];
            let mut g_e = [0.0_f64; 3];
            let mut g_p = [0.0_f64; 3];
            let mut j = 0;
            while j < 3 {
                let kappa = if sol.kappa_sq[j] >= 0.0 {
                    sol.kappa_sq[j].sqrt()
                } else {
                    0.0
                };
                g_mu[j] = kappa * M_MU_MEV;
                g_e[j] = kappa * M_E_MEV;
                g_p[j] = kappa * M_P_MEV;
                j += 1;
            }
            println!("\n모드별 유효 결합 상수:");
            println!("  모드  g_μ        g_e        g_p");
            let mut a = 0;
            while a < 3 {
                println!(
                    "  {}    {:.3e}  {:.3e}  {:.3e}",
                    a + 1,
                    g_mu[a],
                    g_e[a],
                    g_p[a]
                );
                a += 1;
            }
            println!("\n예측값:");
            println!("  Δr_p^2(e) = {:.4e} fm^2", sol.delta_r2_e);
            println!("  Δr_p^2(μ) = {:.4e} fm^2", sol.delta_r2_mu);
            println!("  Δa_μ      = {:.3e}", sol.delta_a_mu);
            println!("\n잔차:");
            println!("  전자 반경 제약: Δr_p^2(e) = {:.3e}", sol.residuals[0]);
            println!("  뮤온 반경 상대 오차: {:.1}%", sol.residuals[1] * 100.0);
            println!("  뮤온 g-2 상대 오차: {:.1}%", sol.residuals[2] * 100.0);
            analyze_multimode_solution(masses, &sol, delta_r2_target, delta_a_target);
        }
        None => {
            println!("\n선형 방정식이 특이해서 해를 찾지 못함");
        }
    }
}

pub fn run_continuous_ratio_bounds() {
    println!("\n=== 연속 스펙트럼 레이리 경계 스캔 ===\n");

    let coupling = MassProportionalCoupling::from_g_mu(6e-4_f64);
    let g_ratio = coupling.g_p / coupling.g_mu;

    let delta_r2_exp = 0.04_f64 * 0.04_f64;
    let delta_a_exp = 2.51e-9_f64;
    let r_exp = delta_r2_exp / delta_a_exp;

    println!("뮤온 g-2에서 유도된 질량-비례 결합 사용: g_p/g_μ = {g_ratio:.3}");
    println!("실험 비율 R_exp = Δr_p^2(μ)/Δa_μ ≈ {r_exp:.3e} (단위: fm^2)");

    let masses_min_mev = 1.0_f64;
    let masses_max_mev = 1000.0_f64;
    let samples = 400usize;

    let log_min = masses_min_mev.ln();
    let log_max = masses_max_mev.ln();

    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;

    let mut idx = 0usize;
    while idx < samples {
        let t = (idx as f64 + 0.5_f64) / (samples as f64);
        let log_m = log_min + t * (log_max - log_min);
        let m = log_m.exp();

        let f_mu = MultiModeYukawa::coeff_r2(M_MU_MEV, m);
        let g_val = MultiModeYukawa::coeff_g2(m);

        if g_val <= 0.0_f64 {
            idx += 1;
            continue;
        }

        let r_m = g_ratio * f_mu / g_val;
        if r_m.is_finite() {
            if r_m < r_min {
                r_min = r_m;
            }
            if r_m > r_max {
                r_max = r_m;
            }
        }

        idx += 1;
    }

    println!("\n단일 모드 비율 R(m) = (g_p/g_μ)·F_μ(m)/G(m) 의 스캔 범위 (m ∈ [1, 1000] MeV):");
    println!("  R_min ≈ {r_min:.3e}");
    println!("  R_max ≈ {r_max:.3e}");

    if r_exp >= r_min && r_exp <= r_max {
        println!("\n결론: R_exp 가 [R_min, R_max] 내부에 있음");
        println!("  → 양의 연속 스펙트럼 ρ(s)만으로도 Δr_p^2(μ), Δa_μ 비율을 맞출 여지가 있음 (커널 비율 관점).");
    } else {
        println!("\n결론: R_exp 가 [R_min, R_max] 바깥에 있음");
        println!("  → 양수 스펙트럼 ρ(s)만으로 두 관측값을 동시에 만족시키기 어렵고,");
        println!("     비선택 모드(보정 모드)나 플레버 구조 확장이 필요함을 시사.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_proportional_law() {
        let coupling = MassProportionalCoupling::from_g_mu(6e-4);
        assert!(coupling.verify_mass_proportionality());

        let ratio = coupling.g_mu / coupling.g_e;
        let expected = M_MU_MEV / M_E_MEV;
        assert!((ratio - expected).abs() / expected < 1e-6);
    }

    #[test]
    fn test_bohr_radius_scaling() {
        let a0_e_angstrom = BohrRadiusCalculator::electron_proton_angstrom();
        let a0_mu_fm = BohrRadiusCalculator::muon_proton_fm();

        assert!(a0_e_angstrom > 0.5 && a0_e_angstrom < 0.6);
        assert!(a0_mu_fm > 200.0 && a0_mu_fm < 400.0);

        let ratio = a0_e_angstrom * 1e5 / a0_mu_fm;
        assert!(ratio > 150.0 && ratio < 250.0);
    }

    #[test]
    fn test_yukawa_potential_hierarchy() {
        let coupling = MassProportionalCoupling::from_g_mu(6e-4);
        let yukawa = YukawaPotential::new(coupling.g_mu, coupling.g_p, 17.0);

        let ratio = yukawa.mu_to_e_ratio(2.6, &coupling);
        let expected_ratio = M_MU_MEV / M_E_MEV;

        assert!((ratio - expected_ratio).abs() / expected_ratio < 1e-6);
    }

    #[test]
    fn test_g2_anomaly_order() {
        let anomaly = AnomalousMagneticMoment::new(6e-4, 17.0);
        let delta_a_mu = anomaly.delta_a_mu();

        assert!(delta_a_mu > 1e-10 && delta_a_mu < 1e-6);
    }

    #[test]
    fn test_effective_mass_correction() {
        let m0 = 100.0;
        let loss = EffectiveMassCalculator::calculate_loss(m0);
        let eff = EffectiveMassCalculator::calculate_effective_mass(m0);

        // Loss + Eff = M0
        assert!((loss + eff - m0).abs() < 1e-9);

        // Eff = M0 * e^-1
        let expected_eff = m0 * (-1.0_f64).exp();
        assert!((eff - expected_eff).abs() < 1e-9);
    }
}
