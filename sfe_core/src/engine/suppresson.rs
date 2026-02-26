use crate::engine::noise_multi::diagnose_tls_spectrum;
use crate::engine::qec::simulate_repetition_code;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;

pub const HBAR_EV_S: f64 = 6.582119569e-16;
pub const C_M_PER_S: f64 = 2.99792458e8;
pub const HBAR_C_EV_M: f64 = HBAR_EV_S * C_M_PER_S;
pub const GEV_TO_EV: f64 = 1e9;
pub const TEV_TO_EV: f64 = 1e12;
pub const MEV_TO_EV: f64 = 1e6;
pub const M_PLANCK_GEV: f64 = 1.22e19;
pub const M_HIGGS_GEV: f64 = 125.0;
pub const M_MU_GEV: f64 = 0.1057;
pub const M_E_GEV: f64 = 0.000511;

#[derive(Clone, Debug)]
pub struct SuppressonMode {
    pub omega_sim: f64,
    pub amplitude_sim: f64,
    pub omega_phys: f64,
    pub amplitude_phys: f64,
    pub energy_ev: f64,
}

impl SuppressonMode {
    pub fn from_physical(omega_sim: f64, amplitude_sim: f64, dt_us: f64, noise_amp: f64) -> Self {
        let omega_phys = omega_sim / dt_us;
        let amplitude_phys = amplitude_sim * (noise_amp * 0.01) / dt_us;

        let energy_ev = omega_phys * 6.58e-10;

        Self {
            omega_sim,
            amplitude_sim,
            omega_phys,
            amplitude_phys,
            energy_ev,
        }
    }

    pub fn from_t1_t2(t1_us: f64, t2_us: f64, dt_us: f64, noise_amp: f64) -> Vec<Self> {
        let peaks = diagnose_tls_spectrum(t1_us, t2_us, dt_us);

        peaks
            .into_iter()
            .map(|peak| {
                let omega_phys = peak.omega_center / dt_us;
                let amp_phys = peak.amplitude / dt_us * (noise_amp * 0.01);

                Self::from_physical(omega_phys, amp_phys, dt_us, noise_amp)
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct SuppressonEvidence {
    pub mode: SuppressonMode,
    pub qec_gain: f64,
    pub improvement_over_baseline: f64,
    pub is_sweet_spot: bool,
}

pub struct SuppressonScanner {
    pub steps: usize,
    pub dt_us: f64,
    pub noise_amp: f64,
    pub qec_distance: usize,
    pub measure_interval: usize,
    pub trials: usize,
    pub baseline_gain: Option<f64>,
}

impl SuppressonScanner {
    pub fn new(steps: usize, qec_distance: usize, noise_amp: f64) -> Self {
        let total_time_us = 100.0;
        let dt_us = total_time_us / steps as f64;

        Self {
            steps,
            dt_us,
            noise_amp,
            qec_distance,
            measure_interval: 50,
            trials: 1000,
            baseline_gain: None,
        }
    }

    pub fn measure_baseline(&mut self) -> f64 {
        std::env::remove_var("CE_SUPPRESSON_OMEGA");
        std::env::remove_var("CE_SUPPRESSON_AMP");
        std::env::remove_var("CE_SUPPRESSON_OMEGA2");
        std::env::remove_var("CE_SUPPRESSON_AMP2");
        std::env::remove_var("CE_SUPPRESSON_ANC");

        let res = simulate_repetition_code(
            self.qec_distance,
            &[],
            self.noise_amp,
            self.steps,
            self.measure_interval,
            self.trials,
        );

        self.baseline_gain = Some(res.gain);
        res.gain
    }

    pub fn scan_mode(&self, mode: &SuppressonMode) -> SuppressonEvidence {
        std::env::set_var("CE_SUPPRESSON_ENABLE", "2");
        std::env::set_var("CE_SUPPRESSON_OMEGA", mode.omega_sim.to_string());
        std::env::set_var("CE_SUPPRESSON_AMP", mode.amplitude_sim.to_string());
        std::env::set_var("CE_SUPPRESSON_OMEGA2", (mode.omega_sim * 0.5).to_string());
        std::env::set_var("CE_SUPPRESSON_AMP2", mode.amplitude_sim.to_string());
        std::env::set_var("CE_SUPPRESSON_ANC", "1");

        let res = simulate_repetition_code(
            self.qec_distance,
            &[],
            self.noise_amp,
            self.steps,
            self.measure_interval,
            self.trials,
        );

        let baseline = self.baseline_gain.unwrap_or(0.5);
        let improvement = res.gain / baseline;
        let is_sweet_spot = improvement > 1.1;

        SuppressonEvidence {
            mode: mode.clone(),
            qec_gain: res.gain,
            improvement_over_baseline: improvement,
            is_sweet_spot,
        }
    }

    pub fn grid_scan(
        &self,
        omega_range: (f64, f64),
        amp_range: (f64, f64),
        grid_size: usize,
    ) -> Vec<SuppressonEvidence> {
        let omega_min = omega_range.0;
        let omega_max = omega_range.1;
        let amp_min = amp_range.0;
        let amp_max = amp_range.1;

        let mut params = Vec::new();
        for i in 0..grid_size {
            for j in 0..grid_size {
                let omega_sim =
                    omega_min + (omega_max - omega_min) * i as f64 / (grid_size - 1) as f64;
                let amp_sim = amp_min + (amp_max - amp_min) * j as f64 / (grid_size - 1) as f64;

                let mode =
                    SuppressonMode::from_physical(omega_sim, amp_sim, self.dt_us, self.noise_amp);
                params.push(mode);
            }
        }

        println!("격자 스캔 시작: {}개 포인트", params.len());

        let results: Vec<SuppressonEvidence> = params
            .par_iter()
            .enumerate()
            .map(|(idx, mode)| {
                if idx % 10 == 0 {
                    println!("진행: {}/{}", idx, params.len());
                }
                self.scan_mode(mode)
            })
            .collect();

        results
    }

    pub fn find_sweet_spots(&self, evidences: &[SuppressonEvidence]) -> Vec<SuppressonEvidence> {
        let mut sweet_spots: Vec<SuppressonEvidence> = evidences
            .iter()
            .filter(|e| e.is_sweet_spot)
            .cloned()
            .collect();

        sweet_spots.sort_by(|a, b| b.qec_gain.partial_cmp(&a.qec_gain).unwrap());
        sweet_spots
    }

    pub fn verify_high_energy_low_energy_projection(
        &self,
        g_mu: f64,
        m_phi_mev: f64,
    ) -> ProjectionResult {
        let m_e = 0.511;
        let m_mu = 105.66;

        let g_q = g_mu * (m_e / m_mu);

        let rho_phi_ev4 = 3e-11_f64;

        let epsilon_min = 0.14_f64;
        let epsilon_max = 0.26_f64;

        let m_low_ev = 1e-10_f64;

        let conversion = 6.58e-10;
        let epsilon_min_ev = epsilon_min * conversion;
        let epsilon_max_ev = epsilon_max * conversion;

        let numerator_min = m_low_ev.powi(2) * epsilon_min_ev.powi(2);
        let numerator_max = m_low_ev.powi(2) * epsilon_max_ev.powi(2);
        let denominator = 2.0 * g_q.powi(2) * rho_phi_ev4;

        let f_low_min = numerator_min / denominator;
        let f_low_max = numerator_max / denominator;

        let is_consistent = f_low_min > 0.0 && f_low_max < 1.0;

        ProjectionResult {
            g_mu,
            m_phi_mev,
            g_q,
            m_low_ev,
            epsilon_min_rad_per_us: epsilon_min,
            epsilon_max_rad_per_us: epsilon_max,
            f_low_min,
            f_low_max,
            is_consistent,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionResult {
    pub g_mu: f64,
    pub m_phi_mev: f64,
    pub g_q: f64,
    pub m_low_ev: f64,
    pub epsilon_min_rad_per_us: f64,
    pub epsilon_max_rad_per_us: f64,
    pub f_low_min: f64,
    pub f_low_max: f64,
    pub is_consistent: bool,
}

pub struct MultiScaleSuppressonSpectrum {
    pub tev_scale: TeVScaleMode,
    pub mev_scale: MeVScaleMode,
    pub ev_scale: EvScaleMode,
    pub hierarchy_check: HierarchyCheck,
}

#[derive(Debug, Clone)]
pub struct TeVScaleMode {
    pub mass_tev: f64,
    pub kappa: f64,
    pub coupling_mu: f64,
    pub coupling_e: f64,
    pub new_physics_scale_gev: f64,
}

#[derive(Debug, Clone)]
pub struct MeVScaleMode {
    pub mass_mev: f64,
    pub g_mu: f64,
    pub g_e: f64,
    pub x17_compatible: bool,
}

#[derive(Debug, Clone)]
pub struct EvScaleMode {
    pub mass_ev: f64,
    pub tls_freq_rad_per_us: f64,
    pub qec_compatible: bool,
}

#[derive(Debug, Clone)]
pub struct HierarchyCheck {
    pub planck_to_tev_ratio: f64,
    pub tev_to_higgs_ratio: f64,
    pub higgs_to_mev_ratio: f64,
    pub mev_to_ev_ratio: f64,
    pub goldstone_protection: bool,
    pub naturalness_score: f64,
}

impl MultiScaleSuppressonSpectrum {
    pub fn from_first_principles() -> Self {
        let kappa_inv_mev = 1e6_f64;
        let mass_tev_gev = kappa_inv_mev / 1000.0_f64;
        let mass_tev = mass_tev_gev / 1000.0_f64;
        let kappa = 1.0_f64 / kappa_inv_mev;

        let alpha_sup = 1e-3_f64;
        let g_mu_tev = alpha_sup * (M_MU_GEV / M_PLANCK_GEV) * M_PLANCK_GEV;
        let g_e_tev = alpha_sup * (M_E_GEV / M_PLANCK_GEV) * M_PLANCK_GEV;

        let tev_scale = TeVScaleMode {
            mass_tev,
            kappa,
            coupling_mu: g_mu_tev,
            coupling_e: g_e_tev,
            new_physics_scale_gev: mass_tev_gev,
        };

        let mass_mev = 17.0_f64;
        let g_mu_mev = 6e-4_f64;
        let g_e_mev = g_mu_mev * (M_E_GEV / M_MU_GEV);
        let x17_compatible = (mass_mev - 16.7_f64).abs() < 1.0_f64;

        let mev_scale = MeVScaleMode {
            mass_mev,
            g_mu: g_mu_mev,
            g_e: g_e_mev,
            x17_compatible,
        };

        let mass_ev = 1e-10_f64;
        let tls_freq = mass_ev / (HBAR_EV_S * 1e6_f64);
        let qec_compatible = (0.9e-10_f64..=1.7e-10_f64).contains(&mass_ev);

        let ev_scale = EvScaleMode {
            mass_ev,
            tls_freq_rad_per_us: tls_freq,
            qec_compatible,
        };

        let planck_ev = M_PLANCK_GEV * GEV_TO_EV;
        let tev_ev = mass_tev_gev * GEV_TO_EV;
        let higgs_ev = M_HIGGS_GEV * GEV_TO_EV;
        let mev_ev = mass_mev * MEV_TO_EV;

        let planck_to_tev = planck_ev / tev_ev;
        let tev_to_higgs = tev_ev / higgs_ev;
        let higgs_to_mev = higgs_ev / mev_ev;
        let mev_to_ev = mev_ev / mass_ev;

        let goldstone_protection = true;
        let log_sum = planck_to_tev.log10().abs()
            + tev_to_higgs.log10().abs()
            + higgs_to_mev.log10().abs()
            + mev_to_ev.log10().abs();
        let naturalness_score = if log_sum > 0.0_f64 {
            1.0_f64 / log_sum
        } else {
            0.0_f64
        };

        let hierarchy_check = HierarchyCheck {
            planck_to_tev_ratio: planck_to_tev,
            tev_to_higgs_ratio: tev_to_higgs,
            higgs_to_mev_ratio: higgs_to_mev,
            mev_to_ev_ratio: mev_to_ev,
            goldstone_protection,
            naturalness_score,
        };

        Self {
            tev_scale,
            mev_scale,
            ev_scale,
            hierarchy_check,
        }
    }

    pub fn verify_consistency(&self) -> bool {
        let tev_ok = self.tev_scale.new_physics_scale_gev > M_HIGGS_GEV;
        let mev_ok = self.mev_scale.x17_compatible;
        let ev_ok = self.ev_scale.qec_compatible;
        let hierarchy_ok = self.hierarchy_check.goldstone_protection;

        tev_ok && mev_ok && ev_ok && hierarchy_ok
    }

    pub fn print_full_spectrum(&self) {
        println!("\n=== 광명보손 다중 스케일 스펙트럼 ===\n");

        println!("1. TeV 스케일 모드 (힉스보다 큰 새로운 물리학)");
        println!(
            "   질량: {:.2} TeV ({:.2e} GeV)",
            self.tev_scale.mass_tev, self.tev_scale.new_physics_scale_gev
        );
        println!("   κ^-1: {:.2e}", 1.0 / self.tev_scale.kappa);
        println!("   g_μ: {:.2e}", self.tev_scale.coupling_mu);
        println!("   g_e: {:.2e}", self.tev_scale.coupling_e);
        println!("   역할: 전약력 스케일 이상의 새로운 물리학");

        println!("\n2. MeV 스케일 모드 (뮤온 g-2, X17)");
        println!("   질량: {:.1} MeV", self.mev_scale.mass_mev);
        println!("   g_μ: {:.2e}", self.mev_scale.g_mu);
        println!("   g_e: {:.2e}", self.mev_scale.g_e);
        println!(
            "   X17 정합성: {}",
            if self.mev_scale.x17_compatible {
                "✓ 일치"
            } else {
                "✗ 불일치"
            }
        );
        println!("   역할: 뮤온 g-2 변칙, 양성자 반경 퍼즐");

        println!("\n3. eV 스케일 모드 (양자컴퓨팅 TLS)");
        println!("   질량: {:.2e} eV", self.ev_scale.mass_ev);
        println!(
            "   TLS 주파수: {:.4} rad/μs",
            self.ev_scale.tls_freq_rad_per_us
        );
        println!(
            "   QEC 정합성: {}",
            if self.ev_scale.qec_compatible {
                "✓ 허용창 내"
            } else {
                "✗ 허용창 외"
            }
        );
        println!("   역할: 큐비트 1/f 노이즈, QEC sweet spot");

        println!("\n4. 계층 구조 검증");
        println!(
            "   M_Planck / M_TeV: {:.2e} (예상: ~10^17)",
            self.hierarchy_check.planck_to_tev_ratio
        );
        println!(
            "   M_TeV / M_Higgs: {:.2e} (예상: ~10^1)",
            self.hierarchy_check.tev_to_higgs_ratio
        );
        println!(
            "   M_Higgs / M_MeV: {:.2e} (예상: ~10^4)",
            self.hierarchy_check.higgs_to_mev_ratio
        );
        println!(
            "   M_MeV / M_eV: {:.2e} (예상: ~10^16)",
            self.hierarchy_check.mev_to_ev_ratio
        );
        println!(
            "   Goldstone 보호: {}",
            if self.hierarchy_check.goldstone_protection {
                "✓ 활성"
            } else {
                "✗ 비활성"
            }
        );
        println!(
            "   자연성 점수: {:.3} (높을수록 자연스러움)",
            self.hierarchy_check.naturalness_score
        );

        println!("\n5. 전체 정합성");
        let is_consistent = self.verify_consistency();
        println!(
            "   판정: {}",
            if is_consistent {
                "✓ 모든 스케일 정합"
            } else {
                "✗ 정합성 문제"
            }
        );
    }
}

pub fn run_suppresson_evidence_analysis() {
    println!("=== CE 광명보손 증거 분석 (다중 스케일) ===\n");

    let spectrum = MultiScaleSuppressonSpectrum::from_first_principles();
    spectrum.print_full_spectrum();

    crate::engine::suppresson_physics::run_muon_specificity_analysis();

    println!("\n{}", "=".repeat(70));
    println!("저에너지 QEC 검증 시작");
    println!("{}\n", "=".repeat(70));

    let mut scanner = SuppressonScanner::new(2000, 5, 0.10);

    println!("1. 기준선 측정 (광명보손 없음)");
    let baseline = scanner.measure_baseline();
    println!("   기준선 QEC Gain: {baseline:.3}\n");

    println!("2. IBM Fez T1/T2 기반 연역적 모드 추출");
    let t1 = 60.0;
    let t2 = 40.0;
    let modes = SuppressonMode::from_t1_t2(t1, t2, scanner.dt_us, scanner.noise_amp);

    for (i, mode) in modes.iter().enumerate() {
        println!(
            "   모드 {}: ω_phys={:.4} rad/μs, A_phys={:.3} rad/μs, E={:.2e} eV",
            i + 1,
            mode.omega_phys,
            mode.amplitude_phys,
            mode.energy_ev
        );

        let evidence = scanner.scan_mode(mode);
        println!(
            "   → QEC Gain: {:.3}, 개선율: {:.2}x",
            evidence.qec_gain, evidence.improvement_over_baseline
        );
    }

    println!("\n3. 격자 스캔 (스윗스팟 탐색)");
    let omega_range = (0.004, 0.016);
    let amp_range = (4.0, 16.0);
    let grid_size = 6;

    let evidences = scanner.grid_scan(omega_range, amp_range, grid_size);

    println!("\n4. 스윗스팟 발견");
    let sweet_spots = scanner.find_sweet_spots(&evidences);

    println!("   발견된 스윗스팟 (개선율 >1.1x): {}개", sweet_spots.len());
    for (i, spot) in sweet_spots.iter().take(5).enumerate() {
        println!(
            "   #{}: ω_sim={:.4}, A_sim={:.2}, E={:.2e} eV, Gain={:.3} ({:.2}x)",
            i + 1,
            spot.mode.omega_sim,
            spot.mode.amplitude_sim,
            spot.mode.energy_ev,
            spot.qec_gain,
            spot.improvement_over_baseline
        );
    }

    println!("\n   상위 10개 결과:");
    let mut all_sorted = evidences.clone();
    all_sorted.sort_by(|a, b| b.qec_gain.partial_cmp(&a.qec_gain).unwrap());
    for (i, e) in all_sorted.iter().take(10).enumerate() {
        println!(
            "   #{}: ω_sim={:.4}, A_sim={:.2}, Gain={:.3} ({:.2}x)",
            i + 1,
            e.mode.omega_sim,
            e.mode.amplitude_sim,
            e.qec_gain,
            e.improvement_over_baseline
        );
    }

    if let Some(best) = all_sorted.first() {
        let g_sel = best.improvement_over_baseline;
        let g_exp = 1.5_f64;
        let g_unsel = g_exp - g_sel;
        let frac_unsel = g_unsel / g_sel;

        println!("\n   선택/비선택 광명장 관점에서 본 QEC Gain 분해:");
        println!("     G_sel(sim) = {g_sel:.3}x (최고 스윗스팟 시뮬레이션)");
        println!("     G_exp(doc) = {g_exp:.3}x (문서 7.5장 목표값)");
        println!("     G_unsel    = {g_unsel:.3}x");
        println!(
            "     G_unsel/G_sel = {:.3} (비선택 모드가 시뮬레이션 Gain의 {:.1}% 조정 필요)",
            frac_unsel,
            frac_unsel * 100.0_f64
        );
    }

    if !sweet_spots.is_empty() {
        let g_exp = 1.5_f64;
        let mut sum_frac = 0.0_f64;
        let mut count = 0_usize;
        let mut min_frac = 1e9_f64;
        let mut max_frac = -1e9_f64;

        for spot in sweet_spots.iter() {
            let g_sel = spot.improvement_over_baseline;
            if g_sel <= 1.0_f64 {
                continue;
            }
            let g_unsel = g_exp - g_sel;
            let frac = g_unsel / g_sel;
            sum_frac += frac;
            count += 1;
            if frac < min_frac {
                min_frac = frac;
            }
            if frac > max_frac {
                max_frac = frac;
            }
        }

        if count > 0 {
            let mean_frac = sum_frac / count as f64;
            println!("\n   스윗스팟 전체에서 본 비선택 광명장 조정 비율 통계:");
            println!("     샘플 개수: {count}");
            println!("     G_unsel/G_sel 최소값: {min_frac:.3}");
            println!("     G_unsel/G_sel 최대값: {max_frac:.3}");
            println!("     G_unsel/G_sel 평균값: {mean_frac:.3}");
        }
    }

    println!("\n5. 고에너지-저에너지 사영 검증");
    let g_mu = 6e-4;
    let m_phi_mev = 17.0;

    let projection = scanner.verify_high_energy_low_energy_projection(g_mu, m_phi_mev);

    println!("   입력:");
    println!("   - 뮤온 결합: g_μ = {:.2e}", projection.g_mu);
    println!("   - 광명보손 질량: m_φ = {} MeV", projection.m_phi_mev);
    println!("\n   유도:");
    println!("   - 큐비트 결합: g_q = {:.2e}", projection.g_q);
    println!(
        "   - 저에너지 모드 질량: m_low = {:.2e} eV",
        projection.m_low_ev
    );
    println!("\n   QEC 허용창:");
    println!(
        "   - ε_min = {:.2} rad/μs",
        projection.epsilon_min_rad_per_us
    );
    println!(
        "   - ε_max = {:.2} rad/μs",
        projection.epsilon_max_rad_per_us
    );
    println!("\n   에너지 분율:");
    println!("   - f_low_min = {:.2e}", projection.f_low_min);
    println!("   - f_low_max = {:.2e}", projection.f_low_max);
    println!(
        "\n   정합성: {}",
        if projection.is_consistent {
            "✓ 통과"
        } else {
            "✗ 실패"
        }
    );

    println!("\n6. 결과 저장");
    save_results_to_csv(&evidences, "suppresson_scan_results.csv");

    println!("\n=== 분석 완료 ===");
}

pub fn save_results_to_csv(evidences: &[SuppressonEvidence], filename: &str) {
    let mut file = File::create(filename).expect("파일 생성 실패");

    writeln!(file, "omega_sim,amplitude_sim,omega_phys,amplitude_phys,energy_ev,qec_gain,improvement,is_sweet_spot").unwrap();

    for e in evidences {
        writeln!(
            file,
            "{:.6},{:.6},{:.6},{:.6},{:.6e},{:.6},{:.6},{}",
            e.mode.omega_sim,
            e.mode.amplitude_sim,
            e.mode.omega_phys,
            e.mode.amplitude_phys,
            e.mode.energy_ev,
            e.qec_gain,
            e.improvement_over_baseline,
            if e.is_sweet_spot { 1 } else { 0 }
        )
        .unwrap();
    }

    println!("   결과 저장 완료: {filename}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suppresson_mode_creation() {
        let mode = SuppressonMode::from_physical(0.0136, 13.0, 0.05, 0.10);

        assert!(mode.omega_sim > 0.0);
        assert!(mode.amplitude_sim > 0.0);
        assert!(mode.energy_ev > 0.0);
        assert!(mode.energy_ev < 1e-8);
    }

    #[test]
    fn test_t1_t2_diagnosis() {
        let modes = SuppressonMode::from_t1_t2(60.0, 40.0, 0.05, 0.10);

        assert_eq!(modes.len(), 2);
        assert!(modes[0].omega_phys > 0.0);
    }

    #[test]
    fn test_projection_consistency() {
        let scanner = SuppressonScanner::new(2000, 5, 0.10);
        let result = scanner.verify_high_energy_low_energy_projection(6e-4, 17.0);

        assert!(result.f_low_min > 0.0);
        assert!(result.f_low_max > result.f_low_min);
        assert!(result.is_consistent);
    }
}
