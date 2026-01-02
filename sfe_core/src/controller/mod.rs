use crate::engine::ibm_api::JobResult;
use crate::engine::noise::PinkNoiseGenerator;
use crate::engine::optimizer::SfeOptimizer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HardwareSpec {
    pub t1: f64, // us
    pub t2: f64, // us
    pub gate_err: f64,
    pub tls_freq: Option<f64>, // rad/us
    pub tls_amp: f64,          // rad/us
    pub drift_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ControlMode {
    Passive,
    Static,
    ANC,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlStrategy {
    pub mode: ControlMode,
    pub pulse_sequence: Vec<f64>,
    pub qec_distance: usize,
    pub expected_gain: f64,
}

pub struct SfeController {
    history: Vec<HardwareSpec>,
}

impl Default for SfeController {
    fn default() -> Self {
        Self::new()
    }
}

impl SfeController {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
        }
    }

    pub fn diagnose(&mut self, t1: f64, t2: f64, gate_err: f64) -> HardwareSpec {
        let rate_t2 = 1.0 / t2;
        let rate_t1_limit = 1.0 / (2.0 * t1);
        let pure_dephasing = rate_t2 - rate_t1_limit;

        let (tls_freq, tls_amp, drift) = if pure_dephasing > 0.01 {
            (Some(0.2713), (pure_dephasing).sqrt() * 2.0, 0.005)
        } else {
            (None, 0.0, 0.0)
        };

        let spec = HardwareSpec {
            t1,
            t2,
            gate_err,
            tls_freq,
            tls_amp,
            drift_rate: drift,
        };
        self.history.push(spec);
        spec
    }

    pub fn select_strategy(&self, spec: &HardwareSpec) -> ControlStrategy {
        let drift_penalty = spec.drift_rate * 100.0;
        let alpha_env: f64 = std::env::var("SFE_NOISE_ALPHA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.6);
        let scale_env: f64 = std::env::var("SFE_NOISE_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0e-15);
        let mut gain_static = if spec.tls_amp > 0.1 { 1.2 } else { 1.05 };
        if drift_penalty > 1.0 {
            gain_static *= 0.8;
        }

        let coherence_rich = spec.t1 > 100.0;
        let strong_tls = spec.tls_amp > 1.0;
        let gain_anc = if strong_tls && coherence_rich {
            3.5
        } else {
            1.1
        };

        let (mode, gain) = if gain_anc > gain_static && gain_anc > 1.3 {
            (ControlMode::ANC, gain_anc)
        } else if gain_static > 1.1 {
            (ControlMode::Static, gain_static)
        } else {
            (ControlMode::Passive, 1.0)
        };

        let p_base = 1e-3;
        let p_eff = p_base / gain;
        let qec_dist = if p_eff < 1e-4 {
            7
        } else if p_eff < 5e-4 {
            5
        } else {
            3
        };

        let seq = if mode == ControlMode::Static {
            // [Real-time Optimization Core]
            // 하드웨어 스펙에 맞춰 즉시 최적 시퀀스 생성
            let steps = 2000;
            let n_pulses = 8;
            let alpha = alpha_env;
            let noise_scale = scale_env;

            // 1. 노이즈 풀 생성 (쾌속 모드: 50 trials)
            let mut gen = PinkNoiseGenerator::new_with_params(steps, alpha, noise_scale);
            let trials = 50;
            let mut noise_pool = Vec::with_capacity(trials);
            for _ in 0..trials {
                noise_pool.push(gen.generate_new());
            }

            // 2. TLS Frequency 변환 (rad/us -> rad/step)
            // Target Time T=100us 가정 (광대역 커버)
            // w_sim = w_phys * (T / steps) = w_phys * (100/2000) = w_phys * 0.05
            let tls_omega_sim = spec.tls_freq.map(|f| f * 0.05);

            // 3. 최적화 수행 (Robust Mode)
            let optimizer = SfeOptimizer::new(50.0);
            optimizer.optimize_for_spec(steps, n_pulses, 0.15, &noise_pool, tls_omega_sim)
        } else {
            vec![]
        };

        ControlStrategy {
            mode,
            pulse_sequence: seq,
            qec_distance: qec_dist,
            expected_gain: gain,
        }
    }

    /// [3단계: 결과 분석]
    /// 수신된 JobResult를 분석하여 P(0) 및 개선 리포트 생성
    pub fn analyze_result(&self, result: &JobResult) {
        println!("\n[SFE-Rust] 결과 분석 리포트");
        println!("---------------------------------------------------");

        if let Some(dists) = &result.quasi_dists {
            let durations = [0, 2222, 4444, 6666, 8888, 11111, 13333, 15555, 17777, 20000];

            println!("{:<10} | {:<10}", "Duration", "P(|0>) Survival");
            println!("{:-<10}-|-{:-<10}", "-", "-");

            let mut p0_values = Vec::new();

            for (i, dist) in dists.iter().enumerate() {
                if i >= durations.len() {
                    break;
                }
                // quasi_dist: {"0": 0.9, "1": 0.1} or similar
                // serde_json Value에서 "0" 키의 값을 추출
                let p0 = dist.get("0").and_then(|v| v.as_f64()).unwrap_or(0.0);
                p0_values.push(p0);

                println!("{:<10} | {:.4}", durations[i], p0);
            }

            // 간단한 통계
            let avg_p0: f64 = p0_values.iter().sum::<f64>() / p0_values.len() as f64;
            println!("---------------------------------------------------");
            println!("평균 생존 확률 (Avg P0): {avg_p0:.4}");

            if avg_p0 > 0.7 {
                println!(">> 상태: 좋음 (Good Coherence)");
            } else if avg_p0 > 0.5 {
                println!(">> 상태: 보통 (Degraded)");
            } else {
                println!(">> 상태: 나쁨 (Decohered)");
            }
        } else {
            println!("데이터 없음 (No quasi-dists found)");
        }
    }
}
