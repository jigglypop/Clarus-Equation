use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use sfe_core::engine::core::QCEngine;
use sfe_core::engine::geometric::GeometricEngine;
use sfe_core::engine::manifold::Manifold; // Trait import added
use sfe_core::engine::qec::CE_PHASE_SCALE;
use sfe_core::{run_pulse_optimizer, run_sweep_benchmark, IbmClient, CeController};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// CE 광명장 동역학 시뮬레이션
    Dynamics {
        #[arg(short, long, default_value_t = 100_000)]
        size: usize,
        #[arg(short, long, default_value_t = 5_000)]
        steps: usize,
        #[arg(short, long, default_value = "ce_output.csv")]
        output: String,
    },
    /// [NEW] Part8 기하학적 통합 방정식 시뮬레이션
    GeometricDynamics {
        #[arg(short, long, default_value_t = 2)]
        dim: usize,
        #[arg(short, long, default_value_t = 1000)]
        steps: usize,
        #[arg(short, long, default_value = "geometric_output.csv")]
        output: String,
    },
    /// [NEW] SCQE(자가보정 양자소자) 작동 시뮬레이션
    ScqeSimulation {
        #[arg(short, long, default_value_t = 1000)]
        steps: usize,
        #[arg(short, long, default_value = "scqe_result.csv")]
        output: String,
    },
    /// [NEW] CE-ARC 노이즈 캔슬러 시뮬레이션
    CeArc {
        #[arg(short, long, default_value_t = 1000)]
        steps: usize,
        #[arg(short, long, default_value = "ce_arc_result.csv")]
        output: String,
        #[arg(long)]
        diag: bool,
    },
    /// [NEW] CE 금융 시장 시뮬레이션 (Econophysics)
    MarketSimulation {
        #[arg(short, long, default_value_t = 5000)]
        steps: usize,
        #[arg(short, long, default_value = "market_sim.csv")]
        output: String,
    },
    /// [NEW] CE 거시경제 위기 예측 시뮬레이션
    MacroCrisis {
        #[arg(short, long, default_value_t = 100)]
        years: usize,
        #[arg(short, long, default_value = "macro_crisis.csv")]
        output: String,
    },
    /// [Deprecated] 양자 노이즈 시뮬레이션 (Sweep 사용 권장)
    QuantumNoise {
        #[arg(short, long, default_value_t = 10_000)]
        steps: usize,
        #[arg(short, long, default_value_t = 1000)]
        trials: usize,
        #[arg(short, long, default_value = "quantum_noise.csv")]
        output: String,
    },
    /// [Deprecated] 디커플링 벤치마크 (Sweep 사용 권장)
    DecouplingBenchmark {
        #[arg(short, long, default_value_t = 10_000)]
        steps: usize,
        #[arg(short, long, default_value_t = 1000)]
        trials: usize,
        #[arg(short, long, default_value = "decoupling_result.csv")]
        output: String,
    },
    /// CE 펄스 최적화기 (Pure Analytic Formula)
    PulseOptimizer {
        #[arg(short, long, default_value_t = 2000)]
        steps: usize,
        #[arg(short, long, default_value_t = 50)]
        pulses: usize,
        #[arg(short, long, default_value_t = 50)]
        generations: usize, // 이제 무시됩니다 (순수 공식 사용)
    },
    /// [NEW] 포괄적 파라미터 스윕 (히트맵 데이터 생성)
    Sweep {
        #[arg(short, long, default_value = "sweep_results.csv")]
        output: String,
    },
    Qec {
        #[arg(short, long, default_value_t = 3)]
        distance: usize,
        #[arg(short, long, default_value_t = 0.10)]
        noise: f64,
    },
    Surface {
        #[arg(short, long, default_value_t = 0.10)]
        noise: f64,
    },
    /// [NEW] CE 상용 컨트롤러 (실시간 고속 진단/제어)
    RunController {
        #[arg(long)]
        t1: f64,
        #[arg(long)]
        t2: f64,
        #[arg(long, default_value_t = 0.001)]
        gate_err: f64,
    },
    /// [NEW] IBM Quantum 하드웨어 연결
    RunIBM {
        #[arg(short, long)]
        api_key: String,
    },
    /// [NEW] 광명보손 증거 스캔 및 분석
    SuppressonScan,
    /// [NEW] 3-모드 광명보손 Yukawa 스캔
    MultiModeScan,
    /// [NEW] 연속 스펙트럼 레이리 경계 스캔
    ContinuousBounds,
}

fn configure_fez_suppresson(noise: f64, steps: usize) {
    if noise <= 0.0 {
        std::env::remove_var("CE_SUPPRESSON_OMEGA");
        std::env::remove_var("CE_SUPPRESSON_AMP");
        std::env::remove_var("CE_SUPPRESSON_OMEGA2");
        std::env::remove_var("CE_SUPPRESSON_AMP2");
        return;
    }
    let mut controller = CeController::new();
    let t1 = 60.0_f64;
    let t2 = 40.0_f64;
    let gate_err = 1.0e-3_f64;
    let spec = controller.diagnose(t1, t2, gate_err);
    let tls_freq = match spec.tls_freq {
        Some(v) => v,
        None => {
            std::env::remove_var("CE_SUPPRESSON_OMEGA");
            std::env::remove_var("CE_SUPPRESSON_AMP");
            std::env::remove_var("CE_SUPPRESSON_OMEGA2");
            std::env::remove_var("CE_SUPPRESSON_AMP2");
            return;
        }
    };
    let total_time_us = 100.0_f64;
    let dt_us = total_time_us / steps as f64;
    let omega_top = tls_freq * dt_us;
    let amp_top = spec.tls_amp * dt_us / (noise.abs() * CE_PHASE_SCALE);
    let omega_bottom = 0.5_f64 * omega_top;
    let amp_bottom = amp_top;
    std::env::set_var("CE_SUPPRESSON_OMEGA", omega_top.to_string());
    std::env::set_var("CE_SUPPRESSON_AMP", amp_top.to_string());
    std::env::set_var("CE_SUPPRESSON_OMEGA2", omega_bottom.to_string());
    std::env::set_var("CE_SUPPRESSON_AMP2", amp_bottom.to_string());
}

fn main() {
    let args = Args::parse();
    println!("==========================================");
    println!("   CE 상용 엔진 v1.9 (Crisis Sim)        ");
    println!("==========================================");

    let start_time = Instant::now();

    match args.command {
        Commands::Dynamics {
            size,
            steps,
            output,
        } => {
            println!("모드: CE 광명장 동역학 시뮬레이션");
            let mut engine = QCEngine::new(size);
            let pb = ProgressBar::new(steps as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} {bar:40} {pos}/{len}")
                    .unwrap(),
            );
            let mut history = Vec::with_capacity(steps);
            for t in 0..steps {
                engine.step();
                if t % 10 == 0 {
                    history.push((t, engine.get_center_value()));
                }
                pb.inc(1);
            }
            pb.finish();
            let mut file = File::create(&output).unwrap();
            writeln!(file, "TimeStep,CenterPhi").unwrap();
            for (t, v) in history {
                writeln!(file, "{t},{v}").unwrap();
            }
        }
        Commands::GeometricDynamics { dim, steps, output } => {
            println!("모드: CE Part8 기하학적 통합 시뮬레이션");
            println!("  차원: {dim}, 스텝: {steps}");
            let engine = GeometricEngine::new(dim);

            let mut x = vec![1.0; dim];
            let mut p = vec![-1.0; dim]; // 원점으로의 복원력
            let dt = 0.01;

            let pb = ProgressBar::new(steps as u64);
            let mut file = File::create(&output).unwrap();
            writeln!(file, "Step,R,Suppression,X0,X1").unwrap();

            for t in 0..steps {
                let (x_next, supp) = engine.step(&x, &p, dt);
                let r = engine.manifold.ricci_scalar(&x);

                writeln!(
                    file,
                    "{},{:.6},{:.6},{:.6},{:.6}",
                    t,
                    r,
                    supp,
                    x[0],
                    x.get(1).unwrap_or(&0.0)
                )
                .unwrap();

                x = x_next;
                // 단순 조화 진동자 운동량 업데이트 (p_new = p - k*x*dt)
                for i in 0..dim {
                    p[i] -= x[i] * dt;
                }

                pb.inc(1);
            }
            pb.finish();
            println!("결과 저장됨: {output}");
        }
        Commands::ScqeSimulation { steps, output } => {
            println!("모드: SCQE(자가보정 양자소자) 시뮬레이션");
            let engine = GeometricEngine::new(2);
            println!("  SCQE 피드백 루프 실행 중...");

            let trajectory = engine.run_scqe_simulation(steps);

            let mut file = File::create(&output).unwrap();
            writeln!(file, "Step,R,Suppression,EnergyDensity").unwrap();
            for (t, (r, s, e)) in trajectory.iter().enumerate() {
                writeln!(file, "{t},{r:.6},{s:.6},{e:.6}").unwrap();
            }
            println!("SCQE 시뮬레이션 완료. 결과 저장됨: {output}");
        }
        Commands::CeArc { steps, output, diag } => {
            let dt = 0.01;
            let warmup = 1000.min(steps / 5);

            if diag {
                println!("=== ARC 잔차 원인 분해 진단 ===\n");
                let configs: Vec<(&str, f64, f64, usize)> = vec![
                    ("기준 (latency=2, full noise)", 0.08, 0.005, 2),
                    ("latency=0 (지연 제거)", 0.08, 0.005, 0),
                    ("latency=1 (1스텝 지연)", 0.08, 0.005, 1),
                    ("process=0 (확률노이즈 제거)", 0.0, 0.005, 2),
                    ("meas=0 (측정잡음 제거)", 0.08, 0.0, 2),
                    ("ideal (noise=0, meas=0, lat=0)", 0.0, 0.0, 0),
                ];

                for (label, pn, mn, lat) in &configs {
                    let mut env =
                        sfe_core::engine::arc::ArcSimulationEnv::with_params(*pn, *mn, *lat);
                    let mut sn = 0.0_f64;
                    let mut sr = 0.0_f64;
                    let mut n = 0_usize;

                    for t in 0..steps {
                        let (noise, resid) = env.step(dt);
                        if t >= warmup {
                            sn += noise * noise;
                            sr += resid * resid;
                            n += 1;
                        }
                    }

                    let rms_n = (sn / n as f64).sqrt();
                    let rms_r = (sr / n as f64).sqrt();
                    let red = if rms_n > 1e-12 {
                        (1.0 - rms_r / rms_n) * 100.0
                    } else if rms_r < 1e-12 {
                        100.0
                    } else {
                        0.0
                    };
                    println!("  {label:38} | RMS_n={rms_n:.6} RMS_r={rms_r:.6} | {red:6.2}%");
                }
                println!();
            }

            println!("모드: CE-ARC (3+1 ADM Coupled Cancellation) 시뮬레이션");
            println!("  결합 모델: 3+1 ADM + CE [R,K,Phi,Pi] 4D 결합 칼만");

            let mut env = sfe_core::engine::arc::ArcSimulationEnv::new();

            let mut file = File::create(&output).unwrap();
            writeln!(file, "Step,TrueNoise,ResidualNoise").unwrap();

            let pb = ProgressBar::new(steps as u64);
            let mut sum_noise_sq = 0.0_f64;
            let mut sum_resid_sq = 0.0_f64;
            let mut n_active = 0_usize;

            for t in 0..steps {
                let (true_noise, residual) = env.step(dt);
                writeln!(file, "{t},{true_noise:.8},{residual:.8}").unwrap();

                if t >= warmup {
                    sum_noise_sq += true_noise * true_noise;
                    sum_resid_sq += residual * residual;
                    n_active += 1;
                }
                pb.inc(1);
            }
            pb.finish();

            if n_active > 0 {
                let rms_noise = (sum_noise_sq / n_active as f64).sqrt();
                let rms_resid = (sum_resid_sq / n_active as f64).sqrt();
                let reduction = if rms_noise > 1e-12 {
                    (1.0 - rms_resid / rms_noise) * 100.0
                } else {
                    0.0
                };
                println!("  웜업: {warmup} steps 제외");
                println!("  RMS 노이즈:  {rms_noise:.6}");
                println!("  RMS 잔차:    {rms_resid:.6}");
                println!("  RMS 감소율:  {reduction:.2}%");
            }
            println!("CE-ARC 시뮬레이션 완료. 결과 저장됨: {output}");
        }
        Commands::MarketSimulation { steps, output } => {
            println!("모드: CE 금융 시장 시뮬레이션 (Surfing Strategy)");

            use rand::Rng;
            use sfe_core::engine::market::CeTradingBot;

            let initial_cash = 10_000.0;
            let mut bot = CeTradingBot::new(initial_cash);
            let mut price = 100.0;

            let mut file = File::create(&output).unwrap();
            writeln!(file, "Step,Price,PortfolioValue").unwrap();

            let pb = ProgressBar::new(steps as u64);
            let mut rng = rand::thread_rng();

            // 시뮬레이션 루프 (Flash Crash 시나리오 포함)
            for t in 0..steps {
                // Random Walk + Jump
                let mut ret = rng.gen_range(-0.01..0.01); // 일반 등락
                let mut volume = 1000.0 + rng.gen_range(-200.0..200.0);

                // 3000 스텝쯤에 폭락 이벤트 발생
                if (3000..3050).contains(&t) {
                    ret = -0.03; // 급락
                    volume = 5000.0; // 투매
                }
                // 3100 스텝쯤에 V자 반등
                if (3100..3150).contains(&t) {
                    ret = 0.03; // 반등
                    volume = 3000.0;
                }

                price *= 1.0 + ret;
                let high = price * 1.005;
                let low = price * 0.995;

                let value = bot.process_tick(price, volume, high, low);

                writeln!(file, "{t},{price:.2},{value:.2}").unwrap();
                pb.inc(1);
            }
            pb.finish();

            let final_value = bot.get_portfolio_value(price);
            let return_rate = (final_value - initial_cash) / initial_cash * 100.0;

            println!("시뮬레이션 완료.");
            println!("  초기 자본: ${initial_cash:.2}");
            println!("  최종 가치: ${final_value:.2}");
            println!("  수익률: {return_rate:.2}%");

            if return_rate > 0.0 {
                println!("  [SUCCESS] CE 전략이 시장을 이겼습니다.");
            }
            println!("결과 저장됨: {output}");
        }
        Commands::MacroCrisis { years, output } => {
            println!("모드: CE 거시경제 위기 예측 시뮬레이션");
            println!("  기간: {years}년 (Quarterly Data)");

            use rand::Rng;
            use sfe_core::engine::market::MacroEconomy;

            let mut economy = MacroEconomy::new();
            let mut rng = rand::thread_rng();
            let quarters = years * 4;

            let mut file = File::create(&output).unwrap();
            writeln!(
                file,
                "Quarter,DebtToGDP,GDP_Growth,Interest,Curvature,ProbCrisis"
            )
            .unwrap();

            let pb = ProgressBar::new(quarters as u64);

            for q in 0..quarters {
                // 시나리오:
                // 1. 부채 주도 성장 (Debt Bubble) -> Debt 증가, GDP 성장 양호
                // 2. 성장 둔화 (Stagnation) -> Debt 유지, GDP 하락
                // 3. 금리 인상 (Tightening) -> Interest 급등
                // 4. 위기 발생 (Minsky Moment)

                let mut debt_shock = 0.005; // 부채는 계속 늘어남
                let mut gdp_shock = rng.gen_range(-0.002..0.003);
                let mut rate_shock = 0.0;
                let m2_shock = 0.001;

                // 시나리오 이벤트 (Quarter 기준)
                if q > 20 && q < 40 {
                    debt_shock = 0.02; // 버블 형성기
                    gdp_shock = 0.005;
                } else if (40..50).contains(&q) {
                    gdp_shock = -0.03; // 강력한 경기 침체 (Depression)
                    rate_shock = 0.01; // 스태그플레이션 (금리 급등)
                } else if q >= 50 {
                    // 위기 후 디레버리징 (자동으로 되어야 하지만 여기선 강제)
                    debt_shock = -0.01;
                    rate_shock = -0.005;
                }

                let (r, prob) =
                    economy.update_and_predict(gdp_shock, debt_shock, rate_shock, m2_shock);

                writeln!(
                    file,
                    "{},{:.3},{:.3},{:.3},{:.6},{:.4}",
                    q, economy.debt_to_gdp, economy.gdp_growth, economy.interest_rate, r, prob
                )
                .unwrap();

                // 경고 임계치 낮춤 (0.5 -> 0.3)
                if prob > 0.3 {
                    println!(
                        "  [CRISIS WARNING] Q{}: Probability {:.1}% (R={:.4})",
                        q,
                        prob * 100.0,
                        r
                    );
                }

                pb.inc(1);
            }
            pb.finish();
            println!("거시경제 시뮬레이션 완료. 결과 저장됨: {output}");
        }
        Commands::QuantumNoise { .. } => {
            println!("(사용 중단됨) 포괄적인 분석을 위해 Sweep 명령을 사용하세요.");
        }
        Commands::DecouplingBenchmark { .. } => {
            println!("(사용 중단됨) 포괄적인 분석을 위해 Sweep 명령을 사용하세요.");
        }
        Commands::PulseOptimizer {
            steps,
            pulses,
            generations,
        } => {
            println!("모드: CE 펄스 최적화 (순수 공식 사용)");
            let (pulse_seq_idx, udd, sfe) = run_pulse_optimizer(steps, pulses, generations, 0.15);
            println!("최종 점수 -> UDD: {udd:.4}, CE: {sfe:.4}");

            let pulse_seq_norm: Vec<f64> = pulse_seq_idx
                .iter()
                .map(|&idx| idx as f64 / steps as f64)
                .collect();
            println!("\n>>> CE 정규화된 시퀀스 (검증용) <<<");
            print!("[");
            for (i, val) in pulse_seq_norm.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{val:.4}");
            }
            println!("]");
            println!(">>> 시퀀스 종료 <<<\n");
        }
        Commands::Sweep { output } => {
            run_sweep_benchmark(output);
        }
        Commands::Qec { distance, noise } => {
            println!("CE+QEC 하이브리드 시뮬레이션 실행 (거리={distance}, 노이즈={noise})");
            let steps = 2000usize;
            let sup_flag: i32 = std::env::var("CE_SUPPRESSON_ENABLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            match sup_flag {
                0 => {
                    std::env::remove_var("CE_SUPPRESSON_OMEGA");
                    std::env::remove_var("CE_SUPPRESSON_AMP");
                    std::env::remove_var("CE_SUPPRESSON_OMEGA2");
                    std::env::remove_var("CE_SUPPRESSON_AMP2");
                }
                1 => {
                    configure_fez_suppresson(noise, steps);
                }
                2 => {}
                _ => {
                    configure_fez_suppresson(noise, steps);
                }
            }
            println!("1. CE 펄스 적용 (Pure Analytic Formula)...");
            let (pulse_seq, _, ce_score) = run_pulse_optimizer(steps, 60, 0, noise);
            println!("   CE 결맞음 점수: {ce_score:.4}");

            println!("2. 반복 코드(Repetition Code) 시뮬레이션 (QEC 계층)...");
            let res = sfe_core::engine::qec::simulate_repetition_code(
                distance, &pulse_seq, noise, steps, 50, 2000,
            );

            println!("---------------------------------------------");
            println!("물리적 오류율 (p_phy): {:.6}", res.physical_error_rate);
            println!("논리적 오류율 (P_L):   {:.6}", res.logical_error_rate);
            println!("이득 (Gain):           {:.2}x", res.gain);
            println!("---------------------------------------------");
        }
        Commands::Surface { noise } => {
            println!("CE+Surface Code 시뮬레이션 실행 (거리=3, 노이즈={noise})");
            println!("1. CE 펄스 적용 (Pure Analytic Formula)...");
            let steps = 2000usize;
            let sup_flag: i32 = std::env::var("CE_SUPPRESSON_ENABLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            match sup_flag {
                0 => {
                    std::env::remove_var("CE_SUPPRESSON_OMEGA");
                    std::env::remove_var("CE_SUPPRESSON_AMP");
                    std::env::remove_var("CE_SUPPRESSON_OMEGA2");
                    std::env::remove_var("CE_SUPPRESSON_AMP2");
                }
                1 => {
                    configure_fez_suppresson(noise, steps);
                }
                2 => {}
                _ => {
                    configure_fez_suppresson(noise, steps);
                }
            }
            let (pulse_seq, _, ce_score) = run_pulse_optimizer(steps, 60, 0, noise);
            println!("   CE 결맞음 점수: {ce_score:.4}");
            println!("2. Surface Code(d=3) 시뮬레이션 (QEC 계층)...");
            let res =
                sfe_core::engine::qec::simulate_surface_code_d3(&pulse_seq, noise, steps, 50, 2000);
            println!("---------------------------------------------");
            println!("물리적 오류율 (p_phy): {:.6}", res.physical_error_rate);
            println!("논리적 오류율 (P_L):   {:.6}", res.logical_error_rate);
            println!("이득 (Gain):           {:.2}x", res.gain);
            println!("---------------------------------------------");
        }
        Commands::RunController { t1, t2, gate_err } => {
            println!("모드: CE 상용 컨트롤러 (Real-time Core)");
            let mut controller = CeController::new();
            let t_diag = Instant::now();
            let spec = controller.diagnose(t1, t2, gate_err);
            let diag_time = t_diag.elapsed().as_nanos();
            println!(
                "[Diagnosis] T1={}us T2={}us => TLS Amp={:.3} (Time: {} ns)",
                spec.t1, spec.t2, spec.tls_amp, diag_time
            );
            let t_strat = Instant::now();
            let strategy = controller.select_strategy(&spec);
            let strat_time = t_strat.elapsed().as_micros();
            println!(
                "[Strategy] Selected Mode: {:?} (Time: {} us)",
                strategy.mode, strat_time
            );
            println!("  - Expected Gain: {:.2}x", strategy.expected_gain);
            println!("  - Rec. QEC Dist: d={}", strategy.qec_distance);
            if !strategy.pulse_sequence.is_empty() {
                println!(
                    "  - Optimal Pulse Seq Generated ({} pulses)",
                    strategy.pulse_sequence.len()
                );
                println!("RAW_SEQ_START");
                println!("{:?}", strategy.pulse_sequence);
                println!("RAW_SEQ_END");
            }
        }
        Commands::RunIBM { api_key } => {
            println!("모드: IBM Quantum 하드웨어 브리지 (CE Rust-Native Controller)");
            let mut controller = CeController::new();
            let t1 = 60.0;
            let t2 = 40.0;
            let gate_err = 1e-3;
            println!("1. 하드웨어 상태 진단 중... (T1={t1}us, T2={t2}us)");
            let spec = controller.diagnose(t1, t2, gate_err);
            println!(
                "   -> 진단 결과: TLS Amp={:.3}, Drift={:.3}",
                spec.tls_amp, spec.drift_rate
            );
            println!("2. 최적 제어 전략 수립 중...");
            let strategy = controller.select_strategy(&spec);
            println!(
                "   -> 선택된 전략: {:?} (Gain {:.2}x)",
                strategy.mode, strategy.expected_gain
            );
            if strategy.pulse_sequence.is_empty() {
                println!("   [!] 전략이 ANC/Passive이므로 펄스 시퀀스 생성을 건너뜁니다.");
                return;
            }
            println!("3. IBM Quantum Runtime API 직접 연결 중...");
            let mut client = IbmClient::new(&api_key);
            match client.authenticate() {
                Ok(_) => {
                    println!("   인증 성공. CE 펄스 작업 제출...");
                    match client.submit_ce_job(&strategy.pulse_sequence) {
                        Ok(job_id) => {
                            println!("성공: 작업 제출 완료!");
                            println!("Job ID: {job_id}");
                            println!("모니터링: https://quantum.ibm.com/jobs/{job_id}");
                            println!("\n4. 작업 완료 대기 중...");
                            match client.wait_for_result(&job_id) {
                                Ok(result) => {
                                    println!("   결과 수신 완료!");
                                    controller.analyze_result(&result);
                                }
                                Err(e) => println!("오류: 결과 수신 실패: {e}"),
                            }
                        }
                        Err(e) => println!("오류: 작업 제출 실패: {e}"),
                    }
                }
                Err(e) => println!("오류: 인증 실패: {e}"),
            }
        }
        Commands::SuppressonScan => {
            println!("모드: 광명보손 증거 스캔 및 분석");
            sfe_core::run_suppresson_evidence_analysis();
        }
        Commands::MultiModeScan => {
            println!("모드: 3-모드 광명보손 Yukawa 스캔");
            sfe_core::engine::suppresson_physics::run_multimode_yukawa_scan();
        }
        Commands::ContinuousBounds => {
            println!("모드: 연속 스펙트럼 레이리 경계 스캔");
            sfe_core::engine::suppresson_physics::run_continuous_ratio_bounds();
        }
    }

    println!("총 소요 시간: {:.2}s", start_time.elapsed().as_secs_f64());
}
