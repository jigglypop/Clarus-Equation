use clap::{Parser, Subcommand};
use std::time::Instant;
use sfe_core::{run_pulse_optimizer, run_sweep_benchmark, IbmClient, SfeController};
use sfe_core::engine::core::QSFEngine;
use sfe_core::engine::qec::SFE_PHASE_SCALE;
use std::fs::File;
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// SFE 억압장 동역학 시뮬레이션
    Dynamics {
        #[arg(short, long, default_value_t = 100_000)]
        size: usize,
        #[arg(short, long, default_value_t = 5_000)]
        steps: usize,
        #[arg(short, long, default_value = "sfe_output.csv")]
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
    /// SFE 펄스 최적화기 (Pure Analytic Formula)
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
    /// [NEW] SFE 상용 컨트롤러 (실시간 고속 진단/제어)
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
    /// [NEW] 억압보손 증거 스캔 및 분석
    SuppressonScan,
    /// [NEW] 3-모드 억압보손 Yukawa 스캔
    MultiModeScan,
}

fn configure_fez_suppresson(noise: f64, steps: usize) {
    if noise <= 0.0 {
        std::env::remove_var("SFE_SUPPRESSON_OMEGA");
        std::env::remove_var("SFE_SUPPRESSON_AMP");
        std::env::remove_var("SFE_SUPPRESSON_OMEGA2");
        std::env::remove_var("SFE_SUPPRESSON_AMP2");
        return;
    }
    let mut controller = SfeController::new();
    let t1 = 60.0_f64;
    let t2 = 40.0_f64;
    let gate_err = 1.0e-3_f64;
    let spec = controller.diagnose(t1, t2, gate_err);
    let tls_freq = match spec.tls_freq {
        Some(v) => v,
        None => {
            std::env::remove_var("SFE_SUPPRESSON_OMEGA");
            std::env::remove_var("SFE_SUPPRESSON_AMP");
            std::env::remove_var("SFE_SUPPRESSON_OMEGA2");
            std::env::remove_var("SFE_SUPPRESSON_AMP2");
            return;
        }
    };
    let total_time_us = 100.0_f64;
    let dt_us = total_time_us / steps as f64;
    let omega_top = tls_freq * dt_us;
    let amp_top = spec.tls_amp * dt_us / (noise.abs() * SFE_PHASE_SCALE);
    let omega_bottom = 0.5_f64 * omega_top;
    let amp_bottom = amp_top;
    std::env::set_var("SFE_SUPPRESSON_OMEGA", omega_top.to_string());
    std::env::set_var("SFE_SUPPRESSON_AMP", amp_top.to_string());
    std::env::set_var("SFE_SUPPRESSON_OMEGA2", omega_bottom.to_string());
    std::env::set_var("SFE_SUPPRESSON_AMP2", amp_bottom.to_string());
}

fn main() {
    let args = Args::parse();
    println!("==========================================");
    println!("   SFE 상용 엔진 v1.6 (Pure Formula)      ");
    println!("==========================================");

    let start_time = Instant::now();

    match args.command {
        Commands::Dynamics { size, steps, output } => {
             println!("모드: SFE 억압장 동역학 시뮬레이션");
             let mut engine = QSFEngine::new(size);
             let pb = ProgressBar::new(steps as u64);
             pb.set_style(ProgressStyle::default_bar().template("{spinner:.green} {bar:40} {pos}/{len}").unwrap());
             let mut history = Vec::with_capacity(steps);
             for t in 0..steps {
                 engine.step();
                 if t % 10 == 0 { history.push((t, engine.get_center_value())); }
                 pb.inc(1);
             }
             pb.finish();
             let mut file = File::create(&output).unwrap();
             writeln!(file, "TimeStep,CenterPhi").unwrap();
             for (t, v) in history { writeln!(file, "{},{}", t, v).unwrap(); }
        },
        Commands::QuantumNoise { .. } => {
            println!("(사용 중단됨) 포괄적인 분석을 위해 Sweep 명령을 사용하세요.");
        },
        Commands::DecouplingBenchmark { .. } => {
            println!("(사용 중단됨) 포괄적인 분석을 위해 Sweep 명령을 사용하세요.");
        },
        Commands::PulseOptimizer { steps, pulses, generations } => {
            println!("모드: SFE 펄스 최적화 (순수 공식 사용)");
            // run_pulse_optimizer는 이제 내부적으로 Analytic Formula를 사용합니다.
            let (pulse_seq_idx, udd, sfe) = run_pulse_optimizer(steps, pulses, generations, 0.15);
            println!("최종 점수 -> UDD: {:.4}, SFE: {:.4}", udd, sfe);

            let pulse_seq_norm: Vec<f64> = pulse_seq_idx
                .iter()
                .map(|&idx| idx as f64 / steps as f64)
                .collect();
            println!("\n>>> SFE 정규화된 시퀀스 (검증용) <<<");
            print!("[");
            for (i, val) in pulse_seq_norm.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:.4}", val);
            }
            println!("]");
            println!(">>> 시퀀스 종료 <<<\n");
        },
        Commands::Sweep { output } => {
            run_sweep_benchmark(output);
        },
        Commands::Qec { distance, noise } => {
            println!("SFE+QEC 하이브리드 시뮬레이션 실행 (거리={}, 노이즈={})", distance, noise);
            let steps = 2000usize;
            let sup_flag: i32 = std::env::var("SFE_SUPPRESSON_ENABLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            match sup_flag {
                0 => {
                    std::env::remove_var("SFE_SUPPRESSON_OMEGA");
                    std::env::remove_var("SFE_SUPPRESSON_AMP");
                    std::env::remove_var("SFE_SUPPRESSON_OMEGA2");
                    std::env::remove_var("SFE_SUPPRESSON_AMP2");
                }
                1 => {
                    configure_fez_suppresson(noise, steps);
                }
                2 => {}
                _ => {
                    configure_fez_suppresson(noise, steps);
                }
            }
            println!("1. SFE 펄스 적용 (Pure Analytic Formula)...");
            // Pure SFE 공식은 매우 빠르므로 즉시 결과가 나옵니다.
            // 노이즈가 강한 상황을 가정하여 최적화 수행
            let (pulse_seq, _, sfe_score) = run_pulse_optimizer(steps, 60, 0, noise);
            println!("   SFE 결맞음 점수: {:.4}", sfe_score);
            
            println!("2. 반복 코드(Repetition Code) 시뮬레이션 (QEC 계층)...");
            let res = sfe_core::engine::qec::simulate_repetition_code(
                distance, &pulse_seq, noise, steps, 50, 2000
            );
            
            println!("---------------------------------------------");
            println!("물리적 오류율 (p_phy): {:.6}", res.physical_error_rate);
            println!("논리적 오류율 (P_L):   {:.6}", res.logical_error_rate);
            println!("이득 (Gain):           {:.2}x", res.gain);
            println!("---------------------------------------------");
        },
        Commands::Surface { noise } => {
            println!("SFE+Surface Code 시뮬레이션 실행 (거리=3, 노이즈={})", noise);
            println!("1. SFE 펄스 적용 (Pure Analytic Formula)...");
            let steps = 2000usize;
            let sup_flag: i32 = std::env::var("SFE_SUPPRESSON_ENABLE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            match sup_flag {
                0 => {
                    std::env::remove_var("SFE_SUPPRESSON_OMEGA");
                    std::env::remove_var("SFE_SUPPRESSON_AMP");
                    std::env::remove_var("SFE_SUPPRESSON_OMEGA2");
                    std::env::remove_var("SFE_SUPPRESSON_AMP2");
                }
                1 => {
                    configure_fez_suppresson(noise, steps);
                }
                2 => {}
                _ => {
                    configure_fez_suppresson(noise, steps);
                }
            }
            let (pulse_seq, _, sfe_score) = run_pulse_optimizer(steps, 60, 0, noise);
            println!("   SFE 결맞음 점수: {:.4}", sfe_score);
            println!("2. Surface Code(d=3) 시뮬레이션 (QEC 계층)...");
            let res = sfe_core::engine::qec::simulate_surface_code_d3(
                &pulse_seq, noise, steps, 50, 2000,
            );
            println!("---------------------------------------------");
            println!("물리적 오류율 (p_phy): {:.6}", res.physical_error_rate);
            println!("논리적 오류율 (P_L):   {:.6}", res.logical_error_rate);
            println!("이득 (Gain):           {:.2}x", res.gain);
            println!("---------------------------------------------");
        },
        Commands::RunController { t1, t2, gate_err } => {
            println!("모드: SFE 상용 컨트롤러 (Real-time Core)");
            let mut controller = SfeController::new();
            
            // 1. 진단 (nanosecond scale)
            let t_diag = Instant::now();
            let spec = controller.diagnose(t1, t2, gate_err);
            let diag_time = t_diag.elapsed().as_nanos();
            
            println!("[Diagnosis] T1={}us T2={}us => TLS Amp={:.3} (Time: {} ns)", 
                spec.t1, spec.t2, spec.tls_amp, diag_time);

            // 2. 전략 수립 (microsecond scale)
            let t_strat = Instant::now();
            let strategy = controller.select_strategy(&spec);
            let strat_time = t_strat.elapsed().as_micros();
            
            println!("[Strategy] Selected Mode: {:?} (Time: {} us)", strategy.mode, strat_time);
            println!("  - Expected Gain: {:.2}x", strategy.expected_gain);
            println!("  - Rec. QEC Dist: d={}", strategy.qec_distance);
            
            if strategy.pulse_sequence.len() > 0 {
                println!("  - Optimal Pulse Seq Generated ({} pulses)", strategy.pulse_sequence.len());
                println!("RAW_SEQ_START");
                println!("{:?}", strategy.pulse_sequence);
                println!("RAW_SEQ_END");
            }
        },
        Commands::RunIBM { api_key } => {
            println!("모드: IBM Quantum 하드웨어 브리지 (SFE Rust-Native Controller)");
            
            let mut controller = SfeController::new();
            
            // 1. 진단 (Fez Typical Data or Fetch)
            let t1 = 60.0;
            let t2 = 40.0;
            let gate_err = 1e-3;
            
            println!("1. 하드웨어 상태 진단 중... (T1={}us, T2={}us)", t1, t2);
            let spec = controller.diagnose(t1, t2, gate_err);
            println!("   -> 진단 결과: TLS Amp={:.3}, Drift={:.3}", spec.tls_amp, spec.drift_rate);

            // 2. 전략 수립
            println!("2. 최적 제어 전략 수립 중...");
            let strategy = controller.select_strategy(&spec);
            println!("   -> 선택된 전략: {:?} (Gain {:.2}x)", strategy.mode, strategy.expected_gain);
            
            if strategy.pulse_sequence.is_empty() {
                println!("   [!] 전략이 ANC/Passive이므로 펄스 시퀀스 생성을 건너뜁니다.");
                return;
            }

            // 3. IBM API 직접 제출
            println!("3. IBM Quantum Runtime API 직접 연결 중...");
            let mut client = IbmClient::new(&api_key);
            
            match client.authenticate() {
                Ok(_) => {
                    println!("   인증 성공. SFE 펄스 작업 제출...");
                    match client.submit_sfe_job(&strategy.pulse_sequence) {
                        Ok(job_id) => {
                            println!("성공: 작업 제출 완료!");
                            println!("Job ID: {}", job_id);
                            println!("모니터링: https://quantum.ibm.com/jobs/{}", job_id);
                            
                            // 4. 결과 대기 및 수신
                            println!("\n4. 작업 완료 대기 중...");
                            match client.wait_for_result(&job_id) {
                                Ok(result) => {
                                    println!("   결과 수신 완료!");
                                    // 5. 자동 분석
                                    controller.analyze_result(&result);
                                },
                                Err(e) => println!("오류: 결과 수신 실패: {}", e),
                            }
                        },
                        Err(e) => println!("오류: 작업 제출 실패: {}", e),
                    }
                },
                Err(e) => println!("오류: 인증 실패: {}", e),
            }
        },
        Commands::SuppressonScan => {
            println!("모드: 억압보손 증거 스캔 및 분석");
            sfe_core::run_suppresson_evidence_analysis();
        }
        Commands::MultiModeScan => {
            println!("모드: 3-모드 억압보손 Yukawa 스캔");
            sfe_core::engine::suppresson_physics::run_multimode_yukawa_scan();
        }
    }

    println!("총 소요 시간: {:.2}s", start_time.elapsed().as_secs_f64());
}
