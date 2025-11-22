use clap::{Parser, Subcommand};
use std::time::Instant;
use sfe_core::{run_pulse_optimizer, run_sweep_benchmark, IbmClient};
use sfe_core::engine::core::QSFEngine; 
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
    /// Field Dynamics Simulation
    Dynamics {
        #[arg(short, long, default_value_t = 100_000)]
        size: usize,
        #[arg(short, long, default_value_t = 5_000)]
        steps: usize,
        #[arg(short, long, default_value = "sfe_output.csv")]
        output: String,
    },
    /// Quantum Noise Simulation
    QuantumNoise {
        #[arg(short, long, default_value_t = 10_000)]
        steps: usize,
        #[arg(short, long, default_value_t = 1000)]
        trials: usize,
        #[arg(short, long, default_value = "quantum_noise.csv")]
        output: String,
    },
    /// Decoupling Benchmark (Single Point)
    DecouplingBenchmark {
        #[arg(short, long, default_value_t = 10_000)]
        steps: usize,
        #[arg(short, long, default_value_t = 1000)]
        trials: usize,
        #[arg(short, long, default_value = "decoupling_result.csv")]
        output: String,
    },
    /// SFE-Genetic Pulse Optimizer
    PulseOptimizer {
        #[arg(short, long, default_value_t = 2000)]
        steps: usize,
        #[arg(short, long, default_value_t = 50)]
        pulses: usize,
        #[arg(short, long, default_value_t = 50)]
        generations: usize,
    },
    /// [NEW] Comprehensive Parameter Sweep (Heatmap Data)
    Sweep {
        #[arg(short, long, default_value = "sweep_results.csv")]
        output: String,
    },
    /// [NEW] Quantum Error Correction Hybrid Simulation
    Qec {
        #[arg(short, long, default_value_t = 3)]
        distance: usize,
        #[arg(short, long, default_value_t = 0.10)]
        noise: f64,
    },
    /// [NEW] Connect to IBM Quantum Hardware
    RunIBM {
        #[arg(short, long)]
        api_key: String,
    }
}

fn main() {
    let args = Args::parse();
    println!("==========================================");
    println!("   SFE Commercial Engine v1.5 (Cloud)     ");
    println!("==========================================");

    let start_time = Instant::now();

    match args.command {
        Commands::Dynamics { size, steps, output } => {
             println!("Mode: Field Dynamics Simulation");
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
            println!("(Deprecated) Use Sweep for comprehensive analysis.");
        },
        Commands::DecouplingBenchmark { .. } => {
            println!("(Deprecated) Use Sweep for comprehensive analysis.");
        },
        Commands::PulseOptimizer { steps, pulses, generations } => {
            let (_, udd, sfe) = run_pulse_optimizer(steps, pulses, generations, 0.15);
            println!("Final Result -> UDD: {:.4}, SFE: {:.4}", udd, sfe);
        },
        Commands::Sweep { output } => {
            run_sweep_benchmark(output);
        },
        Commands::Qec { distance, noise } => {
            println!("Running SFE+QEC Hybrid Simulation (d={}, noise={})", distance, noise);
            println!("1. Optimizing Pulses (SFE Layer)...");
            let (pulse_seq, _, sfe_score) = run_pulse_optimizer(2000, 60, 20, noise);
            println!("   SFE Score (Coherence): {:.4}", sfe_score);
            
            println!("2. Simulating Repetition Code (QEC Layer)...");
            let res = sfe_core::engine::qec::simulate_repetition_code(
                distance, &pulse_seq, noise, 2000, 50, 2000
            );
            
            println!("---------------------------------------------");
            println!("Physical Error Rate: {:.6}", res.physical_error_rate);
            println!("Logical Error Rate:  {:.6}", res.logical_error_rate);
            println!("Gain (Phy/Log): {:.2}", res.gain);
            println!("---------------------------------------------");
        },
        Commands::RunIBM { api_key } => {
            println!("Mode: IBM Quantum Hardware Bridge");
            println!("1. Running SFE Optimizer to find best pulse sequence...");
            // Run optimizer locally first to get the "Recipe"
            let steps_total = 2000;
            let (pulse_seq_idx, _, sfe_score) = run_pulse_optimizer(steps_total, 50, 20, 0.15);
            println!("   Optimization Complete. SFE Score: {:.4}", sfe_score);
            println!("   Pulse Count: {}", pulse_seq_idx.len());

            // Convert indices (steps) to normalized time (0.0 - 1.0)
            let pulse_seq_norm: Vec<f64> = pulse_seq_idx.iter()
                .map(|&idx| idx as f64 / steps_total as f64)
                .collect();
            
            // [NEW] Print Sequence for Python Bridge
            println!("\n>>> COPY THIS SEQUENCE FOR PYTHON BRIDGE <<<");
            print!("[");
            for (i, val) in pulse_seq_norm.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:.4}", val);
            }
            println!("]");
            println!(">>> END SEQUENCE <<<\n");

            println!("2. Connecting to IBM Quantum API...");
            let mut client = IbmClient::new(&api_key);
            
            match client.authenticate() {
                Ok(_) => {
                    match client.submit_sfe_job(&pulse_seq_norm) {
                        Ok(job_id) => {
                            println!("SUCCESS: Job submitted successfully!");
                            println!("Job ID: {}", job_id);
                            println!("Monitor at: https://quantum.ibm.com/jobs");
                        },
                        Err(e) => println!("ERROR: Failed to submit job: {}", e),
                    }
                },
                Err(e) => println!("ERROR: Authentication failed: {}", e),
            }
        }
    }

    println!("Total Time: {:.2}s", start_time.elapsed().as_secs_f64());
}
