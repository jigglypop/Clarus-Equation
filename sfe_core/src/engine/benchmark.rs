use super::optimizer::run_pulse_optimizer;
use indicatif::ProgressBar;
use std::fs::File;
use std::io::Write;

/// [신규] SFE vs UDD 민감도 분석 (스윕)
/// 펄스 개수와 노이즈 진폭에 대해 반복 실행
pub fn run_sweep_benchmark(output: String) {
    println!("포괄적 스윕 벤치마크 시작...");
    println!("스윕 범위: 펄스 수 [10..160], 노이즈 진폭 [0.02..0.30]");

    let pulse_counts = vec![10, 20, 30, 40, 50, 60, 80, 100, 120, 160];
    let noise_levels = vec![0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30];
    let steps = 2000;
    let generations = 10; // 스윕 속도를 위해 축소

    let mut file = File::create(&output).expect("스윕 출력 파일 생성 실패");
    writeln!(
        file,
        "PulseCount,NoiseAmp,UDD_Score,SFE_Score,Improvement_Pct"
    )
    .unwrap();

    let total_iters = pulse_counts.len() * noise_levels.len();
    let pb = ProgressBar::new(total_iters as u64);

    for &pulses in &pulse_counts {
        for &noise in &noise_levels {
            // 최적화 모듈을 사용하여 UDD(기준선)와 SFE(최적화됨) 점수를 모두 얻습니다.
            // run_pulse_optimizer는 (best_seq, udd_score, sfe_score)를 반환합니다.
            let (_, udd_score, sfe_score) = run_pulse_optimizer(steps, pulses, generations, noise);

            let improvement = (sfe_score - udd_score) / udd_score.abs() * 100.0;

            writeln!(
                file,
                "{pulses},{noise},{udd_score:.4},{sfe_score:.4},{improvement:.2}"
            )
            .unwrap();
            pb.inc(1);
        }
    }
    pb.finish();
    println!("스윕 완료. 데이터 저장됨: {output}");
}
