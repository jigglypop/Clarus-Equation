//! RBELayerNorm 레이어 테스트

use crate::nlp::layernorm::{RBELayerNorm, RBELayerNormConfig};
use anyhow::Result;

#[test]
fn 레이어놈_생성_및_기본동작_테스트() -> Result<()> {
    let config = RBELayerNormConfig {
        normalized_shape: vec![128],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: false,  // 표준 구현으로 테스트
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 간단한 입력
    let input = vec![1.0, 2.0, 3.0, 4.0].repeat(32);  // 128개 원소
    let output = layer_norm.forward(&input)?;
    
    assert_eq!(output.len(), input.len());
    
    // 정규화 후 평균은 0에 가까워야 함
    let mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
    assert!(mean.abs() < 0.01, "평균이 0에서 너무 멀리 떨어짐: {}", mean);
    
    // 정규화 후 표준편차는 1에 가까워야 함
    let variance: f32 = output.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / output.len() as f32;
    let std_dev = variance.sqrt();
    assert!((std_dev - 1.0).abs() < 0.01, "표준편차가 1에서 너무 멀리 떨어짐: {}", std_dev);
    
    Ok(())
}

#[test]
fn 수치적_안정성_극한값_테스트() -> Result<()> {
    let config = RBELayerNormConfig {
        normalized_shape: vec![10],
        eps: 1e-5,
        elementwise_affine: false,  // affine 없이 순수 정규화만
        use_fused_ops: true,
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 극한값 테스트 케이스들
    let test_cases = vec![
        // 매우 큰 값
        vec![1e10; 10],
        // 매우 작은 값
        vec![1e-10; 10],
        // 큰 값과 작은 값 혼합
        vec![1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10],
        // 모두 같은 값 (분산 0)
        vec![5.0; 10],
    ];
    
    for (i, input) in test_cases.iter().enumerate() {
        let output = layer_norm.forward(input)?;
        
        // NaN이나 Inf가 없어야 함
        for &val in output.iter() {
            assert!(!val.is_nan(), "테스트 케이스 {}에서 NaN 발생", i);
            assert!(!val.is_infinite(), "테스트 케이스 {}에서 Inf 발생", i);
        }
    }
    
    Ok(())
}

#[test]
fn 카한_합산_정확도_테스트() -> Result<()> {
    let config = RBELayerNormConfig {
        normalized_shape: vec![100],
        eps: 1e-12,  // 매우 작은 epsilon
        elementwise_affine: false,
        use_fused_ops: true,  // Kahan summation 사용
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 수치적으로 까다로운 입력: 큰 값과 작은 값의 조합
    let mut input = vec![1e6; 50];
    input.extend(vec![1.0; 50]);
    
    let output = layer_norm.forward(&input)?;
    
    // 통계 확인
    let stats = layer_norm.compute_statistics(&input)?;
    
    // 평균이 정확히 계산되었는지 확인
    let expected_mean = (50.0 * 1e6 + 50.0) / 100.0;
    let actual_mean = stats.means[0];
    let relative_error = ((actual_mean - expected_mean) / expected_mean).abs();
    
    assert!(relative_error < 1e-6, 
            "Kahan summation 정확도 부족: 상대 오차 {}", relative_error);
    
    Ok(())
}

#[test]
fn 배치_처리_일관성_테스트() -> Result<()> {
    let config = RBELayerNormConfig {
        normalized_shape: vec![64],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: true,
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 배치 입력 (3개 샘플, 각 64차원)
    let mut batch_input = Vec::new();
    for i in 0..(64 * 3) {
        batch_input.push((i % 10) as f32 + 1.0);
    }
    
    // 전체 배치 처리
    let batch_output = layer_norm.forward(&batch_input)?;
    
    // 개별 처리
    let sample1 = &batch_input[0..64];
    let sample2 = &batch_input[64..128];
    let sample3 = &batch_input[128..192];
    
    let output1 = layer_norm.forward(sample1)?;
    let output2 = layer_norm.forward(sample2)?;
    let output3 = layer_norm.forward(sample3)?;
    
    // 배치 처리와 개별 처리 결과가 같아야 함
    for i in 0..64 {
        assert!((batch_output[i] - output1[i]).abs() < 1e-6,
                "샘플 1 불일치 at {}: {} vs {}", i, batch_output[i], output1[i]);
        assert!((batch_output[64 + i] - output2[i]).abs() < 1e-6,
                "샘플 2 불일치 at {}: {} vs {}", i, batch_output[64 + i], output2[i]);
        assert!((batch_output[128 + i] - output3[i]).abs() < 1e-6,
                "샘플 3 불일치 at {}: {} vs {}", i, batch_output[128 + i], output3[i]);
    }
    
    Ok(())
}

#[test]
fn 융합연산_vs_표준연산_일관성_테스트() -> Result<()> {
    let base_config = RBELayerNormConfig {
        normalized_shape: vec![256],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: false,
    };
    
    // 표준 구현
    let standard_ln = RBELayerNorm::new(base_config.clone())?;
    
    // 융합 구현
    let mut fused_config = base_config.clone();
    fused_config.use_fused_ops = true;
    let fused_ln = RBELayerNorm::new(fused_config)?;
    
    // 랜덤 입력
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    let input: Vec<f32> = (0..256).map(|_| rng.gen_range(-10.0..10.0)).collect();
    
    let standard_output = standard_ln.forward(&input)?;
    let fused_output = fused_ln.forward(&input)?;
    
    // 두 구현의 결과가 거의 같아야 함
    let max_diff = standard_output.iter()
        .zip(fused_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    assert!(max_diff < 1e-5, 
            "표준과 융합 구현 차이가 너무 큼: {}", max_diff);
    
    Ok(())
}

#[test]
fn 사전학습_가중치_로드_테스트() -> Result<()> {
    let config = RBELayerNormConfig {
        normalized_shape: vec![32],
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: true,
    };
    
    // 가짜 사전학습 가중치
    let gamma: Vec<f32> = (0..32).map(|i| 1.0 + 0.01 * i as f32).collect();
    let beta: Vec<f32> = (0..32).map(|i| -0.5 + 0.02 * i as f32).collect();
    
    let layer_norm = RBELayerNorm::from_pretrained(
        Some(gamma.clone()),
        Some(beta.clone()),
        config
    )?;
    
    // 테스트 입력
    let input = vec![0.0; 32];  // 모두 0
    let output = layer_norm.forward(&input)?;
    
    // 입력이 모두 0이고 정규화하면 모두 0이 되므로
    // 출력은 beta와 같아야 함
    for i in 0..32 {
        assert!((output[i] - beta[i]).abs() < 1e-5,
                "인덱스 {}에서 예상값과 다름: {} vs {}", i, output[i], beta[i]);
    }
    
    Ok(())
}

#[test]
fn 통계정보_추출_테스트() -> Result<()> {
    let config = RBELayerNormConfig {
        normalized_shape: vec![50],
        eps: 1e-5,
        elementwise_affine: false,
        use_fused_ops: true,
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 2개 배치, 각 50차원
    let mut input = vec![1.0; 50];
    input.extend(vec![10.0; 50]);
    
    let stats = layer_norm.compute_statistics(&input)?;
    
    assert_eq!(stats.means.len(), 2);
    assert_eq!(stats.variances.len(), 2);
    
    // 첫 번째 배치: 모두 1.0
    assert!((stats.means[0] - 1.0).abs() < 1e-6);
    assert!(stats.variances[0] < 1e-6);  // 분산은 0에 가까워야 함
    
    // 두 번째 배치: 모두 10.0
    assert!((stats.means[1] - 10.0).abs() < 1e-6);
    assert!(stats.variances[1] < 1e-6);  // 분산은 0에 가까워야 함
    
    Ok(())
}

#[test]
fn 성능_벤치마크_테스트() -> Result<()> {
    use std::time::Instant;
    
    let config = RBELayerNormConfig {
        normalized_shape: vec![768],  // GPT-2 크기
        eps: 1e-5,
        elementwise_affine: true,
        use_fused_ops: true,
    };
    
    let layer_norm = RBELayerNorm::new(config)?;
    
    // 배치 크기 64
    let input = vec![1.0; 768 * 64];
    
    // 워밍업
    for _ in 0..10 {
        let _ = layer_norm.forward(&input)?;
    }
    
    // 실제 측정
    let start = Instant::now();
    let iterations = 100;
    
    for _ in 0..iterations {
        let _ = layer_norm.forward(&input)?;
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    println!("LayerNorm 평균 실행 시간: {:?}", avg_time);
    println!("처리량: {:.2} samples/sec", 
             64.0 * iterations as f64 / elapsed.as_secs_f64());
    
    // 성능 기준: 배치당 1ms 이하
    assert!(avg_time.as_micros() < 1000, 
            "LayerNorm이 너무 느림: {:?}", avg_time);
    
    Ok(())
} 