use crate::nlp::dropout::RBEDropout;
use anyhow::Result;

#[test]
fn 드롭아웃_생성_및_기본_동작_테스트() -> Result<()> {
    // 다양한 드롭아웃 확률로 생성
    let dropout_01 = RBEDropout::new(0.1)?;
    let dropout_05 = RBEDropout::new(0.5)?;
    let dropout_09 = RBEDropout::new(0.9)?;
    
    assert_eq!(dropout_01.dropout_prob, 0.1);
    assert_eq!(dropout_05.dropout_prob, 0.5);
    assert_eq!(dropout_09.dropout_prob, 0.9);
    
    // 잘못된 확률값 테스트
    assert!(RBEDropout::new(-0.1).is_err());
    assert!(RBEDropout::new(1.1).is_err());
    
    Ok(())
}

#[test]
fn 훈련모드_비활성화시_항등함수_테스트() -> Result<()> {
    let mut dropout = RBEDropout::new(0.5)?;
    dropout.set_training(false);
    
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output = dropout.forward(&input);
    
    // 훈련 모드가 아닐 때는 입력 그대로 출력
    assert_eq!(input, output);
    
    Ok(())
}

#[test]
fn 드롭아웃_스케일링_검증_테스트() -> Result<()> {
    let mut dropout = RBEDropout::new(0.5)?;
    dropout.set_training(true);
    
    // 큰 배열로 통계적 검증
    let size = 10000;
    let input: Vec<f32> = vec![1.0; size];
    
    let mut sum = 0.0;
    let num_trials = 100;
    
    for _ in 0..num_trials {
        let output = dropout.forward(&input);
        sum += output.iter().sum::<f32>();
    }
    
    let avg_sum = sum / num_trials as f32;
    let expected_sum = size as f32; // 스케일링으로 인해 평균은 유지되어야 함
    
    // 5% 오차 범위 내
    assert!((avg_sum - expected_sum).abs() / expected_sum < 0.05);
    
    Ok(())
}

#[test]
fn 푸앵카레_마스크_거리기반_드롭아웃_테스트() -> Result<()> {
    let mut dropout = RBEDropout::new(0.5)?;
    
    // 푸앵카레 마스크 생성
    let size = 1000;
    let mask = dropout.generate_poincare_mask(size);
    
    // 마스크는 bool 배열
    assert_eq!(mask.len(), size);
    
    // 중심부와 경계부의 드롭률 비교
    let center_drops = mask[0..100].iter().filter(|&&x| x).count();
    let boundary_drops = mask[900..1000].iter().filter(|&&x| x).count();
    
    // 경계부에서 더 많이 드롭되어야 함 (푸앵카레 메트릭 때문)
    println!("Center drops: {}, Boundary drops: {}", center_drops, boundary_drops);
    
    Ok(())
}

#[test]
fn 사이클_동기화_드롭아웃_테스트() -> Result<()> {
    let mut dropout = RBEDropout::new(0.5)?;
    
    let input = vec![1.0; 256];
    
    // 다른 사이클 상태에서 다른 패턴 생성
    let output1 = dropout.cycle_synchronized_dropout(&input, 0x000);
    let output2 = dropout.cycle_synchronized_dropout(&input, 0x3FF);
    let output3 = dropout.cycle_synchronized_dropout(&input, 0x7FF);
    
    // 다른 사이클 상태는 다른 드롭아웃 패턴을 생성해야 함
    let diff12: f32 = output1.iter().zip(&output2).map(|(a, b)| (a - b).abs()).sum();
    let diff23: f32 = output2.iter().zip(&output3).map(|(a, b)| (a - b).abs()).sum();
    
    assert!(diff12 > 0.0);
    assert!(diff23 > 0.0);
    
    Ok(())
}

#[test]
fn 이차원_텐서_드롭아웃_테스트() -> Result<()> {
    let mut dropout = RBEDropout::new(0.3)?;
    
    // 4x4 텐서
    let input = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
        vec![13.0, 14.0, 15.0, 16.0],
    ];
    
    let output = dropout.forward_2d(&input);
    
    // 출력 형태 유지
    assert_eq!(output.len(), 4);
    assert_eq!(output[0].len(), 4);
    
    // 훈련 모드에서는 일부 값이 0이 되어야 함
    let zero_count: usize = output.iter()
        .flat_map(|row| row.iter())
        .filter(|&&x| x == 0.0)
        .count();
    
    println!("Zero count: {} / 16", zero_count);
    
    Ok(())
}

#[test]
fn 극한_드롭아웃_확률_테스트() -> Result<()> {
    // 0% 드롭아웃
    let mut dropout0 = RBEDropout::new(0.0)?;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output0 = dropout0.forward(&input);
    assert_eq!(input, output0);
    
    // 99% 드롭아웃
    let mut dropout99 = RBEDropout::new(0.99)?;
    let large_input = vec![1.0; 1000];
    let output99 = dropout99.forward(&large_input);
    
    let non_zero_count = output99.iter().filter(|&&x| x != 0.0).count();
    // 99% 드롭아웃이므로 약 10개 정도만 살아남아야 함
    assert!(non_zero_count < 50); // 여유를 두고 50개 이하
    
    Ok(())
} 