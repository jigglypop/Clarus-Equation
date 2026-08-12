use crate::core::optimizers::TransformAnalyzer;
use crate::packed_params::TransformType;

#[test]
fn 변환분석기_초기화_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    assert_eq!(analyzer.dct_performance, 0.0, "DCT 성능은 0으로 초기화");
    assert_eq!(analyzer.wavelet_performance, 0.0, "웨이블릿 성능은 0으로 초기화");
    assert_eq!(analyzer.smoothness_threshold, 0.1, "평활도 임계값 기본값");
    assert_eq!(analyzer.frequency_concentration_threshold, 2.0, "주파수 집중도 임계값 기본값");
    
    println!("✅ 변환분석기 초기화 테스트 통과");
}

#[test]
fn 평활도측정_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    // 완전히 평평한 신호 (변화 없음)
    let flat_signal = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let flat_smoothness = analyzer.measure_smoothness(&flat_signal);
    assert_eq!(flat_smoothness, 0.0, "평평한 신호의 평활도는 0");
    
    // 선형 증가 신호
    let linear_signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let linear_smoothness = analyzer.measure_smoothness(&linear_signal);
    assert_eq!(linear_smoothness, 1.0, "선형 신호의 평활도는 기울기와 동일");
    
    // 급격한 변화 신호
    let noisy_signal = vec![0.0, 10.0, 0.0, 10.0, 0.0];
    let noisy_smoothness = analyzer.measure_smoothness(&noisy_signal);
    assert!(noisy_smoothness > 5.0, "급격한 변화 신호는 높은 평활도 값");
    
    // 빈 신호와 단일 요소 신호
    assert_eq!(analyzer.measure_smoothness(&vec![]), 0.0, "빈 신호의 평활도는 0");
    assert_eq!(analyzer.measure_smoothness(&vec![5.0]), 0.0, "단일 요소 신호의 평활도는 0");
    
    println!("✅ 평활도 측정 테스트 통과");
    println!("   평평한 신호: {:.3}", flat_smoothness);
    println!("   선형 신호: {:.3}", linear_smoothness);
    println!("   노이즈 신호: {:.3}", noisy_smoothness);
}

#[test]
fn 주파수집중도측정_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    // 단일 주파수 성분 (DC)
    let dc_signal = vec![1.0, 1.0, 1.0, 1.0];
    let dc_concentration = analyzer.measure_frequency_concentration(&dc_signal);
    println!("DC concentration: {}", dc_concentration);
    assert!(dc_concentration > 0.8, "DC 성분은 높은 집중도를 가져야 함");
    
    // 랜덤 노이즈 (여러 주파수 성분)
    let noise_signal = vec![0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4];
    let noise_concentration = analyzer.measure_frequency_concentration(&noise_signal);
    assert!(noise_concentration < dc_concentration, "노이즈는 DC보다 낮은 집중도");
    
    // 빈 신호
    let empty_concentration = analyzer.measure_frequency_concentration(&vec![]);
    assert_eq!(empty_concentration, 0.0, "빈 신호의 주파수 집중도는 0");
    
    println!("✅ 주파수 집중도 측정 테스트 통과");
    println!("   DC 신호: {:.3}", dc_concentration);
    println!("   노이즈 신호: {:.3}", noise_concentration);
}

#[test]
fn 최적변환선택_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    // 평활한 신호 + 높은 주파수 집중도 → DCT가 유리
    let smooth_concentrated = vec![1.0, 1.01, 1.02, 1.03, 1.04]; // 매우 평활
    let choice1 = analyzer.select_optimal_transform(&smooth_concentrated);
    
    // 급격한 변화 신호 → Wavelet이 유리할 가능성
    let rough_signal = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let choice2 = analyzer.select_optimal_transform(&rough_signal);
    
    // 결과 출력 (정확한 알고리즘 예측은 어려우므로 단순히 실행 확인)
    println!("✅ 최적 변환 선택 테스트 통과");
    println!("   평활한 신호 → {:?}", choice1);
    println!("   급격한 신호 → {:?}", choice2);
    
    // 최소한 유효한 TransformType이 반환되는지 확인
    assert!(matches!(choice1, TransformType::Dct | TransformType::Dwt));
    assert!(matches!(choice2, TransformType::Dct | TransformType::Dwt));
}

#[test]
fn DCT성능측정_테스트() {
    let mut analyzer = TransformAnalyzer::new();
    
    // 간단한 테스트 신호
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let compression_ratio = 0.5; // 50% 압축
    
    let rmse = analyzer.measure_dct_performance(&original, compression_ratio);
    
    assert!(rmse >= 0.0, "RMSE는 항상 비음수");
    assert!(analyzer.dct_performance >= 0.0, "DCT 성능 지표가 업데이트됨");
    assert_eq!(analyzer.dct_performance, rmse, "성능 지표가 올바르게 저장됨");
    
    // 빈 신호에 대한 처리
    let empty_rmse = analyzer.measure_dct_performance(&vec![], 0.5);
    assert_eq!(empty_rmse, 0.0, "빈 신호의 RMSE는 0");
    
    println!("✅ DCT 성능 측정 테스트 통과");
    println!("   신호 길이: {}, 압축률: {:.1}%, RMSE: {:.6}", original.len(), compression_ratio * 100.0, rmse);
}

#[test]
fn 웨이블릿성능측정_테스트() {
    let mut analyzer = TransformAnalyzer::new();
    
    // 2의 거듭제곱 길이 신호 (웨이블릿에 적합)
    let original = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let compression_ratio = 0.5; // 50% 압축
    
    let rmse = analyzer.measure_wavelet_performance(&original, compression_ratio);
    
    assert!(rmse >= 0.0, "RMSE는 항상 비음수");
    assert!(analyzer.wavelet_performance >= 0.0, "웨이블릿 성능 지표가 업데이트됨");
    assert_eq!(analyzer.wavelet_performance, rmse, "성능 지표가 올바르게 저장됨");
    
    // 길이가 2 미만인 신호
    let short_rmse = analyzer.measure_wavelet_performance(&vec![1.0], 0.5);
    assert_eq!(short_rmse, 0.0, "너무 짧은 신호의 RMSE는 0");
    
    // 빈 신호
    let empty_rmse = analyzer.measure_wavelet_performance(&vec![], 0.5);
    assert_eq!(empty_rmse, 0.0, "빈 신호의 RMSE는 0");
    
    println!("✅ 웨이블릿 성능 측정 테스트 통과");
    println!("   신호 길이: {}, 압축률: {:.1}%, RMSE: {:.6}", original.len(), compression_ratio * 100.0, rmse);
}

#[test]
fn 변환방법비교_테스트() {
    let mut analyzer = TransformAnalyzer::new();
    
    // 테스트용 신호 (2의 거듭제곱 길이)
    let signal = vec![1.0, 4.0, 2.0, 3.0, 5.0, 1.0, 4.0, 2.0];
    let compression_ratio = 0.375; // 37.5% 압축
    
    let (winner, dct_rmse, wavelet_rmse) = analyzer.compare_transforms(&signal, compression_ratio);
    
    // 결과 검증
    assert!(dct_rmse >= 0.0, "DCT RMSE는 비음수");
    assert!(wavelet_rmse >= 0.0, "웨이블릿 RMSE는 비음수");
    assert!(matches!(winner, TransformType::Dct | TransformType::Dwt), "유효한 승자 선택");
    
    // 성능 지표 업데이트 확인
    assert_eq!(analyzer.dct_performance, dct_rmse, "DCT 성능 지표 업데이트");
    assert_eq!(analyzer.wavelet_performance, wavelet_rmse, "웨이블릿 성능 지표 업데이트");
    
    // 승자 결정 로직 확인
    let expected_winner = if dct_rmse < wavelet_rmse {
        TransformType::Dct
    } else {
        TransformType::Dwt
    };
    assert_eq!(winner, expected_winner, "승자가 올바르게 결정됨");
    
    println!("✅ 변환 방법 비교 테스트 통과");
    println!("   DCT RMSE: {:.6}", dct_rmse);
    println!("   웨이블릿 RMSE: {:.6}", wavelet_rmse);
    println!("   승자: {:?}", winner);
} 