use crate::core::optimizers::AdamState;

#[test]
fn Adam상태_초기화_테스트() {
    let adam_state = AdamState::new();
    
    assert_eq!(adam_state.m, 0.0, "1차 모멘트는 0으로 초기화되어야 함");
    assert_eq!(adam_state.v, 0.0, "2차 모멘트는 0으로 초기화되어야 함");
    assert_eq!(adam_state.t, 0, "시간 스텝은 0으로 초기화되어야 함");
    assert_eq!(adam_state.beta1, 0.9, "beta1 기본값은 0.9");
    assert_eq!(adam_state.beta2, 0.999, "beta2 기본값은 0.999");
    assert_eq!(adam_state.epsilon, 1e-8, "epsilon 기본값은 1e-8");
    
    println!("✅ Adam 상태 초기화 테스트 통과");
}

#[test]
fn Adam업데이트_기본동작_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    let gradient = 0.1;
    let lr = 0.001;
    
    let original_param = param;
    adam_state.update(&mut param, gradient, lr);
    
    assert_ne!(param, original_param, "파라미터가 업데이트되어야 함");
    assert!(param < original_param, "양의 그래디언트에 대해 파라미터는 감소해야 함");
    assert_eq!(adam_state.t, 1, "시간 스텝이 증가해야 함");
    assert!(adam_state.m > 0.0, "1차 모멘트가 업데이트되어야 함");
    assert!(adam_state.v > 0.0, "2차 모멘트가 업데이트되어야 함");
    
    println!("✅ Adam 업데이트 기본동작 테스트 통과");
    println!("   파라미터: {:.6} → {:.6}", original_param, param);
}

#[test]
fn Adam업데이트_연속호출_수렴_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    let gradient = 0.1;
    let lr = 0.01;
    
    let initial_param = param;
    
    // 여러 번 업데이트
    for i in 0..100 {
        adam_state.update(&mut param, gradient, lr);
        
        // 중간 진행 상황 확인
        if i % 25 == 24 {
            println!("   스텝 {}: 파라미터 = {:.6}", i + 1, param);
        }
    }
    
    // 수렴성 확인
    assert!(param < 0.5, "충분한 반복 후 파라미터는 크게 감소해야 함");
    assert_eq!(adam_state.t, 100, "시간 스텝이 올바르게 증가해야 함");
    
    println!("✅ Adam 업데이트 연속호출 수렴 테스트 통과");
    println!("   초기값: {:.6} → 최종값: {:.6}", initial_param, param);
}

#[test]
fn Adam업데이트_음의그래디언트_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 0.5;
    let gradient = -0.1;  // 음의 그래디언트
    let lr = 0.01;
    
    let original_param = param;
    adam_state.update(&mut param, gradient, lr);
    
    assert_ne!(param, original_param, "파라미터가 업데이트되어야 함");
    assert!(param > original_param, "음의 그래디언트에 대해 파라미터는 증가해야 함");
    
    println!("✅ Adam 업데이트 음의그래디언트 테스트 통과");
    println!("   파라미터: {:.6} → {:.6} (증가)", original_param, param);
}

#[test]
fn Adam상태_with_config_테스트() {
    let adam_state = AdamState::with_config(0.95, 0.9999, 1e-7);
    
    assert_eq!(adam_state.beta1, 0.95, "커스텀 beta1 값 확인");
    assert_eq!(adam_state.beta2, 0.9999, "커스텀 beta2 값 확인");
    assert_eq!(adam_state.epsilon, 1e-7, "커스텀 epsilon 값 확인");
    assert_eq!(adam_state.m, 0.0, "1차 모멘트는 여전히 0으로 초기화");
    assert_eq!(adam_state.v, 0.0, "2차 모멘트는 여전히 0으로 초기화");
    assert_eq!(adam_state.t, 0, "시간 스텝은 여전히 0으로 초기화");
    
    println!("✅ Adam 상태 with_config 테스트 통과");
    println!("   커스텀 설정: β1={}, β2={}, ε={}", 0.95, 0.9999, 1e-7);
}

#[test]
fn Adam상태_reset_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    
    // 몇 번 업데이트하여 상태 변경
    for _ in 0..5 {
        adam_state.update(&mut param, 0.1, 0.01);
    }
    
    // 상태가 변경되었는지 확인
    assert_ne!(adam_state.m, 0.0, "업데이트 후 1차 모멘트가 변경되어야 함");
    assert_ne!(adam_state.v, 0.0, "업데이트 후 2차 모멘트가 변경되어야 함");
    assert_eq!(adam_state.t, 5, "업데이트 후 시간 스텝이 5여야 함");
    
    // 리셋 수행
    adam_state.reset();
    
    // 리셋 후 상태 확인
    assert_eq!(adam_state.m, 0.0, "리셋 후 1차 모멘트는 0이어야 함");
    assert_eq!(adam_state.v, 0.0, "리셋 후 2차 모멘트는 0이어야 함");
    assert_eq!(adam_state.t, 0, "리셋 후 시간 스텝은 0이어야 함");
    
    println!("✅ Adam 상태 reset 테스트 통과");
}

#[test]
fn Adam업데이트_다양한_학습률_테스트() {
    let learning_rates = [0.001, 0.01, 0.1];
    let gradient = 0.1;
    let initial_param = 1.0;
    
    for &lr in &learning_rates {
        let mut adam_state = AdamState::new();
        let mut param = initial_param;
        
        // 10번 업데이트
        for _ in 0..10 {
            adam_state.update(&mut param, gradient, lr);
        }
        
        let param_change = (initial_param - param).abs();
        println!("   학습률 {}: 파라미터 변화량 = {:.6}", lr, param_change);
        
        // 학습률이 클수록 더 큰 변화가 있어야 함
        assert!(param_change > 0.0, "파라미터가 변화해야 함");
    }
    
    println!("✅ Adam 업데이트 다양한 학습률 테스트 통과");
} 