use super::super::ErrorController;

#[test]
fn 오차제어기_생성_테스트() {
    let threshold = 1e-3;
    let controller = ErrorController::new(threshold);
    
    assert_eq!(controller.global_error_threshold, threshold);
    assert!(controller.block_errors.is_empty());
    assert!(controller.error_weights.is_empty());
    
    println!("✅ 오차제어기 생성 테스트 통과");
    println!("   임계값: {:.6}", threshold);
}

#[test]
fn 오차_업데이트_테스트() {
    let mut controller = ErrorController::new(1e-3);
    
    controller.update_block_error((0, 0), 0.001);
    controller.update_block_error((0, 1), 0.002);
    controller.update_block_error((1, 0), 0.0005);
    controller.update_block_error((1, 1), 0.003);
    
    assert_eq!(controller.block_errors.len(), 4);
    assert_eq!(controller.block_errors[&(0, 0)], 0.001);
    assert_eq!(controller.block_errors[&(0, 1)], 0.002);
    assert_eq!(controller.block_errors[&(1, 0)], 0.0005);
    assert_eq!(controller.block_errors[&(1, 1)], 0.003);
    
    println!("✅ 오차 업데이트 테스트 통과");
    println!("   등록된 블록 수: {}", controller.block_errors.len());
}

#[test]
fn 전체_오차_계산_테스트() {
    let mut controller = ErrorController::new(1e-3);
    
    let empty_error = controller.compute_total_error();
    assert_eq!(empty_error, 0.0);
    
    controller.update_block_error((0, 0), 0.003);
    controller.update_block_error((0, 1), 0.004);
    
    let total_error = controller.compute_total_error();
    assert!(total_error > 0.0);
    assert!(total_error < 1.0);
    
    println!("✅ 전체 오차 계산 테스트 통과");
    println!("   빈 상태: {:.6}", empty_error);
    println!("   오차 존재: {:.6}", total_error);
}

#[test]
fn 분할_필요성_판단_테스트() {
    let mut controller = ErrorController::new(1e-3);
    
    let should_subdivide_unknown = controller.should_subdivide((0, 0), 1);
    assert!(should_subdivide_unknown);
    
    controller.update_block_error((1, 1), 0.005);
    let should_subdivide_high = controller.should_subdivide((1, 1), 1);
    assert!(should_subdivide_high);
    
    controller.update_block_error((2, 2), 0.0005);
    let should_subdivide_low = controller.should_subdivide((2, 2), 1);
    assert!(!should_subdivide_low);
    
    controller.update_block_error((3, 3), 0.01);
    let should_subdivide_max_depth = controller.should_subdivide((3, 3), 4);
    assert!(!should_subdivide_max_depth);
    
    println!("✅ 분할 필요성 판단 테스트 통과");
} 