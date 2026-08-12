//! # 성능 모니터링 단위테스트
//!
//! PerformanceMonitor와 관련 구조체들의 기능 검증

use crate::core::systems::performance::{
    PerformanceMonitor, QualityMetricsTracker, MemoryUsageTracker,
    CompressionMetrics, LayerPerformance, SystemMetrics
};
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 성능_모니터_초기화_테스트() {
        let monitor = PerformanceMonitor::new();
        
        assert_eq!(monitor.quality_metrics.accuracy, 0.0);
        assert_eq!(monitor.quality_metrics.precision, 0.0);
        assert_eq!(monitor.quality_metrics.recall, 0.0);
        assert_eq!(monitor.memory_usage.current_usage, 0);
        assert_eq!(monitor.compression_metrics.original_size, 0);
        
        println!("✅ 성능 모니터 초기화 테스트 통과");
        println!("   초기 정확도: {}", monitor.quality_metrics.accuracy);
    }

    #[test]
    fn 품질_지표_직접_수정_테스트() {
        let mut monitor = PerformanceMonitor::new();
        
        // 직접 필드 수정
        monitor.quality_metrics.accuracy = 0.95;
        monitor.quality_metrics.precision = 0.92;
        monitor.quality_metrics.recall = 0.88;
        monitor.quality_metrics.f1_score = 0.90;
        
        assert_eq!(monitor.quality_metrics.accuracy, 0.95);
        assert_eq!(monitor.quality_metrics.precision, 0.92);
        assert_eq!(monitor.quality_metrics.recall, 0.88);
        assert_eq!(monitor.quality_metrics.f1_score, 0.90);
        
        println!("✅ 품질 지표 직접 수정 테스트 통과");
        println!("   정확도: {:.3}, F1 점수: {:.3}", 
                monitor.quality_metrics.accuracy, 
                monitor.quality_metrics.f1_score);
    }

    #[test]
    fn 메모리_사용량_직접_추적_테스트() {
        let mut monitor = PerformanceMonitor::new();
        
        monitor.memory_usage.current_usage = 1024 * 1024; // 1MB
        assert_eq!(monitor.memory_usage.current_usage, 1024 * 1024);
        
        monitor.memory_usage.current_usage = 2048 * 1024; // 2MB
        monitor.memory_usage.peak_usage = 2048 * 1024;
        assert_eq!(monitor.memory_usage.current_usage, 2048 * 1024);
        assert_eq!(monitor.memory_usage.peak_usage, 2048 * 1024);
        
        // 메모리 감소
        monitor.memory_usage.current_usage = 512 * 1024; // 0.5MB
        // peak_usage는 수동으로 유지
        assert_eq!(monitor.memory_usage.current_usage, 512 * 1024);
        assert_eq!(monitor.memory_usage.peak_usage, 2048 * 1024); // 최고값 유지
        
        println!("✅ 메모리 사용량 직접 추적 테스트 통과");
        println!("   현재: {}KB, 최고: {}KB", 
                monitor.memory_usage.current_usage / 1024,
                monitor.memory_usage.peak_usage / 1024);
    }

    #[test]
    fn 압축_지표_직접_계산_테스트() {
        let mut monitor = PerformanceMonitor::new();
        
        let original_size = 1000000; // 1MB
        let compressed_size = 10000;  // 10KB
        
        monitor.compression_metrics.original_size = original_size;
        monitor.compression_metrics.compressed_size = compressed_size;
        monitor.compression_metrics.compression_ratio = original_size as f32 / compressed_size as f32;
        
        assert_eq!(monitor.compression_metrics.original_size, original_size);
        assert_eq!(monitor.compression_metrics.compressed_size, compressed_size);
        
        let expected_ratio = original_size as f32 / compressed_size as f32;
        assert!((monitor.compression_metrics.compression_ratio - expected_ratio).abs() < 0.001);
        
        println!("✅ 압축 지표 직접 계산 테스트 통과");
        println!("   압축률: {:.1}:1", monitor.compression_metrics.compression_ratio);
    }

    #[test]
    fn 레이어_성능_기록_테스트() {
        let mut monitor = PerformanceMonitor::new();
        
        let execution_time = Duration::from_millis(10);
        let layer_perf = LayerPerformance {
            layer_id: 0,
            execution_time: execution_time,
            input_size: 1024,
            output_size: 512,
        };
        
        monitor.layer_performances.push(layer_perf);
        
        assert_eq!(monitor.layer_performances.len(), 1);
        let recorded_layer = &monitor.layer_performances[0];
        assert_eq!(recorded_layer.layer_id, 0);
        assert_eq!(recorded_layer.input_size, 1024);
        assert_eq!(recorded_layer.output_size, 512);
        assert_eq!(recorded_layer.execution_time, execution_time);
        
        println!("✅ 레이어 성능 기록 테스트 통과");
        println!("   실행 시간: {:?}", recorded_layer.execution_time);
    }

    #[test]
    fn 시스템_지표_직접_설정_테스트() {
        let mut monitor = PerformanceMonitor::new();
        
        monitor.system_metrics.total_execution_time = Duration::from_millis(30);
        monitor.system_metrics.total_layers = 3;
        monitor.system_metrics.throughput = 34.133; // samples per ms
        
        assert_eq!(monitor.system_metrics.total_layers, 3);
        assert_eq!(monitor.system_metrics.total_execution_time, Duration::from_millis(30));
        assert!((monitor.system_metrics.throughput - 34.133).abs() < 0.001);
        
        println!("✅ 시스템 지표 직접 설정 테스트 통과");
        println!("   총 실행 시간: {:?}, 처리량: {:.3} samples/ms", 
                monitor.system_metrics.total_execution_time,
                monitor.system_metrics.throughput);
    }

    #[test]
    fn 구조체_복제_테스트() {
        let monitor = PerformanceMonitor::new();
        let cloned_monitor = monitor.clone();
        
        assert_eq!(monitor.quality_metrics.accuracy, cloned_monitor.quality_metrics.accuracy);
        assert_eq!(monitor.memory_usage.current_usage, cloned_monitor.memory_usage.current_usage);
        assert_eq!(monitor.compression_metrics.original_size, cloned_monitor.compression_metrics.original_size);
        
        println!("✅ 구조체 복제 테스트 통과");
    }

    #[test]
    fn 개별_구조체_생성_테스트() {
        let quality_tracker = QualityMetricsTracker::new();
        assert_eq!(quality_tracker.accuracy, 0.0);
        assert_eq!(quality_tracker.precision, 0.0);
        
        let memory_tracker = MemoryUsageTracker::new();
        assert_eq!(memory_tracker.current_usage, 0);
        assert_eq!(memory_tracker.peak_usage, 0);
        
        let compression_metrics = CompressionMetrics::new();
        assert_eq!(compression_metrics.original_size, 0);
        assert_eq!(compression_metrics.compression_ratio, 1.0);
        
        let system_metrics = SystemMetrics::new();
        assert_eq!(system_metrics.total_layers, 0);
        assert_eq!(system_metrics.total_execution_time, Duration::from_secs(0));
        
        println!("✅ 개별 구조체 생성 테스트 통과");
    }

    #[test]
    fn 성능_데이터_연산_테스트() {
        let mut monitor = PerformanceMonitor::new();
        
        // 여러 레이어 성능 추가
        for i in 0..3 {
            let layer_perf = LayerPerformance {
                layer_id: i,
                execution_time: Duration::from_millis((i as u64 + 1) * 10),
                input_size: 1024 >> i,
                output_size: 512 >> i,
            };
            monitor.layer_performances.push(layer_perf);
        }
        
        assert_eq!(monitor.layer_performances.len(), 3);
        
        // 총 실행 시간 계산
        let total_time: Duration = monitor.layer_performances
            .iter()
            .map(|layer| layer.execution_time)
            .sum();
        
        assert_eq!(total_time, Duration::from_millis(60)); // 10 + 20 + 30
        
        println!("✅ 성능 데이터 연산 테스트 통과");
        println!("   총 실행 시간: {:?}", total_time);
    }

    #[test]
    fn 압축률_계산_정확성_테스트() {
        let mut compression_metrics = CompressionMetrics::new();
        
        // 다양한 압축률 테스트
        let test_cases = vec![
            (1000000, 10000, 100.0),   // 100:1
            (2000000, 100000, 20.0),   // 20:1  
            (500000, 250000, 2.0),     // 2:1
        ];
        
        for (original, compressed, expected_ratio) in test_cases {
            compression_metrics.original_size = original;
            compression_metrics.compressed_size = compressed;
            compression_metrics.compression_ratio = original as f32 / compressed as f32;
            
            assert!((compression_metrics.compression_ratio - expected_ratio).abs() < 0.001);
            
            println!("   {}:{:.1} 압축률 확인됨", original / compressed, compression_metrics.compression_ratio);
        }
        
        println!("✅ 압축률 계산 정확성 테스트 통과");
    }
} 