#[cfg(test)]
mod tests {
    use anyhow::Result;
    use crate::nlp::rmsnorm::{RBERMSNorm, RBERMSNormConfig};
    use crate::QualityGrade;
    
    #[test]
    fn RMS정규화_기본_동작_테스트() -> Result<()> {
        let config = RBERMSNormConfig {
            normalized_shape: 8,
            epsilon: 1e-5,
            quality_grade: QualityGrade::B,
            enable_parallel: false,
        };
        
        let mut rmsnorm = RBERMSNorm::new(config);
        rmsnorm.init_weights()?;
        
        // 테스트 입력
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = rmsnorm.forward(&input)?;
        
        // 출력 크기 확인
        assert_eq!(output.len(), input.len());
        
        // RMS 계산 검증
        let expected_rms = ((1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0 + 49.0 + 64.0) / 8.0f32).sqrt();
        let scale = 1.0 / expected_rms;
        
        // 대략적인 값 확인 (gamma가 1 근처이므로)
        for i in 0..8 {
            let expected = input[i] * scale; // gamma ≈ 1
            assert!((output[i] - expected).abs() < 0.2, 
                "output[{}] = {}, expected ≈ {}", i, output[i], expected);
        }
        
        Ok(())
    }
    
    #[test]
    fn 배치_처리_테스트() -> Result<()> {
        let config = RBERMSNormConfig {
            normalized_shape: 4,
            epsilon: 1e-6,
            quality_grade: QualityGrade::A,
            enable_parallel: true,
        };
        
        let mut rmsnorm = RBERMSNorm::new(config);
        rmsnorm.init_weights()?;
        
        // 배치 크기 3, 각 4개 원소
        let input = vec![
            1.0, 2.0, 3.0, 4.0,  // 배치 1
            5.0, 6.0, 7.0, 8.0,  // 배치 2
            9.0, 10.0, 11.0, 12.0, // 배치 3
        ];
        
        let output = rmsnorm.forward(&input)?;
        
        // 크기 확인
        assert_eq!(output.len(), 12);
        
        // 각 배치가 독립적으로 정규화되었는지 확인
        for batch in 0..3 {
            let start = batch * 4;
            let batch_output = &output[start..start + 4];
            
            // 각 배치의 RMS가 대략 1인지 확인
            let rms: f32 = batch_output.iter()
                .map(|&x| x * x)
                .sum::<f32>() / 4.0;
            let rms = rms.sqrt();
            
            assert!((rms - 1.0).abs() < 0.2, 
                "배치 {} RMS = {}, expected ≈ 1", batch, rms);
        }
        
        Ok(())
    }
    
    #[test]
    fn 수치적_안정성_테스트() -> Result<()> {
        let config = RBERMSNormConfig {
            normalized_shape: 4,
            epsilon: 1e-5,
            quality_grade: QualityGrade::S,
            enable_parallel: false,
        };
        
        let mut rmsnorm = RBERMSNorm::new(config);
        rmsnorm.init_weights()?;
        
        // 매우 작은 값들
        let tiny_input = vec![1e-8, 2e-8, 3e-8, 4e-8];
        let tiny_output = rmsnorm.forward(&tiny_input)?;
        
        // NaN이나 Inf가 없는지 확인
        for &val in &tiny_output {
            assert!(val.is_finite(), "출력에 NaN/Inf 포함");
        }
        
        // 매우 큰 값들
        let large_input = vec![1e8, 2e8, 3e8, 4e8];
        let large_output = rmsnorm.forward(&large_input)?;
        
        for &val in &large_output {
            assert!(val.is_finite(), "큰 입력에서 NaN/Inf 발생");
        }
        
        Ok(())
    }
    
    #[test]
    fn 메모리_사용량_테스트() {
        let config = RBERMSNormConfig {
            normalized_shape: 768,
            ..Default::default()
        };
        
        let mut rmsnorm = RBERMSNorm::new(config);
        rmsnorm.init_weights().unwrap();
        
        let (compressed_size, compression_ratio) = rmsnorm.memory_usage();
        
        println!("RMSNorm 메모리 사용량:");
        println!("- 압축 크기: {} bytes", compressed_size);
        println!("- 압축률: {:.1}:1", compression_ratio);
        
        // 기본적인 압축이 되었는지 확인
        assert!(compression_ratio > 1.0);
    }
    
    #[test]
    fn 영벡터_입력_테스트() -> Result<()> {
        let config = RBERMSNormConfig {
            normalized_shape: 4,
            epsilon: 1e-5,
            ..Default::default()
        };
        
        let mut rmsnorm = RBERMSNorm::new(config);
        rmsnorm.init_weights()?;
        
        // 모든 값이 0인 입력
        let zero_input = vec![0.0; 4];
        let output = rmsnorm.forward(&zero_input)?;
        
        // epsilon 때문에 0이 아닌 출력이 나와야 함
        for &val in &output {
            assert!(val.abs() < 1.0, "영벡터 입력에서 큰 출력 발생");
        }
        
        Ok(())
    }
} 