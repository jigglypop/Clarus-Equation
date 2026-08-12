#[cfg(test)]
mod tests {
    use crate::nlp::softmax::RBESoftmax;
    use rand::{thread_rng, Rng};
    
    #[test]
    fn 소프트맥스_생성_및_기본_동작_테스트() {
        let softmax = RBESoftmax::new(-1);
        
        // 입력 생성
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = softmax.forward(&input);
        
        // 소프트맥스 결과 검증
        assert_eq!(output.len(), input.len());
        
        // 합이 1인지 확인
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "소프트맥스 합이 1이 아님: {}", sum);
        
        // 단조 증가 속성 확인
        for i in 1..output.len() {
            assert!(output[i] > output[i-1], "소프트맥스가 단조 증가하지 않음");
        }
    }
    
    #[test]
    fn 수치안정성_테스트() {
        let softmax = RBESoftmax::new(-1);
        
        // 큰 값으로 테스트
        let large_input = vec![1000.0, 1001.0, 999.0, 1000.5];
        let output = softmax.forward(&large_input);
        
        // NaN이나 Inf 없는지 확인
        for &val in &output {
            assert!(val.is_finite(), "수치적으로 불안정한 출력: {}", val);
            assert!(val >= 0.0 && val <= 1.0, "소프트맥스 범위 벗어남: {}", val);
        }
        
        // 합이 1인지 확인
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "큰 입력에서 합이 1이 아님: {}", sum);
    }
    
    #[test]
    fn 온도_스케일링_테스트() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        
        // 다양한 온도에서 테스트
        let temperatures = vec![0.5, 1.0, 2.0, 10.0];
        let mut results = Vec::new();
        
        for temp in temperatures {
            let mut softmax = RBESoftmax::new(-1);
            softmax.temperature = temp;
            
            let output = softmax.forward(&input);
            results.push((temp, output));
        }
        
        // 온도가 낮을수록 더 sharp한 분포
        for i in 1..results.len() {
            let (temp1, out1) = &results[i-1];
            let (temp2, out2) = &results[i];
            
            if temp1 < temp2 {
                // 더 낮은 온도는 더 큰 최대값을 가짐
                let max1 = out1.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let max2 = out2.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                assert!(max1 >= max2, "온도 {}의 최대값이 온도 {}보다 작음", temp1, temp2);
            }
        }
    }
    
    #[test]
    fn 푸앵카레_가중치_소프트맥스_테스트() {
        let softmax = RBESoftmax::new(-1);
        
        // 입력
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // 사이클 상태로 테스트
        let output = softmax.forward_poincare(&input, Some(1024));
        
        // 결과 검증
        assert_eq!(output.len(), input.len());
        
        // 합이 1인지 확인
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "푸앵카레 가중 소프트맥스 합이 1이 아님: {}", sum);
    }
    
    #[test]
    fn 로그_소프트맥스_테스트() {
        let softmax = RBESoftmax::new(-1);
        
        let input = vec![1.0, 2.0, 3.0, 4.0];
        
        // 일반 소프트맥스와 로그 소프트맥스 비교
        let softmax_output = softmax.forward(&input);
        let log_softmax_output = softmax.log_forward(&input);
        
        // log(softmax(x)) ≈ log_softmax(x) 확인
        for (i, (&soft, &log_soft)) in softmax_output.iter().zip(&log_softmax_output).enumerate() {
            let expected = soft.ln();
            assert!(
                (expected - log_soft).abs() < 1e-5,
                "인덱스 {}에서 log_softmax 불일치: expected {}, got {}",
                i, expected, log_soft
            );
        }
    }
    
    #[test]
    fn 배치_처리_테스트() {
        let softmax = RBESoftmax::new(-1);
        
        // 배치 입력 생성
        let batch_size = 4;
        let seq_len = 8;
        let mut batch_input = Vec::new();
        
        let mut rng = thread_rng();
        for _ in 0..batch_size {
            let seq: Vec<f32> = (0..seq_len).map(|_| rng.gen_range(-5.0..5.0)).collect();
            batch_input.push(seq);
        }
        
        let batch_output = softmax.forward_batch(&batch_input);
        
        // 각 배치가 올바른지 검증
        assert_eq!(batch_output.len(), batch_size);
        
        for (i, output) in batch_output.iter().enumerate() {
            assert_eq!(output.len(), seq_len, "배치 {}의 출력 길이가 잘못됨", i);
            
            let sum: f32 = output.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "배치 {}의 소프트맥스 합이 1이 아님: {}",
                i, sum
            );
        }
    }
    
    #[test]
    fn 사이클_동기화_테스트() {
        let mut softmax = RBESoftmax::new(-1);
        softmax.cycle_sync = true;
        
        let input = vec![1.0, 2.0, 3.0, 4.0];
        
        // 다양한 사이클 상태에서 테스트
        let cycle_states: Vec<u16> = vec![0, 512, 1024, 1536, 2047];
        
        for state in cycle_states {
            // cycle_state를 전달하여 테스트
            let output = softmax.forward_poincare(&input, Some(state));
            
            // 결과가 유효한지 확인
            assert_eq!(output.len(), input.len());
            let sum: f32 = output.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "사이클 {}에서 합이 1이 아님: {}", state, sum);
        }
    }
    
    #[test]
    fn 곡률_파라미터_테스트() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        
        // 다양한 곡률 값으로 테스트
        let curvatures = vec![0.5, 1.0, 2.0, 5.0];
        
        for curvature in curvatures {
            let mut softmax = RBESoftmax::new(-1);
            softmax.curvature = curvature;
            
            let output = softmax.forward(&input);
            
            // 결과가 유효한지 확인
            assert_eq!(output.len(), input.len());
            let sum: f32 = output.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "곡률 {}에서 소프트맥스 합이 1이 아님: {}",
                curvature, sum
            );
        }
    }
    
    #[test]
    fn 극한값_처리_테스트() {
        let softmax = RBESoftmax::new(-1);
        
        // 모든 값이 같을 때
        let uniform_input = vec![5.0; 4];
        let uniform_output = softmax.forward(&uniform_input);
        
        for &val in &uniform_output {
            assert!(
                (val - 0.25).abs() < 1e-5,
                "균등 입력에서 소프트맥스가 균등하지 않음: {}",
                val
            );
        }
        
        // 매우 큰 차이가 있을 때
        let extreme_input = vec![-100.0, 0.0, 100.0, 0.0];
        let extreme_output = softmax.forward(&extreme_input);
        
        assert!(extreme_output[0] < 1e-10, "매우 작은 입력의 소프트맥스가 0에 가깝지 않음");
        assert!(extreme_output[2] > 0.99, "매우 큰 입력의 소프트맥스가 1에 가깝지 않음");
    }
} 