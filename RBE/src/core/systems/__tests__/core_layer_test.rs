//! # 핵심 레이어 단위테스트
//!
//! EncodedLayer와 FusedEncodedLayer의 기능 검증

use crate::core::systems::core_layer::{EncodedLayer, FusedEncodedLayer};
use crate::core::packed_params::{HybridEncodedBlock, RbeParameters, Packed128, ResidualCoefficient, TransformType};
use nalgebra::DVector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 인코딩된_레이어_생성_테스트() {
        // 테스트 블록 생성
        let rbe_params: RbeParameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let residuals = vec![
            ResidualCoefficient { index: (0, 0), value: 0.01 },
            ResidualCoefficient { index: (0, 1), value: 0.02 },
            ResidualCoefficient { index: (1, 0), value: 0.03 },
            ResidualCoefficient { index: (1, 1), value: 0.04 },
        ];
        let block = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params,
            residuals,
            transform_type: TransformType::Dct,
        };
        
        let blocks = vec![vec![block]];
        let layer = EncodedLayer::new(blocks, 4, 4);
        
        assert_eq!(layer.block_rows, 1);
        assert_eq!(layer.block_cols, 1);
        assert_eq!(layer.total_rows, 4);
        assert_eq!(layer.total_cols, 4);
        
        println!("✅ 인코딩된 레이어 생성 성공");
    }

    #[test]
    fn 융합_레이어_생성_테스트() {
        // Packed128 시드 생성
        let seed = Packed128 {
            hi: 0x123456789ABCDEF0,
            lo: 0x0FEDCBA987654321,
        };
        
        let weight_seeds = vec![vec![seed]];
        let layer = FusedEncodedLayer::new(weight_seeds, 8, 8, 8, 8);
        
        assert_eq!(layer.block_rows, 1);
        assert_eq!(layer.block_cols, 1);
        assert_eq!(layer.block_height, 8);
        assert_eq!(layer.block_width, 8);
        assert_eq!(layer.total_rows, 8);
        assert_eq!(layer.total_cols, 8);
        
        println!("✅ 융합 레이어 생성 성공");
    }

    #[test]
    fn 융합_순전파_정밀도_테스트() {
        // 테스트 시드
        let seed = Packed128 {
            hi: 0x123456789ABCDEF0,
            lo: 0x3F0000003F000000, // r=0.5, theta=0.5
        };
        
        let weight_seeds = vec![vec![seed]];
        let layer = FusedEncodedLayer::new(weight_seeds, 4, 4, 4, 4);
        
        // 입력 벡터
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        
        // 순전파 실행
        let output = layer.fused_forward_precise(&input);
        
        assert_eq!(output.len(), 4);
        
        // 출력값이 합리적인 범위인지 확인
        for &val in output.iter() {
            assert!(val.is_finite(), "출력값이 무한대가 아니어야 함");
        }
        
        println!("✅ 융합 순전파 정밀도 테스트 통과");
        println!("   입력: {:?}", input.as_slice());
        println!("   출력: {:?}", output.as_slice());
    }

    #[test]
    fn 융합_역전파_정밀도_테스트() {
        let seed = Packed128 {
            hi: 0x123456789ABCDEF0,
            lo: 0x3F0000003F000000, // r=0.5, theta=0.5
        };
        
        let weight_seeds = vec![vec![seed]];
        let mut layer = FusedEncodedLayer::new(weight_seeds, 4, 4, 4, 4);
        
        // 입력과 그래디언트
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let loss_gradient = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        
        // 초기 가중치 백업
        let initial_seed = layer.weight_seeds[0][0];
        
        // 역전파 실행
        let input_gradient = layer.fused_backward_precise(&input, &loss_gradient, 0.01);
        
        assert_eq!(input_gradient.len(), 4);
        
        // 가중치가 업데이트되었는지 확인
        let updated_seed = layer.weight_seeds[0][0];
        assert_ne!(initial_seed.lo, updated_seed.lo, "가중치가 업데이트되어야 함");
        
        // 입력 그래디언트가 합리적인지 확인
        for &val in input_gradient.iter() {
            assert!(val.is_finite(), "입력 그래디언트가 무한대가 아니어야 함");
        }
        
        println!("✅ 융합 역전파 정밀도 테스트 통과");
        println!("   입력 그래디언트: {:?}", input_gradient.as_slice());
    }

    #[test]
    fn adam_옵티마이저_융합_역전파_테스트() {
        let seed = Packed128 {
            hi: 0x123456789ABCDEF0,
            lo: 0x3F0000003F000000,
        };
        
        let weight_seeds = vec![vec![seed]];
        let mut layer = FusedEncodedLayer::new(weight_seeds, 4, 4, 4, 4);
        
        // Adam 모멘텀 초기화
        let mut momentum_r = vec![vec![0.0]];
        let mut velocity_r = vec![vec![0.0]];
        let mut momentum_theta = vec![vec![0.0]];
        let mut velocity_theta = vec![vec![0.0]];
        
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let loss_gradient = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        
        // Adam 역전파 실행
        let input_gradient = layer.fused_backward_adam(
            &input,
            &loss_gradient,
            &mut momentum_r,
            &mut velocity_r,
            &mut momentum_theta,
            &mut velocity_theta,
            1, // epoch
            0.001, // learning_rate
        );
        
        assert_eq!(input_gradient.len(), 4);
        
        // 모멘텀이 업데이트되었는지 확인
        assert_ne!(momentum_r[0][0], 0.0, "r 모멘텀이 업데이트되어야 함");
        assert_ne!(velocity_r[0][0], 0.0, "r 속도가 업데이트되어야 함");
        
        println!("✅ Adam 옵티마이저 융합 역전파 테스트 통과");
        println!("   r 모멘텀: {:.6}", momentum_r[0][0]);
        println!("   r 속도: {:.6}", velocity_r[0][0]);
    }

    #[test]
    fn 인코딩된_레이어_순전파_테스트() {
        let rbe_params: RbeParameters = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005];
        let mut residuals = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                residuals.push(ResidualCoefficient { index: (i, j), value: 0.01 });
            }
        }
        let block = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params,
            residuals,
            transform_type: TransformType::Dct,
        };
        
        let blocks = vec![vec![block]];
        let layer = EncodedLayer::new(blocks, 4, 4);
        
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = layer.fused_forward(&input);
        
        assert_eq!(output.len(), 4);
        
        // 출력값이 유효한지 확인
        for &val in output.iter() {
            assert!(val.is_finite(), "출력값이 유효해야 함");
        }
        
        println!("✅ 인코딩된 레이어 순전파 테스트 통과");
        println!("   출력 평균: {:.6}", output.mean());
    }

    #[test]
    fn 인코딩된_레이어_역전파_테스트() {
        let rbe_params: RbeParameters = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005];
        let mut residuals = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                residuals.push(ResidualCoefficient { index: (i, j), value: 0.01 });
            }
        }
        let block = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params,
            residuals,
            transform_type: TransformType::Dct,
        };
        
        let blocks = vec![vec![block]];
        let layer = EncodedLayer::new(blocks, 4, 4);
        
        let input = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let loss_gradient = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        
        let (input_gradient, param_grads) = layer.fused_backward(&input, &loss_gradient);
        
        assert_eq!(input_gradient.len(), 4);
        assert_eq!(param_grads.len(), 1);
        assert_eq!(param_grads[0].len(), 1);
        
        // 그래디언트가 유효한지 확인
        for &val in input_gradient.iter() {
            assert!(val.is_finite(), "입력 그래디언트가 유효해야 함");
        }
        
        println!("✅ 인코딩된 레이어 역전파 테스트 통과");
    }

    #[test]
    fn 다양한_블록_크기_테스트() {
        let test_cases = vec![
            (2, 2, 8, 8),   // 작은 블록, 큰 레이어
            (8, 8, 8, 8),   // 블록과 레이어 크기 동일
            (4, 6, 12, 18), // 비정방형
        ];
        
        for (block_h, block_w, total_h, total_w) in test_cases {
            let seed = Packed128 {
                hi: 0x123456789ABCDEF0,
                lo: 0x3F0000003F000000,
            };
            
            // 필요한 블록 수 계산
            let blocks_h = (total_h + block_h - 1) / block_h;
            let blocks_w = (total_w + block_w - 1) / block_w;
            
            let mut weight_seeds = Vec::new();
            for _ in 0..blocks_h {
                let mut row = Vec::new();
                for _ in 0..blocks_w {
                    row.push(seed);
                }
                weight_seeds.push(row);
            }
            
            let layer = FusedEncodedLayer::new(weight_seeds, block_h, block_w, total_h, total_w);
            
            let input = DVector::from_element(total_w, 1.0);
            let output = layer.fused_forward_precise(&input);
            
            assert_eq!(output.len(), total_h);
            
            println!("✅ 블록 크기 {}×{}, 레이어 크기 {}×{} 테스트 통과", 
                    block_h, block_w, total_h, total_w);
        }
    }
} 