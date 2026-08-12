//! RBE 기반 융합 순전파 테스트

#[cfg(test)]
mod tests {
    use crate::core::{
        decoder::fused_forward::{FusedForwardPass, BlockLayout},
        packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient},
    };
    
    fn create_test_block(rows: usize, cols: usize, seed: f32) -> HybridEncodedBlock {
        HybridEncodedBlock {
            rows,
            cols,
            rbe_params: [seed, seed * 0.5, seed * 0.3, seed * 0.2, 
                        seed * 0.15, seed * 0.1, seed * 0.08, seed * 0.05],
            residuals: vec![],
            transform_type: TransformType::Dct,
        }
    }
    
    #[test]
    fn test_블록_GEMV_기본동작() {
        let fused_forward = FusedForwardPass::new();
        
        // 4x4 블록 하나
        let blocks = vec![create_test_block(4, 4, 1.0)];
        let block_layout = BlockLayout::new(4, 4, 4);
        
        let input = vec![1.0, 0.5, -0.5, 1.0];
        let mut output = vec![0.0; 4];
        
        fused_forward.block_gemv(&blocks, &input, &mut output, &block_layout);
        
        // 출력이 0이 아닌지 확인
        assert!(output.iter().any(|&x| x != 0.0));
        
        // 모든 값이 유한한지 확인
        for &val in &output {
            assert!(val.is_finite());
        }
    }
    
    #[test]
    fn test_여러_블록_GEMV() {
        let fused_forward = FusedForwardPass::new();
        
        // 8x8 행렬을 4x4 블록 4개로
        let blocks = vec![
            create_test_block(4, 4, 1.0),
            create_test_block(4, 4, 0.8),
            create_test_block(4, 4, 0.6),
            create_test_block(4, 4, 0.4),
        ];
        
        let block_layout = BlockLayout::new(8, 8, 4);
        
        let input = vec![1.0; 8];
        let mut output = vec![0.0; 8];
        
        fused_forward.block_gemv(&blocks, &input, &mut output, &block_layout);
        
        // 출력 확인
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_병렬_순차_일관성() {
        let mut fused_forward_parallel = FusedForwardPass::new();
        let mut fused_forward_sequential = FusedForwardPass::new();
        fused_forward_sequential.set_parallel(false);
        
        // 큰 블록들 생성
        let blocks: Vec<HybridEncodedBlock> = (0..9).map(|i| {
            create_test_block(32, 32, 1.0 + i as f32 * 0.1)
        }).collect();
        
        let block_layout = BlockLayout::new(96, 96, 32);
        
        let input = vec![0.5; 96];
        let mut output_parallel = vec![0.0; 96];
        let mut output_sequential = vec![0.0; 96];
        
        // 병렬 처리
        fused_forward_parallel.block_gemv(&blocks, &input, &mut output_parallel, &block_layout);
        
        // 순차 처리
        fused_forward_sequential.block_gemv(&blocks, &input, &mut output_sequential, &block_layout);
        
        // 결과 비교
        for (i, (&p, &s)) in output_parallel.iter().zip(output_sequential.iter()).enumerate() {
            assert!((p - s).abs() < 1e-6, 
                   "병렬/순차 불일치 at index {}: {} vs {}", i, p, s);
        }
    }
    
    #[test]
    fn test_배치_처리() {
        let fused_forward = FusedForwardPass::new();
        
        let blocks = vec![
            create_test_block(4, 4, 1.0),
            create_test_block(4, 4, 0.5),
        ];
        
        let block_layout = BlockLayout::new(4, 8, 4);
        
        // 여러 입력 벡터
        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.5; 8],
        ];
        
        let outputs = fused_forward.batch_forward(&blocks, &inputs, &block_layout);
        
        // 출력 개수 확인
        assert_eq!(outputs.len(), 3);
        
        // 각 출력 확인
        for (i, output) in outputs.iter().enumerate() {
            assert_eq!(output.len(), 4);
            assert!(output.iter().all(|&x| x.is_finite()), 
                   "배치 {} 출력에 무한값", i);
        }
    }
    
    #[test]
    fn test_블록_레이아웃() {
        // 다양한 크기의 레이아웃 테스트
        let layouts = vec![
            (10, 10, 4),  // 정확히 나누어떨어지지 않는 경우
            (16, 16, 4),  // 정확히 나누어떨어지는 경우
            (7, 11, 3),   // 비정방 행렬
        ];
        
        for (rows, cols, block_size) in layouts {
            let layout = BlockLayout::new(rows, cols, block_size);
            
            let expected_grid_rows = (rows + block_size - 1) / block_size;
            let expected_grid_cols = (cols + block_size - 1) / block_size;
            
            assert_eq!(layout.grid_rows, expected_grid_rows);
            assert_eq!(layout.grid_cols, expected_grid_cols);
            
            // 블록 인덱스 변환 테스트
            for block_row in 0..layout.grid_rows {
                for block_col in 0..layout.grid_cols {
                    let idx = layout.get_block_index(block_row, block_col);
                    let (r, c) = layout.get_block_position(idx);
                    assert_eq!(r, block_row);
                    assert_eq!(c, block_col);
                }
            }
        }
    }
    
    #[test]
    fn test_잔차가_있는_블록() {
        let fused_forward = FusedForwardPass::new();
        
        // 잔차가 있는 블록
        let mut block = create_test_block(4, 4, 0.5);
        block.residuals = vec![
            ResidualCoefficient { index: (0, 0), value: 1.0 },
            ResidualCoefficient { index: (1, 1), value: 0.5 },
        ];
        
        let blocks = vec![block];
        let block_layout = BlockLayout::new(4, 4, 4);
        
        let input = vec![1.0; 4];
        let mut output = vec![0.0; 4];
        
        fused_forward.block_gemv(&blocks, &input, &mut output, &block_layout);
        
        // 잔차의 영향이 있는지 확인
        assert!(output.iter().any(|&x| x.abs() > 0.1));
    }
    
    #[test]
    fn test_빈_입력_처리() {
        let fused_forward = FusedForwardPass::new();
        
        // 빈 블록 리스트
        let blocks = vec![];
        let block_layout = BlockLayout::new(4, 4, 4);
        
        let input = vec![1.0; 4];
        let mut output = vec![1.0; 4]; // 0이 아닌 값으로 초기화
        
        fused_forward.block_gemv(&blocks, &input, &mut output, &block_layout);
        
        // 출력이 0으로 초기화되었는지 확인
        assert!(output.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_경계_케이스() {
        let fused_forward = FusedForwardPass::new();
        
        // 1x1 블록
        let blocks = vec![create_test_block(1, 1, 2.0)];
        let block_layout = BlockLayout::new(1, 1, 1);
        
        let input = vec![1.0];
        let mut output = vec![0.0];
        
        fused_forward.block_gemv(&blocks, &input, &mut output, &block_layout);
        
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }
} 