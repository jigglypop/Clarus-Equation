use super::super::{L1Block, L2Block, L3Block, L4Block};
use crate::packed_params::Packed128;
use rand::thread_rng;

#[test]
fn L1블록_생성_테스트() {
    let global_params = Packed128::random(&mut thread_rng());
    let l1_block = L1Block {
        row_start: 0,
        col_start: 0,
        rows: 4096,
        cols: 4096,
        l2_blocks: Vec::new(),
        global_params,
    };
    
    assert_eq!(l1_block.row_start, 0);
    assert_eq!(l1_block.col_start, 0);
    assert_eq!(l1_block.rows, 4096);
    assert_eq!(l1_block.cols, 4096);
    assert!(l1_block.l2_blocks.is_empty());
    
    println!("✅ L1블록 생성 테스트 통과");
    println!("   위치: ({}, {}), 크기: {}×{}", 
             l1_block.row_start, l1_block.col_start, l1_block.rows, l1_block.cols);
}

#[test]
fn L2블록_생성_테스트() {
    let macro_params = Packed128::random(&mut thread_rng());
    let l2_block = L2Block {
        row_start: 1024,
        col_start: 2048,
        rows: 1024,
        cols: 1024,
        l3_blocks: Vec::new(),
        macro_params,
    };
    
    assert_eq!(l2_block.row_start, 1024);
    assert_eq!(l2_block.col_start, 2048);
    assert_eq!(l2_block.rows, 1024);
    assert_eq!(l2_block.cols, 1024);
    assert!(l2_block.l3_blocks.is_empty());
    
    println!("✅ L2블록 생성 테스트 통과");
    println!("   위치: ({}, {}), 크기: {}×{}", 
             l2_block.row_start, l2_block.col_start, l2_block.rows, l2_block.cols);
}

#[test]
fn L3블록_생성_테스트() {
    let mid_params = Packed128::random(&mut thread_rng());
    let l3_block = L3Block {
        row_start: 256,
        col_start: 512,
        rows: 256,
        cols: 256,
        l4_blocks: Vec::new(),
        mid_params,
    };
    
    assert_eq!(l3_block.row_start, 256);
    assert_eq!(l3_block.col_start, 512);
    assert_eq!(l3_block.rows, 256);
    assert_eq!(l3_block.cols, 256);
    assert!(l3_block.l4_blocks.is_empty());
    
    println!("✅ L3블록 생성 테스트 통과");
    println!("   위치: ({}, {}), 크기: {}×{}", 
             l3_block.row_start, l3_block.col_start, l3_block.rows, l3_block.cols);
}

#[test]
fn L4블록_생성_테스트() {
    let detail_params = Packed128::random(&mut thread_rng());
    let l4_block = L4Block {
        row_start: 64,
        col_start: 128,
        rows: 64,
        cols: 64,
        detail_params,
    };
    
    assert_eq!(l4_block.row_start, 64);
    assert_eq!(l4_block.col_start, 128);
    assert_eq!(l4_block.rows, 64);
    assert_eq!(l4_block.cols, 64);
    
    println!("✅ L4블록 생성 테스트 통과");
    println!("   위치: ({}, {}), 크기: {}×{}", 
             l4_block.row_start, l4_block.col_start, l4_block.rows, l4_block.cols);
}

#[test]
fn 계층적_블록_구조_테스트() {
    // L4 블록 생성
    let l4_block = L4Block {
        row_start: 0,
        col_start: 0,
        rows: 64,
        cols: 64,
        detail_params: Packed128::random(&mut thread_rng()),
    };
    
    // L3 블록에 L4 블록 포함
    let mut l3_block = L3Block {
        row_start: 0,
        col_start: 0,
        rows: 256,
        cols: 256,
        l4_blocks: Vec::new(),
        mid_params: Packed128::random(&mut thread_rng()),
    };
    
    // L4 블록들의 행 추가
    let mut l4_row = Vec::new();
    l4_row.push(l4_block);
    l3_block.l4_blocks.push(l4_row);
    
    // L2 블록에 L3 블록 포함
    let mut l2_block = L2Block {
        row_start: 0,
        col_start: 0,
        rows: 1024,
        cols: 1024,
        l3_blocks: Vec::new(),
        macro_params: Packed128::random(&mut thread_rng()),
    };
    
    let mut l3_row = Vec::new();
    l3_row.push(l3_block);
    l2_block.l3_blocks.push(l3_row);
    
    // L1 블록에 L2 블록 포함
    let mut l1_block = L1Block {
        row_start: 0,
        col_start: 0,
        rows: 4096,
        cols: 4096,
        l2_blocks: Vec::new(),
        global_params: Packed128::random(&mut thread_rng()),
    };
    
    let mut l2_row = Vec::new();
    l2_row.push(l2_block);
    l1_block.l2_blocks.push(l2_row);
    
    // 계층 구조 검증
    assert_eq!(l1_block.l2_blocks.len(), 1);
    assert_eq!(l1_block.l2_blocks[0].len(), 1);
    assert_eq!(l1_block.l2_blocks[0][0].l3_blocks.len(), 1);
    assert_eq!(l1_block.l2_blocks[0][0].l3_blocks[0].len(), 1);
    assert_eq!(l1_block.l2_blocks[0][0].l3_blocks[0][0].l4_blocks.len(), 1);
    assert_eq!(l1_block.l2_blocks[0][0].l3_blocks[0][0].l4_blocks[0].len(), 1);
    
    println!("✅ 계층적 블록 구조 테스트 통과");
    println!("   L1 → L2 → L3 → L4 계층 구조 완성");
}

#[test]
fn 블록_복제_테스트() {
    let original_l4 = L4Block {
        row_start: 100,
        col_start: 200,
        rows: 64,
        cols: 64,
        detail_params: Packed128::random(&mut thread_rng()),
    };
    
    let cloned_l4 = original_l4.clone();
    
    assert_eq!(original_l4.row_start, cloned_l4.row_start);
    assert_eq!(original_l4.col_start, cloned_l4.col_start);
    assert_eq!(original_l4.rows, cloned_l4.rows);
    assert_eq!(original_l4.cols, cloned_l4.cols);
    
    println!("✅ 블록 복제 테스트 통과");
}

#[test]
fn 블록_디버그_출력_테스트() {
    let l1_block = L1Block {
        row_start: 0,
        col_start: 0,
        rows: 4096,
        cols: 4096,
        l2_blocks: Vec::new(),
        global_params: Packed128::random(&mut thread_rng()),
    };
    
    let debug_str = format!("{:?}", l1_block);
    
    assert!(debug_str.contains("L1Block"), "디버그 출력에 구조체명이 없음");
    assert!(debug_str.contains("row_start"), "디버그 출력에 필드명이 없음");
    assert!(debug_str.len() > 50, "디버그 출력이 너무 짧음");
    
    println!("✅ 블록 디버그 출력 테스트 통과");
    println!("   Debug 출력 길이: {} 문자", debug_str.len());
}

#[test]
fn 블록_크기_검증_테스트() {
    // 표준 블록 크기들
    let l1_size = 4096;
    let l2_size = 1024;
    let l3_size = 256;
    let l4_size = 64;
    
    // 크기 비율 검증
    assert_eq!(l1_size / l2_size, 4, "L1과 L2 크기 비율");
    assert_eq!(l2_size / l3_size, 4, "L2와 L3 크기 비율");
    assert_eq!(l3_size / l4_size, 4, "L3과 L4 크기 비율");
    
    // 전체 비율
    assert_eq!(l1_size / l4_size, 64, "L1과 L4 전체 크기 비율");
    
    let l1_block = L1Block {
        row_start: 0,
        col_start: 0,
        rows: l1_size,
        cols: l1_size,
        l2_blocks: Vec::new(),
        global_params: Packed128::random(&mut thread_rng()),
    };
    
    let l4_block = L4Block {
        row_start: 0,
        col_start: 0,
        rows: l4_size,
        cols: l4_size,
        detail_params: Packed128::random(&mut thread_rng()),
    };
    
    assert_eq!(l1_block.rows / l4_block.rows, 64);
    assert_eq!(l1_block.cols / l4_block.cols, 64);
    
    println!("✅ 블록 크기 검증 테스트 통과");
    println!("   L1({}×{}) : L2({}×{}) : L3({}×{}) : L4({}×{}) = 64:16:4:1", 
             l1_size, l1_size, l2_size, l2_size, l3_size, l3_size, l4_size, l4_size);
}

#[test]
fn 블록_위치_검증_테스트() {
    // 서로 다른 위치의 블록들
    let blocks = [
        (0, 0, "좌상단"),
        (1024, 0, "우상단"),
        (0, 1024, "좌하단"),
        (1024, 1024, "우하단"),
        (512, 512, "중앙"),
    ];
    
    for &(row_start, col_start, desc) in &blocks {
        let l2_block = L2Block {
            row_start,
            col_start,
            rows: 1024,
            cols: 1024,
            l3_blocks: Vec::new(),
            macro_params: Packed128::random(&mut thread_rng()),
        };
        
        assert_eq!(l2_block.row_start, row_start);
        assert_eq!(l2_block.col_start, col_start);
        
        println!("{} 블록: ({}, {})", desc, row_start, col_start);
    }
    
    println!("✅ 블록 위치 검증 테스트 통과");
}

#[test]
fn 다중_블록_배열_테스트() {
    // 2x2 L2 블록 배열 생성
    let mut l2_blocks = Vec::new();
    
    for i in 0..2 {
        let mut l2_row = Vec::new();
        for j in 0..2 {
            let l2_block = L2Block {
                row_start: i * 1024,
                col_start: j * 1024,
                rows: 1024,
                cols: 1024,
                l3_blocks: Vec::new(),
                macro_params: Packed128::random(&mut thread_rng()),
            };
            l2_row.push(l2_block);
        }
        l2_blocks.push(l2_row);
    }
    
    // 배열 구조 검증
    assert_eq!(l2_blocks.len(), 2, "행 개수");
    assert_eq!(l2_blocks[0].len(), 2, "열 개수");
    assert_eq!(l2_blocks[1].len(), 2, "열 개수");
    
    // 위치 검증
    assert_eq!(l2_blocks[0][0].row_start, 0);
    assert_eq!(l2_blocks[0][0].col_start, 0);
    assert_eq!(l2_blocks[0][1].row_start, 0);
    assert_eq!(l2_blocks[0][1].col_start, 1024);
    assert_eq!(l2_blocks[1][0].row_start, 1024);
    assert_eq!(l2_blocks[1][0].col_start, 0);
    assert_eq!(l2_blocks[1][1].row_start, 1024);
    assert_eq!(l2_blocks[1][1].col_start, 1024);
    
    println!("✅ 다중 블록 배열 테스트 통과");
    println!("   2×2 블록 배열 생성 및 위치 검증 완료");
} 