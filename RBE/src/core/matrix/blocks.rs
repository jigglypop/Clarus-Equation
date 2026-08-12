use crate::packed_params::Packed128;

/// L1 블록 (최상위 레벨)
#[derive(Debug, Clone)]
pub struct L1Block {
    /// 블록 위치
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    /// L2 하위 블록들
    pub l2_blocks: Vec<Vec<L2Block>>,
    /// 전체 블록을 표현하는 단일 파라미터
    pub global_params: Packed128,
}

/// L2 블록 (1024×1024)
#[derive(Debug, Clone)]
pub struct L2Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub l3_blocks: Vec<Vec<L3Block>>,
    pub macro_params: Packed128,
}

/// L3 블록 (256×256)
#[derive(Debug, Clone)]
pub struct L3Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub l4_blocks: Vec<Vec<L4Block>>,
    pub mid_params: Packed128,
}

/// L4 블록 (64×64, 최소 단위)
#[derive(Debug, Clone)]
pub struct L4Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub detail_params: Packed128,
} 