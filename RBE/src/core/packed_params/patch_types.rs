use std::collections::HashMap;
use std::sync::Arc;

/// 패치 태그 - 4비트로 패치 종류 구분
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatchTag {
    Core = 0,        // 기본 코어 (사용 안함)
    DeltaRank1 = 1,  // LoRA-style rank-1 업데이트
    DeltaScalar = 2, // 스칼라 곱셈 보정
    DeltaSVDVec = 3, // SVD 벡터 교체
    MaskDrop = 4,    // 잔차 마스킹
}

/// 단일 패치 - 20 + 32*n 비트
#[derive(Debug, Clone)]
pub struct DeltaPatch {
    pub tag: PatchTag,
    pub block_id: u16,     // 10비트 실제 사용
    pub payload_len: u8,   // 6비트 실제 사용 (32-bit 단위)
    pub payload: Vec<u32>, // 실제 데이터
    pub crc: Option<u8>,   // 선택적 CRC
}

/// 압축된 패치 (비트 패킹)
#[derive(Debug, Clone)]
pub struct PackedPatch {
    pub header: u32,      // tag(4) + block_id(10) + payload_len(6) + reserved(12)
    pub payload: Vec<u32>,
}

impl PackedPatch {
    pub fn from_delta(patch: &DeltaPatch) -> Self {
        let header = ((patch.tag as u32) << 28)
            | ((patch.block_id as u32 & 0x3FF) << 18)
            | ((patch.payload_len as u32 & 0x3F) << 12);
        
        Self {
            header,
            payload: patch.payload.clone(),
        }
    }
    
    pub fn to_delta(&self) -> DeltaPatch {
        let tag = match (self.header >> 28) & 0xF {
            1 => PatchTag::DeltaRank1,
            2 => PatchTag::DeltaScalar,
            3 => PatchTag::DeltaSVDVec,
            4 => PatchTag::MaskDrop,
            _ => PatchTag::Core,
        };
        
        let block_id = ((self.header >> 18) & 0x3FF) as u16;
        let payload_len = ((self.header >> 12) & 0x3F) as u8;
        
        DeltaPatch {
            tag,
            block_id,
            payload_len,
            payload: self.payload.clone(),
            crc: None,
        }
    }
}

/// 패치 로그 - append-only 구조
pub struct PatchLog {
    patches: Vec<PackedPatch>,
    block_index: HashMap<u16, Vec<usize>>, // block_id -> patch indices
}

impl PatchLog {
    pub fn new() -> Self {
        Self {
            patches: Vec::new(),
            block_index: HashMap::new(),
        }
    }
    
    pub fn append(&mut self, patch: DeltaPatch) {
        let packed = PackedPatch::from_delta(&patch);
        let idx = self.patches.len();
        
        self.block_index
            .entry(patch.block_id)
            .or_insert_with(Vec::new)
            .push(idx);
            
        self.patches.push(packed);
    }
    
    pub fn lookup(&self, block_id: u16) -> Vec<DeltaPatch> {
        self.block_index
            .get(&block_id)
            .map(|indices| {
                indices.iter()
                    .map(|&idx| self.patches[idx].to_delta())
                    .collect()
            })
            .unwrap_or_default()
    }
    
    pub fn size_bytes(&self) -> usize {
        self.patches.iter()
            .map(|p| 4 + p.payload.len() * 4) // header + payload
            .sum()
    }
}

/// 확장된 레이어 코드 (코어 + 패치 예약)
#[derive(Debug, Clone)]
pub struct ExtendedLayerCode {
    pub core: super::Packed128,           // 128비트 코어
    pub residuals: Vec<super::ResidualCoefficient>, // 잔차
    pub reserved: u32,                    // 32비트 예약 (패치 포인터 등)
}

impl ExtendedLayerCode {
    pub fn new(core: super::Packed128, residuals: Vec<super::ResidualCoefficient>) -> Self {
        Self {
            core,
            residuals,
            reserved: 0,
        }
    }
    
    pub fn set_patch_offset(&mut self, offset: u16) {
        self.reserved = (self.reserved & 0xFFFF0000) | (offset as u32);
    }
    
    pub fn get_patch_offset(&self) -> Option<u16> {
        let offset = (self.reserved & 0xFFFF) as u16;
        if offset > 0 { Some(offset) } else { None }
    }
}

/// 패치 적용 헬퍼 함수들
pub fn apply_delta_scalar(weights: &mut [f32], alpha: f32) {
    for w in weights.iter_mut() {
        *w *= alpha;
    }
}

pub fn apply_delta_rank1(weights: &mut [f32], rows: usize, cols: usize, u_idx: usize, v_idx: usize, scale: f32) {
    // rank-1 outer product update: W += scale * u * v^T
    let u_val = if u_idx < rows { 1.0 } else { 0.0 };
    let v_val = if v_idx < cols { 1.0 } else { 0.0 };
    
    for i in 0..rows {
        for j in 0..cols {
            if i == u_idx && j == v_idx {
                weights[i * cols + j] += scale * u_val * v_val;
            }
        }
    }
}

/// 해마 메모리 시스템
pub struct HippocampalMemory {
    pub patch_log: Arc<parking_lot::RwLock<PatchLog>>,
    pub update_count: usize,
    pub gc_threshold: usize,
}

impl HippocampalMemory {
    pub fn new(gc_threshold: usize) -> Self {
        Self {
            patch_log: Arc::new(parking_lot::RwLock::new(PatchLog::new())),
            update_count: 0,
            gc_threshold,
        }
    }
    
    pub fn add_patch(&mut self, patch: DeltaPatch) {
        self.patch_log.write().append(patch);
        self.update_count += 1;
    }
    
    pub fn should_gc(&self) -> bool {
        self.update_count >= self.gc_threshold
    }
    
    pub fn get_patches(&self, block_id: u16) -> Vec<DeltaPatch> {
        self.patch_log.read().lookup(block_id)
    }
    
    pub fn memory_usage(&self) -> usize {
        self.patch_log.read().size_bytes()
    }
} 