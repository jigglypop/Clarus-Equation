use std::fmt;
use serde::{Serialize, Deserialize};

/// 하이퍼볼릭 압축 모드 - 푸앵카레 공간의 기하학적 특성 반영
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HyperbolicPackingMode {
    /// 측지선(Geodesic) - 최단 경로 표현 (128비트)
    Geodesic = 0,
    /// 호로사이클(Horocycle) - 평행선 표현 (192비트)
    Horocycle = 1,
    /// 이중곡률(Bicurvature) - 복잡한 곡률 표현 (256비트)
    Bicurvature = 2,
    /// 리만(Riemannian) - 완전한 리만 기하 표현 (512비트)
    Riemannian = 3,
    /// 적응형 메트릭(Adaptive Metric) - 데이터 맞춤형
    AdaptiveMetric = 4,
    /// 계층적 푸앵카레(Hierarchical Poincaré) - 다중 스케일
    HierarchicalPoincare = 5,
    /// 사용자 정의
    Custom = 15,
}

/// 하이퍼볼릭 텐서 - 푸앵카레 공간에서의 압축된 텐서 표현
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbolicTensor {
    /// 헤더: 모드(8) + 메타데이터(24)
    pub header: u32,
    /// 가변 길이 페이로드 - 압축된 텐서 데이터
    pub payload: Vec<u64>,
}

impl HyperbolicTensor {
    pub fn new(mode: HyperbolicPackingMode, metadata: u32) -> Self {
        let header = ((mode as u32) << 24) | (metadata & 0xFFFFFF);
        Self {
            header,
            payload: Vec::new(),
        }
    }
    
    pub fn get_mode(&self) -> HyperbolicPackingMode {
        match (self.header >> 24) & 0xFF {
            0 => HyperbolicPackingMode::Geodesic,
            1 => HyperbolicPackingMode::Horocycle,
            2 => HyperbolicPackingMode::Bicurvature,
            3 => HyperbolicPackingMode::Riemannian,
            4 => HyperbolicPackingMode::AdaptiveMetric,
            5 => HyperbolicPackingMode::HierarchicalPoincare,
            15 => HyperbolicPackingMode::Custom,
            _ => HyperbolicPackingMode::Geodesic,
        }
    }
    
    pub fn get_metadata(&self) -> u32 {
        self.header & 0xFFFFFF
    }
    
    /// 측지선 인코딩 - 하이퍼볼릭 공간의 최단 경로 (128비트)
    pub fn encode_geodesic(params: &[f32; 8], residuals: &[f32]) -> Self {
        let mut block = Self::new(HyperbolicPackingMode::Geodesic, residuals.len() as u32);
        
        // 첫 64비트: rank-4 파라미터
        let p1 = ((params[0].to_bits() as u64) << 32) | (params[1].to_bits() as u64);
        let p2 = ((params[2].to_bits() as u64) << 32) | (params[3].to_bits() as u64);
        block.payload.push(p1);
        block.payload.push(p2);
        
        // 잔차 인코딩 (16비트씩)
        if residuals.len() > 0 {
            let mut residual_bits = 0u64;
            for (i, &r) in residuals.iter().take(4).enumerate() {
                let r_16 = half::f16::from_f32(r).to_bits();
                residual_bits |= (r_16 as u64) << (i * 16);
            }
            block.payload.push(residual_bits);
        }
        
        block
    }
    
    /// 호로사이클 인코딩 - 평행선 표현 (192비트)
    pub fn encode_horocycle(params: &[f32; 8], residuals: &[f32]) -> Self {
        let mut block = Self::new(HyperbolicPackingMode::Horocycle, residuals.len() as u32);
        
        // rank-4 파라미터 (128비트)
        let p1 = ((params[0].to_bits() as u64) << 32) | (params[1].to_bits() as u64);
        let p2 = ((params[2].to_bits() as u64) << 32) | (params[3].to_bits() as u64);
        block.payload.push(p1);
        block.payload.push(p2);
        
        // 잔차 최대 8개 (64비트) - f16으로 저장
        if residuals.len() > 0 {
            let mut residual_bits = 0u64;
            for (i, &r) in residuals.iter().take(4).enumerate() {
                let r_16 = half::f16::from_f32(r).to_bits();
                residual_bits |= (r_16 as u64) << (i * 16);
            }
            block.payload.push(residual_bits);
            
            if residuals.len() > 4 {
                let mut residual_bits2 = 0u64;
                for (i, &r) in residuals.iter().skip(4).take(4).enumerate() {
                    let r_16 = half::f16::from_f32(r).to_bits();
                    residual_bits2 |= (r_16 as u64) << (i * 16);
                }
                block.payload.push(residual_bits2);
            }
        }
        
        block
    }
    
    /// 이중곡률 인코딩 - 복잡한 곡률 표현 (256비트)
    pub fn encode_bicurvature(params: &[f32], residuals: &[f32]) -> Self {
        let mut block = Self::new(HyperbolicPackingMode::Bicurvature, 
                                 ((params.len() as u32) << 16) | (residuals.len() as u32));
        
        // rank-8 파라미터 (256비트)
        for chunk in params.chunks(2) {
            if chunk.len() == 2 {
                let bits = ((chunk[0].to_bits() as u64) << 32) | (chunk[1].to_bits() as u64);
                block.payload.push(bits);
            } else if chunk.len() == 1 {
                let bits = (chunk[0].to_bits() as u64) << 32;
                block.payload.push(bits);
            }
        }
        
        // 잔차 압축 (4비트 양자화)
        for chunk in residuals.chunks(16) {
            let mut bits = 0u64;
            for (i, &r) in chunk.iter().enumerate() {
                let r_4 = (r * 7.0).round().clamp(-8.0, 7.0) as i8;
                bits |= ((r_4 as u8 & 0xF) as u64) << (i * 4);
            }
            block.payload.push(bits);
        }
        
        block
    }
    
    /// 적응형 메트릭 인코딩 - 데이터에 맞춤형 하이퍼볼릭 표현
    pub fn encode_adaptive_metric(data: &[f32], rows: usize, cols: usize) -> Self {
        // 데이터 분석
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let sparsity = data.iter().filter(|&&x| x.abs() < 1e-6).count() as f32 / data.len() as f32;
        
        // 메타데이터: 분산(8비트) + 희소성(8비트) + 크기(8비트)
        let metadata = ((variance * 255.0).round() as u32) << 16
                     | ((sparsity * 255.0).round() as u32) << 8
                     | ((rows.min(255)) as u32);
        
        let mut block = Self::new(HyperbolicPackingMode::AdaptiveMetric, metadata);
        
        // 희소성이 높으면 인덱스-값 쌍으로 저장
        if sparsity > 0.8 {
            let mut sparse_data = Vec::new();
            for (i, &val) in data.iter().enumerate() {
                if val.abs() > 1e-6 {
                    sparse_data.push((i as u16, half::f16::from_f32(val)));
                }
            }
            
            // 개수 저장
            block.payload.push(sparse_data.len() as u64);
            
            // 인덱스-값 쌍 저장 (32비트씩)
            for chunk in sparse_data.chunks(2) {
                let mut bits = 0u64;
                for (i, &(idx, val)) in chunk.iter().enumerate() {
                    bits |= ((idx as u64) << (i * 32)) | ((val.to_bits() as u64) << (i * 32 + 16));
                }
                block.payload.push(bits);
            }
        } else {
            // 일반적인 경우 - SVD 기반 압축
            // 간단한 rank-k 근사 (여기서는 rank-4)
            let k = 4;
            let mut params = vec![0.0f32; k * 2];
            
            // 간단한 평균 기반 근사 (실제로는 SVD 사용)
            for i in 0..k {
                params[i] = data[i * data.len() / k];
                params[k + i] = data[(i + 1) * data.len() / k - 1];
            }
            
            for chunk in params.chunks(2) {
                let bits = ((chunk[0].to_bits() as u64) << 32) | (chunk[1].to_bits() as u64);
                block.payload.push(bits);
            }
        }
        
        block
    }
    
    /// 하이퍼볼릭 디코딩 - 압축된 기하학적 표현을 복원
    pub fn decode(&self, rows: usize, cols: usize) -> Vec<f32> {
        match self.get_mode() {
            HyperbolicPackingMode::Geodesic => self.decode_geodesic(rows, cols),
            HyperbolicPackingMode::Horocycle => self.decode_horocycle(rows, cols),
            HyperbolicPackingMode::Bicurvature => self.decode_bicurvature(rows, cols),
            HyperbolicPackingMode::AdaptiveMetric => self.decode_adaptive_metric(rows, cols),
            _ => vec![0.0; rows * cols],
        }
    }
    
    fn decode_geodesic(&self, rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows * cols];
        
        if self.payload.len() >= 2 {
            // rank-4 파라미터 복원
            let p0 = f32::from_bits((self.payload[0] >> 32) as u32);
            let p1 = f32::from_bits(self.payload[0] as u32);
            let p2 = f32::from_bits((self.payload[1] >> 32) as u32);
            let p3 = f32::from_bits(self.payload[1] as u32);
            
            // 기본 패턴 생성
            for i in 0..rows {
                for j in 0..cols {
                    let x = j as f32 / cols as f32;
                    let y = i as f32 / rows as f32;
                    result[i * cols + j] = p0 + p1 * x + p2 * y + p3 * x * y;
                }
            }
            
            // 잔차 추가
            if self.payload.len() > 2 {
                let residual_bits = self.payload[2];
                for k in 0..4 {
                    let r_16_bits = ((residual_bits >> (k * 16)) & 0xFFFF) as u16;
                    let r = half::f16::from_bits(r_16_bits).to_f32();
                    // 주요 위치에 잔차 추가
                    let idx = k * rows * cols / 4;
                    if idx < result.len() {
                        result[idx] += r;
                    }
                }
            }
        }
        
        result
    }
    
    fn decode_horocycle(&self, rows: usize, cols: usize) -> Vec<f32> {
        let mut result = self.decode_geodesic(rows, cols);
        
        // 추가 잔차 적용 (f16 형식)
        if self.payload.len() > 2 {
            // 첫 4개 잔차
            let residual_bits = self.payload[2];
            for k in 0..4 {
                let r_16_bits = ((residual_bits >> (k * 16)) & 0xFFFF) as u16;
                let r = half::f16::from_bits(r_16_bits).to_f32();
                let idx = k * rows * cols / 8;
                if idx < result.len() {
                    result[idx] += r;
                }
            }
            
            // 다음 4개 잔차
            if self.payload.len() > 3 {
                let residual_bits2 = self.payload[3];
                for k in 0..4 {
                    let r_16_bits = ((residual_bits2 >> (k * 16)) & 0xFFFF) as u16;
                    let r = half::f16::from_bits(r_16_bits).to_f32();
                    let idx = (k + 4) * rows * cols / 8;
                    if idx < result.len() {
                        result[idx] += r;
                    }
                }
            }
        }
        
        result
    }
    
    fn decode_bicurvature(&self, rows: usize, cols: usize) -> Vec<f32> {
        let mut result = vec![0.0f32; rows * cols];
        let metadata = self.get_metadata();
        let param_count = (metadata >> 16) as usize;
        let residual_count = (metadata & 0xFFFF) as usize;
        
        // rank-k 파라미터 복원
        let mut params = Vec::new();
        for i in 0..self.payload.len().min(param_count / 2 + 1) {
            params.push(f32::from_bits((self.payload[i] >> 32) as u32));
            params.push(f32::from_bits(self.payload[i] as u32));
        }
        
        // 고차 기저 함수 적용
        for i in 0..rows {
            for j in 0..cols {
                let x = j as f32 / cols as f32;
                let y = i as f32 / rows as f32;
                let mut val = 0.0;
                
                // rank-8 기저
                if params.len() >= 8 {
                    val = params[0] + params[1] * x + params[2] * y + params[3] * x * y
                        + params[4] * (2.0 * x * x - 1.0)
                        + params[5] * (2.0 * y * y - 1.0)
                        + params[6] * x * (2.0 * y * y - 1.0)
                        + params[7] * y * (2.0 * x * x - 1.0);
                }
                
                result[i * cols + j] = val;
            }
        }
        
        // 4비트 잔차 복원
        let residual_start = param_count / 2 + 1;
        for i in residual_start..self.payload.len() {
            let bits = self.payload[i];
            for k in 0..16 {
                let r_4 = ((bits >> (k * 4)) & 0xF) as i8;
                let r = if r_4 >= 8 { (r_4 - 16) as f32 / 7.0 } else { r_4 as f32 / 7.0 };
                let idx = (i - residual_start) * 16 + k;
                if idx < result.len() && idx < residual_count {
                    result[idx] += r;
                }
            }
        }
        
        result
    }
    
    fn decode_adaptive_metric(&self, rows: usize, cols: usize) -> Vec<f32> {
        let metadata = self.get_metadata();
        let variance = ((metadata >> 16) & 0xFF) as f32 / 255.0;
        let sparsity = ((metadata >> 8) & 0xFF) as f32 / 255.0;
        
        let mut result = vec![0.0f32; rows * cols];
        
        if sparsity > 0.8 && self.payload.len() > 0 {
            // 희소 데이터 복원
            let count = self.payload[0] as usize;
            for i in 1..self.payload.len() {
                let bits = self.payload[i];
                for k in 0..2 {
                    let idx = ((bits >> (k * 32)) & 0xFFFF) as usize;
                    let val_bits = ((bits >> (k * 32 + 16)) & 0xFFFF) as u16;
                    let val = half::f16::from_bits(val_bits).to_f32();
                    if idx < result.len() {
                        result[idx] = val;
                    }
                }
            }
        } else {
            // 일반 데이터 복원
            for i in 0..self.payload.len().min(4) {
                let p0 = f32::from_bits((self.payload[i] >> 32) as u32);
                let p1 = f32::from_bits(self.payload[i] as u32);
                
                // 선형 보간
                let start_idx = i * rows * cols / 4;
                let end_idx = ((i + 1) * rows * cols / 4).min(rows * cols);
                for j in start_idx..end_idx {
                    let t = (j - start_idx) as f32 / (end_idx - start_idx) as f32;
                    result[j] = p0 * (1.0 - t) + p1 * t;
                }
            }
        }
        
        result
    }
    
    /// 압축률 계산
    pub fn compression_ratio(&self, original_size: usize) -> f32 {
        let compressed_bits = 32 + self.payload.len() * 64; // 헤더 + 페이로드
        let original_bits = original_size * 32; // f32
        original_bits as f32 / compressed_bits as f32
    }
    
    /// 메모리 사용량 (바이트)
    pub fn size_bytes(&self) -> usize {
        4 + self.payload.len() * 8
    }
}

impl fmt::Display for HyperbolicTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperbolicTensor({:?}, {} bytes)", self.get_mode(), self.size_bytes())
    }
}

/// 계층적 하이퍼볼릭 텐서 - 멀티스케일 표현
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalHyperbolicTensor {
    /// 레벨별 하이퍼볼릭 텐서
    pub levels: Vec<HyperbolicTensor>,
    /// 레벨 간 연결 정보
    pub connections: Vec<(u8, u8, f32)>, // (from_level, to_level, weight)
}

impl HierarchicalHyperbolicTensor {
    pub fn new() -> Self {
        Self {
            levels: Vec::new(),
            connections: Vec::new(),
        }
    }
    
    pub fn add_level(&mut self, tensor: HyperbolicTensor) {
        self.levels.push(tensor);
    }
    
    pub fn add_connection(&mut self, from: u8, to: u8, weight: f32) {
        self.connections.push((from, to, weight));
    }
    
    /// 멀티스케일 디코딩
    pub fn decode(&self, rows: usize, cols: usize) -> Vec<f32> {
        if self.levels.is_empty() {
            return vec![0.0; rows * cols];
        }
        
        // 각 레벨 디코딩
        let mut level_outputs: Vec<Vec<f32>> = self.levels.iter()
            .map(|block| block.decode(rows, cols))
            .collect();
        
        // 연결 적용
        for &(from, to, weight) in &self.connections {
            if (from as usize) < level_outputs.len() && (to as usize) < level_outputs.len() {
                let from_data = level_outputs[from as usize].clone();
                for (i, &val) in from_data.iter().enumerate() {
                    level_outputs[to as usize][i] += val * weight;
                }
            }
        }
        
        // 최종 레벨 반환
        level_outputs.last().unwrap().clone()
    }
} 