//! # 시스템 구성 설정
//!
//! 7.2 시스템 구성 설정
//! 하이브리드 학습 시스템의 설정과 하이퍼파라미터들

use crate::matrix::QualityLevel;

/// 7.2 시스템 구성 설정
#[derive(Debug, Clone)]
pub struct SystemConfiguration {
    /// 레이어 크기들
    pub layer_sizes: Vec<(usize, usize)>,
    /// 압축률 설정
    pub compression_ratio: f32,
    /// 품질 레벨
    pub quality_level: QualityLevel,
    /// 학습 하이퍼파라미터
    pub learning_params: LearningParameters,
    /// 하드웨어 설정
    pub hardware_config: HardwareConfiguration,
    /// 최적화 설정
    pub optimization_config: OptimizationConfiguration,
    /// 메모리 설정
    pub memory_config: MemoryConfiguration,
}

/// 7.2.1 학습 파라미터
#[derive(Debug, Clone)]
pub struct LearningParameters {
    /// 기본 학습률 (별칭)
    pub learning_rate: f32,
    /// 기본 학습률
    pub base_learning_rate: f32,
    /// 에포크 수 (별칭)
    pub epochs: usize,
    /// 최대 에포크
    pub max_epochs: usize,
    /// 적응적 학습률 설정
    pub adaptive_lr_config: AdaptiveLearningRateConfig,
    /// 손실 함수 가중치
    pub loss_weights: LossWeights,
    /// 배치 크기
    pub batch_size: usize,
}

/// 7.2.2 적응적 학습률 설정
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRateConfig {
    /// 초기 학습률
    pub initial_lr: f32,
    /// 최소 학습률
    pub min_lr: f32,
    /// 최대 학습률
    pub max_lr: f32,
    /// 학습률 조정 인자
    pub adjustment_factor: f32,
    /// 수렴 판단 임계값
    pub convergence_threshold: f32,
}

/// 7.2.3 멀티모달 손실 함수 가중치
#[derive(Debug, Clone)]
pub struct LossWeights {
    /// 데이터 손실 가중치
    pub data_loss_weight: f32,
    /// 푸앵카레 정규화 가중치
    pub poincare_regularization_weight: f32,
    /// 상태 분포 균형 가중치
    pub state_balance_weight: f32,
    /// 잔차 희소성 가중치
    pub sparsity_weight: f32,
}

/// 7.2.4 하드웨어 구성
#[derive(Debug, Clone)]
pub struct HardwareConfiguration {
    /// CPU 스레드 수
    pub num_cpu_threads: usize,
    /// GPU 사용 여부
    pub use_gpu: bool,
    /// GPU 장치 ID
    pub gpu_device_id: usize,
    /// 혼합 정밀도 활성화
    pub enable_mixed_precision: bool,
    /// 메모리 풀 크기
    pub memory_pool_size: usize,
    /// SIMD 최적화 활성화
    pub enable_simd: bool,
}

/// 7.2.5 최적화 구성
#[derive(Debug, Clone)]
pub struct OptimizationConfiguration {
    /// 블록 크기 임계값
    pub block_size_threshold: usize,
    /// 희소성 활성화
    pub enable_sparsity: bool,
    /// 희소성 임계값
    pub sparsity_threshold: f32,
    /// 양자화 활성화
    pub enable_quantization: bool,
    /// 양자화 비트 수
    pub quantization_bits: usize,
}

/// 7.2.6 메모리 구성
#[derive(Debug, Clone)]
pub struct MemoryConfiguration {
    /// 캐시 크기 (MB)
    pub cache_size_mb: usize,
    /// 메모리 매핑 활성화
    pub enable_memory_mapping: bool,
    /// 가중치 미리 로드
    pub preload_weights: bool,
    /// 메모리 풀 크기 (MB)
    pub memory_pool_size_mb: usize,
}

// Default 구현들

impl Default for SystemConfiguration {
    fn default() -> Self {
        Self {
            layer_sizes: vec![(784, 256), (256, 128), (128, 10)],
            compression_ratio: 1000.0,
            quality_level: QualityLevel::High,
            learning_params: LearningParameters::default(),
            hardware_config: HardwareConfiguration::default(),
            optimization_config: OptimizationConfiguration::default(),
            memory_config: MemoryConfiguration::default(),
        }
    }
}

impl Default for LearningParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            base_learning_rate: 0.001,
            epochs: 100,
            max_epochs: 100,
            adaptive_lr_config: AdaptiveLearningRateConfig::default(),
            loss_weights: LossWeights::default(),
            batch_size: 32,
        }
    }
}

impl Default for AdaptiveLearningRateConfig {
    fn default() -> Self {
        Self {
            initial_lr: 0.001,
            min_lr: 1e-6,
            max_lr: 0.1,
            adjustment_factor: 0.5,
            convergence_threshold: 1e-4,
        }
    }
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            data_loss_weight: 1.0,
            poincare_regularization_weight: 0.01,
            state_balance_weight: 0.001,
            sparsity_weight: 0.0001,
        }
    }
}

impl Default for HardwareConfiguration {
    fn default() -> Self {
        Self {
            num_cpu_threads: std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4),
            use_gpu: false,
            gpu_device_id: 0,
            enable_mixed_precision: false,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            enable_simd: true,
        }
    }
}

impl Default for OptimizationConfiguration {
    fn default() -> Self {
        Self {
            block_size_threshold: 64,
            enable_sparsity: true,
            sparsity_threshold: 0.01,
            enable_quantization: false,
            quantization_bits: 8,
        }
    }
}

impl Default for MemoryConfiguration {
    fn default() -> Self {
        Self {
            cache_size_mb: 256,
            enable_memory_mapping: false,
            preload_weights: true,
            memory_pool_size_mb: 512,
        }
    }
} 