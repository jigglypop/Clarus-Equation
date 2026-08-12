/// Optimizer 전체 구성
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Adam 구성
    pub adam: AdamConfig,
    /// Riemannian Adam 구성  
    pub riemannian_adam: RiemannianAdamConfig,
    /// 학습률
    pub learning_rate: f32,
    /// 학습률 스케줄링
    pub lr_schedule: LearningRateSchedule,
    /// 그래디언트 클리핑
    pub gradient_clipping: Option<f32>,
    /// 가중치 감소
    pub weight_decay: f32,
}

/// Adam 옵티마이저 구성
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// 베타1 파라미터 (1차 모멘트 지수 감소율)
    pub beta1: f32,
    /// 베타2 파라미터 (2차 모멘트 지수 감소율)
    pub beta2: f32,
    /// 엡실론 (수치 안정성을 위한 작은 값)
    pub epsilon: f32,
}

/// Riemannian Adam 옵티마이저 구성
#[derive(Debug, Clone)]
pub struct RiemannianAdamConfig {
    /// 베타1 파라미터 (1차 모멘트 지수 감소율)
    pub beta1: f32,
    /// 베타2 파라미터 (2차 모멘트 지수 감소율)
    pub beta2: f32,
    /// 엡실론 (수치 안정성을 위한 작은 값)
    pub epsilon: f32,
    /// 리만 메트릭 정규화 계수
    pub metric_regularization: f32,
}

/// 학습률 스케줄링 방법
#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    /// 고정 학습률
    Constant,
    /// 지수 감소
    ExponentialDecay { decay_rate: f32, decay_steps: usize },
    /// 코사인 어닐링
    CosineAnnealing { min_lr: f32, max_lr: f32, period: usize },
    /// 스텝 감소
    StepDecay { step_size: usize, gamma: f32 },
    /// 적응적 학습률
    Adaptive { patience: usize, factor: f32 },
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            adam: AdamConfig::default(),
            riemannian_adam: RiemannianAdamConfig::default(),
            learning_rate: 0.001,
            lr_schedule: LearningRateSchedule::Constant,
            gradient_clipping: Some(1.0),
            weight_decay: 0.0,
        }
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl Default for RiemannianAdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            metric_regularization: 1e-4,
        }
    }
}

impl OptimizerConfig {
    /// 새 구성 생성
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Adam 구성 설정
    pub fn with_adam_config(mut self, config: AdamConfig) -> Self {
        self.adam = config;
        self
    }
    
    /// Riemannian Adam 구성 설정
    pub fn with_riemannian_adam_config(mut self, config: RiemannianAdamConfig) -> Self {
        self.riemannian_adam = config;
        self
    }
    
    /// 학습률 설정
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
    
    /// 학습률 스케줄 설정
    pub fn with_lr_schedule(mut self, schedule: LearningRateSchedule) -> Self {
        self.lr_schedule = schedule;
        self
    }
    
    /// 그래디언트 클리핑 설정
    pub fn with_gradient_clipping(mut self, clip_value: Option<f32>) -> Self {
        self.gradient_clipping = clip_value;
        self
    }
    
    /// 가중치 감소 설정
    pub fn with_weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }
} 