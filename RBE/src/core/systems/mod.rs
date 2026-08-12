pub mod core_layer;
pub mod hybrid_system;
pub mod config;
pub mod performance;
pub mod state_management;
pub mod compute_engine;

pub use core_layer::{EncodedLayer, FusedEncodedLayer};
pub use hybrid_system::{
    HybridPoincareRBESystem, HybridPoincareLayer, 
    PoincareEncodingLayer, FusionProcessingLayer, HybridLearningLayer
};
pub use config::{
    SystemConfiguration, LearningParameters, AdaptiveLearningRateConfig,
    LossWeights, HardwareConfiguration
};
pub use performance::{
    PerformanceMonitor, MemoryUsageTracker, ComputationTimeTracker,
    QualityMetricsTracker, EnergyEfficiencyTracker
};
pub use state_management::{
    LearningState, LossComponents, ConvergenceStatus,
    StateManager, ParameterManager
};
pub use compute_engine::{
    ResidualCompressor, CORDICEngine, BasisFunctionLUT,
    ParallelGEMMEngine, RiemannianGradientComputer, StateTransitionDifferentiator,
    AdaptiveScheduler, PerformanceAnalyzer, LayerMetrics,
    ActivationStatistics, WeightStatistics, GradientStatistics,
    InternalTransformType, LearningRateStrategy
};

#[cfg(test)]
pub mod __tests__;
