try:
    from .hierarchical_sentence_topic_llm import (
        HierarchicalLLMConfig,
        HierarchicalSentenceTopicLLM,
        SentenceTopicHead,
        MetricContextRouter,
        HierarchicalLMDecoder,
        RCELexicalDecoder,
        HAS_METRIKEY,
    )
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False
    HierarchicalLLMConfig = None
    HierarchicalSentenceTopicLLM = None
    SentenceTopicHead = None
    MetricContextRouter = None
    HierarchicalLMDecoder = None
    RCELexicalDecoder = None
    HAS_METRIKEY = False

try:
    from .transformer_converter import (
        RSULFConfig,
        RSULFTransformerConverter,
        convert_transformer_to_rsulf,
    )
    _HAS_CONVERTER = True
except ImportError:
    _HAS_CONVERTER = False
    RSULFConfig = None
    RSULFTransformerConverter = None
    convert_transformer_to_rsulf = None

from .riemannian_aggregation import RiemannianAggregation
from .residual_reuse import (
    ResidualReuseMLP,
    ResidualReuseStats,
    install_gpt2_residual_reuse,
    reset_residual_reuse,
    residual_reuse_report,
)

__all__ = [
    "RiemannianAggregation",
    "ResidualReuseMLP",
    "ResidualReuseStats",
    "install_gpt2_residual_reuse",
    "reset_residual_reuse",
    "residual_reuse_report",
    "HierarchicalLLMConfig",
    "HierarchicalSentenceTopicLLM",
    "SentenceTopicHead",
    "MetricContextRouter",
    "HierarchicalLMDecoder",
    "RCELexicalDecoder",
    "HAS_METRIKEY",
    "RSULFConfig",
    "RSULFTransformerConverter",
    "convert_transformer_to_rsulf",
    "_HAS_LLM",
    "_HAS_CONVERTER",
]
