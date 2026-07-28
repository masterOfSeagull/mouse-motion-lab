"""Deterministic baseline generators for MouseMotionLab."""

from .base import (
    CONDITION_FEATURE_NAMES, OUTPUT_SIZE, POSITION_COUNT, GeneratedParameterBatch, GenerationRequest,
    GenerationResult, MovementGenerator, ProcessedDataset, TrajectorySample, condition_vector,
    constrain_parameter_output, decode_output, seeded_normal_source,
)
from .pca_mixture import PcaMixtureConfig, PcaMixtureGenerator
from .retrieval import RetrievalConfig, RetrievalGenerator, RetrievalResult
from .dataset import ProcessedDatasetError, load_processed_dataset
from .validation import validate_baseline
from .training import BaselineTrainingError, build_baseline_model, load_generator
from .conditional_flow import ConditionalFlowConfig, ConditionalFlowGenerator
from .flow_training import train_conditional_flow, train_experimental_conditional_flow
from .registry import PromotionError, compare_models, promote_model

__all__ = [
    "BaselineTrainingError", "CONDITION_FEATURE_NAMES", "ConditionalFlowConfig", "ConditionalFlowGenerator", "OUTPUT_SIZE", "POSITION_COUNT", "GeneratedParameterBatch", "GenerationRequest",
    "GenerationResult", "MovementGenerator", "PcaMixtureConfig", "PcaMixtureGenerator", "ProcessedDataset",
    "ProcessedDatasetError", "PromotionError", "RetrievalConfig", "RetrievalGenerator", "RetrievalResult", "TrajectorySample",
    "build_baseline_model", "compare_models", "condition_vector", "constrain_parameter_output", "decode_output", "load_generator",
    "load_processed_dataset", "promote_model", "seeded_normal_source", "train_conditional_flow",
    "train_experimental_conditional_flow", "validate_baseline",
]
