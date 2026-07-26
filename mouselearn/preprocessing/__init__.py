"""Deterministic conversion of immutable raw snapshots into model-ready representations."""

from .runner import PreprocessingError, PreprocessingSpec, preprocess_dataset_snapshot

__all__ = ["PreprocessingError", "PreprocessingSpec", "preprocess_dataset_snapshot"]
