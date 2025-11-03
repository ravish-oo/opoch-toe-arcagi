"""
ARC-AGI Bit-Algebra Solver

Deterministic, receipts-first solver for ARC-AGI tasks.
Spec version: 1.5
"""

__version__ = "0.1.0"
__spec_version__ = "1.5"

# WO-14: Aggregate Mapping (T9) - Features & Size Predictor
from .features import (
    FeatureVector,
    SizeFit,
    ColorMap,
    agg_features,
    agg_size_fit,
    predict_size,
    agg_color_map
)

__all__ = [
    # WO-14 Features
    "FeatureVector",
    "SizeFit",
    "ColorMap",
    "agg_features",
    "agg_size_fit",
    "predict_size",
    "agg_color_map",
]
