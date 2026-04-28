"""
Lookback Lens: attention-based hallucination detection.

Reference: Chuang et al. (2024) https://arxiv.org/pdf/2407.07071
"""
from .extractor import LookbackRatioExtractor
from .classifier import LookbackLensClassifier

__all__ = ["LookbackRatioExtractor", "LookbackLensClassifier"]
