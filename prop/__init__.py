"""Proposed/innovative models package."""

from .Adaptive_CoCoLasso import AdaptiveCoCoLasso
from .RandomForest_Corrected_Lasso import RandomForestCorrectedLasso
from .XGBoost_Corrected_Lasso import XGBoostCorrectedLasso

__all__ = [
	'AdaptiveCoCoLasso',
	'RandomForestCorrectedLasso',
	'XGBoostCorrectedLasso',
]
