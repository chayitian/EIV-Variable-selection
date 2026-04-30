from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .ALasso import ALasso
from .ACLasso import ACLasso
from .ACoCoLasso import ACoCoLasso
from .CLasso import CLasso
from .COLS import COLS
from .CoCoElasticNet import CoCoElasticNet
from .CoCoLasso import CoCoLasso
from .CRidge import CRidge
from .vs_evaluate import selection_accuracy

OLS = LinearRegression

__all__ = [
    'OLS',
    'Lasso',
    'ALasso',
    'ElasticNet',
    'Ridge',
    'COLS',
    'CLasso',
    'CRidge',
    'CoCoLasso',
    'CoCoElasticNet',
    'ACLasso',
    'ACoCoLasso',
    'selection_accuracy',
]
