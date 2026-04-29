# src.models.eiv.canonical.__init__

## 用途
- 导出标准（非自适应）EIV 模型类。

## 逐行说明
- `from .CoCoLasso import CoCoLasso`: 导入凸约束 Lasso。
- `from .CoCoElasticNet import CoCoElasticNet`: 导入凸约束弹性网络。
- `from .CLasso import CLasso`: 导入修正 Lasso。
- `from .COLS import COLS`: 导入修正 OLS。
- `from .CRidge import CRidge`: 导入修正 Ridge。
- `__all__ = [...]`: 声明对外公开的标准 EIV 导出符号。

## 备注
- 这些模型通过传入 `Sigma_uu` 来修正测量误差影响。
