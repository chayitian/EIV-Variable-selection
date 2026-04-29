# src.models.__init__

## 用途
- 聚合基础模型与 EIV 模型，统一在 `models` 命名空间下导出。

## 逐行说明
- `from .base import ALasso, ElasticNet, Lasso, OLS, Ridge`: 引入基础线性模型。
- `from .eiv import (...)`: 引入测量误差修正的 EIV 模型。
- `ACoCoLasso`: 自适应凸约束 Lasso（EIV）。
- `ACLasso`: 自适应修正 Lasso（EIV）。
- `CLasso`: 修正 Lasso（EIV）。
- `COLS`: 修正 OLS（EIV）。
- `CoCoElasticNet`: 修正弹性网络（EIV）。
- `CoCoLasso`: 凸约束 Lasso（EIV）。
- `CRidge`: 修正 Ridge（EIV）。
- `__all__ = [...]`: 列出 `models` 子包对外公开的导出符号。

## 备注
- `__all__` 是通配导入的权威导出列表，需与实际导入保持一致。
