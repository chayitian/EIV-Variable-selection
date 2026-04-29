# src.models.eiv.__init__

## 用途
- 将 EIV（误差-变量）模型聚合到一个命名空间中。

## 逐行说明
- `from .adaptive import ACoCoLasso, ACLasso`: 导入自适应 EIV 模型。
- `from .canonical import CoCoLasso, CoCoElasticNet, CLasso, COLS, CRidge`: 导入标准（非自适应）EIV 模型。
- `__all__ = [...]`: 列出对外公开的 EIV 模型符号。

## 备注
- 该模块由 `src.models` 统一导出，方便在更高层级使用。
