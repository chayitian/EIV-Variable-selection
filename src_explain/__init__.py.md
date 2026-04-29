# src.__init__

## 用途
- 统一导出模型类与评估函数，便于从 `src` 直接导入使用。

## 逐行说明
- `from .models import (...)`: 重新导出模型类，调用方可直接从 `src` 引入。
- `OLS`: 普通最小二乘模型。
- `ALasso`: 自适应 Lasso 基础模型。
- `ElasticNet`: 弹性网络基础模型。
- `ACoCoLasso`: 自适应 CoCoLasso（EIV 模型）。
- `ACLasso`: 自适应修正 Lasso（EIV 模型）。
- `CLasso`: 修正 Lasso（EIV 模型）。
- `COLS`: 修正 OLS（EIV 模型）。
- `CoCoElasticNet`: 修正弹性网络（EIV 模型）。
- `CoCoLasso`: 凸约束 Lasso（EIV 模型）。
- `CRidge`: 修正 Ridge（EIV 模型）。
- `Lasso`: 标准 Lasso 基础模型。
- `from .evaluation import selection_accuracy`: 导出变量选择评估函数。
- `__all__ = [...]`: 显式声明 `from src import *` 的公开接口。
- `__all__` 中字符串与上方导入保持一致，保证导出稳定。

## 备注
- 若重命名或移除模型，需要同步更新导入列表与 `__all__`，避免导入错误。
