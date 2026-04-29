# src.models.eiv.adaptive.__init__

## 用途
- 导出带有数据依赖惩罚项的自适应 EIV 模型。

## 逐行说明
- `from .ACoCoLasso import ACoCoLasso`: 导入自适应凸约束 Lasso。
- `from .ACLasso import ACLasso`: 导入自适应修正 Lasso。
- `__all__ = [...]`: 声明对外公开的自适应 EIV 导出符号。

## 备注
- 自适应方法需要初始系数估计来构建权重。
