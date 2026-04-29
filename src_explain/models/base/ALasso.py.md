# src.models.base.ALasso

## 概述
- 实现自适应 Lasso，通过初始估计对系数进行重加权。
- 支持坐标下降、ADMM、ISTA 与 LARS 求解器。

## 理论与公式
- 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \lambda \sum_{j=1}^p w_j |\beta_j|
  $$
- 自适应权重:
  $$
  w_j = \frac{1}{(|\hat{\beta}_j^{\text{init}}| + \epsilon)^{\gamma}}
  $$
- 加权 Lasso 可通过列缩放或加权阈值来求解。

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.linear_model import LassoLars`: 提供 LARS 求解器。
- `class ALasso:`: 定义自适应 Lasso 估计器。
- 文档字符串行: 说明目标函数、权重、参数与参考。
- `def __init__(...)`: 保存超参数、初始系数与求解器设置。
- `self.coef_`, `self.intercept_`, `self.weights_`, `self.init_coef_`, `self.n_iter_`: 初始化模型状态。
- `_soft_threshold`: 软阈值算子 $S(\rho,\lambda)$。
- `_fit_coordinate_descent`: 通过加权设计 $X/weights$ 做坐标下降。
- `X_weighted = X_centered / weights`: 列缩放使加权 L1 变为标准 L1。
- `beta_weighted = np.zeros(...)`: 从零初始化。
- 内层循环: 计算残差、$\rho_j$、$z_j$ 并阈值化更新。
- `return beta_weighted / weights`: 收敛后映射回原尺度。
- `_fit_admm`: 标准 ADMM，但阈值为 `self.alpha * weights / rho`。
- `_fit_ista`: 使用 Lipschitz 步长并带权阈值。
- `_fit_lars`: 在加权设计上运行 LARS，再按 `1/weights` 还原。
- `def fit(self, X, y):`: 校验输入并构建权重。
- 校验 `alpha` 非负、`solver` 合法、`admm_rho` 正值。
- `if self.init_coef is None`: 自适应 Lasso 需要初始系数。
- `init_coef = np.asarray(self.init_coef, ...)`: 校验形状与有限性。
- `self.weights_ = 1.0 / (|init| + epsilon)^gamma`: 计算权重。
- 中心化: 计算均值并去除截距。
- 求解器分派设置 `self.coef_`。
- `self.intercept_ = y_mean - np.dot(X_mean, self.coef_)`: 恢复截距。
- `def predict(self, X):`: 预测方法。
- `return X @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 该实现要求外部提供 `init_coef`，常来自 OLS 或 Ridge。
- `epsilon` 用于避免初始系数接近 0 时的除零问题。
