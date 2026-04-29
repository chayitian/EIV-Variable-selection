# src.models.base.OLS

## 概述
- 实现普通最小二乘回归，支持可选标准化与多种求解器。
- 支持直接求解、Cholesky、SVD 或梯度下降。

## 理论与公式
- 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - X\beta \rVert_2^2
  $$
- 正规方程:
  $$
  \Sigma_X \beta = \rho,\quad \Sigma_X = \frac{X^T X}{n},\quad \rho = \frac{X^T y}{n}
  $$

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.preprocessing import StandardScaler`: 在 `normalize=True` 时使用。
- `class OLS:`: 定义估计器。
- 文档字符串行: 说明目标函数与求解器选项。
- `def __init__(...)`: 设置截距、标准化与求解器参数。
- `self.coef_`, `self.intercept_`, `self.scaler_X_`, `self.scaler_y_`: 初始化状态。
- `_solve_linear_system`: 统一处理 $\Sigma_X \beta = \rho$ 的求解方法。
- `if self.solver == 'solve'`: 直接使用 `np.linalg.solve`。
- `if self.solver == 'cholesky'`: 对称正定时使用 Cholesky。
- `if self.solver == 'svd'`: 近奇异时使用 SVD 稳定求解。
- `if self.solver == 'gd'`: 使用梯度下降。
- GD path: 校验迭代次数和阈值；未提供步长时用 Lipschitz 常数决定。
- `beta = np.zeros_like(rho)`: GD 初始解。
- `grad = Sigma_X @ beta - rho`: 梯度更新方向。
- 收敛判断采用系数最大变化量。
- `raise ValueError(...)`: 拒绝不支持的求解器参数。
- `def fit(self, X, y):`: 训练入口。
- 校验求解器合法性。
- `if self.normalize`: 标准化 X 并中心化 y。
- `Sigma_X = (X_scaled.T @ X_scaled) / n_samples`: 标准化后的协方差。
- 最小特征值检查，必要时加入对角扰动提升稳定性。
- `rho = (X_scaled.T @ y_centered) / n_samples`: 右侧向量。
- `beta_scaled = self._solve_linear_system(...)`: 在标准化空间求解。
- `beta_original_scale = beta_scaled / X_std`: 反缩放回原尺度。
- `self.intercept_ = scaler_y.mean_ - scaler_X.mean_ @ beta_original_scale`: 恢复截距。
- `else:` 分支处理未标准化输入。
- `if self.fit_intercept`: 显式中心化 X 和 y。
- 计算 `Sigma_X` 与 `rho`。
- `self.coef_ = self._solve_linear_system(...)`: 保存系数。
- `self.intercept_ = y_mean - X_mean @ self.coef_` 或 `0.0`: 设置截距。
- `def predict(self, X):`: 预测方法。
- `return X @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 对角扰动用于避免协方差矩阵非正定导致的数值问题。
- `normalize=True` 时先标准化再反变换，便于比较系数。
