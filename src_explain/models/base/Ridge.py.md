# src.models.base.Ridge

## 概述
- 实现岭回归（Ridge Regression），在 OLS 基础上加入 L2 正则化。
- 支持直接求解、Cholesky、SVD 与梯度下降求解器。

## 理论与公式
- 目标函数:
  $$
  \min_{\beta} \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \frac{\alpha}{2}\lVert \beta \rVert_2^2
  $$
- 正规方程:
  $$
  (\Sigma_X + \alpha I)\beta = \rho,\quad \Sigma_X = \frac{X^T X}{n},\quad \rho = \frac{X^T y}{n}
  $$

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.preprocessing import StandardScaler`: 需要标准化时使用。
- `class Ridge:`: 定义岭回归估计器。
- 文档字符串行: 说明目标函数与求解器设置。
- `def __init__(...)`: 保存 `alpha`、截距与求解器配置。
- `self.coef_`, `self.intercept_`, `self.scaler_X_`, `self.scaler_y_`: 初始化拟合状态。
- `_solve_linear_system`: 求解 $(\Sigma_X + \alpha I)\beta=\rho$。
- `ridge_system = Sigma_X + self.alpha * I`: 构造正则化线性系统。
- `if self.solver == 'solve'`: 直接求解。
- `if self.solver == 'cholesky'`: Cholesky 分解求解。
- `if self.solver == 'svd'`: SVD 稳定求解。
- `if self.solver == 'gd'`: 梯度下降求解。
- `grad = ridge_system @ beta - rho`: 梯度下降的梯度。
- `def fit(self, X, y)`: 训练入口。
- `if self.alpha < 0`: 校验正则化强度。
- `if self.normalize`: 标准化 X、中心化 y。
- `Sigma_X = (X_scaled.T @ X_scaled) / n_samples`: 标准化协方差。
- 最小特征值检查: 对协方差做稳定化抖动。
- `beta_scaled = self._solve_linear_system(...)`: 标准化空间求解。
- `beta_original_scale = beta_scaled / X_std`: 反缩放回原尺度。
- `self.intercept_ = scaler_y.mean_ - scaler_X.mean_ @ beta_original_scale`: 恢复截距。
- 未标准化分支: 视 `fit_intercept` 做中心化并求解。
- `def predict(self, X)`: 预测方法。
- `return X @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- `alpha=0` 时退化为 OLS。
- 标准化有助于让正则化对不同量纲特征更公平。
