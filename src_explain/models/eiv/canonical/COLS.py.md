# src.models.eiv.canonical.COLS

## 概述
- 实现修正 OLS，用于含测量误差的回归。
- 常用于自适应 EIV 方法的初始系数估计。

## 理论与公式
- 修正协方差:
  $$
  \Sigma_{\text{corrected}} = \frac{W^T W}{n} - \Sigma_{uu}
  $$
- 闭式解:
  $$
  \beta = \Sigma_{\text{corrected}}^{-1} \rho,\quad \rho = \frac{W^T y}{n}
  $$

## 逐行说明
- `import numpy as np`: 数值计算。
- `from sklearn.preprocessing import StandardScaler`: 标准化工具。
- `class COLS:`: 定义修正 OLS 估计器。
- 文档字符串行: 描述用途与参数。
- `def __init__(self, Sigma_uu=None)`: 保存测量误差协方差。
- `self.coef_`, `self.intercept_`, `self.scaler_W_`, `self.scaler_y_`: 初始化状态。
- `def fit(self, W, y):`: 训练入口。
- `n_samples, n_features = W.shape`: 维度。
- `if self.Sigma_uu is None`: 默认测量误差为零。
- `if Sigma_uu.shape != (n_features, n_features)`: 校验协方差形状。
- `self.scaler_W_` 与 `self.scaler_y_`: 标准化准备。
- `W_scaled = self.scaler_W_.fit_transform(W)`: 标准化特征。
- `y_centered = self.scaler_y_.fit_transform(...).flatten()`: 中心化响应。
- `W_std` 与 `W_std_safe`: 缩放因子与零方差保护。
- `Sigma_uu_scaled = Sigma_uu / np.outer(W_std_safe, W_std_safe)`: 缩放误差协方差。
- `Sigma_W = (W_scaled.T @ W_scaled) / n_samples`: 经验协方差。
- `Sigma_corrected = Sigma_W - Sigma_uu_scaled`: 修正协方差。
- 对称化与最小特征值检查: 保证 PSD 与数值稳定。
- `rho = (W_scaled.T @ y_centered) / n_samples`: 交叉项。
- `beta_scaled = np.linalg.solve(Sigma_corrected, rho)`: 闭式解求解。
- `beta_original_scale = beta_scaled / W_std_safe`: 反缩放。
- `beta_original_scale = np.where(W_std > 1e-12, ...)`: 零方差特征置零。
- `self.coef_` 与 `self.intercept_`: 保存拟合参数。
- `return self`: 链式调用。
- `def predict(self, W)`: 预测方法。
- `return W @ self.coef_ + self.intercept_`: 线性预测。

## 备注
- 修正协方差需为正定；对角抖动用于保障可逆。
- 作为初始估计可帮助自适应惩罚聚焦强信号。
